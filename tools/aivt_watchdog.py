"""AI-Vtuber アプリのウォッチドッグ（プロセス監視＋自動再起動）

配信中（data/stream_state.json が存在する間）だけ、アプリ
（<YOUR_PATH>\ai-vtuber、http://127.0.0.1:9000）の生死を監視し、異常終了を
検知したら自動で復旧する。状態ファイルが無い間は完全に無動作。

背景: 2026-11-20 の配信テストでエンジンが APPCRASH（ヒープ破壊）で
即死し、無音・静止画の配信が約80分流れ続けた。再発前提の備えとして、
落下そのものではなく「落ちたまま」を短時間で解消するのが目的。

復旧シーケンス:
  1. 異常（HTTP応答なし＋アプリプロセス不在/ハング）が
     FAIL_THRESHOLD_SEC 継続したら検知
  2. 残存プロセスを終了し、既存タスク aivt-logged でアプリ起動
  3. :9000 の応答を待ち、状態ファイルの video_id で /api/start を再発行
  4. 発話再開をアプリログで確認（ベストエフォート）し、各段階を記録

意図した停止との区別（2026-11-21 stop-fix）: /api/stop は処理の冒頭で
状態ファイルに status="stopping" を記録する。ウォッチドッグは
stopping のファイルがある間は監視・復旧を行わず、復旧シーケンスの
開始直前にも状態ファイルを読み直して stopping になっていたら中止
する。これにより、停止処理の途中でアプリが落とされても（フォール
バックの taskkill 等）、意図した停止を落下と誤認して復活させる
ことがない。status の無い旧形式のファイルは配信中として扱う。

安全弁: 再起動は1時間に最大 MAX_RESTARTS_PER_HOUR 回まで。超過時は
復旧を止め、CRITICAL ログを残す。

常駐方法: schtasks aivt-watchdog（ONLOGON、pyw -3.10）。
このスクリプトは標準ライブラリのみで動作する（アプリが壊れた状態
でも動かなければならないため、アプリ側パッケージに依存しない）。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

# ── 設定 ────────────────────────────────────────────────────
APP_DIR = Path(r"<YOUR_PATH>\ai-vtuber")
STATE_FILE = APP_DIR / "data" / "stream_state.json"
HEALTH_URL = "http://127.0.0.1:9000/api/status"
START_URL = "http://127.0.0.1:9000/api/start"
APP_PORT = 9000
APP_TASK = "aivt-logged"
APP_LOG = Path(r"<YOUR_PATH>\aivt.log")

LOG_FILE = Path(r"<YOUR_PATH>\logs\aivt-watchdog.log")
WD_STATE_FILE = Path(r"<YOUR_PATH>\aivt-watchdog-state.json")
PID_FILE = Path(r"<YOUR_PATH>\aivt-watchdog.pid")

POLL_SEC = 10.0            # 監視間隔
FAIL_THRESHOLD_SEC = 30.0  # 異常がこの秒数継続したら復旧に入る
START_GRACE_SEC = 240.0    # 起動直後のプロセスはこの間「起動中」とみなす
MAX_RESTARTS_PER_HOUR = 3
RESTART_WINDOW_SEC = 3600.0
BOOT_WAIT_SEC = 300.0      # 復旧時のアプリ起動待ち上限
SPEECH_WAIT_SEC = 180.0    # 復旧後の発話再開確認の上限
SPEECH_MARKERS = (b"audio/speech", b"speech=")  # aivt.log の発話兆候


# ── ログ ────────────────────────────────────────────────────
def log(message: str, level: str = "INFO") -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} [{level}] {message}"
    try:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        if LOG_FILE.exists() and LOG_FILE.stat().st_size > 5_000_000:
            LOG_FILE.replace(LOG_FILE.with_suffix(".log.old"))
        with LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass
    try:
        print(line, flush=True)
    except Exception:
        pass


# ── 判定ロジック（単体テスト対象） ───────────────────────────
class FailureTracker:
    """異常（down）の継続時間を測る。

    healthy / starting の観測でリセットし、down が threshold_sec
    以上続いたら True を返す。
    """

    def __init__(self, threshold_sec: float = FAIL_THRESHOLD_SEC):
        self.threshold_sec = threshold_sec
        self._down_since: float | None = None

    def observe(self, state: str, now: float) -> bool:
        if state != "down":
            self._down_since = None
            return False
        if self._down_since is None:
            self._down_since = now
            return False
        return (now - self._down_since) >= self.threshold_sec

    def duration(self, now: float) -> float:
        if self._down_since is None:
            return 0.0
        return now - self._down_since


class RestartLimiter:
    """再起動回数の上限管理（window_sec の間に最大 max_restarts 回）。

    履歴は to_list() / history 引数で出し入れでき、ウォッチドッグ側で
    ファイルへ永続化する（ウォッチドッグ自身の再起動を跨いでも数える）。
    """

    def __init__(self, max_restarts: int = MAX_RESTARTS_PER_HOUR,
                 window_sec: float = RESTART_WINDOW_SEC,
                 history: list[float] | None = None):
        self.max_restarts = max_restarts
        self.window_sec = window_sec
        self._history: list[float] = list(history or [])

    def _prune(self, now: float) -> None:
        self._history = [t for t in self._history
                         if now - t < self.window_sec]

    def can_restart(self, now: float) -> bool:
        self._prune(now)
        return len(self._history) < self.max_restarts

    def record(self, now: float) -> None:
        self._prune(now)
        self._history.append(now)

    def count(self, now: float) -> int:
        self._prune(now)
        return len(self._history)

    def to_list(self) -> list[float]:
        return list(self._history)


# ── IO ヘルパー ─────────────────────────────────────────────
def read_stream_state(path: Path = STATE_FILE) -> dict | None:
    """配信状態ファイルを読む。無い・壊れている場合は None。"""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def is_stopping(state: dict | None) -> bool:
    """状態ファイルが「意図した停止の進行中」を示しているか。

    /api/stop が冒頭で status="stopping" を記録する。status の無い
    旧形式や streaming の場合は False（従来どおり監視対象）。
    """
    return bool(state) and state.get("status") == "stopping"


def recovery_abort_reason(path: Path = STATE_FILE) -> str | None:
    """復旧を中止すべき理由があれば文字列、無ければ None。

    復旧シーケンスの開始直前に状態ファイルを読み直すための判定。
    検知から復旧開始までの間に /api/stop が走り出したケースで、
    意図した停止を復活させてしまう競合を防ぐ。
    """
    state = read_stream_state(path)
    if state is None:
        return "stream state file is gone (stream already ended)"
    if is_stopping(state):
        return "stream is stopping intentionally"
    return None


def http_get_json(url: str, timeout: float = 5.0) -> dict | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            if resp.status != 200:
                return None
            return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None


def http_post_json(url: str, payload: dict,
                   timeout: float = 300.0) -> dict | None:
    try:
        req = urllib.request.Request(
            url, data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None


def app_process_pids() -> list[int]:
    """run.py を実行している python プロセスの PID 一覧"""
    ps = ("Get-CimInstance Win32_Process | "
          "Where-Object { $_.Name -like 'python*' -and "
          "$_.CommandLine -match 'run\\.py' } | "
          "ForEach-Object { $_.ProcessId }")
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps],
            capture_output=True, text=True, timeout=30)
        pids = []
        for line in out.stdout.splitlines():
            line = line.strip()
            if line.isdigit():
                pids.append(int(line))
        return pids
    except Exception:
        return []


def port_listener_pid(port: int = APP_PORT) -> int | None:
    """指定ポートを LISTENING しているプロセスの PID"""
    try:
        out = subprocess.run(["netstat", "-ano"], capture_output=True,
                             text=True, timeout=30)
        for line in out.stdout.splitlines():
            parts = line.split()
            if (len(parts) >= 5 and parts[0] == "TCP"
                    and parts[1].endswith(f":{port}")
                    and parts[3] == "LISTENING"):
                return int(parts[4])
    except Exception:
        pass
    return None


def load_restart_history() -> list[float]:
    try:
        data = json.loads(WD_STATE_FILE.read_text(encoding="utf-8"))
        return [float(t) for t in data.get("restarts", [])]
    except Exception:
        return []


def save_restart_history(history: list[float]) -> None:
    try:
        WD_STATE_FILE.write_text(json.dumps({"restarts": history}),
                                 encoding="utf-8")
    except Exception:
        pass


def wait_for_speech(path: Path, offset: int,
                    timeout_sec: float = SPEECH_WAIT_SEC) -> bool:
    """アプリログの offset 以降に発話の兆候が現れるまで待つ"""
    deadline = time.time() + timeout_sec
    pos = offset
    while time.time() < deadline:
        try:
            size = path.stat().st_size
            if size > pos:
                with path.open("rb") as f:
                    f.seek(pos)
                    chunk = f.read()
                pos = size
                if any(m in chunk for m in SPEECH_MARKERS):
                    return True
        except Exception:
            pass
        time.sleep(5)
    return False


# ── ウォッチドッグ本体 ───────────────────────────────────────
class Watchdog:
    def __init__(self):
        self.tracker = FailureTracker()
        self.limiter = RestartLimiter(history=load_restart_history())
        self._seen_pids: dict[int, float] = {}
        self._limit_logged_at = 0.0
        self._stopping_noted = False

    def probe_state(self, now: float) -> str:
        """アプリの状態を 'healthy' | 'starting' | 'down' で返す"""
        if http_get_json(HEALTH_URL) is not None:
            return "healthy"
        pids = app_process_pids()
        alive: dict[int, float] = {}
        for pid in pids:
            alive[pid] = self._seen_pids.get(pid, now)
        self._seen_pids = alive
        if pids and all(now - t < START_GRACE_SEC
                        for t in alive.values()):
            return "starting"
        return "down"

    def recover(self, state: dict, now: float) -> None:
        # 開始直前に状態ファイルを読み直す。検知後に /api/stop が
        # 走り出していたら、意図した停止なので復旧しない（再起動
        # 予算も消費しない）。
        abort = recovery_abort_reason()
        if abort is not None:
            log(f"RECOVERY ABORTED: {abort}")
            return
        video_id = str(state.get("video_id", ""))
        log(f"RECOVERY START video_id={video_id!r} "
            f"stream_started_at={state.get('started_at')}")
        self.limiter.record(now)
        save_restart_history(self.limiter.to_list())
        try:
            speech_offset = APP_LOG.stat().st_size
        except Exception:
            speech_offset = 0

        # 1) 残存プロセスの終了とポート解放
        victims = set(app_process_pids())
        listener = port_listener_pid()
        if listener:
            victims.add(listener)
        for victim in sorted(victims):
            log(f"killing leftover app process pid={victim}")
            try:
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(victim)],
                               capture_output=True, timeout=30)
            except Exception as e:
                log(f"taskkill failed pid={victim}: {e}", "WARN")
        deadline = time.time() + 30
        while time.time() < deadline and port_listener_pid() is not None:
            time.sleep(2)
        if port_listener_pid() is not None:
            log("RECOVERY FAILED: port 9000 still occupied", "ERROR")
            return

        # 2) 既存の起動タスクでアプリ起動
        log(f"starting app via schtasks /Run /TN {APP_TASK}")
        try:
            subprocess.run(["schtasks", "/Run", "/TN", APP_TASK],
                           capture_output=True, timeout=30)
        except Exception as e:
            log(f"RECOVERY FAILED: schtasks /Run error: {e}", "ERROR")
            return

        # 3) ヘルス待ち
        boot_start = time.time()
        healthy = False
        while time.time() - boot_start < BOOT_WAIT_SEC:
            if http_get_json(HEALTH_URL) is not None:
                healthy = True
                break
            time.sleep(5)
        if not healthy:
            log("RECOVERY FAILED: app did not become healthy "
                f"within {BOOT_WAIT_SEC}s", "ERROR")
            return
        log(f"app healthy after {int(time.time() - boot_start)}s")

        # 4) 状態ファイルの video_id でエンジン再開
        resp = http_post_json(START_URL, {"video_id": video_id})
        if resp is None:
            log("RECOVERY FAILED: /api/start did not respond", "ERROR")
            return
        log(f"engine restart requested via /api/start: {resp}")
        status = http_get_json(HEALTH_URL)
        log(f"status after restart: {status}")

        # 5) 発話再開の確認（ベストエフォート）
        if wait_for_speech(APP_LOG, speech_offset):
            log("RECOVERY DONE: speech resumed (observed in app log)")
        else:
            log("RECOVERY DONE (engine restarted), but speech resumption "
                "was not observed within the wait window", "WARN")

    def run_once(self, now: float) -> None:
        state = read_stream_state()
        if state is None:
            # 配信外: 完全に無動作（計測だけリセット）
            self.tracker.observe("healthy", now)
            self._seen_pids = {}
            self._stopping_noted = False
            return
        if is_stopping(state):
            # 意図した停止の進行中: 監視・復旧の対象外
            if not self._stopping_noted:
                self._stopping_noted = True
                log("stream is stopping intentionally; "
                    "watchdog stands down")
            self.tracker.observe("healthy", now)
            self._seen_pids = {}
            return
        self._stopping_noted = False
        probe = self.probe_state(now)
        if not self.tracker.observe(probe, now):
            return
        # 異常が FAIL_THRESHOLD_SEC 以上継続した
        if not self.limiter.can_restart(now):
            if now - self._limit_logged_at >= 600:
                self._limit_logged_at = now
                log(f"RESTART LIMIT REACHED "
                    f"({self.limiter.count(now)}/{self.limiter.max_restarts} "
                    "restarts in the last hour); automatic recovery "
                    "suspended. Manual intervention required.", "CRITICAL")
            return
        log(f"app failure detected (probe={probe}, down for "
            f"{self.tracker.duration(now):.0f}s); starting recovery")
        try:
            self.recover(state, now)
        except Exception as e:
            log(f"RECOVERY ERROR: {e}", "ERROR")
        self.tracker.observe("healthy", time.time())


def main() -> int:
    # 二重起動防止（PIDファイル＋生存確認）
    if PID_FILE.exists():
        try:
            old = int(PID_FILE.read_text(encoding="utf-8").strip())
            out = subprocess.run(["tasklist", "/FI", f"PID eq {old}"],
                                 capture_output=True, text=True, timeout=30)
            if str(old) in out.stdout:
                log(f"another watchdog is running (pid={old}); exiting")
                return 0
        except Exception:
            pass
    try:
        PID_FILE.write_text(str(os.getpid()), encoding="utf-8")
    except Exception:
        pass
    log(f"watchdog started pid={os.getpid()} "
        "(monitors only while the stream state file exists)")
    wd = Watchdog()
    try:
        while True:
            try:
                wd.run_once(time.time())
            except Exception as e:
                log(f"watchdog loop error: {e}", "ERROR")
            time.sleep(POLL_SEC)
    finally:
        try:
            PID_FILE.unlink(missing_ok=True)
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
