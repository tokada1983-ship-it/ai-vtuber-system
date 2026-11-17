"""リソースモニター（CPU・メモリ・GPU/VRAM）

ダッシュボードが開いている間だけ2秒ごとに計測し、WebSocketで配る。
直近5分ぶんの履歴を持つので、画面を開いた直後からグラフが描ける。

VRAM はプロセスごとの使用量が Windows（WDDM）では取れないため、
GPU全体の使用量に加えて「どのサービスが何を載せているか」
（Ollama のモデルと VRAM 量、LM Studio の読み込み中モデル、Irodori-TTS の起動状態）を並べる。
"""
from __future__ import annotations

import asyncio
import os
import re
import shutil
import time
from collections import deque

import httpx
import psutil

from backend.gpu_process import GpuProcessMemory
from backend.logger import get_logger

logger = get_logger(__name__)

HISTORY_SEC = 300
INTERVAL_SEC = 2.0
GB = 1024 ** 3


async def query_gpus() -> list[dict]:
    """nvidia-smi で GPU の使用率・VRAM・温度・消費電力を取る（無ければ空）"""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return []
    try:
        proc = await asyncio.create_subprocess_exec(
            exe, "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
            "--format=csv,noheader,nounits",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
            creationflags=getattr(__import__("subprocess"), "CREATE_NO_WINDOW", 0),
        )
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=4)
    except Exception:
        return []

    def num(v: str) -> float | None:
        try:
            return float(v)
        except ValueError:
            return None

    gpus = []
    for line in out.decode("utf-8", errors="replace").splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 7:
            continue
        used, total = num(parts[3]), num(parts[4])
        gpus.append({
            "index": int(parts[0]), "name": parts[1], "util": num(parts[2]),
            "vram_used_gb": round(used / 1024, 2) if used is not None else None,
            "vram_total_gb": round(total / 1024, 2) if total is not None else None,
            "vram_percent": round(used / total * 100, 1) if used and total else None,
            "temp": num(parts[5]), "power_w": num(parts[6]),
        })
    return gpus


# 見かけ上の値が出るだけで実体の無いプロセス（dwm は共有メモリを数十GBと報告する）
_IGNORE = {"dwm.exe"}
_SYSTEM = {"csrss.exe", "explorer.exe", "searchhost.exe", "startmenuexperiencehost.exe",
           "textinputhost.exe", "shellexperiencehost.exe", "applicationframehost.exe"}
_MIN_GB = 0.1   # これ未満のアプリは「その他のアプリ」にまとめる


def _model_from_path(path: str) -> str:
    name = os.path.basename(path or "")
    return re.sub(r"\.gguf$", "", name, flags=re.I) or "?"


def classify(pid: int, name: str, cmd: list[str], ollama_models: list[str],
             ollama_runners: int, own_pid: int) -> tuple[str, str, str, str]:
    """
    プロセスを (まとめるキー, 表示名, サービス名, 種別) に分類する。
    種別: ai（モデル）/ app（その他のアプリ）/ system（Windows）
    """
    low = name.lower()
    line = " ".join(cmd)
    if "irodori_openai_tts" in line:
        return "irodori", "Irodori-TTS v4.1-Small", "音声合成", "ai"
    if "backend.persona.trainer" in line:
        return "trainer", "人格QLoRA 学習中", "学習", "ai"
    if "llama-server" in low or (low == "ollama.exe" and "runner" in cmd):
        model = ""
        if "--model" in cmd:
            model = cmd[cmd.index("--model") + 1]
        elif "-m" in cmd:
            model = cmd[cmd.index("-m") + 1]
        if ".lmstudio" in line.lower() or "lm studio" in line.lower():
            m = _model_from_path(model)
            return f"lms:{m}", m, "LM Studio", "ai"
        if "ollama" in line.lower():
            # Ollama は内部ファイル名しか見えないので、載っているモデルが1つならその名前にする
            label = ollama_models[0] if len(ollama_models) == 1 and ollama_runners == 1 else "（モデル）"
            return f"ollama:{pid}", label, "Ollama", "ai"
        m = _model_from_path(model)
        return f"llama:{m}", m, "llama.cpp", "ai"
    if pid == own_pid or "backend.main" in line or line.rstrip().endswith("run.py"):
        return "app", "AI-Vtuber（音声認識・アバター）", "AI-Vtuber", "ai"
    if "main.py" in line and "8188" in line:
        return "comfyui", "ComfyUI", "画像生成", "ai"
    if low in _SYSTEM:
        return "system", "Windows（画面表示など）", "システム", "system"
    pretty = re.sub(r"\.exe$", "", name, flags=re.I)
    return f"app:{low}", pretty, "アプリ", "app"


class ResourceMonitor:

    def __init__(self, config, hub):
        self._cfg = config
        self._hub = hub
        self._proc = psutil.Process(os.getpid())
        self._history: deque[dict] = deque(maxlen=int(HISTORY_SEC / INTERVAL_SEC))
        self._latest: dict = {}
        self._users: list[dict] = []
        self._users_at = 0.0
        self._task: asyncio.Task | None = None
        self._gpu_mem = GpuProcessMemory()
        self._ollama_models: list[str] = []
        psutil.cpu_percent(None)          # 1回目は0が返るので空振りしておく
        self._proc.cpu_percent(None)

    # ── 計測 ──────────────────────────────────────────────────

    async def sample(self) -> dict:
        vm = psutil.virtual_memory()
        gpus = await query_gpus()
        try:
            rss = self._proc.memory_info().rss
            app_cpu = self._proc.cpu_percent(None) / (psutil.cpu_count() or 1)
        except psutil.Error:
            rss, app_cpu = 0, 0.0
        snap = {
            "type": "system_stats",
            "ts": time.time(),
            "cpu": psutil.cpu_percent(None),
            "cpu_count": psutil.cpu_count(),
            "ram_used_gb": round((vm.total - vm.available) / GB, 2),
            "ram_total_gb": round(vm.total / GB, 2),
            "ram_percent": vm.percent,
            "app_ram_gb": round(rss / GB, 2),
            "app_cpu": round(app_cpu, 1),
            "gpus": gpus,
        }
        # VRAM を使っているサービスは変化が遅いので10秒ごとに調べる
        if time.time() - self._users_at > 10:
            self._users = await self.vram_users()
            self._users_at = time.time()
        snap["vram_users"] = self._users
        snap["vram_breakdown"] = self.vram_breakdown(gpus[0] if gpus else None)
        snap["warnings"] = self._warnings(snap)
        self._latest = snap
        self._history.append({
            "ts": snap["ts"], "cpu": snap["cpu"], "ram": snap["ram_percent"],
            "vram": gpus[0]["vram_percent"] if gpus else None,
            "gpu": gpus[0]["util"] if gpus else None,
        })
        return snap

    @staticmethod
    def _warnings(s: dict) -> list[str]:
        w = []
        if s["ram_percent"] >= 90:
            w.append(f"メモリが残りわずかです（{s['ram_percent']:.0f}%）")
        for g in s["gpus"]:
            if g["vram_percent"] and g["vram_percent"] >= 92:
                w.append(f"VRAMが残りわずかです（{g['vram_used_gb']}/{g['vram_total_gb']} GB）。"
                         "使っていないモデルをアンロードしてください")
            if g["temp"] and g["temp"] >= 83:
                w.append(f"GPUの温度が高めです（{g['temp']:.0f}℃）")
        if s["cpu"] >= 95:
            w.append(f"CPUが高負荷です（{s['cpu']:.0f}%）")
        return w

    def vram_breakdown(self, gpu: dict | None) -> list[dict]:
        """
        VRAMの内訳（何が何GB使っているか）。大きい順、AIモデルを先に並べる。
        合計がGPU全体の使用量に満たないぶんは「内訳不明」として足す。
        """
        per_pid = self._gpu_mem.read()
        if not per_pid:
            return []
        runners = 0
        procs = []
        for pid, used in per_pid.items():
            if used <= 0:
                continue
            # 名前は取れても起動引数は権限で取れないことがある（dwm.exe など）ので別々に取る
            try:
                p = psutil.Process(pid)
                name = p.name()
            except (psutil.Error, OSError):
                continue
            try:
                cmd = p.cmdline()
            except (psutil.Error, OSError):
                cmd = []
            if name.lower() in _IGNORE:
                continue
            if "llama-server" in name.lower() and "ollama" in " ".join(cmd).lower():
                runners += 1
            procs.append((pid, name, cmd, used))

        groups: dict[str, dict] = {}
        for pid, name, cmd, used in procs:
            key, label, service, kind = classify(pid, name, cmd, self._ollama_models, runners, os.getpid())
            g = groups.setdefault(key, {"key": key, "label": label, "service": service,
                                        "kind": kind, "bytes": 0})
            g["bytes"] += used

        items, small = [], 0
        for g in groups.values():
            if g["bytes"] / GB < 0.01:
                continue
            if g["kind"] == "app" and g["bytes"] / GB < _MIN_GB:
                small += g["bytes"]
            else:
                items.append(g)
        if small:
            items.append({"key": "apps", "label": "その他のアプリ", "service": "アプリ",
                          "kind": "app", "bytes": small})
        total = sum(g["bytes"] for g in items)
        if gpu and gpu.get("vram_used_gb") is not None:
            rest = gpu["vram_used_gb"] * GB - total
            if rest > 0.2 * GB:
                items.append({"key": "unknown", "label": "内訳不明（共有領域など）", "service": "システム",
                              "kind": "system", "bytes": rest})
        order = {"ai": 0, "app": 1, "system": 2}
        items.sort(key=lambda g: (order[g["kind"]], -g["bytes"]))
        for g in items:
            g["gb"] = round(g.pop("bytes") / GB, 2)
        return items

    async def vram_users(self) -> list[dict]:
        """VRAMにモデルを載せている（はずの）サービスを並べる"""
        users: list[dict] = []
        async with httpx.AsyncClient(timeout=2.0) as client:
            # Ollama：モデルごとの VRAM 量まで分かる
            be = self._cfg.llm.get_backend("ollama")
            try:
                r = await client.get(f"{(be.base_url if be else 'http://localhost:11434').rstrip('/')}/api/ps")
                models = r.json().get("models", [])
                self._ollama_models = [m.get("name", "") for m in models]
                for m in models:
                    users.append({"service": "Ollama", "model": m.get("name", ""),
                                  "vram_gb": round((m.get("size_vram") or 0) / GB, 2)})
            except Exception:
                pass
            # LM Studio：読み込み中のモデル名（VRAM量はAPIで取れない）
            lm = self._cfg.llm.get_backend("lmstudio")
            if lm and lm.base_url:
                root = lm.base_url.rstrip("/").removesuffix("/v1")
                try:
                    r = await client.get(f"{root}/api/v0/models")
                    for m in r.json().get("data", []):
                        if m.get("state") == "loaded":
                            users.append({"service": "LM Studio", "model": m.get("id", ""), "vram_gb": None})
                except Exception:
                    pass
            # Irodori-TTS：起動していればモデル（約3GB）を載せている
            ir = self._cfg.tts.backends.get("irodori")
            if ir and self._cfg.tts.primary == "irodori":
                base = ir.base_url.rstrip("/").removesuffix("/v1")
                try:
                    r = await client.get(f"{base}/health")
                    if r.status_code < 500:
                        users.append({"service": "Irodori-TTS", "model": "v4.1-Small", "vram_gb": None})
                except Exception:
                    pass
        return users

    def snapshot(self) -> dict:
        return {**self._latest, "history": list(self._history)}

    # ── 配信ループ ────────────────────────────────────────────

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._loop(), name="resource_monitor")

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
            self._task = None

    async def _loop(self) -> None:
        while True:
            try:
                # ダッシュボードを誰も開いていなければ計測しない（nvidia-smi の起動コストを省く）
                if self._hub.has_clients("dashboard"):
                    await self._hub.broadcast("dashboard", await self.sample())
                await asyncio.sleep(INTERVAL_SEC)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning("resource monitor error", error=str(e))
                await asyncio.sleep(10)