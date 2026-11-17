"""外部のローカルサーバ（TTSサーバなど）を自動で起動・停止する共通部品

  - すでに応答していれば何もしない（ユーザーが手動で起動したものはそのまま使う）
  - 応答しなければ指定コマンドをウィンドウ無しで起動し、ヘルスチェックが通るまで待つ
  - 自分で起動したプロセスだけを終了時に止める
"""
from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx

from backend.logger import get_logger

logger = get_logger(__name__)


def _kill_with_parent(pid: int):
    """
    Windows: 子プロセスを「親が終了したら道連れになる」ジョブに入れる。
    コンソールウィンドウを閉じて強制終了されたときも、裏でTTSサーバだけが
    残り続けて VRAM を握ったままになるのを防ぐ。戻り値のハンドルは保持しておくこと。
    """
    if sys.platform != "win32":
        return None
    import ctypes
    from ctypes import wintypes

    class IO_COUNTERS(ctypes.Structure):
        _fields_ = [(n, ctypes.c_ulonglong) for n in (
            "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
            "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

    class BASIC_LIMIT(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_longlong),
                    ("PerJobUserTimeLimit", ctypes.c_longlong),
                    ("LimitFlags", wintypes.DWORD),
                    ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t),
                    ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.c_size_t),
                    ("PriorityClass", wintypes.DWORD),
                    ("SchedulingClass", wintypes.DWORD)]

    class EXTENDED_LIMIT(ctypes.Structure):
        _fields_ = [("BasicLimitInformation", BASIC_LIMIT),
                    ("IoInfo", IO_COUNTERS),
                    ("ProcessMemoryLimit", ctypes.c_size_t),
                    ("JobMemoryLimit", ctypes.c_size_t),
                    ("PeakProcessMemoryUsed", ctypes.c_size_t),
                    ("PeakJobMemoryUsed", ctypes.c_size_t)]

    JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
    JobObjectExtendedLimitInformation = 9
    PROCESS_SET_QUOTA, PROCESS_TERMINATE = 0x0100, 0x0001
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateJobObjectW.restype = wintypes.HANDLE
    k32.OpenProcess.restype = wintypes.HANDLE
    job = k32.CreateJobObjectW(None, None)
    if not job:
        return None
    info = EXTENDED_LIMIT()
    info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    k32.SetInformationJobObject(wintypes.HANDLE(job), JobObjectExtendedLimitInformation,
                                ctypes.byref(info), ctypes.sizeof(info))
    handle = k32.OpenProcess(PROCESS_SET_QUOTA | PROCESS_TERMINATE, False, pid)
    if handle:
        k32.AssignProcessToJobObject(wintypes.HANDLE(job), wintypes.HANDLE(handle))
        k32.CloseHandle(wintypes.HANDLE(handle))
    return job


class LocalServerProcess:

    def __init__(self, name: str, health_url: str, command: list[str], cwd: str | Path,
                 env: dict | None = None, startup_timeout_sec: float = 300.0,
                 stop_on_exit: bool = True, log_path: str | Path | None = None):
        self.name = name
        self.health_url = health_url
        self.command = command
        self.cwd = Path(cwd)
        self.env = env or {}
        self.startup_timeout_sec = startup_timeout_sec
        self.stop_on_exit = stop_on_exit
        self.log_path = Path(log_path) if log_path else None
        self._proc: subprocess.Popen | None = None
        self._job = None   # Windowsのジョブ（このプロセスが落ちたら子も終了する）
        self._lock = asyncio.Lock()
        self.last_error = ""
        self.ready_at: float | None = None

    @property
    def started_by_us(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    async def is_healthy(self, timeout: float = 3.0) -> bool:
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.get(self.health_url)
                return resp.status_code < 500
        except Exception:
            return False

    async def ensure_running(self) -> bool:
        """応答するまで起動して待つ。使える状態になったら True。"""
        if await self.is_healthy():
            self.ready_at = self.ready_at or time.time()
            return True
        async with self._lock:
            if await self.is_healthy():
                return True
            if not self.cwd.exists():
                self.last_error = f"{self.name}: フォルダが見つかりません（{self.cwd}）"
                logger.warning("local server dir not found", name=self.name, cwd=str(self.cwd))
                return False
            if not self.started_by_us:
                try:
                    self._spawn()
                except OSError as e:
                    self.last_error = f"{self.name} を起動できません: {e}"
                    logger.warning("local server spawn failed", name=self.name, error=str(e))
                    return False
            started = time.monotonic()
            while time.monotonic() - started < self.startup_timeout_sec:
                if await self.is_healthy(timeout=2.0):
                    self.last_error = ""
                    self.ready_at = time.time()
                    logger.info("local server is up", name=self.name,
                                seconds=round(time.monotonic() - started, 1))
                    return True
                if self._proc is not None and self._proc.poll() is not None:
                    self.last_error = (f"{self.name} が終了しました（exit {self._proc.returncode}）"
                                       + (f"。ログ: {self.log_path}" if self.log_path else ""))
                    break
                await asyncio.sleep(1.0)
            else:
                self.last_error = f"{self.name} が {self.startup_timeout_sec:.0f} 秒以内に応答しませんでした"
            logger.warning("local server failed to start", name=self.name, error=self.last_error)
            return False

    def _spawn(self) -> None:
        kwargs: dict = {"cwd": str(self.cwd), "env": {**os.environ, **self.env},
                        "stdin": subprocess.DEVNULL}
        out = subprocess.DEVNULL
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            out = open(self.log_path, "ab")
        kwargs["stdout"] = kwargs["stderr"] = out
        if sys.platform == "win32":
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
        logger.info("starting local server", name=self.name, command=" ".join(self.command))
        self._proc = subprocess.Popen(self.command, **kwargs)
        try:
            self._job = _kill_with_parent(self._proc.pid)
        except Exception as e:   # 失敗しても起動自体は続ける（終了時の後始末が効かないだけ）
            logger.info("could not tie server to parent process", error=str(e))

    def stop(self) -> None:
        """自分で起動したものだけ止める（子プロセスごと）"""
        if not (self.started_by_us and self.stop_on_exit):
            self._proc = None
            return
        logger.info("stopping local server (started by us)", name=self.name)
        if sys.platform == "win32":
            # uv run などは子プロセスを作るので、プロセスツリーごと終了する
            subprocess.run(["taskkill", "/PID", str(self._proc.pid), "/T", "/F"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            self._proc.terminate()
        try:
            self._proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self._proc.kill()
        self._proc = None

    async def status(self) -> dict:
        return {
            "name": self.name,
            "healthy": await self.is_healthy(),
            "started_by_us": self.started_by_us,
            "last_error": self.last_error,
            "log": str(self.log_path) if self.log_path else "",
        }