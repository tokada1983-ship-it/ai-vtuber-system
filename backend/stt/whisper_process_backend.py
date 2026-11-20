"""WhisperProcessBackend: faster-whisper を別プロセスで動かす STT バックエンド

2026-11-20 の APPCRASH 対策 第2段。WhisperModel（ctranslate2）は
backend.stt.whisper_worker プロセスに分離し、本体とは localhost HTTP
だけで通信する。狙いは2つ:

- ctranslate2 とその同梱 DLL（cudnn64_9.dll / libiomp5md.dll）を
  エンジンプロセスから完全に排除する（実測: device=cpu にしても
  ctranslate2 の import 時点で同梱 DLL と torch が全てロードされるため、
  インプロセスのままでは分離にならない）。
- ワーカーがネイティブ障害で落ちてもエンジンは継続し、ワーカーだけを
  自動再起動する。

推論の直列化（第1段）は、クライアント側の threading.Lock で同等に保つ
（HTTP 呼び出し全体をロックする）。ワーカー側にもロックがある。

使い方（config.yaml）:
  stt:
    isolation: process   # inprocess（従来） / process（別プロセス）
"""
from __future__ import annotations

import asyncio
import atexit
import json
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

from backend.logger import get_logger
from backend.stt.base import STTBackend, STTResult
from backend.stt.whisper_backend import FasterWhisperBackend

logger = get_logger(__name__)

# ワーカー再起動の予算（クラッシュループで再起動を繰り返さないため）
_RESTART_WINDOW_SEC = 600.0
_RESTART_MAX_STARTS = 6


class WhisperProcessBackend(STTBackend):

    def __init__(
        self,
        model_size: str = "small",
        device: str = "auto",
        language: str = "ja",
        compute_type: str = "",
        port: int = 0,
        startup_timeout: float = 120.0,
        request_timeout: float = 120.0,
        base_url: str = "",
    ):
        self._model_size = model_size
        self._language = language or None
        self._device, self._compute_type = FasterWhisperBackend._resolve_device(
            device, compute_type)
        self._port = int(port)
        self._startup_timeout = float(startup_timeout)
        self._request_timeout = float(request_timeout)
        # base_url が指定されたら外部ワーカーに接続するだけ（プロセス管理しない）
        self._external_url = base_url.rstrip("/") if base_url else ""
        self._proc: subprocess.Popen | None = None
        self._ready = False
        # 推論の直列化（第1段と同等）。HTTP 呼び出しは to_thread で走るため
        # asyncio.Lock ではなく threading.Lock を使う。
        self._infer_lock = threading.Lock()
        self._manage_lock = asyncio.Lock()
        self._start_history: list[float] = []
        if not self._external_url:
            atexit.register(self._terminate_worker)

    # ── ワーカー管理 ──────────────────────────────────────────

    @property
    def _base_url(self) -> str:
        if self._external_url:
            return self._external_url
        return f"http://127.0.0.1:{self._port}"

    @staticmethod
    def _project_root() -> Path:
        return Path(__file__).resolve().parents[2]

    def _pick_port(self) -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]

    def _health_sync(self) -> bool:
        try:
            with urllib.request.urlopen(
                    self._base_url + "/health", timeout=3) as r:
                data = json.loads(r.read())
            return data.get("status") == "ok" and bool(data.get("ready"))
        except Exception:
            return False

    def _start_worker_sync(self) -> bool:
        """ワーカーを起動し /health が応答するまで待つ（同期処理）"""
        if self._external_url:
            return self._health_sync()
        now = time.time()
        self._start_history = [t for t in self._start_history
                               if now - t < _RESTART_WINDOW_SEC]
        if len(self._start_history) >= _RESTART_MAX_STARTS:
            logger.error("Whisper worker restart budget exhausted",
                         starts=len(self._start_history),
                         window_sec=_RESTART_WINDOW_SEC)
            return False
        self._start_history.append(now)

        if self._port <= 0:
            self._port = self._pick_port()
        root = self._project_root()
        log_path = root / "logs" / "stt_worker.log"
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_file = open(log_path, "ab")
        except Exception:
            log_file = subprocess.DEVNULL
        cmd = [sys.executable, "-m", "backend.stt.whisper_worker",
               "--port", str(self._port),
               "--model", self._model_size,
               "--device", self._device,
               "--compute-type", self._compute_type,
               "--language", self._language or "ja",
               "--parent-pid", str(os.getpid())]
        kwargs = {}
        if os.name == "nt":
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
        try:
            self._proc = subprocess.Popen(
                cmd, cwd=str(root), stdout=log_file, stderr=log_file, **kwargs)
        except Exception as e:
            logger.error("Whisper worker spawn failed", error=str(e))
            self._proc = None
            return False
        logger.info("Whisper worker started", pid=self._proc.pid,
                    port=self._port, device=self._device,
                    compute_type=self._compute_type)

        deadline = time.time() + self._startup_timeout
        while time.time() < deadline:
            if self._proc.poll() is not None:
                logger.error("Whisper worker exited during startup",
                             returncode=self._proc.returncode)
                self._proc = None
                return False
            if self._health_sync():
                self._ready = True
                return True
            time.sleep(0.5)
        logger.error("Whisper worker startup timed out",
                     timeout_sec=self._startup_timeout)
        self._terminate_worker()
        return False

    def _terminate_worker(self) -> None:
        proc, self._proc = self._proc, None
        self._ready = False
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass

    async def _ensure_worker(self) -> bool:
        if self._external_url:
            if await asyncio.to_thread(self._health_sync):
                self._ready = True
                return True
            self._ready = False
            return False
        if self._proc is not None and self._proc.poll() is None \
                and self._ready:
            return True
        async with self._manage_lock:
            if self._proc is not None and self._proc.poll() is None \
                    and self._ready:
                return True
            # 死んでいる・未起動なら（再）起動する
            self._terminate_worker()
            return await asyncio.to_thread(self._start_worker_sync)

    # ── 推論 ──────────────────────────────────────────────────

    def _post_transcribe_sync(self, pcm16: bytes, sample_rate: int) -> dict:
        req = urllib.request.Request(
            self._base_url + "/transcribe", data=pcm16, method="POST",
            headers={"X-Sample-Rate": str(sample_rate)})
        with urllib.request.urlopen(req, timeout=self._request_timeout) as r:
            return json.loads(r.read())

    def _call_transcribe(self, pcm16: bytes, sample_rate: int) -> dict:
        """直列化ロックの中で HTTP 呼び出しする（to_thread から呼ぶ）"""
        wait_start = time.time()
        with self._infer_lock:
            waited_ms = int((time.time() - wait_start) * 1000)
            if waited_ms >= 100:
                logger.info("Whisper transcribe serialized (waited for lock)",
                            waited_ms=waited_ms)
            return self._post_transcribe_sync(pcm16, sample_rate)

    async def warmup(self) -> None:
        """ワーカーを起動し、無音1秒で初回推論も済ませておく"""
        if not await self._ensure_worker():
            logger.warning("Whisper warmup skipped: worker unavailable")
            return
        silence = b"\x00\x00" * 16000
        try:
            await asyncio.to_thread(self._call_transcribe, silence, 16000)
            logger.info("Whisper warmup done", model=self._model_size)
        except Exception as e:
            logger.warning("Whisper warmup failed", error=str(e))

    async def transcribe(self, pcm16: bytes, sample_rate: int = 16000) -> STTResult:
        if not pcm16:
            return STTResult(text="")
        duration_ms = int(len(pcm16) / 2 / sample_rate * 1000)
        if not await self._ensure_worker():
            return STTResult(text="", duration_ms=duration_ms)

        t0 = time.time()
        try:
            res = await asyncio.to_thread(
                self._call_transcribe, pcm16, sample_rate)
        except Exception as e:
            # ワーカー落下時は1回だけ再起動して再試行する
            logger.warning("Whisper worker transcribe failed; restarting",
                           error=str(e))
            self._ready = False
            if not await self._ensure_worker():
                return STTResult(text="", duration_ms=duration_ms)
            try:
                res = await asyncio.to_thread(
                    self._call_transcribe, pcm16, sample_rate)
            except Exception as e2:
                logger.warning("Whisper worker transcribe failed after restart",
                               error=str(e2))
                return STTResult(text="", duration_ms=duration_ms)

        latency_ms = int((time.time() - t0) * 1000)
        text = (res.get("text") or "").strip()
        logger.info("Transcribed", text=text[:40],
                    duration_ms=duration_ms, latency_ms=latency_ms)
        return STTResult(text=text,
                         language=res.get("language", "") or "",
                         duration_ms=duration_ms, latency_ms=latency_ms)

    def close(self) -> None:
        self._terminate_worker()

    @property
    def name(self) -> str:
        return f"faster-whisper/{self._model_size}(process:{self._device})"

    @property
    def is_ready(self) -> bool:
        return self._ready
