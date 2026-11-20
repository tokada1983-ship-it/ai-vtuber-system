"""FasterWhisperBackend: ローカルGPUで動く音声認識

faster-whisper (CTranslate2) を使う。GPUがあれば float16、無ければ CPU int8。
モデルは初回使用時に遅延ロードし、以降はメモリに常駐する。

インストール:
    pip install faster-whisper

モデルサイズの目安（日本語）:
    tiny   : 速いが精度は低い
    base   : 軽量。短い指示程度なら実用
    small  : バランス型（推奨）
    medium : 高精度・要VRAM
    large-v3: 最高精度・要VRAM
"""
from __future__ import annotations

import asyncio
import threading
import time

import numpy as np

from backend.logger import get_logger
from backend.stt.base import STTBackend, STTResult

logger = get_logger(__name__)


class FasterWhisperBackend(STTBackend):

    def __init__(
        self,
        model_size: str = "small",
        device: str = "auto",
        language: str = "ja",
        compute_type: str = "",
    ):
        self._model_size = model_size
        self._language = language or None
        self._device, self._compute_type = self._resolve_device(device, compute_type)
        self._model = None
        self._load_lock = asyncio.Lock()
        # 推論の直列化用ロック。WhisperModel は同時 transcribe 呼び出しに
        # スレッド安全でないため、ネイティブ推論は必ず1つずつ実行する。
        # 推論は asyncio.to_thread のワーカースレッドで走るので
        # asyncio.Lock ではなく threading.Lock を使う。
        self._infer_lock = threading.Lock()
        self._load_failed = False

    @staticmethod
    def _resolve_device(device: str, compute_type: str) -> tuple[str, str]:
        """device='auto' ならCUDAが使えるか判定する"""
        resolved = device
        if device in ("", "auto"):
            try:
                import torch
                resolved = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                resolved = "cpu"
        if compute_type:
            return resolved, compute_type
        return resolved, ("float16" if resolved == "cuda" else "int8")

    def _load_model(self):
        """同期のモデルロード（to_thread から呼ぶ）"""
        from faster_whisper import WhisperModel
        t0 = time.time()
        model = WhisperModel(
            self._model_size,
            device=self._device,
            compute_type=self._compute_type,
        )
        logger.info("Whisper model loaded",
                    model=self._model_size, device=self._device,
                    compute_type=self._compute_type,
                    load_sec=round(time.time() - t0, 1))
        return model

    async def _ensure_model(self) -> bool:
        if self._model is not None:
            return True
        if self._load_failed:
            return False
        async with self._load_lock:
            if self._model is not None:
                return True
            try:
                self._model = await asyncio.to_thread(self._load_model)
                return True
            except ImportError:
                logger.error("faster-whisper is not installed — run: pip install faster-whisper")
                self._load_failed = True
                return False
            except Exception as e:
                logger.error("Whisper model load failed",
                             model=self._model_size, error=str(e))
                self._load_failed = True
                return False

    async def warmup(self) -> None:
        """モデルを事前ロードして、無音1秒を流して初回推論も済ませておく"""
        if not await self._ensure_model():
            return
        silence = np.zeros(16000, dtype=np.float32)
        try:
            await asyncio.to_thread(self._run_transcribe, silence)
            logger.info("Whisper warmup done", model=self._model_size)
        except Exception as e:
            logger.warning("Whisper warmup failed", error=str(e))

    def _run_transcribe(self, audio: np.ndarray) -> tuple[str, str]:
        """同期の推論本体（to_thread から呼ぶ）

        transcribe 本体と warmup の両経路がここを通る。WhisperModel の
        同時呼び出しはネイティブ状態を壊しうるため、_infer_lock で
        直列化する（2026-11-20 の APPCRASH 対策 第1段）。
        """
        wait_start = time.time()
        with self._infer_lock:
            waited_ms = int((time.time() - wait_start) * 1000)
            if waited_ms >= 100:
                logger.info("Whisper transcribe serialized (waited for lock)",
                            waited_ms=waited_ms)
            segments, info = self._model.transcribe(
                audio,
                language=self._language,
                beam_size=5,
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 300},
                condition_on_previous_text=False,  # 幻聴（前文脈の引きずり）を防ぐ
            )
            text = "".join(seg.text for seg in segments).strip()
            return text, getattr(info, "language", "") or ""

    async def transcribe(self, pcm16: bytes, sample_rate: int = 16000) -> STTResult:
        if not pcm16:
            return STTResult(text="")
        if not await self._ensure_model():
            return STTResult(text="")

        # PCM16LE → float32 [-1.0, 1.0]
        audio = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0
        duration_ms = int(len(audio) / sample_rate * 1000)

        t0 = time.time()
        try:
            text, lang = await asyncio.to_thread(self._run_transcribe, audio)
        except Exception as e:
            logger.warning("Whisper transcribe failed", error=str(e))
            return STTResult(text="", duration_ms=duration_ms)

        latency_ms = int((time.time() - t0) * 1000)
        logger.info("Transcribed", text=text[:40], duration_ms=duration_ms, latency_ms=latency_ms)
        return STTResult(text=text, language=lang,
                         duration_ms=duration_ms, latency_ms=latency_ms)

    @property
    def name(self) -> str:
        return f"faster-whisper/{self._model_size}({self._device})"

    @property
    def is_ready(self) -> bool:
        return self._model is not None
