"""STT（faster-whisper）推論の直列化テスト

2026-11-20 の APPCRASH 対策 第1段。WhisperModel は同時 transcribe
呼び出しにスレッド安全でないため、FasterWhisperBackend._run_transcribe
が threading.Lock で直列化されることを検証する。
実モデルは使わず、同時実行数を数えるフェイクモデルで置き換える。
"""
from __future__ import annotations

import asyncio
import threading
import time

import numpy as np

from backend.stt.whisper_backend import FasterWhisperBackend


class _FakeInfo:
    language = "ja"


class _FakeSeg:
    def __init__(self, text):
        self.text = text


class FakeModel:
    """同時実行の最大値を記録するフェイク WhisperModel"""

    def __init__(self, delay: float = 0.05):
        self.delay = delay
        self._lock = threading.Lock()
        self._current = 0
        self.max_concurrent = 0
        self.calls = 0

    def transcribe(self, audio, **kwargs):
        with self._lock:
            self._current += 1
            self.calls += 1
            self.max_concurrent = max(self.max_concurrent, self._current)
        try:
            time.sleep(self.delay)
            return [_FakeSeg("テスト")], _FakeInfo()
        finally:
            with self._lock:
                self._current -= 1


def _backend_with_fake() -> tuple[FasterWhisperBackend, FakeModel]:
    backend = FasterWhisperBackend(model_size="small", device="cpu")
    fake = FakeModel()
    backend._model = fake  # ロード済みにして実モデルを使わない
    return backend, fake


async def test_transcribe_calls_are_serialized():
    backend, fake = _backend_with_fake()
    pcm = b"\x00\x00" * 16000  # 1秒分の無音PCM16
    results = await asyncio.gather(
        *(backend.transcribe(pcm) for _ in range(6)))
    assert all(r.text == "テスト" for r in results)
    assert fake.calls == 6
    assert fake.max_concurrent == 1


async def test_run_transcribe_direct_calls_are_serialized():
    """warmup 経路（_run_transcribe の直接呼び出し）も直列化される"""
    backend, fake = _backend_with_fake()
    audio = np.zeros(16000, dtype=np.float32)
    results = await asyncio.gather(
        *(asyncio.to_thread(backend._run_transcribe, audio)
          for _ in range(4)))
    assert all(text == "テスト" for text, _lang in results)
    assert fake.max_concurrent == 1


def test_infer_lock_is_threading_lock():
    backend, _fake = _backend_with_fake()
    assert isinstance(backend._infer_lock, type(threading.Lock()))
