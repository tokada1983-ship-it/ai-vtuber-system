"""AudioRouter: WAV音声を仮想オーディオデバイスまたはデフォルトデバイスへ出力する"""
from __future__ import annotations

import asyncio
import io
import wave

from backend.logger import get_logger

logger = get_logger(__name__)


class AudioRouter:
    """
    TTSResultのWAVバイト列を sounddevice で再生する。
    device=None のときはシステムのデフォルトデバイスを使用。
    配信用途では device を "CABLE Input (VB-Audio Virtual Cable)" 等の仮想デバイスに設定する。
    """

    def __init__(self, device: str | int | None = None):
        self._device = device

    async def play(self, wav_bytes: bytes) -> None:
        """WAVを再生し、再生終了まで待機する（非同期）"""
        if not wav_bytes:
            return
        await asyncio.to_thread(self._play_sync, wav_bytes)

    def _play_sync(self, wav_bytes: bytes) -> None:
        try:
            import sounddevice as sd
            import numpy as np

            with io.BytesIO(wav_bytes) as buf:
                with wave.open(buf, "rb") as wf:
                    sample_rate = wf.getframerate()
                    n_channels = wf.getnchannels()
                    sampwidth = wf.getsampwidth()
                    frames = wf.readframes(wf.getnframes())

            dtype_map = {1: np.int8, 2: np.int16, 4: np.int32}
            dtype = dtype_map.get(sampwidth, np.int16)
            audio = np.frombuffer(frames, dtype=dtype)

            if n_channels > 1:
                audio = audio.reshape(-1, n_channels)

            sd.play(audio, samplerate=sample_rate, device=self._device)
            sd.wait()

        except Exception as e:
            logger.warning("AudioRouter play error", error=str(e))

    @staticmethod
    def list_devices() -> list[dict]:
        """利用可能なオーディオデバイス一覧を返す（設定確認用）"""
        try:
            import sounddevice as sd
            devices = sd.query_devices()
            return [
                {"index": i, "name": d["name"], "output_channels": d["max_output_channels"]}
                for i, d in enumerate(devices)
                if d["max_output_channels"] > 0
            ]
        except Exception as e:
            logger.warning("list_devices error", error=str(e))
            return []
