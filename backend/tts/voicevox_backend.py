"""VOICEVOXBackend: VOICEVOXエンジンを使ったTTSバックエンド"""
from __future__ import annotations

import io
import wave

import httpx

from backend.config import TTSBackendConfig
from backend.logger import get_logger
from backend.models import TTSResult
from backend.tts.base import TTSBackend

logger = get_logger(__name__)


class VoicevoxBackend(TTSBackend):
    """
    VOICEVOX ENGINE (http://localhost:50021) を使って音声合成する。
    事前に VOICEVOX をローカルで起動しておく必要がある。
    """

    def __init__(self, cfg: TTSBackendConfig):
        self._cfg = cfg
        self._healthy = True

    async def synthesize(self, text: str, emotion: str | None = None) -> TTSResult:
        if not text.strip():
            return TTSResult(audio_wav=b"", duration_ms=0)

        # emotion に応じて速度・ピッチを微調整
        speed = self._cfg.speed
        pitch = self._cfg.pitch
        if emotion == "happy":
            speed += 0.05
            pitch += 0.02
        elif emotion == "sad":
            speed -= 0.05
            pitch -= 0.02
        elif emotion == "angry":
            speed += 0.1

        base = self._cfg.base_url.rstrip("/")

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                # Step 1: audio_query
                q_resp = await client.post(
                    f"{base}/audio_query",
                    params={"text": text, "speaker": self._cfg.speaker_id},
                )
                q_resp.raise_for_status()
                query = q_resp.json()

                query["speedScale"] = speed
                query["pitchScale"] = pitch

                # Step 2: synthesis
                s_resp = await client.post(
                    f"{base}/synthesis",
                    params={"speaker": self._cfg.speaker_id},
                    json=query,
                )
                s_resp.raise_for_status()
                wav_bytes = s_resp.content

                duration_ms = self._get_duration_ms(wav_bytes)
                self._healthy = True
                logger.debug("VOICEVOX synthesized", text=text[:30], duration_ms=duration_ms)
                return TTSResult(audio_wav=wav_bytes, duration_ms=duration_ms)

        except httpx.ConnectError:
            logger.warning("VOICEVOX: connection refused — is VOICEVOX running?")
            self._healthy = False
            return TTSResult(audio_wav=b"", duration_ms=0)
        except Exception as e:
            logger.warning("VOICEVOX error", error=str(e))
            self._healthy = False
            return TTSResult(audio_wav=b"", duration_ms=0)

    def _get_duration_ms(self, wav_bytes: bytes) -> int:
        try:
            with io.BytesIO(wav_bytes) as buf:
                with wave.open(buf, "rb") as wf:
                    frames = wf.getnframes()
                    rate = wf.getframerate()
                    return int(frames / rate * 1000)
        except Exception:
            return 0

    @property
    def name(self) -> str:
        return "voicevox"

    @property
    def is_healthy(self) -> bool:
        return self._healthy
