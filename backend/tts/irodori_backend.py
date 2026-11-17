"""IrodoriTTSBackend: Irodori-TTS-Server (OpenAI互換 /v1/audio/speech) を使ったTTSバックエンド

Irodori-TTS-Server を事前に起動しておく必要がある。
  GitHub: https://github.com/Aratako/Irodori-TTS-Server
  デフォルトポート: 8088

voices/ ディレクトリにリファレンス音声(.wav)を置き、
config.yaml の voice に そのファイル名（拡張子なし）を指定する。
"""
from __future__ import annotations

import io
import wave

import httpx

from backend.logger import get_logger
from backend.models import TTSResult
from backend.tts.base import TTSBackend

logger = get_logger(__name__)

# emotion → voice サフィックスマッピング（voices/ 以下のファイル名規則に合わせて変更可）
_EMOTION_VOICE_MAP: dict[str, str] = {
    "happy":     "happy",
    "sad":       "sad",
    "angry":     "angry",
    "surprised": "surprised",
    "relaxed":   "relaxed",
    "neutral":   "neutral",
}


class IrodoriTTSConfig:
    """IrodoriTTS専用設定（config.yaml の tts.backends.irodori に対応）"""
    def __init__(
        self,
        base_url: str = "http://localhost:8088/v1",
        model: str = "irodori-tts",
        voice: str = "default",                  # voices/ 以下のベース名
        use_emotion_voice: bool = False,          # True なら emotion に応じて voice を切り替える
        emotion_voice_prefix: str = "",           # "{prefix}-{emotion}" 形式で voice を選択
        response_format: str = "wav",
        api_key: str = "not-used",
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.voice = voice
        self.use_emotion_voice = use_emotion_voice
        self.emotion_voice_prefix = emotion_voice_prefix
        self.response_format = response_format
        self.api_key = api_key

    @classmethod
    def from_dict(cls, d: dict) -> "IrodoriTTSConfig":
        return cls(
            base_url=d.get("base_url", "http://localhost:8088/v1"),
            model=d.get("model", "irodori-tts"),
            voice=d.get("voice", "default"),
            use_emotion_voice=d.get("use_emotion_voice", False),
            emotion_voice_prefix=d.get("emotion_voice_prefix", ""),
            response_format=d.get("response_format", "wav"),
            api_key=d.get("api_key", "not-used"),
        )


class IrodoriTTSBackend(TTSBackend):
    """
    Irodori-TTS-Server の OpenAI互換エンドポイントを叩く TTSバックエンド。

    設定例 (config.yaml):
      tts:
        primary: irodori
        backends:
          irodori:
            base_url: "http://localhost:8088/v1"
            model: "irodori-tts"
            voice: "Character2-neutral"          # voices/Character2-neutral.wav に対応
            use_emotion_voice: true          # emotion に応じて Character2-{emotion} に切り替える
            emotion_voice_prefix: "Character2"
            response_format: "wav"
    """

    def __init__(self, cfg: IrodoriTTSConfig, timeout: float = 120.0):
        self._cfg = cfg
        self._healthy = True
        self._timeout = timeout

    def _resolve_voice(self, emotion: str | None) -> str:
        """emotionに応じてvoice名を決定する"""
        if not self._cfg.use_emotion_voice or not emotion:
            return self._cfg.voice

        emotion_suffix = _EMOTION_VOICE_MAP.get(emotion, "neutral")
        prefix = self._cfg.emotion_voice_prefix or self._cfg.voice
        return f"{prefix}-{emotion_suffix}"

    async def synthesize(self, text: str, emotion: str | None = None) -> TTSResult:
        if not text.strip():
            return TTSResult(audio_wav=b"", duration_ms=0)

        voice = self._resolve_voice(emotion)
        url = f"{self._cfg.base_url}/audio/speech"

        payload = {
            "model": self._cfg.model,
            "input": text,
            "voice": voice,
            "response_format": self._cfg.response_format,
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._cfg.api_key}",
        }

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(url, json=payload, headers=headers)
                resp.raise_for_status()
                audio_bytes = resp.content

            # WAV以外のフォーマットはduration計算をスキップ
            duration_ms = 0
            if self._cfg.response_format == "wav":
                duration_ms = self._get_wav_duration_ms(audio_bytes)

            self._healthy = True
            logger.debug(
                "IrodoriTTS synthesized",
                text=text[:30],
                voice=voice,
                duration_ms=duration_ms,
            )
            return TTSResult(audio_wav=audio_bytes, duration_ms=duration_ms)

        except httpx.ConnectError:
            logger.warning(
                "IrodoriTTS: connection refused — is Irodori-TTS-Server running?",
                url=url,
            )
            self._healthy = False
            return TTSResult(audio_wav=b"", duration_ms=0)
        except httpx.TimeoutException:
            # タイムアウトは同時リクエストの詰まりが主因。サーバ自体は生きている可能性が高い
            logger.warning("IrodoriTTS: timed out", url=url,
                           timeout_sec=self._timeout, text=text[:20])
            return TTSResult(audio_wav=b"", duration_ms=0)
        except httpx.HTTPStatusError as e:
            logger.warning(
                "IrodoriTTS HTTP error",
                status=e.response.status_code,
                body=e.response.text[:200],
                voice=voice,
            )
            self._healthy = False
            return TTSResult(audio_wav=b"", duration_ms=0)
        except Exception as e:
            # str(e) が空になる例外があるため型と repr を必ず残す
            logger.warning("IrodoriTTS error",
                           error_type=type(e).__name__, error=repr(e))
            self._healthy = False
            return TTSResult(audio_wav=b"", duration_ms=0)

    def _get_wav_duration_ms(self, wav_bytes: bytes) -> int:
        try:
            with io.BytesIO(wav_bytes) as buf:
                with wave.open(buf, "rb") as wf:
                    return int(wf.getnframes() / wf.getframerate() * 1000)
        except Exception:
            return 0

    @property
    def name(self) -> str:
        return "irodori"

    @property
    def is_healthy(self) -> bool:
        return self._healthy
