"""STTBackend インターフェース（音声認識 = hearing）"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class STTResult:
    text: str
    language: str = ""
    duration_ms: int = 0
    latency_ms: int = 0

    @property
    def ok(self) -> bool:
        return bool(self.text.strip())


class STTBackend(ABC):
    """
    16kHz mono PCM16 のバイト列を受け取ってテキストに変換する。
    失敗時は例外を投げず、空テキストの STTResult を返す。
    """

    @abstractmethod
    async def transcribe(self, pcm16: bytes, sample_rate: int = 16000) -> STTResult: ...

    @abstractmethod
    async def warmup(self) -> None:
        """モデルを事前ロードする（初回発話の待ち時間をなくす）"""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def is_ready(self) -> bool: ...
