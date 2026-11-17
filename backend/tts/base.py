"""TTSBackend インターフェース"""
from __future__ import annotations

from abc import ABC, abstractmethod

from backend.models import TTSResult


class TTSBackend(ABC):

    @abstractmethod
    async def synthesize(self, text: str, emotion: str | None = None) -> TTSResult:
        """テキストをWAV音声に変換して返す。失敗時は空のTTSResultを返す。"""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def is_healthy(self) -> bool: ...
