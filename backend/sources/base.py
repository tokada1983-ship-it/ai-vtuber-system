"""CommentSource インターフェース"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncIterator

from backend.models import Comment


class CommentSource(ABC):
    """コメント取得レイヤの抽象基底クラス。全ての実装はこれを継承する。"""

    @abstractmethod
    async def start(self, video_id: str) -> None:
        """ストリーミング開始。video_id はYouTubeのビデオID。"""

    @abstractmethod
    async def stop(self) -> None:
        """ストリーミング停止。"""

    @abstractmethod
    def comments(self) -> AsyncIterator[Comment]:
        """非同期イテレータ。新着コメントを順に yield する。"""

    @property
    @abstractmethod
    def is_healthy(self) -> bool:
        """接続が正常かどうか。health_check_loopが参照する。"""

    @property
    def name(self) -> str:
        return self.__class__.__name__
