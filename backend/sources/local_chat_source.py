"""LocalChatSource: ローカルユーザーと1対1でチャットするためのコメントソース

配信サイトにはつながらない。ブラウザのチャット画面（/chat）やマイク入力から
push() された発言だけを流す、完全ローカルのテストモード用。

MockSource との違い:
    MockSource       … サンプル発言を勝手に流し続ける（無人での動作確認用）
    LocalChatSource  … 人が入力したときだけ流す（1対1の会話・キャラ調整用）
"""
from __future__ import annotations

import asyncio
import itertools
from datetime import datetime
from typing import AsyncIterator

from backend.logger import get_logger
from backend.models import Comment
from backend.sources.base import CommentSource

logger = get_logger(__name__)


class LocalChatSource(CommentSource):
    """ローカル1対1チャット用のソース"""

    def __init__(self, user_name: str = "マスター", max_pending: int = 20):
        self._user_name = user_name
        self._running = False
        self._queue: asyncio.Queue[Comment] = asyncio.Queue()
        self._seq = itertools.count(1)
        self._max_pending = max_pending

    # ── ライフサイクル ────────────────────────────────────────

    async def start(self, video_id: str) -> None:
        self._running = True
        logger.info("LocalChatSource started", user=self._user_name)

    async def stop(self) -> None:
        self._running = False
        # 停止時に溜まっている未処理の発言は捨てる（再開時に昔の発言が飛ばないように）
        while not self._queue.empty():
            self._queue.get_nowait()
        logger.info("LocalChatSource stopped")

    @property
    def user_name(self) -> str:
        return self._user_name

    @user_name.setter
    def user_name(self, name: str) -> None:
        if name.strip():
            self._user_name = name.strip()

    @property
    def pending(self) -> int:
        return self._queue.qsize()

    # ── 入力 ──────────────────────────────────────────────────

    async def push(self, text: str, author: str | None = None) -> Comment | None:
        """
        ローカルユーザーの発言を1件流し込む。
        空文字や、停止中・詰まりすぎているときは None を返す。
        """
        text = (text or "").strip()
        if not text or not self._running:
            return None
        if self._queue.qsize() >= self._max_pending:
            logger.warning("LocalChatSource queue full, dropping input", text=text[:30])
            return None

        n = next(self._seq)
        comment = Comment(
            id=f"local_{n:05d}",
            author=(author or self._user_name).strip() or self._user_name,
            author_channel_id="local_user",
            message=text,
            timestamp=datetime.now(),
        )
        await self._queue.put(comment)
        return comment

    # ── 出力 ──────────────────────────────────────────────────

    async def comments(self) -> AsyncIterator[Comment]:  # type: ignore[override]
        while self._running or not self._queue.empty():
            try:
                comment = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                yield comment
            except asyncio.TimeoutError:
                continue

    @property
    def is_healthy(self) -> bool:
        return self._running
