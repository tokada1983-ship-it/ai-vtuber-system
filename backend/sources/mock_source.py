"""MockSource: 開発・テスト用コメントソース。JSONLファイルまたはハードコードリストから再生。"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import AsyncIterator

from backend.logger import get_logger
from backend.models import Comment
from backend.sources.base import CommentSource

logger = get_logger(__name__)

# デフォルトのサンプルコメント
_SAMPLE_COMMENTS = [
    {"author": "ユーザーA", "message": "こんにちは！配信みてるよ！"},
    {"author": "ユーザーB", "message": "今日も元気そうで良かった"},
    {"author": "ユーザーC", "message": "最近どんなゲームやってるの？"},
    {"author": "ユーザーD", "message": "プログラミングって難しいですか？"},
    {"author": "ユーザーE", "message": "Character2ちゃん大好きー！"},
    {"author": "ユーザーF", "message": "このシステムどうやって作ったの？"},
    {"author": "ユーザーG", "message": "配信いつも楽しみにしてます"},
    {"author": "ユーザーH", "message": "今日の天気どう思う？"},
    {"author": "ユーザーI", "message": "AIってすごいね！"},
    {"author": "ユーザーJ", "message": "次はどんな話する？"},
]


class MockSource(CommentSource):
    """
    開発用モックソース。
    - jsonl_path を指定すると、JSONLファイルからコメントを再生
    - 未指定ならサンプルコメントをループ
    - interval_sec で投稿間隔を調整（デフォルト3秒）
    """

    def __init__(
        self,
        jsonl_path: str | Path | None = None,
        interval_sec: float = 3.0,
        loop: bool = True,
    ):
        self._jsonl_path = Path(jsonl_path) if jsonl_path else None
        self._interval_sec = interval_sec
        self._loop = loop
        self._running = False
        self._queue: asyncio.Queue[Comment] = asyncio.Queue()
        self._task: asyncio.Task | None = None
        self._comment_list: list[dict] = []

    async def start(self, video_id: str) -> None:
        self._running = True
        if self._jsonl_path and self._jsonl_path.exists():
            with open(self._jsonl_path, encoding="utf-8") as f:
                self._comment_list = [json.loads(line) for line in f if line.strip()]
        else:
            self._comment_list = _SAMPLE_COMMENTS

        logger.info("MockSource started", count=len(self._comment_list), video_id=video_id)
        self._task = asyncio.create_task(self._feed())

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
        logger.info("MockSource stopped")

    async def _feed(self) -> None:
        idx = 0
        while self._running:
            if idx >= len(self._comment_list):
                if self._loop:
                    idx = 0
                else:
                    break
            raw = self._comment_list[idx]
            comment = Comment(
                id=f"mock_{idx:04d}",
                author=raw.get("author", "名無し"),
                author_channel_id=f"mock_ch_{idx}",
                message=raw.get("message", ""),
                timestamp=datetime.now(),
                is_superchat=raw.get("is_superchat", False),
                is_member=raw.get("is_member", False),
                is_moderator=raw.get("is_moderator", False),
            )
            await self._queue.put(comment)
            idx += 1
            await asyncio.sleep(self._interval_sec)

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
