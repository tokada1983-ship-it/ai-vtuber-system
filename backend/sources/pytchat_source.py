"""PytchatSource: pytchat を使ったYouTubeライブチャット取得 (非公式・APIキー不要)

pytchat は内部で signal モジュールを使うためメインスレッド以外から呼べない。
multiprocessing で子プロセスを立ち上げて Queue 経由でコメントを受け取る方式で対応する。
"""
from __future__ import annotations

import asyncio
import multiprocessing
import multiprocessing.queues
from datetime import datetime
from typing import AsyncIterator

from backend.logger import get_logger
from backend.models import Comment
from backend.sources.base import CommentSource

logger = get_logger(__name__)


def _pytchat_worker(video_id: str, q: multiprocessing.Queue, stop_event: multiprocessing.Event) -> None:
    """子プロセスで動くpytchatワーカー。コメントをQueueに送る。"""
    try:
        import pytchat
    except ImportError:
        q.put({"error": "pytchat not installed"})
        return

    try:
        chat = pytchat.create(video_id=video_id)
    except Exception as e:
        q.put({"error": str(e)})
        return

    # 接続直後の最初のポーリングは過去コメントが大量に来るためスキップ
    try:
        first_items = chat.get()
        seen_ids = {item.id for item in first_items.sync_items()}
    except Exception:
        seen_ids = set()

    q.put({"status": "connected", "skipped": len(seen_ids)})

    while not stop_event.is_set() and chat.is_alive():
        try:
            items = chat.get()
            for item in items.sync_items():
                # 既に見たIDはスキップ
                if item.id in seen_ids:
                    continue
                seen_ids.add(item.id)
                # メモリ節約のため古いIDを削除（最新1000件だけ保持）
                if len(seen_ids) > 1000:
                    seen_ids = set(list(seen_ids)[-500:])

                author = item.author
                q.put({
                    "id":               item.id,
                    "author":           getattr(author, "name", "名無し") or "名無し",
                    "author_channel_id": getattr(author, "channelId", "") or "",
                    "message":          item.message,
                    "timestamp":        item.timestamp,
                    "is_superchat":     getattr(item, "type", "") == "superChat",
                    "superchat_amount": int(item.amountValue) if hasattr(item, "amountValue") and item.amountValue else None,
                    "is_member":        getattr(author, "isChatMember", False) or getattr(author, "is_chat_member", False),
                    "is_moderator":     getattr(author, "isChatModerator", False) or getattr(author, "is_chat_moderator", False),
                })
        except Exception as e:
            q.put({"error": str(e)})
            break
        import time
        time.sleep(0.5)

    try:
        chat.terminate()
    except Exception:
        pass
    q.put({"status": "disconnected"})


class PytchatSource(CommentSource):

    MAX_FAILURES = 5

    def __init__(self):
        self._running = False
        self._queue: asyncio.Queue[Comment] = asyncio.Queue()
        self._task: asyncio.Task | None = None
        self._failure_count = 0
        self._video_id = ""
        self._mp_queue: multiprocessing.Queue | None = None
        self._stop_event: multiprocessing.Event | None = None
        self._process: multiprocessing.Process | None = None

    async def start(self, video_id: str) -> None:
        self._video_id = video_id
        self._running = True
        self._failure_count = 0
        logger.info("PytchatSource starting", video_id=video_id)
        self._task = asyncio.create_task(self._fetch_loop())

    async def stop(self) -> None:
        self._running = False
        if self._stop_event:
            self._stop_event.set()
        if self._process and self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=3)
        if self._task:
            self._task.cancel()
        logger.info("PytchatSource stopped")

    async def _fetch_loop(self) -> None:
        while self._running:
            # 子プロセスを起動
            ctx = multiprocessing.get_context("spawn")
            self._mp_queue = ctx.Queue()
            self._stop_event = ctx.Event()
            self._process = ctx.Process(
                target=_pytchat_worker,
                args=(self._video_id, self._mp_queue, self._stop_event),
                daemon=True,
            )
            self._process.start()

            try:
                connected = False
                while self._running:
                    # ノンブロッキングでQueueからメッセージを取得
                    try:
                        msg = await asyncio.wait_for(
                            asyncio.get_event_loop().run_in_executor(
                                None, self._mp_queue.get, True, 2.0
                            ),
                            timeout=3.0,
                        )
                    except (asyncio.TimeoutError, Exception):
                        # プロセスが死んでいたら終了
                        if not self._process.is_alive():
                            break
                        continue

                    if "error" in msg:
                        self._failure_count += 1
                        logger.warning("PytchatSource error", error=msg["error"],
                                       failures=self._failure_count)
                        if self._failure_count >= self.MAX_FAILURES:
                            logger.error("PytchatSource: too many failures")
                            self._running = False
                        break

                    if msg.get("status") == "connected":
                        connected = True
                        self._failure_count = 0
                        skipped = msg.get("skipped", 0)
                        logger.info("pytchat connected",
                                    video_id=self._video_id,
                                    skipped_past=skipped)
                        continue

                    if msg.get("status") == "disconnected":
                        break

                    # コメントデータ
                    if connected and "id" in msg:
                        comment = Comment(
                            id=msg["id"],
                            author=msg["author"],
                            author_channel_id=msg["author_channel_id"],
                            message=msg["message"],
                            timestamp=datetime.fromtimestamp(msg["timestamp"] / 1000),
                            is_superchat=msg.get("is_superchat", False),
                            superchat_amount=msg.get("superchat_amount"),
                            is_member=msg.get("is_member", False),
                            is_moderator=msg.get("is_moderator", False),
                        )
                        await self._queue.put(comment)

            except asyncio.CancelledError:
                break
            finally:
                self._stop_event.set()
                if self._process.is_alive():
                    self._process.terminate()
                    self._process.join(timeout=3)

            if self._running and self._failure_count < self.MAX_FAILURES:
                logger.info("PytchatSource: reconnecting in 5s...")
                await asyncio.sleep(5.0)

    async def comments(self) -> AsyncIterator[Comment]:  # type: ignore[override]
        while self._running or not self._queue.empty():
            try:
                comment = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                yield comment
            except asyncio.TimeoutError:
                continue

    @property
    def is_healthy(self) -> bool:
        return self._running and self._failure_count < self.MAX_FAILURES

    @property
    def name(self) -> str:
        return "pytchat"
