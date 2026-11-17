"""YouTubeDataAPISource: YouTube Data API v3 の公式エンドポイントを使ったチャット取得"""
from __future__ import annotations

import asyncio
from datetime import datetime
from typing import AsyncIterator

import httpx

from backend.logger import get_logger
from backend.models import Comment
from backend.sources.base import CommentSource

logger = get_logger(__name__)

YT_API_BASE = "https://www.googleapis.com/youtube/v3"


class YouTubeDataAPISource(CommentSource):
    """
    公式 YouTube Data API v3 経由でライブチャットを取得する。
    - 必要: YOUTUBE_API_KEY 環境変数
    - クォータ: 1リクエストあたり約5 units。デフォルト10,000 units/日
    - ポーリング間隔はAPIが返す pollingIntervalMillis に従う
    """

    def __init__(self, api_key: str):
        self._api_key = api_key
        self._running = False
        self._queue: asyncio.Queue[Comment] = asyncio.Queue()
        self._task: asyncio.Task | None = None
        self._failure_count = 0
        self._live_chat_id: str | None = None
        self.MAX_FAILURES = 5

    async def start(self, video_id: str) -> None:
        self._running = True
        self._failure_count = 0
        logger.info("YouTubeDataAPISource starting", video_id=video_id)

        live_chat_id = await self._get_live_chat_id(video_id)
        if not live_chat_id:
            logger.error("Failed to retrieve liveChatId", video_id=video_id)
            self._failure_count = self.MAX_FAILURES
            return

        self._live_chat_id = live_chat_id
        self._task = asyncio.create_task(self._poll_loop())

    async def _get_live_chat_id(self, video_id: str) -> str | None:
        url = f"{YT_API_BASE}/videos"
        params = {
            "part": "liveStreamingDetails",
            "id": video_id,
            "key": self._api_key,
        }
        async with httpx.AsyncClient() as client:
            try:
                resp = await client.get(url, params=params, timeout=10.0)
                resp.raise_for_status()
                items = resp.json().get("items", [])
                if not items:
                    return None
                return items[0].get("liveStreamingDetails", {}).get("activeLiveChatId")
            except Exception as e:
                logger.error("get_live_chat_id failed", error=str(e))
                return None

    async def _poll_loop(self) -> None:
        page_token: str | None = None
        polling_interval = 5.0

        async with httpx.AsyncClient() as client:
            while self._running:
                try:
                    url = f"{YT_API_BASE}/liveChat/messages"
                    params = {
                        "liveChatId": self._live_chat_id,
                        "part": "snippet,authorDetails",
                        "key": self._api_key,
                        "maxResults": 200,
                    }
                    if page_token:
                        params["pageToken"] = page_token

                    resp = await client.get(url, params=params, timeout=15.0)
                    resp.raise_for_status()
                    data = resp.json()

                    for item in data.get("items", []):
                        comment = self._parse_item(item)
                        if comment:
                            await self._queue.put(comment)

                    page_token = data.get("nextPageToken")
                    polling_interval = data.get("pollingIntervalMillis", 5000) / 1000
                    self._failure_count = 0

                except asyncio.CancelledError:
                    break
                except httpx.HTTPStatusError as e:
                    self._failure_count += 1
                    if e.response.status_code == 403:
                        logger.error("YouTube API quota exceeded or auth error")
                        self._failure_count = self.MAX_FAILURES
                        break
                    logger.warning("YouTube API HTTP error", status=e.response.status_code)
                    await asyncio.sleep(10.0)
                    continue
                except Exception as e:
                    self._failure_count += 1
                    logger.warning("YouTube API error", error=str(e))
                    if self._failure_count >= self.MAX_FAILURES:
                        logger.error("YouTubeDataAPISource: too many failures")
                        break
                    await asyncio.sleep(5.0)
                    continue

                await asyncio.sleep(polling_interval)

    def _parse_item(self, item: dict) -> Comment | None:
        try:
            snippet = item.get("snippet", {})
            author = item.get("authorDetails", {})
            msg_type = snippet.get("type", "")
            is_sc = msg_type == "superChatEvent"

            return Comment(
                id=item["id"],
                author=author.get("displayName", "名無し"),
                author_channel_id=author.get("channelId", ""),
                message=snippet.get("displayMessage", ""),
                timestamp=datetime.fromisoformat(
                    snippet.get("publishedAt", datetime.now().isoformat()).replace("Z", "+00:00")
                ),
                is_superchat=is_sc,
                superchat_amount=int(snippet.get("superChatDetails", {}).get("amountMicros", 0)) // 1_000_000
                if is_sc else None,
                is_member=author.get("isChatMember", False),
                is_moderator=author.get("isChatModerator", False),
            )
        except Exception as e:
            logger.debug("Failed to parse YT API comment", error=str(e))
            return None

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
        logger.info("YouTubeDataAPISource stopped")

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
