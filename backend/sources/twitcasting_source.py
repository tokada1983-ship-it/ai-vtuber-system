"""TwitcastingSource: ツイキャス API v2 経由でライブコメントを取得する

必要な設定:
  config.yaml:
    twitcasting:
      access_token: "YOUR_ACCESS_TOKEN"   # または環境変数 TWITCASTING_ACCESS_TOKEN
      user_id: "your_screen_name"          # 自分のツイキャスID（@なし）
      polling_interval_ms: 2000

アクセストークンの取得:
  https://twitcasting.tv/developer.php でアプリ登録後、
  Implicit フローで取得: https://apiv2.twitcasting.tv/oauth2/authorize?client_id={CLIENT_ID}&response_type=token
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import AsyncIterator

import httpx

from backend.logger import get_logger
from backend.models import Comment
from backend.sources.base import CommentSource

logger = get_logger(__name__)

TC_API_BASE = "https://apiv2.twitcasting.tv"


class TwitcastingSource(CommentSource):
    """
    ツイキャス API v2 の GET /movies/{movie_id}/comments を
    ポーリングしてコメントを取得する。

    movie_id は start() に渡す video_id として使う。
    空文字の場合は user_id から現在のライブを自動取得する。
    """

    MAX_FAILURES = 5

    def __init__(
        self,
        access_token: str,
        user_id: str,
        polling_interval_ms: int = 2000,
    ):
        self._token = access_token
        self._user_id = user_id
        self._polling_interval = polling_interval_ms / 1000
        self._movie_id: str | None = None
        self._running = False
        self._queue: asyncio.Queue[Comment] = asyncio.Queue()
        self._task: asyncio.Task | None = None
        self._failure_count = 0
        self._seen_ids: set[str] = set()

    @property
    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self._token}",
            "Accept": "application/json",
            "X-Api-Version": "2.0",
        }

    async def _get_current_movie_id(self) -> str | None:
        """現在ライブ中のmovie_idを取得する"""
        url = f"{TC_API_BASE}/users/{self._user_id}/current_live"
        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                resp = await client.get(url, headers=self._headers)
                if resp.status_code == 404:
                    logger.info("TwitcastingSource: no live currently", user=self._user_id)
                    return None
                resp.raise_for_status()
                data = resp.json()
                movie_id = str(data["movie"]["id"])
                logger.info("TwitcastingSource: found live", movie_id=movie_id)
                return movie_id
            except Exception as e:
                logger.warning("TwitcastingSource: get_current_movie failed", error=str(e))
                return None

    async def start(self, video_id: str) -> None:
        self._running = True
        self._failure_count = 0
        self._seen_ids.clear()

        if video_id and video_id not in ("mock", ""):
            self._movie_id = video_id
        else:
            self._movie_id = await self._get_current_movie_id()

        if not self._movie_id:
            logger.warning("TwitcastingSource: no movie_id, polling for live start...")

        self._task = asyncio.create_task(self._poll_loop())
        logger.info("TwitcastingSource started", movie_id=self._movie_id, user=self._user_id)

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
        logger.info("TwitcastingSource stopped")

    async def _poll_loop(self) -> None:
        last_id: int | None = None

        async with httpx.AsyncClient(timeout=15.0) as client:
            while self._running:
                # movie_idがない場合は現在のライブを探す
                if not self._movie_id:
                    self._movie_id = await self._get_current_movie_id()
                    if not self._movie_id:
                        await asyncio.sleep(10.0)
                        continue

                try:
                    url = f"{TC_API_BASE}/movies/{self._movie_id}/comments"
                    params: dict = {"limit": 50}
                    if last_id is not None:
                        params["after_id"] = last_id

                    resp = await client.get(url, headers=self._headers, params=params)
                    resp.raise_for_status()
                    data = resp.json()

                    comments = data.get("comments", [])
                    for item in reversed(comments):  # 古い順に処理
                        comment = self._parse_item(item)
                        if comment and comment.id not in self._seen_ids:
                            self._seen_ids.add(comment.id)
                            await self._queue.put(comment)
                            if last_id is None or int(item["id"]) > last_id:
                                last_id = int(item["id"])

                    self._failure_count = 0

                except asyncio.CancelledError:
                    break
                except httpx.HTTPStatusError as e:
                    self._failure_count += 1
                    if e.response.status_code == 401:
                        logger.error("TwitcastingSource: access token expired or invalid")
                        self._failure_count = self.MAX_FAILURES
                        break
                    logger.warning("TwitcastingSource HTTP error", status=e.response.status_code)
                except Exception as e:
                    self._failure_count += 1
                    logger.warning("TwitcastingSource error", error=str(e))
                    if self._failure_count >= self.MAX_FAILURES:
                        logger.error("TwitcastingSource: too many failures")
                        break

                await asyncio.sleep(self._polling_interval)

    def _parse_item(self, item: dict) -> Comment | None:
        try:
            author = item.get("author", {})
            return Comment(
                id=str(item["id"]),
                author=author.get("name") or author.get("screen_id", "名無し"),
                author_channel_id=author.get("id", ""),
                message=item.get("message", ""),
                timestamp=datetime.fromtimestamp(
                    item.get("created", 0), tz=timezone.utc
                ).astimezone(),
                is_superchat=False,
                is_member=item.get("is_from_supporter", False),
                is_moderator=False,
            )
        except Exception as e:
            logger.debug("TwitcastingSource: parse error", error=str(e))
            return None

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
        return "twitcasting"
