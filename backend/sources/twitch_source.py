"""TwitchSource: Twitch IRC over WebSocket でライブチャットを取得する

必要な設定:
  config.yaml:
    twitch:
      token: "oauth:YOUR_TOKEN"    # https://twitchtokengenerator.com/ で取得
      nick: "your_bot_account"     # Twitchアカウント名（小文字）
      channel: "your_channel"      # 配信チャンネル名（@なし、小文字）

トークンの取得:
  https://twitchtokengenerator.com/ で chat:read スコープのトークンを生成
  "oauth:" プレフィックスを付けて設定する
"""
from __future__ import annotations

import asyncio
import re
from datetime import datetime
from typing import AsyncIterator

from backend.logger import get_logger
from backend.models import Comment
from backend.sources.base import CommentSource

logger = get_logger(__name__)

TWITCH_IRC_URI = "wss://irc-ws.chat.twitch.tv:443"


class TwitchSource(CommentSource):
    """
    Twitch IRC over WebSocket を使ってチャットを取得する。
    twitchio ライブラリを使わずに軽量な独自実装。
    video_id には配信チャンネル名を渡す（channel設定と同じ）。
    """

    MAX_FAILURES = 5

    def __init__(self, token: str, nick: str, channel: str):
        self._token = token if token.startswith("oauth:") else f"oauth:{token}"
        self._nick = nick.lower()
        self._channel = channel.lower().lstrip("#")
        self._running = False
        self._queue: asyncio.Queue[Comment] = asyncio.Queue()
        self._task: asyncio.Task | None = None
        self._failure_count = 0

    async def start(self, video_id: str) -> None:
        self._running = True
        self._failure_count = 0
        # video_id でチャンネルを上書き可能
        if video_id and video_id not in ("mock", ""):
            self._channel = video_id.lower().lstrip("#")
        self._task = asyncio.create_task(self._connect_loop())
        logger.info("TwitchSource starting", channel=self._channel)

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
        logger.info("TwitchSource stopped")

    async def _connect_loop(self) -> None:
        import websockets

        while self._running:
            try:
                async with websockets.connect(TWITCH_IRC_URI) as ws:
                    # 認証
                    await ws.send(f"PASS {self._token}")
                    await ws.send(f"NICK {self._nick}")
                    await ws.send("CAP REQ :twitch.tv/tags twitch.tv/commands")
                    await ws.send(f"JOIN #{self._channel}")

                    logger.info("TwitchSource connected", channel=self._channel)
                    self._failure_count = 0

                    async for raw in ws:
                        if not self._running:
                            break
                        for line in raw.strip().split("\r\n"):
                            comment = self._parse_line(line)
                            if comment:
                                await self._queue.put(comment)
                            # PING/PONG keepalive
                            if line.startswith("PING"):
                                await ws.send("PONG :tmi.twitch.tv")

            except asyncio.CancelledError:
                break
            except Exception as e:
                self._failure_count += 1
                logger.warning("TwitchSource connection error", error=str(e), failures=self._failure_count)
                if self._failure_count >= self.MAX_FAILURES:
                    logger.error("TwitchSource: too many failures")
                    break
                await asyncio.sleep(5.0)

    # IRCメッセージのタグをパース
    _TAG_RE = re.compile(r"@([^ ]+) :(\S+) PRIVMSG #(\S+) :(.*)")
    _SIMPLE_RE = re.compile(r":(\S+)!(\S+)@(\S+) PRIVMSG #(\S+) :(.*)")
    _TAG_PARSE = re.compile(r"([^=;]+)=([^;]*)")

    def _parse_line(self, line: str) -> Comment | None:
        try:
            tags: dict = {}
            display_name = ""
            user_id = ""
            is_sub = False
            is_mod = False

            if line.startswith("@"):
                m = self._TAG_RE.match(line)
                if not m:
                    return None
                tag_str, full_user, channel, message = m.groups()
                # タグをパース
                for tm in self._TAG_PARSE.finditer(tag_str):
                    tags[tm.group(1)] = tm.group(2)
                display_name = tags.get("display-name", full_user.split("!")[0])
                user_id = tags.get("user-id", "")
                is_sub = tags.get("subscriber") == "1"
                is_mod = tags.get("mod") == "1"
                comment_id = tags.get("id", f"twitch_{datetime.now().timestamp()}")
            else:
                m2 = self._SIMPLE_RE.match(line)
                if not m2:
                    return None
                nick, _, _, channel, message = m2.groups()
                display_name = nick
                user_id = nick
                comment_id = f"twitch_{datetime.now().timestamp()}"

            if not message:
                return None

            return Comment(
                id=comment_id,
                author=display_name,
                author_channel_id=user_id,
                message=message,
                timestamp=datetime.now(),
                is_superchat=False,
                is_member=is_sub,
                is_moderator=is_mod,
            )
        except Exception as e:
            logger.debug("TwitchSource: parse error", error=str(e))
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
        return "twitch"
