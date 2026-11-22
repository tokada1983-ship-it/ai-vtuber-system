"""発言ログ欄（speechlog チャネル）への相乗り配送の単体テスト。

2026-11-22 panelfix: 配信画面右端の発言ログ欄は、既存のイベント配送に
相乗りして (1) キャラの発話 = send_chat(assistant) 経由で1発話1件、
(2) 拾ったコメント = send_comment 経由（呼び出しは応答キュー投入時に
限られる）で受け取る。本テストはその配送と履歴再送を検証する。
"""
from __future__ import annotations

import asyncio
import json

from backend.avatar.ws_hub import WebSocketHub


class _FakeWS:
    """受け取ったテキストを貯めるだけの WebSocket スタブ"""

    def __init__(self):
        self.sent: list[dict] = []
        self.accepted = False

    async def accept(self):
        self.accepted = True

    async def send_text(self, text: str):
        self.sent.append(json.loads(text))

    def of_type(self, type_: str) -> list[dict]:
        return [m for m in self.sent if m.get("type") == type_]


def _run(coro):
    return asyncio.run(coro)


def test_speech_chat_assistant_flows_to_speechlog():
    async def main():
        hub, log_ws = await _setup()
        chat_ws = _FakeWS()
        await hub.connect(chat_ws, "chat")
        await hub.send_chat("assistant", "こんにちは！", emotion="happy",
                            author="Character1", character="Character1")
        logs = log_ws.of_type("speech_log")
        assert len(logs) == 1
        assert logs[0]["text"] == "こんにちは！"
        assert logs[0]["character"] == "Character1"
        assert logs[0]["speaker"] == "Character1"
        # 従来の chat チャネルへの配送も従来どおり
        assert len(chat_ws.of_type("chat")) == 1
    _run(main())


async def _setup():
    hub = WebSocketHub()
    log_ws = _FakeWS()
    await hub.connect(log_ws, "speechlog")
    return hub, log_ws


def test_speech_chat_user_role_not_in_speechlog():
    async def main():
        hub, log_ws = await _setup()
        await hub.send_chat("user", "ローカル入力", author="マスター")
        assert log_ws.of_type("speech_log") == []
    _run(main())


def test_send_comment_flows_as_picked_comment():
    async def main():
        hub, log_ws = await _setup()
        dash_ws = _FakeWS()
        await hub.connect(dash_ws, "dashboard")
        await hub.send_comment({
            "author": "視聴者A", "message": "質問です！",
            "priority": 5.0, "is_superchat": False,
        })
        picked = log_ws.of_type("picked_comment")
        assert len(picked) == 1
        assert picked[0]["author"] == "視聴者A"
        assert picked[0]["text"] == "質問です！"  # message → text に正規化
        assert picked[0]["is_superchat"] is False
        assert picked[0]["is_voice"] is False
        # 従来の dashboard への new_comment も従来どおり
        assert len(dash_ws.of_type("new_comment")) == 1
    _run(main())


def test_history_replayed_on_connect_and_capped():
    async def main():
        hub = WebSocketHub()
        for i in range(35):
            await hub.send_chat("assistant", f"発話{i}", emotion="normal",
                                author="Character2", character="Character2")
        assert len(hub._speechlog_history) == WebSocketHub.SPEECHLOG_HISTORY_MAX
        late_ws = _FakeWS()
        await hub.connect(late_ws, "speechlog")
        hist = late_ws.of_type("speechlog_history")
        assert len(hist) == 1
        items = hist[0]["items"]
        assert len(items) == WebSocketHub.SPEECHLOG_HISTORY_MAX
        assert items[0]["text"] == "発話5"   # 古い方から落ちている
        assert items[-1]["text"] == "発話34"
    _run(main())


def test_no_history_message_when_empty():
    async def main():
        hub = WebSocketHub()
        ws = _FakeWS()
        await hub.connect(ws, "speechlog")
        assert ws.of_type("speechlog_history") == []
    _run(main())
