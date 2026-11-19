"""WebSocketHub: フロントエンド（Dashboard・アバター）へのリアルタイム通知"""
from __future__ import annotations

import asyncio
import json
import time
from typing import Any

from fastapi import WebSocket
from backend.logger import get_logger

logger = get_logger(__name__)


class WebSocketHub:
    """
    複数のWebSocketクライアントをチャネルで管理する。
    - "dashboard" : 操作・状態確認用
    - "avatar"    : OBS用アバターレンダラ
    - "subtitle"  : 字幕オーバーレイ
    """

    def __init__(self):
        self._connections: dict[str, list[WebSocket]] = {
            "dashboard": [],
            "avatar": [],
            "subtitle": [],
            "unity": [],
            "chat": [],      # ローカル1対1チャット画面
        }
        self._lock = asyncio.Lock()
        # アバターページの生死判定（ページから届くハートビートで更新する。
        # ページは描画・再生が止まっても WS 接続だけ残ることがあるため、
        # 接続の有無ではなく「フレーム数が増えているか」で健全性を測る）
        self._avatar_health: dict[str, dict[str, Any]] = {}
        self._hb_last_log: dict[str, float] = {}
        self._hb_stall_warned: dict[str, float] = {}

    async def connect(self, ws: WebSocket, channel: str = "dashboard") -> None:
        await ws.accept()
        async with self._lock:
            self._connections.setdefault(channel, []).append(ws)
        logger.info("WS connected", channel=channel)

    async def disconnect(self, ws: WebSocket, channel: str = "dashboard") -> None:
        async with self._lock:
            conns = self._connections.get(channel, [])
            if ws in conns:
                conns.remove(ws)
        logger.info("WS disconnected", channel=channel)

    async def broadcast(self, channel: str, data: dict[str, Any]) -> None:
        """指定チャネルの全接続にJSONを送信する"""
        text = json.dumps(data, ensure_ascii=False)
        dead: list[WebSocket] = []

        async with self._lock:
            conns = list(self._connections.get(channel, []))

        for ws in conns:
            try:
                await ws.send_text(text)
            except Exception:
                dead.append(ws)

        if dead:
            async with self._lock:
                for ws in dead:
                    conns = self._connections.get(channel, [])
                    if ws in conns:
                        conns.remove(ws)

    def has_clients(self, channel: str) -> bool:
        """そのチャネルを開いている画面があるか（誰も見ていない計測を省くため）"""
        return bool(self._connections.get(channel))

    async def broadcast_all(self, data: dict[str, Any]) -> None:
        """全チャネルにブロードキャスト"""
        for channel in self._connections:
            await self.broadcast(channel, data)

    # ── 高レベルAPI ──────────────────────────────────────────
    #
    # アバター向けのメッセージには、誰の発話かを示す character（キャラの id）が付く。
    # アバター画面は /avatar?char=<id> で自分の担当キャラだけを拾う。
    # character を付けない（1体運用の頃の）呼び出しは、全アバター画面に届く。

    async def send_speech_start(self, text: str, emotion: str, audio_url: str | None = None,
                                character: str | None = None, speaker: str | None = None) -> None:
        msg: dict = {"type": "speech_start", "text": text, "emotion": emotion}
        if audio_url:
            msg["audio_url"] = audio_url
        if character:
            msg["character"] = character
        await self.broadcast("avatar", msg)
        sub: dict = {"type": "subtitle", "text": text}
        if speaker:
            sub["speaker"] = speaker
        if character:
            sub["character"] = character
        await self.broadcast("subtitle", sub)
        await self.broadcast("unity", msg)

    async def send_motion(self, motion: str, character: str | None = None) -> None:
        msg: dict = {"type": "motion", "motion": motion}
        if character:
            msg["character"] = character
        await self.broadcast("avatar", msg)
        await self.broadcast("unity", msg)

    async def send_speech_end(self, character: str | None = None) -> None:
        msg: dict = {"type": "speech_end"}
        if character:
            msg["character"] = character
        await self.broadcast("avatar", msg)
        await self.broadcast("unity", msg)

    async def send_unity_move(self, x: float, y: float, z: float) -> None:
        """Unityアバターに移動座標を指示する"""
        await self.broadcast("unity", {"type": "move_to", "x": x, "y": y, "z": z})

    async def send_unity_gimmick(self, gimmick_id: str) -> None:
        """Unityアバターにギミック操作を指示する"""
        await self.broadcast("unity", {"type": "use_gimmick", "gimmick_id": gimmick_id})

    async def send_unity_action(self, action: str, target: str | None = None) -> None:
        """Unityアバターに一般的なアクションを指示する"""
        await self.broadcast("unity", {"type": "action", "action": action, "target": target})

    async def send_status(self, status: dict) -> None:
        await self.broadcast("dashboard", {"type": "status", **status})

    async def send_comment(self, comment_data: dict) -> None:
        await self.broadcast("dashboard", {"type": "new_comment", **comment_data})

    async def send_chat(self, role: str, text: str, **extra) -> None:
        """
        ローカル1対1チャット画面（/chat）へ1行流す。

        role: "user"（ローカルユーザーの発言）/ "assistant"（キャラの発話）
        アバター向けの speech_start は1文につき複数回飛ぶ（音声URLが後から付く）ため、
        チャット吹き出しが二重にならないよう専用のイベントに分けている。
        """
        await self.broadcast("chat", {"type": "chat", "role": role, "text": text, **extra})

    async def send_log(self, level: str, message: str) -> None:
        await self.broadcast("dashboard", {"type": "log", "level": level, "message": message})

    # ── ページ → サーバ（ハートビート／イベント） ──────────────
    #
    # アバターページは数秒ごとに {"type": "heartbeat", ...} を送り、
    # 異常時は {"type": "page_event", "event": ..., "detail": ...} を送る。
    # ここでキャラ別に最終状態を保持し、ログは間引いて残す。

    async def handle_client_message(self, channel: str, data: dict[str, Any]) -> bool:
        """ページから届いたハートビート／イベントを処理する。処理したら True"""
        mtype = data.get("type")
        if mtype == "heartbeat":
            self._note_heartbeat(channel, data)
            return True
        if mtype == "page_event":
            self._note_page_event(data)
            return True
        return False

    def _note_heartbeat(self, channel: str, data: dict[str, Any]) -> None:
        char = str(data.get("character") or "unknown")
        now = time.time()
        frames = data.get("frames")
        prev = self._avatar_health.get(char)
        # フレーム数が前回と同値＝描画・再生が進んでいない（接続だけ生きている）
        stalled = (
            prev is not None
            and frames is not None
            and prev.get("frames") == frames
            and not data.get("context_lost")
        )
        self._avatar_health[char] = {
            "channel": channel,
            "page": data.get("page"),
            "frames": frames,
            "state_time": data.get("state_time"),
            "context_lost": bool(data.get("context_lost")),
            "clip": data.get("clip"),
            "ready_state": data.get("ready_state"),
            "mode": data.get("mode"),
            "received_at": now,
            "count": (prev.get("count", 0) + 1) if prev else 1,
            "stalled": stalled,
        }
        if stalled:
            if now - self._hb_stall_warned.get(char, 0.0) >= 30.0:
                self._hb_stall_warned[char] = now
                logger.warning(
                    "Avatar heartbeat stalled", character=char,
                    frames=frames, clip=data.get("clip"),
                    state_time=data.get("state_time"),
                )
        elif now - self._hb_last_log.get(char, 0.0) >= 60.0:
            self._hb_last_log[char] = now
            logger.info(
                "Avatar heartbeat", character=char, page=data.get("page"),
                frames=frames, state_time=data.get("state_time"),
                clip=data.get("clip"), ready_state=data.get("ready_state"),
            )

    def _note_page_event(self, data: dict[str, Any]) -> None:
        char = str(data.get("character") or "unknown")
        event = str(data.get("event") or "unknown")
        detail = str(data.get("detail") or "")[:300]
        warn_events = {
            "context_lost", "page_error", "tick_error", "watchdog_reload",
            "playback_stall", "media_error", "play_rejected",
        }
        # 注意: structlog は "event" をログ本文の引数名として予約しているため、
        # kwarg 名に event は使えない（event_name に逃がす）
        if event in warn_events:
            logger.warning("Avatar page event", character=char, event_name=event, detail=detail)
        else:
            logger.info("Avatar page event", character=char, event_name=event, detail=detail)

    def avatar_health(self) -> dict[str, dict[str, Any]]:
        """キャラ別の最終ハートビート状態（写し）を返す"""
        return {k: dict(v) for k, v in self._avatar_health.items()}
