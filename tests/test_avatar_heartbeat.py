"""アバターページのハートビート／イベント処理（WebSocketHub）の単体テスト。

2026-11-19 の配信テストで、アバターページが「WS 接続は正常表示のまま
再生だけ凍結」し、バックエンド側から生死判定できなかった問題への対策。
ページから届く heartbeat / page_event をハブがキャラ別に保持し、
フレーム停滞を検出してログに残すことを検証する。
"""
from __future__ import annotations

import asyncio

import backend.avatar.ws_hub as ws_hub_module
from backend.avatar.ws_hub import WebSocketHub


class _Recorder:
    """structlog 風ロガーの記録用スタブ"""

    def __init__(self):
        self.calls: list[tuple[str, str, dict]] = []

    def _add(self, level, msg, kwargs):
        self.calls.append((level, msg, kwargs))

    def info(self, msg, **kwargs):
        self._add("info", msg, kwargs)

    def warning(self, msg, **kwargs):
        self._add("warning", msg, kwargs)

    def error(self, msg, **kwargs):
        self._add("error", msg, kwargs)

    def debug(self, msg, **kwargs):
        self._add("debug", msg, kwargs)

    def by_msg(self, msg):
        return [c for c in self.calls if c[1] == msg]


def _hub_with_recorder(monkeypatch):
    rec = _Recorder()
    monkeypatch.setattr(ws_hub_module, "logger", rec)
    return WebSocketHub(), rec


def _hb(char="Character1", frames=100, **kw):
    data = {
        "type": "heartbeat", "character": char, "page": "video",
        "frames": frames, "state_time": 12.5, "ready_state": 4,
        "clip": "idle_main.webm", "mode": "idle", "context_lost": False,
    }
    data.update(kw)
    return data


def test_heartbeat_stored_per_character(monkeypatch):
    hub, _ = _hub_with_recorder(monkeypatch)
    handled = asyncio.run(hub.handle_client_message("avatar", _hb(frames=100)))
    assert handled is True
    asyncio.run(hub.handle_client_message("avatar", _hb(char="Character2", frames=7)))
    asyncio.run(hub.handle_client_message("avatar", _hb(frames=160)))

    health = hub.avatar_health()
    assert set(health) == {"Character1", "Character2"}
    assert health["Character1"]["frames"] == 160
    assert health["Character1"]["count"] == 2
    assert health["Character1"]["clip"] == "idle_main.webm"
    assert health["Character1"]["stalled"] is False
    assert health["Character2"]["frames"] == 7


def test_heartbeat_stall_detected_when_frames_stop(monkeypatch):
    hub, rec = _hub_with_recorder(monkeypatch)
    asyncio.run(hub.handle_client_message("avatar", _hb(frames=100)))
    asyncio.run(hub.handle_client_message("avatar", _hb(frames=100)))
    assert hub.avatar_health()["Character1"]["stalled"] is True
    warnings = rec.by_msg("Avatar heartbeat stalled")
    assert len(warnings) == 1
    assert warnings[0][2]["character"] == "Character1"

    # 停滞警告は30秒に1回まで（連続ハートビートでログが溢れない）
    asyncio.run(hub.handle_client_message("avatar", _hb(frames=100)))
    assert len(rec.by_msg("Avatar heartbeat stalled")) == 1


def test_heartbeat_info_log_throttled(monkeypatch):
    hub, rec = _hub_with_recorder(monkeypatch)
    for frames in (100, 130, 160, 190):
        asyncio.run(hub.handle_client_message("avatar", _hb(frames=frames)))
    # 情報ログは60秒に1回まで
    assert len(rec.by_msg("Avatar heartbeat")) == 1


def test_context_lost_suppresses_stall_flag(monkeypatch):
    hub, _ = _hub_with_recorder(monkeypatch)
    asyncio.run(hub.handle_client_message("avatar", _hb(frames=100)))
    asyncio.run(hub.handle_client_message("avatar", _hb(frames=100, context_lost=True)))
    entry = hub.avatar_health()["Character1"]
    assert entry["context_lost"] is True
    assert entry["stalled"] is False


def test_heartbeat_without_character_goes_to_unknown(monkeypatch):
    hub, _ = _hub_with_recorder(monkeypatch)
    data = _hb()
    del data["character"]
    asyncio.run(hub.handle_client_message("avatar", data))
    assert "unknown" in hub.avatar_health()


def test_unrelated_message_not_handled(monkeypatch):
    hub, _ = _hub_with_recorder(monkeypatch)
    assert asyncio.run(hub.handle_client_message("avatar", {"cmd": "stop"})) is False
    assert asyncio.run(hub.handle_client_message("dashboard", {"type": "status"})) is False
    assert hub.avatar_health() == {}


def test_page_event_levels_and_truncation(monkeypatch):
    hub, rec = _hub_with_recorder(monkeypatch)
    long_detail = "x" * 500
    handled = asyncio.run(hub.handle_client_message("avatar", {
        "type": "page_event", "character": "Character1",
        "event": "playback_stall", "detail": long_detail,
    }))
    assert handled is True
    warns = rec.by_msg("Avatar page event")
    assert len(warns) == 1
    assert warns[0][0] == "warning"
    assert warns[0][2]["event_name"] == "playback_stall"
    assert len(warns[0][2]["detail"]) == 300

    asyncio.run(hub.handle_client_message("avatar", {
        "type": "page_event", "character": "Character2",
        "event": "context_restored", "detail": "ok",
    }))
    infos = [c for c in rec.by_msg("Avatar page event") if c[0] == "info"]
    assert len(infos) == 1
    assert infos[0][2]["character"] == "Character2"


def test_real_structlog_logger_does_not_raise():
    """実ロガーで page_event / heartbeat を処理しても例外を投げないこと。

    structlog は "event" を引数名として予約しており、kwarg に使うと
    TypeError になる（2026-11-19 の実機スモークで実際に踏んだ回帰）。
    スタブロガーでは検出できないため、実ロガーで固定する。
    """
    from backend.logger import setup_logging
    setup_logging("INFO")
    hub = WebSocketHub()
    asyncio.run(hub.handle_client_message("avatar", {
        "type": "page_event", "character": "Character1",
        "event": "playback_stall", "detail": "clip=idle_main.webm",
    }))
    asyncio.run(hub.handle_client_message("avatar", {
        "type": "page_event", "character": "Character1",
        "event": "context_restored", "detail": "ok",
    }))
    asyncio.run(hub.handle_client_message("avatar", _hb(frames=42)))
    assert hub.avatar_health()["Character1"]["frames"] == 42
