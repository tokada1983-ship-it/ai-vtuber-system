"""ウォッチドッグ（tools/aivt_watchdog.py）の判定ロジックのテスト

異常の継続判定（FailureTracker）と再起動上限（RestartLimiter）、
状態ファイル読み取り、Watchdog.run_once の分岐を検証する。
ウォッチドッグ本体は標準ライブラリのみの独立スクリプトなので、
ファイルパスから直接ロードする。
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_WD_PATH = (Path(__file__).resolve().parent.parent
            / "tools" / "aivt_watchdog.py")
_spec = importlib.util.spec_from_file_location("aivt_watchdog", _WD_PATH)
wd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(wd)


# ── FailureTracker ──────────────────────────────────────────
def test_failure_tracker_triggers_after_threshold():
    t = wd.FailureTracker(threshold_sec=30.0)
    assert t.observe("down", 100.0) is False
    assert t.observe("down", 129.9) is False
    assert t.observe("down", 130.0) is True


def test_failure_tracker_resets_on_healthy():
    t = wd.FailureTracker(threshold_sec=30.0)
    assert t.observe("down", 0.0) is False
    assert t.observe("down", 20.0) is False
    assert t.observe("healthy", 21.0) is False
    assert t.observe("down", 22.0) is False
    assert t.observe("down", 51.9) is False
    assert t.observe("down", 52.0) is True


def test_failure_tracker_resets_on_starting():
    t = wd.FailureTracker(threshold_sec=30.0)
    assert t.observe("down", 0.0) is False
    assert t.observe("starting", 40.0) is False  # 起動中は異常扱いしない
    assert t.observe("down", 41.0) is False
    assert t.observe("down", 71.0) is True


# ── RestartLimiter ──────────────────────────────────────────
def test_restart_limiter_allows_up_to_max():
    lim = wd.RestartLimiter(max_restarts=3, window_sec=3600.0)
    for i in range(3):
        assert lim.can_restart(1000.0 + i) is True
        lim.record(1000.0 + i)
    assert lim.can_restart(1003.0) is False
    assert lim.count(1003.0) == 3


def test_restart_limiter_window_expiry():
    lim = wd.RestartLimiter(max_restarts=1, window_sec=3600.0)
    lim.record(0.0)
    assert lim.can_restart(3599.0) is False
    assert lim.can_restart(3600.0) is True  # 窓を抜けたら再び許可


def test_restart_limiter_history_roundtrip():
    lim = wd.RestartLimiter(max_restarts=3, window_sec=3600.0,
                            history=[100.0, 200.0])
    assert lim.count(300.0) == 2
    lim2 = wd.RestartLimiter(max_restarts=3, window_sec=3600.0,
                             history=lim.to_list())
    assert lim2.count(300.0) == 2


# ── 状態ファイル ─────────────────────────────────────────────
def test_read_stream_state_valid(tmp_path):
    p = tmp_path / "stream_state.json"
    p.write_text('{"video_id": "v1", "started_at": "t"}',
                 encoding="utf-8")
    assert wd.read_stream_state(p)["video_id"] == "v1"


def test_read_stream_state_missing_and_corrupt(tmp_path):
    assert wd.read_stream_state(tmp_path / "none.json") is None
    p = tmp_path / "bad.json"
    p.write_text("not json", encoding="utf-8")
    assert wd.read_stream_state(p) is None
    p.write_text('["not", "a", "dict"]', encoding="utf-8")
    assert wd.read_stream_state(p) is None


# ── Watchdog.run_once の分岐 ─────────────────────────────────
@pytest.fixture()
def fake_watchdog(monkeypatch):
    monkeypatch.setattr(wd, "load_restart_history", lambda: [])
    monkeypatch.setattr(wd, "save_restart_history", lambda h: None)
    # 本番のウォッチドッグログへテストのログを混ぜない
    monkeypatch.setattr(wd, "log", lambda *a, **k: None)
    dog = wd.Watchdog()
    dog.recover_calls = []
    dog.recover = lambda state, now: dog.recover_calls.append((state, now))
    return dog


def test_run_once_idle_without_state_file(monkeypatch, fake_watchdog):
    monkeypatch.setattr(wd, "read_stream_state", lambda: None)
    fake_watchdog.run_once(1000.0)
    assert fake_watchdog.recover_calls == []


def test_run_once_healthy_no_recovery(monkeypatch, fake_watchdog):
    monkeypatch.setattr(wd, "read_stream_state",
                        lambda: {"video_id": "v", "started_at": "t"})
    fake_watchdog.probe_state = lambda now: "healthy"
    fake_watchdog.run_once(1000.0)
    assert fake_watchdog.recover_calls == []


def test_run_once_recovers_after_sustained_down(monkeypatch,
                                                fake_watchdog):
    monkeypatch.setattr(wd, "read_stream_state",
                        lambda: {"video_id": "v", "started_at": "t"})
    fake_watchdog.probe_state = lambda now: "down"
    fake_watchdog.run_once(1000.0)   # 異常の開始
    fake_watchdog.run_once(1029.0)   # まだ閾値未満
    assert fake_watchdog.recover_calls == []
    fake_watchdog.run_once(1031.0)   # 閾値超過 → 復旧
    assert len(fake_watchdog.recover_calls) == 1


def test_run_once_respects_restart_limit(monkeypatch, fake_watchdog):
    monkeypatch.setattr(wd, "read_stream_state",
                        lambda: {"video_id": "v", "started_at": "t"})
    fake_watchdog.probe_state = lambda now: "down"
    fake_watchdog.limiter = wd.RestartLimiter(
        max_restarts=3, window_sec=3600.0, history=[0.0, 1.0, 2.0])
    fake_watchdog.run_once(1000.0)
    fake_watchdog.run_once(1031.0)
    assert fake_watchdog.recover_calls == []


# ── 意図した停止（stopping）の判定（stop-fix） ───────────────
def test_is_stopping():
    assert wd.is_stopping({"status": "stopping"}) is True
    assert wd.is_stopping({"status": "streaming"}) is False
    assert wd.is_stopping({"video_id": "v"}) is False  # 旧形式は配信中
    assert wd.is_stopping(None) is False
    assert wd.is_stopping({}) is False


def test_recovery_abort_reason(tmp_path):
    p = tmp_path / "stream_state.json"
    # ファイルが無い → 中止
    assert wd.recovery_abort_reason(p) is not None
    # streaming → 中止しない
    p.write_text('{"video_id": "v", "status": "streaming"}',
                 encoding="utf-8")
    assert wd.recovery_abort_reason(p) is None
    # 旧形式（status 無し）→ 中止しない
    p.write_text('{"video_id": "v"}', encoding="utf-8")
    assert wd.recovery_abort_reason(p) is None
    # stopping → 中止
    p.write_text('{"video_id": "v", "status": "stopping"}',
                 encoding="utf-8")
    assert wd.recovery_abort_reason(p) is not None


def test_run_once_stands_down_while_stopping(monkeypatch, fake_watchdog):
    """stopping の間はアプリが down でも復旧に入らない"""
    monkeypatch.setattr(
        wd, "read_stream_state",
        lambda: {"video_id": "v", "started_at": "t",
                 "status": "stopping"})
    fake_watchdog.probe_state = lambda now: "down"
    for now in (1000.0, 1031.0, 1200.0):
        fake_watchdog.run_once(now)
    assert fake_watchdog.recover_calls == []


def test_recover_aborts_when_state_turns_stopping(monkeypatch,
                                                  fake_watchdog):
    """復旧の開始直前に stopping へ変わっていたら予算を消費せず中止"""
    monkeypatch.setattr(
        wd, "recovery_abort_reason",
        lambda path=None: "stream is stopping intentionally")
    dog = fake_watchdog
    dog.recover = wd.Watchdog.recover.__get__(dog, wd.Watchdog)
    dog.recover({"video_id": "v", "started_at": "t"}, 1000.0)
    assert dog.limiter.count(1000.0) == 0  # 再起動履歴に記録されない
