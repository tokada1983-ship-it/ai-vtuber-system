"""配信状態ファイル（backend/stream_state）のテスト

/api/start で作成され /api/stop と正常終了で削除される、ウォッチドッグ
連携用の状態ファイルの読み書きを検証する。
"""
from __future__ import annotations

import json

from backend.stream_state import (
    clear_stream_state,
    mark_stream_stopping,
    read_stream_state,
    write_stream_state,
)


def test_write_and_read_roundtrip(tmp_path):
    path = tmp_path / "data" / "stream_state.json"
    write_stream_state("abc123", path=path)
    data = read_stream_state(path=path)
    assert data is not None
    assert data["video_id"] == "abc123"
    assert "started_at" in data
    assert isinstance(data["pid"], int)


def test_write_empty_video_id(tmp_path):
    """MockSource（空 video_id）でも状態ファイルは作られる"""
    path = tmp_path / "stream_state.json"
    write_stream_state("", path=path)
    data = read_stream_state(path=path)
    assert data is not None
    assert data["video_id"] == ""


def test_clear_removes_file(tmp_path):
    path = tmp_path / "stream_state.json"
    write_stream_state("x", path=path)
    assert path.exists()
    clear_stream_state(path=path)
    assert not path.exists()
    assert read_stream_state(path=path) is None


def test_clear_missing_file_is_noop(tmp_path):
    path = tmp_path / "nope.json"
    clear_stream_state(path=path)  # 例外を投げない
    assert read_stream_state(path=path) is None


def test_read_corrupt_file_returns_none(tmp_path):
    path = tmp_path / "stream_state.json"
    path.write_text("{not json", encoding="utf-8")
    assert read_stream_state(path=path) is None


def test_write_overwrites_previous(tmp_path):
    path = tmp_path / "stream_state.json"
    write_stream_state("first", path=path)
    write_stream_state("second", path=path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["video_id"] == "second"


# ── status と停止意図の記録（stop-fix） ──────────────────────
def test_write_records_status_streaming(tmp_path):
    path = tmp_path / "stream_state.json"
    write_stream_state("abc123", path=path)
    data = read_stream_state(path=path)
    assert data["status"] == "streaming"


def test_mark_stopping_preserves_fields(tmp_path):
    path = tmp_path / "stream_state.json"
    write_stream_state("abc123", path=path)
    assert mark_stream_stopping(path=path) is True
    data = read_stream_state(path=path)
    assert data["status"] == "stopping"
    assert data["video_id"] == "abc123"  # 既存フィールドは保持される
    assert "started_at" in data
    assert "stopping_at" in data


def test_mark_stopping_without_file_is_noop(tmp_path):
    path = tmp_path / "nope.json"
    assert mark_stream_stopping(path=path) is False  # 例外を投げない
    assert not path.exists()  # ファイルは作られない


def test_mark_stopping_corrupt_file_is_noop(tmp_path):
    path = tmp_path / "stream_state.json"
    path.write_text("{not json", encoding="utf-8")
    assert mark_stream_stopping(path=path) is False


def test_rewrite_after_stopping_returns_to_streaming(tmp_path):
    """停止マークの後に /api/start で作り直すと streaming に戻る"""
    path = tmp_path / "stream_state.json"
    write_stream_state("v1", path=path)
    mark_stream_stopping(path=path)
    write_stream_state("v2", path=path)
    data = read_stream_state(path=path)
    assert data["status"] == "streaming"
    assert data["video_id"] == "v2"
