"""配信セッションの状態ファイル（ウォッチドッグ連携）

エンジンが配信中であることを、アプリプロセスの外からも分かるように
するための小さな状態ファイル。/api/start の成功時に作成し、
/api/stop とアプリの正常終了時に削除する。

外部のウォッチドッグ（tools/aivt_watchdog.py）は、このファイルが
存在する間だけアプリの生死を監視し、異常終了したら自動復旧する。
アプリが APPCRASH 等で即死した場合はこのファイルが残るため、
ウォッチドッグが「配信中だったのにアプリが居ない」を検知できる。

status フィールドで「配信中（streaming）」と「意図した停止の進行中
（stopping）」を区別する。/api/stop は処理の冒頭で stopping を記録
するため、停止処理が長引いたり途中でアプリが落とされても、
ウォッチドッグは意図した停止を落下と誤認しない（2026-11-21 stop-fix）。
status の無い旧形式のファイルは streaming として扱う。
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

from backend.logger import get_logger

logger = get_logger(__name__)

# <repo root>/data/stream_state.json（このファイルの2つ上が root）
STREAM_STATE_PATH = Path(__file__).resolve().parent.parent / "data" / "stream_state.json"


def write_stream_state(video_id: str, path: Path | None = None) -> None:
    """配信開始を状態ファイルに記録する。失敗しても例外は投げない。"""
    target = path or STREAM_STATE_PATH
    payload = {
        "video_id": video_id,
        "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "pid": os.getpid(),
        "status": "streaming",
    }
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                       encoding="utf-8")
        tmp.replace(target)
        logger.info("Stream state written", video_id=video_id)
    except Exception as e:
        logger.warning("Failed to write stream state", error=str(e))


def mark_stream_stopping(path: Path | None = None) -> bool:
    """状態ファイルに「意図した停止の進行中」を記録する。

    /api/stop の冒頭で呼ぶ。既存の内容（video_id 等）は保持し、
    status="stopping" と stopping_at を書き加える。ファイルが無い・
    壊れている場合は何もしない。戻り値は記録できたかどうか。
    例外は投げない。
    """
    target = path or STREAM_STATE_PATH
    data = read_stream_state(path=target)
    if data is None:
        return False
    data["status"] = "stopping"
    data["stopping_at"] = datetime.now().astimezone().isoformat(
        timespec="seconds")
    try:
        tmp = target.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                       encoding="utf-8")
        tmp.replace(target)
        logger.info("Stream state marked as stopping",
                    video_id=data.get("video_id"))
        return True
    except Exception as e:
        logger.warning("Failed to mark stream state as stopping",
                       error=str(e))
        return False


def clear_stream_state(path: Path | None = None) -> None:
    """状態ファイルを削除する。無ければ何もしない。例外は投げない。"""
    target = path or STREAM_STATE_PATH
    try:
        target.unlink(missing_ok=True)
        logger.info("Stream state cleared")
    except Exception as e:
        logger.warning("Failed to clear stream state", error=str(e))


def read_stream_state(path: Path | None = None) -> dict | None:
    """状態ファイルを読む。無い・壊れている場合は None。"""
    target = path or STREAM_STATE_PATH
    try:
        return json.loads(target.read_text(encoding="utf-8"))
    except Exception:
        return None
