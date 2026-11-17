"""ViewerProfileStore: 視聴者ごとの関係性・親密度・過去の話題を永続管理するストア"""
from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from backend.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ViewerProfile:
    viewer_id: str
    display_name: str
    visit_count: int
    first_seen_at: float
    last_seen_at: float
    familiarity_level: int  # 1: 初見, 2: 顔見知り, 3: 常連, 4: 大常連, 5: 古参
    notes: str
    custom_nickname: str | None = None

    @property
    def level_name(self) -> str:
        levels = {
            1: "初見",
            2: "顔見知り",
            3: "常連",
            4: "大常連",
            5: "古参・特別なリスナー",
        }
        return levels.get(self.familiarity_level, "常連")


class ViewerProfileStore:
    """視聴者の関係性（エンティティ記憶）をSQLiteで管理する"""

    def __init__(self, db_path: Path):
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self) -> None:
        with self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS viewer_profiles (
                    viewer_id         TEXT PRIMARY KEY,
                    display_name      TEXT NOT NULL,
                    visit_count       INTEGER NOT NULL DEFAULT 1,
                    first_seen_at     REAL NOT NULL,
                    last_seen_at      REAL NOT NULL,
                    familiarity_level INTEGER NOT NULL DEFAULT 1,
                    notes             TEXT NOT NULL DEFAULT '',
                    custom_nickname   TEXT
                )
                """
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_last_seen ON viewer_profiles(last_seen_at)"
            )

    @staticmethod
    def calc_familiarity(visit_count: int) -> int:
        """来訪回数から親密度レベルを算出 (1〜5)"""
        if visit_count <= 1:
            return 1
        if visit_count <= 4:
            return 2
        if visit_count <= 9:
            return 3
        if visit_count <= 19:
            return 4
        return 5

    def record_visit(self, viewer_id: str, display_name: str) -> ViewerProfile:
        """視聴者の来訪を記録し、更新後のプロファイルを返す"""
        if not viewer_id:
            viewer_id = "anonymous"
        now = time.time()

        with self._conn:
            row = self._conn.execute(
                "SELECT * FROM viewer_profiles WHERE viewer_id = ?", (viewer_id,)
            ).fetchone()

            if row is None:
                profile = ViewerProfile(
                    viewer_id=viewer_id,
                    display_name=display_name,
                    visit_count=1,
                    first_seen_at=now,
                    last_seen_at=now,
                    familiarity_level=1,
                    notes="",
                    custom_nickname=None,
                )
                self._conn.execute(
                    """
                    INSERT INTO viewer_profiles (
                        viewer_id, display_name, visit_count,
                        first_seen_at, last_seen_at, familiarity_level, notes
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        profile.viewer_id,
                        profile.display_name,
                        profile.visit_count,
                        profile.first_seen_at,
                        profile.last_seen_at,
                        profile.familiarity_level,
                        profile.notes,
                    ),
                )
                logger.info("New viewer recorded", viewer=viewer_id, name=display_name)
                return profile
            else:
                new_visit = row["visit_count"] + 1
                new_level = self.calc_familiarity(new_visit)
                self._conn.execute(
                    """
                    UPDATE viewer_profiles
                    SET display_name = ?,
                        visit_count = ?,
                        last_seen_at = ?,
                        familiarity_level = ?
                    WHERE viewer_id = ?
                    """,
                    (display_name, new_visit, now, new_level, viewer_id),
                )
                return ViewerProfile(
                    viewer_id=viewer_id,
                    display_name=display_name,
                    visit_count=new_visit,
                    first_seen_at=row["first_seen_at"],
                    last_seen_at=now,
                    familiarity_level=new_level,
                    notes=row["notes"] or "",
                    custom_nickname=row["custom_nickname"],
                )

    def get_profile(self, viewer_id: str) -> ViewerProfile | None:
        """視聴者プロファイルを取得"""
        if not viewer_id:
            return None
        row = self._conn.execute(
            "SELECT * FROM viewer_profiles WHERE viewer_id = ?", (viewer_id,)
        ).fetchone()
        if row is None:
            return None
        return ViewerProfile(
            viewer_id=row["viewer_id"],
            display_name=row["display_name"],
            visit_count=row["visit_count"],
            first_seen_at=row["first_seen_at"],
            last_seen_at=row["last_seen_at"],
            familiarity_level=row["familiarity_level"],
            notes=row["notes"] or "",
            custom_nickname=row["custom_nickname"],
        )

    def add_note(self, viewer_id: str, new_note: str) -> None:
        """視聴者に関するメモを追加（重複回避）"""
        if not viewer_id or not new_note:
            return
        cleaned = new_note.strip()
        with self._conn:
            row = self._conn.execute(
                "SELECT notes FROM viewer_profiles WHERE viewer_id = ?", (viewer_id,)
            ).fetchone()
            if row is None:
                return
            existing = row["notes"] or ""
            lines = [l.strip() for l in existing.split("\n") if l.strip()]
            if cleaned not in lines:
                lines.append(cleaned)
                # 最大10件まで保持
                if len(lines) > 10:
                    lines = lines[-10:]
                updated = "\n".join(lines)
                self._conn.execute(
                    "UPDATE viewer_profiles SET notes = ? WHERE viewer_id = ?",
                    (updated, viewer_id),
                )

    def build_prompt(self, viewer_id: str) -> str:
        """プロンプトへ注入する視聴者関係性コンテキストを生成"""
        profile = self.get_profile(viewer_id)
        if not profile:
            return ""

        call_name = profile.custom_nickname or profile.display_name
        lines = [
            f"### コメント相手の情報",
            f"- 呼称: {call_name} (関係性: {profile.level_name} / 親密度Lv{profile.familiarity_level})",
            f"- 累計来訪回数: {profile.visit_count}回",
        ]
        if profile.notes:
            notes_str = ", ".join(profile.notes.split("\n")[:4])
            lines.append(f"- 相手に関する記憶: {notes_str}")

        if profile.familiarity_level == 1:
            lines.append("- 接し方の方針: はじめて配信に来てくれたリスナー。歓迎しつつ初対面として優しく接する。")
        elif profile.familiarity_level >= 4:
            lines.append("- 接し方の方針: いつも来てくれる大常連。くだけた親しみのある態度や特別な信頼感を見せてよい。")

        return "\n".join(lines)

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass
