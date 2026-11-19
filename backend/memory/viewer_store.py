"""ViewerProfileStore: 視聴者ごとの関係性・親密度・過去の話題を永続管理するストア

Phase 4（組み込み検討書 §5 第1段）でポイント台帳とレベル移行を追加した。

- ポイント台帳: 視聴者ごとの累積ポイントを viewer_profiles.points に保持し、
  加算の履歴を viewer_point_events に1件ずつ残す（いつ・何で・何点か）。
- レベル移行: 親密度レベルは来訪回数ではなく累積ポイントの閾値で決める。
- 加算ルール: 基本点＋レート制限（時間窓の回数上限と1セッションの総量上限）。
  同一イベントの連打では稼げない（aituber-onair/kizuna の設計を参考にした）。
- 後方互換: ポイント設定（ViewerPointsConfig）が渡されない、または
  enabled=False の場合は、従来どおり来訪回数ベースでレベルを決める。
  既存プロフィールは初回起動時に「現在のレベルの開始ポイント」へ換算して
  移行するので、レベルが巻き戻ることはない。
"""
from __future__ import annotations

import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from backend.logger import get_logger

if TYPE_CHECKING:
    from backend.config import ViewerPointsConfig

logger = get_logger(__name__)

# レート制限の窓（秒）。イベント種別ごとの上限回数は config 側で持つ
RATE_WINDOW_SEC = 3600.0

# レベル閾値の既定値（Lv1..Lv5 の開始ポイント）。config が正ならそちらを使う
DEFAULT_LEVEL_THRESHOLDS = [0, 50, 200, 500, 1000]

# 台帳に記録するイベント種別
EVENT_VISIT = "visit"                    # 来訪（1セッションに1回だけ）
EVENT_COMMENT = "comment"                # 実質のあるコメント
EVENT_REACTION = "reaction"              # リアクションのみのコメント
EVENT_PICKED = "picked"                  # コメントに応答してもらえた
EVENT_SUPERCHAT = "superchat"            # スパチャ
EVENT_ATTENDANCE_BONUS = "attendance_bonus"  # 参加配信数の節目ボーナス


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
    points: int = 0            # 累積ポイント（Phase 4）
    streams_attended: int = 0  # 参加した配信セッション数（Phase 4）

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

    def __init__(
        self,
        db_path: Path,
        points_cfg: "ViewerPointsConfig | None" = None,
        session_id: str | None = None,
    ):
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._points_cfg = points_cfg
        # セッション＝このストアが生きている配信1回分。来訪ポイントの
        # 二重加算を防ぐ単位で、再起動や出演者の組み直しで新しい値になる
        self._session_id = session_id or uuid.uuid4().hex
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_db()

    # ── ポイント基盤 ────────────────────────────────────────

    @property
    def points_enabled(self) -> bool:
        return self._points_cfg is not None and self._points_cfg.enabled

    @property
    def session_id(self) -> str:
        return self._session_id

    def _thresholds(self) -> list[int]:
        if self._points_cfg is not None:
            th = list(self._points_cfg.level_thresholds or [])
            if len(th) == 5:
                return th
        return list(DEFAULT_LEVEL_THRESHOLDS)

    def level_for_points(self, points: int) -> int:
        """累積ポイントから親密度レベル（1〜5）を決める"""
        level = 1
        for i, th in enumerate(self._thresholds()):
            if points >= th:
                level = i + 1
        return max(1, min(5, level))

    def points_band(self, points: int) -> str:
        """ポイントの大きさを数値でなく言葉で表す（プロンプト用）。

        キャラがポイントの具体的な数字を口にしないよう、プロンプトには
        この定性ラベルだけを載せる。
        """
        th = self._thresholds()
        if points < th[2]:
            return "まだ少なめ"
        if points < th[3]:
            return "少しずつ積み重なってきている"
        if points < th[4]:
            return "しっかり積み重なっている"
        return "とても大きい"

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
                    custom_nickname   TEXT,
                    points            INTEGER NOT NULL DEFAULT 0,
                    streams_attended  INTEGER NOT NULL DEFAULT 0,
                    last_visit_session TEXT
                )
                """
            )
            # 既存DB（Phase 3 まで）には無い列を足す（後方互換な移行）
            cols = {r["name"] for r in self._conn.execute(
                "PRAGMA table_info(viewer_profiles)")}
            if "points" not in cols:
                self._conn.execute(
                    "ALTER TABLE viewer_profiles "
                    "ADD COLUMN points INTEGER NOT NULL DEFAULT 0")
            if "streams_attended" not in cols:
                self._conn.execute(
                    "ALTER TABLE viewer_profiles "
                    "ADD COLUMN streams_attended INTEGER NOT NULL DEFAULT 0")
            if "last_visit_session" not in cols:
                self._conn.execute(
                    "ALTER TABLE viewer_profiles "
                    "ADD COLUMN last_visit_session TEXT")
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_last_seen ON viewer_profiles(last_seen_at)"
            )
            # ポイント台帳（加算履歴）
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS viewer_point_events (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    viewer_id  TEXT NOT NULL,
                    event      TEXT NOT NULL,
                    points     INTEGER NOT NULL,
                    session_id TEXT NOT NULL DEFAULT '',
                    created_at REAL NOT NULL
                )
                """
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_point_events_viewer "
                "ON viewer_point_events(viewer_id, event, created_at)"
            )
            self._migrate_legacy_points()

    def _migrate_legacy_points(self) -> None:
        """既存視聴者の初期ポイント換算（現在のレベルの開始ポイント）。

        来訪回数ベースで育った既存プロフィールが、ポイント制への移行で
        レベルを巻き戻さないための一度きりの換算。points=0 かつ
        visit_count>=2（＝Lv2以上が確定している）行だけが対象で、
        換算は台帳にも移行イベントとして残さない（ポイントの出所を
        イベント加算と混同しないため、列の初期値としてだけ置く）。
        """
        th = self._thresholds()
        rows = self._conn.execute(
            "SELECT viewer_id, familiarity_level FROM viewer_profiles "
            "WHERE points = 0 AND visit_count >= 2"
        ).fetchall()
        for row in rows:
            level = max(1, min(5, int(row["familiarity_level"] or 1)))
            initial = th[level - 1]
            if initial > 0:
                self._conn.execute(
                    "UPDATE viewer_profiles SET points = ? WHERE viewer_id = ?",
                    (initial, row["viewer_id"]),
                )

    @staticmethod
    def calc_familiarity(visit_count: int) -> int:
        """来訪回数から親密度レベルを算出 (1〜5)。ポイント無効時の従来方式"""
        if visit_count <= 1:
            return 1
        if visit_count <= 4:
            return 2
        if visit_count <= 9:
            return 3
        if visit_count <= 19:
            return 4
        return 5

    # ── 台帳への加算（トランザクション内） ───────────────────

    def _ensure_profile_in_tx(self, viewer_id: str, display_name: str,
                              now: float) -> None:
        """プロフィール行が無ければ作る（ポイント加算の前提）"""
        row = self._conn.execute(
            "SELECT viewer_id FROM viewer_profiles WHERE viewer_id = ?",
            (viewer_id,),
        ).fetchone()
        if row is None:
            self._conn.execute(
                """
                INSERT INTO viewer_profiles (
                    viewer_id, display_name, visit_count,
                    first_seen_at, last_seen_at, familiarity_level, notes
                ) VALUES (?, ?, 0, ?, ?, 1, '')
                """,
                (viewer_id, display_name or viewer_id, now, now),
            )

    def _count_in_window(self, viewer_id: str, event: str, now: float) -> int:
        """直近 RATE_WINDOW_SEC 以内にそのイベントで加算した回数"""
        row = self._conn.execute(
            "SELECT COUNT(*) AS n FROM viewer_point_events "
            "WHERE viewer_id = ? AND event = ? AND created_at > ?",
            (viewer_id, event, now - RATE_WINDOW_SEC),
        ).fetchone()
        return int(row["n"] or 0)

    def _session_total(self, viewer_id: str) -> int:
        """このセッションで加算済みの合計（スパチャ分は上限の対象外）"""
        row = self._conn.execute(
            "SELECT COALESCE(SUM(points), 0) AS s FROM viewer_point_events "
            "WHERE viewer_id = ? AND session_id = ? AND event != ?",
            (viewer_id, self._session_id, EVENT_SUPERCHAT),
        ).fetchone()
        return int(row["s"] or 0)

    def _award_in_tx(self, viewer_id: str, event: str, base_points: int,
                     now: float) -> int:
        """台帳へ1件記録し、プロフィールの累積へ足す。実際に足せた点数を返す。

        1セッションの総量上限（スパチャ以外）を超える分は切り捨てる。
        0点になる場合は台帳にも残さない（レート制限の回数数えを汚さない）。
        """
        cfg = self._points_cfg
        points = int(base_points)
        if points <= 0:
            return 0
        if event != EVENT_SUPERCHAT and cfg is not None:
            remaining = cfg.session_point_cap - self._session_total(viewer_id)
            if remaining <= 0:
                logger.info("Viewer points session cap reached",
                            viewer=viewer_id, point_event=event)
                return 0
            points = min(points, remaining)
        self._conn.execute(
            "INSERT INTO viewer_point_events "
            "(viewer_id, event, points, session_id, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (viewer_id, event, points, self._session_id, now),
        )
        self._conn.execute(
            "UPDATE viewer_profiles SET points = points + ? WHERE viewer_id = ?",
            (points, viewer_id),
        )
        return points

    def _award_visit_in_tx(self, viewer_id: str, now: float) -> int:
        """来訪ポイントの加算（セッションが変わったときだけ）。加算合計を返す。

        参加配信数（streams_attended）はポイントが上限で0点になっても
        事実として数える。節目（attendance_milestone_every の倍数）に
        達したら継続参加ボーナスも足す。
        """
        cfg = self._points_cfg
        row = self._conn.execute(
            "SELECT streams_attended, last_visit_session FROM viewer_profiles "
            "WHERE viewer_id = ?", (viewer_id,),
        ).fetchone()
        if row is None or row["last_visit_session"] == self._session_id:
            return 0
        streams = int(row["streams_attended"] or 0) + 1
        self._conn.execute(
            "UPDATE viewer_profiles SET streams_attended = ?, "
            "last_visit_session = ? WHERE viewer_id = ?",
            (streams, self._session_id, viewer_id),
        )
        awarded = self._award_in_tx(viewer_id, EVENT_VISIT, cfg.visit_points, now)
        every = cfg.attendance_milestone_every
        if every > 0 and streams % every == 0:
            awarded += self._award_in_tx(
                viewer_id, EVENT_ATTENDANCE_BONUS,
                cfg.attendance_milestone_points, now)
            logger.info("Viewer attendance milestone",
                        viewer=viewer_id, streams=streams)
        return awarded

    def _sync_level_in_tx(self, viewer_id: str) -> None:
        """累積ポイントからレベルを再計算し、変わっていれば書き戻す"""
        row = self._conn.execute(
            "SELECT points, familiarity_level FROM viewer_profiles "
            "WHERE viewer_id = ?", (viewer_id,),
        ).fetchone()
        if row is None:
            return
        new_level = self.level_for_points(int(row["points"] or 0))
        old_level = int(row["familiarity_level"] or 1)
        if new_level != old_level:
            self._conn.execute(
                "UPDATE viewer_profiles SET familiarity_level = ? "
                "WHERE viewer_id = ?",
                (new_level, viewer_id),
            )
            logger.info("Viewer familiarity level changed",
                        viewer=viewer_id, from_level=old_level,
                        to_level=new_level, points=int(row["points"] or 0))

    # ── 公開API ─────────────────────────────────────────────

    def award_points(self, viewer_id: str, event: str, *,
                     display_name: str = "",
                     superchat_amount: int | None = None) -> int:
        """ポイント加算の公開入口。実際に加算できた点数を返す。

        event: comment / reaction / picked / superchat / visit。
        レート制限（時間窓の回数上限・1セッションの総量上限）に当たると
        0を返す。ポイント無効時・未知のイベントも0（呼び出し側は戻り値を
        気にしなくてよい）。visit は record_visit 経由が正規ルートだが、
        直接呼ばれた場合も同じ規則（セッション1回だけ）で処理する。
        """
        if not self.points_enabled or not viewer_id:
            return 0
        cfg = self._points_cfg
        now = time.time()
        with self._conn:
            self._ensure_profile_in_tx(viewer_id, display_name, now)
            if event == EVENT_VISIT:
                awarded = self._award_visit_in_tx(viewer_id, now)
            elif event == EVENT_COMMENT:
                if self._count_in_window(viewer_id, EVENT_COMMENT, now) \
                        >= cfg.comment_max_per_hour:
                    logger.info("Viewer points rate limited",
                                viewer=viewer_id, point_event=event)
                    return 0
                awarded = self._award_in_tx(
                    viewer_id, EVENT_COMMENT, cfg.comment_points, now)
            elif event == EVENT_REACTION:
                if self._count_in_window(viewer_id, EVENT_REACTION, now) \
                        >= cfg.reaction_max_per_hour:
                    logger.info("Viewer points rate limited",
                                viewer=viewer_id, point_event=event)
                    return 0
                awarded = self._award_in_tx(
                    viewer_id, EVENT_REACTION, cfg.reaction_points, now)
            elif event == EVENT_PICKED:
                if self._count_in_window(viewer_id, EVENT_PICKED, now) \
                        >= cfg.picked_max_per_hour:
                    logger.info("Viewer points rate limited",
                                viewer=viewer_id, point_event=event)
                    return 0
                awarded = self._award_in_tx(
                    viewer_id, EVENT_PICKED, cfg.picked_points, now)
            elif event == EVENT_SUPERCHAT:
                points = cfg.superchat_base_points
                if superchat_amount:
                    bonus = (int(superchat_amount) // 1000) \
                        * cfg.superchat_per_1000_points
                    points += min(bonus, cfg.superchat_bonus_cap)
                awarded = self._award_in_tx(
                    viewer_id, EVENT_SUPERCHAT, points, now)
            else:
                return 0
            if awarded > 0:
                self._sync_level_in_tx(viewer_id)
            return awarded

    def record_visit(self, viewer_id: str, display_name: str) -> ViewerProfile:
        """視聴者の来訪を記録し、更新後のプロファイルを返す。

        ポイント有効時は、セッションが変わった回だけ来訪ポイントと
        参加配信数を加算し、レベルは累積ポイントから決める。
        無効時は従来どおり来訪回数からレベルを決める。
        """
        if not viewer_id:
            viewer_id = "anonymous"
        now = time.time()

        with self._conn:
            row = self._conn.execute(
                "SELECT * FROM viewer_profiles WHERE viewer_id = ?", (viewer_id,)
            ).fetchone()

            if row is None:
                self._conn.execute(
                    """
                    INSERT INTO viewer_profiles (
                        viewer_id, display_name, visit_count,
                        first_seen_at, last_seen_at, familiarity_level, notes
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (viewer_id, display_name, 1, now, now, 1, ""),
                )
                logger.info("New viewer recorded", viewer=viewer_id, name=display_name)
            else:
                self._conn.execute(
                    """
                    UPDATE viewer_profiles
                    SET display_name = ?,
                        visit_count = ?,
                        last_seen_at = ?
                    WHERE viewer_id = ?
                    """,
                    (display_name, row["visit_count"] + 1, now, viewer_id),
                )

            if self.points_enabled:
                self._award_visit_in_tx(viewer_id, now)
                self._sync_level_in_tx(viewer_id)
            else:
                row2 = self._conn.execute(
                    "SELECT visit_count FROM viewer_profiles WHERE viewer_id = ?",
                    (viewer_id,),
                ).fetchone()
                self._conn.execute(
                    "UPDATE viewer_profiles SET familiarity_level = ? "
                    "WHERE viewer_id = ?",
                    (self.calc_familiarity(int(row2["visit_count"])), viewer_id),
                )

            row = self._conn.execute(
                "SELECT * FROM viewer_profiles WHERE viewer_id = ?", (viewer_id,)
            ).fetchone()
            return self._row_to_profile(row)

    @staticmethod
    def _row_to_profile(row: sqlite3.Row) -> ViewerProfile:
        keys = row.keys()
        return ViewerProfile(
            viewer_id=row["viewer_id"],
            display_name=row["display_name"],
            visit_count=row["visit_count"],
            first_seen_at=row["first_seen_at"],
            last_seen_at=row["last_seen_at"],
            familiarity_level=row["familiarity_level"],
            notes=row["notes"] or "",
            custom_nickname=row["custom_nickname"],
            points=int(row["points"] or 0) if "points" in keys else 0,
            streams_attended=int(row["streams_attended"] or 0)
            if "streams_attended" in keys else 0,
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
        return self._row_to_profile(row)

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
        if self.points_enabled:
            # ポイントは定性ラベルだけで伝える（キャラが具体的な数値を
            # 口にしないようにするため、数値そのものは載せない）
            lines.append(f"- 応援の蓄積: {self.points_band(profile.points)}")
            lines.append(
                "- 留意: 応援の蓄積は裏側のポイントで管理されている。"
                "具体的な数値は話題に出さない（聞かれても数字では答えない）。")
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
