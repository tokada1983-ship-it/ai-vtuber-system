"""視聴者ポイント（Phase 4 / 組み込み検討書 §5 第1段）のテスト

ポイント台帳の加算ルール・レート制限・レベル移行、既存プロフィールの
移行、build_prompt への差し込み、コメント選別との整合（落とした
コメントのポイント扱い）を、外部サービスなしで検証する。
"""
import asyncio
import sqlite3
from datetime import datetime
from typing import AsyncIterator

from backend.config import ViewerPointsConfig, load_config
from backend.memory.viewer_store import ViewerProfileStore
from backend.models import Comment
from backend.sources.base import CommentSource
from helpers import RAW, StubLLM, StubTTS, make_orchestrator


def make_store(tmp_path, session_id="s1", **cfgkw):
    cfg = ViewerPointsConfig(**cfgkw)
    return ViewerProfileStore(tmp_path / "viewers.db",
                              points_cfg=cfg, session_id=session_id)


def reopen_store(tmp_path, session_id, **cfgkw):
    cfg = ViewerPointsConfig(**cfgkw)
    return ViewerProfileStore(tmp_path / "viewers.db",
                              points_cfg=cfg, session_id=session_id)


# ── 加算ルール ──────────────────────────────────────────────

class TestAwardRules:
    def test_comment_points(self, tmp_path):
        store = make_store(tmp_path)
        assert store.award_points("u1", "comment") == 2
        assert store.get_profile("u1").points == 2

    def test_reaction_points(self, tmp_path):
        store = make_store(tmp_path)
        assert store.award_points("u1", "reaction") == 1
        assert store.get_profile("u1").points == 1

    def test_picked_points(self, tmp_path):
        store = make_store(tmp_path)
        assert store.award_points("u1", "picked") == 3
        assert store.get_profile("u1").points == 3

    def test_superchat_base_only(self, tmp_path):
        store = make_store(tmp_path)
        assert store.award_points("u1", "superchat") == 10

    def test_superchat_amount_bonus(self, tmp_path):
        store = make_store(tmp_path)
        # 5,000円 → 基本10 + 1,000円ごと1点で +5
        assert store.award_points("u1", "superchat",
                                  superchat_amount=5000) == 15

    def test_superchat_bonus_capped(self, tmp_path):
        store = make_store(tmp_path)
        # 金額上乗せは上限10点まで
        assert store.award_points("u1", "superchat",
                                  superchat_amount=999999) == 20

    def test_unknown_event_is_zero(self, tmp_path):
        store = make_store(tmp_path)
        assert store.award_points("u1", "gift") == 0
        assert store.get_profile("u1").points == 0

    def test_award_creates_profile(self, tmp_path):
        store = make_store(tmp_path)
        store.award_points("newbie", "comment")
        profile = store.get_profile("newbie")
        assert profile is not None and profile.points == 2

    def test_events_are_ledgered(self, tmp_path):
        store = make_store(tmp_path)
        store.award_points("u1", "comment")
        store.award_points("u1", "picked")
        rows = store._conn.execute(
            "SELECT event, points FROM viewer_point_events "
            "WHERE viewer_id = 'u1' ORDER BY id").fetchall()
        assert [(r["event"], r["points"]) for r in rows] == [
            ("comment", 2), ("picked", 3)]


# ── レート制限 ──────────────────────────────────────────────

class TestRateLimits:
    def test_comment_hourly_cap(self, tmp_path):
        store = make_store(tmp_path)  # 既定: コメント加算は1時間に10回まで
        results = [store.award_points("u1", "comment") for _ in range(12)]
        assert results[:10] == [2] * 10
        assert results[10:] == [0, 0]
        assert store.get_profile("u1").points == 20

    def test_reaction_hourly_cap(self, tmp_path):
        store = make_store(tmp_path)  # 既定: リアクションは1時間に5回まで
        results = [store.award_points("u1", "reaction") for _ in range(7)]
        assert results == [1] * 5 + [0, 0]

    def test_picked_hourly_cap(self, tmp_path):
        store = make_store(tmp_path, picked_max_per_hour=3)
        results = [store.award_points("u1", "picked") for _ in range(5)]
        assert results == [3] * 3 + [0, 0]

    def test_session_cap(self, tmp_path):
        store = make_store(tmp_path, session_point_cap=25,
                           comment_max_per_hour=1000)
        total = sum(store.award_points("u1", "comment") for _ in range(20))
        assert total == 25
        assert store.get_profile("u1").points == 25

    def test_superchat_exempt_from_session_cap(self, tmp_path):
        store = make_store(tmp_path, session_point_cap=10,
                           comment_max_per_hour=1000)
        # 通常イベントで上限まで埋める
        assert sum(store.award_points("u1", "comment")
                   for _ in range(5)) == 10
        assert store.award_points("u1", "comment") == 0
        # スパチャは上限の対象外で、上限を消費もしない
        assert store.award_points("u1", "superchat",
                                  superchat_amount=5000) == 15
        assert store.get_profile("u1").points == 25


# ── 来訪と参加配信数 ────────────────────────────────────────

class TestVisitAndAttendance:
    def test_visit_once_per_session(self, tmp_path):
        store = make_store(tmp_path)
        store.record_visit("u1", "アリス")
        store.record_visit("u1", "アリス")
        profile = store.record_visit("u1", "アリス")
        assert profile.visit_count == 3
        assert profile.points == 10          # 来訪ポイントはセッション1回だけ
        assert profile.streams_attended == 1

    def test_visit_again_in_new_session(self, tmp_path):
        store = make_store(tmp_path, session_id="s1")
        store.record_visit("u1", "アリス")
        store.close()
        store2 = reopen_store(tmp_path, session_id="s2")
        profile = store2.record_visit("u1", "アリス")
        assert profile.points == 20
        assert profile.streams_attended == 2

    def test_attendance_milestone_bonus(self, tmp_path):
        kw = dict(attendance_milestone_every=2, attendance_milestone_points=20)
        store = make_store(tmp_path, session_id="s1", **kw)
        store.record_visit("u1", "アリス")
        assert store.get_profile("u1").points == 10
        store.close()
        store2 = reopen_store(tmp_path, session_id="s2", **kw)
        profile = store2.record_visit("u1", "アリス")
        # 2回目の参加で節目ボーナス +20（来訪10と合わせて累計40）
        assert profile.streams_attended == 2
        assert profile.points == 40

    def test_visit_via_award_points(self, tmp_path):
        store = make_store(tmp_path)
        assert store.award_points("u1", "visit") == 10
        assert store.award_points("u1", "visit") == 0  # 同一セッションは1回だけ


# ── レベル移行 ──────────────────────────────────────────────

class TestLevels:
    def test_level_for_points_boundaries(self, tmp_path):
        store = make_store(tmp_path)
        assert store.level_for_points(0) == 1
        assert store.level_for_points(49) == 1
        assert store.level_for_points(50) == 2
        assert store.level_for_points(199) == 2
        assert store.level_for_points(200) == 3
        assert store.level_for_points(500) == 4
        assert store.level_for_points(999) == 4
        assert store.level_for_points(1000) == 5

    def test_level_up_by_awards(self, tmp_path):
        store = make_store(tmp_path, level_thresholds=[0, 10, 20, 30, 40],
                           comment_max_per_hour=1000, session_point_cap=10000)
        for _ in range(5):   # 2点 × 5 = 10点 → Lv2
            store.award_points("u1", "comment")
        profile = store.get_profile("u1")
        assert profile.points == 10
        assert profile.familiarity_level == 2
        assert profile.level_name == "顔見知り"
        for _ in range(5):   # 累計20点 → Lv3
            store.award_points("u1", "comment")
        profile = store.get_profile("u1")
        assert profile.familiarity_level == 3
        assert profile.level_name == "常連"

    def test_level_follows_points_not_visits(self, tmp_path):
        # ポイント有効時は、来訪（コメント処理）回数を重ねても
        # ポイントが伸びなければレベルは上がらない
        store = make_store(tmp_path)
        for _ in range(10):
            profile = store.record_visit("u1", "アリス")
        assert profile.visit_count == 10
        assert profile.points == 10  # 来訪はセッション1回分だけ
        assert profile.familiarity_level == 1


# ── 既存プロフィールの移行 ──────────────────────────────────

def make_legacy_db(path):
    """Phase 3 までのスキーマ（ポイント列なし）でDBを作る"""
    conn = sqlite3.connect(str(path))
    conn.execute(
        """
        CREATE TABLE viewer_profiles (
            viewer_id         TEXT PRIMARY KEY,
            display_name      TEXT NOT NULL,
            visit_count       INTEGER NOT NULL DEFAULT 1,
            first_seen_at     REAL NOT NULL,
            last_seen_at      REAL NOT NULL,
            familiarity_level INTEGER NOT NULL DEFAULT 1,
            notes             TEXT NOT NULL DEFAULT '',
            custom_nickname   TEXT
        )
        """)
    conn.execute(
        "INSERT INTO viewer_profiles VALUES "
        "('regular', '常連さん', 5, 1000.0, 2000.0, 3, '', NULL)")
    conn.execute(
        "INSERT INTO viewer_profiles VALUES "
        "('veteran', '古参さん', 25, 1000.0, 2000.0, 5, '', NULL)")
    conn.execute(
        "INSERT INTO viewer_profiles VALUES "
        "('newface', '新人さん', 1, 1000.0, 2000.0, 1, '', NULL)")
    conn.commit()
    conn.close()


class TestMigration:
    def test_legacy_levels_preserved_as_points(self, tmp_path):
        db = tmp_path / "viewers.db"
        make_legacy_db(db)
        store = ViewerProfileStore(db, points_cfg=ViewerPointsConfig(),
                                   session_id="s1")
        regular = store.get_profile("regular")
        assert regular.points == 200           # Lv3 の開始ポイントへ換算
        assert regular.familiarity_level == 3  # レベルは巻き戻らない
        veteran = store.get_profile("veteran")
        assert veteran.points == 1000
        assert veteran.familiarity_level == 5
        newface = store.get_profile("newface")
        assert newface.points == 0             # Lv1 は換算対象外

    def test_migrated_viewer_keeps_level_after_visit(self, tmp_path):
        db = tmp_path / "viewers.db"
        make_legacy_db(db)
        store = ViewerProfileStore(db, points_cfg=ViewerPointsConfig(),
                                   session_id="s1")
        profile = store.record_visit("regular", "常連さん")
        assert profile.points == 210           # 換算200 + 来訪10
        assert profile.familiarity_level == 3
        assert profile.streams_attended == 1


# ── 無効化と後方互換 ────────────────────────────────────────

class TestDisabledAndLegacy:
    def test_no_config_is_legacy(self, tmp_path):
        # 設定なしで作ったストアは従来動作（来訪回数ベース）
        store = ViewerProfileStore(tmp_path / "viewers.db")
        for _ in range(5):
            profile = store.record_visit("u1", "アリス")
        assert profile.familiarity_level == 3
        assert profile.points == 0
        assert store.award_points("u1", "comment") == 0
        assert "応援の蓄積" not in store.build_prompt("u1")

    def test_disabled_config_is_legacy(self, tmp_path):
        store = make_store(tmp_path, enabled=False)
        for _ in range(5):
            profile = store.record_visit("u1", "アリス")
        assert profile.familiarity_level == 3
        assert store.award_points("u1", "comment") == 0
        assert "応援の蓄積" not in store.build_prompt("u1")


# ── プロンプトへの差し込み ──────────────────────────────────

class TestBuildPrompt:
    def test_prompt_has_band_not_numbers(self, tmp_path):
        store = make_store(tmp_path, comment_points=23,
                           comment_max_per_hour=100, session_point_cap=10000)
        for _ in range(10):   # 23点 × 10 = 230点
            store.award_points("u1", "comment")
        profile = store.get_profile("u1")
        assert profile.points == 230
        prompt = store.build_prompt("u1")
        assert "応援の蓄積" in prompt
        assert "少しずつ積み重なってきている" in prompt
        assert "具体的な数値は話題に出さない" in prompt
        # ポイントの生数値はプロンプトに載せない（キャラが口にしないため）
        assert "230" not in prompt

    def test_prompt_level_uses_points(self, tmp_path):
        store = make_store(tmp_path, level_thresholds=[0, 10, 20, 30, 40],
                           comment_max_per_hour=1000, session_point_cap=10000)
        for _ in range(10):   # 20点 → Lv3
            store.award_points("u1", "comment")
        prompt = store.build_prompt("u1")
        assert "親密度Lv3" in prompt
        assert "常連" in prompt


# ── 設定 ────────────────────────────────────────────────────

class TestConfig:
    def test_defaults(self):
        cfg = ViewerPointsConfig()
        assert cfg.enabled is True
        assert cfg.visit_points == 10
        assert cfg.level_thresholds == [0, 50, 200, 500, 1000]

    def test_load_config_section(self, tmp_path):
        path = tmp_path / "config.yaml"
        path.write_text(
            "viewer_points:\n"
            "  enabled: false\n"
            "  comment_points: 5\n"
            "  level_thresholds: [0, 10, 20, 30, 40]\n",
            encoding="utf-8")
        cfg = load_config(path)
        assert cfg.viewer_points.enabled is False
        assert cfg.viewer_points.comment_points == 5
        assert cfg.viewer_points.level_thresholds == [0, 10, 20, 30, 40]


# ── Orchestrator 結合（コメント選別との整合） ────────────────

class FakeSource(CommentSource):
    """用意したコメントを順に流すだけのソース"""

    def __init__(self, comments: list[Comment]):
        self._comments = comments

    async def start(self, video_id: str) -> None:
        pass

    async def stop(self) -> None:
        pass

    async def comments(self) -> AsyncIterator[Comment]:
        for c in self._comments:
            yield c

    @property
    def is_healthy(self) -> bool:
        return True


async def run_ingest(orch, comments):
    orch._source = FakeSource(comments)
    orch._running = True
    task = asyncio.create_task(orch._comment_ingest_loop())
    await asyncio.wait_for(task, timeout=10)


def make_comment(msg: str, author: str = "視聴者A", channel: str = "chA",
                 **kw) -> Comment:
    return Comment(id="c", author=author, author_channel_id=channel,
                   message=msg, timestamp=datetime.now(), **kw)


class TestOrchestratorPoints:
    async def test_dropped_comments_points_by_category(self):
        orch, _hub = make_orchestrator(StubLLM(RAW), StubTTS())
        # 通常コメントは確率0で落とす（always系だけが拾われる設定）
        orch._cfg.comment_filter.base_pickup_probability = 0.0

        comments = [
            make_comment("こんにちは！", author="視聴者A", channel="chA"),
            make_comment("草", author="視聴者A", channel="chA"),
            make_comment("こんにちは！", author="視聴者A", channel="chA"),
            make_comment("ああああああ", author="視聴者B", channel="chB"),
            make_comment("今日はいい天気ですね", author="視聴者B", channel="chB"),
        ]
        await run_ingest(orch, comments)

        store = orch._viewer_store
        a = store.get_profile("視聴者A")
        # A: 来訪10（最初の落選コメントで記録）＋ リアクション1。
        # 重複コメントは0点。拾われた初コメはまだ処理前なので加算なし
        assert a.points == 11
        b = store.get_profile("視聴者B")
        # B: 来訪10（連打コメントで記録、連打自体は0点）＋
        # 確率で落とした通常コメントの参加分2
        assert b.points == 12

    async def test_responded_comment_points(self):
        orch, _hub = make_orchestrator(StubLLM(RAW), StubTTS())
        await orch._process_comment(
            make_comment("こんばんは！", author="視聴者C", channel="chC"))
        profile = orch._viewer_store.get_profile("視聴者C")
        # 来訪10 + コメント2 + 拾われた3
        assert profile.points == 15
        assert profile.visit_count == 1

    async def test_responded_superchat_points(self):
        orch, _hub = make_orchestrator(StubLLM(RAW), StubTTS())
        await orch._process_comment(make_comment(
            "応援してます！", author="視聴者D", channel="chD",
            is_superchat=True, superchat_amount=3000))
        profile = orch._viewer_store.get_profile("視聴者D")
        # 来訪10 + コメント2 + スパチャ(基本10 + 3,000円で+3) + 拾われた3
        assert profile.points == 28
