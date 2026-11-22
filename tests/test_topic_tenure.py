"""話題の任期（topic tenure、2026-11-22 longstream）の回帰テスト

同じ主題（弧）が長くなりすぎた場合に、緊急リセット（LoopGuard や
飽和リセット）に頼らず、通常の判断として次話題へ移行する機構:
  - 主題チェーン（台帳ベース）の計測と閾値判定（発話数／経過時間）
  - 決定境界での強制クローズ（締め発話）→ 完遂として記録
  - 材料が無い場合は完遂として閉じて通常の新規選択へ委ねる
  - 移行した主題のクールダウン除外（ニュース選択・復帰）
  - 無効化スイッチと状態の永続化

外部サービスには依存せず、スタブで検証する（実リクエストは行わない）。
"""
from __future__ import annotations

import time

from backend.knowledge.models import WebDocument
from backend.knowledge.topic_planner import TopicPlanner

TITLE = "新型ロボットが発表"
FOLLOWUP = WebDocument(title="新型ロボットが発表 続報まとめ",
                       url="https://a.com/robot2", snippet="続報",
                       source="google_news", site="テスト")
SAME_DOC = WebDocument(title="新型ロボットが発表",
                       url="https://a.com/robot", snippet="初報",
                       source="google_news", site="テスト")
UNRELATED = WebDocument(title="地元の祭りの準備が進行中",
                        url="https://a.com/fes", snippet="祭り",
                        source="google_news", site="テスト")


class FakeSearch:
    """WebSearchClient のスタブ"""

    last_error = ""

    def __init__(self, docs=None):
        self._docs = docs or []
        self.calls: list[str] = []

    async def search(self, query, limit=None, use_cache=True):
        self.calls.append(query)
        return list(self._docs)


class FakeNews:
    """NewsFeed のスタブ（planner が使う unused_items / mark_used のみ）"""

    def __init__(self, docs):
        self._docs = list(docs)

    def unused_items(self):
        return list(self._docs)

    def mark_used(self, doc):
        self._docs = [d for d in self._docs if d.key != doc.key]


def make_planner(**kw) -> TopicPlanner:
    args = dict(
        news=None,
        search=None,
        keyword_provider=lambda: [],
        theme_provider=lambda: "",
        memory_provider=lambda: {
            "text": "先週の配信で視聴者と雑談した思い出の記録です"},
        state_path="",
        topic_tenure_speeches=6,
        topic_tenure_sec=1200.0,
        topic_tenure_cooldown_sec=1800.0,
        topic_tenure_window_sec=1800.0,
    )
    args.update(kw)
    return TopicPlanner(**args)


def ledger_entries(title, n, at=None):
    at = time.time() if at is None else at
    return [{"kind": "news", "title": title, "query": title,
             "at": at - i * 10} for i in range(n)]


def open_thread(planner, title=TITLE, speeches=1):
    now = time.time()
    key = planner._arc_key_for(title, title)
    planner._thread = {
        "kind": "news", "title": title, "query": title,
        "arc_key": key, "speeches": speeches, "continuations": 0,
        "started_at": now - 60, "at": now - 5, "doc_key": "doc-orig",
        "closing": False,
    }
    return key


# ── チェーン計測と閾値 ──────────────────────────────────────

class TestChainMeasurement:
    def test_chain_counts_ledger_repetitions(self):
        planner = make_planner()
        planner._ledger = ledger_entries(TITLE, 6)
        chain = planner._tenure_exceeded(time.time())
        assert chain is not None
        assert chain["speeches"] == 6

    def test_below_threshold(self):
        planner = make_planner()
        planner._ledger = ledger_entries(TITLE, 5)
        assert planner._tenure_exceeded(time.time()) is None

    def test_elapsed_threshold(self):
        # 経過時間での到達は、最小発話数（3件）を伴う場合に限る
        # （2026-11-22 verifyfix で OR 条件から見直し）。経過は
        # セッション開始以降でのみ測るため、開始時刻を過去に置く
        planner = make_planner()
        planner._born_at = time.time() - 2000
        planner._ledger = ledger_entries(TITLE, 3, at=time.time() - 1300)
        chain = planner._tenure_exceeded(time.time())
        assert chain is not None
        assert chain["elapsed"] >= 1200

    def test_memory_entries_not_counted(self):
        planner = make_planner()
        planner._ledger = [
            {"kind": "memory", "title": "覚えていること", "query": "",
             "at": time.time()} for _ in range(8)]
        assert planner._tenure_exceeded(time.time()) is None

    def test_disabled(self):
        planner = make_planner(topic_tenure_enabled=False)
        planner._ledger = ledger_entries(TITLE, 8)
        assert planner._tenure_exceeded(time.time()) is None

    def test_excluded_subject_does_not_refire(self):
        planner = make_planner()
        planner._ledger = ledger_entries(TITLE, 8)
        key = planner._arc_key_for(TITLE, TITLE)
        planner._register_tenure_exclusion(key, TITLE, time.time())
        assert planner._tenure_exceeded(time.time()) is None


# ── 移行の振る舞い ──────────────────────────────────────────

class TestTransition:
    async def test_forced_close_then_commit_counts_tenure(self):
        planner = make_planner(news=FakeNews([FOLLOWUP]),
                               search=FakeSearch([]))
        open_thread(planner)
        planner._ledger = ledger_entries(TITLE, 6)
        topic = await planner.plan()
        assert topic is not None
        # tenure 到達時は「締め」の発話として弧を進める
        assert topic.arc_stage == "close"
        planner.commit(topic)
        # 完遂として記録され、移行として数えられ、主題は除外される
        assert planner._arc_stats["tenure"] == 1
        assert planner._arc_stats["completed"] >= 1
        assert planner._thread is None
        assert planner._is_tenure_excluded(TITLE, "")

    async def test_no_material_closes_arc_as_completed(self):
        planner = make_planner(news=FakeNews([]), search=FakeSearch([]))
        open_thread(planner)
        planner._ledger = ledger_entries(TITLE, 6)
        topic = await planner.plan()
        # 締め材料が無くても弧は完遂として閉じられ、通常選択へ進む
        assert planner._thread is None
        assert planner._arc_stats["tenure"] == 1
        assert planner._is_tenure_excluded(TITLE, "")
        assert topic is None or topic.kind != "news"

    async def test_no_transition_below_threshold(self):
        planner = make_planner(news=FakeNews([FOLLOWUP]),
                               search=FakeSearch([]))
        open_thread(planner)
        planner._ledger = ledger_entries(TITLE, 2)
        await planner.plan()
        assert planner._arc_stats["tenure"] == 0
        assert not planner._is_tenure_excluded(TITLE, "")


# ── 除外の適用 ──────────────────────────────────────────────

class TestExclusionApplication:
    def test_news_pick_avoids_excluded(self):
        planner = make_planner(news=FakeNews([SAME_DOC, UNRELATED]))
        key = planner._arc_key_for(TITLE, TITLE)
        planner._register_tenure_exclusion(key, TITLE, time.time())
        doc = planner._pick_news_doc()
        assert doc is not None and doc.key == UNRELATED.key

    def test_news_pick_falls_back_when_all_excluded(self):
        planner = make_planner(news=FakeNews([SAME_DOC]))
        key = planner._arc_key_for(TITLE, TITLE)
        planner._register_tenure_exclusion(key, TITLE, time.time())
        doc = planner._pick_news_doc()
        # 全滅時は話題を枯らさないことを優先し、従来プールから選ぶ
        assert doc is not None and doc.key == SAME_DOC.key

    async def test_resume_skips_excluded(self):
        planner = make_planner()
        key = planner._arc_key_for(TITLE, TITLE)
        planner._register_tenure_exclusion(key, TITLE, time.time())
        planner._stack = [{"kind": "news", "title": TITLE, "query": TITLE,
                           "arc_key": key,
                           "suspended_at": time.time() - 10,
                           "speeches": 1}]
        assert await planner._try_resume(True) is None
        assert planner._stack == []

    def test_exclusion_expires(self):
        planner = make_planner(topic_tenure_cooldown_sec=0.0)
        key = planner._arc_key_for(TITLE, TITLE)
        planner._register_tenure_exclusion(key, TITLE, time.time())
        assert not planner._is_tenure_excluded(TITLE, "")

    def test_status_reports_tenure(self):
        planner = make_planner()
        key = planner._arc_key_for(TITLE, TITLE)
        planner._register_tenure_exclusion(key, TITLE, time.time())
        st = planner.status()
        assert st["tenure"]["enabled"] is True
        assert len(st["tenure"]["excluded"]) == 1

    def test_exclusion_persists(self, tmp_path):
        state = str(tmp_path / "planner_state.json")
        p1 = make_planner(state_path=state)
        key = p1._arc_key_for(TITLE, TITLE)
        p1._register_tenure_exclusion(key, TITLE, time.time())
        p2 = make_planner(state_path=state)
        assert p2._is_tenure_excluded(TITLE, "")
