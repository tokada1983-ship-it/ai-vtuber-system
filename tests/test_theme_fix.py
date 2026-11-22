"""配信テーマ機構の改修（2026-11-22 themefix）の回帰テスト

BFラン（2026-11-21）で「stream_theme を設定しても発話のテーマ言及が
0件」だった問題への対策:
  - TopicPlanner の話題決定にテーマ枠（占有率目標）を新設
  - テーマ語を反復抑制（ban）の対象から除外
  - force_topic をプランナの優先キューへの種まきに変更
  - curiosity クエリを「種そのまま／テーマ＋キーワード1つ」に単純化
  - テーマ言及率の計測（status の stream_theme）

外部サービスには依存せず、スタブで検証する（実リクエストは行わない）。
"""
from __future__ import annotations

from backend.config import AppConfig
from backend.knowledge.models import WebDocument
from backend.knowledge.topic_planner import TopicPlanner
from backend.llm.prompt_builder import build_topic_prompt

THEME = "ブラックフライデー"

BF_DOC = WebDocument(title="ブラックフライデー特集", url="https://a.com/bf",
                     snippet="セール情報", source="google_news", site="テスト")


class FakeSearch:
    """WebSearchClient のスタブ（固定の資料を返し、呼び出しを記録）"""

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
        theme_provider=lambda: THEME,
        memory_provider=lambda: {
            "text": "先週の配信で視聴者と雑談した思い出の記録です"},
        state_path="",
    )
    args.update(kw)
    return TopicPlanner(**args)


# ── テーマ枠（占有率目標） ────────────────────────────────────

class TestThemeQuota:
    async def test_theme_topics_reach_target_share(self):
        planner = make_planner(search=FakeSearch([BF_DOC]))
        kinds = []
        for _ in range(12):
            topic = await planner.plan()
            planner.commit(topic)
            kinds.append(topic.kind)
        # 台帳が空の最初の決定はテーマ枠が効く
        assert kinds[0] == "theme"
        # 目標 0.34 に対して、12件中おおむね1/3がテーマ話題になる
        assert kinds.count("theme") >= 3
        assert planner.theme_share() >= 0.25

    async def test_no_theme_no_theme_topics(self):
        planner = make_planner(search=FakeSearch([BF_DOC]),
                               theme_provider=lambda: "")
        for _ in range(6):
            topic = await planner.plan()
            planner.commit(topic)
            assert topic.kind != "theme"

    async def test_theme_disabled_by_zero_target(self):
        planner = make_planner(search=FakeSearch([BF_DOC]),
                               theme_share_target=0.0)
        for _ in range(6):
            topic = await planner.plan()
            planner.commit(topic)
            assert topic.kind != "theme"


# ── ban 免除 ─────────────────────────────────────────────────

class TestThemeBanExemption:
    def test_theme_word_is_not_banned(self):
        planner = make_planner()
        planner.ban_word(THEME)
        assert THEME not in planner.banned_words()
        # テーマでない語は従来どおり ban される
        planner.ban_word("元気")
        assert "元気" in planner.banned_words()

    def test_build_query_keeps_theme_after_ban_attempt(self):
        planner = make_planner(keyword_provider=lambda: ["家電"])
        planner.ban_word(THEME)
        q = planner._build_query()
        assert q.startswith(THEME)


# ── クエリ単純化 ─────────────────────────────────────────────

class TestQuerySimplification:
    def test_theme_queries_are_simple(self):
        for _ in range(20):
            planner = make_planner(keyword_provider=lambda: ["家電"])
            q = planner._build_query()
            # テーマ単体か、テーマ＋キーワード1つだけ。テンプレート
            # （最新／おすすめ等）の付いた複合クエリにはならない
            assert q in (THEME, f"{THEME} 家電")

    def test_keyword_only_query_has_no_template(self):
        planner = make_planner(theme_provider=lambda: "",
                               keyword_provider=lambda: ["家電"])
        assert planner._build_query() == "家電"


# ── force_topic の種まき ─────────────────────────────────────

class TestForcedTopics:
    async def test_forced_news_is_filtered_by_theme(self):
        docs = [
            WebDocument(title="ブラックフライデーで家電が大セール",
                        url="https://a.com/1", snippet="詳細",
                        source="google_news", site="テスト"),
            WebDocument(title="日経平均が続伸した",
                        url="https://a.com/2", snippet="詳細",
                        source="google_news", site="テスト"),
        ]
        planner = make_planner(news=FakeNews(docs))
        planner.enqueue_forced_topic(kind="news", query=THEME)
        topic = await planner.plan()
        assert topic.kind == "news"
        assert THEME in topic.documents[0].title
        assert planner.status()["stream_theme"]["forced_pending"] == 0

    async def test_forced_query_becomes_theme_topic(self):
        planner = make_planner(search=FakeSearch([BF_DOC]))
        planner.enqueue_forced_topic(kind="", query="冬のセール")
        topic = await planner.plan()
        assert topic.kind == "theme"
        assert topic.query == "冬のセール"

    async def test_forced_auto_uses_stream_theme(self):
        planner = make_planner(search=FakeSearch([BF_DOC]))
        planner.enqueue_forced_topic(kind="auto")
        topic = await planner.plan()
        assert topic.kind == "theme"
        assert topic.query == THEME


# ── テーマ話題の材料フォールバック ───────────────────────────

class TestThemeMaterialFallback:
    async def test_theme_topic_without_material_becomes_prompt_topic(self):
        planner = make_planner(
            search=FakeSearch([]),
            memory_provider=lambda: {
                "text": "全然関係ない日常の思い出メモです"})
        topic = await planner.plan(kind="theme")
        assert topic.kind == "theme"
        # 検索も記憶も材料が無ければ、お題ドキュメント（source="theme"）
        assert topic.documents and topic.documents[0].source == "theme"


# ── テーマ言及率の計測 ───────────────────────────────────────

class TestThemeMentionStats:
    def test_mentions_are_counted(self):
        planner = make_planner()
        planner.note_speech(idle=True,
                            text="今日はブラックフライデーの話をしよう")
        planner.note_speech(idle=True, text="別の話題の発話です")
        st = planner.status()["stream_theme"]
        assert st["theme"] == THEME
        assert st["speeches"] == 2
        assert st["mentions"] == 1
        assert st["mention_rate"] == 0.5


# ── プロンプト（案C） ────────────────────────────────────────

class TestThemePrompt:
    def _cfg(self) -> AppConfig:
        cfg = AppConfig()
        cfg.character.stream_theme = THEME
        return cfg

    def test_topic_prompt_adds_theme_note_for_other_kinds(self):
        docs = [WebDocument(title="猫が惑星発見", url="https://a.com/1",
                            snippet="詳細", site="ねこ新聞")]
        prompt = build_topic_prompt(self._cfg(), docs, kind="news")
        assert THEME in prompt
        assert "無理にねじ込まない" in prompt

    def test_theme_topic_prompt_with_material(self):
        docs = [WebDocument(title="セール特集", url="https://a.com/2",
                            snippet="詳細", source="google_news",
                            site="テスト")]
        prompt = build_topic_prompt(self._cfg(), docs, kind="theme",
                                    query=THEME)
        assert "配信テーマに沿って" in prompt

    def test_theme_prompt_topic_without_material(self):
        docs = [WebDocument(title=f"配信テーマ: {THEME}",
                            snippet="自分の感想を交えて話す",
                            source="theme", site="配信テーマ")]
        prompt = build_topic_prompt(self._cfg(), docs, kind="theme",
                                    query=THEME)
        assert "自律トークモード（配信テーマ）" in prompt
        assert "<web_result>" not in prompt
