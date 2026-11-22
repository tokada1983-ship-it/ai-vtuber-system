"""5時間検証ラン残課題4件の修正（2026-11-22 verifyfix）の回帰テスト

対象:
  1. 記憶整理の発動構造: 文脈リセットで整理カウンタ／pending を消さない
  2. 話題任期: 弧なし到達でも完遂記録が出る／時間だけの到達には最小
     発話数を要求する／経過はセッション開始以降でのみ測る／commit が
     arc_key を台帳へ保存し、弧の継続中もチェーン履歴が評価される
  3. テーマの言語化: テーマ話題は allow_web ゲート対象外で検索できる／
     プロンプトにテーマ名を口にする明示指示が入る
  4. 同一文漏れ: 反復チェックの比較窓を60発話へ拡大／リセット後も反復
     記憶を保持／短句の完全一致は2回目からブロック

外部サービスには依存せず、スタブで検証する（実リクエストは行わない）。
"""
from __future__ import annotations

import time

from backend.config import AppConfig
from backend.knowledge.models import WebDocument
from backend.knowledge.topic_planner import TopicPlanner
from backend.llm.prompt_builder import build_topic_prompt
from backend.memory.conversation_log import ConversationLog
from backend.models import ChatMessage
from backend.orchestrator import Orchestrator
from helpers import FakeHub, StubLLM, StubTTS, make_orchestrator

THEME = "ブラックフライデー"
TITLE = "新型ロボットが発表"
BF_DOC = WebDocument(title="ブラックフライデー特集", url="https://a.com/bf",
                     snippet="セール情報", source="google_news",
                     site="テスト")


class FakeSearch:
    last_error = ""

    def __init__(self, docs=None):
        self._docs = docs or []
        self.calls: list[str] = []

    async def search(self, query, limit=None, use_cache=True):
        self.calls.append(query)
        return list(self._docs)


def make_planner(**kw) -> TopicPlanner:
    args = dict(
        news=None,
        search=None,
        keyword_provider=lambda: [],
        theme_provider=lambda: THEME,
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


def ledger_entries(title, n, at=None, **extra):
    at = time.time() if at is None else at
    return [{"kind": "news", "title": title, "query": title,
             "at": at - i * 10, **extra} for i in range(n)]


# ── 1. 記憶整理の発動構造 ────────────────────────────────────

class TestConsolidationSurvivesReset:
    def test_clear_context_preserves_pending_and_counter(self):
        log = ConversationLog()
        for i in range(30):
            log.add_idle(f"発話{i}")
        assert log.turns_since_consolidation == 30
        pending_before = log.pending_size
        log.clear_context()
        assert log.history_size == 0
        assert log.pending_size == pending_before
        assert log.turns_since_consolidation == 30

    def test_consolidation_fires_after_context_clears(self):
        log = ConversationLog()
        # リセット（文脈クリア）を挟みながら50ターン溜めれば整理対象になる
        for round_ in range(3):
            for i in range(20):
                log.add_idle(f"発話{round_}-{i}")
            log.clear_context()
        assert log.should_consolidate(50)

    def test_full_clear_still_resets(self):
        log = ConversationLog()
        log.add_idle("発話")
        log.clear()
        assert log.pending_size == 0
        assert log.turns_since_consolidation == 0

    def test_orchestrator_reset_preserves_pending(self):
        orch, _hub = make_orchestrator(StubLLM("{}"), StubTTS())
        agent = orch.agents[0]
        for i in range(10):
            agent.log.add_idle(f"発話{i}")
        orch._reset_saturated_context("テスト")
        assert agent.log.history_size == 0
        assert agent.log.pending_size > 0
        assert agent.log.turns_since_consolidation == 10


# ── 2. 話題任期 ──────────────────────────────────────────────

class TestTenureCompletion:
    async def test_arcless_reach_records_completion(self):
        planner = make_planner()
        planner._ledger = ledger_entries(TITLE, 6)
        topic = await planner._try_tenure_transition(True)
        assert topic is None
        # 弧が開いていなくても完遂として数えられ、主題は除外される
        assert planner._arc_stats["tenure"] == 1
        assert planner._is_tenure_excluded(TITLE, "")

    def test_time_only_reach_needs_min_speeches(self):
        planner = make_planner()
        planner._born_at = time.time() - 2000
        # 発話1件・経過だけ長いチェーンは到達しない
        planner._ledger = ledger_entries(TITLE, 1,
                                         at=time.time() - 1300)
        assert planner._tenure_exceeded(time.time()) is None
        # 発話3件＋経過が閾値超なら到達する
        planner._ledger = ledger_entries(TITLE, 3,
                                         at=time.time() - 1300)
        assert planner._tenure_exceeded(time.time()) is not None

    def test_restored_chain_does_not_fire_at_session_start(self):
        planner = make_planner()
        # 前セッションから復元された台帳（開始時刻はセッション前）。
        # 経過はセッション開始以降でのみ測るため、開始直後は到達しない
        planner._ledger = ledger_entries(TITLE, 3,
                                         at=time.time() - 1700)
        assert planner._tenure_exceeded(time.time()) is None

    async def test_commit_stores_arc_key(self):
        planner = make_planner(search=FakeSearch([BF_DOC]))
        topic = await planner._plan_theme("", True)
        assert topic is not None and topic.arc_key
        planner.commit(topic)
        entry = planner._ledger[-1]
        assert entry["arc_key"] == topic.arc_key
        assert entry["source"] == "google_news"

    def test_theme_memory_entries_excluded_by_source(self):
        planner = make_planner()
        planner._born_at = time.time() - 2000
        planner._ledger = [
            {"kind": "theme", "title": "覚えていること", "query": THEME,
             "source": "memory", "at": time.time() - i * 10}
            for i in range(8)]
        assert planner._tenure_exceeded(time.time()) is None


# ── 3. テーマの言語化 ────────────────────────────────────────

class TestThemeWebExemption:
    async def test_theme_search_allowed_when_web_gated(self):
        search = FakeSearch([BF_DOC])
        planner = make_planner(search=search)
        # 全体の allow_web=False でも、テーマ検索が許可されていれば
        # 材料付きのテーマ話題になる（お題に劣化しない）
        topic = await planner.plan(kind="theme", allow_web=False,
                                   allow_web_theme=True)
        assert topic is not None and topic.kind == "theme"
        assert search.calls, "テーマ検索が実行されていない"
        assert topic.documents[0].source == "google_news"

    async def test_theme_degrades_without_theme_web(self):
        search = FakeSearch([BF_DOC])
        planner = make_planner(search=search)
        topic = await planner.plan(kind="theme", allow_web=False)
        assert topic is not None
        assert not search.calls
        assert topic.documents[0].source in ("theme", "memory")


class TestThemePromptNaming:
    def _cfg(self) -> AppConfig:
        cfg = AppConfig()
        cfg.character.stream_theme = THEME
        return cfg

    def test_prompt_topic_names_theme(self):
        docs = [WebDocument(title=f"配信テーマ: {THEME}",
                            snippet="自分の感想を交えて話す",
                            source="theme", site="配信テーマ")]
        prompt = build_topic_prompt(self._cfg(), docs, kind="theme",
                                    query=THEME)
        assert "一度は自然に口にしてください" in prompt

    def test_material_theme_prompt_names_theme(self):
        docs = [WebDocument(title="セール特集", url="https://a.com/2",
                            snippet="詳細", source="google_news",
                            site="テスト")]
        prompt = build_topic_prompt(self._cfg(), docs, kind="theme",
                                    query=THEME)
        assert "一度は口にしてください" in prompt


# ── 4. 同一文漏れ ────────────────────────────────────────────

LONG_SPEECH = "お茶が香ばしくて最高！ボクが調べてきた預金事情の最新ニュース、今からたっぷりお届けします"
SHORT_SPEECH = "ボクの調査能力、なかなかじゃない？"


class TestRepeatWindow:
    def _orch(self):
        orch, _hub = make_orchestrator(StubLLM("{}"), StubTTS())
        return orch

    def test_repeat_beyond_12_speeches_still_blocked(self):
        orch = self._orch()
        norm = Orchestrator._normalize_speech(LONG_SPEECH)
        orch._recent_speech_norms.append(norm)
        # 間に20発話挟んでも（旧窓12では溢れる）再演を捕捉できる
        for i in range(20):
            orch._recent_speech_norms.append(f"別の発話{i}ですよ")
        orch._recent_speech_norms = orch._recent_speech_norms[
            -Orchestrator._RECENT_SPEECH_NORMS_MAX:]
        assert orch._should_block_repeat(LONG_SPEECH)

    def test_reset_preserves_repeat_memory(self):
        orch = self._orch()
        norm = Orchestrator._normalize_speech(LONG_SPEECH)
        orch._recent_speech_norms.append(norm)
        orch._short_phrase_events.append((time.time(), "短句"))
        orch._reset_saturated_context("テスト")
        assert norm in orch._recent_speech_norms
        assert orch._short_phrase_events

    def test_short_phrase_tiered_allowance_kept(self):
        # 短句の完全一致は段階許可を維持する判断（2026-11-22
        # verifyfix）。2回目は許可、120秒窓で3回目からブロック
        orch = self._orch()
        norm = Orchestrator._normalize_speech(SHORT_SPEECH)
        assert len(norm) <= Orchestrator._SHORT_PHRASE_MAX_LEN
        orch._recent_speech_norms.append(norm)
        assert not orch._should_block_repeat(SHORT_SPEECH)
        now = time.time()
        orch._short_phrase_events.append((now, norm))
        orch._short_phrase_events.append((now, norm))
        assert orch._should_block_repeat(SHORT_SPEECH)

    def test_short_phrase_count_survives_reset(self):
        # リセット後も短句イベントが保持され、回数判定が盲目にならない
        orch = self._orch()
        norm = Orchestrator._normalize_speech(SHORT_SPEECH)
        orch._recent_speech_norms.append(norm)
        now = time.time()
        orch._short_phrase_events.append((now, norm))
        orch._short_phrase_events.append((now, norm))
        orch._reset_saturated_context("テスト")
        assert orch._should_block_repeat(SHORT_SPEECH)

    def test_short_phrase_variant_still_tiered(self):
        orch = self._orch()
        # 言い回しの亜種（完全一致でない短句）は従来どおり段階許可の対象
        orch._recent_speech_norms.append(
            Orchestrator._normalize_speech("ありがとうございます！"))
        # 正規化後に完全一致にならない亜種は、段階許可で通る
        assert not orch._should_block_repeat("ありがとうございますね！")
