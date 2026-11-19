"""Phase 3 の回帰テスト: 事実質問→調査の接続と検索まわりの質

2026-11-17 の1時間テストで発生した問題への対策:
  - トリガーワードを含まない事実質問が無回答に落ちた（P3）
  - 「調べるね」と宣言だけして調査が走らなかった（宣言と実行の不一致）
  - 検索プロバイダの 429 連発と、curiosity クエリの自己言及（P5）

外部サービスには依存せず、スタブで検証する（実リクエストは行わない）。
"""
from __future__ import annotations

import json
import random
import tempfile
import time
from pathlib import Path

import httpx
import pytest
import yaml

from backend.config import AppConfig, CharacterConfig, EnsembleConfig
from backend.knowledge.models import WebDocument
from backend.knowledge.news import NewsFeed
from backend.knowledge.topic_planner import Topic, TopicPlanner
from backend.knowledge.web_search import WebSearchClient
from backend.llm.prompt_builder import (build_answer_pending_prompt,
                                        build_topic_prompt)
from backend.orchestrator import Orchestrator
from helpers import (RAW, FakeHub, NoAudio, StubLLM, StubTTS, comment,
                     make_orchestrator)

FACT_QUESTION = "昨日の日経平均株価の終値って、いくらだったか分かりますか？"
NIKKEI_DOC = WebDocument(
    title="日経平均終値", url="https://example.com/nikkei",
    snippet="終値は59,662.02円でした", source="google_news", site="テスト")


def _json_raw(speech: str) -> str:
    return json.dumps({"emotion": "neutral", "motion": "idle",
                       "skip": False, "speech": speech}, ensure_ascii=False)


def _write_char(dirpath: Path, name: str) -> str:
    path = dirpath / f"{name}.yaml"
    path.write_text(yaml.safe_dump(
        {"name": name, "description": f"{name}の説明",
         "personality": f"{name}の口調"}, allow_unicode=True), encoding="utf-8")
    return str(path)


class RecordingLLM(StubLLM):
    """受け取った system_prompt を記録するLLM（streaming / 一括の両対応）"""

    def __init__(self, raw: str):
        super().__init__(raw)
        self.prompts: list[str] = []

    async def generate(self, system_prompt, messages, *, route="respond", temperature=None):
        self.prompts.append(system_prompt)
        return await super().generate(system_prompt, messages, route=route)

    async def generate_stream(self, system_prompt, messages, *, route="respond", temperature=None):
        self.prompts.append(system_prompt)
        async for chunk in super().generate_stream(system_prompt, messages,
                                                  route=route):
            yield chunk


class FakeSearch:
    """WebSearchClient のスタブ（呼び出しを記録し、固定の資料を返す）"""

    last_error = ""

    def __init__(self, docs: list[WebDocument]):
        self._docs = docs
        self.calls: list[str] = []

    async def search(self, query, limit=None, use_cache=True):
        self.calls.append(query)
        return list(self._docs)


def make_duo_p3(llm_a=None, llm_b=None):
    """掛け合いなし（確率0）の2体オーケストレータを作る"""
    tmp = Path(tempfile.mkdtemp(prefix="aivt_p3_"))
    cfg = AppConfig()
    cfg.stt.enabled = False
    cfg.ollama.auto_start = False
    cfg.ollama.preload_on_start = False
    cfg.tts.irodori_server.auto_start = False
    cfg.knowledge.enabled = False
    cfg.characters = [
        CharacterConfig(name="Character1", id="Character1", data_dir=str(tmp),
                        prompt_file=_write_char(tmp, "Character1"),
                        aliases=["character1"]),
        CharacterConfig(name="Character2", id="Character2", data_dir=str(tmp),
                        prompt_file=_write_char(tmp, "Character2")),
    ]
    cfg.ensemble = EnsembleConfig(turn_gap_sec=0, idle_banter_probability=0.0,
                                  banter_probability=0.0)
    hub = FakeHub()
    orch = Orchestrator(cfg, hub)
    a, b = orch.agents
    a.llm, a.tts = llm_a or RecordingLLM(RAW), StubTTS()
    b.llm, b.tts = llm_b or RecordingLLM(RAW), StubTTS()
    orch._audio = NoAudio()
    orch._audio_cache = Path(tempfile.mkdtemp(prefix="aivt_audio_"))
    return orch, hub


def _enable_knowledge(orch: Orchestrator, search) -> TopicPlanner:
    """知識系を有効化し、検索スタブと実プランナ（状態保存なし）を差す"""
    orch._cfg.knowledge.enabled = True
    planner = TopicPlanner(news=None, search=None, state_path="")
    orch._planner = planner
    orch._search = search
    return planner


def _speakers(hub: FakeHub) -> list[str]:
    return [d["character"] for ch, d in hub.events if ch == "speech_end"]


# ── (1) 事実質問の判定 ──────────────────────────────────────

def test_needs_search_classifies_factual_question():
    orch, _ = make_orchestrator(StubLLM(RAW), StubTTS())
    orch._cfg.knowledge.enabled = True
    orch._search = object()          # 検索クライアントがある想定

    # 2026-11-17 に落ちた実例（トリガーワードを1語も含まない）
    assert orch._needs_search(FACT_QUESTION)
    factual, reason = orch._classify_factual_question(FACT_QUESTION)
    assert factual and reason
    # トリガーワード経由は従来どおり
    assert orch._needs_search("最新のニュースを教えて")
    # カテゴリ語＋疑問形
    assert orch._needs_search("明日の天気はどう？")
    # 通常の感想コメントは対象外
    assert not orch._needs_search("今日の配信も楽しいね！")
    assert not orch._needs_search("こんにちは！今日も元気だね")
    # 意見質問は対象外（パン派／ごはん派の実験コメント）
    assert not orch._needs_search("朝ごはん、パン派？ごはん派？")
    assert not orch._needs_search("それ、どう思う？")


# ── (2) 間隔ゲートでは質問を捨てずキューへ回す ──────────────

@pytest.mark.asyncio
async def test_interval_gate_defers_question_to_queue():
    llm = RecordingLLM(_json_raw("わかった、調べるね。あとで教えるね。"))
    orch, hub = make_duo_p3(llm_a=llm)
    search = FakeSearch([NIKKEI_DOC])
    planner = _enable_knowledge(orch, search)
    orch._last_search_time = time.time()     # 直前に検索済み＝ゲートが閉じている

    await orch._process_comment(comment(FACT_QUESTION, author="視聴者A"))

    entries = planner.pending_answers()
    assert len(entries) == 1
    assert entries[0]["state"] == "waiting"
    assert entries[0]["question"] == FACT_QUESTION
    assert search.calls == []                       # 同期検索は走っていない
    assert planner._comment_hook is None            # hook に残らない
    # 即応は「依頼済み」プロンプトで生成された（この分岐でのみ宣言が許可される）
    assert any("依頼済み" in p for p in llm.prompts)
    assert _speakers(hub) == ["Character1"]


# ── (3) 保留回答は plan() で comment hook より優先される ────

@pytest.mark.asyncio
async def test_pending_answer_beats_comment_hook_in_plan():
    planner = TopicPlanner(news=None, search=None, state_path="")
    pid = planner.enqueue_answer("視聴者A", FACT_QUESTION,
                                 query="日経平均株価 終値")
    planner.attach_answer_documents(pid, [NIKKEI_DOC])
    # 別コメントの hook を発火可能な状態で用意しておく
    planner.note_comment("視聴者B", "さっきのゲームの話、面白かったね")
    planner._comment_hook["at"] = time.time() - 100
    assert planner.has_deliverable_answer()

    topic = await planner.plan()

    assert topic.kind == "answer"
    assert topic.pending_id == pid
    assert topic.author == "視聴者A"
    assert "59,662.02" in topic.documents[0].snippet
    planner.commit(topic)
    assert planner.pending_answers() == []          # 発話成功で消費される


# ── (4) ワーカーの結果が Topic(kind="answer") として差し込まれる ──

@pytest.mark.asyncio
async def test_worker_result_becomes_answer_topic():
    orch, _ = make_orchestrator(StubLLM(RAW), StubTTS())
    search = FakeSearch([NIKKEI_DOC])
    planner = _enable_knowledge(orch, search)
    pid = planner.enqueue_answer("視聴者A", FACT_QUESTION,
                                 query="日経平均株価 終値")

    assert await orch._research_one_pending() is True
    assert search.calls == ["日経平均株価 終値"]

    entry = planner.pending_answers()[0]
    assert entry["id"] == pid and entry["state"] == "ready"

    topic = await planner.plan()
    assert topic.kind == "answer"
    assert topic.pending_id == pid
    assert "59,662.02" in topic.documents[0].snippet


@pytest.mark.asyncio
async def test_worker_retries_then_marks_failed():
    """検索が空振りし続けたら再試行を尽くして failed になること"""
    orch, _ = make_orchestrator(StubLLM(RAW), StubTTS())
    search = FakeSearch([])
    planner = _enable_knowledge(orch, search)
    planner.enqueue_answer("視聴者A", FACT_QUESTION, query="日経平均株価 終値")

    # 再試行間隔 [30, 120, 300] ＝ 3回まで defer、4回目の試行で failed
    for _ in range(3):
        assert await orch._research_one_pending() is True
        entry = planner.pending_answers()[0]
        assert entry["state"] == "waiting"
        planner._pending_answers[0]["next_try_at"] = 0.0   # 間隔経過を模す
    assert await orch._research_one_pending() is True
    assert planner.pending_answers()[0]["state"] == "failed"
    assert len(search.calls) == 4


# ── (5) 調査全滅時の締め発話は一度だけ ──────────────────────

@pytest.mark.asyncio
async def test_failed_answer_closing_spoken_once():
    planner = TopicPlanner(news=None, search=None, state_path="")
    pid = planner.enqueue_answer("視聴者A", "明日の天気は？")
    planner.fail_answer(pid)
    assert planner.has_deliverable_answer()

    topic = await planner.plan()
    assert topic.kind == "answer"
    assert topic.pending_id == pid
    assert topic.documents[0].source == "answer_failed"

    # 発話失敗（release）では消えない
    planner.release(topic)
    assert len(planner.pending_answers()) == 1
    topic2 = await planner.plan()
    assert topic2.kind == "answer" and topic2.pending_id == pid

    # 発話成功（commit）で消費され、締めは二度と出ない
    planner.commit(topic2)
    assert planner.pending_answers() == []
    topic3 = await planner.plan()
    assert topic3.kind != "answer"


# ── (6) 宣言だけの応答は保留キューへ遡及投入される ──────────

@pytest.mark.asyncio
async def test_unfulfilled_promise_retroactively_queued():
    llm = RecordingLLM(_json_raw("まかせて、すぐに調べてくるね！"))
    orch, hub = make_duo_p3(llm_a=llm)
    planner = _enable_knowledge(orch, FakeSearch([NIKKEI_DOC]))

    # 事実質問ではない通常コメントへの応答で「調べてくる」と言ってしまった
    await orch._process_comment(comment("こんにちは！今日も元気だね",
                                        author="視聴者A"))

    entries = planner.pending_answers()
    assert len(entries) == 1
    assert entries[0]["question"] == "こんにちは！今日も元気だね"
    # 遡及投入された質問は comment hook にも残らない（プランナ側の保険）
    assert planner._comment_hook is None


# ── (7) プロバイダ別バックオフ ──────────────────────────────

def _http_429(retry_after: int | None = None) -> httpx.HTTPStatusError:
    req = httpx.Request("GET", "https://news.google.com/rss/search")
    headers = {"Retry-After": str(retry_after)} if retry_after else {}
    resp = httpx.Response(429, request=req, headers=headers)
    return httpx.HTTPStatusError("429 Too Many Requests",
                                 request=req, response=resp)


@pytest.mark.asyncio
async def test_provider_cooldown_skips_and_resets(monkeypatch):
    monkeypatch.delenv("GOOGLE_SEARCH_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_CSE_ID", raising=False)
    monkeypatch.delenv("BRAVE_SEARCH_API_KEY", raising=False)
    client = WebSearchClient(provider="auto", min_request_interval_sec=0)
    calls = {"duckduckgo": 0, "google_news": 0}

    async def fake(name, query, limit):
        calls[name] += 1
        if name == "google_news":
            raise _http_429()
        return []                       # DDG は空振り（例外なし）

    monkeypatch.setattr(client, "_search_with", fake)

    assert await client.search("テスト") == []
    assert calls == {"duckduckgo": 1, "google_news": 1}
    assert client._in_cooldown("google_news")
    assert client._cooldown_remaining("google_news") >= 50   # 基準60秒＋ジッタ

    # 冷却中はリクエスト自体を送らない
    assert await client.search("別のクエリ") == []
    assert calls["google_news"] == 1
    assert calls["duckduckgo"] == 2

    # 成功でリセットされる
    client._note_provider_success("google_news")
    assert not client._in_cooldown("google_news")
    assert client._provider_fails.get("google_news", 0) == 0


def test_cooldown_respects_retry_after_and_ignores_other_errors():
    client = WebSearchClient(min_request_interval_sec=0,
                             cooldown_base_sec=60, cooldown_max_sec=900)
    client._note_provider_failure("google_news", _http_429(retry_after=300))
    assert client._cooldown_remaining("google_news") >= 300

    # 429 でない失敗では冷却しない
    client2 = WebSearchClient(min_request_interval_sec=0)
    client2._note_provider_failure("duckduckgo", RuntimeError("timeout"))
    assert not client2._in_cooldown("duckduckgo")


# ── (8) curiosity の種の衛生フィルタ ────────────────────────

def test_curiosity_seed_ng_filters_self_referential():
    assert not TopicPlanner._seed_ok("調査の品質向上とスピードの維持")
    assert not TopicPlanner._seed_ok("安定したパフォーマンスの維持")
    assert not TopicPlanner._seed_ok("情報収集の効率化")
    assert not TopicPlanner._seed_ok("Character1は元気な女の子です")   # 文の形
    assert TopicPlanner._seed_ok("三条まつり")
    assert TopicPlanner._seed_ok("新型グラフィックボード")


def test_build_query_excludes_ng_seeds_from_all_sources():
    planner = TopicPlanner(
        news=None, search=None, state_path="",
        keyword_provider=lambda: ["情報収集の効率化"],
        theme_provider=lambda: "",
        intention_provider=lambda: [
            {"title": "調査の品質向上とスピードの維持", "next_action": "続ける"}],
    )
    assert planner._build_query() == ""


def test_build_query_uses_clean_intention_seed(monkeypatch):
    planner = TopicPlanner(
        news=None, search=None, state_path="",
        keyword_provider=lambda: [],
        theme_provider=lambda: "",
        intention_provider=lambda: [
            {"title": "三条まつりの準備", "next_action": "進める"}],
    )
    monkeypatch.setattr(random, "random", lambda: 0.0)   # 意図種を必ず選ぶ
    q = planner._build_query()
    assert "三条まつりの準備" in q


# ── ニュース取得のバックオフ（P5）──────────────────────────

@pytest.mark.asyncio
async def test_news_failure_backoff_and_recovery(monkeypatch):
    feed = NewsFeed(feeds=["https://example.com/rss"], retry_base_sec=120)

    async def boom(url):
        raise RuntimeError("down")

    monkeypatch.setattr(feed, "_fetch_feed", boom)
    await feed.refresh()
    assert feed.fail_streak == 1
    assert not feed.is_stale(900)          # 丸々15分沈黙せずバックオフ待ち

    feed._next_retry_at = time.time() - 1  # バックオフ経過を模す
    assert feed.is_stale(900)

    doc = WebDocument(title="記事", url="https://example.com/1")

    async def ok(url):
        return [doc]

    monkeypatch.setattr(feed, "_fetch_feed", ok)
    await feed.refresh()
    assert feed.fail_streak == 0
    assert feed.items and feed.items[0].title == "記事"
    assert not feed.is_stale(900)          # 成功直後は新鮮


# ── 回答プロンプト（P3）────────────────────────────────────

def test_answer_prompts():
    cfg = AppConfig()
    p = build_topic_prompt(cfg, [NIKKEI_DOC], kind="answer",
                           query="終値はいくら？", question_author="視聴者A")
    assert "調べが終わりました" in p
    assert "視聴者Aさん" in p

    failed_doc = WebDocument(title="調べた質問", snippet="終値はいくら？",
                             source="answer_failed")
    p2 = build_topic_prompt(cfg, [failed_doc], kind="answer",
                            query="終値はいくら？", question_author="視聴者A")
    assert "分かりませんでした" in p2

    p3 = build_answer_pending_prompt(cfg, question="終値はいくら？")
    assert "依頼済み" in p3
    assert "推測して言わない" in p3


# ── 優先順位：保留回答は話題の強制切替より優先（Phase 2 との取り決め）──

class FakePlannerP3:
    """保留回答を持つプランナのスタブ"""

    def __init__(self, deliverable: bool):
        self._deliverable = deliverable
        self.plan_calls: list[dict] = []
        self.reset_thread_calls = 0

    def has_deliverable_answer(self):
        return self._deliverable

    async def plan(self, kind="", query="", allow_web=True, context=None):
        self.plan_calls.append({"kind": kind, "query": query})
        if kind:
            return Topic(kind=kind, label="強制話題", documents=[NIKKEI_DOC],
                         topic_id="fake-forced")
        return Topic(kind="answer", label="質問への回答", query="終値はいくら？",
                     documents=[NIKKEI_DOC], topic_id="fake-answer",
                     pending_id="pa1", author="視聴者A")

    def commit(self, topic):
        pass

    def release(self, topic):
        pass

    def reset_thread(self):
        self.reset_thread_calls += 1

    def note_speech(self, idle=False, text=""):
        pass

    def ban_word(self, word, seconds=1800):
        pass

    def banned_words(self):
        return set()


@pytest.mark.asyncio
async def test_speak_idle_pending_answer_overrides_force_switch():
    orch, hub = make_duo_p3()
    planner = FakePlannerP3(deliverable=True)
    orch._planner = planner
    orch._force_switch_pending = True

    await orch._speak_idle()

    # 回答が差し込み可能な間は強制切替が効かず、消費もされない
    assert planner.plan_calls[0]["kind"] == ""
    assert planner.reset_thread_calls == 0
    assert orch._force_switch_pending is True
    assert _speakers(hub) == ["Character1"]
