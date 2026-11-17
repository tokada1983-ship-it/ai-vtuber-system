"""Phase 3.5 の回帰テスト: 検索クエリ整形と資料の十分性判定

2026-11-17 のローカル検証で残った未達点への対策:
  - 検索クエリが質問文の丸投げで、終値そのものを含む資料を引けなかった
  - 資料は返ったが答えの数値が無いケースで保留キューに入らず、
    調査ワーカーの再試行が起きなかった

外部サービスには依存せず、スタブで検証する（実リクエストは行わない）。
"""
from __future__ import annotations

import json
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import yaml

from backend.config import AppConfig, CharacterConfig, EnsembleConfig
from backend.knowledge.models import WebDocument
from backend.knowledge.topic_planner import TopicPlanner
from backend.orchestrator import Orchestrator
from helpers import (RAW, FakeHub, NoAudio, StubLLM, StubTTS, comment,
                     make_orchestrator)

JST = timezone(timedelta(hours=9))
# 2026-11-17 00:30 JST（UTC では 2026-11-16 15:30。JST 基準の確認用）
NOW_JST = datetime(2026, 11, 17, 0, 30, tzinfo=JST).timestamp()

FACT_QUESTION = "昨日の日経平均株価の終値って、いくらだったか分かりますか？"
NIKKEI_DOC = WebDocument(
    title="日経平均終値", url="https://example.com/nikkei",
    snippet="終値は59,662.02円でした", source="google_news", site="テスト")
NUMBERLESS_DOC = WebDocument(
    title="日経平均、続伸", url="https://example.com/market",
    snippet="午前の取引で値を上げ、市場は堅調に推移しました",
    source="google_news", site="テスト")


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


class SeqSearch:
    """呼ぶたびに用意した資料リストを順に返す検索スタブ"""

    last_error = ""

    def __init__(self, seq: list[list[WebDocument]]):
        self._seq = list(seq)
        self.calls: list[str] = []

    async def search(self, query, limit=None, use_cache=True):
        self.calls.append(query)
        if self._seq:
            return list(self._seq.pop(0))
        return []


def make_duo_35(llm_a=None, llm_b=None):
    """掛け合いなし（確率0）の2体オーケストレータを作る"""
    tmp = Path(tempfile.mkdtemp(prefix="aivt_p35_"))
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


# ── (1) 整形クエリに期待要素が含まれる ──────────────────────

def test_shaped_query_contains_key_elements():
    orch, _ = make_orchestrator(StubLLM(RAW), StubTTS())
    queries = orch._build_search_queries(FACT_QUESTION, now=NOW_JST)
    assert 1 <= len(queries) <= 3
    first = queries[0]
    assert "日経平均" in first
    assert "終値" in first
    assert "11月16日" in first            # 「昨日」を JST で解決した日付
    assert first != FACT_QUESTION          # 質問文の丸投げではない
    # 本命クエリ（現時刻基準）もカテゴリ要素を含む
    assert "日経平均" in orch._build_search_query(FACT_QUESTION)
    assert "終値" in orch._build_search_query(FACT_QUESTION)


# ── (2) 相対日付の解決は JST 基準 ───────────────────────────

def test_relative_date_resolution_uses_jst():
    orch, _ = make_orchestrator(StubLLM(RAW), StubTTS())
    # NOW_JST は 2026-11-17 00:30 JST（UTC では前日 15:30）。
    # UTC 基準で解決すると「昨日」が 11月15日 になってしまうため、
    # この期待値が成り立つこと自体が JST 基準の証拠になる
    assert orch._resolve_date_tokens("昨日の天気", now=NOW_JST) == ["11月16日"]
    assert orch._resolve_date_tokens("今日の天気", now=NOW_JST) == ["11月17日"]
    assert orch._resolve_date_tokens("一昨日の株価", now=NOW_JST) == ["11月15日"]
    assert orch._resolve_date_tokens("明日の天気", now=NOW_JST) == ["11月18日"]
    assert orch._resolve_date_tokens("明後日の天気", now=NOW_JST) == ["11月19日"]
    # 明示日付はそのまま拾う
    assert "11月3日" in orch._resolve_date_tokens("11月3日の終値", now=NOW_JST)


def test_extract_category_words_dedupes_compounds():
    orch, _ = make_orchestrator(StubLLM(RAW), StubTTS())
    cats = orch._extract_category_words(FACT_QUESTION)
    assert cats[0] == "日経平均"
    assert "終値" in cats
    # 「発売日」の中に内包される「発売」は残らない
    assert orch._extract_category_words("発売日はいつ？") == ["発売日"]


# ── 十分性判定の単体 ────────────────────────────────────────

def test_numeric_token_sufficiency():
    assert Orchestrator._docs_have_numeric_token([NIKKEI_DOC])
    assert not Orchestrator._docs_have_numeric_token([NUMBERLESS_DOC])
    # 「3時間前」のような経過表現だけでは答えらしい数値とはみなさない
    stale = WebDocument(title="市場の声", url="https://example.com/a",
                        snippet="3時間前に更新された記事です",
                        source="google_news", site="テスト")
    assert not Orchestrator._docs_have_numeric_token([stale])

    orch, _ = make_orchestrator(StubLLM(RAW), StubTTS())
    assert orch._expects_numeric_answer(FACT_QUESTION)
    assert not orch._expects_numeric_answer("明日の天気はどう？")


# ── (3) 同期資料に数値が無い → 保留キューへ ─────────────────

@pytest.mark.asyncio
async def test_sync_docs_without_number_are_queued():
    llm = RecordingLLM(_json_raw("わかった、調べるね。あとで教えるね。"))
    orch, hub = make_duo_35(llm_a=llm)
    search = FakeSearch([NUMBERLESS_DOC])
    planner = _enable_knowledge(orch, search)

    await orch._process_comment(comment(FACT_QUESTION, author="視聴者A"))

    entries = planner.pending_answers()
    assert len(entries) == 1
    assert entries[0]["state"] == "waiting"
    assert entries[0]["question"] == FACT_QUESTION
    assert entries[0]["query"] == orch._build_search_query(FACT_QUESTION)
    # 同期検索は整形クエリ（本命）で走った
    assert search.calls == [orch._build_search_query(FACT_QUESTION)]
    # 即応は数値の無い資料では答えず「検索を依頼済み」プロンプトになった
    # （判別は見出しで行う。「依頼済み」という語自体は共通ルール文にも
    # 含まれるため、Phase 3 の緩い判別はここでは使わない）
    assert any("調べもの（検索を依頼済み）" in p for p in llm.prompts)
    assert not any("調べもの（資料あり）" in p for p in llm.prompts)
    assert _speakers(hub) == ["Character1"]


# ── (4) 同期資料に数値がある → そのまま同期回答 ─────────────

@pytest.mark.asyncio
async def test_sync_docs_with_number_answer_directly():
    llm = RecordingLLM(_json_raw("昨日の終値は59,662.02円だったよ。"))
    orch, hub = make_duo_35(llm_a=llm)
    search = FakeSearch([NIKKEI_DOC])
    planner = _enable_knowledge(orch, search)

    await orch._process_comment(comment(FACT_QUESTION, author="視聴者A"))

    assert planner.pending_answers() == []          # キューには入らない
    assert search.calls == [orch._build_search_query(FACT_QUESTION)]
    # 資料回答プロンプト（資料本文つき）で生成され、保留プロンプトではない
    assert any("調べもの（資料あり）" in p for p in llm.prompts)
    assert any("59,662.02" in p for p in llm.prompts)
    assert not any("調べもの（検索を依頼済み）" in p for p in llm.prompts)
    assert _speakers(hub) == ["Character1"]


# ── (5) ワーカー再試行でクエリバリエーションが使われる ──────

@pytest.mark.asyncio
async def test_worker_cycles_query_variants():
    orch, _ = make_orchestrator(StubLLM(RAW), StubTTS())
    search = FakeSearch([])
    planner = _enable_knowledge(orch, search)
    planner.enqueue_answer("視聴者A", FACT_QUESTION,
                           query=orch._build_search_query(FACT_QUESTION))

    # 再試行間隔 [30, 120, 300] ＝ 3回まで defer、4回目の試行で failed
    for _ in range(3):
        assert await orch._research_one_pending() is True
        planner._pending_answers[0]["next_try_at"] = 0.0   # 間隔経過を模す
    assert await orch._research_one_pending() is True

    assert planner.pending_answers()[0]["state"] == "failed"
    assert len(search.calls) == 4
    assert search.calls[0] == orch._build_search_query(FACT_QUESTION)
    # 最初の3回はすべて別のバリエーションで検索している
    assert len(set(search.calls[:3])) == 3
    assert "日経平均" in search.calls[1]


@pytest.mark.asyncio
async def test_worker_retries_when_docs_lack_number():
    orch, _ = make_orchestrator(StubLLM(RAW), StubTTS())
    search = SeqSearch([[NUMBERLESS_DOC], [NIKKEI_DOC]])
    planner = _enable_knowledge(orch, search)
    planner.enqueue_answer("視聴者A", FACT_QUESTION,
                           query=orch._build_search_query(FACT_QUESTION))

    # 1回目: 資料はあるが数値が無い → 不十分として ready にならない
    assert await orch._research_one_pending() is True
    assert planner.pending_answers()[0]["state"] == "waiting"

    # 2回目（次のバリエーション）: 数値入りの資料 → ready
    planner._pending_answers[0]["next_try_at"] = 0.0
    assert await orch._research_one_pending() is True
    entry = planner.pending_answers()[0]
    assert entry["state"] == "ready"
    assert "59,662.02" in entry["documents"][0].snippet
    assert search.calls[0] != search.calls[1]
