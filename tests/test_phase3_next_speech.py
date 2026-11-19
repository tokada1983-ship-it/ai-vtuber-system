# -*- coding: utf-8 -*-
"""Phase 3（次発言プランニング）のユニットテスト。

対象:
- ルール版プランナー: 次発言の意図（introduce/deepen/ask/react/close/
  transition）の決定ルール（STUDY §6-(a1)）
- 話題の弧: 段階遷移（導入→展開→締め）と完遂判定（ADDENDUM）。
  弧の解放は「締めの発話の commit」か「上限（発話数・継続回数・時間）
  の超過」のときだけ
- 割り込み整合: コメント応答（寄り道）で弧がリセットされず、復帰が
  弧の続きとして数えられること。締め発話の失敗（release）では
  完遂しないこと
- 先行生成の前提照合（STUDY §6.4-3）: 計画後に別の発話が挟まったら
  先行生成を破棄すること
- プランナー無効化で Phase 2 までの動作（意図なし・従来の段階付け）
  に戻ること
"""
from __future__ import annotations

import asyncio
import time

from backend.config import AppConfig
from backend.knowledge.models import WebDocument
from backend.knowledge.topic_planner import Topic, TopicPlanner
from backend.llm.prompt_builder import build_topic_prompt
from backend.models import LLMResponse
from helpers import StubLLM, StubTTS, make_orchestrator


class _Feed:
    """mark_used で減っていくニュースフィードのスタブ"""

    def __init__(self, docs):
        self._docs = list(docs)

    def unused_items(self):
        return list(self._docs)

    def mark_used(self, doc):
        self._docs = [d for d in self._docs if d.key != doc.key]


def _quake_feed():
    return _Feed([
        WebDocument(title="大きな地震が発生しました",
                    url="https://example.com/1", snippet="本文1",
                    source="news"),
        WebDocument(title="大きな地震が発生しました続報",
                    url="https://example.com/2", snippet="本文2",
                    source="news"),
        WebDocument(title="大きな地震の被害状況と避難情報",
                    url="https://example.com/3", snippet="本文3",
                    source="news"),
    ])


def _planner(**kw):
    kw.setdefault("state_path", "")
    kw.setdefault("arc_ask_probability", 0.0)   # 決定的にする
    return TopicPlanner(**kw)


def _open_arc(p):
    """ニュース話題を1つ開いて commit し、弧を開始した状態にする"""
    t = p._plan_news()
    assert t is not None
    t.topic_id = "t-open"
    p.commit(t)
    return t


# ── 意図決定のルール分岐 ──────────────────────────────────────

class TestIntentDecision:
    def test_fresh_news_is_introduce(self):
        p = _planner(news=_quake_feed())
        t = p._plan_news()
        assert t.arc_stage == "introduce"
        assert t.intent == "introduce"
        assert t.arc_key == "大きな地震が発生しました"

    def test_first_continuation_deepens(self, monkeypatch):
        monkeypatch.setattr(
            "backend.knowledge.topic_planner.random.random", lambda: 0.0)
        p = _planner(news=_quake_feed())
        _open_arc(p)
        t = asyncio.run(p._try_continue(True))
        assert t is not None and t.resumed
        assert t.arc_stage == "deepen" and t.intent == "deepen"

    def test_continuation_at_cap_closes(self, monkeypatch):
        monkeypatch.setattr(
            "backend.knowledge.topic_planner.random.random", lambda: 0.0)
        p = _planner(news=_quake_feed())
        _open_arc(p)
        t1 = asyncio.run(p._try_continue(True))
        t1.topic_id = "t-c1"
        p.commit(t1)
        t2 = asyncio.run(p._try_continue(True))
        assert t2.arc_stage == "close" and t2.intent == "close"

    def test_speech_cap_forces_close(self, monkeypatch):
        monkeypatch.setattr(
            "backend.knowledge.topic_planner.random.random", lambda: 0.0)
        # 継続回数に余裕があっても、発話数上限に達する回は締めになる
        p = _planner(news=_quake_feed(), arc_max_speeches=2,
                     arc_max_continuations=5)
        _open_arc(p)
        t = asyncio.run(p._try_continue(True))
        assert t.arc_stage == "close" and t.intent == "close"

    def test_ask_intent_by_probability_and_energy(self, monkeypatch):
        monkeypatch.setattr(
            "backend.knowledge.topic_planner.random.random", lambda: 0.0)
        p = _planner(news=_quake_feed(), arc_ask_probability=1.0)
        p._energy = 0.9
        p._energy_at = time.time()
        _open_arc(p)
        t = asyncio.run(p._try_continue(True))
        assert t.arc_stage == "deepen" and t.intent == "ask"

    def test_ask_needs_energy(self, monkeypatch):
        monkeypatch.setattr(
            "backend.knowledge.topic_planner.random.random", lambda: 0.0)
        p = _planner(news=_quake_feed(), arc_ask_probability=1.0)
        p._energy = 0.0
        p._energy_at = time.time()
        _open_arc(p)
        t = asyncio.run(p._try_continue(True))
        assert t.intent == "deepen"

    def test_memory_topic_is_introduce(self):
        p = _planner(memory_provider=lambda: {
            "text": "昔の思い出の本文です。十分な長さがあります。"})
        t = asyncio.run(p.plan())
        assert t.kind == "memory" and t.intent == "introduce"

    def test_comment_hook_is_react(self, monkeypatch):
        monkeypatch.setattr(
            "backend.knowledge.topic_planner.random.random", lambda: 0.0)
        p = _planner()
        p._comment_hook = {"author": "視聴者A", "text": "その話もっと聞きたい",
                           "at": time.time() - 100}
        t = asyncio.run(p.plan())
        assert t.kind == "comment" and t.intent == "react"

    def test_pending_answer_is_react(self):
        p = _planner()
        pid = p.enqueue_answer("視聴者A", "昨日の天気はどうでしたか")
        assert pid
        doc = WebDocument(title="天気", snippet="晴れでした",
                          source="search")
        assert p.attach_answer_documents(pid, [doc])
        t = asyncio.run(p.plan())
        assert t.kind == "answer" and t.intent == "react"

    def test_transition_intent_after_completion(self):
        p = _planner(news=_quake_feed())
        p._last_arc_end_at = time.time()
        t = p._plan_news()
        assert t.intent == "transition"
        # 完遂から時間が経っていれば通常の導入に戻る
        p._last_arc_end_at = time.time() - 400
        t = p._plan_news()
        assert t.intent == "introduce"


# ── 弧の遷移と完遂判定 ────────────────────────────────────────

class TestArcCompletion:
    def test_commit_opens_arc(self):
        p = _planner(news=_quake_feed())
        _open_arc(p)
        arc = p.status()["arc"]
        assert arc == {"kind": "news", "stage": "introduce",
                       "speeches": 1, "closing": False}

    def test_arc_walks_to_close_and_completes(self, monkeypatch):
        monkeypatch.setattr(
            "backend.knowledge.topic_planner.random.random", lambda: 0.0)
        p = _planner(news=_quake_feed())
        _open_arc(p)
        t1 = asyncio.run(p._try_continue(True))
        assert t1.arc_stage == "deepen"
        t1.topic_id = "t-c1"
        p.commit(t1)
        assert p._thread is not None and p._thread["speeches"] == 2
        t2 = asyncio.run(p._try_continue(True))
        assert t2.arc_stage == "close"
        t2.topic_id = "t-c2"
        p.commit(t2)
        # 締めの発話が commit されて初めて弧が解放される
        assert p._thread is None
        assert p.status()["arc"] is None
        assert p.status()["arc_stats"]["completed"] == 1
        last = p._ledger[-1]
        assert last["arc_completed"] is True
        assert last["arc_speeches"] == 3
        assert last["intent"] == "close" and last["arc_stage"] == "close"

    def test_release_of_close_keeps_arc_open(self, monkeypatch):
        monkeypatch.setattr(
            "backend.knowledge.topic_planner.random.random", lambda: 0.0)
        p = _planner(news=_quake_feed())
        _open_arc(p)
        t1 = asyncio.run(p._try_continue(True))
        t1.topic_id = "t-c1"
        p.commit(t1)
        t2 = asyncio.run(p._try_continue(True))
        assert t2.arc_stage == "close"
        assert p._thread["closing"] is True
        # 締めの生成が失敗したら、完遂にはならず弧は開いたまま
        p.release(t2)
        assert p._thread is not None
        assert p._thread["closing"] is False
        assert p.status()["arc_stats"]["completed"] == 0

    def test_arc_expires_by_total_time_cap(self):
        now = time.time()
        p = _planner(news=_quake_feed(), arc_max_sec=900.0)
        p._thread = {"kind": "news", "title": "大きな地震が発生しました",
                     "query": "", "at": now, "continuations": 0,
                     "speeches": 1, "started_at": now - 1000,
                     "stage": "introduce", "arc_key": "k", "closing": False}
        t = asyncio.run(p._try_continue(True))
        assert t is None
        assert p._thread is None
        assert p.status()["arc_stats"]["expired"] == 1

    def test_arc_expires_by_inactivity(self):
        now = time.time()
        p = _planner(news=_quake_feed())
        p._thread = {"kind": "news", "title": "大きな地震が発生しました",
                     "query": "", "at": now - 700, "continuations": 0,
                     "speeches": 1, "started_at": now - 700,
                     "stage": "introduce", "arc_key": "k", "closing": False}
        t = asyncio.run(p._try_continue(True))
        assert t is None
        assert p._thread is None
        assert p.status()["arc_stats"]["expired"] == 1


# ── 割り込み後の復帰 ──────────────────────────────────────────

class TestInterruptResume:
    def test_comment_detour_preserves_and_resumes_arc(self, monkeypatch):
        monkeypatch.setattr(
            "backend.knowledge.topic_planner.random.random", lambda: 0.0)
        p = _planner(news=_quake_feed())
        _open_arc(p)
        started_at = p._thread["started_at"]
        p.note_speech(idle=True, text="大きな地震があったみたいだね")
        # コメント応答（寄り道）: 弧はスタックへ退避され、スレッドは残る
        p.note_speech(idle=False, text="コメントありがとう、様子を見ててね")
        assert len(p._stack) == 1
        assert p._stack[0]["arc_speeches"] == 1
        assert p._thread is not None
        # 復帰は弧の続きとして数えられ、直前の発話も渡る
        t = asyncio.run(p._try_resume(True))
        assert t is not None and t.resumed
        assert t.arc_stage == "deepen" and t.intent == "deepen"
        assert t.prev_speech == "コメントありがとう、様子を見ててね"
        t.topic_id = "t-resume"
        p.commit(t)
        assert p._thread["speeches"] == 2
        assert p._thread["continuations"] == 1
        assert p._thread["started_at"] == started_at
        # 残り1回の継続で締めに届き、完遂する
        t2 = asyncio.run(p._try_continue(True))
        assert t2.arc_stage == "close"
        t2.topic_id = "t-close"
        p.commit(t2)
        assert p._thread is None
        assert p.status()["arc_stats"]["completed"] == 1

    def test_resume_of_expired_arc_is_dropped(self):
        now = time.time()
        p = _planner(arc_max_sec=900.0)
        p._stack = [{"kind": "news", "title": "大きな地震が発生しました",
                     "query": "", "topic_id": "t-old", "at": now - 2000,
                     "suspended_at": now, "arc_speeches": 1,
                     "arc_started_at": now - 2000, "continuations": 0,
                     "arc_key": "k"}]
        t = asyncio.run(p._try_resume(True))
        assert t is None
        assert p._stack == []
        assert p.status()["arc_stats"]["expired"] == 1

    def test_speeches_do_not_reset_arc(self, monkeypatch):
        # 掛け合いで話者が替わっても（発話通知がどちらから来ても）、
        # 弧は話題単位で管理され、リセットされない
        monkeypatch.setattr(
            "backend.knowledge.topic_planner.random.random", lambda: 0.0)
        p = _planner(news=_quake_feed())
        _open_arc(p)
        arc_key = p._thread["arc_key"]
        p.note_speech(idle=True, text="相方の発話です")
        p.note_speech(idle=False, text="もう一方の応答です")
        assert p._thread is not None
        assert p._thread["arc_key"] == arc_key
        t = asyncio.run(p._try_continue(True))
        assert t is not None and t.arc_key == arc_key


# ── プランナー無効化（Phase 2 までの動作） ────────────────────

class TestPlannerDisabled:
    def test_no_intents_when_disabled(self, monkeypatch):
        monkeypatch.setattr(
            "backend.knowledge.topic_planner.random.random", lambda: 0.0)
        p = _planner(news=_quake_feed(), planner_enabled=False)
        t = _open_arc(p)
        assert t.intent == ""
        t1 = asyncio.run(p._try_continue(True))
        assert t1.arc_stage == "deepen" and t1.intent == ""
        t1.topic_id = "t-c1"
        p.commit(t1)
        t2 = asyncio.run(p._try_continue(True))
        # 段階付けは従来規則（継続上限で締め）のまま、意図は付かない
        assert t2.arc_stage == "close" and t2.intent == ""
        t2.topic_id = "t-c2"
        p.commit(t2)
        # 完遂判定は働かず、スレッドは従来どおり残る
        assert p._thread is not None
        assert p.status()["arc_stats"]["completed"] == 0


# ── プロンプトへの意図描画 ────────────────────────────────────

DOC = WebDocument(title="記事タイトル", url="https://example.com",
                  snippet="記事の本文です", source="news", site="Example")


class TestIntentPrompt:
    def test_ask_line_rendered(self):
        p = build_topic_prompt(AppConfig(), [DOC], kind="news",
                               resumed=True, arc_stage="deepen",
                               prev_speech="さっきの発話の実文です",
                               intent="ask")
        assert "問いかけてください" in p

    def test_transition_line_rendered(self):
        p = build_topic_prompt(AppConfig(), [DOC], kind="news",
                               intent="transition")
        assert "新しい話題です" in p

    def test_no_intent_no_extra_lines(self):
        p = build_topic_prompt(AppConfig(), [DOC], kind="news",
                               resumed=True, arc_stage="deepen",
                               prev_speech="さっきの発話の実文です")
        assert "問いかけてください" not in p
        assert "新しい話題です" not in p


# ── 先行生成の前提照合（Orchestrator） ────────────────────────

class TestPrefetchPremise:
    def _orch_with_prefetch(self):
        orch, hub = make_orchestrator(StubLLM("{}"), StubTTS())
        planner = TopicPlanner(state_path="")
        orch._planner = planner
        planner.note_speech(idle=True, text="前提になった発話です")
        orch._prefetched = {
            "at": time.time(),
            "agent_key": orch.lead.key,
            "topic": None,
            "response": LLMResponse(speech="先行生成した発話です"),
            "actions": [],
            "premise_speech": "前提になった発話です",
        }
        return orch, planner

    def test_prefetch_used_when_premise_unchanged(self):
        orch, planner = self._orch_with_prefetch()
        pf = orch._take_prefetched(orch.lead)
        assert pf is not None
        assert pf["response"].speech == "先行生成した発話です"

    def test_prefetch_discarded_when_speech_intervened(self):
        orch, planner = self._orch_with_prefetch()
        # 計画の後にコメント応答が挟まった = 前提の最終発話が変わった
        planner.note_speech(idle=False, text="割り込んだコメント応答です")
        pf = orch._take_prefetched(orch.lead)
        assert pf is None
        assert orch._prefetched is None
