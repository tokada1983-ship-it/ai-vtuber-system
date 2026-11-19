# -*- coding: utf-8 -*-
"""Phase 1（組み込み検討書）のユニットテスト。

対象:
- プランナー穴埋め: memory占有率ハードキャップ（STUDY §3-C1）、
  フォールバック終端の多様化＝定番小噺（§3-C2）、話題化した記憶の
  L3常駐注入クールダウン（§3-C3 の最小形）
- 消費側の堅牢化（§1-C）: 切断JSONのサルベージ、no_json の
  致命/非致命の分離計測、peek 済み emotion/motion の採用
- 継続フレーミング（§6）: resumed/prev_speech/arc_stage が
  生成プロンプトに届くこと、話題の弧の状態が Topic に付くこと
"""
from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

from backend.config import AppConfig
from backend.ensemble.agent import CharacterAgent
from backend.knowledge.topic_planner import Topic, TopicPlanner
from backend.llm.parser import (parse_llm_output, parse_outcome_stats,
                                salvage_broken_json)
from backend.llm.prompt_builder import build_topic_prompt
from backend.llm.streaming import SpeechStreamParser
from backend.knowledge.models import WebDocument
from backend.models import Emotion, Motion


def _planner_with_ledger(kinds, **kw):
    kw.setdefault("state_path", "")
    p = TopicPlanner(**kw)
    for k in kinds:
        p._ledger.append({"topic_id": "", "kind": k, "label": "",
                          "query": "", "title": "", "at": 0.0})
    return p


# ── memory占有率ハードキャップ

class _Feed:
    def __init__(self, docs):
        self._docs = docs

    def unused_items(self):
        return list(self._docs)

class TestMemoryShareCap:
    def test_cap_blocks_memory_when_dominant(self):
        # 直近20件中15件が memory（占有率0.75 ≥ キャップ0.45）。
        # 他候補（news）のスコアが立つようフィードも与える
        doc = WebDocument(title="新しい記事", url="https://example.com",
                          snippet="本文", source="news")
        p = _planner_with_ledger(["memory"] * 15 + ["news"] * 5,
                                 news=_Feed([doc]))
        picked = p._pick_kind(allow_web=True)
        assert picked != "memory"
        assert p._last_utilities["memory"] == 0.0

    def test_cap_not_applied_with_few_samples(self):
        # 台帳が少ないうちは誤爆を避けてキャップを効かせない
        p = _planner_with_ledger(["memory"] * 5)
        p._pick_kind(allow_web=True)
        assert p._last_utilities["memory"] > 0

    def test_cap_not_applied_without_alternative(self):
        # Web系が使えない（他候補のスコアが0）ときは memory を残す
        p = _planner_with_ledger(["memory"] * 15 + ["news"] * 5)
        p._pick_kind(allow_web=False)
        assert p._last_utilities["memory"] > 0

    def test_memory_share_helper(self):
        p = _planner_with_ledger(["memory", "news", "memory", "curiosity"])
        assert p.memory_share() == 0.5
        assert p.status()["memory_share"] == 0.5


# ── フォールバック終端の多様化（定番小噺） ────────────────────

class TestFillerFallback:
    def test_filler_when_all_material_exhausted(self):
        p = TopicPlanner(memory_provider=lambda: None,
                         filler_enabled=True, state_path="")
        topic = asyncio.run(p.plan())
        assert topic.kind == "filler"
        assert topic.documents and topic.documents[0].title

    def test_no_filler_when_disabled(self):
        # 既定は無効：従来どおり材料なしの memory 話題に落ちる
        p = TopicPlanner(memory_provider=lambda: None, state_path="")
        topic = asyncio.run(p.plan())
        assert topic.kind == "memory"
        assert not topic.documents

    def test_filler_rotates(self):
        p = TopicPlanner(memory_provider=lambda: None,
                         filler_enabled=True, state_path="")
        t1 = asyncio.run(p.plan())
        p.commit(t1)
        t2 = asyncio.run(p.plan())
        assert t2.kind == "filler"
        assert t2.documents[0].title != t1.documents[0].title


# ── 話題化した記憶のクールダウン（L3注入との共有） ────────────

class TestTopicMemoryCooldown:
    def test_committed_memory_is_recent(self):
        body = "お菓子作りの思い出の本文です"
        p = TopicPlanner(memory_provider=lambda: {"text": body},
                         state_path="")
        topic = asyncio.run(p.plan())
        assert topic.kind == "memory"
        assert not p.is_topic_memory_recent(body)
        p.commit(topic)
        assert p.is_topic_memory_recent(body)
        assert not p.is_topic_memory_recent("全然関係ない別の記憶の本文です")


class _Entry:
    def __init__(self, text):
        self.text = text


class _Store:
    def __init__(self, texts):
        self._texts = texts

    def sample_recent(self, limit=5):
        return [_Entry(t) for t in self._texts[:limit]]


def _bare_agent(store, planner):
    agent = CharacterAgent.__new__(CharacterAgent)
    agent.long_term_memory = None
    agent.memory_store = store
    orch = SimpleNamespace(idle_episodic_entries=2,
                           idle_core_memory_chars=4000, _planner=planner)
    agent.app = SimpleNamespace(orchestrator=orch)
    return agent


class TestIdleMemoryCooldown:
    TEXTS = ["お菓子作りの思い出の本文です", "別の記憶Aです",
             "別の記憶Bです", "別の記憶Cです"]

    def _planner_with_topic_memory(self):
        p = TopicPlanner(state_path="")
        topic = Topic(kind="memory", label="記憶からの話題", documents=[
            WebDocument(title="タイトル", snippet=self.TEXTS[0],
                        source="memory")])
        p.commit(topic)
        return p

    def test_topic_memory_excluded_from_idle_injection(self):
        agent = _bare_agent(_Store(self.TEXTS),
                            self._planner_with_topic_memory())
        text = agent.build_idle_memory()
        assert "お菓子作りの思い出" not in text
        # 除外で空いた枠は別の記憶で埋まる
        assert "別の記憶Aです" in text and "別の記憶Bです" in text

    def test_no_planner_keeps_recent(self):
        agent = _bare_agent(_Store(self.TEXTS), None)
        text = agent.build_idle_memory()
        assert "お菓子作りの思い出" in text


# ── 切断JSONのサルベージと計測 ────────────────────────────────

TRUNCATED_OK = ('{"emotion":"happy","motion":"wave",'
                '"speech":"今日はいい天気だね！散歩に行こう')
TRUNCATED_FRAGMENT = '{"emotion":"neutral","motion":"idle","speech":"あらら'


class TestSalvage:
    def test_salvages_complete_sentence_from_truncated_json(self):
        resp = parse_llm_output(TRUNCATED_OK)
        assert not resp.skip
        assert resp.speech == "今日はいい天気だね！散歩に行こう"
        # emotion/motion は壊れたJSONからでも peek して採用する
        assert resp.emotion == Emotion.HAPPY
        assert resp.motion == Motion.WAVE
        assert resp.raw.get("salvaged")

    def test_fragment_is_never_salvaged(self):
        # 06:41:08 の回帰：未完の断片は発話として確定していない
        assert salvage_broken_json(TRUNCATED_FRAGMENT) is None
        resp = parse_llm_output(TRUNCATED_FRAGMENT)
        assert resp.skip and resp.speech == ""

    def test_explicit_skip_is_honored(self):
        resp = parse_llm_output(
            '{"skip": true, "speech": "これは十分に長い文章ですよ、話します')
        assert resp.skip and resp.speech == ""

    def test_meta_leak_is_not_salvaged(self):
        # 推論マーカーを含む発話は、長さがあっても救済しない
        resp = parse_llm_output(
            '{"speech":"検討：これは十分に長い文章のはずです、続けます')
        assert resp.skip and resp.speech == ""

    def test_outcome_counters_separate_fatal_and_salvaged(self):
        before = parse_outcome_stats()
        parse_llm_output(TRUNCATED_OK)
        parse_llm_output(TRUNCATED_FRAGMENT)
        parse_llm_output('{"emotion":"neutral","motion":"idle",'
                         '"speech":"普通の文章です。","target_comment_id": null}')
        after = parse_outcome_stats()
        assert after["no_json_salvaged"] == before["no_json_salvaged"] + 1
        assert after["no_json_fatal"] == before["no_json_fatal"] + 1
        assert after["ok"] == before["ok"] + 1


class TestStreamingSalvage:
    def test_stream_keeps_speech_and_peeked_enums(self):
        asm = SpeechStreamParser()
        asm.feed(TRUNCATED_OK)
        resp = asm.build_response()
        assert resp.speech == "今日はいい天気だね！散歩に行こう"
        assert resp.emotion == Emotion.HAPPY
        assert resp.motion == Motion.WAVE

    def test_stream_salvage_adopts_peek_when_parser_cannot(self):
        # パーサ単体では救えない短い発話でも、ストリームで確定した文は
        # peek 済みの emotion/motion で発話として成立させる
        asm = SpeechStreamParser()
        asm.feed('{"emotion":"happy","motion":"wave","speech":"うん、そうだね')
        resp = asm.build_response()
        assert resp.speech == "うん、そうだね"
        assert resp.emotion == Emotion.HAPPY
        assert resp.motion == Motion.WAVE
        assert resp.raw.get("stream_salvaged")


# ── 継続フレーミング ──────────────────────────────────────────

DOC = WebDocument(title="記事タイトル", url="https://example.com",
                  snippet="記事の本文です", source="news", site="Example")


class TestContinuationFraming:
    def test_resumed_prompt_quotes_prev_speech(self):
        p = build_topic_prompt(AppConfig(), [DOC], kind="news",
                               resumed=True, arc_stage="deepen",
                               prev_speech="さっきの発話の実文です")
        assert "さっきの発話の実文です" in p
        assert "続きです" in p

    def test_close_stage_prompt_asks_to_wrap_up(self):
        p = build_topic_prompt(AppConfig(), [DOC], kind="news",
                               resumed=True, arc_stage="close")
        assert "締めくく" in p

    def test_fresh_prompt_has_no_continuation_framing(self):
        p = build_topic_prompt(AppConfig(), [DOC], kind="news")
        assert "さっき自分が話した話題の続き" not in p
        assert "締めくく" not in p

    def test_filler_prompt_is_not_a_web_document(self):
        doc = WebDocument(title="季節の変わり目", snippet="季節の話",
                          source="filler", site="定番の小噺")
        p = build_topic_prompt(AppConfig(), [doc], kind="filler")
        assert "<web_result>" not in p
        assert "季節の変わり目" in p



class TestArcState:
    def test_fresh_news_topic_starts_arc(self):
        doc = WebDocument(title="新しい記事", url="https://example.com",
                          snippet="本文", source="news")
        p = TopicPlanner(news=_Feed([doc]), state_path="")
        topic = p._plan_news()
        assert topic is not None and topic.arc_stage == "introduce"

    def test_continue_marks_close_at_limit(self, monkeypatch):
        # 継続は確率60%で発火する仕様なので、乱数を固定して検証する
        monkeypatch.setattr(
            "backend.knowledge.topic_planner.random.random", lambda: 0.0)
        follow = WebDocument(title="大きな地震が発生しました続報",
                             url="https://example.com", snippet="続報の本文",
                             source="news")
        p = TopicPlanner(news=_Feed([follow]), state_path="")
        p.note_speech(text="最初の発話の実文です")
        p._thread = {"kind": "news", "title": "大きな地震が発生しました",
                     "query": "", "docs": [], "used": [],
                     "continuations": 1, "at": time.time()}
        topic = asyncio.run(p._try_continue(allow_web=True))
        assert topic is not None and topic.resumed
        assert topic.arc_stage == "close"
        assert topic.prev_speech == "最初の発話の実文です"
