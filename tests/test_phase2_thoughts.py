"""Phase 2（inner thoughts／システム発火／内心表示チャネル）のテスト

- thought キーの抽出（parser／streaming サルベージ）と LLMResponse への搭載
- 内心の検証・配送（ThoughtsDispatcher）とクールダウン
- TTS 分離：thought は合成に一切渡らないこと（発話経路の結合テスト）
- プロンプトへの内心節の注入
- システム発火エフェクト（EffectsDispatcher.dispatch_system）
"""
import asyncio
from types import SimpleNamespace

import pytest

from backend.config import AppConfig, EffectEntryConfig
from backend.effects import (EffectsDispatcher, ThoughtsDispatcher,
                             resolve_thoughts, validate_thought)
from backend.llm.parser import parse_llm_output
from backend.llm.prompt_builder import _RESPONSE_FORMAT, _thoughts_section
from backend.llm.streaming import SpeechStreamParser
from backend.models import LLMResponse
from helpers import StubLLM, StubTTS, make_orchestrator

RAW_WITH_THOUGHT = (
    '{"emotion":"happy","motion":"wave",'
    '"speech":"やっほー！今日も来てくれてありがとう。",'
    '"thought":"つい笑っちゃった"}'
)
THOUGHT_TEXT = "つい笑っちゃった"
SPEECH_TEXT = "やっほー！今日も来てくれてありがとう。"


# ── parser: thought 抽出 ──────────────────────────────────────

class TestParserThought:
    def test_thought_extracted(self):
        resp = parse_llm_output(RAW_WITH_THOUGHT)
        assert resp.skip is False
        assert resp.speech == SPEECH_TEXT
        assert resp.thought == THOUGHT_TEXT

    def test_thought_absent_is_empty(self):
        resp = parse_llm_output(
            '{"emotion":"happy","motion":"wave","speech":"こんにちは！"}')
        assert resp.thought == ""

    def test_thought_non_string_ignored(self):
        resp = parse_llm_output(
            '{"emotion":"happy","motion":"wave","speech":"こんにちは！",'
            '"thought":123}')
        assert resp.thought == ""
        resp = parse_llm_output(
            '{"emotion":"happy","motion":"wave","speech":"こんにちは！",'
            '"thought":{"text":"x"}}')
        assert resp.thought == ""

    def test_thought_salvaged_from_broken_json(self):
        # 全文パースが失敗する切断JSONでも、完結した thought は拾う
        raw = ('{"emotion":"happy","motion":"wave",'
               '"speech":"今日も元気にいくよ、みんな。",'
               '"thought":"ちょっと緊張するな","effect":{"type":"flip"')
        resp = parse_llm_output(raw)
        assert resp.skip is False
        assert resp.thought == "ちょっと緊張するな"


# ── streaming: extract_thought ────────────────────────────────

class TestStreamingThought:
    def test_extract_after_complete(self):
        parser = SpeechStreamParser()
        parser.feed(RAW_WITH_THOUGHT)
        assert parser.extract_thought() == THOUGHT_TEXT

    def test_extract_incomplete_is_empty(self):
        parser = SpeechStreamParser()
        parser.feed('{"emotion":"happy","speech":"途中だよ。","thought":"つい笑')
        assert parser.extract_thought() == ""

    def test_extract_absent_is_empty(self):
        parser = SpeechStreamParser()
        parser.feed('{"emotion":"happy","speech":"普通の発話だよ。"}')
        assert parser.extract_thought() == ""

    def test_build_response_carries_thought(self):
        parser = SpeechStreamParser()
        parser.feed(RAW_WITH_THOUGHT)
        parser.flush()
        resp = parser.build_response()
        assert resp.speech == SPEECH_TEXT
        assert resp.thought == THOUGHT_TEXT

    def test_thought_not_in_speech(self):
        # thought の文が speech 側に混ざらないこと
        parser = SpeechStreamParser()
        parser.feed(RAW_WITH_THOUGHT)
        parser.flush()
        resp = parser.build_response()
        assert THOUGHT_TEXT not in resp.speech


# ── 検証とキャラ別解決 ───────────────────────────────────────

class TestValidateThought:
    def setup_method(self):
        self.resolved = resolve_thoughts(AppConfig(), None)

    def test_plain_text_passes(self):
        assert validate_thought("  本音だよ  ", self.resolved) == "本音だよ"

    def test_truncated_at_max_chars(self):
        assert len(validate_thought("あ" * 40, self.resolved)) == 30

    def test_newlines_folded(self):
        assert validate_thought("一行目\n二行目", self.resolved) == "一行目 二行目"

    def test_empty_and_non_string_dropped(self):
        assert validate_thought("", self.resolved) == ""
        assert validate_thought("   ", self.resolved) == ""
        assert validate_thought(None, self.resolved) == ""
        assert validate_thought(123, self.resolved) == ""
        assert validate_thought({"text": "x"}, self.resolved) == ""

    def test_tag_fragment_dropped(self):
        assert validate_thought("<speech: 漏れた", self.resolved) == ""
        assert validate_thought("<think>考え中", self.resolved) == ""


class TestResolveThoughts:
    def test_character_override(self):
        cfg = AppConfig()
        char = cfg.characters[0]
        char.thoughts.max_chars = 12
        char.thoughts.cooldown_sec = 3.0
        resolved = resolve_thoughts(cfg, char)
        assert resolved.max_chars == 12
        assert resolved.cooldown_sec == 3.0
        assert resolved.enabled is True          # 未指定は全体設定を使う
        assert cfg.thoughts.max_chars == 30      # 全体設定は書き換わらない

    def test_no_character_returns_global(self):
        cfg = AppConfig()
        assert resolve_thoughts(cfg, None) is cfg.thoughts


# ── ThoughtsDispatcher ────────────────────────────────────────

class StubHub:
    def __init__(self):
        self.broadcasts: list[tuple[str, dict]] = []

    async def broadcast(self, channel, data):
        self.broadcasts.append((channel, data))


def make_thoughts(cfg=None):
    cfg = cfg or AppConfig()
    hub = StubHub()
    return ThoughtsDispatcher(cfg, hub), hub


class TestThoughtsDispatcher:
    async def test_broadcast_with_speaker(self):
        dispatcher, hub = make_thoughts()
        agent = SimpleNamespace(cfg=None, key="Character1", name="Character1")
        result = await dispatcher.dispatch(THOUGHT_TEXT, agent=agent,
                                           speech=SPEECH_TEXT)
        assert result == {"ok": True, "text": THOUGHT_TEXT}
        channel, msg = hub.broadcasts[0]
        assert channel == "avatar"
        assert msg == {"type": "thought", "text": THOUGHT_TEXT,
                       "character": "Character1", "speaker": "Character1"}

    async def test_cooldown_blocks_second(self):
        dispatcher, hub = make_thoughts()
        assert (await dispatcher.dispatch("一言め"))["ok"] is True
        second = await dispatcher.dispatch("二言め")
        assert second == {"ok": False, "reason": "cooldown"}
        assert len(hub.broadcasts) == 1

    async def test_disabled(self):
        cfg = AppConfig()
        cfg.thoughts.enabled = False
        dispatcher, hub = make_thoughts(cfg)
        result = await dispatcher.dispatch("本音")
        assert result == {"ok": False, "reason": "thoughts disabled"}
        assert hub.broadcasts == []

    async def test_invalid_dropped(self):
        dispatcher, hub = make_thoughts()
        result = await dispatcher.dispatch("   ")
        assert result["ok"] is False
        assert hub.broadcasts == []

    async def test_duplicate_of_speech_dropped(self):
        dispatcher, hub = make_thoughts()
        result = await dispatcher.dispatch(
            "今日も来てくれてありがとう", speech=SPEECH_TEXT)
        assert result == {"ok": False, "reason": "duplicates speech"}
        assert hub.broadcasts == []

    async def test_manual_bypasses_cooldown(self):
        dispatcher, hub = make_thoughts()
        assert (await dispatcher.dispatch("一言め"))["ok"] is True
        assert (await dispatcher.dispatch("二言め", manual=True))["ok"] is True
        assert len(hub.broadcasts) == 2


# ── プロンプト ────────────────────────────────────────────────

class TestThoughtsPrompt:
    def test_section_present_by_default(self):
        cfg = AppConfig()
        section = _thoughts_section(cfg, cfg.characters[0])
        assert "内心" in section
        assert "thought" in section
        assert "控えめ" in section

    def test_section_hidden_when_disabled(self):
        cfg = AppConfig()
        cfg.thoughts.enabled = False
        assert _thoughts_section(cfg, cfg.characters[0]) == ""

    def test_response_format_mentions_thought(self):
        assert "thought" in _RESPONSE_FORMAT


# ── Orchestrator 結合: TTS 分離 ───────────────────────────────

def thought_messages(hub) -> list[dict]:
    return [msg for kind, msg in hub.events
            if kind == "avatar" and isinstance(msg, dict)
            and msg.get("type") == "thought"]


class TestOrchestratorThoughts:
    async def test_speak_dispatches_thought_not_to_tts(self):
        tts = StubTTS()
        orch, hub = make_orchestrator(StubLLM("{}"), tts)
        response = LLMResponse(speech="お知らせだよ。", thought="どきどきする")
        await orch._speak(response)
        msgs = thought_messages(hub)
        assert len(msgs) == 1
        assert msgs[0]["text"] == "どきどきする"
        assert msgs[0]["character"] == orch.lead.key
        # TTS に渡るのは speech のみ
        assert tts.calls == ["お知らせだよ。"]

    async def test_speak_without_thought_sends_nothing(self):
        orch, hub = make_orchestrator(StubLLM("{}"), StubTTS())
        await orch._speak(LLMResponse(speech="普通の発話だよ。"))
        assert thought_messages(hub) == []

    async def test_streaming_dispatches_thought_not_to_tts(self):
        tts = StubTTS()
        orch, hub = make_orchestrator(StubLLM(RAW_WITH_THOUGHT), tts)
        response = await orch._generate_and_speak("sys")
        assert response is not None
        assert response.speech == SPEECH_TEXT
        assert response.thought == THOUGHT_TEXT
        msgs = thought_messages(hub)
        assert len(msgs) == 1
        assert msgs[0]["text"] == THOUGHT_TEXT
        # TTS に thought が混ざらないこと
        assert all(THOUGHT_TEXT not in call for call in tts.calls)
        assert tts.calls == ["やっほー！", "今日も来てくれてありがとう。"]

    async def test_streaming_thought_disabled_by_config(self):
        orch, hub = make_orchestrator(StubLLM(RAW_WITH_THOUGHT), StubTTS())
        orch._cfg.thoughts.enabled = False
        response = await orch._generate_and_speak("sys")
        assert response is not None
        assert thought_messages(hub) == []


# ── システム発火エフェクト ────────────────────────────────────

def make_effects_cfg(tmp_path=None) -> AppConfig:
    cfg = AppConfig()
    cfg.effects.se["sparkle"] = EffectEntryConfig(
        file="assets/effects/se/sparkle.wav", label="キラキラ")
    cfg.effects.flip["announcement"] = EffectEntryConfig(
        text="重大発表！", label="重大発表")
    cfg.effects.system.cutin_probability = 0.0   # SE/フリップのみで決定的に
    return cfg


class PlayRecorder:
    def __init__(self):
        self.played: list[bytes] = []

    async def __call__(self, wav: bytes):
        self.played.append(wav)


def make_effects_dispatcher(cfg=None):
    cfg = cfg or make_effects_cfg()
    hub = StubHub()
    return EffectsDispatcher(cfg, hub, PlayRecorder()), hub


class TestDispatchSystem:
    async def test_fires_from_catalog(self, tmp_path, monkeypatch):
        se = tmp_path / "assets" / "effects" / "se" / "sparkle.wav"
        se.parent.mkdir(parents=True)
        se.write_bytes(b"RIFF-fake")
        monkeypatch.chdir(tmp_path)
        dispatcher, hub = make_effects_dispatcher()
        result = await dispatcher.dispatch_system(reason="no_material")
        assert result["ok"] is True
        assert len(hub.broadcasts) == 1
        msg = hub.broadcasts[0][1]
        assert msg["type"] == "effect"
        assert msg["effect_type"] in ("se", "flip")

    async def test_system_cooldown_blocks_second(self):
        cfg = make_effects_cfg()
        cfg.effects.cooldown_sec = 0.0       # 共有クールダウンは無効化し
        cfg.effects.system.cooldown_sec = 300.0
        dispatcher, hub = make_effects_dispatcher(cfg)
        assert (await dispatcher.dispatch_system())["ok"] is True
        second = await dispatcher.dispatch_system()
        assert second == {"ok": False, "reason": "system cooldown"}
        assert len(hub.broadcasts) == 1

    async def test_shared_cooldown_with_spontaneous(self):
        # 自発発火の直後は、システム発火も全体クールダウンで落ちる
        dispatcher, hub = make_effects_dispatcher()
        raw = {"type": "flip", "text": "自発の一言"}
        assert (await dispatcher.dispatch(raw))["ok"] is True
        result = await dispatcher.dispatch_system()
        assert result == {"ok": False, "reason": "cooldown"}
        assert len(hub.broadcasts) == 1

    async def test_disabled(self):
        cfg = make_effects_cfg()
        cfg.effects.system.enabled = False
        dispatcher, hub = make_effects_dispatcher(cfg)
        result = await dispatcher.dispatch_system()
        assert result == {"ok": False, "reason": "system effects disabled"}
        assert hub.broadcasts == []

    async def test_no_catalog_available(self):
        cfg = AppConfig()
        dispatcher, hub = make_effects_dispatcher(cfg)
        result = await dispatcher.dispatch_system()
        assert result == {"ok": False, "reason": "no system effect available"}
        assert hub.broadcasts == []

    async def test_system_fire_sets_shared_clock(self):
        # システム発火も全体クールダウンの時計を進める（自発の連発防止）
        dispatcher, hub = make_effects_dispatcher()
        assert (await dispatcher.dispatch_system())["ok"] is True
        result = await dispatcher.dispatch({"type": "flip", "text": "自発"})
        assert result == {"ok": False, "reason": "cooldown"}
