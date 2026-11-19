"""演出エフェクト（カットイン動画・効果音・フリップ表示）のテスト

検証・カタログ解決・クールダウン・プロンプト注入・ストリーミング抽出と、
Orchestrator の発話経路への結合を、外部サービスなしで検証する。
"""
import asyncio
from types import SimpleNamespace

import pytest

from backend.config import AppConfig, EffectEntryConfig
from backend.effects import (EffectsDispatcher, file_to_url, resolve_effects,
                             validate_effect)
from backend.llm.parser import parse_llm_output
from backend.llm.prompt_builder import _effects_section
from backend.llm.streaming import SpeechStreamParser
from backend.models import LLMResponse
from helpers import StubLLM, StubTTS, make_orchestrator

RAW_WITH_EFFECT = (
    '{"emotion":"happy","motion":"wave",'
    '"speech":"やっほー！今日も来てくれてありがとう。",'
    '"effect":{"type":"flip","text":"重大発表！"}}'
)


def make_cfg() -> AppConfig:
    cfg = AppConfig()
    cfg.effects.cutin["duo"] = EffectEntryConfig(
        file="models/effects/cutin/duo.mp4", label="二人のカットイン")
    cfg.effects.se["sparkle"] = EffectEntryConfig(
        file="assets/effects/se/sparkle.wav", label="キラキラ")
    cfg.effects.flip["announcement"] = EffectEntryConfig(
        text="重大発表！", label="重大発表")
    return cfg


# ── 検証 ──────────────────────────────────────────────────────

class TestValidateEffect:
    def setup_method(self):
        self.resolved = resolve_effects(make_cfg(), None)

    def test_se_by_name(self):
        spec = validate_effect({"type": "se", "name": "sparkle"}, self.resolved)
        assert spec is not None
        assert spec.file == "assets/effects/se/sparkle.wav"
        assert spec.label == "キラキラ"

    def test_cutin_url_resolved(self):
        spec = validate_effect({"type": "cutin", "name": "duo"}, self.resolved)
        assert spec is not None
        assert spec.url == "/models/effects/cutin/duo.mp4"

    def test_flip_by_name_uses_catalog_text(self):
        spec = validate_effect({"type": "flip", "name": "announcement"},
                               self.resolved)
        assert spec is not None
        assert spec.text == "重大発表！"

    def test_flip_free_text_truncated(self):
        spec = validate_effect({"type": "flip", "text": "あ" * 40}, self.resolved)
        assert spec is not None
        assert len(spec.text) == 30

    def test_unknown_name_dropped(self):
        assert validate_effect({"type": "se", "name": "nope"},
                               self.resolved) is None

    def test_unknown_type_dropped(self):
        assert validate_effect({"type": "bogus", "name": "sparkle"},
                               self.resolved) is None

    def test_non_dict_dropped(self):
        assert validate_effect("sparkle", self.resolved) is None
        assert validate_effect(None, self.resolved) is None

    def test_flip_without_text_or_name_dropped(self):
        assert validate_effect({"type": "flip"}, self.resolved) is None
        assert validate_effect({"type": "flip", "text": "  "},
                               self.resolved) is None

    def test_type_and_name_normalized(self):
        spec = validate_effect({"type": " SE ", "name": " sparkle "},
                               self.resolved)
        assert spec is not None and spec.type == "se"


# ── カタログ解決（キャラ別上書き） ────────────────────────────

class TestResolveEffects:
    def test_character_override_merges(self):
        cfg = make_cfg()
        char = cfg.characters[0]
        char.effects.cutin["intro"] = EffectEntryConfig(
            file="models/effects/cutin/intro.mp4", label="登場")
        resolved = resolve_effects(cfg, char)
        assert set(resolved.cutin) == {"duo", "intro"}
        assert set(resolved.se) == {"sparkle"}        # 全体分も残る
        # 全体設定は書き換わらない
        assert "intro" not in cfg.effects.cutin

    def test_character_cooldown_override(self):
        cfg = make_cfg()
        char = cfg.characters[0]
        char.effects.cooldown_sec = 5.0
        assert resolve_effects(cfg, char).cooldown_sec == 5.0
        assert resolve_effects(cfg, None).cooldown_sec == 90.0

    def test_no_character_returns_global(self):
        cfg = make_cfg()
        assert resolve_effects(cfg, None) is cfg.effects


class TestFileToUrl:
    def test_models_path(self):
        assert file_to_url("models/effects/cutin/duo.mp4") == \
            "/models/effects/cutin/duo.mp4"

    def test_windows_separator(self):
        assert file_to_url("models\\effects\\cutin\\duo.mp4") == \
            "/models/effects/cutin/duo.mp4"

    def test_non_models_path_not_served(self):
        assert file_to_url("assets/effects/se/sparkle.wav") == ""
        assert file_to_url("") == ""


# ── パーサ（非ストリーミング） ────────────────────────────────

class TestParserEffect:
    def test_effect_dict_kept(self):
        r = parse_llm_output(RAW_WITH_EFFECT)
        assert r.effect == {"type": "flip", "text": "重大発表！"}

    def test_effect_non_dict_dropped(self):
        raw = ('{"emotion":"happy","motion":"wave","speech":"やっほー！",'
               '"effect":"sparkle"}')
        r = parse_llm_output(raw)
        assert r.effect is None
        assert r.speech == "やっほー！"

    def test_no_effect_is_none(self):
        r = parse_llm_output(
            '{"emotion":"happy","motion":"wave","speech":"やっほー！"}')
        assert r.effect is None


# ── ストリーミング抽出（サルベージ） ──────────────────────────

def feed_all(parser: SpeechStreamParser, raw: str, chunk: int = 5):
    for i in range(0, len(raw), chunk):
        parser.feed(raw[i:i + chunk])
    parser.flush()
    return parser


class TestStreamingExtractEffect:
    def test_complete_effect(self):
        p = feed_all(SpeechStreamParser(), RAW_WITH_EFFECT)
        assert p.extract_effect() == {"type": "flip", "text": "重大発表！"}
        assert p.build_response().effect == {"type": "flip", "text": "重大発表！"}

    def test_incomplete_effect_is_none(self):
        p = feed_all(SpeechStreamParser(),
                     '{"emotion":"happy","speech":"途中","effect":{"type":"se"')
        assert p.extract_effect() is None

    def test_salvage_when_whole_json_broken(self):
        """JSON全体が壊れていても、完結した effect は抽出できること。

        ストリーミング経路の effect 配送が「発話JSON全体のパース成功」に
        依存しないための性質（GitHub調査の設計インプットを受けた担保）。
        """
        import json
        raw = RAW_WITH_EFFECT + "xyz{"
        with pytest.raises(json.JSONDecodeError):
            json.loads(raw)
        p = feed_all(SpeechStreamParser(), raw)
        assert p.extract_effect() == {"type": "flip", "text": "重大発表！"}

    def test_braces_inside_string(self):
        p = feed_all(SpeechStreamParser(),
                     '{"speech":"あ","effect":{"type":"flip","text":"{すごい}！"}}')
        assert p.extract_effect() == {"type": "flip", "text": "{すごい}！"}

    def test_no_effect(self):
        p = feed_all(SpeechStreamParser(),
                     '{"emotion":"happy","speech":"無し"}')
        assert p.extract_effect() is None


# ── プロンプト注入 ────────────────────────────────────────────

class TestPromptSection:
    def test_section_lists_catalog(self):
        cfg = make_cfg()
        section = _effects_section(cfg, cfg.characters[0])
        assert "演出エフェクト" in section
        assert "sparkle" in section
        assert "announcement" in section
        assert "控えめ" in section

    def test_section_hidden_when_disabled(self):
        cfg = make_cfg()
        cfg.effects.enabled = False
        assert _effects_section(cfg, cfg.characters[0]) == ""

    def test_section_hidden_when_catalog_empty(self):
        cfg = AppConfig()
        assert _effects_section(cfg, cfg.characters[0]) == ""


# ── ディスパッチャ ────────────────────────────────────────────

class StubHub:
    def __init__(self):
        self.broadcasts: list[tuple[str, dict]] = []

    async def broadcast(self, channel, data):
        self.broadcasts.append((channel, data))


class PlayRecorder:
    def __init__(self):
        self.played: list[bytes] = []

    async def __call__(self, wav: bytes):
        self.played.append(wav)


def make_dispatcher(cfg=None):
    cfg = cfg or make_cfg()
    hub = StubHub()
    player = PlayRecorder()
    return EffectsDispatcher(cfg, hub, player), hub, player


class TestDispatcher:
    async def test_flip_broadcast(self):
        dispatcher, hub, _ = make_dispatcher()
        agent = SimpleNamespace(cfg=None, key="Character1", name="Character1")
        result = await dispatcher.dispatch(
            {"type": "flip", "name": "announcement"}, agent=agent)
        assert result["ok"] is True
        channel, msg = hub.broadcasts[0]
        assert channel == "avatar"
        assert msg["type"] == "effect"
        assert msg["effect_type"] == "flip"
        assert msg["text"] == "重大発表！"
        assert msg["character"] == "Character1"
        assert msg["speaker"] == "Character1"
        assert msg["duration_sec"] == 5.0

    async def test_cooldown_blocks_second(self):
        dispatcher, hub, _ = make_dispatcher()
        raw = {"type": "flip", "text": "一言"}
        assert (await dispatcher.dispatch(raw))["ok"] is True
        second = await dispatcher.dispatch(raw)
        assert second == {"ok": False, "reason": "cooldown"}
        assert len(hub.broadcasts) == 1

    async def test_manual_bypasses_and_preserves_cooldown(self):
        dispatcher, hub, _ = make_dispatcher()
        raw = {"type": "flip", "text": "一言"}
        assert (await dispatcher.dispatch(raw))["ok"] is True
        # 手動はクールダウン中でも発火できる
        assert (await dispatcher.dispatch(raw, manual=True))["ok"] is True
        # 手動発火は自発側の時計を更新しない（直後の自発はまだクールダウン中）
        assert (await dispatcher.dispatch(raw))["reason"] == "cooldown"
        assert len(hub.broadcasts) == 2

    async def test_invalid_dropped_silently(self):
        dispatcher, hub, _ = make_dispatcher()
        result = await dispatcher.dispatch({"type": "se", "name": "nope"})
        assert result["ok"] is False
        assert hub.broadcasts == []

    async def test_disabled(self):
        cfg = make_cfg()
        cfg.effects.enabled = False
        dispatcher, hub, _ = make_dispatcher(cfg)
        result = await dispatcher.dispatch({"type": "flip", "text": "一言"})
        assert result == {"ok": False, "reason": "effects disabled"}
        assert hub.broadcasts == []

    async def test_se_plays_via_audio(self, tmp_path, monkeypatch):
        se_dir = tmp_path / "assets" / "effects" / "se"
        se_dir.mkdir(parents=True)
        (se_dir / "sparkle.wav").write_bytes(b"RIFF-fake-wav")
        monkeypatch.chdir(tmp_path)
        dispatcher, hub, player = make_dispatcher()
        result = await dispatcher.dispatch({"type": "se", "name": "sparkle"})
        assert result["ok"] is True
        await asyncio.sleep(0.1)   # 背景再生タスクの完了待ち
        assert player.played == [b"RIFF-fake-wav"]
        # 画面側にも通知は届く（フロントは何も表示しない）
        assert hub.broadcasts[0][1]["effect_type"] == "se"

    async def test_cutin_broadcast_with_url(self, tmp_path, monkeypatch):
        cutin = tmp_path / "models" / "effects" / "cutin" / "duo.mp4"
        cutin.parent.mkdir(parents=True)
        cutin.write_bytes(b"fake-mp4")
        monkeypatch.chdir(tmp_path)
        dispatcher, hub, _ = make_dispatcher()
        result = await dispatcher.dispatch({"type": "cutin", "name": "duo"})
        assert result["ok"] is True
        msg = hub.broadcasts[0][1]
        assert msg["effect_type"] == "cutin"
        assert msg["url"] == "/models/effects/cutin/duo.mp4"

    async def test_cutin_missing_file_dropped(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)   # ファイルが無い場所
        dispatcher, hub, _ = make_dispatcher()
        result = await dispatcher.dispatch({"type": "cutin", "name": "duo"})
        assert result == {"ok": False, "reason": "cutin file missing"}
        assert hub.broadcasts == []


# ── Orchestrator 結合 ─────────────────────────────────────────

def effect_messages(hub) -> list[dict]:
    # FakeHub は broadcast を ("avatar", msg) として events に記録する
    return [msg for kind, msg in hub.events
            if kind == "avatar" and isinstance(msg, dict)
            and msg.get("type") == "effect"]


class TestOrchestratorEffects:
    async def test_speak_dispatches_effect(self):
        orch, hub = make_orchestrator(StubLLM("{}"), StubTTS())
        orch._cfg.effects.flip["announcement"] = EffectEntryConfig(
            text="重大発表！", label="重大発表")
        response = LLMResponse(
            speech="お知らせだよ。",
            effect={"type": "flip", "name": "announcement"})
        await orch._speak(response)
        msgs = effect_messages(hub)
        assert len(msgs) == 1
        assert msgs[0]["text"] == "重大発表！"
        assert msgs[0]["character"] == orch.lead.key

    async def test_speak_without_effect_sends_nothing(self):
        orch, hub = make_orchestrator(StubLLM("{}"), StubTTS())
        await orch._speak(LLMResponse(speech="普通の発話だよ。"))
        assert effect_messages(hub) == []

    async def test_streaming_dispatches_effect(self):
        orch, hub = make_orchestrator(StubLLM(RAW_WITH_EFFECT), StubTTS())
        response = await orch._generate_and_speak("sys")
        assert response is not None
        assert response.speech == "やっほー！今日も来てくれてありがとう。"
        msgs = effect_messages(hub)
        assert len(msgs) == 1
        assert msgs[0]["effect_type"] == "flip"
        assert msgs[0]["text"] == "重大発表！"

    async def test_streaming_salvage_dispatches_effect(self):
        """発話JSON全体が壊れていても effect は配送されること"""
        orch, hub = make_orchestrator(StubLLM(RAW_WITH_EFFECT + "xyz{"),
                                      StubTTS())
        response = await orch._generate_and_speak("sys")
        assert response is not None
        assert response.speech == "やっほー！今日も来てくれてありがとう。"
        msgs = effect_messages(hub)
        assert len(msgs) == 1
        assert msgs[0]["text"] == "重大発表！"

    async def test_streaming_invalid_effect_dropped(self):
        raw = ('{"emotion":"happy","motion":"wave",'
               '"speech":"やっほー！今日も来てくれてありがとう。",'
               '"effect":{"type":"se","name":"存在しない名前"}}')
        orch, hub = make_orchestrator(StubLLM(raw), StubTTS())
        response = await orch._generate_and_speak("sys")
        assert response is not None
        assert response.speech == "やっほー！今日も来てくれてありがとう。"
        assert effect_messages(hub) == []

    async def test_trigger_effect_api_path(self):
        orch, hub = make_orchestrator(StubLLM("{}"), StubTTS())
        orch._cfg.effects.flip["announcement"] = EffectEntryConfig(
            text="重大発表！", label="重大発表")
        result = await orch.trigger_effect("flip", name="announcement")
        assert result["ok"] is True
        assert len(effect_messages(hub)) == 1
        bad = await orch.trigger_effect("se", name="存在しない")
        assert bad["ok"] is False

    async def test_effects_catalog_summary(self):
        orch, _ = make_orchestrator(StubLLM("{}"), StubTTS())
        orch._cfg.effects.se["sparkle"] = EffectEntryConfig(
            file="assets/effects/se/sparkle.wav", label="キラキラ")
        summary = orch.effects_catalog()
        assert summary["enabled"] is True
        assert [e["name"] for e in summary["se"]] == ["sparkle"]
