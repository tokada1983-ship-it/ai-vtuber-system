"""Phase 5 Stage D（タグ方式の消費側移行）のテスト

- parse_tag_output / scan_tag_output: タグ走査、タグ破損時の発話成立、
  fx/th の配送形、ok / degraded / fatal の計測（JSON 側と分離）
- TagSpeechStreamParser: タグの途中で分割されても文が正しく出ること、
  peek / extract_effect / extract_thought / build_response
- プロンプト: protocol=tag でフォーマット節・fx節・th節がタグ記法に
  なること、protocol=json では従来と変わらないこと
- 切替: config 既定は json、Ollama の format:json 抑制、factory の
  スタンプ、knowledge_status の計測切替
- 結合: タグ方式で _generate_and_speak が喋り、thought が TTS に
  混ざらないこと。json 方式の従来動作が壊れていないこと（ロールバック）
"""
from __future__ import annotations

from backend.config import AppConfig, EffectEntryConfig, LLMBackendConfig
from backend.effects import resolve_effects, validate_effect
from backend.factory import build_llm
from backend.llm.ollama_backend import OllamaBackend
from backend.llm.parser import (parse_outcome_stats, parse_tag_output,
                                scan_tag_output, tag_parse_outcome_stats)
from backend.llm.prompt_builder import (_RESPONSE_FORMAT,
                                        _RESPONSE_FORMAT_TAG,
                                        _effects_section, _thoughts_section,
                                        build_system_prompt)
from backend.llm.streaming import TagSpeechStreamParser
from tests.helpers import RAW, StubLLM, StubTTS, make_orchestrator

SPEECH_TEXT = "やっほー！今日も来てくれてありがとう。"
THOUGHT_TEXT = "つい笑っちゃった"
TAG_RAW = ("[emo:happy][mot:wave]" + SPEECH_TEXT
           + "[th:" + THOUGHT_TEXT + "]")


def _delta(before: dict, after: dict) -> dict:
    return {k: after.get(k, 0) - before.get(k, 0) for k in after}


# ── parse_tag_output ─────────────────────────────────────────

class TestParseTagOutput:
    def test_well_formed_ok(self):
        before = tag_parse_outcome_stats()
        resp = parse_tag_output("[emo:happy][mot:wave]" + SPEECH_TEXT)
        after = tag_parse_outcome_stats()
        assert not resp.skip
        assert resp.speech == SPEECH_TEXT
        assert resp.emotion.value == "happy"
        assert resp.motion.value == "wave"
        assert _delta(before, after) == {"ok": 1, "degraded": 0, "fatal": 0}

    def test_fx_and_thought_extracted(self):
        resp = parse_tag_output(
            "[emo:happy][mot:nod]すごいね！"
            "[fx:se:fanfare][th:" + THOUGHT_TEXT + "]")
        assert resp.effect == {"type": "se", "name": "fanfare"}
        assert resp.thought == THOUGHT_TEXT
        assert THOUGHT_TEXT not in resp.speech

    def test_flip_text_and_name_resolution(self):
        resp = parse_tag_output("[emo:happy][mot:nod]見て！[fx:flip:大ニュースだよ]")
        assert resp.effect == {"type": "flip", "text": "大ニュースだよ"}
        resp = parse_tag_output("[emo:happy][mot:nod]見て！[fx:flip:big_news]",
                                flip_names={"big_news"})
        assert resp.effect == {"type": "flip", "name": "big_news"}

    def test_missing_motion_degraded_with_default(self):
        before = tag_parse_outcome_stats()
        resp = parse_tag_output("[emo:sad]ごめんね、今日は元気がないんだ。")
        after = tag_parse_outcome_stats()
        assert not resp.skip
        assert resp.motion.value == "idle"
        assert _delta(before, after)["degraded"] == 1

    def test_invalid_values_degraded_with_defaults(self):
        resp = parse_tag_output("[emo:excited][mot:fly]とぶよー！")
        assert not resp.skip
        assert resp.emotion.value == "neutral"
        assert resp.motion.value == "idle"

    def test_broken_trailing_tag_keeps_speech(self):
        # 文頭タグが未完のまま切断されても、流れていた発話は成立する
        resp = parse_tag_output("途中で切れても大丈夫だよ[emo:hap")
        assert not resp.skip
        assert resp.speech == "途中で切れても大丈夫だよ"

    def test_unknown_bracket_fragment_removed(self):
        resp = parse_tag_output("[emo:happy][mot:nod]これは[注意]テストだよ。")
        assert resp.speech == "これはテストだよ。"

    def test_tags_only_is_skip_and_counts_ok(self):
        before = tag_parse_outcome_stats()
        resp = parse_tag_output("[emo:neutral][mot:idle]")
        after = tag_parse_outcome_stats()
        assert resp.skip and resp.speech == ""
        assert _delta(before, after) == {"ok": 1, "degraded": 0, "fatal": 0}

    def test_empty_output_is_fatal(self):
        before = tag_parse_outcome_stats()
        resp = parse_tag_output("")
        after = tag_parse_outcome_stats()
        assert resp.skip
        assert _delta(before, after)["fatal"] == 1

    def test_meta_leak_blocked(self):
        resp = parse_tag_output("[emo:neutral][mot:idle]ユーザーはそう言った")
        assert resp.skip

    def test_json_stats_not_polluted(self):
        before = parse_outcome_stats()
        parse_tag_output("[emo:happy][mot:wave]分離の確認だよ。")
        assert parse_outcome_stats() == before
        assert parse_outcome_stats("tag") == tag_parse_outcome_stats()


# ── scan_tag_output ──────────────────────────────────────────

class TestScanTagOutput:
    def test_partial_tag_held_while_streaming(self):
        scan = scan_tag_output("[emo:hap", final=False)
        assert scan.held_from == 0
        assert scan.speech == ""
        assert scan.emotion_raw == ""

    def test_partial_tag_dropped_when_final(self):
        scan = scan_tag_output("発話だよ[emo:hap", final=True)
        assert scan.speech == "発話だよ"
        assert "partial_dropped" in scan.issues

    def test_leading_order_issue(self):
        scan = scan_tag_output("先に発話[emo:happy][mot:nod]続き")
        assert scan.emotion_raw == "happy"
        assert "emo_not_leading" in scan.issues


# ── TagSpeechStreamParser ────────────────────────────────────

def _feed_in_chunks(parser: TagSpeechStreamParser, raw: str,
                    chunk: int) -> list[str]:
    out: list[str] = []
    for i in range(0, len(raw), chunk):
        out += parser.feed(raw[i:i + chunk])
    out += parser.flush()
    return out


class TestTagStreamParser:
    def test_sentences_despite_tag_splits(self):
        parser = TagSpeechStreamParser()
        sentences = _feed_in_chunks(parser, TAG_RAW, 3)
        assert "".join(sentences) == SPEECH_TEXT
        assert all("[" not in s for s in sentences)

    def test_peek_fills_in_as_tags_arrive(self):
        parser = TagSpeechStreamParser()
        parser.feed("[emo:ha")
        assert parser.peek("emotion") == ""
        parser.feed("ppy][mot:wave]やっほー！")
        assert parser.peek("emotion") == "happy"
        assert parser.peek("motion") == "wave"

    def test_extract_effect_and_thought(self):
        parser = TagSpeechStreamParser()
        _feed_in_chunks(parser,
                        "[emo:happy][mot:nod]すごいね！"
                        "[fx:se:fanfare][th:" + THOUGHT_TEXT + "]", 5)
        assert parser.extract_effect() == {"type": "se", "name": "fanfare"}
        assert parser.extract_thought() == THOUGHT_TEXT

    def test_build_response_fields_and_ok_count(self):
        before = tag_parse_outcome_stats()
        parser = TagSpeechStreamParser()
        _feed_in_chunks(parser, TAG_RAW, 4)
        resp = parser.build_response()
        after = tag_parse_outcome_stats()
        assert not resp.skip
        assert resp.speech == SPEECH_TEXT
        assert resp.emotion.value == "happy"
        assert resp.motion.value == "wave"
        assert resp.thought == THOUGHT_TEXT
        assert _delta(before, after)["ok"] == 1

    def test_build_response_degraded_when_tag_missing(self):
        before = tag_parse_outcome_stats()
        parser = TagSpeechStreamParser()
        _feed_in_chunks(parser, "[emo:happy]モーション無しだよ。", 4)
        resp = parser.build_response()
        after = tag_parse_outcome_stats()
        assert not resp.skip and resp.motion.value == "idle"
        assert _delta(before, after)["degraded"] == 1

    def test_tags_only_stream_is_skip(self):
        parser = TagSpeechStreamParser()
        out = _feed_in_chunks(parser, "[emo:neutral][mot:idle]", 3)
        assert out == []
        resp = parser.build_response()
        assert resp.skip

    def test_is_skip_never_aborts_mid_stream(self):
        parser = TagSpeechStreamParser()
        parser.feed("[emo:neutral][mot:idle]")
        assert parser.is_skip is False


# ── fx の配送互換（Phase 2 の検証を通ること） ─────────────────

class TestEffectCompatibility:
    def test_se_effect_passes_validation(self):
        cfg = AppConfig()
        cfg.effects.se["fanfare"] = EffectEntryConfig(
            file="assets/effects/se/fanfare.wav", label="ファンファーレ")
        resolved = resolve_effects(cfg, None)
        resp = parse_tag_output("[emo:happy][mot:nod]やったー！[fx:se:fanfare]")
        spec = validate_effect(resp.effect, resolved)
        assert spec is not None
        assert spec.type == "se" and spec.name == "fanfare"

    def test_flip_name_resolves_to_catalog_text(self):
        cfg = AppConfig()
        cfg.effects.flip["big"] = EffectEntryConfig(text="大ニュース", label="大ニュース")
        resolved = resolve_effects(cfg, None)
        resp = parse_tag_output("[emo:happy][mot:nod]発表だよ[fx:flip:big]",
                                flip_names={"big"})
        spec = validate_effect(resp.effect, resolved)
        assert spec is not None
        assert spec.text == "大ニュース"


# ── プロンプト ───────────────────────────────────────────────

class TestTagPrompt:
    def test_tag_format_section_matches_training_text(self):
        # 学習データの正文（PROTOCOL §6）と実行時が一致すること
        assert "[emo:happy][mot:wave]発話テキスト" in _RESPONSE_FORMAT_TAG
        assert "タグ付きの発話テキストだけを出力すること" in _RESPONSE_FORMAT_TAG
        assert "JSONのみ出力すること" not in _RESPONSE_FORMAT_TAG

    def test_system_prompt_switches_format(self):
        cfg = AppConfig()
        json_prompt = build_system_prompt(cfg)
        assert "JSONのみ出力すること" in json_prompt
        cfg.llm.protocol = "tag"
        tag_prompt = build_system_prompt(cfg)
        assert "[emo:happy][mot:wave]発話テキスト" in tag_prompt
        assert "JSONのみ出力すること" not in tag_prompt

    def test_effects_section_tag_wording(self):
        cfg = AppConfig()
        cfg.effects.se["sparkle"] = EffectEntryConfig(
            file="assets/effects/se/sparkle.wav", label="キラキラ")
        sec_tag = _effects_section(cfg, cfg.characters[0], "tag")
        assert "[fx:se:sparkle]" in sec_tag
        sec_json = _effects_section(cfg, cfg.characters[0])
        assert '"effect"' in sec_json
        assert "[fx:" not in sec_json

    def test_thoughts_section_tag_wording(self):
        cfg = AppConfig()
        sec_tag = _thoughts_section(cfg, cfg.characters[0], "tag")
        assert "[th:" in sec_tag
        sec_json = _thoughts_section(cfg, cfg.characters[0])
        assert '"thought"' in sec_json
        assert _RESPONSE_FORMAT  # JSON 側の正文は不変


# ── プロトコル切替 ───────────────────────────────────────────

class TestProtocolSwitch:
    def test_config_default_is_json(self):
        assert AppConfig().llm.protocol == "json"

    def test_ollama_format_json_suppressed_only_in_tag(self):
        cfg = LLMBackendConfig(base_url="http://localhost:11434", model="m",
                               extra_payload={"think": False, "format": "json"})
        backend = OllamaBackend(cfg)
        payload = backend._build_payload("sys", [], stream=False)
        assert payload["format"] == "json"
        backend.protocol = "tag"
        payload = backend._build_payload("sys", [], stream=False)
        assert "format" not in payload
        assert payload["think"] is False  # think: false は方式によらず維持

    def test_factory_stamps_protocol(self):
        cfg = AppConfig()
        cfg.llm.protocol = "tag"
        primary, fallback = build_llm(cfg)
        assert primary.protocol == "tag"
        if fallback is not None:
            assert fallback.protocol == "tag"

    def test_knowledge_status_switches_stats(self):
        orch, _hub = make_orchestrator(StubLLM("{}"), StubTTS())
        js = orch.knowledge_status()["parse_outcomes"]
        assert "no_json_fatal" in js
        orch._cfg.llm.protocol = "tag"
        before = tag_parse_outcome_stats()
        parse_tag_output("[emo:happy][mot:nod]テストだよ。")
        tg = orch.knowledge_status()["parse_outcomes"]
        assert "no_json_fatal" not in tg
        assert tg["ok"] == before["ok"] + 1


# ── Orchestrator 結合 ────────────────────────────────────────

def thought_messages(hub) -> list[dict]:
    return [msg for kind, msg in hub.events
            if kind == "avatar" and isinstance(msg, dict)
            and msg.get("type") == "thought"]


class TestOrchestratorTagSpeak:
    async def test_tag_streaming_speaks_and_separates_thought(self):
        tts = StubTTS()
        orch, hub = make_orchestrator(StubLLM(TAG_RAW), tts)
        orch._cfg.llm.protocol = "tag"
        response = await orch._generate_and_speak("sys")
        assert response is not None
        assert response.speech == SPEECH_TEXT
        assert response.thought == THOUGHT_TEXT
        assert response.emotion.value == "happy"
        assert response.motion.value == "wave"
        # TTS に渡るのは発話文のみ（タグ・内心は混ざらない）
        assert tts.calls == ["やっほー！", "今日も来てくれてありがとう。"]
        assert all("[" not in call for call in tts.calls)
        msgs = thought_messages(hub)
        assert len(msgs) == 1
        assert msgs[0]["text"] == THOUGHT_TEXT

    async def test_tag_streaming_broken_output_still_speaks(self):
        # 文頭タグが壊れていても発話は成立する（タグ方式の本旨）
        tts = StubTTS()
        orch, _hub = make_orchestrator(
            StubLLM("[emo:hap]途中で切れても喋るよ、大丈夫。"), tts)
        orch._cfg.llm.protocol = "tag"
        response = await orch._generate_and_speak("sys")
        assert response is not None
        assert "途中で切れても喋るよ、大丈夫。" in response.speech
        assert all("[" not in call for call in tts.calls)

    async def test_json_protocol_unchanged(self):
        # ロールバック確認: protocol=json（既定）では従来どおり動く
        tts = StubTTS()
        orch, _hub = make_orchestrator(StubLLM(RAW), tts)
        assert orch._cfg.llm.protocol == "json"
        response = await orch._generate_and_speak("sys")
        assert response is not None
        assert response.speech == SPEECH_TEXT
        assert tts.calls == ["やっほー！", "今日も来てくれてありがとう。"]
