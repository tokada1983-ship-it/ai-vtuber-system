"""コアコンポーネントのユニットテスト"""
import asyncio
from datetime import datetime

import pytest

from backend.config import load_config
from backend.llm.parser import parse_llm_output
from backend.memory.conversation_log import ConversationLog
from backend.models import Comment, Emotion, LLMResponse, Motion
from backend.selection.comment_selector import CommentSelector
from backend.sources.mock_source import MockSource


def make_comment(msg: str, author: str = "テスト", **kwargs) -> Comment:
    return Comment(
        id="test_1",
        author=author,
        author_channel_id="ch_test",
        message=msg,
        timestamp=datetime.now(),
        **kwargs,
    )


# ── LLM Parser ────────────────────────────────────────────────

class TestLLMParser:
    def test_valid_json(self):
        raw = '{"speech": "こんにちは！", "emotion": "happy", "motion": "wave", "target_comment_id": null, "skip": false}'
        r = parse_llm_output(raw)
        assert r.speech == "こんにちは！"
        assert r.emotion == Emotion.HAPPY
        assert r.motion == Motion.WAVE
        assert r.skip is False

    def test_dirty_json_with_preamble(self):
        raw = 'Sure! Here is my response:\n{"speech": "テスト", "emotion": "neutral", "motion": "idle", "skip": false}'
        r = parse_llm_output(raw)
        assert r.speech == "テスト"
        assert r.skip is False

    def test_code_fenced_json(self):
        raw = '```json\n{"speech": "hello", "emotion": "neutral", "motion": "nod", "skip": false}\n```'
        r = parse_llm_output(raw)
        assert r.speech == "hello"

    def test_skip_true(self):
        raw = '{"speech": "", "emotion": "neutral", "motion": "idle", "skip": true}'
        r = parse_llm_output(raw)
        assert r.skip is True

    def test_empty_speech_becomes_skip(self):
        raw = '{"speech": "", "emotion": "neutral", "motion": "idle", "skip": false}'
        r = parse_llm_output(raw)
        assert r.skip is True

    def test_invalid_json(self):
        r = parse_llm_output("これはJSONではありません")
        assert r.skip is True

    def test_unknown_emotion_fallback(self):
        raw = '{"speech": "hi", "emotion": "unknown_value", "motion": "idle", "skip": false}'
        r = parse_llm_output(raw)
        assert r.emotion == Emotion.NEUTRAL


# ── CommentSelector ───────────────────────────────────────────

class TestCommentSelector:
    def setup_method(self):
        cfg = load_config("config.yaml")
        self.sel = CommentSelector(cfg.orchestrator, "Character1")

    def test_normal_comment(self):
        c = make_comment("普通のコメント")
        assert self.sel.score(c) == 2.0

    def test_superchat_score(self):
        c = make_comment("ありがとう！", is_superchat=True, superchat_amount=1000)
        assert self.sel.score(c) > 5.0

    def test_question_score(self):
        c = make_comment("ゲームは好きですか？")
        assert self.sel.score(c) > 2.0

    def test_character_name_mention(self):
        c = make_comment("Character1ちゃんかわいい！")
        assert self.sel.score(c) > 2.0

    def test_too_long_comment_filtered(self):
        c = make_comment("あ" * 300)
        assert self.sel.score(c) == -1.0

    def test_empty_comment_filtered(self):
        c = make_comment("   ")
        assert self.sel.score(c) == -1.0

    def test_member_score(self):
        c = make_comment("こんにちは", is_member=True)
        assert self.sel.score(c) > 2.0


# ── ConversationLog ───────────────────────────────────────────

class TestConversationLog:
    def test_add_and_get(self):
        log = ConversationLog(max_turns=3)
        c = make_comment("テストコメント")
        log.add_comment(c)
        log.add_response(LLMResponse(speech="応答テスト", emotion=Emotion.HAPPY, motion=Motion.NOD))
        msgs = log.get_messages()
        assert len(msgs) == 2
        assert msgs[0].role == "user"
        assert msgs[1].role == "assistant"

    def test_max_turns_limit(self):
        log = ConversationLog(max_turns=2)
        for i in range(5):
            log.add_comment(make_comment(f"コメント{i}", author=f"user{i}"))
        msgs = log.get_messages()
        # max_turns*2 = 4 を超えない
        assert len(msgs) <= 4

    def test_injection_wrap(self):
        log = ConversationLog()
        c = make_comment("テスト", author="Injector")
        log.add_comment(c)
        msgs = log.get_messages()
        assert "<viewer_comment" in msgs[0].content
        assert "Injector" in msgs[0].content

    def test_skip_response_not_added(self):
        log = ConversationLog()
        c = make_comment("コメント")
        log.add_comment(c)
        log.add_response(LLMResponse(speech="", skip=True))
        msgs = log.get_messages()
        assert len(msgs) == 1  # assistant は追加されない


# ── MockSource ────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_mock_source_yields_comments():
    src = MockSource(interval_sec=0.05)
    await src.start("mock")
    comments = []
    async for c in src.comments():
        comments.append(c)
        if len(comments) >= 3:
            break
    await src.stop()
    assert len(comments) == 3
    assert all(c.message for c in comments)


@pytest.mark.asyncio
async def test_mock_source_is_healthy():
    src = MockSource(interval_sec=0.1)
    await src.start("mock")
    assert src.is_healthy is True
    await src.stop()
