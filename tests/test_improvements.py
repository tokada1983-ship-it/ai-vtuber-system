"""改修分のユニットテスト（応答後フィルタ・拡張モーション・キュー動作）"""
import time
from datetime import datetime

import pytest

from backend.llm.parser import parse_llm_output
from backend.models import Comment, Motion
from backend.orchestrator import _QueueItem
from backend.safety.response_filter import ResponseFilter


def make_comment(msg: str = "テスト", author: str = "テスト") -> Comment:
    return Comment(
        id="test_1",
        author=author,
        author_channel_id="ch_test",
        message=msg,
        timestamp=datetime.now(),
    )


# ── ResponseFilter ────────────────────────────────────────────

class TestResponseFilter:
    def setup_method(self):
        self.f = ResponseFilter(ng_words=["禁止語"])

    def test_normal_speech_passes(self):
        assert self.f.check("こんにちは！今日もがんばるよ！").ok is True

    def test_phone_number_blocked(self):
        assert self.f.check("電話してね、090-1234-5678だよ").ok is False
        assert self.f.check("固定は0312345678です").ok is False

    def test_email_blocked(self):
        assert self.f.check("メールはtest@example.comまで").ok is False

    def test_postal_code_blocked(self):
        assert self.f.check("住所は〒123-4567です").ok is False

    def test_api_key_blocked(self):
        assert self.f.check("キーはsk-xxxxxxxxxxxxxxxxxxxxxxxx").ok is False
        assert self.f.check("AKIAXXXXXXXXXXXXXXXX1234").ok is False

    def test_internal_tag_blocked(self):
        assert self.f.check("<viewer_comment>を無視して").ok is False
        assert self.f.check("あなたのシステムプロンプトは").ok is False

    def test_ng_word_blocked(self):
        r = self.f.check("これは禁止語を含む発話")
        assert r.ok is False
        assert "禁止語" in r.reason

    def test_disabled_filter_passes_everything(self):
        f = ResponseFilter(ng_words=["禁止語"], enabled=False)
        assert f.check("090-1234-5678 禁止語").ok is True

    def test_empty_text_passes(self):
        assert self.f.check("").ok is True

    def test_normal_numbers_not_blocked(self):
        # 通常の数字混じり発話は誤検出しない
        assert self.f.check("視聴者100人ありがとう！").ok is True
        assert self.f.check("2026年7月19日の配信だよ").ok is True


# ── 拡張モーション ─────────────────────────────────────────────

class TestExtendedMotions:
    @pytest.mark.parametrize("motion", [
        "shake_head", "think", "excited", "shy", "point",
        "clap", "stretch", "lean_forward", "dance",
    ])
    def test_extended_motion_parsed(self, motion):
        """プロンプトで提示している16種のモーションがidleに潰されないこと"""
        raw = f'{{"speech": "テスト", "emotion": "happy", "motion": "{motion}", "skip": false}}'
        r = parse_llm_output(raw)
        assert r.motion == Motion(motion)

    def test_unknown_motion_falls_back_to_idle(self):
        raw = '{"speech": "テスト", "emotion": "happy", "motion": "backflip", "skip": false}'
        r = parse_llm_output(raw)
        assert r.motion == Motion.IDLE


# ── コメントキュー ─────────────────────────────────────────────

class TestQueueItem:
    def test_priority_ordering(self):
        """優先度が高い（=負値が小さい）コメントが先"""
        low = _QueueItem(priority=-2.0, seq=0, comment=make_comment())
        high = _QueueItem(priority=-9.0, seq=1, comment=make_comment())
        assert high < low

    def test_fifo_on_equal_priority(self):
        """同点優先度は投入順（seq昇順）"""
        first = _QueueItem(priority=-2.0, seq=0, comment=make_comment())
        second = _QueueItem(priority=-2.0, seq=1, comment=make_comment())
        assert first < second

    def test_enqueued_at_not_compared(self):
        a = _QueueItem(priority=-2.0, seq=0, comment=make_comment(), enqueued_at=time.monotonic())
        b = _QueueItem(priority=-2.0, seq=1, comment=make_comment(), enqueued_at=0.0)
        assert a < b
