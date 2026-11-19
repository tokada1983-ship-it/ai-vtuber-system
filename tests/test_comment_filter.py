"""コメント選別（トリアージ）のテスト

CommentTriage の判定カテゴリ（必ず拾う／確率で拾う／拾わない）と、
Orchestrator の取り込みループへの結合（落としたコメントも記録と
視聴者プロフィールは更新されること）を、外部サービスなしで検証する。
"""
import asyncio
import json
from datetime import datetime
from typing import AsyncIterator

from backend.config import AppConfig, CommentFilterConfig, load_config
from backend.memory.stream_archive import StreamArchiver
from backend.models import Comment
from backend.selection.comment_triage import CommentTriage
from backend.sources.base import CommentSource
from helpers import RAW, StubLLM, StubTTS, make_orchestrator

NAMES = ["Character1", "Character2"]


def make_comment(msg: str, author: str = "視聴者A", channel: str = "chA",
                 **kw) -> Comment:
    return Comment(id="c", author=author, author_channel_id=channel,
                   message=msg, timestamp=datetime.now(), **kw)


def make_triage(rng=None, known=None, **cfgkw):
    """決定論的な rng / clock を注入したトリアージと、時刻を進める箱を返す"""
    cfg = CommentFilterConfig(**cfgkw)
    now = [1000.0]
    triage = CommentTriage(cfg, NAMES, known_author=known,
                           rng=rng or (lambda: 0.0),
                           clock=lambda: now[0])
    return triage, now


def seen_author_triage(rng=None, **cfgkw):
    """視聴者A を既見にしたトリアージ（初コメ判定を外した検証用）"""
    triage, now = make_triage(rng=rng, **cfgkw)
    first = triage.evaluate(make_comment("こんにちは、見てます"),
                            score=2.0, queue_size=0)
    assert first.category == "always_first"
    return triage, now


# ── 必ず拾う ────────────────────────────────────────────────

class TestAlwaysPick:
    def test_name_call(self):
        triage, _ = seen_author_triage(base_pickup_probability=0.0)
        r = triage.evaluate(make_comment("Character1、聞いて聞いて"),
                            score=4.0, queue_size=0)
        assert r.picked and r.category == "always_name"

    def test_question_sentence(self):
        triage, _ = seen_author_triage(base_pickup_probability=0.0)
        r = triage.evaluate(make_comment("今日の天気はどうですか"),
                            score=3.5, queue_size=0)
        assert r.picked and r.category == "always_question"

    def test_question_mark(self):
        triage, _ = seen_author_triage(base_pickup_probability=0.0)
        r = triage.evaluate(make_comment("それ本当？"),
                            score=3.5, queue_size=0)
        assert r.picked and r.category == "always_question"

    def test_first_comment(self):
        triage, _ = make_triage(base_pickup_probability=0.0)
        r = triage.evaluate(make_comment("はじめまして、見てます"),
                            score=2.0, queue_size=0)
        assert r.picked and r.category == "always_first"

    def test_superchat(self):
        triage, _ = seen_author_triage(base_pickup_probability=0.0)
        r = triage.evaluate(make_comment("応援してます", is_superchat=True),
                            score=7.0, queue_size=0)
        assert r.picked and r.category == "always_superchat"

    def test_known_author_is_not_first(self):
        triage, _ = make_triage(rng=lambda: 0.9, base_pickup_probability=0.5,
                                known=lambda c: True)
        r = triage.evaluate(make_comment("また来ました"),
                            score=2.0, queue_size=0)
        assert not r.picked and r.category == "probability_miss"

    def test_known_author_lookup_failure_counts_as_first(self):
        def boom(_comment):
            raise RuntimeError("db error")
        triage, _ = make_triage(known=boom)
        r = triage.evaluate(make_comment("こんにちは"),
                            score=2.0, queue_size=0)
        assert r.picked and r.category == "always_first"


# ── 拾わない（ハードドロップ） ──────────────────────────────

class TestHardDrop:
    def test_ng_word(self):
        triage, _ = make_triage(ng_words=["禁止ワード"])
        r = triage.evaluate(make_comment("禁止ワードって何ですか"),
                            score=3.5, queue_size=0)
        assert not r.picked and r.category == "ng_word"

    def test_ng_word_beats_name_call(self):
        triage, _ = make_triage(ng_words=["禁止ワード"])
        r = triage.evaluate(make_comment("Character1、禁止ワードだよ"),
                            score=5.5, queue_size=0)
        assert not r.picked and r.category == "ng_word"

    def test_duplicate_within_window(self):
        triage, _ = seen_author_triage()
        msg = "同じ文を送ります"
        r1 = triage.evaluate(make_comment(msg), score=2.0, queue_size=0)
        assert r1.picked
        r2 = triage.evaluate(make_comment(msg), score=2.0, queue_size=0)
        assert not r2.picked and r2.category == "duplicate"

    def test_duplicate_expires_after_window(self):
        triage, now = seen_author_triage()
        msg = "同じ文を送ります"
        triage.evaluate(make_comment(msg), score=2.0, queue_size=0)
        now[0] += 61.0  # duplicate_window_sec=60 を過ぎる
        r = triage.evaluate(make_comment(msg), score=2.0, queue_size=0)
        assert r.picked and r.category == "scored"

    def test_same_text_other_author_is_not_duplicate(self):
        triage, _ = seen_author_triage()
        triage.evaluate(make_comment("同じ文", author="視聴者A", channel="chA"),
                        score=2.0, queue_size=0)
        r = triage.evaluate(make_comment("同じ文", author="視聴者B", channel="chB"),
                            score=2.0, queue_size=0)
        assert r.picked and r.category == "always_first"

    def test_spam_repeat(self):
        triage, _ = make_triage()
        r = triage.evaluate(make_comment("ああああああ"),
                            score=2.0, queue_size=0)
        assert not r.picked and r.category == "spam_repeat"

    def test_question_mark_flood_is_spam_not_question(self):
        triage, _ = make_triage()
        r = triage.evaluate(make_comment("？？？？？？"),
                            score=3.5, queue_size=0)
        assert not r.picked and r.category == "spam_repeat"

    def test_emphasis_is_not_flood(self):
        triage, _ = seen_author_triage()
        r = triage.evaluate(make_comment("すごい！！！！！"),
                            score=2.0, queue_size=0)
        assert r.picked and r.category == "scored"

    def test_reaction_only_kusa(self):
        triage, _ = seen_author_triage()
        r = triage.evaluate(make_comment("草"), score=2.0, queue_size=0)
        assert not r.picked and r.category == "reaction_only"

    def test_reaction_only_www(self):
        triage, _ = seen_author_triage()
        r = triage.evaluate(make_comment("www"), score=2.0, queue_size=0)
        assert not r.picked and r.category == "reaction_only"

    def test_reaction_only_emoji(self):
        triage, _ = seen_author_triage()
        r = triage.evaluate(make_comment("😆😆😆"), score=2.0, queue_size=0)
        assert not r.picked and r.category == "reaction_only"

    def test_reaction_word_with_substance_is_picked(self):
        triage, _ = seen_author_triage()
        r = triage.evaluate(make_comment("草、今日も面白いね"),
                            score=2.0, queue_size=0)
        assert r.picked and r.category == "scored"

    def test_first_comment_reaction_is_picked_as_first(self):
        """初コメはリアクションだけでも拾う（歓迎のため。連打・NGは別）"""
        triage, _ = make_triage()
        r = triage.evaluate(make_comment("草"), score=2.0, queue_size=0)
        assert r.picked and r.category == "always_first"


# ── 確率で拾う（混雑度） ────────────────────────────────────

class TestProbability:
    def test_base_probability_hit_and_miss(self):
        triage, _ = seen_author_triage(rng=lambda: 0.3,
                                       base_pickup_probability=0.5)
        r = triage.evaluate(make_comment("面白いね"), score=2.0, queue_size=0)
        assert r.picked and r.category == "scored"

        triage2, _ = seen_author_triage(rng=lambda: 0.7,
                                        base_pickup_probability=0.5)
        r2 = triage2.evaluate(make_comment("面白いね"), score=2.0, queue_size=0)
        assert not r2.picked and r2.category == "probability_miss"

    def test_congestion_lowers_probability(self):
        # rng=0.3: 通常 p=1.0 で拾い、滞留 p=0.5 でも拾い、飽和 p=0.2 で落とす
        triage, _ = seen_author_triage(rng=lambda: 0.3)
        msg = "普通の感想です"
        r = triage.evaluate(make_comment(msg + "1"), score=5.0, queue_size=0)
        assert r.picked
        r = triage.evaluate(make_comment(msg + "2"), score=5.0, queue_size=5)
        assert r.picked
        r = triage.evaluate(make_comment(msg + "3"), score=5.0, queue_size=15)
        assert not r.picked and r.category == "probability_miss"

    def test_saturated_low_score_hard_drop(self):
        triage, _ = seen_author_triage(rng=lambda: 0.0)
        r = triage.evaluate(make_comment("低スコアの感想"),
                            score=2.0, queue_size=15)
        assert not r.picked and r.category == "saturated_low_score"

    def test_saturated_high_score_uses_probability(self):
        triage, _ = seen_author_triage(rng=lambda: 0.1)
        r = triage.evaluate(make_comment("高スコアの感想"),
                            score=9.0, queue_size=15)
        assert r.picked and r.category == "scored"


# ── 無効化 ──────────────────────────────────────────────────

class TestDisabled:
    def test_disabled_passes_everything(self):
        triage, _ = make_triage(enabled=False, ng_words=["禁止ワード"])
        r = triage.evaluate(make_comment("禁止ワードですか"),
                            score=3.5, queue_size=99)
        assert r.picked and r.category == "disabled"
        r = triage.evaluate(make_comment("ああああああ"),
                            score=2.0, queue_size=99)
        assert r.picked and r.category == "disabled"


# ── 設定 ────────────────────────────────────────────────────

class TestConfig:
    def test_defaults(self):
        cfg = AppConfig()
        assert cfg.comment_filter.enabled is True
        assert cfg.comment_filter.base_pickup_probability == 1.0
        assert cfg.comment_filter.ng_words == []

    def test_load_config_section(self, tmp_path):
        path = tmp_path / "config.yaml"
        path.write_text(
            "comment_filter:\n"
            "  enabled: false\n"
            "  congested_queue_size: 3\n"
            "  ng_words: [だめ]\n",
            encoding="utf-8")
        cfg = load_config(path)
        assert cfg.comment_filter.enabled is False
        assert cfg.comment_filter.congested_queue_size == 3
        assert cfg.comment_filter.ng_words == ["だめ"]


# ── Orchestrator 結合 ───────────────────────────────────────

class FakeSource(CommentSource):
    """用意したコメントを順に流すだけのソース"""

    def __init__(self, comments: list[Comment]):
        self._comments = comments

    async def start(self, video_id: str) -> None:
        pass

    async def stop(self) -> None:
        pass

    async def comments(self) -> AsyncIterator[Comment]:
        for c in self._comments:
            yield c

    @property
    def is_healthy(self) -> bool:
        return True


async def run_ingest(orch, comments):
    orch._source = FakeSource(comments)
    orch._running = True
    task = asyncio.create_task(orch._comment_ingest_loop())
    await asyncio.wait_for(task, timeout=10)


def drain_messages(orch) -> list[str]:
    out = []
    while not orch._comment_queue.empty():
        out.append(orch._comment_queue.get_nowait().comment.message)
    return out


class TestOrchestratorIntegration:
    async def test_triage_filters_and_records(self, tmp_path):
        orch, _hub = make_orchestrator(StubLLM(RAW), StubTTS())
        archiver = StreamArchiver(tmp_path)
        archiver.start_run("testvideo")
        orch._archiver = archiver

        comments = [
            make_comment("こんにちは！", author="視聴者A", channel="chA"),
            make_comment("草", author="視聴者A", channel="chA"),
            make_comment("ああああああ", author="視聴者B", channel="chB"),
            make_comment("Character2ちゃん、今日の予定は？",
                         author="視聴者B", channel="chB"),
            make_comment("こんにちは！", author="視聴者A", channel="chA"),
        ]
        await run_ingest(orch, comments)

        # キューに入るのは拾った2件だけ（初コメと、名前呼び＋質問）
        queued = drain_messages(orch)
        assert sorted(queued) == sorted(
            ["こんにちは！", "Character2ちゃん、今日の予定は？"])

        # 落とした3件もアーカイブには記録される（記録と応答の分離）
        journal = next(tmp_path.glob("*.jsonl"))
        entries = [json.loads(line) for line in
                   journal.read_text(encoding="utf-8").splitlines() if line.strip()]
        recorded = [e["text"] for e in entries if e["type"] == "comment"]
        assert sorted(recorded) == sorted(["草", "ああああああ", "こんにちは！"])

        # 落としたコメントも視聴者プロフィールが更新される
        store = orch._viewer_store
        assert store.get_profile("視聴者A").visit_count == 2  # 草＋重複
        assert store.get_profile("視聴者B").visit_count == 1  # 連打

    async def test_disabled_filter_enqueues_everything(self):
        orch, _hub = make_orchestrator(StubLLM(RAW), StubTTS())
        orch._cfg.comment_filter.enabled = False
        comments = [
            make_comment("ああああああ", author="視聴者B", channel="chB"),
            make_comment("草", author="視聴者B", channel="chB"),
        ]
        await run_ingest(orch, comments)
        assert sorted(drain_messages(orch)) == sorted(["ああああああ", "草"])
