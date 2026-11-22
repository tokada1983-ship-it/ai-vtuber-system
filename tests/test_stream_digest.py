"""配信中の記憶整理（2026-11-22 longstream）の回帰テスト

長時間配信で膨らむ配信中の記憶への対策:
  - 「ここまでの流れ」ダイジェストの更新（消化済み範囲の二重取り込み防止、
    失敗時は番兵を進めず次回再試行、例外を外へ投げない）
  - 配信中に増えたエピソード（web/auto）の圧縮（直近と保護対象は残し、
    過去の記憶・固定記憶には触れない。要約が成立しない場合は無削除）

LLM と記憶ストアはスタブ／一時 DB で検証する（実リクエストは行わない）。
"""
from __future__ import annotations

import time
from types import SimpleNamespace

from backend.memory.memory_store import MemoryStore
from backend.memory.stream_digest import StreamDigest
from backend.models import ChatMessage

SUMMARY = "これは配信中のまとめ要約です。テスト用の十分な長さの文章になっています。"


class FakeLLM:
    """LLMBackend のスタブ（_generate_raw の非ストリーミング経路）"""

    supports_streaming = False

    def __init__(self, text=SUMMARY, fail=False):
        self._text = text
        self._fail = fail
        self.calls = 0

    async def generate(self, system_prompt, messages, route="idle"):
        self.calls += 1
        if self._fail:
            raise RuntimeError("fake llm failure")
        return SimpleNamespace(raw=None, speech=self._text)


def msgs(*texts):
    return [ChatMessage(role="assistant", content=t) for t in texts]


def make_store(tmp_path) -> MemoryStore:
    return MemoryStore(db_path=str(tmp_path / "memory.db"))


def add_episodes(store, n, source="web"):
    for i in range(n):
        store.add(f"配信中の記憶その{i}番目の内容です", source=source,
                  dedupe=False)


def session_entries(store, since):
    return [e for e in store.all()
            if e.created_at >= since and e.source in ("web", "auto")]


# ── 流れダイジェスト ────────────────────────────────────────

class TestDigestUpdate:
    async def test_digest_updated_from_messages(self, tmp_path):
        keeper = StreamDigest(session_started_at=time.time())
        llm = FakeLLM("序盤はロボットの話をして、次に祭りの話をした。")
        result = await keeper.update(
            messages=msgs("ロボットの話をした", "祭りの話もした"),
            llm=llm, store=None, character_name="test")
        assert result["digest_updated"] is True
        assert "ロボット" in keeper.text
        assert llm.calls == 1

    async def test_same_batch_not_digested_twice(self, tmp_path):
        keeper = StreamDigest(session_started_at=time.time())
        llm = FakeLLM("ダイジェスト本文です。十分な長さがあります。")
        batch = msgs("あれをした", "これもした")
        await keeper.update(messages=batch, llm=llm, store=None)
        result = await keeper.update(messages=batch, llm=llm, store=None)
        assert result["digest_updated"] is False
        assert llm.calls == 1

    async def test_incremental_batch_only_new_messages(self, tmp_path):
        keeper = StreamDigest(session_started_at=time.time())
        llm = FakeLLM("ダイジェスト本文です。十分な長さがあります。")
        await keeper.update(messages=msgs("一つ目"), llm=llm, store=None)
        result = await keeper.update(
            messages=msgs("一つ目", "二つ目"), llm=llm, store=None)
        assert result["digest_updated"] is True
        assert llm.calls == 2

    async def test_failure_does_not_raise_and_retries_next_time(self, tmp_path):
        keeper = StreamDigest(session_started_at=time.time())
        bad = FakeLLM(fail=True)
        result = await keeper.update(messages=msgs("何か話した"),
                                     llm=bad, store=None)
        assert result["digest_updated"] is False
        assert keeper.text == ""
        good = FakeLLM("復旧後のダイジェストです。長さも十分あります。")
        result = await keeper.update(messages=msgs("何か話した"),
                                     llm=good, store=None)
        assert result["digest_updated"] is True

    async def test_empty_generation_keeps_sentinel(self, tmp_path):
        keeper = StreamDigest(session_started_at=time.time())
        llm = FakeLLM("")
        await keeper.update(messages=msgs("何か話した"), llm=llm, store=None)
        await keeper.update(messages=msgs("何か話した"), llm=llm, store=None)
        # 空応答では番兵を進めないため、次回も同じバッチで再試行される
        assert llm.calls == 2
        assert keeper.text == ""


# ── 配信中エピソードの圧縮 ──────────────────────────────────

class TestEpisodeCompaction:
    async def test_compaction_reduces_session_episodes(self, tmp_path):
        store = make_store(tmp_path)
        start = time.time() - 10
        add_episodes(store, 10)
        keeper = StreamDigest(episode_cap=3, episode_keep_recent=2,
                              episode_protect_sec=0.0,
                              session_started_at=start)
        result = await keeper.update(messages=[], llm=FakeLLM(),
                                     store=store)
        assert result["compacted"] == 8
        remaining = session_entries(store, start)
        assert len(remaining) == 2          # 直近2件は原文のまま残る
        digests = [e for e in store.all() if e.source == "stream_digest"]
        assert len(digests) == 1            # 古い8件は要約1件に置き換わる
        assert store.count() == 3

    async def test_no_compaction_under_cap(self, tmp_path):
        store = make_store(tmp_path)
        start = time.time() - 10
        add_episodes(store, 3)
        keeper = StreamDigest(episode_cap=40, episode_keep_recent=15,
                              session_started_at=start)
        result = await keeper.update(messages=[], llm=FakeLLM(),
                                     store=store)
        assert result["compacted"] == 0
        assert store.count() == 3

    async def test_pre_session_and_pinned_untouched(self, tmp_path):
        store = make_store(tmp_path)
        add_episodes(store, 6)
        pinned = store.add("固定された大事な記憶です", source="web",
                           dedupe=False)
        store.set_pinned(pinned.id, True)
        # セッション開始を未来に置くと、全件が「開始前」の記憶になる
        keeper = StreamDigest(episode_cap=1, episode_keep_recent=0,
                              episode_protect_sec=0.0,
                              session_started_at=time.time() + 1000)
        result = await keeper.update(messages=[], llm=FakeLLM(),
                                     store=store)
        assert result["compacted"] == 0
        assert store.count() == 7

    async def test_pinned_not_compacted_within_session(self, tmp_path):
        store = make_store(tmp_path)
        start = time.time() - 10
        add_episodes(store, 5)
        pinned = store.add("固定された大事な記憶です", source="web",
                           dedupe=False)
        store.set_pinned(pinned.id, True)
        keeper = StreamDigest(episode_cap=2, episode_keep_recent=0,
                              episode_protect_sec=0.0,
                              session_started_at=start)
        await keeper.update(messages=[], llm=FakeLLM(), store=store)
        texts = [e.text for e in store.all()]
        assert any("固定された大事な記憶" in t for t in texts)
        assert store.count() == 3           # 原文1件＋要約1件＋固定1件

    async def test_bad_summary_aborts_without_deletion(self, tmp_path):
        store = make_store(tmp_path)
        start = time.time() - 10
        add_episodes(store, 6)
        keeper = StreamDigest(episode_cap=2, episode_keep_recent=0,
                              episode_protect_sec=0.0,
                              session_started_at=start)
        result = await keeper.update(messages=[], llm=FakeLLM("短い"),
                                     store=store)
        # 要約として成立しない出力（20字未満）では原典を消さない
        assert result["compacted"] == 0
        assert store.count() == 6

    async def test_deferred_when_too_many_protected(self, tmp_path):
        store = make_store(tmp_path)
        start = time.time() - 10
        add_episodes(store, 6)
        # 直近5件を無条件保持すると、必要件数に届かず見送りになる
        keeper = StreamDigest(episode_cap=2, episode_keep_recent=5,
                              episode_protect_sec=0.0,
                              session_started_at=start)
        result = await keeper.update(messages=[], llm=FakeLLM(),
                                     store=store)
        assert result["compacted"] == 0
        assert store.count() == 6
