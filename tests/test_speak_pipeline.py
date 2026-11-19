"""発話パイプラインの結合テスト

外部サービス（LM Studio / TTSサーバ）に依存せず、
スタブバックエンドでOrchestratorの発話経路を検証する。
"""
import asyncio
import json

import pytest

from backend.llm.prompt_builder import _speech_limit
from backend.models import LLMResponse
from helpers import RAW, StubLLM, StubTTS, comment, make_orchestrator


# ── ストリーミング発話 ────────────────────────────────────────

@pytest.mark.asyncio
async def test_streaming_speaks_sentence_by_sentence():
    llm, tts = StubLLM(RAW), StubTTS()
    orch, hub = make_orchestrator(llm, tts)

    response = await orch._generate_and_speak("sys")

    assert llm.stream_calls == 1
    assert llm.generate_calls == 0                   # 一括生成にフォールバックしていない
    assert tts.calls == ["やっほー！", "今日も来てくれてありがとう。"]
    assert response.speech == "やっほー！今日も来てくれてありがとう。"
    assert response.emotion.value == "happy"
    assert response.motion.value == "wave"


@pytest.mark.asyncio
async def test_emotion_and_motion_sent_before_first_audio():
    """1文目の音声より前に表情とモーションがアバターへ届くこと"""
    llm, tts = StubLLM(RAW), StubTTS()
    orch, hub = make_orchestrator(llm, tts)
    await orch._generate_and_speak("sys")

    kinds = [k for k, _ in hub.events]
    first_motion = kinds.index("motion")
    first_audio = next(i for i, (k, d) in enumerate(hub.events)
                       if k == "speech_start" and d.get("audio_url"))
    assert first_motion < first_audio
    assert hub.events[first_motion][1]["motion"] == "wave"
    assert kinds[-1] == "speech_end"


@pytest.mark.asyncio
async def test_tts_requests_are_serialized():
    """ローカルTTSサーバを詰まらせないよう、合成は1本ずつ実行されること"""
    llm, tts = StubLLM(RAW), StubTTS(delay=0.05)
    orch, _ = make_orchestrator(llm, tts)
    await orch._generate_and_speak("sys")
    assert tts.max_concurrent == 1


@pytest.mark.asyncio
async def test_hung_tts_does_not_block_forever():
    """TTSがハングしても、タイムアウトして字幕だけで先に進むこと"""
    llm, tts = StubLLM(RAW), StubTTS(hang=True)
    orch, hub = make_orchestrator(llm, tts, tts_timeout=0.2)

    response = await asyncio.wait_for(orch._generate_and_speak("sys"), timeout=10)

    assert response is not None                       # 字幕としては発話できている
    assert response.speech == "やっほー！今日も来てくれてありがとう。"
    assert all(d.get("audio_url") is None
               for k, d in hub.events if k == "speech_start")


@pytest.mark.asyncio
async def test_filter_aborts_streaming_midway():
    """NGな内容が出たら、その文以降を喋らずに中断すること"""
    raw = '{"emotion":"neutral","motion":"idle","speech":"まず普通の話。電話は090-1234-5678だよ。"}'
    llm, tts = StubLLM(raw), StubTTS()
    orch, hub = make_orchestrator(llm, tts)

    response = await orch._generate_and_speak("sys")

    assert response is None
    assert tts.calls == ["まず普通の話。"]              # 2文目は合成すらしていない
    assert any(k == "log" and d["level"] == "warn" for k, d in hub.events)


@pytest.mark.asyncio
async def test_unity_action_tags_are_not_spoken():
    """アクションタグは読み上げず、除去した本文だけを喋ること"""
    raw = '{"emotion":"happy","motion":"idle","speech":"[action: wave] こんにちは！"}'
    llm, tts = StubLLM(raw), StubTTS()
    orch, _ = make_orchestrator(llm, tts)

    response = await orch._generate_and_speak("sys")

    assert tts.calls == ["こんにちは！"]
    assert "action" not in response.speech


@pytest.mark.asyncio
async def test_falls_back_to_batch_when_stream_yields_nothing():
    """ストリームが空でも、一括生成でリカバリすること"""
    llm, tts = StubLLM(""), StubTTS()
    llm._raw = ""
    orch, _ = make_orchestrator(llm, tts)
    # generate() 側だけは有効な応答を返すようにする
    llm.generate = lambda *a, **k: _immediate(LLMResponse(speech="復帰したよ。"))

    response = await orch._generate_and_speak("sys")

    assert response is not None
    assert response.speech == "復帰したよ。"
    assert tts.calls == ["復帰したよ。"]


def _immediate(value):
    async def _coro():
        return value
    return _coro()


@pytest.mark.asyncio
async def test_non_streaming_backend_still_works():
    """ストリーミング非対応バックエンドでも従来どおり発話できること"""
    llm, tts = StubLLM(RAW, streaming=False), StubTTS()
    orch, hub = make_orchestrator(llm, tts)

    response = await orch._generate_and_speak("sys")

    assert llm.stream_calls == 0
    assert llm.generate_calls == 1
    assert tts.calls == ["やっほー！今日も来てくれてありがとう。"]   # 一括で1回
    assert response.speech == "やっほー！今日も来てくれてありがとう。"


# ── コメントキューの優先度 ────────────────────────────────────

@pytest.mark.asyncio
async def test_high_priority_comment_evicts_low_priority_when_full():
    """キューが一杯でも、スパチャや音声入力は雑談を押しのけて入れること"""
    orch, _ = make_orchestrator(StubLLM(RAW), StubTTS(), max_queue_size=3)

    for i in range(3):
        assert await orch.enqueue_comment(comment(f"雑談{i}"), score=2.0)
    assert orch._comment_queue.qsize() == 3

    assert await orch.enqueue_comment(comment("スパチャ！"), score=9.0) is True
    assert orch._comment_queue.qsize() == 3

    items = []
    while not orch._comment_queue.empty():
        items.append(orch._comment_queue.get_nowait())
    messages = [i.comment.message for i in items]
    assert "スパチャ！" in messages
    assert messages[0] == "スパチャ！"          # 最優先で取り出される
    assert len(messages) == 3


@pytest.mark.asyncio
async def test_low_priority_comment_dropped_when_full():
    """一杯のときに同等以下の優先度なら、新規側を捨てて既存を守ること"""
    orch, _ = make_orchestrator(StubLLM(RAW), StubTTS(), max_queue_size=2)
    await orch.enqueue_comment(comment("先客1"), score=5.0)
    await orch.enqueue_comment(comment("先客2"), score=5.0)

    assert await orch.enqueue_comment(comment("後から来た雑談"), score=2.0) is False
    assert orch._comment_queue.qsize() == 2
# ── Phase 1: 思考リーク・断片・上限・即時フェイルオーバー ────
# 2026-11-17 の1時間テストで実際に発生した問題の回帰テスト。

# Character2の生推論がそのまま発話化した件（06:22:13）の再構成
THOUGHT_RAW = (
    "<thought\nThinking Process:\n"
    "1. **Input Analysis**:\n"
    " * User Input: `<partner_speech name=\"Character1\">やっほー！</partner_speech>`"
    " への返答を考える\n"
    "2. **Response Strategy**: 短く相づちを返す"
)


def _json_raw(speech: str) -> str:
    return json.dumps(
        {"emotion": "neutral", "motion": "idle", "skip": False,
         "speech": speech},
        ensure_ascii=False)


class DeadLLM(StubLLM):
    """接続エラーで不健全になったバックエンドの再現（何も生成できない）"""

    @property
    def is_healthy(self):
        return False

    async def generate(self, system_prompt, messages, *, route="respond", temperature=None):
        self.generate_calls += 1
        return LLMResponse(speech="", skip=True)

    async def generate_stream(self, system_prompt, messages, *, route="respond", temperature=None):
        self.stream_calls += 1
        return
        yield ""  # async generator にするためのダミー


@pytest.mark.asyncio
async def test_thought_output_is_never_spoken():
    """生推論の塊は、ストリーミングでも一括でも喋らせないこと"""
    llm, tts = StubLLM(THOUGHT_RAW), StubTTS()
    orch, _ = make_orchestrator(llm, tts)

    response = await orch._generate_and_speak("sys")

    assert response is None
    assert tts.calls == []


@pytest.mark.asyncio
async def test_fragment_only_output_is_never_spoken():
    """「あ、」だけの応答は喋らせないこと（06:55:52 の回帰）"""
    llm, tts = StubLLM(_json_raw("あ、")), StubTTS()
    orch, _ = make_orchestrator(llm, tts)

    response = await orch._generate_and_speak("sys")

    assert response is None
    assert tts.calls == []


@pytest.mark.asyncio
async def test_unfinished_json_fragment_is_never_spoken():
    """未完JSONの「あらら」は喋らせないこと（06:41:08 の回帰）"""
    raw = '{"emotion":"neutral","motion":"idle","skip":false,"speech":"あらら'
    llm, tts = StubLLM(raw), StubTTS()
    orch, _ = make_orchestrator(llm, tts)

    response = await orch._generate_and_speak("sys")

    assert response is None
    assert tts.calls == []


@pytest.mark.asyncio
async def test_streaming_speech_limit_truncates_at_sentence_boundary():
    """上限を超える文は喋らず、応答を文境界で終わらせること"""
    llm, tts = StubLLM(""), StubTTS()
    orch, _ = make_orchestrator(llm, tts)
    limit = _speech_limit(orch._lead.cfg)
    s1 = "あ" * (limit - 10) + "。"
    s2 = "い" * 20 + "。"
    llm._raw = _json_raw(s1 + s2)

    response = await orch._generate_and_speak("sys")

    assert response is not None
    assert response.speech == s1
    assert tts.calls == [s1]


@pytest.mark.asyncio
async def test_non_streaming_truncates_at_sentence_boundary():
    """一括生成でも上限超過分は文境界で切り捨てること"""
    llm, tts = StubLLM("", streaming=False), StubTTS()
    orch, _ = make_orchestrator(llm, tts)
    limit = _speech_limit(orch._lead.cfg)
    s1 = "あ" * (limit - 10) + "。"
    s2 = "い" * 20 + "。"
    llm._raw = _json_raw(s1 + s2)

    response = await orch._generate_and_speak("sys")

    assert response is not None
    assert response.speech == s1
    assert tts.calls == [s1]


@pytest.mark.asyncio
async def test_non_streaming_single_sentence_over_limit_is_dropped():
    """1文だけの応答が上限を超える場合は、文の途中で切らず丸ごと落とすこと"""
    llm, tts = StubLLM("", streaming=False), StubTTS()
    orch, _ = make_orchestrator(llm, tts)
    limit = _speech_limit(orch._lead.cfg)
    llm._raw = _json_raw("あ" * (limit + 10) + "。")

    response = await orch._generate_and_speak("sys")

    assert response is None
    assert tts.calls == []


@pytest.mark.asyncio
async def test_immediate_failover_on_double_failure():
    """ストリーミングも一括も失敗しバックエンドが不健全なら、その場で
    フェイルオーバーすること（90秒周期のヘルスチェックを待たない）"""
    llm, tts = DeadLLM(""), StubTTS()
    orch, _ = make_orchestrator(llm, tts)
    fallback = StubLLM(RAW)
    orch._lead.llm_fallback = fallback

    response = await orch._generate_and_speak("sys")

    assert response is None
    assert orch._lead.llm is fallback          # 入れ替わった
    assert orch._lead.llm_fallback is llm      # 元の primary は fallback 側へ


@pytest.mark.asyncio
async def test_no_immediate_failover_when_backend_healthy():
    """モデルが空を返しただけ（バックエンドは健全）なら切替えないこと"""
    llm, tts = StubLLM(""), StubTTS()
    orch, _ = make_orchestrator(llm, tts)
    fallback = StubLLM(RAW)
    orch._lead.llm_fallback = fallback

    response = await orch._generate_and_speak("sys")

    assert response is None
    assert orch._lead.llm is llm               # 入れ替わっていない


@pytest.mark.asyncio
async def test_immediate_failover_has_cooldown():
    """即時フェイルオーバーはクールダウン中は再発しないこと（flapping防止）"""
    llm, tts = DeadLLM(""), StubTTS()
    orch, _ = make_orchestrator(llm, tts)
    orch._lead.llm_fallback = StubLLM(RAW)

    await orch._generate_and_speak("sys")
    assert orch._lead.llm is not llm           # 1回目は切り替わる

    # 切り替わった先も死んだ想定にしても、クールダウン中は戻さない
    dead2 = DeadLLM("")
    orch._lead.llm = dead2
    orch._lead.llm_fallback = StubLLM(RAW)
    response = await orch._generate_and_speak("sys")

    assert response is None
    assert orch._lead.llm is dead2             # 入れ替わっていない
