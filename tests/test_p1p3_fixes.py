"""P1〜P3 修正（2026-11-17 1時間テスト再実施レポートの残課題）の回帰テスト。

- P3-A: TTS ウォームアップ（開始時に両キャラの声を温める）
- P3-B: 序盤 TTS 保護（開始直後はタイムアウト延長＋1回リトライ）
- P1-B: 反復ブロックの段階化（短い汎用句は短時間に3回目で初めてブロック）
- P2-A: 定型相づちクールダウン（型が1回出たら10分間は再演を差し止め）
- P2-B: プロンプトへ直近の自分の発話を明示（繰り返し回避）
- P1-D: 救済リトライ時のプロンプト縮小（compact retry）
- P1-C: アイドル発話の先行生成（プリフェッチ）
"""
from __future__ import annotations

import asyncio
import json
import tempfile
import time
from pathlib import Path

import yaml

from backend.config import AppConfig, CharacterConfig, EnsembleConfig
from backend.knowledge.models import WebDocument
from backend.knowledge.topic_planner import Topic
from backend.llm.prompt_builder import build_idle_prompt
from backend.models import LLMResponse, TTSResult
from backend.orchestrator import Orchestrator
from helpers import RAW, FakeHub, NoAudio, StubLLM, StubTTS, make_orchestrator

AIZUCHI = "Character1ちゃん、それはどういうことなの？ わたしにも教えて？"
SHORT_PHRASE = "ありがとうございます。"
NORMAL_A = "新しい話題だよ。今日は天気の話をしよう。"
NORMAL_B = "Character2の新しい話題だよ。別の切り口で話すね。"
LONG_REPEAT = "調査の結果を報告するね。今日のニュースを詳しく調べてみたよ。"


def _json_raw(speech: str) -> str:
    return json.dumps({"emotion": "happy", "motion": "wave",
                       "speech": speech}, ensure_ascii=False)


def _write_char(dirpath: Path, name: str) -> str:
    path = dirpath / f"{name}.yaml"
    path.write_text(yaml.safe_dump(
        {"name": name, "description": f"{name}の説明",
         "personality": f"{name}の口調"}, allow_unicode=True), encoding="utf-8")
    return str(path)


class ScriptedLLM(StubLLM):
    """呼び出しごとに返す原稿を切り替えるLLM（最後の原稿を使い続ける）。"""

    def __init__(self, raws: list[str]):
        super().__init__(raws[0])
        self._raws = list(raws)
        self.prompts: list[str] = []

    async def generate_stream(self, system_prompt, messages, *,
                              route="respond", temperature=None):
        self.prompts.append(system_prompt)
        idx = min(len(self.prompts) - 1, len(self._raws) - 1)
        self._raw = self._raws[idx]
        async for chunk in super().generate_stream(
                system_prompt, messages, route=route):
            yield chunk


class RecordingLLM(StubLLM):
    """generate / generate_stream の両方の呼び出しを記録する LLM。"""

    def __init__(self, raws: list[str]):
        super().__init__(raws[0])
        self._raws = list(raws)
        self.calls: list[tuple[str, str, list]] = []

    def _pick(self) -> str:
        idx = min(len(self.calls) - 1, len(self._raws) - 1)
        return self._raws[idx]

    async def generate(self, system_prompt, messages, *,
                       route="respond", temperature=None):
        self.calls.append(("generate", system_prompt, list(messages)))
        self._raw = self._pick()
        return await super().generate(
            system_prompt, messages, route=route, temperature=temperature)

    async def generate_stream(self, system_prompt, messages, *,
                              route="respond", temperature=None):
        self.calls.append(("stream", system_prompt, list(messages)))
        self._raw = self._pick()
        async for chunk in super().generate_stream(
                system_prompt, messages, route=route):
            yield chunk


class FlakyTTS:
    """最初の N 回はハング（タイムアウト級）し、その後は即返す。"""

    def __init__(self, hang_times: int = 1):
        self.calls: list[str] = []
        self._hang = hang_times

    async def synthesize(self, text: str, emotion: str | None = None) -> TTSResult:
        self.calls.append(text)
        if len(self.calls) <= self._hang:
            await asyncio.sleep(3600)
        return TTSResult(audio_wav=b"RIFF....", duration_ms=100)


class RaisingTTS(StubTTS):
    async def synthesize(self, text, emotion=None):
        self.calls.append(text)
        raise RuntimeError("tts down")


class FakePlannerEx:
    """has_deliverable_answer を備えたフェイクプランナ。"""

    def __init__(self, deliverable: bool = False):
        self.plan_calls: list[dict] = []
        self.committed: list = []
        self.released: list = []
        self._deliverable = deliverable
        self.last_comment_speech_at = 0.0

    async def plan(self, kind: str = "", query: str = "",
                   allow_web: bool = True, context=None) -> Topic:
        self.plan_calls.append({"kind": kind, "query": query})
        doc = WebDocument(title="テスト記事", url="https://example.com/1",
                          snippet="テスト用の資料テキストです。")
        return Topic(kind=kind or "news", label="テスト話題",
                     documents=[doc],
                     topic_id=f"fake-{len(self.plan_calls)}")

    def commit(self, topic) -> None:
        self.committed.append(topic)

    def release(self, topic) -> None:
        self.released.append(topic)

    def ban_word(self, word: str, seconds: int = 1800) -> None:
        pass

    def banned_words(self) -> set:
        return set()

    def reset_thread(self) -> None:
        pass

    def note_speech(self, idle: bool = False, text: str = "") -> None:
        pass

    def note_comment(self, author: str, text: str) -> None:
        pass

    def has_deliverable_answer(self) -> bool:
        return self._deliverable


def make_duo_p1p3(llm_a=None, llm_b=None, fallback_a=(), fallback_b=(),
                  planner=None, **over):
    """掛け合いなし（確率0）の2体オーケストレータ。over は orchestrator 設定。"""
    tmp = Path(tempfile.mkdtemp(prefix="aivt_p1p3_"))
    cfg = AppConfig()
    cfg.stt.enabled = False
    cfg.ollama.auto_start = False
    cfg.ollama.preload_on_start = False
    cfg.tts.irodori_server.auto_start = False
    cfg.knowledge.enabled = False
    cfg.characters = [
        CharacterConfig(name="Character1", id="Character1", data_dir=str(tmp),
                        prompt_file=_write_char(tmp, "Character1"),
                        aliases=["character1"], fallback_lines=list(fallback_a)),
        CharacterConfig(name="Character2", id="Character2", data_dir=str(tmp),
                        prompt_file=_write_char(tmp, "Character2"),
                        fallback_lines=list(fallback_b)),
    ]
    cfg.ensemble = EnsembleConfig(turn_gap_sec=0, idle_banter_probability=0.0,
                                  banter_probability=0.0)
    for key, value in over.items():
        setattr(cfg.orchestrator, key, value)
    hub = FakeHub()
    orch = Orchestrator(cfg, hub)
    if planner is not None:
        orch._planner = planner
    tts_a, tts_b = StubTTS(), StubTTS()
    a, b = orch.agents
    a.llm, a.tts = llm_a or StubLLM(_json_raw(NORMAL_A)), tts_a
    b.llm, b.tts = llm_b or StubLLM(_json_raw(NORMAL_B)), tts_b
    orch._audio = NoAudio()
    orch._audio_cache = Path(tempfile.mkdtemp(prefix="aivt_audio_"))
    return orch, hub, tts_a, tts_b


# ---------------------------------------------------------------- P3-A

async def test_p3a_warmup_synthesizes_for_all_characters():
    orch, hub, tts_a, tts_b = make_duo_p1p3()
    await orch._warmup_tts_voices()
    assert tts_a.calls == [Orchestrator._TTS_WARMUP_LINE]
    assert tts_b.calls == [Orchestrator._TTS_WARMUP_LINE]


async def test_p3a_warmup_failure_does_not_raise():
    orch, hub, tts_a, tts_b = make_duo_p1p3()
    orch.agents[0].tts = RaisingTTS()
    orch.agents[1].tts = RaisingTTS()
    await orch._warmup_tts_voices()  # 例外なく戻ること（開始を止めない）


# ---------------------------------------------------------------- P3-B

async def test_p3b_early_window_retries_once():
    tts = FlakyTTS(hang_times=1)
    orch, hub = make_orchestrator(StubLLM(RAW), tts, tts_timeout=0.05)
    orch._session_start_mono = time.monotonic()
    orch._TTS_EARLY_TIMEOUT_SEC = 0.2
    result = await orch._synthesize("テストです", "neutral")
    assert result.audio_wav  # リトライで音声が取れた
    assert len(tts.calls) == 2
    assert orch._tts_timeouts == 0  # 最終的に成功したので数えない


async def test_p3b_outside_window_no_retry():
    tts = FlakyTTS(hang_times=1)
    orch, hub = make_orchestrator(StubLLM(RAW), tts, tts_timeout=0.05)
    result = await orch._synthesize("テストです", "neutral")
    assert not result.audio_wav
    assert len(tts.calls) == 1  # 序盤窓の外では従来どおりリトライしない
    assert orch._tts_timeouts == 1


# ---------------------------------------------------------------- P1-B

async def test_p1b_short_phrase_tiered_blocking():
    orch, hub = make_orchestrator(StubLLM(RAW), StubTTS())
    norm_long = Orchestrator._normalize_speech(LONG_REPEAT)
    orch._recent_speech_norms.append(norm_long)
    assert orch._should_block_repeat(LONG_REPEAT) is True  # 長文は即ブロック

    norm_short = Orchestrator._normalize_speech(SHORT_PHRASE)
    orch._recent_speech_norms.append(norm_short)
    assert orch._should_block_repeat(SHORT_PHRASE) is False  # 2回目相当は許可
    now = time.time()
    orch._short_phrase_events.append((now, norm_short))
    assert orch._should_block_repeat(SHORT_PHRASE) is False  # なお許可
    orch._short_phrase_events.append((now, norm_short))
    assert orch._should_block_repeat(SHORT_PHRASE) is True  # 3回目相当でブロック
    orch._short_phrase_events = [(now - 1000.0, norm_short)] * 3
    assert orch._should_block_repeat(SHORT_PHRASE) is False  # 窓外は数えない
    assert orch._should_block_repeat("まだ言っていない新しい発話です。") is False


async def test_p1b_short_phrase_third_occurrence_blocked():
    raw = _json_raw(SHORT_PHRASE)
    orch, hub, tts_a, tts_b = make_duo_p1p3(
        llm_a=ScriptedLLM([raw]), llm_b=ScriptedLLM([raw]))
    await orch._speak_idle()
    await orch._speak_idle()
    # 汎用句の2回目まではブロックされず、そのまま話せる
    assert tts_a.calls + tts_b.calls == [SHORT_PHRASE, SHORT_PHRASE]
    assert orch._total_block_streak == 0
    await orch._speak_idle()
    # 短時間に3回目 → ブロックされ、音声化されない
    assert tts_a.calls + tts_b.calls == [SHORT_PHRASE, SHORT_PHRASE]
    assert orch._total_block_streak >= 1


# ---------------------------------------------------------------- P2-A

async def test_p2a_pattern_detection_and_cooldown():
    orch, hub = make_orchestrator(StubLLM(RAW), StubTTS())
    assert orch._aizuchi_pattern_of(AIZUCHI) != ""
    assert orch._aizuchi_pattern_of(NORMAL_A) == ""
    assert orch._aizuchi_in_cooldown(AIZUCHI) == ""
    until = time.time() + 600
    for pid, _pat in Orchestrator._AIZUCHI_PATTERNS:
        orch._aizuchi_cooldown_until[pid] = until
    assert orch._aizuchi_in_cooldown(AIZUCHI) == "douiu_koto"
    # 文単位で届く前半・後半だけでも拾える
    assert orch._aizuchi_in_cooldown("Character1ちゃん、それはどういうことなの？") \
        == "douiu_stem"
    assert orch._aizuchi_in_cooldown("わたしにも教えて？") == "watashi_mo"
    assert orch._aizuchi_in_cooldown(NORMAL_A) == ""


async def test_p2a_reperformance_diverted_to_recovery():
    raw_a = _json_raw(AIZUCHI)
    llm_a = ScriptedLLM([raw_a, raw_a, _json_raw(NORMAL_A)])
    llm_b = ScriptedLLM([raw_a, _json_raw(NORMAL_B)])
    orch, hub, tts_a, tts_b = make_duo_p1p3(llm_a=llm_a, llm_b=llm_b)
    await orch._speak_idle()
    # 初回は話せる（一人称はペルソナ補正で変わり得るので後半で数える）
    assert sum(c.count("にも教えて？") for c in tts_a.calls + tts_b.calls) == 1
    await orch._speak_idle()
    all_tts = tts_a.calls + tts_b.calls
    assert sum(c.count("にも教えて？") for c in all_tts) == 1  # 再演なし
    joined = "".join(all_tts)  # ストリーミングは文単位で合成される
    assert (NORMAL_A in joined) or (NORMAL_B in joined)  # 回復生成が話された
    assert orch._total_block_streak == 0  # ブロックではなく差し止め扱い


# ---------------------------------------------------------------- P2-B

async def test_p2b_recent_own_speeches_in_prompt_kwargs():
    orch, hub, tts_a, tts_b = make_duo_p1p3()
    agent = orch.agents[0]
    assert agent.recent_own_speeches() == []
    agent.log.add_idle("さっき話した内容だよ。")
    agent.log.add_idle("もう一つ前の発話だよ。")
    assert agent.recent_own_speeches() == [
        "さっき話した内容だよ。", "もう一つ前の発話だよ。"]
    kwargs = agent.prompt_kwargs()
    assert kwargs["recent_own_speeches"] == [
        "さっき話した内容だよ。", "もう一つ前の発話だよ。"]


def test_p2b_prompt_section_rendering():
    cfg = AppConfig()
    p = build_idle_prompt(cfg, recent_own_speeches=["さっき話した内容だよ。"])
    assert "繰り返しの回避" in p
    assert "さっき話した内容だよ。" in p
    p2 = build_idle_prompt(cfg)
    assert "繰り返しの回避" not in p2


# ---------------------------------------------------------------- P1-D

async def test_p1d_compact_retry_trims_context():
    llm_a = RecordingLLM([_json_raw(LONG_REPEAT), _json_raw(NORMAL_A)])
    orch, hub, tts_a, tts_b = make_duo_p1p3(llm_a=llm_a)
    agent = orch.agents[0]
    # 既定の履歴窓（maxlen）が小さいと切り詰め効果が見えないため広げる
    from collections import deque
    agent.log._history = deque(maxlen=100)
    for i in range(6):
        agent.log.add_idle(f"過去の発話その{i}です。")
    norm = Orchestrator._normalize_speech(LONG_REPEAT)
    orch._recent_speech_norms.append(norm)
    await orch._speak_idle()
    stream_calls = [c for c in llm_a.calls if c[0] == "stream"]
    assert len(stream_calls) == 2
    assert len(stream_calls[0][2]) >= 6  # 初回は全履歴
    assert len(stream_calls[1][2]) == Orchestrator._COMPACT_RETRY_MESSAGES
    assert "再生成（短縮）" in stream_calls[1][1]
    assert NORMAL_A in "".join(tts_a.calls)


# ---------------------------------------------------------------- P1-C

async def test_p1c_prefetch_produce_and_consume():
    llm_a = RecordingLLM([_json_raw(NORMAL_A)])
    planner = FakePlannerEx()
    orch, hub, tts_a, tts_b = make_duo_p1p3(llm_a=llm_a, planner=planner)
    orch._running = True
    await orch._prefetch_idle()
    pf = orch._prefetched
    assert pf is not None and pf["response"].speech == NORMAL_A
    assert any(c[0] == "generate" for c in llm_a.calls)
    orch._running = False
    calls_before = len(llm_a.calls)
    await orch._speak_idle()
    assert NORMAL_A in tts_a.calls  # 先行分がそのまま話された
    assert len(llm_a.calls) == calls_before  # 消費時に追加生成なし
    assert len(planner.committed) == 1


async def test_p1c_maybe_start_prefetch_creates_task():
    llm_a = RecordingLLM([_json_raw(NORMAL_A)])
    orch, hub, tts_a, tts_b = make_duo_p1p3(llm_a=llm_a, planner=FakePlannerEx())
    orch._running = True
    orch._maybe_start_prefetch()
    assert orch._prefetch_task is not None
    await orch._prefetch_task
    assert orch._prefetched is not None
    orch._running = False


async def test_p1c_prefetch_stale_is_discarded():
    planner = FakePlannerEx()
    orch, hub, tts_a, tts_b = make_duo_p1p3(
        llm_a=ScriptedLLM([_json_raw(NORMAL_A)]), planner=planner)
    topic = await planner.plan(kind="news")
    orch._prefetched = {
        "at": time.time() - Orchestrator._PREFETCH_TTL_SEC - 60,
        "agent_key": orch.agents[0].name,
        "topic": topic,
        "response": LLMResponse(speech="古い先行発話なので使われないはずです。"),
        "actions": [],
    }
    await orch._speak_idle()
    assert "古い先行発話なので使われないはずです。" not in (tts_a.calls + tts_b.calls)
    assert topic in planner.released  # 破棄時に話題を解放する
    assert NORMAL_A in "".join(tts_a.calls + tts_b.calls)  # 通常生成へ


async def test_p1c_prefetch_discarded_when_answer_pending():
    planner = FakePlannerEx(deliverable=True)
    orch, hub, tts_a, tts_b = make_duo_p1p3(
        llm_a=ScriptedLLM([_json_raw(NORMAL_A)]), planner=planner)
    topic = await planner.plan(kind="news")
    orch._prefetched = {
        "at": time.time(),
        "agent_key": orch.agents[0].name,
        "topic": topic,
        "response": LLMResponse(speech="先行発話ですが保留回答が優先されます。"),
        "actions": [],
    }
    await orch._speak_idle()
    assert "先行発話ですが保留回答が優先されます。" not in (tts_a.calls + tts_b.calls)
    assert topic in planner.released


async def test_p1c_prefetch_disabled_by_config():
    orch, hub, tts_a, tts_b = make_duo_p1p3(prefetch_idle=False)
    orch._running = True
    orch._maybe_start_prefetch()
    assert orch._prefetch_task is None
