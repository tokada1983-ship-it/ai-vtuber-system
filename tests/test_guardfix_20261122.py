"""2026-11-22 生成ガード改修（guardfix）のテスト

対象は BFラン（2026-11-21）ログ解析の問題1・2・4・5・6:
- 問題1: Ollama の生成レイテンシ常時ログ／probe の実生成化／
  ストリーム失敗（stream_error）の表面化
- 問題2: 言語混入チェック、日記の品質ゲート、プリフェッチ消費直前の再検査
- 問題4: 文脈リセットの共有クールダウン
- 問題5: 定型相づち同一文カウンタのターン跨ぎ保持（時間窓）
- 問題6: リセット前プリフェッチの破棄、curiosity 検索の時間上限
"""
from __future__ import annotations

import asyncio
import json
import tempfile
import time
from pathlib import Path

import httpx
import pytest
from structlog.testing import capture_logs

from backend.config import AppConfig, CharacterConfig, EnsembleConfig, LLMBackendConfig
from backend.knowledge.topic_planner import TopicPlanner
from backend.llm.ollama_backend import OllamaBackend
from backend.llm.parser import foreign_contamination_marker
from backend.memory.conversation_log import ConversationLog
from backend.memory.long_term_memory import LongTermMemory
from backend.memory.reflection_engine import ReflectionEngine
from backend.models import LLMResponse
from backend.orchestrator import Orchestrator

from helpers import FakeHub, NoAudio, StubLLM, StubTTS


def _write_char(tmp: Path, name: str) -> str:
    p = tmp / f"{name}.md"
    p.write_text(f"あなたは{name}です。", encoding="utf-8")
    return str(p)


def make_duo():
    tmp = Path(tempfile.mkdtemp(prefix="aivt_guardfix_"))
    cfg = AppConfig()
    cfg.stt.enabled = False
    cfg.ollama.auto_start = False
    cfg.ollama.preload_on_start = False
    cfg.tts.irodori_server.auto_start = False
    cfg.knowledge.enabled = False
    cfg.characters = [
        CharacterConfig(name="クロエ", id="chloe", data_dir=str(tmp),
                        prompt_file=_write_char(tmp, "クロエ"),
                        aliases=["くろえ"]),
        CharacterConfig(name="ミライ", id="mirai", data_dir=str(tmp),
                        prompt_file=_write_char(tmp, "ミライ")),
    ]
    cfg.ensemble = EnsembleConfig(
        turn_gap_sec=0, idle_banter_probability=0.0,
        banter_probability=0.0)
    hub = FakeHub()
    orch = Orchestrator(cfg, hub)
    a, b = orch.agents
    a.llm, a.tts = StubLLM("{}"), StubTTS()
    b.llm, b.tts = StubLLM("{}"), StubTTS()
    orch._audio = NoAudio()
    orch._audio_cache = Path(tempfile.mkdtemp(prefix="aivt_audio_"))
    return orch, hub


# ── 問題2: 言語混入マーカー ──────────────────────────────────

def test_foreign_marker_detects_chinese_contamination():
    assert foreign_contamination_marker("待ち合ってるから啦，もう早く") == "啦"
    assert foreign_contamination_marker("これはテストです我们") == "们"
    assert foreign_contamination_marker("全角コンニング，これも") == "，"


def test_foreign_marker_ignores_plain_japanese():
    assert foreign_contamination_marker("今日は良い天気ですね。掃除を頑張ります！") == ""
    assert foreign_contamination_marker("読書と日本語の文章です。") == ""
    assert foreign_contamination_marker("") == ""


# ── 問題4: 文脈リセットの共有クールダウン ────────────────────

def test_context_reset_shared_cooldown():
    orch, _hub = make_duo()
    with capture_logs() as logs:
        orch._reset_saturated_context("語1", reason="saturation")
    hits = [e for e in logs if e.get("event") == "context reset (saturation)"]
    assert len(hits) == 1
    first_at = orch._last_context_reset_at
    assert first_at > 0

    # 直後の別駆動元（loop_guard）のリセットは抑止される
    with capture_logs() as logs2:
        orch._reset_saturated_context("語2", reason="loop_guard")
    hits2 = [e for e in logs2 if e.get("event") == "context reset (saturation)"]
    suppressed = [e for e in logs2
                  if e.get("event") == "context reset suppressed (cooldown)"]
    assert hits2 == []
    assert len(suppressed) == 1
    assert suppressed[0]["reason"] == "loop_guard"
    assert orch._last_context_reset_at == first_at

    # クールダウンが明ければ再び発火する
    orch._last_context_reset_at = time.time() - Orchestrator._CONTEXT_RESET_COOLDOWN_SEC - 1
    with capture_logs() as logs3:
        orch._reset_saturated_context("語3", reason="loop_guard")
    hits3 = [e for e in logs3 if e.get("event") == "context reset (saturation)"]
    assert len(hits3) == 1
    assert hits3[0]["reason"] == "loop_guard"


def test_context_reset_discards_prefetch():
    orch, _hub = make_duo()
    orch._prefetched = {"at": time.time(), "agent_key": "chloe",
                        "topic": None, "response": LLMResponse(speech="x"),
                        "actions": [], "premise_speech": ""}
    with capture_logs() as logs:
        orch._reset_saturated_context("語")
    assert orch._prefetched is None
    assert any(e.get("event") == "prefetch discarded"
               and e.get("reason") == "context reset" for e in logs)


# ── 問題5: 相づち同一文カウンタのターン跨ぎ保持 ──────────────

AIZUCHI = "あらあら、Character1ちゃん、それはどういうことなの？わたしにも教えて？"


def test_aizuchi_drop_counter_survives_turn_boundary():
    orch, _hub = make_duo()
    orch._note_aizuchi_drop(AIZUCHI)
    assert orch._aizuchi_drop_count == 1
    assert orch._aizuchi_early_switch is False
    # _speak_idle のターン開始処理（窓の掃除＋フラグだけ落とす）を再現
    orch._prune_aizuchi_drop_clusters()
    orch._aizuchi_early_switch = False
    # 別ターンで同じ文がもう1回差し止められたら、窓内なので数えが届く
    orch._note_aizuchi_drop(AIZUCHI)
    assert orch._aizuchi_drop_count == 2
    assert orch._aizuchi_early_switch is True


def test_aizuchi_drop_counter_expires_with_window():
    orch, _hub = make_duo()
    orch._note_aizuchi_drop(AIZUCHI)
    # 窓を過ぎたことにする
    orch._aizuchi_drop_clusters[0][2] = (
        time.time() - Orchestrator._AIZUCHI_DROP_WINDOW_SEC - 1)
    orch._prune_aizuchi_drop_clusters()
    orch._aizuchi_early_switch = False
    orch._note_aizuchi_drop(AIZUCHI)
    assert orch._aizuchi_drop_count == 1
    assert orch._aizuchi_early_switch is False


# ── 問題2/6: プリフェッチ消費直前の再検査と破棄理由 ──────────

def _pf(speech: str, at: float | None = None) -> dict:
    return {"at": time.time() if at is None else at,
            "agent_key": "chloe", "topic": None,
            "response": LLMResponse(speech=speech),
            "actions": [], "premise_speech": ""}


def test_prefetch_recheck_blocks_repeat_of_latest_speech():
    orch, _hub = make_duo()
    speech = "今日はブラックフライデーの話をしようね、みんな準備はいい？"
    orch._recent_speech_norms.append(orch._normalize_speech(speech))
    # F10: 真偽値ラッパ _prefetch_passes_recheck は削除され、
    # 理由を返す _prefetch_recheck_reason に一本化された
    assert orch._prefetch_recheck_reason(_pf(speech)) != ""
    assert orch._prefetch_recheck_reason(
        _pf("全然違う新しい話題を持ってきたよ。")) == ""


def test_prefetch_recheck_blocks_foreign_contamination():
    orch, _hub = make_duo()
    assert orch._prefetch_recheck_reason(
        _pf("ボクも待ってますよ！啦，早く終わりたいな")) != ""


def test_take_prefetched_discards_pre_reset_prefetch():
    orch, hub = make_duo()
    speaker = orch.agents[0]
    orch._prefetched = _pf("リセット前の文脈で作った発話だよ。",
                           at=time.time() - 10)
    orch._last_context_reset_at = time.time()
    with capture_logs() as logs:
        assert orch._take_prefetched(speaker) is None
    assert any(e.get("event") == "prefetch discarded"
               and e.get("reason") == "context reset" for e in logs)


# ── 問題2: 日記の品質ゲート ──────────────────────────────────

class _DiaryLLM(StubLLM):
    def __init__(self, speech: str):
        super().__init__("{}")
        self._speech = speech

    async def generate(self, system_prompt, messages, *,
                       route="respond", temperature=None):
        self.generate_calls += 1
        return LLMResponse(speech=self._speech)


def _diary_setup(tmp: Path):
    log = ConversationLog()
    log.add_idle("今日はブラックフライデーの話で盛り上がったね。"
                 "みんなの反応が面白かったなあ、また来週も楽しみにしててね。")
    log.add_idle("掃除の話もしたね。竹ぼうきの手入れは大事なんだよ。"
                 "明日は天気も良さそうだし、配信の準備を進めておこう。")
    prompts_dir = tmp / "prompts"
    prompts_dir.mkdir(parents=True)
    ltm = LongTermMemory("クロエ", prompts_dir=prompts_dir)
    return log, ltm


@pytest.mark.asyncio
async def test_diary_rejected_when_foreign_contaminated():
    tmp = Path(tempfile.mkdtemp(prefix="aivt_diary_"))
    log, ltm = _diary_setup(tmp)
    engine = ReflectionEngine()
    result = await engine.reflect_and_update(
        log, None, ltm, _DiaryLLM("今日は楽しかった啦，また明日ね。"), "クロエ")
    assert result is None
    assert "最近の思い出・日記" not in (ltm.get_raw() or "")


@pytest.mark.asyncio
async def test_diary_rejected_when_meta_leak():
    tmp = Path(tempfile.mkdtemp(prefix="aivt_diary_"))
    log, ltm = _diary_setup(tmp)
    engine = ReflectionEngine()
    result = await engine.reflect_and_update(
        log, None, ltm,
        _DiaryLLM("thinking process: 今日の配信を振り返る必要がある。"
                  "リスナーの話を思い出そう。"), "クロエ")
    assert result is None
    assert "最近の思い出・日記" not in (ltm.get_raw() or "")


@pytest.mark.asyncio
async def test_diary_written_when_clean():
    tmp = Path(tempfile.mkdtemp(prefix="aivt_diary_"))
    log, ltm = _diary_setup(tmp)
    engine = ReflectionEngine()
    result = await engine.reflect_and_update(
        log, None, ltm,
        _DiaryLLM("今日はみんなとお買い物の話で盛り上がれて楽しかった。"
                  "また来週も一緒に話そうね。"), "クロエ")
    assert result is not None
    assert "盛り上がれて楽しかった" in (ltm.get_raw() or "")


# ── 問題1: Ollama probe の実生成化と stream_error ────────────

class _FakeStreamResp:
    def __init__(self, lines: list[str], error: Exception | None = None):
        self._lines = lines
        self._error = error

    def raise_for_status(self) -> None:
        pass

    async def aiter_lines(self):
        for line in self._lines:
            yield line
        if self._error is not None:
            raise self._error


class _FakeStreamCM:
    def __init__(self, resp: _FakeStreamResp):
        self._resp = resp

    async def __aenter__(self):
        return self._resp

    async def __aexit__(self, *exc):
        return False


class _FakeClient:
    is_closed = False

    def __init__(self, resp: _FakeStreamResp):
        self._resp = resp
        self.payloads: list[dict] = []

    def stream(self, method, url, json=None, timeout=None):
        self.payloads.append(json)
        return _FakeStreamCM(self._resp)


def _ollama_backend(client: _FakeClient) -> OllamaBackend:
    cfg = LLMBackendConfig(model="m", base_url="http://localhost:11434")
    backend = OllamaBackend(cfg)
    backend._client = client  # _get_client がそのまま返す
    return backend


def _ndjson_done(eval_count: int) -> str:
    return json.dumps({"done": True, "done_reason": "stop",
                       "eval_count": eval_count, "prompt_eval_count": 120,
                       "total_duration": 3_000_000_000,
                       "eval_duration": 2_000_000_000})


@pytest.mark.asyncio
async def test_probe_requires_sustained_streaming_generation():
    chunk = json.dumps({"message": {"content": "あ"}})
    client = _FakeClient(_FakeStreamResp([chunk, _ndjson_done(48)]))
    backend = _ollama_backend(client)
    assert await backend.probe() is True
    payload = client.payloads[0]
    assert payload["stream"] is True
    assert payload["options"]["num_predict"] >= 32


@pytest.mark.asyncio
async def test_probe_rejects_one_token_response():
    # 旧 probe（num_predict=1）なら通ってしまう「即 done」を弾く
    client = _FakeClient(_FakeStreamResp([_ndjson_done(1)]))
    backend = _ollama_backend(client)
    assert await backend.probe() is False
    assert backend.is_healthy is False


@pytest.mark.asyncio
async def test_probe_rejects_stream_timeout():
    client = _FakeClient(_FakeStreamResp(
        [], error=httpx.ReadTimeout("stall")))
    backend = _ollama_backend(client)
    assert await backend.probe() is False


@pytest.mark.asyncio
async def test_stream_error_recorded_in_generation_meta():
    client = _FakeClient(_FakeStreamResp(
        [json.dumps({"message": {"content": "あ"}})],
        error=httpx.ReadTimeout("stall")))
    backend = _ollama_backend(client)
    async for _delta in backend.generate_stream("sys", []):
        pass
    meta = backend.last_generation_meta()
    assert meta.get("stream_error") == "ReadTimeout"
    assert backend.is_healthy is False


@pytest.mark.asyncio
async def test_stream_success_records_latency_meta():
    lines = [json.dumps({"message": {"content": "やっほー"}}),
             _ndjson_done(30)]
    client = _FakeClient(_FakeStreamResp(lines))
    backend = _ollama_backend(client)
    got = [d async for d in backend.generate_stream("sys", [])]
    assert got == ["やっほー"]
    meta = backend.last_generation_meta()
    assert meta.get("eval_count") == 30
    assert meta.get("total_duration_ms") == 3000
    assert "stream_error" not in meta


# ── 問題6: curiosity 検索の時間上限 ──────────────────────────

class _SlowSearch:
    async def search(self, query, limit=None):
        await asyncio.sleep(30)
        return []


@pytest.mark.asyncio
async def test_curiosity_search_timeout_returns_none():
    planner = TopicPlanner.__new__(TopicPlanner)
    planner._search = _SlowSearch()
    planner._history = []
    planner._search_failures = []
    planner.max_documents = 3
    planner._CURIOSITY_SEARCH_TIMEOUT_SEC = 0.2
    started = time.monotonic()
    topic = await planner._plan_curiosity("テストの話題")
    assert topic is None
    assert time.monotonic() - started < 5
    assert len(planner._search_failures) == 1
