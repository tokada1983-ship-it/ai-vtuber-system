# -*- coding: utf-8 -*-
"""Phase 4 の回帰テスト: JSONモード伝搬／温度オーバーライド／failback／話題 案D・E・F

2026-11-17 の1時間テスト後の改善（Phase 4）:
  - Ollama JSON モード（format: "json"）を extra_payload から伝搬する
  - LLMBackend に temperature 引数を追加し、回復リトライ1回目だけ引き上げる
  - フェイルオーバー後の failback（プライマリ回復の能動プローブ＋連続成功で復帰）
  - 案D: テーマのクラスタ化・ヒント後の即時エスカレーション・answer 発話の計数除外
  - 案E: 記憶プールの拡大（50件）・同一セッション上限・支配 gram の重み下げ
  - 案F: 飽和検出でキャラ名を除去してから 3-gram を数える

外部サービスには依存しない（ペイロード構築・スタブ・フェイクで検証）。
"""
from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
import yaml

from backend.config import (AppConfig, CharacterConfig, EnsembleConfig,
                            LLMBackendConfig)
from backend.knowledge.models import WebDocument
from backend.knowledge.topic_planner import Topic, TopicPlanner
from backend.llm.base import LLMBackend
from backend.llm.hybrid_router import HybridRouter
from backend.llm.lmstudio_backend import LMStudioBackend
from backend.llm.ollama_backend import OllamaBackend
from backend.llm.parser import parse_llm_output
from backend.models import ChatMessage, LLMResponse
from backend.orchestrator import Orchestrator
from helpers import FakeHub, NoAudio, StubLLM, StubTTS

MSGS = [ChatMessage(role="user", content="テスト")]
REPEAT_SPEECH = "同じ話を繰り返すよ。反復チェック用の文だよ。"


def _json_raw(speech: str) -> str:
    return json.dumps({"emotion": "neutral", "motion": "idle",
                       "skip": False, "speech": speech}, ensure_ascii=False)


def _write_char(dirpath: Path, name: str) -> str:
    path = dirpath / f"{name}.yaml"
    path.write_text(yaml.safe_dump(
        {"name": name, "description": f"{name}の説明",
         "personality": f"{name}の口調"}, allow_unicode=True), encoding="utf-8")
    return str(path)


def make_duo_p4(llm_a: StubLLM | None = None, llm_b: StubLLM | None = None):
    """Character1／Character2の2人編成（FakeHub・プランナなし）を作る。

    構成は test_phase2_recovery.make_duo_p2 に合わせる（実サービスを
    起動しない設定＋使い捨てデータディレクトリ）。
    """
    tmp = Path(tempfile.mkdtemp(prefix="aivt-p4-"))
    cfg = AppConfig()
    cfg.stt.enabled = False
    cfg.ollama.auto_start = False
    cfg.ollama.preload_on_start = False
    cfg.tts.irodori_server.auto_start = False
    cfg.knowledge.enabled = False
    cfg.characters = [
        CharacterConfig(name="Character1", id="Character1", data_dir=str(tmp),
                        prompt_file=_write_char(tmp, "Character1"),
                        aliases=["Character1ちゃん"]),
        CharacterConfig(name="Character2", id="Character2", data_dir=str(tmp),
                        prompt_file=_write_char(tmp, "Character2"),
                        aliases=["Character2ちゃん"]),
    ]
    cfg.ensemble = EnsembleConfig(turn_gap_sec=0, idle_banter_probability=0.0,
                                  banter_probability=0.0)
    hub = FakeHub()
    orch = Orchestrator(cfg, hub)
    a, b = orch.agents
    default_raw = _json_raw("わらわが答えてやろう。")
    a.llm = llm_a if llm_a is not None else StubLLM(default_raw)
    a.tts = StubTTS()
    b.llm = llm_b if llm_b is not None else StubLLM(default_raw)
    b.tts = StubTTS()
    orch._audio = NoAudio()
    orch._audio_cache = Path(tempfile.mkdtemp(prefix="aivt_audio_"))
    return orch, hub


class FakePlannerP4:
    """ban の記録と最小限の話題提供だけを行うフェイクプランナ"""

    def __init__(self):
        self.plan_calls = 0
        self.committed: list = []
        self.released: list = []
        self.banned: list[tuple[str, int]] = []

    async def plan(self, kind="", query="", allow_web=True, context=None):
        self.plan_calls += 1
        doc = WebDocument(title="テスト記事", url="https://example.com/1",
                          snippet="テスト用の資料テキストです。")
        return Topic(kind=kind or "news", label="テスト話題",
                     documents=[doc], topic_id=f"fake-{self.plan_calls}")

    def commit(self, topic):
        self.committed.append(topic)

    def release(self, topic):
        self.released.append(topic)

    def ban_word(self, word, seconds=600):
        self.banned.append((word, int(seconds)))

    def banned_words(self):
        return {w for w, _ in self.banned}

    def note_speech(self, idle, speaker_key="", text=""):
        pass

    def note_comment(self, author, text):
        pass

    def reset_thread(self):
        pass

    def has_deliverable_answer(self):
        return False


class RecordingLLM(StubLLM):
    """生成ごとに渡された temperature を記録するLLM（原稿は呼び出し順）。

    temperature kwargs 自体が渡されなかった呼び出しは SENTINEL で記録する。
    """

    SENTINEL = "UNSET"

    def __init__(self, raws: list[str]):
        super().__init__(raws[0])
        self._raws = list(raws)
        self.stream_temps: list = []
        self.gen_temps: list = []

    def _pick(self, n: int) -> str:
        return self._raws[min(n, len(self._raws) - 1)]

    async def generate_stream(self, system_prompt, messages, *, route="respond",
                              temperature=SENTINEL):
        self.stream_temps.append(temperature)
        self._raw = self._pick(len(self.stream_temps) - 1)
        async for chunk in StubLLM.generate_stream(self, system_prompt, messages,
                                                   route=route):
            yield chunk

    async def generate(self, system_prompt, messages, *, route="respond",
                       temperature=SENTINEL):
        self.gen_temps.append(temperature)
        self._raw = self._pick(len(self.gen_temps) - 1)
        return await StubLLM.generate(self, system_prompt, messages, route=route)

    def temps(self) -> list:
        return self.stream_temps or self.gen_temps


# ── JSON モード: extra_payload → ペイロード伝搬 ──────────────────

def _ollama_cfg(extra):
    return LLMBackendConfig(base_url="http://localhost:11434", model="m",
                            temperature=0.8, max_tokens=500, extra_payload=extra)


def test_ollama_payload_format_json_propagates():
    backend = OllamaBackend(_ollama_cfg({"think": False, "format": "json"}))
    payload = backend._build_payload("sys", [], stream=False)
    assert payload["format"] == "json"
    assert payload["think"] is False


def test_ollama_payload_has_no_format_by_default():
    backend = OllamaBackend(_ollama_cfg({"think": False}))
    payload = backend._build_payload("sys", [], stream=False)
    assert "format" not in payload


# ── 温度オーバーライド: バックエンド／ルーターの伝搬 ─────────────

def test_ollama_payload_temperature_override():
    backend = OllamaBackend(_ollama_cfg({}))
    assert backend._build_payload("sys", [], stream=False)["options"]["temperature"] == 0.8
    assert backend._build_payload("sys", [], stream=False,
                                  temperature=0.9)["options"]["temperature"] == 0.9


def test_lmstudio_payload_temperature_override():
    cfg = LLMBackendConfig(base_url="http://localhost:1234/v1", model="m",
                           temperature=0.8, max_tokens=100)
    backend = LMStudioBackend(cfg)
    assert backend._build_payload("sys", [], stream=False)["temperature"] == 0.8
    assert backend._build_payload("sys", [], stream=False,
                                  temperature=1.0)["temperature"] == 1.0


class _RecordingRouterBackend(LLMBackend):
    def __init__(self, label, healthy=True):
        self._label = label
        self._healthy = healthy
        self.gen_temps: list = []
        self.stream_temps: list = []

    async def generate(self, system_prompt, messages, *, route="respond",
                       temperature="UNSET"):
        self.gen_temps.append(temperature)
        return LLMResponse(speech=f"{self._label}:こんにちは")

    async def generate_stream(self, system_prompt, messages, *, route="respond",
                              temperature="UNSET"):
        self.stream_temps.append(temperature)
        yield f"{self._label}stream"

    @property
    def supports_streaming(self):
        return True

    @property
    def name(self):
        return self._label

    @property
    def is_healthy(self):
        return self._healthy


@pytest.mark.asyncio
async def test_router_forwards_temperature_only_when_set():
    idle = _RecordingRouterBackend("local")
    fast = _RecordingRouterBackend("cloud")
    router = HybridRouter(idle, fast)
    await router.generate("sys", MSGS, route="idle", temperature=0.9)
    assert idle.gen_temps == [0.9]
    await router.generate("sys", MSGS, route="idle")
    # 未指定のときは kwargs 自体を渡さない（子のデフォルトが効く）
    assert idle.gen_temps == [0.9, "UNSET"]


@pytest.mark.asyncio
async def test_recovery_retry_uses_raised_temperature():
    """回復リトライ1回目の生成だけ温度が 0.8→0.9 に上がること"""
    llm = RecordingLLM([_json_raw(REPEAT_SPEECH),
                        _json_raw("新しい話題を持ってきたよ。今日は天気の話をしよう。")])
    orch, hub = make_duo_p4(llm_a=llm)
    orch._planner = FakePlannerP4()
    orch._recent_speech_norms.append(Orchestrator._normalize_speech(REPEAT_SPEECH))

    await orch._speak_idle()

    assert llm.temps() == [RecordingLLM.SENTINEL, 0.9]
    assert orch._temp_override == {}  # 消費済み


@pytest.mark.asyncio
async def test_temperature_override_only_for_idle_route():
    llm = RecordingLLM([_json_raw("普通の発話だよ。")])
    orch, hub = make_duo_p4(llm_a=llm)
    agent = orch.agents[0]
    orch._arm_recovery_temperature(agent)
    assert orch._temp_override[agent.key][0] == pytest.approx(0.9)

    resp = await orch._generate_and_speak("sys", agent, route="respond")
    assert resp is not None
    assert llm.temps() == [RecordingLLM.SENTINEL]   # respond では使わない
    assert agent.key in orch._temp_override          # 消費もされない

    resp = await orch._generate_and_speak("sys", agent, route="idle")
    assert resp is not None
    assert llm.temps()[-1] == 0.9
    assert agent.key not in orch._temp_override


@pytest.mark.asyncio
async def test_temperature_override_expires():
    llm = RecordingLLM([_json_raw("普通の発話だよ。")])
    orch, hub = make_duo_p4(llm_a=llm)
    agent = orch.agents[0]
    orch._temp_override[agent.key] = (0.9, time.monotonic() - 1.0)  # 失効済み
    resp = await orch._generate_and_speak("sys", agent, route="idle")
    assert resp is not None
    assert llm.temps() == [RecordingLLM.SENTINEL]
    assert agent.key not in orch._temp_override


# ── failback: 能動プローブと連続成功のヒステリシス ───────────────

class FakePrimaryBackend(LLMBackend):
    """probe() を持つプライマリ役。結果列を順に返し、健全フラグも連動させる"""

    def __init__(self, probe_results):
        self._probe_results = list(probe_results)
        self._healthy = False
        self.probe_calls = 0

    async def generate(self, system_prompt, messages, *, route="respond",
                       temperature=None):
        return LLMResponse(speech="", skip=True)

    async def probe(self):
        self.probe_calls += 1
        ok = self._probe_results.pop(0) if self._probe_results else True
        self._healthy = ok
        return ok

    @property
    def name(self):
        return "ollama-primary"

    @property
    def is_healthy(self):
        return self._healthy


class FakeCurrentBackend(LLMBackend):
    """フォールバック先（LM Studio 役）。probe は持たない"""

    def __init__(self, healthy=True):
        self._healthy = healthy

    async def generate(self, system_prompt, messages, *, route="respond",
                       temperature=None):
        return LLMResponse(speech="つなぎ")

    @property
    def name(self):
        return "lmstudio-fallback"

    @property
    def is_healthy(self):
        return self._healthy


def _agent_on_fallback(orch, probe_results):
    agent = orch.agents[0]
    primary = FakePrimaryBackend(probe_results)
    current = FakeCurrentBackend(healthy=True)
    agent.llm = primary
    agent.llm_fallback = current
    assert agent.failover_llm() is True
    assert agent.on_fallback
    return agent, primary, current


@pytest.mark.asyncio
async def test_failback_requires_two_consecutive_probe_successes():
    orch, hub = make_duo_p4()
    agent, primary, current = _agent_on_fallback(orch, [True, True])

    await orch._health_check_cycle()
    assert agent.llm is current          # 1回成功では戻らない
    assert agent.on_fallback

    await orch._health_check_cycle()
    assert agent.llm is primary          # 2回連続成功で復帰
    assert not agent.on_fallback
    assert primary.probe_calls == 2


@pytest.mark.asyncio
async def test_failback_probe_failure_resets_streak():
    orch, hub = make_duo_p4()
    agent, primary, current = _agent_on_fallback(orch, [True, False, True, True])

    await orch._health_check_cycle()     # 成功1
    await orch._health_check_cycle()     # 失敗 → リセット
    assert orch._failback_probe_success[agent.key] == 0
    assert agent.llm is current
    await orch._health_check_cycle()     # 成功1
    assert agent.llm is current
    await orch._health_check_cycle()     # 成功2 → 復帰
    assert agent.llm is primary


def test_failback_llm_guards():
    orch, hub = make_duo_p4()
    agent = orch.agents[0]
    primary = FakePrimaryBackend([])
    primary._healthy = True
    current = FakeCurrentBackend(healthy=False)
    agent.llm = current
    agent.llm_fallback = primary
    agent._on_fallback = True

    assert agent.failback_llm() is False   # 現行が不健全なら戻さない
    current._healthy = True
    assert agent.failback_llm() is True
    assert agent.llm is primary
    assert not agent.on_fallback
    assert agent.failback_llm() is False   # 退避中でなければ何もしない


@pytest.mark.asyncio
async def test_probe_primary_delegation():
    orch, hub = make_duo_p4()
    agent = orch.agents[0]
    primary = FakePrimaryBackend([True])
    agent.llm_fallback = primary
    assert await orch._probe_primary(agent) is True
    assert primary.probe_calls == 1
    # probe を持たないバックエンドは健全フラグで判断する
    agent.llm_fallback = FakeCurrentBackend(healthy=False)
    assert await orch._probe_primary(agent) is False


@pytest.mark.asyncio
async def test_ollama_probe_success_and_failure():
    backend = OllamaBackend(_ollama_cfg({}))

    def ok_handler(request):
        return httpx.Response(200, json={"message": {"role": "assistant",
                                                     "content": ""}})

    backend._client = httpx.AsyncClient(transport=httpx.MockTransport(ok_handler))
    assert await backend.probe() is True
    assert backend.is_healthy

    def ng_handler(request):
        return httpx.Response(500, json={})

    backend._client = httpx.AsyncClient(transport=httpx.MockTransport(ng_handler))
    assert await backend.probe() is False
    assert not backend.is_healthy


# ── 案D: テーマのクラスタ化・ヒント後検証・answer 除外 ───────────

def _fire_research_theme(orch) -> None:
    for text in ("調査の結果を報告しますね", "調査スキルを磨きたいな",
                 "調査結果はこちらです"):
        orch._note_theme_repetition(text)


def test_theme_cluster_escalates_across_labels():
    """別ラベルでも同じ gram の再検知ならエスカレーション（第2段階 ban）"""
    orch, hub = make_duo_p4()
    planner = FakePlannerP4()
    orch._planner = planner
    _fire_research_theme(orch)
    assert planner.banned == [("調査結果", 600)]
    assert not orch._force_switch_pending

    orch._note_theme_repetition("調査スキルがまた上がったよ")
    assert ("調査", 600) in planner.banned
    assert orch._force_switch_pending


def test_hint_ignored_escalates_immediately():
    """テーマヒント消費直後の発話に同じ gram が残っていたら即エスカレーション"""
    orch, hub = make_duo_p4()
    planner = FakePlannerP4()
    orch._planner = planner
    _fire_research_theme(orch)
    assert orch._theme_hint_pending == "調査結果"

    orch._consume_theme_hint("base prompt")
    assert orch._theme_hint_consumed_gram == "調査"
    planner.banned.clear()

    orch._note_theme_repetition("調査の続きを話しますね")
    assert ("調査", 600) in planner.banned
    assert orch._force_switch_pending
    assert orch._theme_hint_consumed_gram == ""


def test_hint_followed_clears_marker_quietly():
    orch, hub = make_duo_p4()
    planner = FakePlannerP4()
    orch._planner = planner
    _fire_research_theme(orch)
    orch._consume_theme_hint("base prompt")
    planner.banned.clear()

    orch._note_theme_repetition("天気の話をしましょう、良い天気ですね")
    assert planner.banned == []
    assert not orch._force_switch_pending
    assert orch._theme_hint_consumed_gram == ""


def test_theme_hint_names_gram_after_second_fire():
    """同じ gram が2回目の発火をしたら、ヒント文で gram を名指しする"""
    orch, hub = make_duo_p4()
    orch._planner = FakePlannerP4()
    _fire_research_theme(orch)
    orch._consume_theme_hint("base prompt")
    orch._note_theme_repetition("天気の話をしましょう、良い天気ですね")
    for text in ("調査の続きを話しますね", "調査の結果はまだです",
                 "調査結果の続報です"):
        orch._note_theme_repetition(text)
    prompt = orch._consume_theme_hint("base prompt")
    assert "特に「調査」という言葉や話題は、今回は避けてください" in prompt


def test_answer_speech_excluded_from_theme_counting():
    orch, hub = make_duo_p4()
    agent = orch.agents[0]
    resp = parse_llm_output(
        '{"emotion":"neutral","motion":"idle",'
        '"speech":"調査結果は59,662円でした"}')
    before = len(orch._recent_theme_grams)
    orch._after_speech(agent, resp, idle=True, topic_kind="answer")
    assert len(orch._recent_theme_grams) == before       # 数えない
    orch._after_speech(agent, resp, idle=True)
    assert len(orch._recent_theme_grams) == before + 1   # 通常発話は数える


# ── 案E: 記憶プールの拡大・セッション上限・支配 gram 重み ────────

def test_memory_seed_uses_wider_pool(monkeypatch):
    orch, hub = make_duo_p4()
    a, b = orch.agents
    seen = {}
    entry = SimpleNamespace(text="天気の良い日の散歩の思い出です", source="episodic")

    def fake_sample_recent(limit=20):
        seen["limit"] = limit
        return [entry]

    a.memory_store = SimpleNamespace(sample_recent=fake_sample_recent)
    b.memory_store = SimpleNamespace(sample_recent=lambda limit=20: [])
    seed = orch._memory_topic_seed()
    assert seen["limit"] == 50
    assert seed is not None and seed["text"] == entry.text


def test_seed_pool_session_cap():
    orch, hub = make_duo_p4()
    base = 1763136000.0  # 2025-11-15 頃
    entries = [SimpleNamespace(text=f"思い出テキスト{i}です", source="episodic",
                               created_at=base) for i in range(9)]
    entries.append(SimpleNamespace(text="別の日の思い出テキストです",
                                   source="episodic", created_at=base + 86400.0))
    kept = orch._cap_seed_pool_by_session(entries)
    same_day = [e for e in kept if e.created_at == base]
    assert len(same_day) <= 3          # 上限 = 10 // 3
    assert len(kept) == 4


def test_seed_session_key_stream_episode_uses_body_date():
    orch, hub = make_duo_p4()
    entry = SimpleNamespace(
        text="2026年11月17日の配信で話した内容の要約です",
        source="stream_episode", created_at=0.0)
    assert orch._seed_session_key(entry) == "ep:2026年11月17日"
    other = SimpleNamespace(text="普通の思い出テキストです", source="episodic",
                            created_at=1763136000.0)
    assert orch._seed_session_key(other)  # 日付文字列が返る（例外にならない）


def test_seed_pool_weights_dominant_gram():
    orch, hub = make_duo_p4()
    dominant = [SimpleNamespace(text=f"調査ノートを見返した話その{i}について",
                                source="episodic") for i in range(4)]
    others = [SimpleNamespace(text="海へ泳ぎに行った夏の話", source="episodic"),
              SimpleNamespace(text="山で星空を眺めた夜の話", source="episodic")]
    weights = orch._seed_pool_weights(dominant + others)
    assert weights[:4] == [0.25] * 4   # 支配 gram（調査）を含むものは軽く
    assert weights[4:] == [1.0, 1.0]


def _planner_with_memory(items):
    seq = list(items)
    return TopicPlanner(news=None, search=None, state_path="",
                        memory_provider=lambda: seq.pop(0) if seq else None)


def test_plan_memory_never_adopts_banned_candidate():
    planner = _planner_with_memory(
        [{"text": "調査についての長い思い出話です", "source": "episodic"}] * 5)
    planner.ban_word("調査", seconds=600)
    assert planner._plan_memory() is None


def test_plan_memory_redraws_past_banned_candidate():
    planner = _planner_with_memory([
        {"text": "調査についての長い思い出話です", "source": "episodic"},
        {"text": "天気の良い日の散歩の思い出です", "source": "episodic"},
    ])
    planner.ban_word("調査", seconds=600)
    topic = planner._plan_memory()
    assert topic is not None
    assert "散歩" in topic.documents[0].snippet


def test_plan_memory_still_adopts_overlapping_candidate():
    """ban でなければ、重なりだけの候補は従来どおり最後に採用される"""
    planner = _planner_with_memory(
        [{"text": "調査の結果についての思い出話です", "source": "episodic"}] * 5)
    planner.note_speech(idle=True, text="調査の結果についての話")
    topic = planner._plan_memory()
    assert topic is not None


# ── 案F: 飽和検出のキャラ名除去 ──────────────────────────────────

def test_character_name_forms_include_chan_variants():
    orch, hub = make_duo_p4()
    forms = orch._character_name_forms()
    assert "Character2" in forms and "Character2ちゃん" in forms
    assert "Character1" in forms and "Character1ちゃん" in forms


def test_saturation_word_ignores_character_name_fragments():
    orch, hub = make_duo_p4()
    speeches = [
        "Character2も調査の話をしたよ",
        "Character2も調査の続きが気になる",
        "Character2も調査の結果を見たいな",
        "Character2も調査の話題で盛り上がった",
        "Character2も調査のノートを開いた",
        "Character2も調査の話ばかりだね",
    ]
    orch._recent_speech_norms = [Orchestrator._normalize_speech(s) for s in speeches]
    word = orch._saturation_word()
    assert word  # 内容語が拾える
    assert word not in ("Character2", "ター2も", "2も調")
