"""2026-11-19 num_ctx 適応制御のテスト

問題2対策で num_ctx を固定 8192 に広げたが、文脈が育てば再び天井に
当たるため、岡田さんの指示で段階式の自動増減に変更した。

検証する仕様:
- 段（ラダー）[4096, 6144, 8192, 12288, 16384]、初期段 8192
- 上げる: prompt_eval_count が現段の85%超、または done_reason=length
  → 次のターン境界（begin_turn）で1段上がる。観測だけでは変わらない
- 下げる: 現段の45%未満が持続（50回連続 or 30分継続）→ 1段下がる
- ヒステリシス: 段変更後はクールダウン（既定300秒）の間は再変更しない
- 最上段では上がらず、溢れの兆候があれば生成前の履歴切り詰めが効く
- 段の状態はバックエンド（=キャラ）ごとに独立
- adaptive_ctx=False では従来どおり extra_payload の固定値が効く
- HybridRouter は begin_turn / context_status を内側へ委譲する
- knowledge_status にキャラ別の llm_context が出る
"""
from __future__ import annotations

from types import SimpleNamespace

from backend.config import LLMBackendConfig, LLMConfig
from backend.factory import _apply_adaptive_ctx
from backend.llm.adaptive_ctx import AdaptiveContextController
from backend.llm.base import LLMBackend
from backend.llm.hybrid_router import HybridRouter
from backend.llm.ollama_backend import OllamaBackend
from backend.models import ChatMessage, LLMResponse
from helpers import StubLLM, StubTTS, make_orchestrator

TIERS = [4096, 6144, 8192, 12288, 16384]


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def advance(self, sec: float) -> None:
        self.t += sec


def _controller(clock: FakeClock | None = None, **kw) -> AdaptiveContextController:
    return AdaptiveContextController(
        TIERS, 8192, clock=clock or FakeClock(), **kw)


# ── コントローラ: 上げる ──────────────────────────────────────

class TestRaise:
    def test_initial_tier(self):
        c = _controller()
        assert c.current_num_ctx == 8192
        assert c.index == 2

    def test_raise_on_high_prompt_usage(self):
        c = _controller()
        c.observe({"prompt_eval_count": 7000, "done_reason": "stop"})
        # 観測だけでは段は変わらず、保留に積まれるだけ
        assert c.current_num_ctx == 8192
        assert c.pending_num_ctx == 12288
        change = c.begin_turn()
        assert change is not None
        assert change["from_num_ctx"] == 8192
        assert change["to_num_ctx"] == 12288
        assert change["reason"] == "high_prompt_usage"
        assert c.current_num_ctx == 12288

    def test_no_raise_below_threshold(self):
        c = _controller()
        c.observe({"prompt_eval_count": 6900, "done_reason": "stop"})
        assert c.pending_num_ctx is None
        assert c.begin_turn() is None
        assert c.current_num_ctx == 8192

    def test_raise_on_length_truncation(self):
        c = _controller()
        c.observe({"prompt_eval_count": 100, "done_reason": "length"})
        assert c.pending_num_ctx == 12288
        change = c.begin_turn()
        assert change is not None and change["reason"] == "length_truncation"

    def test_raise_stops_at_top(self):
        c = AdaptiveContextController(TIERS, 16384, clock=FakeClock())
        c.observe({"prompt_eval_count": 16000, "done_reason": "length"})
        assert c.pending_num_ctx is None
        assert c.current_num_ctx == 16384


# ── コントローラ: 下げる ──────────────────────────────────────

class TestLower:
    def test_lower_after_sustained_turns(self):
        c = _controller()
        for _ in range(49):
            c.observe({"prompt_eval_count": 1000, "done_reason": "stop"})
        assert c.pending_num_ctx is None
        c.observe({"prompt_eval_count": 1000, "done_reason": "stop"})
        assert c.pending_num_ctx == 6144
        change = c.begin_turn()
        assert change is not None and change["reason"] == "sustained_low_usage"
        assert c.current_num_ctx == 6144

    def test_lower_after_time_window(self):
        clock = FakeClock()
        c = _controller(clock)
        c.observe({"prompt_eval_count": 1000, "done_reason": "stop"})
        clock.advance(900)
        c.observe({"prompt_eval_count": 1000, "done_reason": "stop"})
        assert c.pending_num_ctx is None
        clock.advance(900)
        c.observe({"prompt_eval_count": 1000, "done_reason": "stop"})
        assert c.pending_num_ctx == 6144

    def test_low_streak_resets_on_normal_usage(self):
        c = _controller()
        for _ in range(49):
            c.observe({"prompt_eval_count": 1000, "done_reason": "stop"})
        # 45%以上・85%以下の観測で連続が途切れる（上げもしない）
        c.observe({"prompt_eval_count": 5000, "done_reason": "stop"})
        for _ in range(49):
            c.observe({"prompt_eval_count": 1000, "done_reason": "stop"})
        assert c.pending_num_ctx is None
        c.observe({"prompt_eval_count": 1000, "done_reason": "stop"})
        assert c.pending_num_ctx == 6144

    def test_lower_stops_at_floor(self):
        c = AdaptiveContextController(TIERS, 4096, clock=FakeClock())
        for _ in range(60):
            c.observe({"prompt_eval_count": 100, "done_reason": "stop"})
        assert c.pending_num_ctx is None
        assert c.current_num_ctx == 4096


# ── コントローラ: ヒステリシス・独立性 ────────────────────────

class TestHysteresis:
    def test_cooldown_blocks_flapping(self):
        clock = FakeClock()
        c = _controller(clock)
        c.observe({"prompt_eval_count": 7000, "done_reason": "stop"})
        assert c.begin_turn() is not None
        assert c.current_num_ctx == 12288
        # 直後に低使用が持続しても、クールダウン中は下げない
        for _ in range(50):
            c.observe({"prompt_eval_count": 1000, "done_reason": "stop"})
        assert c.pending_num_ctx == 8192
        clock.advance(100)
        assert c.begin_turn() is None
        assert c.current_num_ctx == 12288
        clock.advance(201)
        change = c.begin_turn()
        assert change is not None and change["to_num_ctx"] == 8192

    def test_instances_are_independent(self):
        a, b = _controller(), _controller()
        a.observe({"prompt_eval_count": 100, "done_reason": "length"})
        a.begin_turn()
        assert a.current_num_ctx == 12288
        assert b.current_num_ctx == 8192
        assert b.pending_num_ctx is None


# ── コントローラ: 最上段の溢れ兆候 ────────────────────────────

class TestPreTruncateSignal:
    def test_signal_on_length_at_top(self):
        c = AdaptiveContextController(TIERS, 16384, clock=FakeClock())
        c.observe({"prompt_eval_count": 9000, "done_reason": "length"})
        assert c.needs_pre_truncate(500) is True

    def test_signal_on_prompt_overflow_at_top(self):
        c = AdaptiveContextController(TIERS, 16384, clock=FakeClock())
        c.observe({"prompt_eval_count": 16000, "done_reason": "stop"})
        assert c.needs_pre_truncate(500) is True

    def test_no_signal_below_top(self):
        c = _controller()
        c.observe({"prompt_eval_count": 100, "done_reason": "length"})
        assert c.needs_pre_truncate(500) is False

    def test_no_signal_when_fits(self):
        c = AdaptiveContextController(TIERS, 16384, clock=FakeClock())
        c.observe({"prompt_eval_count": 8000, "done_reason": "stop"})
        assert c.needs_pre_truncate(500) is False


# ── バックエンド配線 ──────────────────────────────────────────

def _backend(extra: dict | None = None) -> OllamaBackend:
    cfg = LLMBackendConfig(
        base_url="http://localhost:11434", model="m", max_tokens=500,
        extra_payload=extra if extra is not None
        else {"think": False, "num_ctx": 8192})
    return OllamaBackend(cfg)


def _stamp(backend: OllamaBackend, llm_cfg: LLMConfig) -> OllamaBackend:
    _apply_adaptive_ctx(backend, SimpleNamespace(llm=llm_cfg))
    return backend


class TestBackendWiring:
    def test_config_defaults(self):
        cfg = LLMConfig()
        assert cfg.adaptive_ctx is True
        assert cfg.context_tiers == TIERS
        assert cfg.context_initial in cfg.context_tiers
        assert cfg.context_raise_ratio == 0.85
        assert cfg.context_lower_ratio == 0.45

    def test_adaptive_payload_progression(self):
        b = _stamp(_backend(), LLMConfig())
        assert b._adaptive is not None
        payload = b._build_payload("sys", [], stream=False)
        assert payload["options"]["num_ctx"] == 8192
        # 高使用率を観測 → ターン境界で上がる → 次のペイロードに反映
        b._last_gen_meta = {"prompt_eval_count": 7000,
                            "done_reason": "stop", "eval_count": 10}
        b._observe_ctx_meta()
        b.begin_turn()
        payload = b._build_payload("sys", [], stream=False)
        assert payload["options"]["num_ctx"] == 12288
        status = b.context_status()
        assert status["adaptive"] is True
        assert status["num_ctx"] == 12288
        assert status["last_prompt_eval_count"] == 7000

    def test_adaptive_overrides_fixed_extra_payload(self):
        b = _stamp(_backend({"think": False, "num_ctx": 4096}), LLMConfig())
        payload = b._build_payload("sys", [], stream=False)
        assert payload["options"]["num_ctx"] == 8192

    def test_disabled_falls_back_to_fixed(self):
        b = _stamp(_backend(), LLMConfig(adaptive_ctx=False))
        assert b._adaptive is None
        payload = b._build_payload("sys", [], stream=False)
        assert payload["options"]["num_ctx"] == 8192
        # 観測・ターン境界を経ても固定値のまま
        b._last_gen_meta = {"prompt_eval_count": 7000,
                            "done_reason": "length", "eval_count": 5}
        b._observe_ctx_meta()
        b.begin_turn()
        payload = b._build_payload("sys", [], stream=False)
        assert payload["options"]["num_ctx"] == 8192
        assert b.context_status()["adaptive"] is False

    def test_pre_truncate_at_top_tier(self):
        llm_cfg = LLMConfig(context_tiers=[4096], context_initial=4096)
        b = _stamp(_backend(), llm_cfg)
        b._last_gen_meta = {"prompt_eval_count": 4000,
                            "done_reason": "length", "eval_count": 5}
        b._observe_ctx_meta()
        messages = [ChatMessage(role="user", content=f"{i}" + "x" * 999)
                    for i in range(10)]
        payload = b._build_payload("sys", messages, stream=False)
        sent = payload["messages"]
        # 予算（4096-500-256=3340トークン）に収まるまで古い順に落ち、
        # system + 直近3件が残る（入力リスト自体は不変）
        assert len(sent) == 4
        assert sent[0]["role"] == "system"
        assert sent[1]["content"].startswith("7")
        assert sent[-1]["content"].startswith("9")
        assert len(messages) == 10
        assert payload["options"]["num_ctx"] == 4096


# ── ルーター委譲 ──────────────────────────────────────────────

class _RecordingBackend(LLMBackend):
    def __init__(self, status: dict | None = None):
        self.turns = 0
        self._status = status or {}

    async def generate(self, system_prompt, messages, *, route="respond",
                       temperature=None) -> LLMResponse:
        return LLMResponse(speech="", skip=True)

    def begin_turn(self) -> None:
        self.turns += 1

    def context_status(self) -> dict:
        return dict(self._status)

    @property
    def name(self) -> str:
        return "recording"

    @property
    def is_healthy(self) -> bool:
        return True


class TestRouterDelegation:
    def test_begin_turn_and_status_delegate(self):
        idle = _RecordingBackend({"num_ctx": 8192, "adaptive": True})
        fast = _RecordingBackend()
        router = HybridRouter(idle, fast)
        router.begin_turn()
        assert idle.turns == 1 and fast.turns == 1
        assert router.context_status()["num_ctx"] == 8192


# ── knowledge_status ──────────────────────────────────────────

class TestKnowledgeStatus:
    def test_llm_context_included(self):
        orch, _hub = make_orchestrator(StubLLM(""), StubTTS())
        backend = _stamp(_backend(), LLMConfig())

        class _FakeAgent:
            key = "char1"
            llm = backend

        orch._agents = [_FakeAgent()]
        status = orch.knowledge_status()
        assert "llm_context" in status
        ctx = status["llm_context"]["char1"]
        assert ctx["adaptive"] is True
        assert ctx["num_ctx"] == 8192
        assert ctx["tiers"] == TIERS
