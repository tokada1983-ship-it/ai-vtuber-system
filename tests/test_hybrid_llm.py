"""ハイブリッドLLM振り分けのテスト

待機系（route="idle"）はローカル、即応（route="respond"）はクラウドへ流れ、
クラウドが不調・空応答のときはローカルへ戻ることを検証する。
config の往復（hybrid 節・gemini バックエンド・extra_payload）も確認する。
"""
import pytest

from backend import factory
from backend.config import AppConfig, LLMBackendConfig, load_config, save_config
from backend.llm.base import LLMBackend
from backend.llm.gemini_backend import GeminiBackend
from backend.llm.hybrid_router import HybridRouter
from backend.models import ChatMessage, LLMResponse


class FakeBackend(LLMBackend):
    """呼び出しを記録する偽バックエンド"""

    def __init__(self, label: str, *, speech: str = "こんにちは", healthy: bool = True):
        self._label = label
        self._speech = speech
        self._healthy = healthy
        self.calls: list[str] = []
        self.stream_calls: list[str] = []

    async def generate(self, system_prompt, messages, *, route="respond", temperature=None):
        # ルーターは route を消費して子には渡さないので、ここでは呼ばれた事実だけ記録する
        self.calls.append("generate")
        if not self._speech:
            return LLMResponse(speech="", skip=True)
        return LLMResponse(speech=f"{self._label}:{self._speech}")

    async def generate_stream(self, system_prompt, messages, *, route="respond", temperature=None):
        self.stream_calls.append("stream")
        if self._speech:
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


MSGS = [ChatMessage(role="user", content="テスト")]


@pytest.mark.asyncio
async def test_route_idle_goes_local_respond_goes_fast():
    idle, fast = FakeBackend("local"), FakeBackend("cloud")
    router = HybridRouter(idle, fast)
    r_idle = await router.generate("sys", MSGS, route="idle")
    r_fast = await router.generate("sys", MSGS, route="respond")
    assert r_idle.speech.startswith("local:")
    assert r_fast.speech.startswith("cloud:")
    assert idle.calls == ["generate"]
    assert fast.calls == ["generate"]


@pytest.mark.asyncio
async def test_default_route_is_fast():
    idle, fast = FakeBackend("local"), FakeBackend("cloud")
    router = HybridRouter(idle, fast)
    await router.generate("sys", MSGS)
    assert fast.calls == ["generate"]
    assert idle.calls == []


@pytest.mark.asyncio
async def test_fast_unhealthy_falls_back_to_local():
    idle = FakeBackend("local")
    fast = FakeBackend("cloud", healthy=False)
    router = HybridRouter(idle, fast)
    r = await router.generate("sys", MSGS, route="respond")
    assert r.speech.startswith("local:")
    assert fast.calls == []


@pytest.mark.asyncio
async def test_fast_empty_retries_on_local():
    idle = FakeBackend("local")
    fast = FakeBackend("cloud", speech="")
    router = HybridRouter(idle, fast)
    r = await router.generate("sys", MSGS, route="respond")
    assert r.speech.startswith("local:")
    assert fast.calls == ["generate"]
    assert idle.calls == ["generate"]


@pytest.mark.asyncio
async def test_stream_routing_and_empty_retry():
    idle, fast = FakeBackend("local"), FakeBackend("cloud")
    router = HybridRouter(idle, fast)
    out = [d async for d in router.generate_stream("sys", MSGS, route="idle")]
    assert out == ["localstream"]
    out = [d async for d in router.generate_stream("sys", MSGS, route="respond")]
    assert out == ["cloudstream"]

    empty_fast = FakeBackend("cloud", speech="")
    router2 = HybridRouter(idle, empty_fast)
    out = [d async for d in router2.generate_stream("sys", MSGS, route="respond")]
    assert out == ["localstream"]


def test_router_health_represents_idle_side():
    router = HybridRouter(FakeBackend("local", healthy=False),
                          FakeBackend("cloud", healthy=True))
    assert router.is_healthy is False
    assert router.supports_streaming is True
    assert "hybrid" in router.name


def test_gemini_payload_disables_thinking():
    backend = GeminiBackend(LLMBackendConfig(model="gemini-3.5-flash-lite"))
    payload = backend._build_payload("sys", MSGS, stream=False)
    assert payload["model"] == "gemini-3.5-flash-lite"
    # reasoning_effort と thinking_config の同時指定は Gemini が 400 を返すので、
    # 既定では thinking_config だけを使う
    assert "reasoning_effort" not in payload
    # 3.5系では thinking_budget=0 が受理されず、thinking_level="minimal" が既定
    assert payload["extra_body"]["google"]["thinking_config"] == {
        "thinking_level": "minimal"
    }


def test_extra_payload_overrides_defaults():
    cfg = LLMBackendConfig(model="gemini-3.5-flash",
                           extra_payload={"reasoning_effort": "low"})
    backend = GeminiBackend(cfg)
    payload = backend._build_payload("sys", MSGS, stream=False)
    assert payload["reasoning_effort"] == "low"
    # 明示指定がある場合は thinking_config を付けない（同時指定は400になる）
    assert "thinking_config" not in str(payload.get("extra_body", {}))


def _hybrid_config(tmp_path) -> AppConfig:
    cfg = AppConfig()
    cfg.llm.primary = "lmstudio"
    cfg.llm.backends = {
        "lmstudio": LLMBackendConfig(base_url="http://localhost:1234/v1",
                                     model="local-model"),
        "gemini": LLMBackendConfig(base_url="https://generativelanguage.googleapis.com/v1beta/openai",
                                   model="gemini-3.5-flash-lite",
                                   api_key_env="GEMINI_API_KEY"),
    }
    cfg.llm.hybrid.enabled = True
    cfg.llm.hybrid.idle_backend = "lmstudio"
    cfg.llm.hybrid.fast_backend = "gemini"
    return cfg


def test_factory_wraps_primary_in_router(tmp_path):
    cfg = _hybrid_config(tmp_path)
    primary, _fallback = factory.build_llm(cfg)
    assert isinstance(primary, HybridRouter)
    assert isinstance(primary.fast_backend, GeminiBackend)
    assert primary.idle_backend.name == "lmstudio"


def test_factory_without_hybrid_returns_plain_backend(tmp_path):
    cfg = _hybrid_config(tmp_path)
    cfg.llm.hybrid.enabled = False
    primary, _fallback = factory.build_llm(cfg)
    assert not isinstance(primary, HybridRouter)
    assert primary.name == "lmstudio"


def test_config_roundtrip_keeps_hybrid(tmp_path):
    path = tmp_path / "config.yaml"
    save_config(_hybrid_config(tmp_path), path)
    loaded = load_config(path)
    assert loaded.llm.hybrid.enabled is True
    assert loaded.llm.hybrid.idle_backend == "lmstudio"
    assert loaded.llm.hybrid.fast_backend == "gemini"
    assert loaded.llm.backends["gemini"].api_key_env == "GEMINI_API_KEY"
    assert loaded.llm.backends["gemini"].model == "gemini-3.5-flash-lite"
