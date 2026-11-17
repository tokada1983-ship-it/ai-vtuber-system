"""人格QLoRA機能と、再構築した基盤（factory / LLM共通化 / 記憶DB移行）のテスト

GPU・学習用パッケージ・外部サーバには依存しない。
"""
import asyncio
import json
import sqlite3

import httpx
import pytest

from backend import factory
from backend.avatar.actions import extract_actions
from backend.config import AppConfig, LLMBackendConfig, load_config, save_config
from backend.llm.lmstudio_backend import LMStudioBackend
from backend.llm.prompt_builder import build_idle_prompt, build_system_prompt
from backend.memory.memory_store import MemoryStore
from backend.models import ChatMessage
from backend.persona import export
from backend.persona.dataset import (
    PersonaDatasetBuilder,
    extract_sample_lines,
    shorten,
    viewer_turn,
)
from backend.persona.trainer import encode, load_rows
from helpers import RAW, StubLLM, StubTTS, make_orchestrator

CHARACTER_YAML = """name: テスト狐
description: |
  名前： テスト狐
  種族： 電脳妖狐
personality: |
  -一人称・二人称
  -一人称： 「わらわ」
  -セリフのサンプル集
  1. 配信開始の挨拶
  「皆の者、よう来たのう！ 今日もわらわが楽しませてやるぞ！」
  2. からかわれた時
  「な、なんじゃと！？ わらわはポンコツではないわ！」
rules: ''
"""


@pytest.fixture
def char_cfg(tmp_path):
    cfg = AppConfig()
    cfg.character.name = "テスト狐"
    path = tmp_path / "character.yaml"
    path.write_text(CHARACTER_YAML, encoding="utf-8")
    cfg.character.prompt_file = str(path)
    cfg.character.data_dir = str(tmp_path / "data")
    cfg.character.stream_theme = ""
    return cfg


# ── 設定 ────────────────────────────────────────────────────

class TestConfig:
    def test_save_and_load_roundtrip_includes_every_field(self, tmp_path):
        cfg = AppConfig()
        cfg.persona.enabled = True
        cfg.persona.runtime_model = "persona-x"
        cfg.local_chat.user_name = "テスター"
        cfg.llm.backends["ollama"] = LLMBackendConfig(base_url="http://x", model="m")
        path = tmp_path / "c.yaml"
        save_config(cfg, path)
        loaded = load_config(path)
        assert loaded == cfg
        assert not (tmp_path / "c.yaml.tmp").exists()   # 一時ファイルは残らない


# ── 記憶DBの移行（起動できなくなっていた不具合の回帰テスト）─────

def test_memory_store_opens_legacy_db_without_viewer_column(tmp_path):
    db = tmp_path / "memories.db"
    conn = sqlite3.connect(db)
    conn.executescript("""
        CREATE TABLE memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT, text TEXT NOT NULL,
            source TEXT NOT NULL DEFAULT 'auto', created_at REAL NOT NULL,
            last_used_at REAL, use_count INTEGER NOT NULL DEFAULT 0,
            importance REAL NOT NULL DEFAULT 1.0, pinned INTEGER NOT NULL DEFAULT 0);
        INSERT INTO memories (text, created_at) VALUES ('旧い記憶', 1.0);
    """)
    conn.close()
    store = MemoryStore(db_path=db)
    assert store.count() == 1
    store.add("新しい記憶", viewer_name="たろう")
    assert [e.text for e in store.by_viewer("たろう")] == ["新しい記憶"]
    store.close()


# ── factory ─────────────────────────────────────────────────

class TestFactory:
    def _cfg(self):
        cfg = AppConfig()
        cfg.llm.primary, cfg.llm.fallback = "lmstudio", "ollama"
        cfg.llm.backends = {
            "lmstudio": LLMBackendConfig(base_url="http://localhost:1234/v1", model="base"),
            "ollama": LLMBackendConfig(base_url="http://localhost:11434", model="llama"),
        }
        return cfg

    def test_without_persona(self):
        primary, fallback = factory.build_llm(self._cfg())
        assert primary.name == "lmstudio" and fallback.name == "ollama"
        assert not getattr(primary, "is_persona", False)

    def test_persona_becomes_primary_and_base_is_fallback(self):
        cfg = self._cfg()
        cfg.persona.enabled = True
        cfg.persona.backend = "ollama"
        cfg.persona.runtime_model = "persona-fox"
        primary, fallback = factory.build_llm(cfg)
        assert primary.is_persona and primary.name == "ollama"
        assert primary._cfg.model == "persona-fox"
        assert cfg.llm.backends["ollama"].model == "llama"   # 元の設定は書き換えない
        assert fallback.name == "lmstudio"

    def test_persona_without_model_name_is_ignored(self):
        cfg = self._cfg()
        cfg.persona.enabled = True
        primary, _ = factory.build_llm(cfg)
        assert primary.name == "lmstudio"

    def test_unknown_backend(self):
        assert factory.create_llm("nope", LLMBackendConfig()) is None
        assert factory.create_llm("llamacpp", LLMBackendConfig()).name == "openai_compat"

    def test_sources(self):
        cfg = AppConfig()
        from backend.sources.local_chat_source import LocalChatSource
        from backend.sources.mock_source import MockSource
        assert isinstance(factory.build_sources(cfg, "local")[0], LocalChatSource)
        assert isinstance(factory.build_sources(cfg, "mock")[0], MockSource)


# ── LLM共通実装 ─────────────────────────────────────────────

class TestOpenAICompat:
    def test_lmstudio_payload_and_connection_reuse(self):
        seen = []

        def handler(request: httpx.Request):
            seen.append(json.loads(request.content))
            return httpx.Response(200, json={"choices": [{"message": {"content": RAW}}]})

        be = LMStudioBackend(LLMBackendConfig(base_url="http://lm/v1", model="auto"))
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        be._client = client

        async def run():
            r1 = await be.generate("sys", [ChatMessage(role="user", content="hi")])
            r2 = await be.generate("sys", [ChatMessage(role="user", content="hi")])
            return r1, r2

        r1, r2 = asyncio.run(run())
        assert r1.speech and r2.speech
        assert be._get_client() is client                      # 毎回作り直さない
        assert "model" not in seen[0]                           # auto は送らない
        assert seen[0]["reasoning_effort"] == "none"
        assert seen[0]["messages"][-1]["content"].endswith("/no_think")

    def test_connect_error_marks_unhealthy(self):
        def handler(request):
            raise httpx.ConnectError("refused")

        be = LMStudioBackend(LLMBackendConfig(base_url="http://lm/v1"))
        be._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        r = asyncio.run(be.generate("sys", []))
        assert r.skip and not be.is_healthy


# ── slimプロンプト ──────────────────────────────────────────

class TestSlimPrompt:
    def test_slim_drops_personality_and_is_shorter(self, char_cfg):
        full = build_system_prompt(char_cfg)
        slim = build_system_prompt(char_cfg, slim_persona=True)
        assert "セリフのサンプル集" in full and "セリフのサンプル集" not in slim
        assert "テスト狐" in slim and "出力フォーマット" in slim
        assert "Constitutional Persona Anchor" not in slim
        assert len(slim) < len(full)

    def test_idle_prompt_passes_through(self, char_cfg):
        assert "セリフのサンプル集" not in build_idle_prompt(char_cfg, slim_persona=True)

    def test_orchestrator_uses_slim_only_while_persona_model_answers(self):
        orch, _ = make_orchestrator(StubLLM(RAW), StubTTS())
        assert not orch.slim_persona_prompt
        orch._llm.is_persona = True
        orch._cfg.persona.prompt_mode = "slim"
        assert orch.slim_persona_prompt
        orch._cfg.persona.prompt_mode = "full"
        assert not orch.slim_persona_prompt


# ── データセット ────────────────────────────────────────────

class TestDataset:
    def test_extract_sample_lines_with_situations(self, char_cfg):
        import yaml
        lines = extract_sample_lines(yaml.safe_load(CHARACTER_YAML))
        assert ("配信開始の挨拶", "皆の者、よう来たのう！ 今日もわらわが楽しませてやるぞ！") in lines
        assert all("わらわ" != line for _, line in lines)   # 一人称の説明は拾わない

    def test_shorten_cuts_at_sentence(self):
        text = "あ" * 50 + "。" + "い" * 50 + "。"
        assert shorten(text, limit=60) == "あ" * 50 + "。"

    def test_from_history_builds_pairs_with_context(self, char_cfg, tmp_path):
        hist = tmp_path / "chat_history.jsonl"
        rows = [
            {"role": "user", "text": "こんばんは", "author": "マスター"},
            {"role": "assistant", "text": "よう来たのう！", "emotion": "happy"},
            {"role": "user", "text": "今日は何する？", "author": "マスター"},
            {"role": "assistant", "text": "わらわと雑談じゃ！", "emotion": "relaxed"},
        ]
        hist.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        exs = PersonaDatasetBuilder(char_cfg, chat_history_path=hist).from_history()
        assert len(exs) == 2
        last = exs[1].messages
        assert last[0]["role"] == "system" and "セリフのサンプル集" not in last[0]["content"]
        assert [m["role"] for m in last[1:]] == ["user", "assistant", "user", "assistant"]
        out = json.loads(last[-1]["content"])
        assert out == {"emotion": "relaxed", "motion": "idle", "speech": "わらわと雑談じゃ！"}

    def test_synthesize_uses_full_prompt_for_teacher_and_slim_for_training(self, char_cfg):
        prompts = []

        class Teacher(StubLLM):
            async def generate(self, system_prompt, messages, *, route="respond", temperature=None):
                prompts.append(system_prompt)
                return await super().generate(system_prompt, messages)

        builder = PersonaDatasetBuilder(char_cfg, seed=1)
        exs, rejected = asyncio.run(builder.synthesize(Teacher(RAW), n=6, concurrency=2))
        assert rejected == 0 and len(exs) == 6
        assert all("セリフのサンプル集" in p for p in prompts)
        assert all("セリフのサンプル集" not in e.messages[0]["content"] for e in exs)
        assert json.loads(exs[0].messages[-1]["content"])["speech"]

    def test_write_dedupes_and_splits(self, char_cfg, tmp_path):
        builder = PersonaDatasetBuilder(char_cfg, seed=0)
        exs, _ = asyncio.run(builder.synthesize(StubLLM(RAW), n=5))
        exs += builder.from_samples()
        report = builder.write(exs, tmp_path / "ds")
        # 合成5件は同じセリフなので1件にまとまる
        assert report.total == 1 + len(builder.from_samples())
        rows = load_rows(report.path)
        assert len(rows) == report.train
        assert (tmp_path / "ds" / "meta.json").exists()

    def test_viewer_turn_escapes(self):
        assert viewer_turn('a"b', "<x>") == '<viewer_comment author="a&quot;b">&lt;x&gt;</viewer_comment>'


# ── 学習データのトークン化 ─────────────────────────────────

class FakeTokenizer:
    """1文字=1トークンの簡易トークナイザ（ChatMLもどきのテンプレート）"""
    eos_token = "|"

    def apply_chat_template(self, messages, add_generation_prompt=False, tokenize=False):
        text = "".join(f"<{m['role']}>{m['content']}|" for m in messages)
        return text + ("<assistant>" if add_generation_prompt else "")

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) for c in text]}


class TestEncode:
    def test_only_assistant_tokens_are_labeled(self):
        msgs = [{"role": "system", "content": "S"}, {"role": "user", "content": "U"},
                {"role": "assistant", "content": "AB"}]
        enc = encode(FakeTokenizer(), msgs, max_len=100)
        labeled = [chr(t) for t, l in zip(enc["input_ids"], enc["labels"]) if l != -100]
        assert "".join(labeled) == "AB|"     # 応答と終端だけを学習する

    def test_too_long_is_dropped(self):
        msgs = [{"role": "user", "content": "x" * 50}, {"role": "assistant", "content": "y"}]
        assert encode(FakeTokenizer(), msgs, max_len=10) is None


# ── エクスポート ────────────────────────────────────────────

class TestExport:
    def test_model_name_is_ollama_safe(self):
        name = export.runtime_model_name("Character1", "20260929-163000")
        assert name == name.lower() and all(c.isalnum() or c in "-._" for c in name)
        assert name == export.runtime_model_name("Character1", "20260929-163000")   # 決定的
        assert export.runtime_model_name("Sample Noir", "r1") == "persona-sample-noir-r1"

    def test_modelfile_modes(self, tmp_path):
        (tmp_path / "merged").mkdir()
        (tmp_path / "adapter").mkdir()
        text = export.write_modelfile(tmp_path, "merged").read_text(encoding="utf-8")
        assert text.startswith("FROM ./merged")
        text = export.write_modelfile(tmp_path, "adapter", ollama_base="qwen2.5:7b").read_text(encoding="utf-8")
        assert "FROM qwen2.5:7b" in text and "ADAPTER ./adapter" in text
        with pytest.raises(ValueError):
            export.write_modelfile(tmp_path, "adapter")

    def test_gguf_modelfile_requires_converted_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            export.write_modelfile(tmp_path, "gguf")
        export.gguf_path(tmp_path, "q8_0").write_bytes(b"GGUF")
        assert export.write_modelfile(tmp_path, "gguf").read_text(encoding="utf-8") \
            .startswith("FROM ./model-q8_0.gguf")

    def test_convert_reports_missing_script(self, tmp_path):
        ok, msg = asyncio.run(export.convert_to_gguf(tmp_path, tmp_path / "nope", "q8_0"))
        assert not ok and "convert_hf_to_gguf.py" in msg
        ok, msg = asyncio.run(export.convert_to_gguf(tmp_path, tmp_path, "q4_0"))
        assert not ok and "未対応" in msg


# ── 切り替え ────────────────────────────────────────────────

class TestManager:
    def _manager(self, tmp_path, monkeypatch):
        from backend.persona import manager as mgr_mod
        monkeypatch.setattr(mgr_mod, "save_config", lambda cfg: None)   # 本物のconfig.yamlを書かない
        orch, hub = make_orchestrator(StubLLM(RAW), StubTTS())
        orch._cfg.persona.output_dir = str(tmp_path / "persona")
        orch._cfg.llm.backends = {"ollama": LLMBackendConfig(base_url="http://localhost:11434", model="base")}
        orch._cfg.llm.primary = "ollama"
        return mgr_mod.PersonaManager(orch._cfg, hub, orch), orch

    def test_activate_and_deactivate_switch_llm(self, tmp_path, monkeypatch):
        mgr, orch = self._manager(tmp_path, monkeypatch)
        asyncio.run(mgr.activate(runtime_model="persona-fox", backend="ollama"))
        assert orch.persona_model_active and orch._llm._cfg.model == "persona-fox"
        assert orch.slim_persona_prompt
        asyncio.run(mgr.deactivate())
        assert not orch.persona_model_active and orch._llm._cfg.model == "base"

    def test_activate_requires_model(self, tmp_path, monkeypatch):
        mgr, _ = self._manager(tmp_path, monkeypatch)
        with pytest.raises(ValueError):
            asyncio.run(mgr.activate())

    def test_runs_listing_and_path_guard(self, tmp_path, monkeypatch):
        mgr, _ = self._manager(tmp_path, monkeypatch)
        run_dir = tmp_path / "persona" / mgr.character / "r1"
        (run_dir / "adapter").mkdir(parents=True)
        (run_dir / "adapter" / "adapter_config.json").write_text("{}", encoding="utf-8")
        (run_dir / "manifest.json").write_text(json.dumps({"status": "done", "final_loss": 0.5}),
                                               encoding="utf-8")
        runs = mgr.list_runs()
        assert runs[0]["run"] == "r1" and runs[0]["has_adapter"]
        with pytest.raises(ValueError):
            mgr._run_dir("../../etc")

    def test_update_settings_casts_types(self, tmp_path, monkeypatch):
        mgr, _ = self._manager(tmp_path, monkeypatch)
        cfg = mgr.update_settings({"epochs": "2", "include_chat_history": "false",
                                   "enabled": True, "unknown": 1})
        assert cfg["epochs"] == 2.0 and cfg["include_chat_history"] is False
        assert cfg["enabled"] is False     # 有効化は activate() でしかできない

    def test_teacher_uses_its_own_model(self, tmp_path, monkeypatch):
        mgr, orch = self._manager(tmp_path, monkeypatch)
        orch._cfg.persona.teacher_backend = "ollama"
        orch._cfg.persona.teacher_model = "big-teacher"
        teacher = mgr._build_teacher()
        assert teacher._cfg.model == "big-teacher"
        assert orch._cfg.llm.backends["ollama"].model == "base"   # 普段のモデルは変えない
        orch._cfg.persona.teacher_model = ""
        assert mgr._build_teacher()._cfg.model == "base"

    def test_training_requires_dataset(self, tmp_path, monkeypatch):
        mgr, _ = self._manager(tmp_path, monkeypatch)
        with pytest.raises(RuntimeError):
            asyncio.run(mgr.start_training())


# ── アバターアクション ──────────────────────────────────────

def test_extract_actions():
    text, acts = extract_actions("座るね [action: sit, target: chair] [アクション: 手を振る]")
    assert text == "座るね"
    assert acts == [{"action": "sit", "target": "chair"}, {"action": "手を振る", "target": None}]

# ── Ollama の自動起動・事前ロード ────────────────────────────

class TestOllamaService:
    def _svc(self, monkeypatch, running_after_spawn=True, exe="C:/ollama.exe"):
        from backend.llm import ollama_service as osvc
        svc = osvc.OllamaService("http://localhost:11434", startup_timeout_sec=2)
        state = {"running": False, "spawned": 0}

        async def is_running(timeout=2.0):
            return state["running"]

        def spawn(path):
            state["spawned"] += 1
            state["running"] = running_after_spawn

        monkeypatch.setattr(svc, "is_running", is_running)
        monkeypatch.setattr(svc, "_spawn", spawn)
        monkeypatch.setattr(osvc, "find_ollama", lambda configured="": exe)
        return svc, state

    def test_starts_server_when_down(self, monkeypatch):
        svc, state = self._svc(monkeypatch)
        assert asyncio.run(svc.ensure_running()) and state["spawned"] == 1

    def test_does_not_spawn_when_already_running(self, monkeypatch):
        svc, state = self._svc(monkeypatch)
        state["running"] = True
        assert asyncio.run(svc.ensure_running()) and state["spawned"] == 0

    def test_reports_missing_executable(self, monkeypatch):
        svc, state = self._svc(monkeypatch, exe=None)
        assert not asyncio.run(svc.ensure_running())
        assert "見つかりません" in svc.last_error and state["spawned"] == 0

    def test_auto_start_disabled(self, monkeypatch):
        svc, state = self._svc(monkeypatch)
        svc.auto_start = False
        assert not asyncio.run(svc.ensure_running()) and state["spawned"] == 0

    def test_remote_server_is_not_spawned(self, monkeypatch):
        from backend.llm import ollama_service as osvc
        svc = osvc.OllamaService("http://127.0.0.1:11434")

        async def down(timeout=2.0):
            return False
        monkeypatch.setattr(svc, "is_running", down)
        assert not asyncio.run(svc.ensure_running()) and "リモート" in svc.last_error

    def test_ensure_model_skips_missing_model_without_auto_pull(self, monkeypatch):
        svc, state = self._svc(monkeypatch)

        async def empty():
            return []
        monkeypatch.setattr(svc, "running_models", empty)
        monkeypatch.setattr(svc, "list_models", empty)
        assert not asyncio.run(svc.ensure_model("persona-x"))
        assert "ありません" in svc.last_error

    def test_normalize_model(self):
        from backend.llm.ollama_service import normalize_model
        assert normalize_model("persona-x") == "persona-x:latest"
        assert normalize_model("qwen2.5:7b") == "qwen2.5:7b"


def test_ollama_backend_starts_server_and_retries_on_connect_error(monkeypatch):
    from backend.llm import ollama_service as osvc
    from backend.llm.ollama_backend import OllamaBackend

    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ConnectError("refused")
        return httpx.Response(200, json={"message": {"content": RAW}})

    be = OllamaBackend(LLMBackendConfig(base_url="http://localhost:11434", model="m"))
    be._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    revived = []

    async def ensure_running():
        revived.append(True)
        return True
    monkeypatch.setattr(osvc.get_service("http://localhost:11434"), "ensure_running", ensure_running)
    r = asyncio.run(be.generate("sys", []))
    assert revived and calls["n"] == 2 and r.speech


# ── 外部サーバ（Irodori-TTS）の自動起動 ───────────────────────

class TestLocalServer:
    def test_spawns_waits_and_stops(self, tmp_path):
        import socket
        import sys
        from backend.services.local_server import LocalServerProcess
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        svc = LocalServerProcess("dummy", f"http://127.0.0.1:{port}/",
                                 [sys.executable, "-m", "http.server", str(port), "--bind", "127.0.0.1"],
                                 cwd=tmp_path, startup_timeout_sec=20, log_path=tmp_path / "srv.log")
        assert asyncio.run(svc.ensure_running())
        assert svc.started_by_us
        svc.stop()
        assert not asyncio.run(svc.is_healthy(timeout=1.0))

    def test_reports_missing_dir(self, tmp_path):
        from backend.services.local_server import LocalServerProcess
        svc = LocalServerProcess("dummy", "http://127.0.0.1:9/", ["nope"], cwd=tmp_path / "missing",
                                 startup_timeout_sec=1)
        assert not asyncio.run(svc.ensure_running()) and "見つかりません" in svc.last_error


class TestIrodoriService:
    def _cfg(self, tmp_path, **over):
        from backend.config import TTSBackendConfig
        cfg = AppConfig()
        cfg.tts.primary = over.get("primary", "irodori")
        cfg.tts.backends["irodori"] = TTSBackendConfig(base_url=over.get("url", "http://localhost:8088/v1"))
        cfg.tts.irodori_server.dir = str(tmp_path)
        cfg.tts.irodori_server.auto_start = over.get("auto_start", True)
        return cfg

    def test_only_when_irodori_is_used_locally(self, tmp_path):
        from backend.tts import irodori_service
        assert irodori_service.get_service(self._cfg(tmp_path)) is not None
        assert irodori_service.get_service(self._cfg(tmp_path, primary="voicevox")) is None
        assert irodori_service.get_service(self._cfg(tmp_path, auto_start=False)) is None
        assert irodori_service.get_service(self._cfg(tmp_path, url="http://127.0.0.1:8088/v1")) is None

    def test_health_url_and_command(self, tmp_path):
        from backend.tts import irodori_service
        svc = irodori_service.get_service(self._cfg(tmp_path, url="http://localhost:8090/v1"))
        assert svc.health_url == "http://localhost:8090/health"
        assert svc.command[-4:] == ["--host", "127.0.0.1", "--port", "8090"]
        venv = tmp_path / ".venv" / "Scripts"
        venv.mkdir(parents=True)
        (venv / "python.exe").write_bytes(b"")
        cmd = irodori_service.build_command(tmp_path, 8088)
        assert cmd[0].endswith("python.exe") and cmd[1:3] == ["-m", "irodori_openai_tts"]


def test_ollama_payload_think_passthrough():
    """extra_payload の think は Ollama API のトップレベルへ移る（Gemma 4 の思考OFF用）"""
    from backend.config import LLMBackendConfig
    from backend.llm.ollama_backend import OllamaBackend

    be = OllamaBackend(LLMBackendConfig(
        base_url="http://localhost:11434", model="gemma4:12b",
        extra_payload={"think": False}))
    payload = be._build_payload("sys", [], stream=False)
    assert payload["think"] is False

    be2 = OllamaBackend(LLMBackendConfig(
        base_url="http://localhost:11434", model="llama3.1:8b"))
    assert "think" not in be2._build_payload("sys", [], stream=False)
