"""2026-11-21 分析D・E・F のテスト

D: Ollama バックエンドの例外ログ改善（stream error / error に例外の型名と
   repr を残す）。2026-11-19 夜の実配信で no_speech 6件を分類した際、
   "Ollama stream error" の error= が空（str(e) が空の例外）で原因を
   確定できなかった反省から。分類の結論として「ヘッダのみ＋stop」の
   skip 様式は 0 件だったため、経路分離は実装していない。
E: video.html の早期リカバリはフロント JS のため本ファイルの対象外
   （スモークで通常系の誤発火なしを確認する）。
F: 適応制御の閾値は現状維持（変更なし）のため対象外。
"""
import json

import httpx
import pytest
from structlog.testing import capture_logs

from backend.config import LLMBackendConfig
from backend.llm.ollama_backend import OllamaBackend


def _backend(handler) -> OllamaBackend:
    cfg = LLMBackendConfig(
        base_url="http://localhost:11434", model="m",
        extra_payload={"think": False})
    backend = OllamaBackend(cfg)
    backend._client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler))
    return backend


class TestStreamErrorLogging:
    async def test_stream_error_keeps_type_and_repr_for_empty_str_exception(self):
        async def handler(request):
            raise httpx.ReadError("", request=request)

        backend = _backend(handler)
        with capture_logs() as logs:
            chunks = [c async for c in backend.generate_stream("sys", [])]
        assert chunks == []
        assert backend.is_healthy is False
        errors = [e for e in logs if e.get("event") == "Ollama stream error"]
        assert len(errors) == 1
        assert errors[0]["error_type"] == "ReadError"
        assert "ReadError" in errors[0]["error"]

    async def test_generate_error_keeps_type_and_repr(self):
        async def handler(request):
            raise RuntimeError("kaput")

        backend = _backend(handler)
        with capture_logs() as logs:
            result = await backend.generate("sys", [])
        assert result.skip is True
        errors = [e for e in logs if e.get("event") == "Ollama error"]
        assert len(errors) == 1
        assert errors[0]["error_type"] == "RuntimeError"
        assert "kaput" in errors[0]["error"]

    async def test_stream_success_still_flows_and_keeps_meta(self):
        body = b"\n".join([
            json.dumps({"message": {"content": "こん"}}).encode(),
            json.dumps({"message": {"content": "にちは"}}).encode(),
            json.dumps({"message": {"content": ""}, "done": True,
                        "done_reason": "stop", "eval_count": 12,
                        "prompt_eval_count": 345}).encode(),
        ])

        async def handler(request):
            return httpx.Response(200, content=body)

        backend = _backend(handler)
        chunks = [c async for c in backend.generate_stream("sys", [])]
        assert "".join(chunks) == "こんにちは"
        meta = backend.last_generation_meta()
        assert meta["done_reason"] == "stop"
        assert meta["eval_count"] == 12
        assert meta["prompt_eval_count"] == 345
