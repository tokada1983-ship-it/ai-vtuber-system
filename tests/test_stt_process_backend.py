"""STT 別プロセス化（APPCRASH対策 第2段）のテスト

WhisperProcessBackend が、外部ワーカースタブ（HTTP）に対して
- 文字起こしを往復できること
- 同時呼び出しがクライアント側ロックで直列化されること（第1段と同等）
- ワーカー障害時に例外を投げず空結果を返し、再試行で回復すること
- config の isolation で factory が実装を切り替えること
を検証する。実モデルは使わない。
"""
from __future__ import annotations

import asyncio
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from backend.stt.whisper_process_backend import WhisperProcessBackend


class _StubState:
    def __init__(self):
        self.lock = threading.Lock()
        self.current = 0
        self.max_concurrent = 0
        self.calls = 0
        self.fail_first = 0  # この回数だけ /transcribe を 500 で落とす


def _make_stub(state: _StubState):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def _json(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                self._json(200, {"status": "ok", "pid": 999, "ready": True})
            else:
                self._json(404, {})

        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            if length:
                self.rfile.read(length)
            if self.path != "/transcribe":
                self._json(404, {})
                return
            with state.lock:
                state.calls += 1
                state.current += 1
                state.max_concurrent = max(state.max_concurrent, state.current)
                fail = state.fail_first > 0
                if fail:
                    state.fail_first -= 1
            try:
                if fail:
                    self._json(500, {"error": "stub failure"})
                    return
                time.sleep(0.05)
                self._json(200, {"text": "テスト", "language": "ja",
                                 "infer_ms": 50})
            finally:
                with state.lock:
                    state.current -= 1

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


@pytest.fixture()
def stub():
    state = _StubState()
    server = _make_stub(state)
    yield state, server.server_address[1]
    server.shutdown()


def _backend(port: int) -> WhisperProcessBackend:
    return WhisperProcessBackend(
        model_size="small", device="cpu",
        base_url=f"http://127.0.0.1:{port}")


async def test_transcribe_roundtrip(stub):
    _state, port = stub
    backend = _backend(port)
    pcm = b"\x00\x00" * 16000  # 1秒分
    result = await backend.transcribe(pcm)
    assert result.text == "テスト"
    assert result.language == "ja"
    assert result.duration_ms == 1000
    assert backend.is_ready


async def test_transcribe_calls_are_serialized(stub):
    state, port = stub
    backend = _backend(port)
    pcm = b"\x00\x00" * 16000
    results = await asyncio.gather(
        *(backend.transcribe(pcm) for _ in range(6)))
    assert all(r.text == "テスト" for r in results)
    assert state.calls == 6
    assert state.max_concurrent == 1


async def test_transcribe_retries_after_failure(stub):
    state, port = stub
    state.fail_first = 1  # 1回目は500、再試行で成功する想定
    backend = _backend(port)
    result = await backend.transcribe(b"\x00\x00" * 16000)
    assert result.text == "テスト"
    assert state.calls == 2


async def test_transcribe_worker_down_returns_empty():
    backend = WhisperProcessBackend(
        model_size="small", device="cpu",
        base_url="http://127.0.0.1:1")  # 誰も listen していない
    result = await backend.transcribe(b"\x00\x00" * 16000)
    assert result.text == ""
    assert not backend.is_ready


async def test_warmup_uses_transcribe_path(stub):
    state, port = stub
    backend = _backend(port)
    await backend.warmup()
    assert state.calls == 1
    assert backend.is_ready


def test_infer_lock_is_threading_lock(stub):
    _state, port = stub
    backend = _backend(port)
    assert isinstance(backend._infer_lock, type(threading.Lock()))


def test_config_isolation_default_and_factory_selection():
    from backend.config import STTConfig
    assert STTConfig().isolation == "inprocess"

    from backend.config import AppConfig
    from backend.factory import build_stt
    from backend.stt.whisper_backend import FasterWhisperBackend

    cfg_proc = AppConfig.model_validate(
        {"stt": {"isolation": "process", "device": "cpu"}})
    backend = build_stt(cfg_proc)
    assert isinstance(backend, WhisperProcessBackend)

    cfg_inproc = AppConfig.model_validate({"stt": {"device": "cpu"}})
    backend2 = build_stt(cfg_inproc)
    assert isinstance(backend2, FasterWhisperBackend)
