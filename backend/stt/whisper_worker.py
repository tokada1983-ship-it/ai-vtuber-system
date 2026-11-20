"""STT ワーカープロセス（faster-whisper の別プロセス化）

2026-11-20 の APPCRASH 対策 第2段。ctranslate2 とその同梱 DLL
（cudnn64_9.dll / libiomp5md.dll）をエンジン本体のプロセスから分離する
ため、WhisperModel をこの独立プロセスに閉じ込め、エンジンとは
localhost HTTP だけで通信する。このプロセスがネイティブ障害で落ちても
エンジンは無事で、クライアント側がワーカーだけを再起動する。

プロトコル:
  GET  /health     -> {"status": "ok", "pid": int, "ready": true}
  POST /transcribe -> body: PCM16LE (mono), header: X-Sample-Rate (既定16000)
                      {"text": str, "language": str, "infer_ms": int}

単体起動:
  py -3.10 -m backend.stt.whisper_worker --port 18711 --model small \
      --device cuda --compute-type float16 --language ja --parent-pid <pid>

--parent-pid を渡すと親プロセスの死活を監視し、親が消えたら自分も
終了する（エンジン異常終了時にワーカーが孤児として残らないため）。
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

_model = None
_language = None
_infer_lock = threading.Lock()


def _transcribe(audio: np.ndarray) -> tuple[str, str]:
    """生産コードと同一パラメータで推論する（同時実行はロックで直列化）"""
    with _infer_lock:
        segments, info = _model.transcribe(
            audio,
            language=_language,
            beam_size=5,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 300},
            condition_on_previous_text=False,  # 幻聴（前文脈の引きずり）を防ぐ
        )
        text = "".join(seg.text for seg in segments).strip()
        return text, getattr(info, "language", "") or ""


def _to_float32_16k(pcm16: bytes, sample_rate: int) -> np.ndarray:
    audio = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0
    if sample_rate and sample_rate != 16000:
        n = int(len(audio) * 16000 / sample_rate)
        if n > 0:
            audio = np.interp(
                np.linspace(0, len(audio) - 1, n),
                np.arange(len(audio)), audio).astype(np.float32)
    return audio


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):  # アクセスログは出さない
        pass

    def _send_json(self, code: int, obj: dict):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._send_json(200, {"status": "ok", "pid": os.getpid(),
                                  "ready": _model is not None})
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/transcribe":
            self._send_json(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length > 0 else b""
            sample_rate = int(self.headers.get("X-Sample-Rate", "16000"))
            audio = _to_float32_16k(raw, sample_rate)
            t0 = time.time()
            text, lang = _transcribe(audio)
            self._send_json(200, {
                "text": text, "language": lang,
                "infer_ms": int((time.time() - t0) * 1000),
            })
        except Exception as e:
            self._send_json(500, {"error": str(e)})


def _parent_alive(pid: int) -> bool:
    """Windows: プロセスが存在し終了していないか"""
    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    STILL_ACTIVE = 259
    kernel32 = ctypes.windll.kernel32
    handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        return False
    try:
        code = ctypes.c_ulong()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
            return False
        return code.value == STILL_ACTIVE
    finally:
        kernel32.CloseHandle(handle)


def _watch_parent(pid: int):
    while True:
        time.sleep(5)
        if not _parent_alive(pid):
            print(f"parent process {pid} is gone; worker exiting", flush=True)
            os._exit(0)


def main():
    global _model, _language
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--model", default="small")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--compute-type", default="float16")
    ap.add_argument("--language", default="ja")
    ap.add_argument("--parent-pid", type=int, default=0)
    args = ap.parse_args()

    _language = args.language or None

    from faster_whisper import WhisperModel
    t0 = time.time()
    _model = WhisperModel(args.model, device=args.device,
                          compute_type=args.compute_type)
    print(f"whisper worker ready model={args.model} device={args.device} "
          f"compute_type={args.compute_type} load_sec={time.time() - t0:.1f} "
          f"pid={os.getpid()}", flush=True)

    if args.parent_pid > 0:
        threading.Thread(target=_watch_parent, args=(args.parent_pid,),
                         daemon=True).start()

    server = ThreadingHTTPServer((args.host, args.port), _Handler)
    print(f"whisper worker listening on {args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
