"""テスト用のスタブ実装

外部サービス（LM Studio / TTSサーバ / Whisper）に依存せず、
Orchestratorの発話経路だけを検証するための差し替え部品。
"""
from __future__ import annotations

import asyncio
import tempfile
from datetime import datetime
from typing import AsyncIterator

from backend.config import AppConfig
from backend.llm.base import LLMBackend
from backend.models import Comment, LLMResponse, TTSResult
from backend.orchestrator import Orchestrator
from backend.tts.base import TTSBackend

# 標準的なLLM応答（2文に分かれる）
RAW = '{"emotion":"happy","motion":"wave","speech":"やっほー！今日も来てくれてありがとう。"}'


class FakeHub:
    """WebSocketHubの代わりに送信内容を記録する"""

    def __init__(self):
        self.events: list[tuple[str, dict]] = []

    async def broadcast(self, channel, data): self.events.append((channel, data))

    async def send_speech_start(self, text, emotion, audio_url=None, character=None, speaker=None):
        self.events.append(("speech_start",
                            {"text": text, "emotion": emotion, "audio_url": audio_url,
                             "character": character}))

    async def send_motion(self, motion, character=None):
        self.events.append(("motion", {"motion": motion, "character": character}))

    async def send_speech_end(self, character=None):
        self.events.append(("speech_end", {"character": character}))
    async def send_chat(self, role, text, **extra):
        self.events.append(("chat", {"role": role, "text": text, **extra}))
    async def send_status(self, s): pass
    async def send_comment(self, c): self.events.append(("comment", c))
    async def send_log(self, level, msg):
        self.events.append(("log", {"level": level, "message": msg}))
    async def send_unity_action(self, action, target=None):
        self.events.append(("unity_action", {"action": action, "target": target}))
    async def send_unity_move(self, x, y, z): self.events.append(("unity_move", {}))


class StubLLM(LLMBackend):
    """指定した生テキストを少しずつストリームで返すLLM"""

    def __init__(self, raw: str, streaming: bool = True, chunk: int = 4):
        self._raw, self._streaming, self._chunk = raw, streaming, chunk
        self.stream_calls = 0
        self.generate_calls = 0

    async def generate(self, system_prompt, messages, *, route="respond", temperature=None) -> LLMResponse:
        self.generate_calls += 1
        from backend.llm.parser import parse_llm_output
        return parse_llm_output(self._raw)

    async def generate_stream(self, system_prompt, messages, *, route="respond", temperature=None) -> AsyncIterator[str]:
        self.stream_calls += 1
        for i in range(0, len(self._raw), self._chunk):
            await asyncio.sleep(0)
            yield self._raw[i:i + self._chunk]

    @property
    def supports_streaming(self): return self._streaming
    @property
    def name(self): return "stub"
    @property
    def is_healthy(self): return True


class StubTTS(TTSBackend):
    """合成の呼ばれ方と同時実行数を記録するTTS"""

    def __init__(self, delay: float = 0.0, fail: bool = False, hang: bool = False):
        self.delay, self.fail, self.hang = delay, fail, hang
        self.calls: list[str] = []
        self.concurrent = 0
        self.max_concurrent = 0

    async def synthesize(self, text, emotion=None) -> TTSResult:
        self.calls.append(text)
        self.concurrent += 1
        self.max_concurrent = max(self.max_concurrent, self.concurrent)
        try:
            if self.hang:
                await asyncio.sleep(3600)
            if self.delay:
                await asyncio.sleep(self.delay)
            if self.fail:
                return TTSResult(audio_wav=b"", duration_ms=0)
            return TTSResult(audio_wav=b"RIFFfake", duration_ms=100)
        finally:
            self.concurrent -= 1

    @property
    def name(self): return "stub-tts"
    @property
    def is_healthy(self): return True


class NoAudio:
    """スピーカー出力の代わり（何もしない）"""
    async def play(self, wav): pass


def make_orchestrator(llm: StubLLM, tts: TTSBackend, **cfg_over) -> tuple[Orchestrator, FakeHub]:
    cfg = AppConfig()
    cfg.stt.enabled = False              # テストでWhisperをロードしない
    # 記憶DB・会話ログは使い捨てディレクトリへ（本物のキャラデータを汚さない）
    cfg.character.data_dir = tempfile.mkdtemp(prefix="aivt_test_")
    # テスト中に本物の ollama serve を起動・モデルをロードしない
    cfg.ollama.auto_start = False
    cfg.ollama.preload_on_start = False
    cfg.tts.irodori_server.auto_start = False   # 本物のTTSサーバを立ち上げない
    cfg.knowledge.enabled = False        # 既定ではWeb検索・ニュース取得を行わない
    cfg.orchestrator.streaming_tts = True
    cfg.tts.timeout_sec = cfg_over.pop("tts_timeout", 30.0)
    for k, v in cfg_over.items():
        setattr(cfg.orchestrator, k, v)
    hub = FakeHub()
    orch = Orchestrator(cfg, hub)
    orch._llm = llm
    orch._tts = tts
    orch._audio = NoAudio()
    # 合成した音声のキャッシュも使い捨てディレクトリへ（本物の audio_cache を消さない）
    from pathlib import Path
    orch._audio_cache = Path(tempfile.mkdtemp(prefix="aivt_audio_"))
    return orch, hub


def comment(msg: str, author: str = "視聴者") -> Comment:
    return Comment(id="c1", author=author, author_channel_id="ch1",
                   message=msg, timestamp=datetime.now())
