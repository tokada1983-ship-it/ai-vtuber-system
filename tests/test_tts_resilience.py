"""TTSサーバが落ちた/ハングしたときの縮退動作のテスト

ローカルTTSサーバ（VOICEVOX / Irodori 等）はハングすることがある。
そのとき配信全体が止まらず、字幕だけで進行し、復帰したら自動で音声に戻ること。
"""
import asyncio

import pytest

from helpers import RAW, StubLLM, StubTTS, make_orchestrator

# RAW は2文（「やっほー！」「今日も来てくれてありがとう。」）に分割される。
# 音声が無いときは字幕の表示時間ぶんだけ待つので、その合計をここに置く。
SUBTITLE_READ_SEC = 0.6 + 1.68


class FlakyTTS(StubTTS):
    """最初のN回はハングし、その後は正常に返すTTS"""

    def __init__(self, hang_times: int):
        super().__init__()
        self._remaining = hang_times

    async def synthesize(self, text, emotion=None):
        self.calls.append(text)
        if self._remaining > 0:
            self._remaining -= 1
            await asyncio.sleep(3600)
        from backend.models import TTSResult
        return TTSResult(audio_wav=b"RIFFfake", duration_ms=100)


@pytest.mark.asyncio
async def test_one_utterance_waits_for_at_most_one_timeout():
    """2文喋る発話でも、TTSハング時の待ちはタイムアウト1回ぶんに収まること"""
    timeout = 2.0
    orch, _ = make_orchestrator(StubLLM(RAW), StubTTS(hang=True), tts_timeout=timeout)
    orch._TTS_FAIL_LIMIT = 99          # 字幕のみモードの影響を受けないようにする

    started = asyncio.get_event_loop().time()
    await asyncio.wait_for(orch._generate_and_speak("sys"), timeout=30)
    elapsed = asyncio.get_event_loop().time() - started

    assert orch._tts_timeouts == 1                       # 2文あってもタイムアウトは1回
    assert elapsed < timeout * 2 + SUBTITLE_READ_SEC     # 2回ぶんは待っていない


@pytest.mark.asyncio
async def test_enters_subtitle_only_mode_after_repeated_timeouts():
    """発話をまたいで連続失敗したら、以降はタイムアウトすら待たなくなること"""
    timeout = 2.0
    orch, _ = make_orchestrator(StubLLM(RAW), StubTTS(hang=True), tts_timeout=timeout)
    orch._TTS_FAIL_LIMIT = 2

    for _ in range(2):
        await asyncio.wait_for(orch._generate_and_speak("sys"), timeout=30)
    assert orch._tts_skip_until > 0        # 字幕のみモードに入っている

    # 3回目の発話ではTTSを呼ばず、字幕の表示時間だけで完了する
    tts3 = StubTTS(hang=True)
    orch._tts = tts3
    started = asyncio.get_event_loop().time()
    await asyncio.wait_for(orch._generate_and_speak("sys"), timeout=30)
    elapsed = asyncio.get_event_loop().time() - started

    assert tts3.calls == []                          # 合成を試みていない
    assert elapsed < SUBTITLE_READ_SEC + 1.0         # 字幕の表示時間しか経っていない


@pytest.mark.asyncio
async def test_recovers_when_tts_comes_back():
    """TTSが復帰したら、次の発話から音声つきに戻ること"""
    tts = FlakyTTS(hang_times=1)
    orch, hub = make_orchestrator(StubLLM(RAW), tts, tts_timeout=0.2)
    orch._TTS_FAIL_LIMIT = 99              # 字幕のみモードには入らせない

    await asyncio.wait_for(orch._generate_and_speak("sys"), timeout=30)   # ハングする発話
    assert orch._tts_timeouts == 1

    hub.events.clear()
    await asyncio.wait_for(orch._generate_and_speak("sys"), timeout=30)   # 復帰後の発話

    audio_events = [d for k, d in hub.events if k == "speech_start" and d.get("audio_url")]
    assert audio_events, "復帰後は音声つきで再生されるはず"
    assert orch._tts_timeouts == 0         # 成功でカウンタがリセットされている


@pytest.mark.asyncio
async def test_subtitle_is_not_sent_twice_for_first_sentence():
    """TTS失敗時に1文目の字幕が二重送信されないこと"""
    orch, hub = make_orchestrator(StubLLM(RAW), StubTTS(fail=True))

    await orch._generate_and_speak("sys")

    sends = [d for k, d in hub.events
             if k == "speech_start" and d.get("text") == "やっほー！"]
    assert len(sends) == 1
