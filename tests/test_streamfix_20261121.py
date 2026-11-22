"""2026-11-21 配信テスト2h分析の修正（streamfix）のテスト

修正1: video.html の early_replay 再設計（静的検証。JS の振る舞い
       自体はヘッドレス検証済みで、ここでは設計パラメータと
       イベント名の退行を防ぐ）
修正4: 定型相づち同一文の差し止めが続いたら話者交代を早める
修正5: プリフェッチ破棄理由のログ（_take_prefetched）
（修正2は test_watchdog.py、修正3は test_loop_guard.py で検証）
"""
from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

import pytest
import yaml
from structlog.testing import capture_logs

from backend.config import AppConfig, CharacterConfig, EnsembleConfig
from backend.models import LLMResponse
from backend.orchestrator import Orchestrator

from helpers import FakeHub, NoAudio, StubLLM, StubTTS

# 2026-11-21 の実事象の文（定型相づち型 douiu_koto に一致する）
AIZUCHI = "あらあら、Character1ちゃん、それはどういうことなの？わたしにも教えて？"
# AIZUCHI と包含・類似の関係にない別文（同一文判定の対象外）
AIZUCHI_OTHER = "全然違う文だよ。今日は良い天気だね。"


def _json_raw(speech: str) -> str:
    return json.dumps({"emotion": "neutral", "motion": "idle",
                       "skip": False, "speech": speech}, ensure_ascii=False)


class ScriptedLLM(StubLLM):
    """呼び出しごとに返す原稿を切り替えるLLM（最後の原稿を使い続ける）。

    test_phase2_recovery の同名クラスと同じ方式。受け取った
    system_prompt を prompts に記録する。生成回数の判定には
    StubLLM 側の stream_calls / generate_calls を使う。
    """

    def __init__(self, raws: list[str]):
        super().__init__(raws[0])
        self._raws = list(raws)
        self.prompts: list[str] = []
        self._calls = 0

    def _next_raw(self, system_prompt) -> None:
        self.prompts.append(system_prompt)
        idx = min(self._calls, len(self._raws) - 1)
        self._calls += 1
        self._raw = self._raws[idx]

    async def generate_stream(self, system_prompt, messages, *,
                              route="respond", temperature=None):
        self._next_raw(system_prompt)
        async for chunk in super().generate_stream(
                system_prompt, messages, route=route):
            yield chunk

    async def generate(self, system_prompt, messages, *,
                       route="respond", temperature=None):
        self._next_raw(system_prompt)
        return await super().generate(
            system_prompt, messages, route=route, temperature=temperature)


def _write_char(dirpath: Path, name: str) -> str:
    path = dirpath / f"{name}.yaml"
    path.write_text(yaml.safe_dump(
        {"name": name, "description": f"{name}の説明",
         "personality": f"{name}の口調"}, allow_unicode=True),
        encoding="utf-8")
    return str(path)


def make_duo(llm_a=None, llm_b=None, streaming: bool = True):
    """掛け合いなし（確率0）の2体オーケストレータを作る"""
    tmp = Path(tempfile.mkdtemp(prefix="aivt_streamfix_"))
    cfg = AppConfig()
    cfg.stt.enabled = False
    cfg.orchestrator.streaming_tts = streaming
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
    a.llm, a.tts = llm_a or StubLLM("{}"), StubTTS()
    b.llm, b.tts = llm_b or StubLLM("{}"), StubTTS()
    orch._audio = NoAudio()
    orch._audio_cache = Path(tempfile.mkdtemp(prefix="aivt_audio_"))
    return orch, hub


def _speakers(hub: FakeHub) -> list[str]:
    return [d["character"] for ch, d in hub.events if ch == "speech_end"]


def _arm_aizuchi_cooldown(orch: Orchestrator) -> None:
    """定型相づちが1回発話された直後の状態（全パターンにクールダウン）"""
    until = time.time() + Orchestrator._AIZUCHI_COOLDOWN_SEC
    for pid, _pat in Orchestrator._AIZUCHI_PATTERNS:
        orch._aizuchi_cooldown_until[pid] = until


# ── 修正1: video.html の early_replay 再設計（静的検証） ──────

def test_video_html_early_replay_redesigned():
    html = (Path(__file__).resolve().parent.parent
            / "frontend" / "video.html").read_text(encoding="utf-8")
    # 進捗確認まで繰り返す方式の定数（2026-11-22 panelfix でペース
    # 判定に変更: 「経過時間に対する再生位置の割合」で progressed を
    # 判定し、0.25秒まで進んで凍結する型の誤判定を防ぐ）
    assert "EARLY_REPLAY_PACE_RATIO = 0.5" in html
    assert "EARLY_REPLAY_MAX_CHECKS = 4" in html
    assert "function watchEarlyReplay" in html
    assert "watchEarlyReplay(url, els[front])" in html
    # 判定結果（発火・不発火の理由）を残すイベント
    assert "early_replay_check" in html
    assert "result=progressed" in html
    assert "'lagging'" in html
    # 旧・単発チェック（再生位置 <0.05 の即時判定）が残っていないこと
    assert "(startedEl.currentTime || 0) < 0.05" not in html


# ── 修正4: 定型相づち同一文の早期話者交代 ────────────────────

def test_note_aizuchi_drop_counts_same_sentence():
    orch, _hub = make_duo()
    orch._note_aizuchi_drop(AIZUCHI)
    assert orch._aizuchi_drop_count == 1
    assert orch._aizuchi_early_switch is False
    orch._note_aizuchi_drop(AIZUCHI)
    assert orch._aizuchi_drop_count == 2
    assert orch._aizuchi_early_switch is True


def test_note_aizuchi_drop_different_sentence_resets():
    orch, _hub = make_duo()
    orch._note_aizuchi_drop(AIZUCHI)
    orch._note_aizuchi_drop(AIZUCHI_OTHER)   # 別文では数え直し
    assert orch._aizuchi_drop_count == 1
    assert orch._aizuchi_early_switch is False
    orch._note_aizuchi_drop(AIZUCHI_OTHER)
    assert orch._aizuchi_drop_count == 2
    assert orch._aizuchi_early_switch is True


@pytest.mark.asyncio
async def test_repeated_aizuchi_switches_speaker_early():
    """同一の相づち文が drop→divert されたら、ヒント再生成を挟まず
    1回目の失敗で話者が交代すること（実事象の再現）"""
    llm_a = ScriptedLLM([_json_raw(AIZUCHI)])
    llm_b = ScriptedLLM([_json_raw("ミライの新しい話題だよ。別の切り口で話すね。")])
    orch, hub = make_duo(llm_a=llm_a, llm_b=llm_b)
    _arm_aizuchi_cooldown(orch)

    await orch._speak_idle()

    assert _speakers(hub) == ["mirai"]
    # クロエのストリーミング生成は1回だけ（従来フローではヒント付き
    # 再生成でもう1サイクル回ってから交代する）
    assert llm_a.stream_calls == 1
    assert orch._speaker_cooldown_until.get("chloe", 0) > time.time()
    assert orch._aizuchi_early_switch is False   # 消費済み


@pytest.mark.asyncio
async def test_single_aizuchi_divert_keeps_normal_recovery():
    """相づちの差し止めが同一文1回だけなら、従来どおりヒント付き
    再生成で同じ話者が回復すること（早期交代は2回から）。

    非ストリーミング経路では差し止めは divert の1回だけになるため、
    「1回」を再現できる。ストリーミング経路では drop＋divert で
    1生成のうちに2回へ達し、早期交代が発動する（上のテスト）。"""
    llm_a = ScriptedLLM([
        _json_raw(AIZUCHI),
        _json_raw("新しい話題を持ってきたよ。今日は天気の話をしよう。")])
    orch, hub = make_duo(llm_a=llm_a, streaming=False)
    _arm_aizuchi_cooldown(orch)

    await orch._speak_idle()

    assert _speakers(hub) == ["chloe"]
    assert llm_a.generate_calls == 2   # ヒント再生成が回った
    assert "言い回しの変更" in llm_a.prompts[1]


# ── 修正5: プリフェッチ破棄理由のログ ────────────────────────

def _set_prefetch(orch: Orchestrator, *, at: float, agent_key: str,
                  speech: str = "先行生成した発話だよ。") -> None:
    orch._prefetched = {
        "at": at, "agent_key": agent_key, "topic": None,
        "response": LLMResponse(speech=speech), "actions": [],
        "premise_speech": "",
    }


def test_take_prefetched_logs_expired_reason():
    orch, _hub = make_duo()
    speaker = orch._lead
    _set_prefetch(orch, at=time.time() - 1000, agent_key=speaker.key)
    with capture_logs() as logs:
        assert orch._take_prefetched(speaker) is None
    hits = [e for e in logs if e.get("event") == "prefetch discarded"]
    assert hits and hits[0]["reason"] == "expired"


def test_take_prefetched_logs_speaker_changed_reason():
    orch, _hub = make_duo()
    speaker = orch._lead
    _set_prefetch(orch, at=time.time(), agent_key="someone_else")
    with capture_logs() as logs:
        assert orch._take_prefetched(speaker) is None
    hits = [e for e in logs if e.get("event") == "prefetch discarded"]
    assert hits and hits[0]["reason"] == "speaker changed"


def test_take_prefetched_usable_returns_without_discard_log():
    orch, _hub = make_duo()
    speaker = orch._lead
    _set_prefetch(orch, at=time.time(), agent_key=speaker.key)
    with capture_logs() as logs:
        pf = orch._take_prefetched(speaker)
    assert pf is not None
    assert not [e for e in logs if e.get("event") == "prefetch discarded"]


def test_note_aizuchi_drop_matches_fragments_of_same_speech():
    """ストリーミングで分割された断片と全体文は同一文として数えること。

    実事象では1文が「？」で2断片に分かれて drop され、続いて全体文が
    divert された。断片→全体文の包含一致で同一文と判定できないと、
    生成1回の中で2回に届かず早期交代が発動しない。"""
    orch, _hub = make_duo()
    orch._note_aizuchi_drop("あらあら、Character1ちゃん、それはどういうことなの？")
    assert orch._aizuchi_early_switch is False
    orch._note_aizuchi_drop(AIZUCHI)   # 全体文（断片を含む）
    assert orch._aizuchi_drop_count == 2
    assert orch._aizuchi_early_switch is True
