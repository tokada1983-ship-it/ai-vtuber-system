"""発話ループ検出（backend/loop_guard.py）のテスト。

検出器単体（反復で発火／多様では不発火の両方向、クールダウン、昇格、
無効化）と、Orchestrator 統合（_after_speech 経由の介入セット、ヒント
消費、昇格時の文脈リセット、計測露出）を検証する。
"""
from __future__ import annotations

import time

from backend.config import AppConfig
from backend.loop_guard import SpeechLoopGuard
from backend.models import LLMResponse

from tests.helpers import StubLLM, StubTTS, RAW, make_orchestrator

# 多様な話題の発話（相互に似ないよう語彙を散らしてある）
DIVERSE = [
    "今日の天気は快晴で、洗濯物がよく乾きそうですね。",
    "昨夜作ったカレーライス、隠し味にチョコを入れたら深みが出ました。",
    "新しい自転車のブレーキ調整、思ったより繊細な作業でしたよ。",
    "駅前の書店で、深海生物の図鑑を衝動買いしてしまいました。",
    "ピアノの練習曲、左手のパートがやっと弾けるようになったんです。",
    "週末は渓流でキャンプをして、焚き火で焼き芋を作る予定です。",
    "古いフィルムカメラの現像、仕上がりを見る瞬間が一番わくわくします。",
    "庭のトマトが赤く色づいて、収穫のタイミングを計っているところです。",
    "電車の窓から見えた虹、二重になっていて珍しかったですね。",
    "宇宙食のアイスクリームって、実際はどんな味なんでしょう。",
    "編み物のセーター、袖の長さを間違えて編み直しになりました。",
    "港町で食べた海鮮丼、ネタの厚さにびっくりしましたよ。",
    "将棋の詰め手筋、昨日の問題は三手詰めで解けました。",
    "水族館のクラゲ展示、照明が幻想的でずっと見ていられます。",
    "登山靴のソールが剥がれて、修理に出すかどうか迷っています。",
    "手紙を書くための万年筆、インクの色は青墨にしました。",
    "時計塔の鐘の音、正午に十二回鳴るのが町の合図なんです。",
    "傘の骨が折れたので、骨董市で丈夫なものを探しています。",
    "写真展の受付で、来場者から構図の質問をたくさん受けました。",
    "雪国の温泉宿、露天風呂から見る星空が格別だそうです。",
    "畑の土づくりには、落ち葉を混ぜると良いと聞きました。",
    "オーケストラの演奏会、管楽器の音色がホールに響いていました。",
    "陶芸教室で作った湯呑み、釉薬の色合いが狙い通りに出ました。",
    "古地図を広げて、今は無くなった街道の跡を辿ってみました。",
]

# 2026-11-20 のループ区間を模した、言い換えローテーションの発話群
LOOP_CORE = [
    "ボクもみんなと一緒に過ごせて、とっても幸せです！",
    "あらあら、Character1ちゃんもそう言ってもらえて、私も嬉しいです！",
    "ボクも、こうして一緒に頑張れることがとっても幸せです！",
    "あらあら、Character1ちゃん、元気です。わたしも一緒に頑張りますね！",
    "本当ですよ！ボクも、こうして一緒に頑張れることがとっても幸せです！",
    "はい、Character1ちゃん、よろしくお願いしますね。",
    "えっへん、ボクに任せなさい！今日も元気に配信していこうね！",
    "あらあら、Character1ちゃんもそう言ってもらえて、わたしも嬉しいですね！",
    "ボクもみんなと一緒に過ごせて、とっても幸せです！",
    "もちろんです！ボクが全力で精一杯に解説しますよ！",
    "あらあら、Character1ちゃん、それは嬉しいですね！みなさんも嬉しいでしょう？",
    "ボクも、こうして一緒に頑張れることがとっても幸せです！",
    "えっへん、ボクに任せなさい！今日も元気に配信していこうね！",
    "あらあら、Character1ちゃんもそう言ってもらえて、私も嬉しいです！",
    "もう、ボクもう最高に幸せな気分です！Character2も、みんなも、最高ですよ！",
    "はい、Character1ちゃん、よろしくお願いしますね。",
    "ボクもみんなと一緒に過ごせて、とっても幸せです！",
    "もちろんです！ボクが全力で精一杯に解説しますよ！",
]

BASE_TS = 1_700_000_000.0


def _feed(guard: SpeechLoopGuard, speeches, start=BASE_TS, step=40.0):
    """時刻を進めながら発話を流し、(index, event) のリストを返す"""
    events = []
    for i, text in enumerate(speeches):
        ev = guard.note_speech(text, now=start + i * step)
        if ev is not None:
            events.append((i, ev))
    return events


# ── 検出器単体 ──────────────────────────────────────────────

def test_loop_sequence_fires():
    guard = SpeechLoopGuard()
    events = _feed(guard, DIVERSE[:8] + LOOP_CORE)
    assert len(events) == 1
    _i, ev = events[0]
    assert ev.level == 1
    assert ev.dup_count >= guard.dup_min_count
    assert ev.word  # 代表語が取れている
    assert guard.stats()["fires"] == 1


def test_diverse_sequence_never_fires():
    guard = SpeechLoopGuard()
    events = _feed(guard, DIVERSE)
    assert events == []
    assert guard.stats()["recent_dup_count"] == 0


def test_callback_repetition_does_not_fire():
    """定番フレーズの単発コールバック（3回まで）は誤発火しない"""
    greeting = "えっへん、ボクに任せなさい！今日も元気に配信していこうね！"
    seq = (DIVERSE[:6] + [greeting] + DIVERSE[6:12] + [greeting]
           + DIVERSE[12:18] + [greeting] + DIVERSE[18:24])
    guard = SpeechLoopGuard()
    events = _feed(guard, seq)
    assert events == []


def test_requires_full_window():
    """窓が埋まるまでは、同じ文の連続でも発火しない"""
    guard = SpeechLoopGuard()
    line = "ボクもみんなと一緒に過ごせて、とっても幸せです！"
    events = _feed(guard, [line] * 10)
    assert events == []
    events = _feed(guard, [line] * 8, start=BASE_TS + 10 * 40)
    assert len(events) == 1


def test_cooldown_and_escalation():
    guard = SpeechLoopGuard()
    events = _feed(guard, DIVERSE[:8] + LOOP_CORE)
    assert len(events) == 1
    fire_ts = BASE_TS + events[0][0] * 40
    # クールダウン中は再演が続いても発火しない
    events2 = _feed(guard, LOOP_CORE[:4], start=fire_ts + 40)
    assert events2 == []
    # クールダウン明け（昇格窓の中）の再発はレベル2
    events3 = _feed(guard, LOOP_CORE[:4], start=fire_ts + 601)
    assert len(events3) == 1
    assert events3[0][1].level == 2
    stats = guard.stats()
    assert stats["fires"] == 2
    assert stats["last_level"] == 2


def test_disabled_never_fires():
    guard = SpeechLoopGuard(enabled=False)
    events = _feed(guard, DIVERSE[:8] + LOOP_CORE)
    assert events == []
    # 無効な間は記録もしない
    assert guard.stats()["recent_dup_count"] == 0


def test_config_defaults():
    cfg = AppConfig()
    lg = cfg.orchestrator.loop_guard
    assert lg.enabled is True
    assert lg.window_speeches == 16
    assert lg.dup_min_count == 6


# ── Orchestrator 統合 ───────────────────────────────────────

def _make():
    return make_orchestrator(StubLLM(RAW), StubTTS())


def _speak(orch, speeches):
    agents = orch._agents
    for i, text in enumerate(speeches):
        orch._after_speech(agents[i % len(agents)],
                            LLMResponse(speech=text), idle=True)


def test_orchestrator_arms_intervention_on_loop():
    orch, _hub = _make()
    _speak(orch, DIVERSE[:8] + LOOP_CORE)
    assert orch._loop_hint_pending is True
    assert orch._force_switch_pending is True
    status = orch.knowledge_status()
    assert status["loop_guard"]["fires"] >= 1
    assert status["loop_guard"]["enabled"] is True


def test_orchestrator_diverse_no_intervention():
    orch, _hub = _make()
    _speak(orch, DIVERSE)
    assert orch._loop_hint_pending is False
    assert orch.knowledge_status()["loop_guard"]["fires"] == 0


def test_loop_hint_consumed_once():
    orch, _hub = _make()
    _speak(orch, DIVERSE[:8] + LOOP_CORE)
    assert orch._loop_hint_pending is True
    out = orch._consume_loop_hint("BASE")
    assert out.startswith("BASE")
    assert "立て直し" in out
    # 1回だけ: 2回目は素通し
    assert orch._consume_loop_hint("BASE") == "BASE"


def test_second_fire_escalates_to_context_reset():
    orch, _hub = _make()
    _speak(orch, DIVERSE[:8] + LOOP_CORE)
    assert orch._loop_guard.stats()["fires"] == 1
    assert orch._last_context_reset_at == 0.0
    # クールダウンが明けたことにする（昇格窓の中）
    orch._loop_guard._last_fire_at = time.time() - 601
    _speak(orch, LOOP_CORE[:3])
    assert orch._loop_guard.stats()["last_level"] == 2
    assert orch._last_context_reset_at > 0.0
