"""Phase 2 の回帰テスト: 反復ブロック回復フロー＋話題の強制力

2026-11-17 の1時間テストで発生した問題への対策:
  - Character2の反復ブロック32回・約9分半×2回の無言（ライブロック）
  - 話題（調査テーマ）の張り付き（プランナの強制力不足）

外部サービスには依存せず、スタブバックエンドで検証する。
"""
from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from backend.config import AppConfig, CharacterConfig, EnsembleConfig
from backend.knowledge.models import WebDocument
from backend.knowledge.topic_planner import Topic, TopicPlanner
from backend.orchestrator import Orchestrator
from helpers import RAW, FakeHub, NoAudio, StubLLM, StubTTS, make_orchestrator

RAW_A = '{"emotion":"happy","motion":"wave","speech":"わらわが答えてやろう。"}'
RAW_B = '{"emotion":"surprised","motion":"nod","speech":"えっ、そうなの？"}'

REPEAT_SPEECH = "同じ話を繰り返すよ。反復チェック用の文だよ。"


def _json_raw(speech: str) -> str:
    return json.dumps({"emotion": "neutral", "motion": "idle",
                       "skip": False, "speech": speech}, ensure_ascii=False)


def _write_char(dirpath: Path, name: str) -> str:
    path = dirpath / f"{name}.yaml"
    path.write_text(yaml.safe_dump(
        {"name": name, "description": f"{name}の説明",
         "personality": f"{name}の口調"}, allow_unicode=True), encoding="utf-8")
    return str(path)


class ScriptedLLM(StubLLM):
    """呼び出しごとに返す原稿を切り替えるLLM（最後の原稿を使い続ける）。

    受け取った system_prompt を prompts に記録する。
    """

    def __init__(self, raws: list[str]):
        super().__init__(raws[0])
        self._raws = list(raws)
        self.prompts: list[str] = []

    async def generate_stream(self, system_prompt, messages, *, route="respond", temperature=None):
        self.prompts.append(system_prompt)
        idx = min(len(self.prompts) - 1, len(self._raws) - 1)
        self._raw = self._raws[idx]
        async for chunk in super().generate_stream(system_prompt, messages,
                                                  route=route):
            yield chunk


class FakePlanner:
    """話題プランナの記録用スタブ（材料ありの話題を返す）"""

    def __init__(self):
        self.plan_calls: list[dict] = []
        self.committed: list = []
        self.released: list = []
        self.banned: list[tuple[str, float]] = []
        self.reset_thread_calls = 0
        self.last_comment_speech_at = 0.0

    async def plan(self, kind="", query="", allow_web=True, context=None):
        self.plan_calls.append({"kind": kind, "query": query})
        doc = WebDocument(title="テスト記事", url="https://example.com/1",
                          snippet="テスト用の資料テキストです。")
        return Topic(kind=kind or "news", label="テスト話題",
                     documents=[doc], topic_id=f"fake-{len(self.plan_calls)}")

    def commit(self, topic):
        self.committed.append(topic)

    def release(self, topic):
        self.released.append(topic)

    def ban_word(self, word, seconds=1800):
        self.banned.append((word, seconds))

    def banned_words(self):
        return {w for w, _ in self.banned}

    def reset_thread(self):
        self.reset_thread_calls += 1

    def note_speech(self, idle=False, text=""):
        pass

    def note_comment(self, author, text):
        pass


def make_duo_p2(llm_a=None, llm_b=None, fallback_a=(), fallback_b=(),
                **ensemble):
    """掛け合いなし（確率0）の2体オーケストレータを作る"""
    tmp = Path(tempfile.mkdtemp(prefix="aivt_p2_"))
    cfg = AppConfig()
    cfg.stt.enabled = False
    cfg.ollama.auto_start = False
    cfg.ollama.preload_on_start = False
    cfg.tts.irodori_server.auto_start = False
    cfg.knowledge.enabled = False
    cfg.characters = [
        CharacterConfig(name="Character1", id="Character1", data_dir=str(tmp),
                        prompt_file=_write_char(tmp, "Character1"),
                        aliases=["character1"], fallback_lines=list(fallback_a)),
        CharacterConfig(name="Character2", id="Character2", data_dir=str(tmp),
                        prompt_file=_write_char(tmp, "Character2"),
                        fallback_lines=list(fallback_b)),
    ]
    ens = {"turn_gap_sec": 0, "idle_banter_probability": 0.0,
           "banter_probability": 0.0}
    ens.update(ensemble)
    cfg.ensemble = EnsembleConfig(**ens)
    hub = FakeHub()
    orch = Orchestrator(cfg, hub)
    tts_a, tts_b = StubTTS(), StubTTS()
    a, b = orch.agents
    a.llm, a.tts = llm_a or StubLLM(RAW_A), tts_a
    b.llm, b.tts = llm_b or StubLLM(RAW_B), tts_b
    orch._audio = NoAudio()
    orch._audio_cache = Path(tempfile.mkdtemp(prefix="aivt_audio_"))
    return orch, hub, tts_a, tts_b


def _speakers(hub: FakeHub) -> list[str]:
    return [d["character"] for ch, d in hub.events if ch == "speech_end"]


def _seed_repeat(orch: Orchestrator) -> None:
    """REPEAT_SPEECH を「直近で話した」ことにして反復ブロックを誘発する"""
    orch._recent_speech_norms.append(Orchestrator._normalize_speech(REPEAT_SPEECH))


# ── 回復フロー ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_retry_with_recovery_hint_and_topic_redraw():
    """1回目のブロックでは回復ヒント付きで再生成し、話題も引き直すこと"""
    llm = ScriptedLLM([_json_raw(REPEAT_SPEECH),
                       _json_raw("新しい話題を持ってきたよ。今日は天気の話をしよう。")])
    orch, hub, tts_a, _ = make_duo_p2(llm_a=llm)
    planner = FakePlanner()
    orch._planner = planner
    _seed_repeat(orch)

    await orch._speak_idle()

    assert len(llm.prompts) == 2                       # 再生成された
    assert "言い回しの変更" in llm.prompts[1]           # 回復ヒントが付いた
    assert "言い回しの変更" not in llm.prompts[0]
    assert len(planner.plan_calls) == 2                # 話題を引き直した
    assert len(planner.released) == 1                  # 失敗した話題は解放
    assert len(planner.committed) == 1                 # 成功した話題は消費確定
    assert _speakers(hub) == ["Character1"]
    assert tts_a.calls == ["新しい話題を持ってきたよ。", "今日は天気の話をしよう。"]
    assert orch._block_streak.get("Character1", 0) == 0     # 成功でリセット
    assert orch._total_block_streak == 0


@pytest.mark.asyncio
async def test_second_block_switches_speaker_and_cooldown_applies():
    """2回目のブロックで話者が交代し、クールダウン中は選ばれないこと"""
    llm_a = ScriptedLLM([_json_raw(REPEAT_SPEECH)])
    llm_b = ScriptedLLM([
        _json_raw("Character2の新しい話題だよ。別の切り口で話すね。"),
        _json_raw("今度はCharacter2の二つ目の話題だよ。続けていこう。"),
    ])
    orch, hub, _, _ = make_duo_p2(llm_a=llm_a, llm_b=llm_b)
    _seed_repeat(orch)

    await orch._speak_idle()

    assert _speakers(hub) == ["Character2"]                 # 相方が喋った
    assert len(llm_a.prompts) == 2                     # Character1は2回試して詰まった
    assert orch._block_streak.get("Character1") == 2
    assert orch._speaker_cooldown_until.get("Character1", 0) > time.time() + 100

    hub.events.clear()
    await orch._speak_idle()

    # クールダウン中のCharacter1は選ばれず（生成も呼ばれず）、Character2が続ける
    assert _speakers(hub) == ["Character2"]
    assert len(llm_a.prompts) == 2


@pytest.mark.asyncio
async def test_fallback_line_spoken_when_all_attempts_fail():
    """全試行が詰まっても、フォールバック台詞で無言を回避すること"""
    llm_a = ScriptedLLM([_json_raw(REPEAT_SPEECH)])
    llm_b = ScriptedLLM([_json_raw(REPEAT_SPEECH)])
    orch, hub, _, tts_b = make_duo_p2(
        llm_a=llm_a, llm_b=llm_b,
        fallback_a=["Character1の繋ぎ台詞だよ。場を温めるね。"],
        fallback_b=["Character2の繋ぎ台詞だよ。のんびり行こう。"])
    _seed_repeat(orch)

    await orch._speak_idle()

    # 話者交代後のCharacter2のプールから1本喋り、通常の発話として履歴に残る
    assert _speakers(hub) == ["Character2"]
    assert tts_b.calls == ["Character2の繋ぎ台詞だよ。のんびり行こう。"]
    assert Orchestrator._normalize_speech("Character2の繋ぎ台詞だよ。のんびり行こう。") \
        in orch._recent_speech_norms


def test_block_streak_escalates_to_context_reset_at_six():
    """合算の連続ブロックが6回で文脈リセットへエスカレーションすること"""
    orch, _ = make_orchestrator(StubLLM(RAW), StubTTS())
    agent = orch._lead
    calls: list[str] = []
    orch._reset_saturated_context = lambda word: calls.append(word)

    for _ in range(5):
        orch._note_blocked(agent)
    assert calls == []
    assert orch._total_block_streak == 5

    orch._note_blocked(agent)
    assert len(calls) == 1
    assert orch._total_block_streak == 0               # 発火時点で戻る


# ── 文型レベルの反復 ────────────────────────────────────────

def test_is_formulaic_opening():
    orch, _ = make_orchestrator(StubLLM(RAW), StubTTS())
    assert orch._is_formulaic_opening("はいはい、すごいですね。")
    assert orch._is_formulaic_opening("あらあら、すごいですね。")
    # 内容が続けば定型文型ではない
    assert not orch._is_formulaic_opening("はいはい、すごいですね。Character2も精一杯ですね。")
    assert not orch._is_formulaic_opening("今日は良い天気ですね。")
    # 相づちが無ければ対象外
    assert not orch._is_formulaic_opening("すごいですね。")


@pytest.mark.asyncio
async def test_formulaic_line_diverted_not_counted_as_block():
    """文型反復は喋らせずに回復リトライへ回し、ブロックとしては数えないこと"""
    llm = ScriptedLLM([_json_raw("はいはい、すごいですね。"),
                       _json_raw("新しい話題だよ。天気の話をしよう。")])
    orch, hub, tts_a, _ = make_duo_p2(llm_a=llm)

    await orch._speak_idle()

    assert tts_a.calls == ["新しい話題だよ。", "天気の話をしよう。"]
    assert "言い回しの変更" in llm.prompts[1]
    assert orch._block_streak.get("Character1", 0) == 0
    assert orch._total_block_streak == 0


# ── 話題の強制力（案A・B・C）────────────────────────────────

def test_theme_detection_bans_label_then_gram_on_escalation():
    """テーマ反復の検知で ban が呼ばれ、再検知では gram まで広がること"""
    orch, _ = make_orchestrator(StubLLM(RAW), StubTTS())
    planner = FakePlanner()
    orch._planner = planner

    orch._note_theme_repetition("調査の結果を報告しますね")
    orch._note_theme_repetition("調査スキルを磨きたいな")
    orch._note_theme_repetition("調査結果はこちらです")

    # 初回検知: ラベル（表層フレーズ）を600秒 ban、強制切替はまだ
    assert planner.banned == [("調査結果", 600)]
    assert orch._force_switch_pending is False

    # 抑制期間が明けて同じテーマが再検知されたらエスカレーション
    orch._theme_hint_fired["調査結果"] = time.time() - 601
    orch._note_theme_repetition("調査結果をもう一度見ます")

    assert planner.banned == [("調査結果", 600), ("調査結果", 600), ("調査", 600)]
    assert orch._force_switch_pending is True


def test_banned_word_excluded_from_memory_seed(monkeypatch):
    """ban された語を含む記憶は、自律トークの材料から外れること"""
    orch, _, _, _ = make_duo_p2()
    orch._planner = TopicPlanner(news=None, search=None, state_path="")
    a, b = orch.agents

    entry1 = SimpleNamespace(text="調査結果についての長い思い出話です",
                             source="episodic")
    monkeypatch.setattr(a.memory_store, "sample_recent",
                        lambda limit=20: [entry1])
    monkeypatch.setattr(b.memory_store, "sample_recent", lambda limit=20: [])

    seed = orch._memory_topic_seed()
    assert seed is not None and "調査" in seed["text"]

    orch._planner.ban_word("調査", seconds=600)
    entry2 = SimpleNamespace(text="調査の思い出はまだまだたくさんあります",
                             source="episodic")
    entry3 = SimpleNamespace(text="天気の良い日の散歩の思い出です",
                             source="episodic")
    monkeypatch.setattr(a.memory_store, "sample_recent",
                        lambda limit=20: [entry2, entry3])

    seed = orch._memory_topic_seed()
    assert seed is not None
    assert seed["text"] == entry3.text                 # 調査を含む方は選ばれない


def test_memory_score_cooldown_and_decay():
    """memory のスコアに同種クールダウンと鮮度減衰が効くこと（案B）"""
    planner = TopicPlanner(news=None, search=None, state_path="",
                           weights={"memory": 1.6, "news": 2.0,
                                    "curiosity": 1.0})
    now = time.time()

    # 直後に memory を話した: 鮮度 0.45 × クールダウン 0.4
    planner._last_kind_at = {"memory": now}
    planner._ledger = [{"kind": "memory"}]
    scores = planner._score_kinds(allow_web=False, context={})
    assert scores["memory"] == pytest.approx(1.6 * 0.45 * 0.4, rel=0.02)

    # 240秒前・直近は別種: 鮮度だけ（0.45 + 0.55×0.5 = 0.725）
    planner._last_kind_at = {"memory": now - 240}
    planner._ledger = [{"kind": "news"}]
    scores = planner._score_kinds(allow_web=False, context={})
    assert scores["memory"] == pytest.approx(1.6 * 0.725, rel=0.02)

    # 480秒以上前: 全快する
    planner._last_kind_at = {"memory": now - 1000}
    scores = planner._score_kinds(allow_web=False, context={})
    assert scores["memory"] == pytest.approx(1.6, rel=0.02)


@pytest.mark.asyncio
async def test_force_switch_forces_news_kind():
    """エスカレーション後の idle は kind=news が強制され、スレッドも捨てること"""
    orch, hub, *_ = make_duo_p2()
    planner = FakePlanner()
    orch._planner = planner
    orch._news = SimpleNamespace(unused_items=lambda: [object()])
    orch._force_switch_pending = True

    await orch._speak_idle()

    assert planner.reset_thread_calls == 1
    assert planner.plan_calls[0]["kind"] == "news"
    assert orch._force_switch_pending is False
    assert _speakers(hub) == ["Character1"]


@pytest.mark.asyncio
async def test_force_switch_uses_curiosity_without_news_stock():
    """ニュース在庫が無ければ kind=curiosity を強制すること"""
    orch, hub, *_ = make_duo_p2()
    planner = FakePlanner()
    orch._planner = planner
    orch._news = None
    orch._search = object()                            # 検索は使える想定
    orch._force_switch_pending = True

    await orch._speak_idle()

    assert planner.reset_thread_calls == 1
    assert planner.plan_calls[0]["kind"] == "curiosity"
