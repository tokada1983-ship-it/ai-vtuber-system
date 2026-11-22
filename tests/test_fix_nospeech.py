"""2026-11-19 問題2対策のテスト: タグ方式 "No speech in tag output" 多発

配信テストで失敗120件を観測。観測追加後のスモークで原因が確定した:
プロンプトが num_ctx=4096 に達し、生成の残り枠が数トークンしかなく、
発話がヘッダ途中で done_reason=length により切断されていた
（思考チャネルでもヘッダ直後EOSでもない）。

検証する対策:
- 計測: 両タグ完備＋本文なしを ok から分離し no_speech として数える
- 観測: no-speech の診断フィールド（strip前プレビュー・思考除去量・
  done_reason / eval_count / prompt_eval_count）
- ストリーミング無発話の計測と報告（report_no_speech）
- Ollama ペイロードへの num_ctx 伝搬（コンテキスト枠の拡張）
- フォールバック差別化: 一括再試行は温度を下げ、length 切断時は
  文脈も切り詰める
- 回復ポリシー: no-speech 系の失敗では温度を上げず、ターン内の
  no-speech 失敗が上限に達したら生成を打ち切る
- 生成予算: idle ターン内の生成回数に上限がある
"""
from __future__ import annotations

from backend.config import LLMBackendConfig
from backend.knowledge.models import WebDocument
from backend.knowledge.topic_planner import Topic
from backend.llm.ollama_backend import OllamaBackend
from backend.llm.parser import (TAG_OUTCOME_FATAL, TAG_OUTCOME_NO_SPEECH,
                                is_tag_failure_outcome, parse_tag_output,
                                tag_no_speech_fields, tag_outcome_of,
                                tag_parse_outcome_stats)
from backend.llm.streaming import TagSpeechStreamParser
from backend.models import ChatMessage
from helpers import StubLLM, StubTTS, make_orchestrator

VALID_TAG = "[emo:happy][mot:wave]やっほー！今日も来てくれてありがとう。"
PARTIAL_TAG = "[emo:happy][mot:"


def _delta(before: dict, after: dict) -> dict:
    return {k: after.get(k, 0) - before.get(k, 0) for k in after}


# ── 計測の是正 ───────────────────────────────────────────────

class TestNoSpeechMetrics:
    def test_header_only_counts_no_speech_not_ok(self):
        before = tag_parse_outcome_stats()
        resp = parse_tag_output("[emo:neutral][mot:idle]")
        after = tag_parse_outcome_stats()
        assert resp.skip
        d = _delta(before, after)
        assert d["no_speech"] == 1
        assert d["ok"] == 0 and d["fatal"] == 0

    def test_partial_header_counts_fatal(self):
        before = tag_parse_outcome_stats()
        resp = parse_tag_output(PARTIAL_TAG)
        after = tag_parse_outcome_stats()
        assert resp.skip
        assert _delta(before, after)["fatal"] == 1

    def test_no_speech_fields_carry_diagnostics(self):
        raw = "<think>考え中</think>[emo:hap"
        meta = {"done_reason": "length", "eval_count": 7,
                "prompt_eval_count": 4091, "thinking_chars": 0}
        fields = tag_no_speech_fields(raw, meta)
        # strip 前の生出力が残る（従来は strip 後の断片だけだった）
        assert fields["raw_preview"] == raw
        assert fields["after_strip_preview"] == "[emo:hap"
        assert fields["think_stripped_chars"] > 0
        assert fields["done_reason"] == "length"
        assert fields["eval_count"] == 7
        assert fields["prompt_eval_count"] == 4091


# ── ストリーミング無発話の計測 ───────────────────────────────

class TestStreamReportNoSpeech:
    def test_header_only_stream_reports_no_speech(self):
        before = tag_parse_outcome_stats()
        parser = TagSpeechStreamParser()
        parser.feed("[emo:neutral][mot:idle]")
        parser.flush()
        outcome = parser.report_no_speech()
        after = tag_parse_outcome_stats()
        assert outcome == "no_speech"
        assert _delta(before, after)["no_speech"] == 1

    def test_partial_header_stream_reports_fatal(self):
        before = tag_parse_outcome_stats()
        parser = TagSpeechStreamParser()
        parser.feed(PARTIAL_TAG)
        parser.flush()
        outcome = parser.report_no_speech({"done_reason": "length"})
        after = tag_parse_outcome_stats()
        assert outcome == "fatal"
        assert _delta(before, after)["fatal"] == 1

    def test_speech_stream_reports_nothing(self):
        before = tag_parse_outcome_stats()
        parser = TagSpeechStreamParser()
        parser.feed(VALID_TAG)
        parser.flush()
        assert parser.report_no_speech() == ""
        assert tag_parse_outcome_stats() == before


# ── Ollama ペイロード（num_ctx 伝搬） ────────────────────────

class TestOllamaPayloadNumCtx:
    def test_num_ctx_propagated_to_options(self):
        cfg = LLMBackendConfig(
            base_url="http://localhost:11434", model="m",
            extra_payload={"think": False, "num_ctx": 8192})
        payload = OllamaBackend(cfg)._build_payload("sys", [], stream=False)
        assert payload["options"]["num_ctx"] == 8192
        assert payload["think"] is False

    def test_num_ctx_absent_by_default(self):
        cfg = LLMBackendConfig(
            base_url="http://localhost:11434", model="m",
            extra_payload={"think": False})
        payload = OllamaBackend(cfg)._build_payload("sys", [], stream=True)
        assert "num_ctx" not in payload["options"]


# ── フォールバック差別化 ─────────────────────────────────────

class RecordingLLM(StubLLM):
    """ストリームと一括生成で別々の原稿を返し、生成条件を記録する"""

    def __init__(self, stream_raw: str, gen_raw: str):
        super().__init__(stream_raw)
        self._gen_raw = gen_raw
        self.generate_temperatures: list = []
        self.generate_messages: list = []

    async def generate(self, system_prompt, messages, *, route="respond",
                       temperature=None):
        self.generate_calls += 1
        self.generate_temperatures.append(temperature)
        self.generate_messages.append(list(messages))
        from backend.llm.parser import parse_tag_output
        return parse_tag_output(self._gen_raw)


class TestFallbackDifferentiation:
    async def test_fallback_lowers_temperature(self):
        llm = RecordingLLM(PARTIAL_TAG, VALID_TAG)
        orch, _hub = make_orchestrator(llm, StubTTS())
        orch._cfg.llm.protocol = "tag"
        response = await orch._generate_and_speak("sys")
        assert response is not None
        assert response.speech.startswith("やっほー！")
        # ベース 0.8 から 0.3 下げた 0.5 で再試行される（同一条件にしない）
        assert llm.generate_temperatures == [0.5]

    async def test_fallback_compacts_messages_on_length(self):
        llm = RecordingLLM(PARTIAL_TAG, VALID_TAG)
        llm._last_gen_meta = {"done_reason": "length", "eval_count": 5,
                              "prompt_eval_count": 4091}
        orch, _hub = make_orchestrator(llm, StubTTS())
        orch._cfg.llm.protocol = "tag"
        orch._lead.log.restore(
            [ChatMessage(role="user", content=f"履歴{i}") for i in range(8)])
        response = await orch._generate_and_speak("sys")
        assert response is not None
        # length 切断時は文脈を直近4件に切り詰めて生成枠を確保する
        assert len(llm.generate_messages[0]) == 4

    async def test_fallback_keeps_messages_without_length(self):
        llm = RecordingLLM(PARTIAL_TAG, VALID_TAG)
        llm._last_gen_meta = {"done_reason": "stop", "eval_count": 9}
        orch, _hub = make_orchestrator(llm, StubTTS())
        orch._cfg.llm.protocol = "tag"
        orch._lead.log.restore(
            [ChatMessage(role="user", content=f"履歴{i}") for i in range(8)])
        response = await orch._generate_and_speak("sys")
        assert response is not None
        assert len(llm.generate_messages[0]) == 8


# ── 回復ポリシー（温度上昇の停止・生成打ち切り・予算） ────────

class FakePlanner:
    """材料ありの話題を返すプランナのスタブ（phase2 テストと同形）"""

    def __init__(self):
        self.committed: list = []
        self.released: list = []

    async def plan(self, kind="", query="", allow_web=True, context=None):
        doc = WebDocument(title="テスト記事", url="https://example.com/1",
                          snippet="テスト用の資料テキストです。")
        return Topic(kind=kind or "news", label="テスト話題",
                     documents=[doc], topic_id="fake-1")

    def commit(self, topic):
        self.committed.append(topic)

    def release(self, topic):
        self.released.append(topic)

    def has_deliverable_answer(self):
        return False


class TestRecoveryPolicy:
    async def test_no_temp_raise_and_early_stop_on_no_speech(self):
        # ストリームも一括もヘッダのみで終わるLLM（全生成が no-speech 系）
        llm = RecordingLLM(PARTIAL_TAG, PARTIAL_TAG)
        tts = StubTTS()
        orch, _hub = make_orchestrator(llm, tts)
        orch._cfg.llm.protocol = "tag"
        orch._planner = FakePlanner()
        orch._lead.cfg.fallback_lines = ["つなぎの台詞だよ、みんな。"]
        await orch._speak_idle()
        # 1試行目（ストリーム＋一括）で no-speech 失敗が上限2件に達し、
        # 2試行目の生成には進まない
        assert llm.stream_calls == 1
        assert llm.generate_calls == 1
        # no-speech 系の失敗では回復温度を装填しない
        assert orch._temp_override == {}
        # それでも無言にはならず、フォールバック台詞が喋られる
        assert tts.calls

    async def test_gen_budget_consumption(self):
        orch, _hub = make_orchestrator(StubLLM(VALID_TAG), StubTTS())
        assert orch._consume_gen_budget() is True   # 予算外は常に可
        orch._gen_budget = 2
        assert orch._consume_gen_budget() is True
        assert orch._consume_gen_budget() is True
        assert orch._consume_gen_budget() is False  # 上限で止まる
        orch._gen_budget = None
        assert orch._consume_gen_budget() is True

    async def test_tag_outcome_attributed_per_generation(self):
        # F3: 失敗の帰属は生成単位（応答に刻まれたパース区分）で取る。
        # 共有カウンタの前後差分は、並行生成の失敗と取り違えるため廃止した
        orch, _hub = make_orchestrator(StubLLM(VALID_TAG), StubTTS())
        assert tag_outcome_of(parse_tag_output(PARTIAL_TAG)) \
            == TAG_OUTCOME_FATAL
        assert tag_outcome_of(parse_tag_output("[emo:neutral][mot:idle]")) \
            == TAG_OUTCOME_NO_SPEECH
        assert tag_outcome_of(parse_tag_output(VALID_TAG)) == "ok"
        report_a: dict = {}
        report_b: dict = {}
        orch._note_gen_outcome(
            report_a, tag_outcome_of(parse_tag_output(PARTIAL_TAG)))
        orch._note_gen_outcome(
            report_a, tag_outcome_of(parse_tag_output(VALID_TAG)))
        assert report_a.get("tag_failures") == 1
        # 別の試行の報告先には混ざらない
        assert report_b.get("tag_failures", 0) == 0
        assert is_tag_failure_outcome(TAG_OUTCOME_FATAL)
        assert is_tag_failure_outcome(TAG_OUTCOME_NO_SPEECH)
        assert not is_tag_failure_outcome("ok")
