"""ストリーミング発話パーサとローカルサービス診断のテスト"""
import json

import pytest

from backend.config import AppConfig
from backend.discovery import ProviderStatus, check_config
from backend.llm.parser import (contains_meta_leak, parse_llm_output,
                                truncate_at_meta_leak)
from backend.llm.streaming import SpeechStreamParser
from backend.models import Emotion, Motion


def stream(raw: str, chunk_size: int = 3) -> tuple[list[str], SpeechStreamParser]:
    """rawをchunk_size文字ずつ流し込み、確定した文のリストとパーサを返す"""
    p = SpeechStreamParser()
    out: list[str] = []
    for i in range(0, len(raw), chunk_size):
        out += p.feed(raw[i:i + chunk_size])
    out += p.flush()
    return out, p


# ── SpeechStreamParser ────────────────────────────────────────

class TestSpeechStreamParser:
    def test_splits_on_sentence_end(self):
        raw = '{"emotion":"happy","motion":"wave","skip":false,"speech":"わー、ありがとう！うれしいよ。"}'
        sents, p = stream(raw)
        assert sents == ["わー、ありがとう！", "うれしいよ。"]
        assert p.speech == "わー、ありがとう！うれしいよ。"

    def test_chunk_size_does_not_change_result(self):
        raw = '{"emotion":"happy","motion":"nod","skip":false,"speech":"こんにちは。今日もがんばるよ！"}'
        for size in (1, 2, 5, 17, 500):
            sents, p = stream(raw, chunk_size=size)
            assert p.speech == "こんにちは。今日もがんばるよ！", f"chunk_size={size}"
            assert sents == ["こんにちは。", "今日もがんばるよ！"], f"chunk_size={size}"

    def test_strips_think_tags(self):
        raw = '<think>どう答えようかな</think>{"emotion":"happy","motion":"nod","skip":false,"speech":"やっほー！"}'
        sents, p = stream(raw)
        assert sents == ["やっほー！"]
        assert "think" not in p.speech

    def test_decodes_json_escapes(self):
        raw = r'{"emotion":"neutral","motion":"idle","skip":false,"speech":"彼は\"やあ\"と言った。"}'
        sents, p = stream(raw)
        assert p.speech == '彼は"やあ"と言った。'

    def test_decodes_unicode_escape(self):
        raw = r'{"emotion":"neutral","motion":"idle","skip":false,"speech":"こんにちは。"}'
        _, p = stream(raw)
        assert p.speech == "こんにちは。"

    def test_skip_true_yields_nothing(self):
        raw = '{"emotion":"neutral","motion":"idle","skip":true,"speech":""}'
        sents, p = stream(raw)
        assert sents == []
        assert p.build_response().skip is True

    def test_plain_text_fallback(self):
        """JSONを返さないモデルでも発話として拾えること"""
        sents, p = stream("今日はいい天気だね。散歩に行こうかな。")
        assert sents == ["今日はいい天気だね。", "散歩に行こうかな。"]

    def test_long_text_without_sentence_end_is_broken_up(self):
        """文末記号が来なくても、長くなれば区切って喋り出せること"""
        body = "あのね、" * 25   # 100文字、文末記号なし
        raw = '{"emotion":"neutral","motion":"idle","skip":false,"speech":"' + body + '"}'
        sents, _ = stream(raw)
        assert len(sents) >= 2
        assert "".join(sents).replace(" ", "") == body.replace(" ", "")

    def test_emotion_motion_available_before_speech_ends(self):
        """emotion/motionがspeechより前にあるので、発話開始前に表情を確定できる"""
        p = SpeechStreamParser()
        p.feed('{"emotion":"surprised","motion":"jump","skip":false,"speech":"えっ')
        assert p.peek("emotion") == "surprised"
        assert p.peek("motion") == "jump"

    def test_build_response_carries_emotion_and_motion(self):
        raw = '{"emotion":"sad","motion":"bow","skip":false,"speech":"ごめんなさい。"}'
        _, p = stream(raw)
        r = p.build_response()
        assert r.emotion == Emotion.SAD
        assert r.motion == Motion.BOW
        assert r.speech == "ごめんなさい。"
        assert r.skip is False

    def test_no_duplicate_or_lost_characters(self):
        """チャンク境界でエスケープが割れても文字が重複・欠落しないこと"""
        raw = r'{"emotion":"happy","motion":"nod","skip":false,"speech":"A\"B\\CあD。"}'
        for size in (1, 2, 3, 4, 7):
            _, p = stream(raw, chunk_size=size)
            assert p.speech == 'A"B\\CあD。', f"chunk_size={size}"


# ── 設定診断 ──────────────────────────────────────────────────

class TestConfigDiagnosis:
    def _cfg(self) -> AppConfig:
        cfg = AppConfig()
        cfg.llm.primary = "lmstudio"
        cfg.tts.primary = "irodori"
        from backend.config import LLMBackendConfig, TTSBackendConfig
        cfg.llm.backends = {"lmstudio": LLMBackendConfig(model="auto")}
        cfg.tts.backends = {"irodori": TTSBackendConfig(voice="tsumugi")}
        cfg.stt.enabled = False
        return cfg

    def test_detects_missing_voice(self):
        """今回実際に無音の原因になった『存在しないvoice指定』を検出できること"""
        cfg = self._cfg()
        providers = [
            ProviderStatus("lmstudio", "llm", "LM Studio", "u", running=True, items=["m"]),
            ProviderStatus("irodori", "tts", "Irodori TTS", "u", running=True,
                           items=["none", "zundamon"]),
        ]
        issues = check_config(cfg, providers)
        voice_issues = [i for i in issues if i.target.endswith("voice")]
        assert len(voice_issues) == 1
        assert voice_issues[0].severity == "error"
        assert "tsumugi" in voice_issues[0].message
        assert "zundamon" in voice_issues[0].hint

    def test_accepts_existing_voice(self):
        cfg = self._cfg()
        cfg.tts.backends["irodori"].voice = "zundamon"
        providers = [
            ProviderStatus("lmstudio", "llm", "LM Studio", "u", running=True, items=["m"]),
            ProviderStatus("irodori", "tts", "Irodori TTS", "u", running=True,
                           items=["none", "zundamon"]),
        ]
        assert check_config(cfg, providers) == []

    def test_detects_stopped_llm(self):
        cfg = self._cfg()
        providers = [
            ProviderStatus("lmstudio", "llm", "LM Studio", "http://localhost:1234/v1", running=False),
            ProviderStatus("irodori", "tts", "Irodori TTS", "u", running=True, items=["tsumugi"]),
        ]
        issues = check_config(cfg, providers)
        assert any(i.target == "llm.primary" and i.severity == "error" for i in issues)

    def test_detects_unknown_model(self):
        cfg = self._cfg()
        cfg.llm.backends["lmstudio"].model = "nonexistent-model"
        providers = [
            ProviderStatus("lmstudio", "llm", "LM Studio", "u", running=True,
                           items=["gemma-4-12b", "qwen3-8b"]),
            ProviderStatus("irodori", "tts", "Irodori TTS", "u", running=True, items=["tsumugi"]),
        ]
        issues = check_config(cfg, providers)
        assert any("nonexistent-model" in i.message for i in issues)

    def test_auto_model_is_not_flagged(self):
        """LM Studioの 'auto' はロード済みモデルを使う指定なので警告しない"""
        cfg = self._cfg()
        providers = [
            ProviderStatus("lmstudio", "llm", "LM Studio", "u", running=True, items=["gemma-4-12b"]),
            ProviderStatus("irodori", "tts", "Irodori TTS", "u", running=True, items=["tsumugi"]),
        ]
        assert check_config(cfg, providers) == []
# ── Phase 1: 思考リーク・断片発話の回帰テスト ─────────────────
# 2026-11-17 の1時間テストで実際に発生した生出力を素材にする。

# Character2の生推論がそのまま発話化した件（06:22:13）の再構成
THOUGHT_RAW = (
    "<thought\nThinking Process:\n"
    "1. **Input Analysis**:\n"
    " * User Input: `<partner_speech name=\"Character1\">やっほー！</partner_speech>`"
    " への返答を考える\n"
    "2. **Response Strategy**: 短く相づちを返す"
)

# 配信アーカイブ（archive.jsonl）の正常発話。異常3件（思考リーク・断片2件）
# と重複1件を除いたもの。これらが無変更で通ることが過剰除去の検出になる。
NORMAL_SPEECHES = [
    "やっほー、ボクCharacter1！今日も元気にいくよ！",
    "はいはい、すごいですね。今日もCharacter2がサポートしますよ",
    "ん！Character2に頼り切っちゃうのはボクのプライドが許さないよ。でも、ボクの調査スキルなら完璧にこなせるからね！",
    "ええ、Character1さんの調査スキルは本当に信頼していますよ。",
    "そんなことないよ！Character2がいつもお茶を用意して支えてくれてるから、ボクも頑張れるんだ。ありがとう！",
    "あらら、ありがとうございます。Character1ちゃんも頑張ってくださいね",
    "えっへん、ボクに任せなさい！Character2、最高の調査結果をみんなに届けるよ！",
    "ん？Character2、今のなに？ボクの調べものに何か新しいヒントがあるのかな？",
    "あ、それはCharacter1ちゃんが調査で新しいことを発見したのでしょうか？",
    "おっ、小尾詩織さん初めまして！昨日の日経平均？ちょっと待って、すぐに調べ上げるからね！Character2も手伝って！",
    "はい、Character1ちゃん。一緒に調べていきましょうね",
    "えっへん！ボクに任せなさい。調査とまとめはボクの得意分野だから、完璧にやってみせるよ！",
    "あらら、すごいですね。Character1ちゃん、頼もしいです。",
    "えへへ、ボクのこと、Character2も頼もしいと思う？ボクはみんなに役立つ情報を届けるのが大好きだからね！",
    "今日もお菓子と調査の話で盛り上がりましょうね",
    "えっへん、もちろんです！ボクの調査スキルとCharacter2の知識を合わせれば、最高の情報がまとまるはずですよ！",
    "はいはい、すごいですね。Character2も精一杯ですね。",
    "そんなことないよ！Character2も十分頑張ってるよ！ボクも負けないように精一杯頑張るね！",
    "おねし、Character1ちゃん。Character2も全力でサポートしますからね。",
    "えっへん！ボクもCharacter2に負けないよう、全力で調査と整理に挑むよ！Character2、ボクの背中を押しなよ！",
    "小尾詩織さん、質問ありがとう！ボクは断然パン派！",
    "あ、パン派ですか。お菓子もパンも大好きなので、それもいいですね",
    "えっへん、ボクはパンの香ばしさが好きだからね。Character2もパン派なら、今度一緒に食べようよ！",
    "あらら、それも嬉しいです。Character2もパンが大好きなので、ぜひ一緒にいただきたいですね。",
    "えっへん、決まりだね！ボクの調査スキルで最高に美味しいパンの情報を探して、Character2に教えるよ！",
    "ん？Character2、どうしたの？ボクの調査に何かボクが見落としている情報があるのかな？",
    "あはは、ボクの調査は常に最新！Character2も驚くような面白い情報をたくさん見つけてくるよ！",
    "はい、Character1ちゃん。期待していますね",
    "えっへん、期待に全力で応えるよ！ボクの調査スキルなら、最高の結果をまとめられるからね。",
    "あらあら、すごいですね。Character1ちゃん、期待しています",
    "えっへん、ボクの調査スキルなら完璧にこなせるからね！Character2、期待に応えるよ！",
]


def _json_raw(speech: str) -> str:
    return json.dumps(
        {"emotion": "neutral", "motion": "idle", "skip": False,
         "speech": speech},
        ensure_ascii=False)


class TestThoughtLeakRegression:
    def test_thought_block_yields_no_speech(self):
        """生推論の塊は1文も払い出さないこと"""
        sents, p = stream(THOUGHT_RAW)
        assert sents == []
        assert p.build_response().skip is True

    def test_parse_llm_output_thought_only_is_skip(self):
        assert parse_llm_output(THOUGHT_RAW).skip is True

    def test_leaked_utterance_is_meta_leak(self):
        """実際に配信へ乗った49字がメタ漏れとして検出されること"""
        leaked = "<thought Thinking Process: 1. **Input Analysis**:"
        assert contains_meta_leak(leaked) is True
        assert truncate_at_meta_leak(leaked) == ""

    def test_truncate_rejects_scaffolding_head(self):
        """切断位置の先頭側自体が思考なら、8字以上あっても返さないこと"""
        intermediate = "<thought\nThinking Process:\n1. **Input Analysis**:"
        assert truncate_at_meta_leak(intermediate) == ""

    def test_thought_tag_stripped_before_json(self):
        raw = "<thought>どう答えよう</thought>" + _json_raw("やっほー！")
        sents, p = stream(raw)
        assert sents == ["やっほー！"]
        assert p.speech == "やっほー！"

    def test_gemma_channel_thought_stripped(self):
        raw = "<|channel>thought\n考え中<channel|>" + _json_raw("やっほー！")
        sents, _ = stream(raw)
        assert sents == ["やっほー！"]

    def test_turn_debris_after_speech_is_cut(self):
        """正常な発話の後ろのターン記号は従来どおり切り落とされること"""
        raw = ("今日も調査を頑張るよ。\n<|turn>user\n"
               "<partner_speech name=\"Character2\">ダミー</partner_speech>")
        sents, _ = stream(raw)
        assert sents == ["今日も調査を頑張るよ。"]


class TestSoleFragmentDiscard:
    def test_unfinished_json_fragment_discarded(self):
        """未完JSONから残った「あらら」（06:41:08）は喋らせないこと"""
        raw = '{"emotion":"neutral","motion":"idle","skip":false,"speech":"あらら'
        sents, _ = stream(raw)
        assert sents == []

    def test_sole_fragment_json_discarded(self):
        """「あ、」（06:55:52）だけの応答は喋らせないこと"""
        sents, _ = stream(_json_raw("あ、"))
        assert sents == []

    def test_sole_fragment_plain_discarded(self):
        sents, _ = stream("あ、")
        assert sents == []

    def test_short_sentence_kept_when_not_sole(self):
        """複数文の中の短い文は断片として捨てないこと"""
        sents, _ = stream(_json_raw("うん。今日も頑張るよ。"))
        assert sents == ["うん。", "今日も頑張るよ。"]


class TestNormalSpeechesUnchanged:
    def test_normal_speeches_pass_through(self):
        """正常発話がすべて無変更で払い出されること（過剰除去の検出）"""
        for speech in NORMAL_SPEECHES:
            sents, p = stream(_json_raw(speech))
            assert "".join(sents) == speech, speech
            assert p.speech == speech, speech

    def test_normal_speeches_are_not_meta_leaks(self):
        for speech in NORMAL_SPEECHES:
            assert contains_meta_leak(speech) is False, speech
            assert parse_llm_output(_json_raw(speech)).speech == speech, speech


class TestFallbackRawSave:
    def test_fallback_raw_is_saved(self):
        """フォールバック発動時に生出力が logs/fallback_raw/ へ残ること"""
        from pathlib import Path
        dest = Path(__file__).resolve().parents[1] / "logs" / "fallback_raw"
        before = set(dest.glob("fallback-*.txt")) if dest.exists() else set()
        marker = "フォールバック保存の確認テキストです。"
        sents, _ = stream(marker)
        assert sents == [marker]
        new_files = [f for f in dest.glob("fallback-*.txt") if f not in before]
        assert new_files, "fallback raw was not saved"
        assert any(marker in f.read_text(encoding="utf-8") for f in new_files)
