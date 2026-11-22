"""SpeechStreamParser: LLMのトークンストリームから発話文を逐次取り出す

LLMは `{"speech":"...","emotion":"happy","motion":"wave","skip":false}` を返すが、
全文の生成完了を待ってからTTSに投げると、コメント受信から発話開始まで数秒かかる。

このパーサはストリーム途中の断片から `speech` の値だけを逐次抽出し、
文の切れ目（。！？）で確定した分をすぐ返す。呼び出し側はそれをTTSに流し込むことで、
1文目を喋りながら2文目を合成できる。

使い方:
    parser = SpeechStreamParser()
    async for delta in llm.generate_stream(...):
        for sentence in parser.feed(delta):
            await tts_queue.put(sentence)
    for sentence in parser.flush():
        await tts_queue.put(sentence)
    response = parser.build_response()   # emotion/motion を含む最終LLMResponse
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

from backend.llm.parser import (PARSE_OUTCOME_FATAL, PARSE_OUTCOME_OK,
                                PARSE_OUTCOME_SALVAGED,
                                TAG_OUTCOME_DEGRADED, TAG_OUTCOME_FATAL,
                                TAG_OUTCOME_NO_SPEECH, TAG_OUTCOME_OK,
                                contains_meta_leak,
                                dedupe_repeated_sentences, log_tag_no_speech,
                                parse_llm_output, record_parse_outcome,
                                record_tag_parse_outcome, scan_tag_output,
                                strip_think_tags, truncate_at_meta_leak)
from backend.logger import get_logger
from backend.models import Emotion, LLMResponse, Motion

logger = get_logger(__name__)

# 文末とみなす文字
_SENTENCE_END = "。．！？!?…♪"
# 文末が来ないまま長くなった場合に区切ってよい文字
_SOFT_BREAK = "、，,・ 　"

_SPEECH_KEY = re.compile(r'"speech"\s*:\s*"')

_JSON_DEBRIS_TAIL = re.compile(r'["”\'}\]]+\s*$')

# 発話として意味のある文字（これが1つも無い断片は喋らせない）
_WORD_CHAR = re.compile(r"[0-9A-Za-zぁ-んァ-ヶ一-龠ー]")


def _has_word_chars(text: str) -> bool:
    return bool(_WORD_CHAR.search(text))


def _normalized_len(text: str) -> int:
    """記号・空白を除いた実質文字数（断片判定用。正規化は他所と同じ）"""
    return len(re.sub(r"[\s、。，．！？!?'\"“”‘’]", "", text or ""))


def _strip_json_debris(text: str) -> str:
    """末尾が } や ] で終わる場合に限り、JSON 残骸（引用符・波括弧）を取り除く"""
    if not re.search(r"[}\]]\s*$", text):
        return text
    return _JSON_DEBRIS_TAIL.sub("", text).rstrip()


def _enum_or_default(enum_cls, value: str, default):
    """peek した文字列を Enum に変換する。不正値なら既定値を返す"""
    try:
        return enum_cls(value)
    except ValueError:
        return default

_ESCAPES = {
    "n": "\n", "t": " ", "r": "", '"': '"', "\\": "\\", "/": "/",
    "b": "", "f": "",
}


class SpeechStreamParser:
    """
    ストリームされた生テキストを食わせると、確定した発話文を順に返す。

    Args:
        min_chunk_chars: この文字数未満の断片は次の文とまとめる（短すぎる音声合成を防ぐ）
        max_chunk_chars: 文末が来なくてもこの長さで区切る（喋り出しが遅れるのを防ぐ）
    """

    def __init__(self, min_chunk_chars: int = 8, max_chunk_chars: int = 60):
        self._min = min_chunk_chars
        self._max = max_chunk_chars
        self._raw = ""       # 受信した生テキスト全体
        self._value = ""     # speech値としてデコード済みの全文
        self._consumed = 0   # _value のうち、既に文として払い出した文字数
        self._closed = False  # speech値の閉じ引用符に到達したか
        self._held_fragment: str | None = None  # 保留中の断片（唯一の文なら破棄）
        self._emitted_count = 0  # 払い出した文の数

    # ── 入力 ──────────────────────────────────────────────────

    def feed(self, delta: str) -> list[str]:
        """ストリーム断片を追加し、確定した文のリストを返す"""
        if not delta:
            return []
        self._raw += delta
        self._reparse()
        return self._pop_sentences(final=False)

    def flush(self) -> list[str]:
        """ストリーム終了時に、残っている分をすべて吐き出す"""
        self._reparse()
        if not self._value:
            # speechキーが見つからなかった場合、生テキストを発話として扱う
            self._save_fallback_raw()
            fallback = self._fallback_text()
            if fallback:
                self._value = fallback
        out = self._pop_sentences(final=True)
        rest = self._value[self._consumed:].strip()
        if not self._closed:
            # 閉じ引用符が来ないまま終わった生成では、末尾に JSON の残骸
            # （"} や ”} など）が付くことがある。発話に混ぜないよう除く。
            out = [s for s in (_strip_json_debris(s) for s in out) if s]
            rest = _strip_json_debris(rest)
        if rest and _has_word_chars(rest):
            self._consumed = len(self._value)
            self._accept(out, rest)
        if self._held_fragment is not None:
            # 後続の文が来なかった＝断片がこの応答の唯一の文だったので破棄する
            logger.warning("Sole fragment discarded", text=self._held_fragment[:40])
            self._held_fragment = None
        return [s for s in out if s.strip()]

    def _save_fallback_raw(self) -> None:
        """フォールバック発動時の生出力を全文保存する（観測性の改善）。

        通常ログの raw_preview / after_strip は先頭300字で切れるため、思考
        ブロック等の全文が残らない。保存先は logs/fallback_raw/ で、直近
        50件だけ残す。保存に失敗しても発話処理は止めない。
        """
        try:
            dest_dir = Path(__file__).resolve().parents[2] / "logs" / "fallback_raw"
            dest_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3]
            path = dest_dir / f"fallback-{stamp}.txt"
            path.write_text(self._raw, encoding="utf-8")
            logger.warning("Plain-text fallback activated; raw output saved",
                           path=str(path), chars=len(self._raw))
            saved = sorted(dest_dir.glob("fallback-*.txt"))
            for old in saved[:-50]:
                old.unlink(missing_ok=True)
        except Exception:
            pass

    # ── 内部処理 ──────────────────────────────────────────────

    def _reparse(self) -> None:
        """生テキスト全体から speech 値を復元し直す（チャンク境界に依存しない）"""
        if self._closed:
            return
        text = self._strip_think(self._raw)
        m = _SPEECH_KEY.search(text)
        if not m:
            return
        self._value, self._closed = self._decode_value(text, m.end())

    @staticmethod
    def _decode_value(text: str, start: int) -> tuple[str, bool]:
        """
        JSON文字列リテラルの中身を start から読み取る。
        Returns: (デコード済み文字列, 閉じ引用符に到達したか)
        """
        buf: list[str] = []
        i = start
        n = len(text)
        while i < n:
            ch = text[i]
            if ch == "\\":
                if i + 1 >= n:
                    break  # エスケープ対象がまだ届いていない
                nxt = text[i + 1]
                if nxt == "u":
                    if i + 6 > n:
                        break  # \uXXXX が未完
                    try:
                        buf.append(chr(int(text[i + 2:i + 6], 16)))
                    except ValueError:
                        pass
                    i += 6
                    continue
                buf.append(_ESCAPES.get(nxt, nxt))
                i += 2
                continue
            if ch == '"':
                return "".join(buf), True
            buf.append(ch)
            i += 1
        return "".join(buf), False

    def _pop_sentences(self, final: bool) -> list[str]:
        """未払い出し部分から、確定した文を切り出す"""
        out: list[str] = []
        while True:
            pending = self._value[self._consumed:]
            if not pending:
                break
            idx = self._find_break(pending, final)
            if idx is None:
                break
            chunk = pending[:idx + 1]
            remaining = len(pending) - len(chunk)
            # 短すぎる断片は、続きが来る見込みがあるなら次とまとめる
            if not final and not self._closed and remaining == 0 \
                    and len(chunk.strip()) < self._min:
                break
            self._consumed += len(chunk)
            stripped = chunk.strip()
            if stripped and _has_word_chars(stripped):
                self._accept(out, stripped)
        return out

    def _accept(self, out: list[str], sentence: str) -> None:
        """確定した文を払い出しリストへ追加する。

        ただし、正規化3字以下の断片が応答の最初の文で、まだ何も払い出して
        いない場合は保留する。後続の文が来れば先に払い出すが、最後まで
        来なければ flush() で破棄する（断片だけの応答を喋らせないための
        対策。複数文の中にある短い相づちは通常どおり残す）。
        """
        if self._held_fragment is not None:
            out.append(self._held_fragment)
            self._emitted_count += 1
            self._held_fragment = None
        if self._emitted_count == 0 and _normalized_len(sentence) <= 3:
            self._held_fragment = sentence
            return
        out.append(sentence)
        self._emitted_count += 1

    def _find_break(self, text: str, final: bool) -> int | None:
        """区切ってよい位置（インデックス）を返す。無ければ None。"""
        for i, ch in enumerate(text):
            if ch in _SENTENCE_END:
                return i
        if len(text) >= self._max:
            # 文末が来ないので、読点や空白で妥協して区切る
            for i in range(len(text) - 1, self._min - 1, -1):
                if text[i] in _SOFT_BREAK:
                    return i
            return len(text) - 1
        return None

    @staticmethod
    def _strip_think(text: str) -> str:
        """思考タグを除去する（<think> / <reasoning> / <thought> と Gemma の
        channel 形式。閉じていない場合は以降を全部落とす）。実装は parser
        側の strip_think_tags に一本化してある"""
        return strip_think_tags(text)

    def _fallback_text(self) -> str:
        """JSONを返さないモデル向けのフォールバック：生テキストをそのまま発話にする

        ただし擬似タグ形式（<emotion: ..> <motion: ..> <speech: ..>）の場合は
        speech の中身だけを抜き、タグ断片そのものは絶対に発話にしない
        （2026-11-14の劣化対策：タグ文字列がそのまま読み上げられた）。
        """
        text = self._strip_think(self._raw).strip()
        text = re.sub(r"```(?:json)?", "", text).strip()
        if not text or text.lstrip().startswith("{"):
            return ""
        m = re.search(r"<speech\s*:\s*", text, flags=re.IGNORECASE)
        if m:
            text = text[m.end():]
        text = re.sub(r"<[^>]{0,80}>", "", text).strip().strip("'\"")
        # 推論・ターン記号の漏れは混入位置で切り、同一文の繰り返しは打ち切る
        text = truncate_at_meta_leak(text)
        if not text:
            return ""
        text = dedupe_repeated_sentences(text)
        if not _has_word_chars(text):
            return ""
        # 切断しきれなかった思考マーカー・ターン記号が残るものと、記号で
        # 始まるものは発話として採用しない（そのターンは skip に倒す）
        if contains_meta_leak(text) or text[:1] in ("<", "*", "#"):
            return ""
        return text

    # ── 結果取得 ──────────────────────────────────────────────

    @property
    def raw(self) -> str:
        return self._raw

    @property
    def speech(self) -> str:
        return self._value.strip()

    @property
    def is_skip(self) -> bool:
        """ストリーム途中でも skip:true が確定していれば True"""
        return bool(re.search(r'"skip"\s*:\s*true', self._raw))

    def peek(self, key: str) -> str:
        """
        ストリーム途中でも取れている文字列キーの値を返す（未到達なら空文字）。
        emotion/motion を speech より前に置くことで、発話開始前に表情を確定できる。
        """
        m = re.search(rf'"{key}"\s*:\s*"([^"]*)"', self._strip_think(self._raw))
        return m.group(1) if m else ""

    def extract_effect(self) -> dict | None:
        """ストリーム中の "effect" オブジェクトを取り出す（生成途中の早期発火用）。

        オブジェクトが完結していない（生成途中）場合は None。
        検証は配送側（backend.effects）が行うため、ここでは dict 化まで。
        """
        text = strip_think_tags(self._raw)
        m = re.search(r'"effect"\s*:\s*\{', text)
        if not m:
            return None
        start = m.end() - 1
        depth = 0
        in_str = False
        escaped = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start:i + 1])
                    except ValueError:
                        return None
                    return obj if isinstance(obj, dict) else None
        return None

    def extract_thought(self) -> str:
        """ストリーム中の "thought" 文字列を取り出す（内心の早期表示用）。

        文字列が閉じていない（生成途中）場合は "" を返す。effect の
        サルベージ抽出と同方針で、検証は配送側（backend.effects）が
        行うため、ここではデコードまで。
        """
        text = strip_think_tags(self._raw)
        m = re.search(r'"thought"\s*:\s*"', text)
        if not m:
            return ""
        value, closed = self._decode_value(text, m.end())
        return value.strip() if closed else ""

    def build_response(self) -> LLMResponse:
        """
        ストリーム全体から最終的なLLMResponseを作る。
        emotion/motion はJSONから拾い、speechはストリームで組み立てた文字列を使う。

        全文パースが失敗しても、ストリームで確定した文（speech）があれば
        発話として成立させる（Phase 1: 消費側の堅牢化）。その場合の
        emotion/motion は peek 済みの値（発話開始前にアバターへ送った値）を
        採用し、実際に表示した表情と記録を一致させる。パース結果は
        致命/非致命を分けて計測する（parser.parse_outcome_stats）。
        計測はここで一元化するため、parse_llm_output 側の計数は止める。
        """
        parsed = parse_llm_output(self._raw, _count=False)
        speech = self.speech or parsed.speech
        salvaged = bool(isinstance(parsed.raw, dict)
                        and parsed.raw.get("salvaged"))
        parse_failed = parsed.skip and not parsed.raw
        # 発話テキストがある以上、既に喋り始めている可能性が高いのでskip扱いにはしない
        if not speech:
            record_parse_outcome(PARSE_OUTCOME_FATAL if parse_failed
                                 else PARSE_OUTCOME_OK)
            return LLMResponse(speech="", skip=True, raw=parsed.raw)
        if salvaged or parse_failed:
            record_parse_outcome(PARSE_OUTCOME_SALVAGED)
        else:
            record_parse_outcome(PARSE_OUTCOME_OK)
        if parse_failed:
            # 全文パースは失敗したが、ストリームで確定した文がある。
            # peek 済みの emotion/motion と effect のサルベージで組み立てる
            logger.info("Stream speech kept despite JSON parse failure",
                        speech=speech[:40])
            return LLMResponse(
                speech=speech,
                emotion=_enum_or_default(Emotion, self.peek("emotion"),
                                         parsed.emotion),
                motion=_enum_or_default(Motion, self.peek("motion"),
                                        parsed.motion),
                target_comment_id=parsed.target_comment_id,
                skip=False,
                raw={"stream_salvaged": True},
                effect=self.extract_effect() or parsed.effect,
                thought=parsed.thought or self.extract_thought(),
            )
        return LLMResponse(
            speech=speech,
            emotion=parsed.emotion,
            motion=parsed.motion,
            target_comment_id=parsed.target_comment_id,
            skip=False,
            raw=parsed.raw,
            effect=parsed.effect,
            thought=parsed.thought or self.extract_thought(),
        )


class TagSpeechStreamParser(SpeechStreamParser):
    """タグ方式版のストリームパーサ（Phase 5 Stage D）。

    SpeechStreamParser と同じインターフェース（feed / flush / peek /
    extract_effect / extract_thought / build_response）を持ち、
    プロトコルに応じて orchestrator が差し替える。文の切り出し・
    短い断片の保留といった払い出し規則は JSON 版と共通（基底クラス）。
    タグの走査そのものは parser.scan_tag_output に一本化してある。
    """

    def __init__(self, min_chunk_chars: int = 8, max_chunk_chars: int = 60,
                 flip_names=None):
        super().__init__(min_chunk_chars, max_chunk_chars)
        # flip の値が定型文の名前かどうかを判定するためのカタログ名
        self._flip_names = set(flip_names or ())
        self._scan = scan_tag_output("", final=False)

    def _reparse(self) -> None:
        """生テキスト全体を走査し直す（チャンク境界に依存しない）"""
        text = strip_think_tags(self._raw)
        self._scan = scan_tag_output(text, final=False,
                                     flip_names=self._flip_names)
        self._value = self._scan.speech

    def flush(self) -> list[str]:
        """ストリーム終了時に、残っている分をすべて吐き出す"""
        text = strip_think_tags(self._raw)
        self._scan = scan_tag_output(text, final=True,
                                     flip_names=self._flip_names)
        self._value = self._scan.speech
        self._closed = True
        out = self._pop_sentences(final=True)
        rest = self._value[self._consumed:].strip()
        if rest and _has_word_chars(rest):
            self._consumed = len(self._value)
            self._accept(out, rest)
        if self._held_fragment is not None:
            # 後続の文が来なかった＝断片がこの応答の唯一の文だったので破棄する
            logger.warning("Sole fragment discarded", text=self._held_fragment[:40])
            self._held_fragment = None
        return [s for s in out if s.strip()]

    @property
    def is_skip(self) -> bool:
        # タグ方式の skip は「タグのみ・本文なし」で、確定は生成終了時。
        # 途中では打ち切らない（発話は最大でも数十字で短い）
        return False

    def peek(self, key: str) -> str:
        """文頭タグの値を返す（未到達なら空文字）。JSON 版の peek と同役"""
        if key == "emotion":
            return self._scan.emotion_raw
        if key == "motion":
            return self._scan.motion_raw
        return ""

    def extract_effect(self) -> dict | None:
        """走査済みの fx タグを配送側の形（dict）で返す"""
        return self._scan.effect

    def extract_thought(self) -> str:
        """走査済みの th タグ（内心）を返す"""
        return (self._scan.thought or "").strip()

    def build_response(self) -> LLMResponse:
        """ストリーム全体から最終的な LLMResponse を作り、計測する。

        計測区分は PROTOCOL §4-5（ok / degraded / fatal）に no_speech を
        加えたもの。タグだけの出力（両タグ完備＋本文なし）は skip の
        意思表示かもしれないが、ヘッダ直後で止まった生成と区別できない
        ため ok には数えない。発話の安全検査（メタ漏れ等）は払い出し時に
        orchestrator が文単位で行っているため、ここでは行わない。
        """
        text = strip_think_tags(self._raw)
        scan = scan_tag_output(text, final=True, flip_names=self._flip_names)
        self._scan = scan
        speech = scan.speech.strip()
        if not speech:
            if scan.emotion_raw and scan.motion_raw:
                # 両タグ完備＋本文なしは ok と分離して数える（計測是正）
                outcome = TAG_OUTCOME_NO_SPEECH
            else:
                outcome = TAG_OUTCOME_FATAL
            record_tag_parse_outcome(outcome)
            return LLMResponse(
                speech="", skip=True,
                raw={"protocol": "tag", "tag_outcome": outcome})
        outcome = TAG_OUTCOME_DEGRADED if scan.issues else TAG_OUTCOME_OK
        record_tag_parse_outcome(outcome)
        return LLMResponse(
            speech=speech,
            emotion=_enum_or_default(Emotion, scan.emotion_raw,
                                     Emotion.NEUTRAL),
            motion=_enum_or_default(Motion, scan.motion_raw, Motion.IDLE),
            skip=False,
            raw={"protocol": "tag", "tag_outcome": outcome},
            effect=scan.effect,
            thought=scan.thought.strip(),
        )

    def report_no_speech(self, gen_meta: dict | None = None) -> str:
        """ストリームが1文も生まなかった生成の計測と診断ログ。

        build_response は発話が成立したときしか呼ばれないため、無発話で
        終わったストリームは従来どの区分にも数えられず、内訳が見えな
        かった。最終走査で本文が無い場合に限り、「両タグ完備＋本文なし」
        なら no_speech、それ以外（ヘッダ途中で切断・空）なら fatal と
        して数え、生出力と生成メタを警告に残す。戻り値は記録した区分
        （本文があった場合は "" で、何も記録しない）。
        """
        text = strip_think_tags(self._raw)
        scan = scan_tag_output(text, final=True, flip_names=self._flip_names)
        self._scan = scan
        if scan.speech.strip():
            return ""
        if scan.emotion_raw and scan.motion_raw:
            outcome = TAG_OUTCOME_NO_SPEECH
        else:
            outcome = TAG_OUTCOME_FATAL
        record_tag_parse_outcome(outcome)
        log_tag_no_speech(self._raw, source="streaming", gen_meta=gen_meta)
        return outcome
