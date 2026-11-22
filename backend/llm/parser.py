"""LLMのJSON出力をパースしてLLMResponseに変換する"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from backend.logger import get_logger
from backend.models import Emotion, LLMResponse, Motion

logger = get_logger(__name__)


# ── パース結果の計測（Phase 1: no_json の致命/非致命の分離） ──────
# 3時間テストで「No JSON object found」が203回出たが、その大半が
# 「ストリーム側で発話が成立していた（非致命）」のか「ターンが丸ごと
# 消えた（致命）」のか区別できなかった。parse_llm_output の結果を
# 3分類して数え、長時間テストで内訳を追えるようにする。
PARSE_OUTCOME_OK = "ok"
PARSE_OUTCOME_SALVAGED = "no_json_salvaged"
PARSE_OUTCOME_FATAL = "no_json_fatal"
_parse_outcomes: dict[str, int] = {
    PARSE_OUTCOME_OK: 0,
    PARSE_OUTCOME_SALVAGED: 0,
    PARSE_OUTCOME_FATAL: 0,
}


def parse_outcome_stats(protocol: str = "json") -> dict[str, int]:
    """パース結果の累計（ok / no_json_salvaged / no_json_fatal）。

    protocol="tag" のときはタグ方式の計測（ok / degraded / fatal）を返す。
    knowledge_status は config の protocol に応じて片方だけを出す。
    """
    if str(protocol or "json").lower() == "tag":
        return tag_parse_outcome_stats()
    return dict(_parse_outcomes)


def record_parse_outcome(outcome: str) -> None:
    """パース結果を1件数える（streaming 側の分類からも使う）"""
    if outcome in _parse_outcomes:
        _parse_outcomes[outcome] += 1


# ── メタ漏れ（推論・ターン記号）検出と同一文の繰り返し除去 ──────
# 文脈が劣化するとQ4モデルが思考過程（「検討：…」）や会話のターン記号
# （user / model）をそのまま出力し、フォールバック経路で読み上げられた
# （2026-11-14の2時間テストで発生）。発話前に落とすための判定。
_TURN_MARKER_RE = re.compile(
    r"(?:^|[\s'\"“「、。・])(?i:user|model|assistant|system)(?:[:：]|\s|$)"
    r"|(?:^|[^\w])(?i:user|model|mode)\s*$")
_REASONING_RE = re.compile(
    r"検討\s*[：:]|考察\s*[：:]|ユーザーの入力|ユーザーは|ユーザーからの"
    r"|システムプロンプト|出力形式|応答文のみ|会話履歴を踏まえ"
    r"|<thought\b|(?i:thinking\s+process|thought\s*[:：])"
    r"|^\s*>\s*\d|^\s*\d+\.\s+\S")
_MD_LINE_RE = re.compile(r"^\s*(#{1,3}\s|[-*]\s)")
# 英語の思考マーカー（モデルの生推論の書き出し）。正当な発話がこれで始まる
# ことはないため、truncate_at_meta_leak では先頭一致を全体無効の根拠に使う
_THINKING_MARKER_RE = re.compile(
    r"<thought\b|(?i:thinking\s+process|thought\s*[:：])")
# 行頭の番号行。単独では通常発話の箇条書きと区別できないため、思考マーカーと
# 併存する場合にだけ推論の書き出しとみなす（contains_meta_leak 参照）
_NUMBERED_LINE_RE = re.compile(r"^\s*\d+\.\s", re.MULTILINE)


# ── 思考タグの除去 ────────────────────────────────────────────
# <think> / <reasoning> に加え、モデルが content へ直接書き出す <thought> と
# Gemma 系の channel 形式（<|channel>thought … <channel|>）も思考として扱う。
# 2026-11-17 の1時間テストで <thought 冒頭の生推論がそのまま発話化した件の対策。
_THINK_TAGS = ("think", "reasoning", "thought")


def strip_think_tags(text: str) -> str:
    """思考タグを取り除く。閉じていない場合は、その位置以降を切り捨てる"""
    t = text or ""
    for tag in _THINK_TAGS:
        t = re.sub(rf"<{tag}\b[^>]*>.*?</{tag}\s*>", "", t, flags=re.DOTALL)
    t = re.sub(r"<\|channel>thought.*?<channel\|>", "", t, flags=re.DOTALL)
    cut = None
    for tag in _THINK_TAGS:
        m = re.search(rf"<{tag}\b", t)
        if m:
            cut = m.start() if cut is None else min(cut, m.start())
    m = re.search(r"<\|channel>thought", t)
    if m:
        cut = m.start() if cut is None else min(cut, m.start())
    if cut is not None:
        t = t[:cut]
    return t


def contains_meta_leak(text: str) -> bool:
    """推論・ターン記号など、発話に混ざってはいけない断片を含むか"""
    t = text or ""
    if _TURN_MARKER_RE.search(t) or _REASONING_RE.search(t):
        return True
    if any(_MD_LINE_RE.match(line) for line in t.splitlines()):
        return True
    # 番号行＋思考マーカーの併存は、箇条書き形式の推論の書き出しとみなす
    # （番号行だけなら通常発話の手順説明等と区別できないため単独では落とさない）
    if _NUMBERED_LINE_RE.search(t) and _THINKING_MARKER_RE.search(t):
        return True
    return False


# ── 言語混入（中国語）検出 ────────────────────────────────────
# フォールバック側のモデル（LM Studio 等）が中国語の断片を混ぜることが
# ある（2026-11-21 BFラン: 「啦，」を含む文がそのまま放送された）。
# 日本語の通常テキストには現れないマーカーだけを拾う:
#   ， … 全角コンマ（日本語は「、」を使う）
#   啦・呢・麼 … 中国語の語気助詞（日本語では使わない）
#   簡体字のうち日本語の字体とコードポイントが異なる常用字
_FOREIGN_MARKER_RE = re.compile(
    r"[，啦呢麼]"
    r"|[们这么说对还个时为过进远达书车马龙听读语识记讲许论证设访谈诉试该详际观实]")


def foreign_contamination_marker(text: str) -> str:
    """中国語混入のマーカー文字を返し、無ければ空文字を返す。

    発話・日記など「日本語として読み上げ・記録されるテキスト」の
    品質ゲート用。マーカー自体を返すのは、ログで何が引っかかったかを
    追えるようにするため。
    """
    m = _FOREIGN_MARKER_RE.search(text or "")
    return m.group(0) if m else ""


def truncate_at_meta_leak(text: str) -> str:
    """メタ漏れの開始位置で文を切る。先頭付近なら丸ごと無効（空文字）"""
    t = text or ""
    # 思考マーカーで始まるテキストは全体が推論そのものなので、「正常な発話の
    # 後ろに scaffolding が続く」形と区別し、切断位置によらず丸ごと無効にする
    if _THINKING_MARKER_RE.match(t.lstrip()):
        return ""
    cut = None
    for rx in (_TURN_MARKER_RE, _REASONING_RE):
        m = rx.search(t)
        if m:
            cut = m.start() if cut is None else min(cut, m.start())
    for line in t.splitlines():
        if _MD_LINE_RE.match(line):
            pos = t.find(line)
            cut = pos if cut is None else min(cut, pos)
    # 箇条書き形式の推論（番号行＋思考マーカーの併存）は最初の番号行で切る
    if _THINKING_MARKER_RE.search(t):
        m = _NUMBERED_LINE_RE.search(t)
        if m:
            cut = m.start() if cut is None else min(cut, m.start())
    if cut is None:
        return t
    head = t[:cut].strip()
    if len(head) < 8:
        return ""
    # 先頭側自体がメタ内容なら、正常な発話として返さない。思考の前文が
    # 「8字以上あるので有効」として漏れた 2026-11-17 の件への対策
    if contains_meta_leak(head):
        return ""
    return head


def dedupe_repeated_sentences(text: str) -> str:
    """同一文の繰り返し（劣化時のループ出力）を2回目の出現で打ち切る"""
    parts = re.split(r"(?<=[。！？!?])", text or "")
    seen: set[str] = set()
    out: list[str] = []
    for part in parts:
        norm = re.sub(r"[\s、。，．！？!?'\"“”‘’]", "", part)
        if len(norm) >= 6:
            if norm in seen:
                break
            seen.add(norm)
        out.append(part)
    return "".join(out).strip()


# ── 挨拶・自己紹介テンプレ（冒頭）の除去 ─────────────────────
# 学習データに挨拶テンプレが高頻度で混ざっているため、会話の途中でも
# 発話の冒頭に「やっほー！ボクCharacter1！」のような自己紹介が割り込む
# （2026-11-16のコメント会話テストで発生）。開始直後の正規の挨拶まで
# 消さないよう、除去するかどうかの判断は呼び出し側（orchestrator）が
# 持ち、ここではテキスト変換だけを行う。
# 通常の挨拶（おはよう・こんにちは等）はテンプレに含めない。
# コメントへの返答としての挨拶は正規の応答だからである。

_GREETING_HEAD_RES = [
    # 呼びかけ型の挨拶（やっほー等）で始まる文
    # 「よっほー」系が欠けていて、先頭文がマッチしないと prefix-run 除去
    # 全体が止まり「よっほー、Character2！ボクCharacter1、準備は万端！」型の
    # 複合テンプレがすり抜けた（2026-11-16 Phase 4 ソークで実測）。
    re.compile(r"^(やっほー|やっほ|よっほー|よっほ|ヤッホー|ヤッホ|ヨッホー|ヨッホ)"),
    # 店員風の挨拶で始まる文
    re.compile(r"^いらっしゃいませ"),
    # 名乗り（ボクCharacter1／わたし、Character2です等）を含む文
    re.compile(r"(ボク|わたし|私|あたし)[、,は]?\s*(Character1|Character2)(です|だよ|だ|！|!|。|、)"),
    # 開始時の定型句（よろしくお願いします）で始まる文
    re.compile(r"^(今日も|本日も)?\s*よろしくお願い(します|いたします)"),
]


def _is_greeting_template_sentence(sentence: str) -> bool:
    s = (sentence or "").strip()
    if not s:
        return False
    return any(rx.search(s) for rx in _GREETING_HEAD_RES)


# 発話冒頭の呼びかけ（「マスター、」等）。呼びかけが前置すると文頭アンカーの
# テンプレ判定に掛からず素通りしたため、判定前に呼びかけ部を分離する
# （2026-11-16 P0b: 「マスター、いらっしゃいませ。…」が素通りした件）。
# 対象は呼びかけに使う語＋区切り記号に限定し、通常文の読点では分離しない。
_VOCATIVE_RE = re.compile(
    r"^(マスター|ご主人様|旦那様|お客様|みんな|みなさん|皆さん|皆様)"
    r"[\s、,！!。…♪]+")


def _strip_greeting_head(t: str) -> str:
    """先頭からテンプレ文が続く間だけ除去する（strip_greeting_template の本体）"""
    parts = [p for p in re.split(r"(?<=[。！？!?])", t) if p.strip()]
    idx = 0
    while idx < len(parts) and _is_greeting_template_sentence(parts[idx]):
        idx += 1
    if idx == 0:
        return t
    return "".join(parts[idx:]).strip()


def strip_greeting_template(text: str) -> str:
    """発話冒頭に連続する挨拶・自己紹介テンプレ文を取り除く。

    先頭からテンプレ文が続く間だけ除去し、最初の通常文以降には触らない。
    全部テンプレだった場合は空文字を返す。
    冒頭の呼びかけ（「マスター、」等）は判定の対象から外し、テンプレ文を
    除去した後の発話には呼びかけを付け戻す。呼びかけ以外が全部テンプレ
    だった場合は、呼びかけだけを残さず空文字を返す。
    """
    t = (text or "").strip()
    if not t:
        return t
    prefix = ""
    body = t
    m = _VOCATIVE_RE.match(t)
    if m and m.end() < len(t):
        prefix = t[:m.end()]
        body = t[m.end():]
    stripped = _strip_greeting_head(body)
    if stripped == body:
        return t
    if not stripped:
        return ""
    return prefix + stripped


# ── 切断JSONからのサルベージ（Phase 1: 消費側の堅牢化） ──────────
# 生成が途中で切れると JSON としては壊れるが、speech の値は途中まで
# （あるいは全部）届いていることが多い。全文パースの失敗だけを理由に
# 発話を捨てると回復リトライが連鎖するため、speech/emotion/motion を
# 個別に拾って発話として成立させる。救う基準は控えめにする：
# 「あらら」のような未完の断片（06:41:08 の回帰）はここでも救わない。
_SPEECH_VALUE_RE = re.compile(r'"speech"\s*:\s*"')
_SKIP_TRUE_RE = re.compile(r'"skip"\s*:\s*true')
_SENTENCE_END_CHARS = "。．！？!?…"

_STRING_ESCAPES = {
    "n": "\n", "t": " ", "r": "", '"': '"', "\\": "\\", "/": "/",
    "b": "", "f": "",
}


def _decode_json_string_prefix(text: str, start: int) -> tuple[str, bool]:
    """JSON文字列リテラルの中身を start から読み取る（streaming と同仕様）。

    Returns: (デコード済み文字列, 閉じ引用符に到達したか)
    """
    buf: list[str] = []
    i = start
    n = len(text)
    while i < n:
        ch = text[i]
        if ch == "\\":
            if i + 1 >= n:
                break  # エスケープ対象がまだ届いていない（切断）
            nxt = text[i + 1]
            if nxt == "u":
                if i + 6 > n:
                    break  # \uXXXX が未完（切断）
                try:
                    buf.append(chr(int(text[i + 2:i + 6], 16)))
                except ValueError:
                    pass
                i += 6
                continue
            buf.append(_STRING_ESCAPES.get(nxt, nxt))
            i += 2
            continue
        if ch == '"':
            return "".join(buf), True
        buf.append(ch)
        i += 1
    return "".join(buf), False


def _peek_string_value(text: str, key: str) -> str:
    """壊れたJSONからでも、完結している文字列キーの値を拾う"""
    m = re.search(rf'"{key}"\s*:\s*"([^"]*)"', text)
    return m.group(1) if m else ""


def _extract_effect_object(text: str) -> dict | None:
    """末尾の "effect" オブジェクトを括弧バランスで取り出す（streaming と同方式）"""
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


def salvage_broken_json(text: str) -> LLMResponse | None:
    """壊れたJSONから発話を救済する。救えなければ None。

    救済するのは「発話として確定している」と言えるものだけ：
      - speech の文字列が閉じている、または
      - 文末記号を含む（文が完結している）、または
      - 正規化8字以上の長さがある（文の途中で切れても内容が成立）
    明示的な skip:true と、安全性チェック（タグ断片・メタ漏れ）に
    掛かるものは救済しない（通常経路と同じ基準）。
    """
    if not text or _SKIP_TRUE_RE.search(text):
        return None
    m = _SPEECH_VALUE_RE.search(text)
    if not m:
        return None
    value, closed = _decode_json_string_prefix(text, m.end())
    speech = value.strip()
    if not speech:
        return None
    normalized = re.sub(r"[\s、。，．！？!?'\"“”‘’]", "", speech)
    if not closed and len(normalized) < 8 \
            and not any(c in speech for c in _SENTENCE_END_CHARS):
        # 未完の断片（「あらら」等）は発話として確定していないので救わない
        return None
    if re.search(r"<\s*(emotion|motion|speech)\s*:|<\|[^>]*\|>", speech):
        return None
    if contains_meta_leak(speech):
        return None
    speech = dedupe_repeated_sentences(speech)
    if not speech:
        return None
    try:
        emotion = Emotion(_peek_string_value(text, "emotion"))
    except ValueError:
        emotion = Emotion.NEUTRAL
    try:
        motion = Motion(_peek_string_value(text, "motion"))
    except ValueError:
        motion = Motion.IDLE
    return LLMResponse(
        speech=speech,
        emotion=emotion,
        motion=motion,
        skip=False,
        raw={"salvaged": True},
        effect=_extract_effect_object(text),
        # 内心も完結していれば拾う（エスケープを含む場合は途切れうるが、
        # サルベージ用途として許容し、最終検証は配送側で行う）
        thought=_peek_string_value(text, "thought"),
    )


def _looks_like_json_attempt(text: str) -> bool:
    """JSONを返そうとした形跡があるか（プレーンテキスト出力と区別する）"""
    t = (text or "").lstrip()
    return t.startswith("{") or '"speech"' in t


def parse_llm_output(raw_text: str, _count: bool = True) -> LLMResponse:
    """
    LLMが返したテキストからJSONを抽出してLLMResponseに変換する。
    以下のパターンに対応:
    - qwen3の <think>...</think> タグを除去
    - ```json ... ``` フェンス除去
    - 前後のテキストを無視してJSONだけ抽出
    失敗した場合はskip=Trueを返す（例外は投げない）。
    """
    text = raw_text.strip()

    # デバッグ用: 生テキストの先頭をログ出力
    logger.debug("LLM raw output", preview=text[:200])

    # 思考タグを除去（<think> / <reasoning> / <thought> と Gemma の channel
    # 形式。閉じていない場合は以降を切り捨てる）
    text = strip_think_tags(text).strip()

    # ```json ... ``` フェンスを除去
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```", "", text)
    text = text.strip()

    # 最初の { から最後の } までを抽出
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        # JSONとしては壊れているが、speech が確定していれば救済する
        # （Phase 1: 全文パース失敗だけを理由に発話を捨てない）
        salvaged = salvage_broken_json(text)
        if salvaged is not None:
            if _count:
                record_parse_outcome(PARSE_OUTCOME_SALVAGED)
            logger.warning("No JSON object found; speech salvaged",
                           speech=salvaged.speech[:40])
            return salvaged
        if _count and _looks_like_json_attempt(text):
            record_parse_outcome(PARSE_OUTCOME_FATAL)
        logger.warning("No JSON object found in LLM output",
                       raw_preview=raw_text[:300],
                       after_strip=text[:300])
        return LLMResponse(speech="", skip=True)

    json_str = text[start : end + 1]

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        salvaged = salvage_broken_json(text)
        if salvaged is not None:
            if _count:
                record_parse_outcome(PARSE_OUTCOME_SALVAGED)
            logger.warning("JSON parse error; speech salvaged",
                           error=str(e), speech=salvaged.speech[:40])
            return salvaged
        if _count:
            record_parse_outcome(PARSE_OUTCOME_FATAL)
        logger.warning("JSON parse error", error=str(e), raw=json_str[:300])
        return LLMResponse(speech="", skip=True)

    if _count:
        record_parse_outcome(PARSE_OUTCOME_OK)

    # skip フラグ
    if data.get("skip", False):
        return LLMResponse(speech="", skip=True, raw=data)

    speech = str(data.get("speech", "")).strip()
    if not speech:
        logger.warning("Empty speech in LLM output", data=data)
        return LLMResponse(speech="", skip=True, raw=data)

    # 制御タグの断片が発話文に混ざっていたら喋らせない（2026-11-14対策）
    if re.search(r"<\s*(emotion|motion|speech)\s*:|<\|[^>]*\|>", speech):
        logger.warning("Tag fragments in speech, skipping", speech=speech[:120])
        return LLMResponse(speech="", skip=True, raw=data)

    # 推論・ターン記号の漏れが混ざっていたら喋らせない（2026-11-14対策2）
    if contains_meta_leak(speech):
        logger.warning("Meta leak in speech, skipping", speech=speech[:120])
        return LLMResponse(speech="", skip=True, raw=data)
    speech = dedupe_repeated_sentences(speech)
    if not speech:
        return LLMResponse(speech="", skip=True, raw=data)

    # emotion
    try:
        emotion = Emotion(data.get("emotion", "neutral"))
    except ValueError:
        emotion = Emotion.NEUTRAL

    # motion
    try:
        motion = Motion(data.get("motion", "idle"))
    except ValueError:
        motion = Motion.IDLE

    # 演出エフェクト（任意）。ここでは dict であることだけを確認し、
    # カタログ照合などの検証は配送側（backend.effects）が行う
    effect = data.get("effect")
    if not isinstance(effect, dict):
        effect = None

    # 内心（任意）。文字列であることだけを確認し、長さ・内容の検証は
    # 配送側（backend.effects）が行う。声には絶対に出さない
    thought = data.get("thought")
    if not isinstance(thought, str):
        thought = ""

    return LLMResponse(
        speech=speech,
        emotion=emotion,
        motion=motion,
        target_comment_id=data.get("target_comment_id"),
        skip=False,
        raw=data,
        effect=effect,
        thought=thought.strip(),
    )


# ── タグ方式パーサ（Phase 5 Stage D） ─────────────────────────
# PROTOCOL.md の文法 `[emo:][mot:]発話[fx:][th:]` を解釈する。
# JSON 方式と違い、タグが欠落・破損しても発話本文は成立させる。
# 計測は ok / degraded / fatal / no_speech の区分（PROTOCOL §4-5 に
# no_speech を追加）で、JSON 側の parse_outcomes とは別カウンタに
# 分ける（knowledge_status は protocol に応じて片方だけを出す）。

TAG_OUTCOME_OK = "ok"
TAG_OUTCOME_DEGRADED = "degraded"
TAG_OUTCOME_FATAL = "fatal"
# 両タグ完備＋本文なし。PROTOCOL §3 では skip の意思表示だが、意図せず
# ヘッダ直後で生成が止まったケースと区別がつかないため、ok から分離
# して数える（2026-11-19 の配信テストで多発した件の計測是正）
TAG_OUTCOME_NO_SPEECH = "no_speech"
_tag_parse_outcomes: dict[str, int] = {
    TAG_OUTCOME_OK: 0,
    TAG_OUTCOME_DEGRADED: 0,
    TAG_OUTCOME_FATAL: 0,
    TAG_OUTCOME_NO_SPEECH: 0,
}


def tag_parse_outcome_stats() -> dict[str, int]:
    """タグ方式パース結果の累計（ok / degraded / fatal / no_speech）"""
    return dict(_tag_parse_outcomes)


def record_tag_parse_outcome(outcome: str) -> None:
    """タグ方式のパース結果を1件数える"""
    if outcome in _tag_parse_outcomes:
        _tag_parse_outcomes[outcome] += 1


def tag_outcome_of(response) -> str:
    """応答オブジェクトに刻まれた、その生成のパース区分を返す（F3）。

    パース区分は生成のたびに応答の raw["tag_outcome"] にも刻まれる。
    失敗の帰属を「モジュール共有カウンタの前後差分」で推測すると、
    ロック外で並行する生成（プリフェッチ等）の失敗と取り違えるため、
    制御側の判定はこの生成単位の値を読むこと。グローバルカウンタは
    表示・集計専用として残す。
    """
    raw = getattr(response, "raw", None)
    if isinstance(raw, dict):
        return str(raw.get("tag_outcome") or "")
    return ""


def is_tag_failure_outcome(outcome: str) -> bool:
    """パース区分が no-speech 系の失敗（fatal / no_speech）か"""
    return outcome in (TAG_OUTCOME_FATAL, TAG_OUTCOME_NO_SPEECH)


def tag_no_speech_fields(raw_text: str,
                         gen_meta: dict | None = None) -> dict:
    """no-speech 診断用のログフィールドを組み立てる。

    strip 前の生出力プレビュー、思考タグ除去で消えた文字数、バックエンドが
    保持した生成メタ（done_reason / eval_count / 思考フィールドの文字数）を
    返す。2026-11-19 の配信テストでは strip 後の断片しかログに残らず、
    「思考チャネルが生成予算を食って本文が切れた」のか「ヘッダ直後で
    EOS した」のか切り分けできなかったための観測追加。
    """
    raw = raw_text or ""
    stripped = strip_think_tags(raw)
    meta = gen_meta or {}
    return {
        "raw_preview": raw[:300],
        "after_strip_preview": stripped.strip()[:300],
        "think_stripped_chars": len(raw) - len(stripped),
        "thinking_chars": meta.get("thinking_chars"),
        "done_reason": meta.get("done_reason"),
        "eval_count": meta.get("eval_count"),
        # プロンプト側のトークン数。num_ctx に近ければコンテキスト溢れ
        "prompt_eval_count": meta.get("prompt_eval_count"),
    }


def log_tag_no_speech(raw_text: str, *, source: str,
                      gen_meta: dict | None = None) -> None:
    """タグ方式で発話が得られなかった生成を、診断情報つきで警告に残す"""
    logger.warning("No speech in tag output", source=source,
                   **tag_no_speech_fields(raw_text, gen_meta))


_TAG_TOKEN_RE = re.compile(r"\[(emo|mot|fx|th):([^\[\]]*)\]")
# 未完タグの判定（'[' の後ろがタグ名の途中までで途切れている形）
_TAG_PARTIAL_RE = re.compile(r"\[(?:e|em|emo|m|mo|mot|f|fx|t|th)?(?::[^\[\]]*)?")
_TAG_HEAD_RE = re.compile(r"\[(?:emo|mot|fx|th):")
# 角括弧断片の最大長（PROTOCOL §4-2 の保留上限）
_TAG_FRAG_MAX = 64

_EMOTION_VALUES = {e.value for e in Emotion}
_MOTION_VALUES = {m.value for m in Motion}


@dataclass
class TagScan:
    """タグ走査の結果。emotion/motion は検証前の生値（"" は未出現）"""
    speech: str = ""
    emotion_raw: str = ""
    motion_raw: str = ""
    effect: dict | None = None
    thought: str = ""
    issues: set[str] = field(default_factory=set)
    held_from: int | None = None  # 保留中の未完断片の開始位置（final=False のみ）


def _parse_fx_value(val: str, flip_names: set, issues: set) -> dict | None:
    """[fx:<type>:<value>] の値を配送側（validate_effect）の形に変換する。

    cutin/se は name、flip は「定型文の名前なら name、それ以外は text」
    として渡す（名前→定型文の解決は validate_effect が行う）。
    """
    if ":" not in val:
        issues.add("fx_broken")
        return None
    ftype, fval = val.split(":", 1)
    ftype = ftype.strip().lower()
    fval = fval.strip()
    if ftype in ("cutin", "se"):
        if not fval:
            issues.add("fx_broken")
            return None
        return {"type": ftype, "name": fval}
    if ftype == "flip":
        if not fval:
            issues.add("fx_broken")
            return None
        if fval in flip_names:
            return {"type": "flip", "name": fval}
        return {"type": "flip", "text": fval}
    issues.add("fx_broken")
    return None


def scan_tag_output(text: str, *, final: bool = True,
                    flip_names=None) -> TagScan:
    """タグ付き出力を走査して発話・タグ・劣化要因に分解する。

    final=False（ストリーミング途中）では、末尾の未完の '[' 断片を
    保留し（held_from に位置を返す）、発話にもタグにも数えない。
    final=True では未完断片は破棄し、劣化として記録する。
    角括弧断片はタグ文法に一致するものだけを制御タグとして扱い、
    一致しない断片も発話からは除去する（PROTOCOL §4-1）。
    入力は思考タグ除去済みのテキストを想定する。
    """
    scan = TagScan()
    flip_name_set = set(flip_names or ())
    parts: list[str] = []
    speech_started = False
    fx_seen = False
    th_seen = False
    i = 0
    n = len(text)
    while i < n:
        j = text.find("[", i)
        if j == -1:
            seg = text[i:]
            if "]" in seg:
                scan.issues.add("stray_bracket")
                seg = seg.replace("]", "")
            parts.append(seg)
            if seg.strip():
                speech_started = True
            break
        if j > i:
            seg = text[i:j]
            if "]" in seg:
                scan.issues.add("stray_bracket")
                seg = seg.replace("]", "")
            parts.append(seg)
            if seg.strip():
                speech_started = True
            i = j
        # text[i] == "["
        k = text.find("]", i + 1, i + _TAG_FRAG_MAX)
        if k == -1:
            rest = text[i:]
            if not final and len(rest) <= _TAG_FRAG_MAX \
                    and _TAG_PARTIAL_RE.fullmatch(rest):
                # 未完のタグかもしれないので保留する（次の断片で確定する）
                scan.held_from = i
                break
            k2 = text.find("]", i + 1)
            if k2 != -1:
                # 長すぎる角括弧断片はタグではないが発話にも混ぜない
                scan.issues.add("unknown_fragment")
                i = k2 + 1
                continue
            # 末尾まで完結しない断片は破棄する
            if _TAG_HEAD_RE.match(rest) or _TAG_PARTIAL_RE.fullmatch(rest):
                scan.issues.add("partial_dropped")
            else:
                scan.issues.add("stray_bracket")
            break
        frag = text[i:k + 1]
        m = _TAG_TOKEN_RE.fullmatch(frag)
        if m is None:
            # 未知の角括弧断片は読み上げ防止のため発話から除去する
            scan.issues.add("unknown_fragment")
            i = k + 1
            continue
        kind, val = m.group(1), m.group(2).strip()
        if kind == "emo":
            if scan.emotion_raw:
                scan.issues.add("extra_tag")
            else:
                scan.emotion_raw = val
                if speech_started:
                    scan.issues.add("emo_not_leading")
                if val not in _EMOTION_VALUES:
                    scan.issues.add("emo_invalid")
        elif kind == "mot":
            if scan.motion_raw:
                scan.issues.add("extra_tag")
            else:
                scan.motion_raw = val
                if speech_started:
                    scan.issues.add("mot_not_leading")
                if val not in _MOTION_VALUES:
                    scan.issues.add("mot_invalid")
        elif kind == "fx":
            if fx_seen:
                scan.issues.add("fx_extra")
            else:
                fx_seen = True
                scan.effect = _parse_fx_value(val, flip_name_set, scan.issues)
        elif kind == "th":
            if th_seen:
                scan.issues.add("extra_tag")
            else:
                th_seen = True
                scan.thought = val
        i = k + 1
    scan.speech = "".join(parts)
    if final:
        if not scan.emotion_raw:
            scan.issues.add("emo_missing")
        if not scan.motion_raw:
            scan.issues.add("mot_missing")
    return scan


def parse_tag_output(raw_text: str, _count: bool = True,
                     flip_names=None,
                     gen_meta: dict | None = None) -> LLMResponse:
    """タグ方式の全文パース（非ストリーミング経路用）。

    JSON 方式の parse_llm_output と同じ安全基準（思考タグ除去・
    メタ漏れ遮断・同一文の繰り返し除去）を適用し、タグが壊れて
    いても発話が成立すれば返す。失敗時は skip=True（例外は投げない）。

    gen_meta はバックエンドが保持した生成メタ（done_reason 等）。
    no-speech の警告ログに診断情報として添える。
    """
    raw_in = (raw_text or "").strip()
    logger.debug("LLM raw output (tag)", preview=raw_in[:200])
    text = strip_think_tags(raw_in).strip()
    text = re.sub(r"```(?:json|text)?\s*", "", text)
    text = re.sub(r"```", "", text).strip()

    scan = scan_tag_output(text, final=True, flip_names=flip_names)
    speech = scan.speech.strip()

    if not speech:
        # タグだけの出力は skip の意思表示（PROTOCOL §3）だが、意図せず
        # ヘッダ直後で止まった生成と区別できないため、ok には数えず
        # no_speech として分離する（2026-11-19 計測是正）
        if scan.emotion_raw and scan.motion_raw:
            if _count:
                record_tag_parse_outcome(TAG_OUTCOME_NO_SPEECH)
            return LLMResponse(
                speech="", skip=True,
                raw={"protocol": "tag", "tag_outcome": TAG_OUTCOME_NO_SPEECH})
        if _count:
            record_tag_parse_outcome(TAG_OUTCOME_FATAL)
        log_tag_no_speech(raw_in, source="non-streaming", gen_meta=gen_meta)
        return LLMResponse(
            speech="", skip=True,
            raw={"protocol": "tag", "tag_outcome": TAG_OUTCOME_FATAL})

    outcome = TAG_OUTCOME_DEGRADED if scan.issues else TAG_OUTCOME_OK
    if _count:
        record_tag_parse_outcome(outcome)
    raw_out = {"protocol": "tag", "tag_outcome": outcome}

    # 安全基準は JSON 方式と同一（メタ漏れは喋らせない）
    if contains_meta_leak(speech):
        logger.warning("Meta leak in speech (tag), skipping",
                       speech=speech[:120])
        return LLMResponse(speech="", skip=True, raw=raw_out)
    speech = dedupe_repeated_sentences(speech)
    if not speech:
        return LLMResponse(speech="", skip=True, raw=raw_out)

    try:
        emotion = Emotion(scan.emotion_raw)
    except ValueError:
        emotion = Emotion.NEUTRAL
    try:
        motion = Motion(scan.motion_raw)
    except ValueError:
        motion = Motion.IDLE
    return LLMResponse(
        speech=speech,
        emotion=emotion,
        motion=motion,
        skip=False,
        raw=raw_out,
        effect=scan.effect,
        thought=scan.thought.strip(),
    )


def render_tag_output(resp: LLMResponse) -> str:
    """LLMResponse をタグ形式テキストに戻す。

    非ストリーミングバックエンドの generate_stream 既定実装が、
    JSON ではなくタグ形式で下流（TagSpeechStreamParser）へ流すために使う。
    """
    out = f"[emo:{resp.emotion.value}][mot:{resp.motion.value}]{resp.speech}"
    if resp.effect:
        etype = str(resp.effect.get("type", "")).strip()
        if etype in ("cutin", "se"):
            out += f"[fx:{etype}:{resp.effect.get('name', '')}]"
        elif etype == "flip":
            val = str(resp.effect.get("text") or resp.effect.get("name") or "")
            if val:
                out += f"[fx:flip:{val}]"
    if resp.thought:
        out += f"[th:{resp.thought}]"
    return out
