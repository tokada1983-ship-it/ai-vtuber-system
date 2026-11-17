"""LLMのJSON出力をパースしてLLMResponseに変換する"""
from __future__ import annotations

import json
import re

from backend.logger import get_logger
from backend.models import Emotion, LLMResponse, Motion

logger = get_logger(__name__)


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


def parse_llm_output(raw_text: str) -> LLMResponse:
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
        logger.warning("No JSON object found in LLM output",
                       raw_preview=raw_text[:300],
                       after_strip=text[:300])
        return LLMResponse(speech="", skip=True)

    json_str = text[start : end + 1]

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.warning("JSON parse error", error=str(e), raw=json_str[:300])
        return LLMResponse(speech="", skip=True)

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

    return LLMResponse(
        speech=speech,
        emotion=emotion,
        motion=motion,
        target_comment_id=data.get("target_comment_id"),
        skip=False,
        raw=data,
    )
