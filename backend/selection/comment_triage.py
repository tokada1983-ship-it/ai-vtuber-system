"""CommentTriage: コメント応答の選別（トリアージ）

配信コメントは全部に応答する必要はない。この段はコメント応答キューの
手前で「応答するか／しないか」だけを決め、判定カテゴリと理由をログに残す。
記録（StreamArchiver）と視聴者プロフィールの更新は応答の有無と分離して
従来どおり行われる（落としたコメントの記録は Orchestrator 側が担う）。

判定の順序（上から順に評価し、最初に当たったもので決める）:
  1. ng_word          … config の NG ワードを含む → 落とす
  2. duplicate        … 同一ユーザーの同一文が短時間に再送された → 落とす
  3. spam_repeat      … 同一文字の連打（「ああああ」「？？？？」等）→ 落とす
  4. always_*         … スパチャ／キャラ名の呼びかけ／質問形／初コメ → 必ず拾う
  5. reaction_only    … 「草」「www」のみ・絵文字のみ等のリアクションだけ → 落とす
  6. scored           … それ以外は混雑度（キュー滞留）に応じた確率で拾う。
                        飽和時にスコアが低いものは確率判定なしで落とす

無効化（comment_filter.enabled=false）すれば、この段は素通りし従来動作に戻る。
"""
from __future__ import annotations

import random
import re
import time
from collections import Counter
from dataclasses import dataclass
from typing import Callable

from backend.config import CommentFilterConfig
from backend.logger import get_logger
from backend.models import Comment

logger = get_logger(__name__)

# 質問形とみなす表現（CommentSelector のスコア付けと概ね揃える）
QUESTION_MARKERS = (
    "？", "?", "ですか", "ますか", "でしょうか",
    "かな", "かしら", "なんで", "どうして", "教えて",
)

# リアクションだけのコメントとみなす文字（草・笑・w・！・拍手の8 等）
_REACTION_ONLY_RE = re.compile(r"[草ｗwW笑！!8８]+")
# リアクション判定で取り除く文字（残りに実質的な文字が無ければリアクションだけ）
_REACTION_STRIP_RE = re.compile(r"[草笑ｗwW！!]")


@dataclass
class TriageResult:
    """トリアージの判定結果"""
    picked: bool          # 応答キューへ進めるか
    category: str         # 判定カテゴリ（ログ・テスト用）
    reason: str           # 人間可読な理由
    probability: float = 1.0  # 確率判定に使った拾う確率（確定判定では 1.0 / 0.0）


def _normalize_text(message: str) -> str:
    """重複判定用に空白を畳んで正規化する"""
    return " ".join((message or "").split())


def _is_repeat_flood(text: str, min_length: int, ratio: float) -> bool:
    """同一文字の連打かどうか（例: ああああああ、？？？？？？）"""
    if len(text) < max(1, min_length):
        return False
    most = Counter(text).most_common(1)[0][1]
    return most / len(text) >= ratio


def _is_reaction_only(text: str) -> bool:
    """「草」「www」のみ・絵文字のみ等の、実質を伴わないリアクションだけか"""
    if _REACTION_ONLY_RE.fullmatch(text):
        return True
    core = _REACTION_STRIP_RE.sub("", text)
    return not any(c.isalnum() for c in core)


class CommentTriage:
    """コメント1件ごとに「応答するか」を判定する（状態はセッション内のみ）"""

    def __init__(
        self,
        cfg: CommentFilterConfig,
        character_names: str | list[str] = "",
        known_author: Callable[[Comment], bool] | None = None,
        rng: Callable[[], float] | None = None,
        clock: Callable[[], float] | None = None,
    ):
        self._cfg = cfg
        names = [character_names] if isinstance(character_names, str) \
            else list(character_names)
        self._character_names = [n for n in names if n]
        # 過去にコメントしたことがある視聴者か（視聴者プロフィール照会用）
        self._known_author = known_author
        self._rng = rng or random.random
        self._clock = clock or time.time
        # このセッションで見た発言者（初コメ判定用）
        self._seen_authors: set[str] = set()
        # 重複判定: {(発言者キー, 正規化文): 最後に見た時刻}
        self._recent_texts: dict[tuple[str, str], float] = {}

    # ── 判定 ────────────────────────────────────────────────

    def evaluate(self, comment: Comment, *, score: float,
                 queue_size: int) -> TriageResult:
        """コメントを評価し、応答キューへ進めるかどうかを返す。

        呼び出しごとに「見た発言者」と「直近の同一文」の状態を更新する
        （落としたコメントも次回以降の初コメ・重複判定の材料になる）。
        """
        cfg = self._cfg
        if not cfg.enabled:
            return TriageResult(True, "disabled", "comment filter disabled")

        now = self._clock()
        author_key = comment.author_channel_id or comment.author or ""
        text = _normalize_text(comment.message)
        is_first = self._is_first_comment(comment, author_key)

        result = self._judge(comment, text=text, score=score,
                             queue_size=queue_size, is_first=is_first,
                             author_key=author_key, now=now)

        # 状態更新（判定の後。今回のコメント自体は次回以降の材料になる）
        if author_key:
            self._seen_authors.add(author_key)
        if text:
            self._recent_texts[(author_key, text)] = now
            self._prune_recent(now)

        logger.info(
            "Comment triage",
            picked=result.picked, category=result.category,
            reason=result.reason, author=comment.author,
            score=score, queue=queue_size,
            message=(comment.message or "")[:40],
        )
        return result

    def _judge(self, comment: Comment, *, text: str, score: float,
               queue_size: int, is_first: bool, author_key: str,
               now: float) -> TriageResult:
        cfg = self._cfg
        message = comment.message or ""

        # 1. NGワード（必ず拾う条件より優先する）
        for word in cfg.ng_words:
            if word and word in message:
                return TriageResult(False, "ng_word", f"NG word: {word}", 0.0)

        # 2. 重複（同一ユーザーの同一文が短時間に再送）
        if cfg.duplicate_window_sec > 0 and text:
            last = self._recent_texts.get((author_key, text))
            if last is not None and now - last < cfg.duplicate_window_sec:
                return TriageResult(False, "duplicate",
                                    "same text resent within window", 0.0)

        # 3. 同一文字の連打（「？？？？」も質問として拾わないよう質問判定より前）
        if _is_repeat_flood(text, cfg.spam_repeat_min_length,
                            cfg.spam_repeat_ratio):
            return TriageResult(False, "spam_repeat",
                                "single-character flood", 0.0)

        # 4. 必ず拾う条件
        if comment.is_superchat:
            return TriageResult(True, "always_superchat", "superchat")
        if any(n in message for n in self._character_names):
            return TriageResult(True, "always_name", "character name call")
        if any(m in message for m in QUESTION_MARKERS):
            return TriageResult(True, "always_question", "question form")
        if is_first:
            return TriageResult(True, "always_first", "first comment")

        # 5. リアクションだけのコメント
        if text and _is_reaction_only(text):
            return TriageResult(False, "reaction_only",
                                "reaction only (no substance)", 0.0)

        # 6. 混雑度に応じた確率判定
        if queue_size >= cfg.saturated_queue_size:
            if score < cfg.saturated_min_score:
                return TriageResult(
                    False, "saturated_low_score",
                    f"queue saturated and score {score} < "
                    f"{cfg.saturated_min_score}", 0.0)
            probability = cfg.saturated_pickup_probability
        elif queue_size >= cfg.congested_queue_size:
            probability = cfg.congested_pickup_probability
        else:
            probability = cfg.base_pickup_probability

        if probability >= 1.0:
            return TriageResult(True, "scored", "picked (p=1.0)", 1.0)
        if probability <= 0.0 or self._rng() >= probability:
            return TriageResult(False, "probability_miss",
                                f"probability roll missed (p={probability})",
                                probability)
        return TriageResult(True, "scored",
                            f"probability roll hit (p={probability})",
                            probability)

    # ── 状態 ────────────────────────────────────────────────

    def _is_first_comment(self, comment: Comment, author_key: str) -> bool:
        """この視聴者の初コメかどうか（セッション内の目撃＋プロフィール照会）"""
        if author_key and author_key in self._seen_authors:
            return False
        if self._known_author is not None:
            try:
                if self._known_author(comment):
                    return False
            except Exception:
                pass  # 照会に失敗したら初コメ扱い（拾う側に倒す）
        return True

    def _prune_recent(self, now: float) -> None:
        """重複判定の古い記録を捨てる（メモリの際限ない増加を防ぐ）"""
        window = max(float(self._cfg.duplicate_window_sec), 1.0)
        if len(self._recent_texts) > 512:
            self._recent_texts = {
                k: v for k, v in self._recent_texts.items()
                if now - v < window * 2
            }
