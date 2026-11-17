"""CommentSelector: コメントに優先度を付ける"""
from __future__ import annotations

import time

from backend.config import OrchestratorConfig
from backend.models import Comment


class CommentSelector:
    """
    各コメントにスコアを付与する。
    Orchestratorはこのスコア順のキューを処理する。
    """

    def __init__(self, cfg: OrchestratorConfig, character_name: str | list[str] = ""):
        self._cfg = cfg
        # 掛け合いでは出演者全員の名前（どれかが呼ばれていれば加点）
        names = [character_name] if isinstance(character_name, str) else list(character_name)
        self._character_names = [n for n in names if n]
        # 同一ユーザーの最終応答時刻 {channel_id: timestamp}
        self._last_response_time: dict[str, float] = {}

    def score(self, comment: Comment) -> float:
        """0.0 〜 10.0 のスコアを返す。-1.0 なら無視（フィルタ）。"""

        # 長すぎるコメントは弾く
        if len(comment.message) > self._cfg.max_comment_length:
            return -1.0

        # 空コメント無視
        if not comment.message.strip():
            return -1.0

        # 連投抑制
        last = self._last_response_time.get(comment.author_channel_id, 0.0)
        if time.time() - last < self._cfg.comment_dedupe_window_sec:
            return 0.5  # 低スコアだが弾かない

        score = 2.0  # 基本スコア

        # スパチャ（最優先）
        if comment.is_superchat:
            score += 5.0
            if comment.superchat_amount:
                score += min(comment.superchat_amount / 1000, 3.0)

        # メンバー・モデレーター
        if comment.is_member:
            score += 2.0
        if comment.is_moderator:
            score += 1.5

        # キャラ名呼びかけ
        if any(n in comment.message for n in self._character_names):
            score += 2.0

        # 質問形
        if any(c in comment.message for c in ["？", "?", "ですか", "ますか", "かな"]):
            score += 1.5

        return min(score, 10.0)

    def mark_responded(self, comment: Comment) -> None:
        """応答済みとしてマーク（連投抑制用）"""
        self._last_response_time[comment.author_channel_id] = time.time()
