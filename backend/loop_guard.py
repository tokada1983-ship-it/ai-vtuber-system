"""発話ループ検出（言い換えレベルの反復に対する最後の段）。

既存の防御は2段あるが、どちらも「会話全体がループに入った」状態は
検出できない:

- リピートブロック（orchestrator._should_block_repeat）: 直近12発話との
  完全一致・包含・高類似（SequenceMatcher > 0.85）だけを見る。言い換えの
  ローテーションと、窓の外（実測で約10分超）での再演はすり抜ける。
- テーマ反復検知（_note_theme_repetition）: 単語（バイグラム）単位の
  禁止とヒント。モデルが同義語へ乗り換えると追いつけず、強制切替で
  材料を変えても発話の型（定型文の循環）が残る。

本モジュールは発話の「新規性」を数える: 直近 window_speeches 件の
発話のうち、過去の全発話のどれかと文字 trigram の Jaccard 類似度が
dup_threshold 以上（＝再演・言い換え）であるものの件数が
dup_min_count 以上になったら、会話がループに入ったと判定する。
正常なコールバック（定番フレーズの単発の再演）では割合が上がらない
ため、誤発火しにくい。

閾値の根拠（2026-11-21、過去配信アーカイブ5本でオフライン検証）:
window=16 / threshold=0.45 / min_count=6 / cooldown=600秒 のとき
- 2026-11-20 ループ夜（249発話）: 20:31・20:43・20:54・21:50・22:20・
  22:33 に発火（問題の 22:37 区間を含む）
- 2026-11-19 夜（165発話、内容的にも定型文の循環が多かった）: 9回発火
- 2026-11-19 3時間テスト（140発話）・1時間テスト（35発話）: 0回
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass

# orchestrator._normalize_speech と同じ正規化（句読点・記号・空白を落とす）
_NORMALIZE_RE = re.compile(r"[\s。、！？!?…♪・「」『』（）()\"“”'’]")

# 代表語の選定から外す trigram（助詞・語尾などの機能語だけのもの）
_STOP_GRAMS = {
    "ですね", "ますね", "ですよ", "ますよ", "でしょ", "ました", "ません",
    "ください", "ださい", "ありが", "りがと", "がとう", "うござ", "ござい",
    "という", "ていま", "てます", "でいる", "ている",
}


def normalize_speech(text: str) -> str:
    """反復判定用に、句読点・記号・空白を落として正規化する"""
    return _NORMALIZE_RE.sub("", text or "")


def _trigrams(norm: str) -> set[str]:
    if len(norm) < 3:
        return set()
    return {norm[i:i + 3] for i in range(len(norm) - 2)}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


@dataclass
class LoopEvent:
    """ループ検出の発火イベント"""
    level: int             # 1=初回（ヒント＋強制切替）, 2=再発（文脈リセットへ昇格）
    dup_count: int         # 窓の中の再演発話数
    window: int            # 窓の発話数
    max_similarity: float  # 発火した発話の、過去発話に対する最大類似度
    word: str              # 再演クラスタの代表 trigram（禁止語・表示用）


class SpeechLoopGuard:
    """発話の再演率を見て会話ループを検出する（両キャラ合算）"""

    def __init__(self, *, enabled: bool = True,
                 window_speeches: int = 16,
                 dup_threshold: float = 0.45,
                 dup_min_count: int = 6,
                 cooldown_sec: float = 600.0,
                 escalate_window_sec: float = 1800.0,
                 history_max: int = 1000) -> None:
        self.enabled = enabled
        self.window_speeches = max(1, window_speeches)
        self.dup_threshold = dup_threshold
        self.dup_min_count = max(1, dup_min_count)
        self.cooldown_sec = cooldown_sec
        self.escalate_window_sec = escalate_window_sec
        self.history_max = max(10, history_max)
        self._history: list[set[str]] = []   # 過去発話の trigram 集合
        self._dup_flags: list[bool] = []     # 直近発話の再演フラグ
        self._fires = 0
        self._last_fire_at = 0.0
        self._last_event: LoopEvent | None = None

    def note_speech(self, text: str, *, now: float | None = None) -> LoopEvent | None:
        """発話を1件記録し、ループに入っていたら LoopEvent を返す。

        無効化されている間は何も記録しない（有効化した時点で
        まっさらな状態から数え直す）。
        """
        if not self.enabled:
            return None
        now = time.time() if now is None else now
        grams = _trigrams(normalize_speech(text))
        max_sim = 0.0
        for prev in self._history:
            sim = _jaccard(grams, prev)
            if sim > max_sim:
                max_sim = sim
        is_dup = bool(grams) and max_sim >= self.dup_threshold
        self._history.append(grams)
        del self._history[:-self.history_max]
        self._dup_flags.append(is_dup)
        del self._dup_flags[:-self.window_speeches]
        if len(self._dup_flags) < self.window_speeches:
            return None
        dup_count = sum(self._dup_flags)
        if dup_count < self.dup_min_count:
            return None
        if now - self._last_fire_at < self.cooldown_sec:
            return None
        level = 2 if (self._last_fire_at
                      and now - self._last_fire_at <= self.escalate_window_sec) else 1
        event = LoopEvent(level=level, dup_count=dup_count,
                          window=self.window_speeches,
                          max_similarity=max_sim,
                          word=self._dominant_gram())
        self._fires += 1
        self._last_fire_at = now
        self._last_event = event
        if level >= 2:
            # 昇格後は窓を仕切り直す（リセットされた文脈で再評価する）。
            # 履歴は残す: 文脈リセットは LLM 側の都合で、聞き手にとっての
            # 「もう聞いた話」は消えないため。
            self._dup_flags.clear()
        return event

    def _dominant_gram(self) -> str:
        """現在の窓の再演発話に最もよく現れる内容 trigram を返す"""
        window = self._history[-self.window_speeches:]
        flags = self._dup_flags[-len(window):]
        from collections import Counter
        coverage: Counter = Counter()
        for grams, is_dup in zip(window, flags):
            if not is_dup:
                continue
            for g in grams:
                # 漢字・カタカナを含む gram だけを代表語の候補にする
                if any(("\u4e00" <= c <= "\u9fff") or ("\u30a0" <= c <= "\u30ff")
                       for c in g):
                    coverage[g] += 1
        for gram, _n in coverage.most_common():
            if gram in _STOP_GRAMS:
                continue
            # 出演者名・一人称を含む断片は代表語にしない（飽和検出の
            # 停止グラムと同じ考え方。名前は話題ではないため）
            if any(s in gram for s in ("ボク", "Character1", "Character2", "わたし")):
                continue
            return gram
        return ""

    def stats(self) -> dict:
        """ダッシュボード/API 用の計測スナップショット"""
        return {
            "enabled": self.enabled,
            "fires": self._fires,
            "last_fire_at": self._last_fire_at or None,
            "last_level": self._last_event.level if self._last_event else 0,
            "last_dup_count": self._last_event.dup_count if self._last_event else 0,
            "recent_dup_count": sum(self._dup_flags),
            "window": self.window_speeches,
        }
