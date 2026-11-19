"""AdaptiveContextController: Ollama の num_ctx を段階式に自動調整する

問題2（2026-11-19）の真因は、プロンプトが num_ctx=4096 に達して生成の
残り枠が数トークンしか残らず、発話が done_reason=length で切断された
ことだった。固定値 8192 への拡張は当座の対策で、文脈が育てば再び天井に
当たる。ここでは生成ごとの観測（prompt_eval_count / done_reason）を
見て、段（tier）を1段ずつ上げ下げする制御を実装する。

設計:
- 段は config.llm.context_tiers（既定 [4096, 6144, 8192, 12288, 16384]）、
  初期段は context_initial（既定 8192）以下で最大の段。
- 上げる: 直近の生成で prompt_eval_count が現段の raise_ratio(0.85) 超、
  または done_reason=length の切断 → 次のターン境界で1段上げる。
- 下げる: prompt_eval_count が現段の lower_ratio(0.45) 未満の状態が
  持続したら1段下げる。持続の判定は「lower_turns 回連続」または
  「lower_window_sec 秒以上継続（最低3回観測）」のどちらか早い方。
  生成頻度は配信状況で大きく変わるため、回数だけでは静かな配信で
  永遠に下がらず、時間だけでは疎な観測で誤判定する。両方を見る。
  下げると Ollama の KV キャッシュが縮み、VRAM を Whisper / TTS /
  描画へ返せる。
- 段の適用はターン境界（begin_turn）でのみ行い、生成途中では変えない。
  観測は保留（pending）として積み、適用時に初めて段が動く。
  適用後は cooldown_sec の間は再変更しない（上げ下げの往復防止）。
- 最上段でも収まらない兆候（前回が length 切断、または前回の
  prompt_eval_count + 生成上限が最上段を超える）がある場合は
  needs_pre_truncate() が True になり、呼び出し側（OllamaBackend）が
  生成前に履歴の古い部分から切り詰める。

このクラスは Ollama への依存を持たない純粋な状態機械で、時刻は
clock 注入で差し替えられる（テスト用）。
"""
from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class _PendingChange:
    to_index: int
    reason: str
    prompt_eval_count: int | None
    done_reason: str | None


class AdaptiveContextController:
    """num_ctx の段を観測に基づいて上げ下げする状態機械（バックエンド毎）"""

    def __init__(
        self,
        tiers: list[int],
        initial: int = 8192,
        *,
        raise_ratio: float = 0.85,
        lower_ratio: float = 0.45,
        cooldown_sec: float = 300.0,
        lower_turns: int = 50,
        lower_window_sec: float = 1800.0,
        clock=time.monotonic,
    ):
        cleaned = sorted({int(t) for t in tiers if int(t) > 0})
        self._tiers: list[int] = cleaned or [8192]
        self._idx = self._nearest_index(int(initial))
        self._raise_ratio = float(raise_ratio)
        self._lower_ratio = float(lower_ratio)
        self._cooldown_sec = float(cooldown_sec)
        self._lower_turns = max(1, int(lower_turns))
        self._lower_window_sec = float(lower_window_sec)
        self._clock = clock
        self._pending: _PendingChange | None = None
        self._last_change_at: float | None = None
        self._low_count = 0
        self._low_since: float | None = None
        self.last_prompt_eval_count: int | None = None
        self.last_done_reason: str | None = None

    # ── 基本状態 ──────────────────────────────────────────────

    def _nearest_index(self, value: int) -> int:
        """value 以下で最大の段のインデックス（無ければ最下段）"""
        idx = 0
        for i, t in enumerate(self._tiers):
            if t <= value:
                idx = i
        return idx

    @property
    def tiers(self) -> list[int]:
        return list(self._tiers)

    @property
    def index(self) -> int:
        return self._idx

    @property
    def current_num_ctx(self) -> int:
        return self._tiers[self._idx]

    @property
    def at_top(self) -> bool:
        return self._idx >= len(self._tiers) - 1

    @property
    def pending_num_ctx(self) -> int | None:
        if self._pending is None:
            return None
        return self._tiers[self._pending.to_index]

    # ── 観測 ──────────────────────────────────────────────────

    def observe(self, meta: dict) -> None:
        """生成1回分のメタ（prompt_eval_count / done_reason）を取り込む。

        ここでは段そのものは変えず、変更が必要なら保留に積むだけ。
        実際の適用は begin_turn()（ターン境界）で行う。
        """
        pec = meta.get("prompt_eval_count")
        if pec is not None:
            try:
                pec = int(pec)
            except (TypeError, ValueError):
                pec = None
        done = meta.get("done_reason")
        if pec is not None:
            self.last_prompt_eval_count = pec
        if done is not None:
            self.last_done_reason = done
        cur = self.current_num_ctx

        # 上げる条件: 切断が起きた、またはプロンプトが現段の上限間近
        raise_needed = done == "length" or (
            pec is not None and pec > cur * self._raise_ratio)
        if raise_needed:
            self._low_count = 0
            self._low_since = None
            if not self.at_top:
                reason = ("length_truncation" if done == "length"
                          else "high_prompt_usage")
                self._pending = _PendingChange(
                    self._idx + 1, reason, pec, done)
            return

        # 下げる条件: 低使用が持続した
        if pec is not None and pec < cur * self._lower_ratio:
            now = self._clock()
            if self._low_since is None:
                self._low_since = now
                self._low_count = 1
            else:
                self._low_count += 1
            sustained = self._low_count >= self._lower_turns or (
                self._low_count >= 3
                and now - self._low_since >= self._lower_window_sec)
            if sustained and self._idx > 0 and self._pending is None:
                self._pending = _PendingChange(
                    self._idx - 1, "sustained_low_usage", pec, done)
        else:
            self._low_count = 0
            self._low_since = None

    # ── ターン境界での適用 ────────────────────────────────────

    def begin_turn(self) -> dict | None:
        """保留中の段変更を適用する。適用したら変更内容の dict を返す。

        クールダウン中は適用せず保留を維持する（往復防止のヒステリシス）。
        """
        if self._pending is None or self._pending.to_index == self._idx:
            self._pending = None
            return None
        now = self._clock()
        if (self._last_change_at is not None
                and now - self._last_change_at < self._cooldown_sec):
            return None
        change = {
            "from_num_ctx": self.current_num_ctx,
            "to_num_ctx": self._tiers[self._pending.to_index],
            "reason": self._pending.reason,
            "prompt_eval_count": self._pending.prompt_eval_count,
            "done_reason": self._pending.done_reason,
        }
        self._idx = self._pending.to_index
        self._pending = None
        self._last_change_at = now
        self._low_count = 0
        self._low_since = None
        return change

    # ── 最上段での溢れ対策 ────────────────────────────────────

    def needs_pre_truncate(self, max_tokens: int) -> bool:
        """最上段でも前回のプロンプトが収まらなかった兆候があるか。

        True の間、呼び出し側は生成前に履歴を切り詰める。実際の
        切り詰め量は呼び出し側が推定トークン数を見て決めるため、
        兆候があっても推定が収まっていれば何も起きない。
        """
        if not self.at_top:
            return False
        if self.last_done_reason == "length":
            return True
        pec = self.last_prompt_eval_count
        return pec is not None and pec + int(max_tokens) > self.current_num_ctx

    # ── 状態表示 ──────────────────────────────────────────────

    def status(self) -> dict:
        return {
            "num_ctx": self.current_num_ctx,
            "tier_index": self._idx,
            "tiers": list(self._tiers),
            "pending_num_ctx": self.pending_num_ctx,
            "last_prompt_eval_count": self.last_prompt_eval_count,
            "last_done_reason": self.last_done_reason,
        }
