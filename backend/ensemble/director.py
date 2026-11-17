"""TurnDirector: 掛け合い配信で「次に誰が喋るか」を決める

決め方（どれも出演者が1体なら常にその1体を返す）:
  - コメントへの返事  : 名指しされたキャラ → いなければしばらく喋っていない方
                       （ensemble.responder: lead なら名指しが無い限りリーダー）
  - 独り言・話題振り  : しばらく喋っていない方
  - 相方の反応       : 直前の話者以外で、しばらく喋っていない方。確率と回数上限つき
"""
from __future__ import annotations

import random
from typing import Protocol, Sequence

from backend.config import EnsembleConfig
from backend.models import Comment


class _Speaker(Protocol):
    key: str
    last_spoke_at: float

    def is_addressed(self, text: str) -> bool: ...


class TurnDirector:
    def __init__(self, cfg: EnsembleConfig, rng: random.Random | None = None):
        self._cfg = cfg
        self._rng = rng or random.Random()

    @staticmethod
    def _least_recent(pool: Sequence[_Speaker]):
        # 同点（まだ誰も喋っていない）なら並び順＝リーダーが先
        return min(pool, key=lambda a: a.last_spoke_at)

    def pick_responder(self, agents: Sequence[_Speaker], comment: Comment):
        if len(agents) == 1:
            return agents[0]
        addressed = [a for a in agents if a.is_addressed(comment.message)]
        if len(addressed) == 1:
            return addressed[0]
        if len(addressed) > 1:
            # 「Character2ちゃん、Character1ってどんな子？」のように複数の名前が出たら、
            # 先に出た方（呼びかけ）が答える。位置が取れなければしばらく喋っていない方
            first = self._first_mentioned(addressed, comment.message)
            return first or self._least_recent(addressed)
        if self._cfg.responder == "lead":
            return agents[0]
        return self._least_recent(agents)

    @staticmethod
    def _first_mentioned(agents: Sequence[_Speaker], text: str):
        best, best_pos = None, None
        for a in agents:
            pos = getattr(a, "mention_position", None)
            if pos is None:
                return None
            p = pos(text)
            if p >= 0 and (best_pos is None or p < best_pos):
                best, best_pos = a, p
        return best

    def pick_idle_speaker(self, agents: Sequence[_Speaker]):
        return self._least_recent(agents)

    def pick_reactor(self, agents: Sequence[_Speaker], speaker: _Speaker):
        others = [a for a in agents if a.key != speaker.key]
        return self._least_recent(others) if others else None

    def wants_banter(self, after: str, turn: int) -> bool:
        """
        直前の発話に相方が反応するか。
        after: "comment"（コメントへの返事のあと）/ "idle"（話題振りのあと）/ "banter"
        turn : この話題で既に何回キャラ同士でやり取りしたか
        """
        if turn >= max(0, self._cfg.max_banter_turns):
            return False
        # 2往目以降（after="banter"）は話題振りと同じ確率。長さは max_banter_turns で抑える
        prob = (self._cfg.banter_probability if after == "comment"
                else self._cfg.idle_banter_probability)
        return self._rng.random() < max(0.0, min(1.0, prob))
