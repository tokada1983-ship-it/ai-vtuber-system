"""ConversationLog: LLMに渡す会話履歴と、記憶整理待ちの会話を管理する"""
from __future__ import annotations

from collections import deque

from backend.models import ChatMessage, Comment, LLMResponse


class ConversationLog:
    """
    2つの用途で会話を保持する。

      _history : LLMに毎ターン渡す直近の文脈。長いとコンテキストを食うので短く保つ
      _pending : まだ長期記憶に整理していない会話。整理が走るまで捨ててはいけない

    この2つを分けているのは、以前は _history だけを持っており、
    「整理は50ターンごと、文脈は20ターンぶん保持」という設定だと
    整理が走る前に30ターンぶんが deque から押し出されて消えていたため。
    整理に渡すぶんは別に貯めておく。
    """

    # 整理待ちが青天井にならないよう上限を設ける（1ターン=2メッセージ）
    _PENDING_HARD_LIMIT = 600

    def __init__(self, max_turns: int = 100):
        self._max_turns = max_turns
        self._history: deque[ChatMessage] = deque(maxlen=max_turns * 2)  # user + assistant で×2
        self._pending: deque[ChatMessage] = deque(maxlen=self._PENDING_HARD_LIMIT)
        self._total_turns: int = 0       # 累計ターン数（クリアしてもリセットしない）
        self._turns_since_consolidation: int = 0  # 前回整理からのターン数

    def _append(self, msg: ChatMessage) -> None:
        self._history.append(msg)
        self._pending.append(msg)

    def add_comment(self, comment: Comment) -> None:
        """視聴者コメントを履歴に追加"""
        content = (
            f'<viewer_comment author="{_sanitize(comment.author)}" id="{_sanitize(comment.id)}">'
            f"{_sanitize(comment.message)}"
            f"</viewer_comment>"
        )
        self._append(ChatMessage(role="user", content=content, comment_id=comment.id))

    def add_partner(self, name: str, text: str) -> None:
        """
        共演者（掛け合いの相方）の発言を履歴に追加する。

        自分の発言ではないので assistant にはせず、user 側に名札つきで積む。
        ターン数は数えない（ターン＝自分が喋った回数）。
        """
        text = (text or "").strip()
        if not text:
            return
        content = (f'<partner_speech name="{_sanitize(name)}">'
                   f"{_sanitize(text)}</partner_speech>")
        self._append(ChatMessage(role="user", content=content))

    def add_response(self, response: LLMResponse) -> None:
        """AIの応答を履歴に追加"""
        if not response.skip and response.speech:
            self._append(ChatMessage(role="assistant", content=response.speech))
            self._total_turns += 1
            self._turns_since_consolidation += 1

    def add_idle(self, text: str) -> None:
        """独り言を履歴に追加"""
        self._append(ChatMessage(role="assistant", content=text))
        self._total_turns += 1
        self._turns_since_consolidation += 1

    def restore(self, messages: list[ChatMessage]) -> int:
        """
        保存済みの過去ログを文脈として読み戻す（起動直後の1ターン目から
        「さっきの話」が通じるようにするため）。

        整理待ち(_pending)には入れない。過去ログは前回のセッションで
        整理済み、または整理する機会を逃したぶんであり、ここで混ぜると
        同じ会話を二重に記憶へ書き込むことになる。
        """
        added = 0
        for msg in messages:
            if not msg.content:
                continue
            self._history.append(msg)
            added += 1
        return added

    def get_messages(self) -> list[ChatMessage]:
        """LLMに渡すメッセージリストを返す（直近の文脈のみ）"""
        return list(self._history)

    def get_pending(self) -> list[ChatMessage]:
        """まだ長期記憶に整理していない会話をすべて返す"""
        return list(self._pending)

    def mark_consolidated(self) -> None:
        """整理完了を記録し、整理待ちバッファとターンカウンターをリセットする"""
        self._turns_since_consolidation = 0
        self._pending.clear()

    def should_consolidate(self, trigger_turns: int) -> bool:
        """
        trigger_turns ターン以上経過していれば True を返す。
        trigger_turns <= 0 の場合は常に False。
        """
        if trigger_turns <= 0:
            return False
        return self._turns_since_consolidation >= trigger_turns

    def set_max_turns(self, max_turns: int) -> None:
        """文脈として保持するターン数を変更する（既存の履歴は引き継ぐ）"""
        if max_turns <= 0 or max_turns == self._max_turns:
            return
        self._max_turns = max_turns
        self._history = deque(self._history, maxlen=max_turns * 2)

    def clear_context(self) -> None:
        """文脈（履歴）だけを破棄し、整理待ちと整理カウンタは保持する。

        文脈リセット（飽和・LoopGuard 由来）の目的は「LLM に渡す文脈を
        捨てる」ことであり、まだ長期記憶に整理していない会話
        （_pending）と整理トリガのカウンタまで消してしまうと、リセット
        が頻発する配信では整理閾値へ到達できなくなる（2026-11-22
        5時間検証で実証：リセット約20ターンごとにカウンタが初期化され、
        閾値50へ構造的に届かなかった）。保持した pending は次の整理
        サイクルが通常どおり拾う。
        """
        self._history.clear()

    def clear(self) -> None:
        """会話をすべて破棄する（整理待ちも含む）"""
        self._history.clear()
        self._pending.clear()
        # total_turnsはクリアしない（統計用）
        self._turns_since_consolidation = 0

    @property
    def total_turns(self) -> int:
        return self._total_turns

    @property
    def turns_since_consolidation(self) -> int:
        return self._turns_since_consolidation

    @property
    def history_size(self) -> int:
        return len(self._history)

    @property
    def pending_size(self) -> int:
        return len(self._pending)

    @property
    def max_turns(self) -> int:
        return self._max_turns


def _sanitize(text: str) -> str:
    """
    視聴者が <viewer_comment> タグを閉じて外に出るのを防ぐ。

    タグで囲むだけの防御はエスケープが無いと成立しない。実際、
    `</viewer_comment>` を含むコメントを送るとタグの外に文字列を置けてしまい、
    そのまま記憶整理にも流れて長期記憶に residue が残る恐れがあった。
    """
    if not text:
        return ""
    return (
        text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
    )
