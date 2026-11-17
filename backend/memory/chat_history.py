"""ChatHistory: ローカル1対1チャットの会話ログを永続化する

エピソード記憶（MemoryStore）は「要約された事実」を貯めるのに対して、
ここは「発言そのもの」をキャラクターごとにJSONLで残す。用途は2つ。

  1. チャット画面を開き直したとき、前回までの会話を表示する
  2. 再起動後の最初のターンから、直近の会話を文脈としてLLMに渡す

1行1JSONの追記式なので、途中で壊れても壊れた行を飛ばせば読み続けられる。
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from backend.logger import get_logger
from backend.models import ChatMessage

logger = get_logger(__name__)

_FILENAME = "chat_history.jsonl"


class ChatHistory:
    """キャラクター1体ぶんのチャット履歴（JSONL追記）"""

    def __init__(
        self,
        character_name: str,
        prompts_dir: str | Path = "backend/llm/prompts",
        max_entries: int = 1000,
    ):
        self._character = character_name
        self._path = Path(prompts_dir) / "characters" / character_name / _FILENAME
        self._max_entries = max_entries
        self._count = -1     # 遅延カウント（初回アクセス時に数える）

    # ── 基本情報 ──────────────────────────────────────────────

    @property
    def path(self) -> Path:
        return self._path

    @property
    def character_name(self) -> str:
        return self._character

    @property
    def exists(self) -> bool:
        return self._path.exists()

    def count(self) -> int:
        if self._count < 0:
            self._count = len(self._read_lines())
        return self._count

    # ── 読み書き ──────────────────────────────────────────────

    def _read_lines(self) -> list[dict]:
        if not self._path.exists():
            return []
        out: list[dict] = []
        try:
            with open(self._path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue           # 壊れた行は飛ばして読み進める
                    if isinstance(entry, dict) and entry.get("text"):
                        out.append(entry)
        except OSError as e:
            logger.warning("ChatHistory read failed", error=str(e))
            return []
        return out

    def append(self, role: str, text: str, author: str = "", emotion: str = "") -> dict | None:
        """
        1件追記する。role は "user" / "assistant" / "partner"。
        partner は掛け合いの相方の発言で、author に相方の名前を入れる。
        """
        text = (text or "").strip()
        if not text or role not in ("user", "assistant", "partner"):
            return None
        entry = {"ts": time.time(), "role": role, "text": text}
        if author:
            entry["author"] = author
        if emotion:
            entry["emotion"] = emotion
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError as e:
            logger.warning("ChatHistory append failed", error=str(e))
            return None

        # 追記済みなので、まだ数えていなければ数え直し、そうでなければ+1でよい
        if self._count < 0:
            self._count = len(self._read_lines())
        else:
            self._count += 1
        # 増えすぎたら古いぶんを落とす（上限の1.5倍を超えたらまとめて整理）
        if self._max_entries > 0 and self._count > int(self._max_entries * 1.5):
            self.trim()
        return entry

    def recent(self, limit: int = 50) -> list[dict]:
        """新しい順ではなく、古い→新しいの順で直近 limit 件を返す"""
        if limit <= 0:
            return []
        return self._read_lines()[-limit:]

    def all(self) -> list[dict]:
        return self._read_lines()

    def trim(self, max_entries: int | None = None) -> int:
        """古い行を捨てて上限内に収める。捨てた件数を返す。"""
        keep = max_entries if max_entries is not None else self._max_entries
        if keep <= 0:
            return 0
        entries = self._read_lines()
        if len(entries) <= keep:
            self._count = len(entries)
            return 0
        dropped = len(entries) - keep
        self._write_all(entries[-keep:])
        return dropped

    def clear(self) -> int:
        """履歴を全部消す。消した件数を返す。"""
        n = self.count()
        self._write_all([])
        return n

    def _write_all(self, entries: list[dict]) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(".jsonl.tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                for e in entries:
                    f.write(json.dumps(e, ensure_ascii=False) + "\n")
            tmp.replace(self._path)
            self._count = len(entries)
        except OSError as e:
            logger.warning("ChatHistory write failed", error=str(e))

    # ── LLM文脈への復元 ───────────────────────────────────────

    def to_messages(self, turns: int = 8, user_tag: str = "viewer_comment") -> list[ChatMessage]:
        """
        直近 turns ターンぶんを ConversationLog に戻せる形にする。

        ユーザー発言は通常の会話と同じくタグで囲む。タグの中身は
        「話題として扱う」ルールがシステムプロンプト側に書いてあるため、
        復元した過去ログだけが素通しにならないようにするのが目的。
        """
        entries = self.recent(max(0, turns) * 2)
        messages: list[ChatMessage] = []
        for e in entries:
            text = str(e.get("text", "")).strip()
            if not text:
                continue
            if e.get("role") == "user":
                author = _escape(str(e.get("author", "")) or "ユーザー")
                messages.append(ChatMessage(
                    role="user",
                    content=f'<{user_tag} author="{author}">{_escape(text)}</{user_tag}>',
                ))
            elif e.get("role") == "partner":
                name = _escape(str(e.get("author", "")) or "共演者")
                messages.append(ChatMessage(
                    role="user",
                    content=f'<partner_speech name="{name}">{_escape(text)}</partner_speech>',
                ))
            else:
                messages.append(ChatMessage(role="assistant", content=text))
        return messages


def _escape(text: str) -> str:
    """ConversationLog と同じエスケープ（タグを閉じて外に出られないように）"""
    return (
        text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
    )
