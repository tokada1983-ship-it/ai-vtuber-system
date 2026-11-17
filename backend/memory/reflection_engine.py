"""ReflectionEngine: 配信終了時や定期的な自己内省（振り返り日記）を生成し、コア記憶を更新する"""
from __future__ import annotations

import re
import time
from datetime import datetime
from typing import TYPE_CHECKING

from backend.logger import get_logger

if TYPE_CHECKING:
    from backend.llm.base import LLMBackend
    from backend.memory.conversation_log import ConversationLog
    from backend.memory.long_term_memory import LongTermMemory
    from backend.memory.memory_store import MemoryStore

logger = get_logger(__name__)

_MAX_DIARY_ENTRIES = 5  # memory.md に保持する最新の日記件数
_REFLECTION_SYSTEM_TEMPLATE = """あなたは{character_name}です。
今回の配信（会話）を振り返り、あなた自身の日記・思い出を100〜200文字程度で執筆してください。

# 条件:
- あなたの口調・性格・一人称を完全に維持すること
- 印象に残ったリスナーとのやり取り、盛り上がった話題、新しく知ったことを素直な感想として書く
- 「〇〇さんと話したこと」「今日起きたこと」など具体的な思い出を含める
- 見出し（##）や日付は付けず、日記の本文テキストのみを出力すること"""


class ReflectionEngine:
    """配信全体の出来事から高次の思い出・自己認識を生成するエンジン"""

    def __init__(self, max_diary_entries: int = _MAX_DIARY_ENTRIES):
        self._max_diary_entries = max_diary_entries
        self._last_reflection: str | None = None
        self._last_reflected_at: float | None = None

    async def reflect_and_update(
        self,
        conversation_log: "ConversationLog",
        memory_store: "MemoryStore | None",
        long_term_memory: "LongTermMemory",
        llm: "LLMBackend",
        character_name: str,
    ) -> str | None:
        """
        会話ログとエピソード記憶から内省日記を生成し、memory.md を自動更新する
        """
        messages = conversation_log.get_pending() or conversation_log.get_messages()
        if not messages or len(messages) < 2:
            logger.info("ReflectionEngine: conversation too short, skipping")
            return None

        # 会話テキストの成形
        conv_text = self._format_conversation(messages)
        if len(conv_text) < 100:
            logger.info("ReflectionEngine: conversation text too small, skipping")
            return None

        recent_memories = ""
        if memory_store:
            recent_entries = memory_store.sample_recent(5)
            if recent_entries:
                recent_memories = "【最近覚えたこと】:\n" + "\n".join(f"- {e.text}" for e in recent_entries)

        prompt = f"""以下の会話記録を振り返ってください。

{recent_memories}

【今回の会話記録】:
{conv_text}

上記を振り返り、{character_name}としての素直な気持ち・日記本文（100〜200文字程度）を出力してください。"""

        system_prompt = _REFLECTION_SYSTEM_TEMPLATE.format(character_name=character_name)

        try:
            logger.info("ReflectionEngine: generating reflection diary...", character=character_name)
            raw = await self._generate_raw(llm, system_prompt, prompt)
            diary_text = self._clean_diary(raw)
            if not diary_text:
                logger.warning("ReflectionEngine: empty diary generated")
                return None

            # memory.md の更新
            today_str = datetime.now().strftime("%Y-%m-%d %H:%M")
            entry_header = f"### [{today_str}] 配信の思い出"
            new_entry = f"{entry_header}\n{diary_text}"

            self._append_to_core_memory(long_term_memory, new_entry)
            self._last_reflection = diary_text
            self._last_reflected_at = time.time()

            logger.info("ReflectionEngine: memory.md updated with new reflection",
                        date=today_str, preview=diary_text[:50])
            return diary_text

        except Exception as e:
            logger.error("ReflectionEngine error", error=str(e))
            return None

    def _append_to_core_memory(self, long_term_memory: "LongTermMemory", new_entry: str) -> None:
        """memory.md に日記セクションを追加・更新する"""
        current_md = long_term_memory.get_raw() or ""
        section_title = "## 最近の思い出・日記"

        if section_title in current_md:
            # 既存セクションの分割
            parts = current_md.split(section_title)
            before = parts[0].rstrip()
            after_section = parts[1].strip()

            # 次の大見出し（## ）があるか確認
            next_header = re.search(r"\n## [^#]", after_section)
            if next_header:
                section_body = after_section[:next_header.start()].strip()
                rest = after_section[next_header.start():]
            else:
                section_body = after_section
                rest = ""

            # 既存のエントリ（### [日付] ...）を行単位またはブロック単位で抽出
            entries = [e.strip() for e in re.split(r"(?=### \[\d{4}-\d{2}-\d{2})", section_body) if e.strip()]
            entries.insert(0, new_entry)
            # 最大件数に収める
            if len(entries) > self._max_diary_entries:
                entries = entries[:self._max_diary_entries]

            new_section = section_title + "\n\n" + "\n\n".join(entries)
            updated_md = f"{before}\n\n{new_section}\n\n{rest}".strip() if before else f"{new_section}\n\n{rest}".strip()
        else:
            # セクション新規作成
            new_section = f"{section_title}\n\n{new_entry}"
            if current_md.strip():
                updated_md = f"{current_md.rstrip()}\n\n{new_section}"
            else:
                updated_md = new_section

        long_term_memory.save(updated_md)

    @staticmethod
    def _format_conversation(messages) -> str:
        lines: list[str] = []
        for m in messages:
            content = getattr(m, "content", "")
            role = getattr(m, "role", "")
            # viewer_comment タグのパース
            vm = re.search(r'<viewer_comment\s+author="([^"]*)"[^>]*>(.*?)</viewer_comment>', content, re.DOTALL)
            pm = re.search(r'<partner_speech\s+name="([^"]*)"[^>]*>(.*?)</partner_speech>', content, re.DOTALL)
            if vm:
                lines.append(f"リスナー({vm.group(1)}): {vm.group(2).strip()}")
            elif pm:
                lines.append(f"共演者({pm.group(1)}): {pm.group(2).strip()}")
            elif role == "user":
                lines.append(f"ユーザー: {content}")
            else:
                lines.append(f"自分: {content}")
        text = "\n".join(lines)
        if len(text) > 4000:
            text = "…（中略）…\n" + text[-4000:]
        return text

    @staticmethod
    def _clean_diary(raw: str) -> str:
        text = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
        text = re.sub(r"```.*?", "", text).strip()
        text = re.sub(r"^#+\s*.*?\n", "", text)  # 見出し行削除
        return text.strip()

    @staticmethod
    async def _generate_raw(llm: "LLMBackend", system_prompt: str, prompt: str) -> str:
        from backend.models import ChatMessage
        messages = [ChatMessage(role="user", content=prompt)]
        # 内省日記は待機系の作業なので、ハイブリッド時はローカル側へ寄せる
        resp = await llm.generate(system_prompt, messages, route="idle")
        return resp.speech or ""
