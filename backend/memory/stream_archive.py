"""StreamArchiver: 配信の全発話を記録し、終了時に「配信エピソード」として長期記憶へ残す

背景（現状の穴）:
  配信モードでは発話は ChatHistory に保存されず（ローカルチャット専用）、
  終了時の記憶整理も ConversationLog の直近100ターン／6000字しか見ないため、
  配信で話した内容の大半が記憶に残らなかった。

仕組み:
  - 配信中: 発話とコメントを JSONL ジャーナルへ逐次追記する
    （data/streams/<開始時刻>-<video_id>.jsonl）
  - 配信終了時: ジャーナル全体をチャンク要約→統合して「配信エピソード」を作り、
    出演者全員の MemoryStore へ source="stream_episode" で投入する。
    要約と全発話は <journal>.episode.md にも残す（人が読めるアーカイブ）
  - ジャーナルだけ置かれた未処理ファイル（過去分のバックフィル等）は
    pending_journals() で拾い、次回起動時に同じ経路でエピソード化する
"""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

from backend.logger import get_logger
from backend.memory.memory_consolidator import MemoryConsolidator
from backend.models import ChatMessage

logger = get_logger(__name__)

_CHUNK_CHARS = 5500     # チャンク要約に渡す記録の上限
_MIN_SPEECHES = 5       # これ未満の短い run はエピソード化しない

_CHUNK_SYSTEM = """あなたは配信の会話記録を要約するアシスタントです。
渡された記録の一部から、後で「その配信で何があったか」を具体的に思い出すための材料を箇条書きで抽出してください。
- 話した話題・ニュースは、固有名詞（人名・作品名・製品名・地名・数値など）を落とさず具体的に書く
- 話題について決まったこと・結論が出た場合は、その結論まで書く
- 誰の話題か（Character1／Character2／視聴者とのやり取りか）が分かるように書く
- 「楽しかった」「盛り上がった」など感想だけの行は禁止。必ず「何について話したか」を書く
- 発言の丸写しはしない。要点だけを1行ずつ
- 箇条書きのみを出力する"""

_EPISODE_SYSTEM = """あなたは配信の記録係です。配信の要約材料から、その配信のエピソード記憶を作ってください。
必ず三人称の中立文で書いてください（「ボク」「わたし」などキャラクターの一人称は使わず、主語は「Character1は」「Character2は」「二人は」のように明示する）。
出力は次の形式のみ：
1行目: 「話題：」に続けて、その配信で扱った主な話題のキーワード（固有名詞を中心に最大5つ、カンマ区切り）
2行目: 「{date}の配信では、…」で始まる2〜3文の要約（扱った話題・固有名詞・出た結論を具体的に含める。感想だけの要約は禁止。視聴者との印象的なやり取りがあれば含める）
3行目以降: 「- 」で始まる主な話題の箇条書き（最大5件、1件1行、話題ごとに独立した文にする）"""


class StreamArchiver:
    """配信ジャーナルの記録と、終了時のエピソード化を担う"""

    def __init__(self, base_dir: str | Path):
        self._dir = Path(base_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._run_path: Path | None = None
        self._video_id: str = ""

    # ── 記録 ────────────────────────────────────────────────

    def start_run(self, video_id: str) -> None:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        safe = "".join(c for c in (video_id or "local")
                       if c.isalnum() or c in "-_")[:32] or "local"
        self._run_path = self._dir / f"{ts}-{safe}.jsonl"
        self._video_id = video_id or ""
        self._append({"type": "run_start", "ts": time.time(),
                      "video_id": self._video_id})
        logger.info("StreamArchiver: run started", path=str(self._run_path))

    @property
    def has_run(self) -> bool:
        return self._run_path is not None

    def _append(self, entry: dict) -> None:
        if self._run_path is None:
            return
        try:
            with open(self._run_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError as e:
            logger.warning("StreamArchiver append failed", error=str(e))

    def record_speech(self, speaker: str, text: str, kind: str = "talk") -> None:
        text = (text or "").strip()
        if not text:
            return
        self._append({"type": "speech", "ts": time.time(),
                      "speaker": speaker, "kind": kind, "text": text})

    def record_comment(self, author: str, text: str) -> None:
        text = (text or "").strip()
        if not text:
            return
        self._append({"type": "comment", "ts": time.time(),
                      "author": author, "text": text})

    # ── エピソード化 ────────────────────────────────────────

    def pending_journals(self) -> list[Path]:
        """エピソード化されていない過去ジャーナル（現在の run を除く）"""
        out: list[Path] = []
        for p in sorted(self._dir.glob("*.jsonl")):
            if p == self._run_path:
                continue
            if not p.with_suffix(".episode.md").exists():
                out.append(p)
        return out

    async def build_episode(self, llm, stores: dict, topics=None) -> dict | None:
        """現在の run をエピソード化する（停止時に呼ぶ）"""
        if self._run_path is None:
            return None
        path, self._run_path = self._run_path, None
        return await self.build_episode_from_journal(path, llm, stores,
                                                     topics=topics)

    async def build_episode_from_journal(self, path, llm, stores: dict,
                                         topics=None) -> dict | None:
        path = Path(path)
        entries: list[dict] = []
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    entries.append(json.loads(line))
        except Exception as e:
            logger.warning("StreamArchiver: journal read failed",
                           path=str(path), error=str(e))
            return None

        speeches = [e for e in entries if e.get("type") == "speech"]
        if len(speeches) < _MIN_SPEECHES:
            logger.info("StreamArchiver: too few speeches, skip episode",
                        count=len(speeches))
            return None

        started = next((e.get("ts") for e in entries
                        if e.get("type") == "run_start"),
                       speeches[0].get("ts", time.time()))
        date_str = datetime.fromtimestamp(started).strftime("%Y年%m月%d日")
        transcript = self._format_transcript(entries)

        # 1) チャンクごとに材料を抽出
        chunks = self._split(transcript, _CHUNK_CHARS)
        materials: list[str] = []
        for ch in chunks:
            raw = await MemoryConsolidator._generate_raw(
                llm, _CHUNK_SYSTEM,
                [ChatMessage(role="user",
                             content=f"次は配信の会話記録の一部です。\n\n{ch}")])
            if raw.strip():
                materials.append(raw.strip())
        if not materials:
            logger.warning("StreamArchiver: no materials extracted",
                           path=str(path))
            return None

        # 2) エピソードへ統合
        topic_lines = ""
        if topics:
            labels = [t.get("label") or t.get("title") or t.get("query") or ""
                      for t in topics]
            labels = [x for x in labels if x]
            if labels:
                topic_lines = ("\n\n## 話題プランナーの記録（話した話題）\n"
                               + "\n".join(f"- {x}" for x in labels[-20:]))
        episode_raw = await MemoryConsolidator._generate_raw(
            llm, _EPISODE_SYSTEM,
            [ChatMessage(role="user", content=(
                f"配信日: {date_str}\n\n## 要約材料\n"
                + "\n".join(materials) + topic_lines))])
        episode = episode_raw.strip()
        if not episode:
            logger.warning("StreamArchiver: episode generation empty",
                           path=str(path))
            return None

        # 3) アーカイブ（要約＋全発話）を残す
        md_path = path.with_suffix(".episode.md")
        try:
            md_path.write_text(
                f"# {date_str}の配信エピソード\n\n{episode}\n\n"
                f"---\n\n## 全発話記録\n\n{transcript}\n",
                encoding="utf-8")
        except OSError as e:
            logger.warning("StreamArchiver: episode md write failed",
                           error=str(e))

        # 4) 各キャラの MemoryStore へ投入
        lines = [l.strip() for l in episode.splitlines() if l.strip()]
        # 1行目の「話題：」行はキーワード保持用。MemoryStore に専用
        # フィールドが無いため、要約エントリの文頭に含めて残す
        topic_line = ""
        if lines and lines[0].startswith("話題："):
            topic_line = lines.pop(0)
        summary = lines[0] if lines else ""
        bullets = [l[2:].strip() for l in lines[1:] if l.startswith("- ")]
        mem_texts = []
        if summary:
            mem_texts.append(f"{topic_line} {summary}" if topic_line else summary)
        mem_texts += [f"（{date_str}の配信）{b}" for b in bullets]
        added_total = 0
        for store in stores.values():
            if store is None:
                continue
            for text in mem_texts:
                if store.add(text, source="stream_episode",
                             importance=1.6) is not None:
                    added_total += 1
        logger.info("StreamArchiver: episode stored",
                    speeches=len(speeches), added=added_total,
                    summary=summary[:50])
        return {"summary": summary, "added": added_total,
                "speeches": len(speeches), "episode": episode}

    # ── 整形 ────────────────────────────────────────────────

    @staticmethod
    def _format_transcript(entries: list[dict]) -> str:
        lines: list[str] = []
        for e in entries:
            ts = e.get("ts") or 0
            hm = datetime.fromtimestamp(ts).strftime("%H:%M") if ts else "--:--"
            if e.get("type") == "speech":
                lines.append(f"[{hm}] {e.get('speaker', '?')}: {e.get('text', '')}")
            elif e.get("type") == "comment":
                lines.append(f"[{hm}] 視聴者({e.get('author', '')}): {e.get('text', '')}")
        return "\n".join(lines)

    @staticmethod
    def _split(text: str, size: int) -> list[str]:
        if len(text) <= size:
            return [text]
        chunks, cur = [], ""
        for line in text.splitlines():
            if len(cur) + len(line) + 1 > size and cur:
                chunks.append(cur)
                cur = ""
            cur += line + "\n"
        if cur:
            chunks.append(cur)
        return chunks
