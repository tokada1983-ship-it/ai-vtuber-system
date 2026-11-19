"""ElythEpisodeImporter: ELYTH日次エピソードを、対象キャラの MemoryStore へ取り込む

仕様: elyth-memory/SPEC.md（PC側取り込み）

VM側の cron が前日分の ELYTH 会話をまとめたエピソードファイル
`ELYTH-EP-YYYYMMDD.md` を Google Drive `Muse連携/elyth-episodes/` へ置く。
このクラスは、その Drive ローカルコピーのフォルダを走査し、未取り込みの
ファイルを対象キャラ（既定: Character1）の MemoryStore へ投入する。
Character2は ELYTH に参加していないため対象外（個体分離ルール）。

配信エピソード（StreamArchiver.build_episode の後段）と同方式:
  - 「## 記憶行（取り込み対象）」の `- ` 行を順に取り込む
  - 1行目（要約）はそのまま、2行目以降は「（{date}のELYTH）」を前置する
  - source="elyth_episode"、importance は frontmatter の値（既定 1.4）
  - provenance は mind_loop の _PROV_BY_MEMORY_SOURCE で
    "elyth_episode" → "experienced" に解決される

状態管理は PC ローカル側の台帳（ファイル名＋SHA-256 ハッシュ）で行い、
Drive 上の原本には一切手を付けない。取り込みに失敗したファイルは台帳に
載せないので、次回走査で自動的に再試行される（StreamArchiver の
pending 処理と同じ隔離方式）。内容が改訂されたファイル（ハッシュ不一致）
は再取り込みし、重複は MemoryStore.add の dedupe が吸収する。
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from datetime import datetime
from pathlib import Path

from backend.logger import get_logger

logger = get_logger(__name__)

SOURCE = "elyth_episode"
DEFAULT_IMPORTANCE = 1.4
_MEMORY_HEADING = "## 記憶行（取り込み対象）"
_FILE_RE = re.compile(r"^ELYTH-EP-\d{8}\.md$")


def parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """最小 frontmatter パーサ（`key: value` の単純な行だけを読む）。

    閉じの `---` が無いものは frontmatter として扱わず、本文全体を返す。
    """
    if not text.startswith("---"):
        return {}, text
    lines = text.splitlines()
    meta: dict[str, str] = {}
    for i in range(1, len(lines)):
        line = lines[i]
        if line.strip() == "---":
            return meta, "\n".join(lines[i + 1:])
        if ":" in line:
            key, _, value = line.partition(":")
            meta[key.strip().lower()] = value.strip()
    return {}, text


def extract_memory_lines(body: str) -> list[str]:
    """「## 記憶行（取り込み対象）」節の `- ` 行を順に取り出す"""
    out: list[str] = []
    in_section = False
    for line in body.splitlines():
        stripped = line.strip()
        if stripped.startswith("## "):
            in_section = stripped == _MEMORY_HEADING
            continue
        if stripped.startswith("# "):
            in_section = False
            continue
        if in_section and stripped.startswith("- "):
            text = stripped[2:].strip()
            if text:
                out.append(text)
    return out


def _date_prefix(meta: dict[str, str], path: Path) -> str:
    """箇条書き行の前置プレフィックス（配信エピソードの「（…の配信）」と同方式）"""
    dt = None
    raw = (meta.get("date") or "").strip()
    if raw:
        try:
            dt = datetime.strptime(raw[:10], "%Y-%m-%d")
        except ValueError:
            dt = None
    if dt is None:
        m = re.search(r"(\d{4})(\d{2})(\d{2})", path.name)
        if m:
            try:
                dt = datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            except ValueError:
                dt = None
    if dt is not None:
        return f"（{dt.year}年{dt.month}月{dt.day}日のELYTH）"
    return "（ELYTH）"


class ElythEpisodeImporter:
    """elyth-episodes フォルダの走査と取り込み（同期処理。失敗はファイル単位で隔離）"""

    def __init__(self, episodes_dir: str | Path, ledger_path: str | Path,
                 character: str = "Character1"):
        self._dir = Path(episodes_dir)
        self._ledger_path = Path(ledger_path)
        self._character = character

    # ── 台帳（PCローカル。Drive原本には触れない） ─────────────

    def _load_ledger(self) -> dict:
        try:
            data = json.loads(self._ledger_path.read_text(encoding="utf-8"))
            if isinstance(data, dict) and isinstance(data.get("files"), dict):
                return data
        except (OSError, ValueError):
            pass
        return {"files": {}}

    def _save_ledger(self, ledger: dict) -> None:
        self._ledger_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._ledger_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(ledger, ensure_ascii=False, indent=1),
                       encoding="utf-8")
        tmp.replace(self._ledger_path)

    # ── 走査・取り込み ────────────────────────────────────────

    def episode_files(self) -> list[Path]:
        if not self._dir.is_dir():
            return []
        return sorted(p for p in self._dir.glob("ELYTH-EP-*.md")
                      if _FILE_RE.match(p.name))

    def scan(self, store) -> dict:
        """未取り込みファイルをすべて取り込む。1ファイルの失敗は他を妨げない"""
        result = {"files": 0, "imported": 0, "draft": 0, "already": 0,
                  "skipped": 0, "failed": 0, "added": 0}
        if store is None:
            return result
        ledger = self._load_ledger()
        changed = False
        for path in self.episode_files():
            result["files"] += 1
            try:
                status, added = self._import_one(path, store, ledger)
            except Exception as e:
                # 台帳に載せない＝次回走査で再試行（pending と同じ隔離方式）
                logger.warning("ELYTH episode import failed",
                               path=str(path), error=str(e))
                result["failed"] += 1
                continue
            if status == "imported":
                result["imported"] += 1
                result["added"] += added
                changed = True
            elif status in result:
                result[status] += 1
        if changed:
            try:
                self._save_ledger(ledger)
            except OSError as e:
                logger.warning("ELYTH import ledger save failed", error=str(e))
        if result["imported"] or result["failed"]:
            logger.info("ELYTH episode scan done", **result)
        return result

    def _import_one(self, path: Path, store, ledger: dict) -> tuple[str, int]:
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        entry = ledger["files"].get(path.name)
        if entry and entry.get("sha256") == digest:
            return "already", 0

        meta, body = parse_frontmatter(raw.decode("utf-8"))
        # 下書き（status に DRAFT を含む）は取り込まない。正式版は status 行が無い。
        # 台帳にも載せないので、正式版に差し替われば次回走査で取り込まれる
        if "DRAFT" in (meta.get("status") or "").upper():
            return "draft", 0
        # 個体分離: 対象キャラ以外のエピソードは取り込まない
        char = (meta.get("character") or "").strip()
        if char and char != self._character:
            logger.warning("ELYTH episode character mismatch, skip",
                           path=str(path), character=char)
            return "skipped", 0

        lines = extract_memory_lines(body)
        if not lines:
            raise ValueError("no memory lines found")

        importance = DEFAULT_IMPORTANCE
        if meta.get("importance"):
            try:
                importance = float(meta["importance"])
            except ValueError:
                pass

        prefix = _date_prefix(meta, path)
        added = 0
        for i, line in enumerate(lines):
            mem_text = line if i == 0 else f"{prefix}{line}"
            if store.add(mem_text, source=SOURCE,
                         importance=importance) is not None:
                added += 1
        ledger["files"][path.name] = {
            "sha256": digest,
            "date": meta.get("date", ""),
            "imported_at": time.time(),
            "lines": len(lines),
            "added": added,
        }
        return "imported", added
