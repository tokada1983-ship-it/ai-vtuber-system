"""MemoryStore: エピソード記憶を構造化して保持する（airi の Memory 層に相当）

## なぜ分けるのか

従来は「配信中に自動で得た記憶」を人手編集用の memory.md に追記し続けていた。
その方式には長期運用で次の問題があった。

  - 追記だけで上限も剪定もないため、記憶ファイルが単調増加する
  - 独り言モードでは全文をプロンプトに入れるので、記憶が育つほど
    コンテキストを圧迫し、最終的にLLMが空応答を返すようになる
  - 重複検出が効かないと同じ内容が積み上がる
  - 人が書いた設定と自動生成された記憶が混ざり、手で直しづらい

そこで記憶を2層に分ける。

  コア記憶     : memory.md。人が書く設定。自動追記しない（LongTermMemory）
  エピソード記憶: このクラス。配信中に自動で貯まる。件数上限・忘却つき

## 忘却について

全部を覚え続けるのは現実的でないので、スコアの低いものから捨てる。

    score = importance × 時間減衰 × 利用実績

時間減衰は半減期モデル（既定30日）。「最近使われた」「よく参照される」記憶は
生き残り、一度きりで二度と引かれない記憶は自然に消える。
"""
from __future__ import annotations

import math
import re
import sqlite3
import time
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

from backend.logger import get_logger

logger = get_logger(__name__)

# 埋め込みモデルが無いときの重複判定。実測では
#   ほぼ同一        : 0.97
#   言い換え        : 0.67〜0.89
#   文型が同じ別事実: 0.82
# なので、0.90 なら別の事実を誤って1つにまとめることがない。
# 言い換えの取りこぼしは件数上限（compact）が受け持つ。
_LITERAL_DEDUPE_RATIO = 0.90

_SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    text         TEXT    NOT NULL,
    source       TEXT    NOT NULL DEFAULT 'auto',
    viewer_name  TEXT,
    created_at   REAL    NOT NULL,
    last_used_at REAL,
    use_count    INTEGER NOT NULL DEFAULT 0,
    importance   REAL    NOT NULL DEFAULT 1.0,
    pinned       INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_created ON memories(created_at);
"""


@dataclass
class MemoryEntry:
    id: int
    text: str
    source: str
    created_at: float
    last_used_at: float | None
    use_count: int
    importance: float
    pinned: bool = False
    viewer_name: str | None = None

    def score(self, now: float | None = None, half_life_days: float = 30.0) -> float:
        """忘却判定用スコア。大きいほど残す価値がある。"""
        now = now or time.time()
        ref = self.last_used_at or self.created_at
        age_days = max(0.0, (now - ref) / 86400.0)
        recency = 0.5 ** (age_days / half_life_days) if half_life_days > 0 else 1.0
        usage = 1.0 + math.log1p(self.use_count)
        return self.importance * recency * usage

    def to_dict(self, now: float | None = None) -> dict:
        return {
            "id": self.id,
            "text": self.text,
            "source": self.source,
            "viewer_name": self.viewer_name,
            "created_at": self.created_at,
            "last_used_at": self.last_used_at,
            "use_count": self.use_count,
            "importance": round(self.importance, 3),
            "pinned": self.pinned,
            "score": round(self.score(now), 4),
        }


class MemoryStore:
    """
    キャラクター1体ぶんのエピソード記憶。
    SQLiteファイル1つで完結し、検索は RAGRetriever に委譲する。
    """

    def __init__(
        self,
        db_path: str | Path,
        retriever=None,
        max_entries: int = 300,
        half_life_days: float = 30.0,
    ):
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._retriever = retriever
        self._max_entries = max_entries
        self._half_life = half_life_days
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._migrate_db()
        self._conn.commit()
        self._index_dirty = True
        self._indexed_ids: list[int] = []

    def _migrate_db(self) -> None:
        """既存DBにカラムが足りない場合の自動マイグレーション"""
        cols = [c["name"] for c in self._conn.execute("PRAGMA table_info(memories)").fetchall()]
        if "viewer_name" not in cols:
            try:
                self._conn.execute("ALTER TABLE memories ADD COLUMN viewer_name TEXT")
            except Exception as e:
                logger.warning("MemoryStore migration error", error=str(e))
        # 索引はカラムを足した後に張る（_SCHEMA側で張ると旧DBで起動できなくなる）
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_viewer ON memories(viewer_name)")

    # ── 基本操作 ──────────────────────────────────────────────

    def _row_to_entry(self, row: sqlite3.Row) -> MemoryEntry:
        keys = row.keys()
        return MemoryEntry(
            id=row["id"], text=row["text"], source=row["source"],
            created_at=row["created_at"], last_used_at=row["last_used_at"],
            use_count=row["use_count"], importance=row["importance"],
            pinned=bool(row["pinned"]),
            viewer_name=row["viewer_name"] if "viewer_name" in keys else None,
        )

    def all(self) -> list[MemoryEntry]:
        rows = self._conn.execute(
            "SELECT * FROM memories ORDER BY created_at DESC"
        ).fetchall()
        return [self._row_to_entry(r) for r in rows]

    def count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]

    def add(
        self,
        text: str,
        source: str = "auto",
        importance: float = 1.0,
        dedupe: bool = True,
        viewer_name: str | None = None,
    ) -> MemoryEntry | None:
        """
        記憶を1件追加する。既存とほぼ同じ内容なら追加せず None を返し、
        代わりに既存エントリの重要度を少し上げる（何度も話題に出る＝大事）。
        """
        text = text.strip()
        if not text:
            return None

        if dedupe:
            dup = self.find_duplicate(text)
            if dup is not None:
                self._conn.execute(
                    "UPDATE memories SET importance = MIN(importance + 0.2, 3.0),"
                    " last_used_at = ? WHERE id = ?",
                    (time.time(), dup.id),
                )
                self._conn.commit()
                logger.debug("MemoryStore: duplicate reinforced",
                             new=text[:40], existing=dup.text[:40])
                return None

        now = time.time()
        cur = self._conn.execute(
            "INSERT INTO memories (text, source, viewer_name, created_at, last_used_at,"
            " use_count, importance) VALUES (?, ?, ?, ?, NULL, 0, ?)",
            (text, source, viewer_name, now, importance),
        )
        self._conn.commit()
        self._index_dirty = True
        entry = MemoryEntry(id=cur.lastrowid, text=text, source=source,
                            viewer_name=viewer_name,
                            created_at=now, last_used_at=None,
                            use_count=0, importance=importance)
        return entry

    def by_viewer(self, viewer_name: str, limit: int = 5) -> list[MemoryEntry]:
        """特定の視聴者に紐づくエピソード記憶を取得"""
        if not viewer_name:
            return []
        rows = self._conn.execute(
            "SELECT * FROM memories WHERE viewer_name = ? ORDER BY created_at DESC LIMIT ?",
            (viewer_name, limit),
        ).fetchall()
        return [self._row_to_entry(r) for r in rows]

    def delete(self, entry_id: int) -> bool:
        cur = self._conn.execute("DELETE FROM memories WHERE id = ?", (entry_id,))
        self._conn.commit()
        if cur.rowcount:
            self._index_dirty = True
            return True
        return False

    def set_pinned(self, entry_id: int, pinned: bool) -> bool:
        """ピン留めした記憶は忘却の対象外にする"""
        cur = self._conn.execute(
            "UPDATE memories SET pinned = ? WHERE id = ?", (1 if pinned else 0, entry_id)
        )
        self._conn.commit()
        return bool(cur.rowcount)

    def set_importance(self, entry_id: int, importance: float) -> bool:
        """重要度を直接設定する（睡眠時統合の固定化・減衰加速用）"""
        cur = self._conn.execute(
            "UPDATE memories SET importance = ? WHERE id = ?",
            (max(0.0, min(3.0, float(importance))), entry_id),
        )
        self._conn.commit()
        return bool(cur.rowcount)

    def clear(self) -> int:
        n = self.count()
        self._conn.execute("DELETE FROM memories")
        self._conn.commit()
        self._index_dirty = True
        return n

    # ── 検索 ──────────────────────────────────────────────────

    def _ensure_index(self) -> list[int]:
        """RAG索引をエントリ本文で張り直す。返り値は索引順のID列。"""
        if self._retriever is None:
            return []
        if not self._index_dirty and self._indexed_ids:
            return self._indexed_ids
        rows = self._conn.execute(
            "SELECT id, text FROM memories ORDER BY id"
        ).fetchall()
        self._indexed_ids = [r["id"] for r in rows]
        self._retriever.index_texts([r["text"] for r in rows])
        self._index_dirty = False
        return self._indexed_ids

    def find_duplicate(self, text: str) -> MemoryEntry | None:
        """
        ほぼ同じ内容の既存エントリを返す（無ければ None）。

        判定方法は検索モードで変える。
          埋め込みあり: 意味の近さ（言い換えも重複として扱える）
          埋め込みなし: 文字列の一致度。TF-IDFのスコアは登録件数が少ないうちは
                        飽和してしまい、「タナカさんはゲームが好き」と
                        「タナカさんは猫が好き」すら同一と見なしてしまうため、
                        語彙ベクトルではなく素直に文字列を比べる。
        """
        if self._retriever is None or self.count() == 0:
            return None
        self._ensure_index()
        hits = self._retriever.search(text, top_k=3)
        if not hits:
            return None

        use_embeddings = self._retriever.uses_embeddings
        threshold = self._retriever.dedupe_threshold
        for hit in hits:
            if use_embeddings:
                is_dup = hit.score >= threshold
            else:
                is_dup = SequenceMatcher(None, text, hit.text).ratio() >= _LITERAL_DEDUPE_RATIO
            if not is_dup:
                continue
            # 索引順とID列は対応しているので source_line からIDを引く
            idx = hit.source_line
            if 0 <= idx < len(self._indexed_ids):
                row = self._conn.execute(
                    "SELECT * FROM memories WHERE id = ?", (self._indexed_ids[idx],)
                ).fetchone()
                if row:
                    return self._row_to_entry(row)
        return None

    def search(
        self,
        query: str,
        top_k: int = 3,
        touch: bool = True,
        min_score: float = 0.0,
    ) -> list[MemoryEntry]:
        """
        クエリに関連する記憶を返す。参照した記憶は使用実績を更新する。

        min_score は「無関係な記憶をプロンプトに混ぜない」ための下限。
        日本語埋め込みの類似度はベースラインが高め（無関係でも0.15〜0.43程度）
        なので、低すぎると毎回全件が返ってしまう。
        """
        if self._retriever is None or self.count() == 0 or not query.strip():
            return []
        self._ensure_index()
        hits = [h for h in self._retriever.search(query, top_k=top_k)
                if h.score >= min_score]
        out: list[MemoryEntry] = []
        for h in hits:
            idx = h.source_line
            if not (0 <= idx < len(self._indexed_ids)):
                continue
            row = self._conn.execute(
                "SELECT * FROM memories WHERE id = ?", (self._indexed_ids[idx],)
            ).fetchone()
            if row:
                out.append(self._row_to_entry(row))
        if touch and out:
            self.touch([e.id for e in out])
        return out

    def find_episodes_on(self, month: int, day: int,
                         limit: int = 5) -> list[MemoryEntry]:
        """指定した月日の配信エピソードを、本文中の日付ラベルで直接引く。

        配信エピソード（source='stream_episode'）の本文には
        「2026年11月14日の配信では、…」のように放送日が埋め込まれている。
        意味検索は日付指定の open な質問（「11月14日の配信で話した話題は？」）
        で当該エピソードを取りこぼすため、日付が明示された質問では
        この直接引きを優先する（Phase 4.5）。
        要約行（「…の配信では」を含む行）を先に、箇条書き行を後に返す。
        """
        rows = self._conn.execute(
            "SELECT * FROM memories WHERE source = 'stream_episode'"
            " AND text LIKE ? ORDER BY created_at",
            (f"%{day}日%",),
        ).fetchall()
        pat = re.compile(rf"(?<!\d){month}月{day}日(?!\d)")
        entries = [self._row_to_entry(r) for r in rows
                   if pat.search(r["text"] or "")]
        summaries = [e for e in entries if "の配信では" in e.text]
        bullets = [e for e in entries if "の配信では" not in e.text]
        out = (summaries + bullets)[:limit]
        if out:
            self.touch([e.id for e in out])
        return out

    def touch(self, ids: list[int]) -> None:
        """参照された記憶の使用実績を更新する（忘却されにくくなる）"""
        if not ids:
            return
        now = time.time()
        self._conn.executemany(
            "UPDATE memories SET use_count = use_count + 1, last_used_at = ? WHERE id = ?",
            [(now, i) for i in ids],
        )
        self._conn.commit()

    def sample_recent(self, limit: int = 8) -> list[MemoryEntry]:
        """独り言のお題用に、最近の記憶を少しだけ取り出す"""
        rows = self._conn.execute(
            "SELECT * FROM memories ORDER BY COALESCE(last_used_at, created_at) DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [self._row_to_entry(r) for r in rows]

    # ── 忘却 ──────────────────────────────────────────────────

    def compact(self) -> list[MemoryEntry]:
        """
        件数上限を超えていたら、スコアの低い記憶から削除する。
        ピン留め・手動追加分は残す。削除したエントリを返す。
        """
        entries = self.all()
        if len(entries) <= self._max_entries:
            return []

        now = time.time()
        droppable = [e for e in entries if not e.pinned and e.source != "manual"]
        keep_forced = len(entries) - len(droppable)
        allowed = max(0, self._max_entries - keep_forced)

        droppable.sort(key=lambda e: e.score(now, self._half_life), reverse=True)
        to_drop = droppable[allowed:]
        if not to_drop:
            return []

        self._conn.executemany(
            "DELETE FROM memories WHERE id = ?", [(e.id,) for e in to_drop]
        )
        self._conn.commit()
        self._index_dirty = True
        logger.info("MemoryStore: forgot low-score memories",
                    dropped=len(to_drop), remaining=self.count())
        return to_drop

    # ── プロンプト生成・統計 ──────────────────────────────────

    def build_prompt(
        self, query: str, top_k: int = 3, max_chars: int = 400, min_score: float = 0.0
    ) -> str:
        """コメントに関連する記憶をプロンプト断片にする"""
        entries = self.search(query, top_k=top_k, min_score=min_score)
        if not entries:
            return ""
        lines = ["## 関連する記憶"]
        for e in entries:
            lines.append(f"- {e.text}")
        text = "\n".join(lines)
        return text[:max_chars]

    def stats(self) -> dict:
        now = time.time()
        entries = self.all()
        return {
            "count": len(entries),
            "max_entries": self._max_entries,
            "half_life_days": self._half_life,
            "pinned": sum(1 for e in entries if e.pinned),
            "path": str(self._path),
            "oldest": min((e.created_at for e in entries), default=None),
            "newest": max((e.created_at for e in entries), default=None),
            "avg_score": (round(sum(e.score(now, self._half_life) for e in entries)
                                / len(entries), 3) if entries else 0.0),
        }

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass


# 旧方式で memory.md に自動追記されていたセクション
_LEGACY_SECTION = "## 配信の記憶・印象"


def migrate_legacy_section(long_term_memory, store: "MemoryStore") -> int:
    """
    memory.md の「## 配信の記憶・印象」を MemoryStore に取り込み、MDからは取り除く。

    旧方式では自動生成された記憶が人手編集用のMDに混ざって溜まり続けていた。
    初回起動時に一度だけ引っ越して、以降 MD は人が書く設定だけにする。

    Returns: 取り込んだ件数
    """
    content = long_term_memory.get_raw()
    if _LEGACY_SECTION not in content:
        return 0

    head, _, tail = content.partition(_LEGACY_SECTION)
    imported = 0
    for line in tail.splitlines():
        line = line.strip()
        if line.startswith("<!--") or not line:
            continue
        if line.startswith("#"):
            break  # 別の見出しに入ったら終了（そこから先はMDに残す）
        if line.startswith(("-", "*")):
            text = line.lstrip("-* ").strip()
            # dedupe=False にしているのは、ここで重複判定を走らせると
            # 起動時に埋め込みモデルのロードが始まって初動が数秒遅くなるため。
            # 元ファイルの内容をそのまま引き継ぐだけなのでこれでよい。
            if text and store.add(text, source="imported", dedupe=False) is not None:
                imported += 1

    if imported:
        long_term_memory.save(head.rstrip() + "\n")
        logger.info("Migrated legacy memory section into MemoryStore",
                    imported=imported, character=long_term_memory.character_name)
    return imported
