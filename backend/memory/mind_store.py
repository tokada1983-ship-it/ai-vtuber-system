"""MindStore: 「個」の置き場所（意見・意図・選好・自己状態・自己物語・思考ジャーナル）

設計: AUTONOMOUS_MIND_DESIGN_V2.md §2 / §5 / §6

MemoryStore（エピソード記憶）と同じ memories.db に同居する新テーブル群を扱う。
DBファイルはキャラごとに分かれている（characters/<名前>/memories.db）ため、
このクラスもキャラ1体ぶんで、Character1とCharacter2の「個」はファイル単位で分離される。

原則:
  - 全レコードに provenance（出所タグ）を必須とする。経験したことと
    自分で考えたこと・夢に見たことを同じ重さで扱わないための仕組み。
  - 意見は上書きせず、改訂したら旧版を superseded として残す（考えが
    変わった履歴そのものが個になる）。
  - 育つ信条（L0b）は self_state の "beliefs" キーに保持する。変更経路は
    set_beliefs() のみとし、thought_journal への記録を必須にする（§6）。
"""
from __future__ import annotations

import json
import re
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

from backend.logger import get_logger

logger = get_logger(__name__)

PROVENANCE_VALUES = ("experienced", "told", "researched", "inferred", "dreamed")
# 意見の confidence のハードキャップ（設計 §2.4/§9: 同じ主題の再思考だけで
# 確信度が単調に上がる自己強化を、保存層でも構造的に防ぐ）。
OPINION_CONFIDENCE_CAP = 0.9
STANCE_VALUES = ("賛成", "反対", "両面", "未定")
INTENTION_STATES = ("active", "paused", "done", "dropped", "superseded")
INTENTION_ORIGINS = ("self", "master", "conversation")
JOURNAL_KINDS = ("reflection", "opinion", "intention", "curiosity",
                 "contradiction", "dream", "sleep", "narrative", "belief",
                 "belief_proposal", "archive")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS opinions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    topic_key     TEXT    NOT NULL,
    topic_label   TEXT    NOT NULL,
    stance        TEXT    NOT NULL,
    reasons       TEXT    NOT NULL DEFAULT '',
    confidence    REAL    NOT NULL DEFAULT 0.5,
    provenance    TEXT    NOT NULL,
    formed_at     REAL    NOT NULL,
    updated_at    REAL    NOT NULL,
    revision      INTEGER NOT NULL DEFAULT 1,
    status        TEXT    NOT NULL DEFAULT 'active',
    supersedes_id INTEGER
);
CREATE INDEX IF NOT EXISTS idx_opinions_topic ON opinions(topic_key, status);
CREATE TABLE IF NOT EXISTS intentions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    title       TEXT    NOT NULL,
    detail      TEXT    NOT NULL DEFAULT '',
    origin      TEXT    NOT NULL DEFAULT 'self',
    state       TEXT    NOT NULL DEFAULT 'active',
    next_action TEXT    NOT NULL DEFAULT '',
    progress    TEXT    NOT NULL DEFAULT '',
    salience    REAL    NOT NULL DEFAULT 0.5,
    provenance  TEXT    NOT NULL,
    created_at  REAL    NOT NULL,
    updated_at  REAL    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_intentions_state ON intentions(state);
CREATE TABLE IF NOT EXISTS preferences (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    item           TEXT    NOT NULL,
    polarity       TEXT    NOT NULL,
    strength       REAL    NOT NULL DEFAULT 0.5,
    evidence_count INTEGER NOT NULL DEFAULT 1,
    provenance     TEXT    NOT NULL,
    first_seen     REAL    NOT NULL,
    last_seen      REAL    NOT NULL
);
CREATE TABLE IF NOT EXISTS self_state (
    key        TEXT PRIMARY KEY,
    value      TEXT NOT NULL,
    provenance TEXT NOT NULL,
    updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS narrative_versions (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    version    INTEGER NOT NULL,
    text       TEXT    NOT NULL,
    basis      TEXT    NOT NULL DEFAULT '',
    approved   TEXT    NOT NULL DEFAULT 'auto',
    provenance TEXT    NOT NULL,
    created_at REAL    NOT NULL
);
CREATE TABLE IF NOT EXISTS thought_journal (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ts              REAL    NOT NULL,
    kind            TEXT    NOT NULL,
    input_refs      TEXT    NOT NULL DEFAULT '',
    output_text     TEXT    NOT NULL DEFAULT '',
    applied_changes TEXT    NOT NULL DEFAULT '',
    provenance      TEXT    NOT NULL
);
CREATE TABLE IF NOT EXISTS thought_journal_archive (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    orig_id         INTEGER,
    ts              REAL    NOT NULL,
    kind            TEXT    NOT NULL,
    input_refs      TEXT    NOT NULL DEFAULT '',
    output_text     TEXT    NOT NULL DEFAULT '',
    applied_changes TEXT    NOT NULL DEFAULT '',
    provenance      TEXT    NOT NULL,
    archived_at     REAL    NOT NULL,
    batch           TEXT    NOT NULL DEFAULT ''
);
"""


def _check_provenance(provenance: str) -> str:
    if provenance not in PROVENANCE_VALUES:
        raise ValueError(f"invalid provenance: {provenance!r} "
                         f"(must be one of {PROVENANCE_VALUES})")
    return provenance


def _normalize_topic_key(topic_label: str) -> str:
    return "".join(topic_label.split()).lower()


@dataclass
class Opinion:
    id: int
    topic_key: str
    topic_label: str
    stance: str
    reasons: str
    confidence: float
    provenance: str
    formed_at: float
    updated_at: float
    revision: int
    status: str
    supersedes_id: int | None = None

    def summary_line(self) -> str:
        line = f"{self.topic_label}: {self.stance}"
        if self.reasons:
            line += f"（{self.reasons}）"
        return line


@dataclass
class Intention:
    id: int
    title: str
    detail: str
    origin: str
    state: str
    next_action: str
    progress: str
    salience: float
    provenance: str
    created_at: float
    updated_at: float


@dataclass
class Preference:
    id: int
    item: str
    polarity: str
    strength: float
    evidence_count: int
    provenance: str
    first_seen: float
    last_seen: float


@dataclass
class Narrative:
    id: int
    version: int
    text: str
    basis: str
    approved: str
    provenance: str
    created_at: float


@dataclass
class MindSnapshot:
    """プロンプトの L0b〜L2 に載せる「自己」のスナップショット。

    発話のたびにDBを引かず、エージェント側で保持して発話間にだけ差し替える
    （生成途中で自己が変わると KV プレフィックスも崩れるため）。
    """
    beliefs: list[str] = field(default_factory=list)
    narrative: str = ""
    mood: str = ""
    energy: float | None = None
    interests: list[str] = field(default_factory=list)
    focus_topic: str = ""
    intentions: list[str] = field(default_factory=list)
    opinions: list[str] = field(default_factory=list)
    # 焦点ローテーション用: 直近に「主焦点」として提示した項目のキー
    # （古い順。提示のたびに presentation_focus() が積む。Phase 4.6）
    focus_history: list[str] = field(default_factory=list)

    def has_l2(self) -> bool:
        return bool(self.mood or self.interests or self.focus_topic
                    or self.intentions or self.opinions)

    # ── 焦点ローテーション（Phase 4.6）──────────────────────
    # 発話生成のたびに常に最有力の意図・意見だけが焦点として提示され
    # 続けると、モデルがその自己像（例:「調査が得意」）に張り付く。
    # Phase 4.5 でプランナー側の種クールダウンは入れたが、掛け合い発話は
    # プランナーの話題ではなく相方発話＋このスナップショットから生成
    # されるため、提示側にもローテーションが要る。DB の中身（意見・
    # 意図そのものや confidence）は一切変えず、提示順と「気になって
    # いること」スロットの選択だけを回す（Phase 2.5 の自己強化対策と
    # 矛盾しないよう、提示履歴は confidence に影響させない）。

    _FOCUS_HISTORY_MAX = 12   # 提示履歴の保持件数
    _FOCUS_RECENT_KEYS = 6    # 直近この件数のキーに入る項目を避ける/下げる

    def presentation_focus(self) -> tuple[str, list[str], list[str]]:
        """L2 提示用の (気になっていること, 意図, 意見) を返す。

        提示した主項目（スロットの内容・意図の先頭・意見の先頭）を
        focus_history に記録し、次回以降の提示で後ろへ回す。
        """
        recent = set(self.focus_history[-self._FOCUS_RECENT_KEYS:])

        def key(text: str) -> str:
            return _normalize_topic_key(text or "")

        # 「気になっていること」スロット: focus_topic 本体を第一候補に、
        # 関心・意図も候補として、直近に主焦点にしたものを避けて選ぶ。
        # focus_topic が空なら従来どおりスロット自体を出さない。
        focus = ""
        if self.focus_topic:
            focus = self.focus_topic
            candidates: list[str] = []
            for cand in [self.focus_topic, *self.interests, *self.intentions]:
                cand = (cand or "").strip()
                if cand and key(cand) not in {key(c) for c in candidates}:
                    candidates.append(cand)
            for cand in candidates:
                if key(cand) not in recent:
                    focus = cand
                    break

        def demote(items: list[str], keyfn) -> list[str]:
            # 直近に主焦点として提示した項目を後ろへ（同群内は元の順序）
            return sorted(items, key=lambda it: 1 if keyfn(it) in recent else 0)

        intentions = demote(list(self.intentions), key)

        def opinion_key(line: str) -> str:
            head = re.split(r"[:：]", line or "", maxsplit=1)[0]
            return key(head)

        opinions = demote(list(self.opinions), opinion_key)

        presented: list[str] = []
        if focus:
            presented.append(key(focus))
        if intentions:
            presented.append(key(intentions[0]))
        if opinions:
            presented.append(opinion_key(opinions[0]))
        for k in presented:
            if k in self.focus_history:
                self.focus_history.remove(k)
            self.focus_history.append(k)
        del self.focus_history[:-self._FOCUS_HISTORY_MAX]
        return focus, intentions, opinions


class MindStore:
    """キャラクター1体ぶんの「個」のストア。memories.db に同居する。"""

    def __init__(self, db_path: str | Path):
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        # MemoryStore と同じファイルを開くため、ロック競合を避ける設定にする
        self._conn.execute("PRAGMA busy_timeout = 5000")
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # ── 意見 ──────────────────────────────────────────────────

    def add_opinion(
        self,
        topic_label: str,
        stance: str,
        reasons: str = "",
        confidence: float = 0.5,
        provenance: str = "inferred",
        topic_key: str | None = None,
    ) -> Opinion:
        """意見を登録する。同じ主題の active な意見があれば改訂として扱い、
        旧版を superseded にして履歴を残す。"""
        _check_provenance(provenance)
        if stance not in STANCE_VALUES:
            raise ValueError(f"invalid stance: {stance!r}")
        topic_label = topic_label.strip()
        if not topic_label:
            raise ValueError("topic_label is empty")
        key = topic_key or _normalize_topic_key(topic_label)
        confidence = max(0.0, min(OPINION_CONFIDENCE_CAP, float(confidence)))
        now = time.time()

        old = self.get_opinion(key)
        revision = 1
        supersedes_id = None
        formed_at = now
        if old is not None:
            self._conn.execute(
                "UPDATE opinions SET status = 'superseded', updated_at = ? WHERE id = ?",
                (now, old.id),
            )
            revision = old.revision + 1
            supersedes_id = old.id
            formed_at = old.formed_at
        cur = self._conn.execute(
            "INSERT INTO opinions (topic_key, topic_label, stance, reasons,"
            " confidence, provenance, formed_at, updated_at, revision, status,"
            " supersedes_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'active', ?)",
            (key, topic_label, stance, reasons.strip(), confidence, provenance,
             formed_at, now, revision, supersedes_id),
        )
        self._conn.commit()
        return Opinion(
            id=cur.lastrowid, topic_key=key, topic_label=topic_label,
            stance=stance, reasons=reasons.strip(), confidence=confidence,
            provenance=provenance, formed_at=formed_at, updated_at=now,
            revision=revision, status="active", supersedes_id=supersedes_id,
        )

    def _row_to_opinion(self, row: sqlite3.Row) -> Opinion:
        return Opinion(
            id=row["id"], topic_key=row["topic_key"],
            topic_label=row["topic_label"], stance=row["stance"],
            reasons=row["reasons"], confidence=row["confidence"],
            provenance=row["provenance"], formed_at=row["formed_at"],
            updated_at=row["updated_at"], revision=row["revision"],
            status=row["status"], supersedes_id=row["supersedes_id"],
        )

    def get_opinion(self, topic_key: str) -> Opinion | None:
        row = self._conn.execute(
            "SELECT * FROM opinions WHERE topic_key = ? AND status = 'active'",
            (topic_key,),
        ).fetchone()
        return self._row_to_opinion(row) if row else None

    def list_opinions(self, status: str = "active", limit: int = 20) -> list[Opinion]:
        rows = self._conn.execute(
            "SELECT * FROM opinions WHERE status = ?"
            " ORDER BY updated_at DESC LIMIT ?",
            (status, limit),
        ).fetchall()
        return [self._row_to_opinion(r) for r in rows]

    def search_opinions(self, query: str, limit: int = 5) -> list[Opinion]:
        """主題・理由への部分一致で active な意見を探す（Phase 1 の最小検索）"""
        q = f"%{query.strip()}%"
        rows = self._conn.execute(
            "SELECT * FROM opinions WHERE status = 'active'"
            " AND (topic_label LIKE ? OR reasons LIKE ?)"
            " ORDER BY confidence DESC, updated_at DESC LIMIT ?",
            (q, q, limit),
        ).fetchall()
        return [self._row_to_opinion(r) for r in rows]

    # ── 意図 ──────────────────────────────────────────────────

    def get_intention_by_title(self, title: str) -> Intention | None:
        """タイトル完全一致（空白・大小を正規化）の active な意図を返す"""
        key = _normalize_topic_key(title or "")
        if not key:
            return None
        for it in self.list_intentions(limit=50):
            if _normalize_topic_key(it.title) == key:
                return it
        return None

    def add_intention(
        self,
        title: str,
        detail: str = "",
        origin: str = "self",
        next_action: str = "",
        salience: float = 0.5,
        provenance: str = "inferred",
    ) -> Intention:
        _check_provenance(provenance)
        if origin not in INTENTION_ORIGINS:
            raise ValueError(f"invalid origin: {origin!r}")
        title = title.strip()
        if not title:
            raise ValueError("title is empty")
        # 意図の重複防止（Phase 4.5）: 同一タイトルの active な意図が既に
        # あれば新規作成せず既存を返す（意見の topic_key 改訂と同じ考え方。
        # 意図には意見のようなマージ規律が無く、同一タイトルが4件溜まった
        # 実測への対処）。next_action が空なら新しい方で補う。
        existing = self.get_intention_by_title(title)
        if existing is not None:
            if not existing.next_action and next_action.strip():
                self.update_intention(existing.id,
                                      next_action=next_action.strip())
                existing.next_action = next_action.strip()
            return existing
        now = time.time()
        cur = self._conn.execute(
            "INSERT INTO intentions (title, detail, origin, state, next_action,"
            " progress, salience, provenance, created_at, updated_at)"
            " VALUES (?, ?, ?, 'active', ?, '', ?, ?, ?, ?)",
            (title, detail.strip(), origin, next_action.strip(),
             max(0.0, min(1.0, float(salience))), provenance, now, now),
        )
        self._conn.commit()
        return Intention(
            id=cur.lastrowid, title=title, detail=detail.strip(), origin=origin,
            state="active", next_action=next_action.strip(), progress="",
            salience=float(salience), provenance=provenance,
            created_at=now, updated_at=now,
        )

    def _row_to_intention(self, row: sqlite3.Row) -> Intention:
        return Intention(
            id=row["id"], title=row["title"], detail=row["detail"],
            origin=row["origin"], state=row["state"],
            next_action=row["next_action"], progress=row["progress"],
            salience=row["salience"], provenance=row["provenance"],
            created_at=row["created_at"], updated_at=row["updated_at"],
        )

    def list_intentions(self, state: str = "active", limit: int = 10) -> list[Intention]:
        rows = self._conn.execute(
            "SELECT * FROM intentions WHERE state = ?"
            " ORDER BY salience DESC, updated_at DESC LIMIT ?",
            (state, limit),
        ).fetchall()
        return [self._row_to_intention(r) for r in rows]

    def update_intention(
        self,
        intention_id: int,
        *,
        state: str | None = None,
        next_action: str | None = None,
        append_progress: str | None = None,
        salience: float | None = None,
    ) -> bool:
        row = self._conn.execute(
            "SELECT * FROM intentions WHERE id = ?", (intention_id,)
        ).fetchone()
        if row is None:
            return False
        if state is not None and state not in INTENTION_STATES:
            raise ValueError(f"invalid state: {state!r}")
        progress = row["progress"]
        if append_progress:
            stamp = time.strftime("%Y-%m-%d", time.localtime())
            entry = f"[{stamp}] {append_progress.strip()}"
            progress = f"{progress}\n{entry}" if progress else entry
        self._conn.execute(
            "UPDATE intentions SET state = ?, next_action = ?, progress = ?,"
            " salience = ?, updated_at = ? WHERE id = ?",
            (
                state if state is not None else row["state"],
                next_action if next_action is not None else row["next_action"],
                progress,
                salience if salience is not None else row["salience"],
                time.time(),
                intention_id,
            ),
        )
        self._conn.commit()
        return True

    # ── 選好 ──────────────────────────────────────────────────

    def add_preference(
        self,
        item: str,
        polarity: str,
        provenance: str = "experienced",
        strength: float = 0.5,
    ) -> Preference:
        """選好を記録する。同じ item＋polarity があれば裏付け回数を増やす
        （1回の出来事で好き嫌いを決めないための evidence_count）。"""
        _check_provenance(provenance)
        if polarity not in ("好き", "嫌い"):
            raise ValueError(f"invalid polarity: {polarity!r}")
        item = item.strip()
        if not item:
            raise ValueError("item is empty")
        now = time.time()
        row = self._conn.execute(
            "SELECT * FROM preferences WHERE item = ? AND polarity = ?",
            (item, polarity),
        ).fetchone()
        if row is not None:
            self._conn.execute(
                "UPDATE preferences SET evidence_count = evidence_count + 1,"
                " strength = MIN(1.0, strength + 0.1), last_seen = ? WHERE id = ?",
                (now, row["id"]),
            )
            self._conn.commit()
            row = self._conn.execute(
                "SELECT * FROM preferences WHERE id = ?", (row["id"],)
            ).fetchone()
        else:
            cur = self._conn.execute(
                "INSERT INTO preferences (item, polarity, strength,"
                " evidence_count, provenance, first_seen, last_seen)"
                " VALUES (?, ?, ?, 1, ?, ?, ?)",
                (item, polarity, max(0.0, min(1.0, float(strength))),
                 provenance, now, now),
            )
            self._conn.commit()
            row = self._conn.execute(
                "SELECT * FROM preferences WHERE id = ?", (cur.lastrowid,)
            ).fetchone()
        return Preference(
            id=row["id"], item=row["item"], polarity=row["polarity"],
            strength=row["strength"], evidence_count=row["evidence_count"],
            provenance=row["provenance"], first_seen=row["first_seen"],
            last_seen=row["last_seen"],
        )

    def adjust_preference(self, item: str, polarity: str,
                          delta: float) -> bool:
        """選好の強さを加減する（Phase 2: 矛盾点検の解決用）。
        該当レコードが無ければ False。強さは 0.1〜1.0 に丸める。"""
        row = self._conn.execute(
            "SELECT * FROM preferences WHERE item = ? AND polarity = ?",
            (item, polarity),
        ).fetchone()
        if row is None:
            return False
        strength = max(0.1, min(1.0, float(row["strength"]) + float(delta)))
        self._conn.execute(
            "UPDATE preferences SET strength = ?, last_seen = ? WHERE id = ?",
            (strength, time.time(), row["id"]),
        )
        self._conn.commit()
        return True

    def list_preferences(self, min_evidence: int = 1, limit: int = 20) -> list[Preference]:
        rows = self._conn.execute(
            "SELECT * FROM preferences WHERE evidence_count >= ?"
            " ORDER BY evidence_count DESC, strength DESC LIMIT ?",
            (min_evidence, limit),
        ).fetchall()
        return [
            Preference(
                id=r["id"], item=r["item"], polarity=r["polarity"],
                strength=r["strength"], evidence_count=r["evidence_count"],
                provenance=r["provenance"], first_seen=r["first_seen"],
                last_seen=r["last_seen"],
            )
            for r in rows
        ]

    # ── 自己状態（self_state）────────────────────────────────

    def set_state(self, key: str, value, provenance: str = "experienced") -> None:
        _check_provenance(provenance)
        self._conn.execute(
            "INSERT INTO self_state (key, value, provenance, updated_at)"
            " VALUES (?, ?, ?, ?)"
            " ON CONFLICT(key) DO UPDATE SET value = excluded.value,"
            " provenance = excluded.provenance, updated_at = excluded.updated_at",
            (key, json.dumps(value, ensure_ascii=False), provenance, time.time()),
        )
        self._conn.commit()

    def get_state(self, key: str, default=None):
        row = self._conn.execute(
            "SELECT value FROM self_state WHERE key = ?", (key,)
        ).fetchone()
        if row is None:
            return default
        try:
            return json.loads(row["value"])
        except (ValueError, TypeError):
            return row["value"]

    def all_state(self) -> dict:
        return {
            r["key"]: self.get_state(r["key"])
            for r in self._conn.execute("SELECT key FROM self_state").fetchall()
        }

    def set_beliefs(self, beliefs: list[str], reason: str,
                    provenance: str = "inferred") -> None:
        """育つ信条（L0b）の差し替え。§6 の歯止めとして、変更理由と
        旧信条を thought_journal に必ず残す（版管理の最小形）。"""
        old = self.get_state("beliefs", []) or []
        self.set_state("beliefs", list(beliefs), provenance)
        self.add_journal(
            "belief",
            output_text=reason,
            applied_changes={"old": old, "new": list(beliefs)},
            provenance=provenance,
        )
        logger.info("MindStore: beliefs updated", count=len(beliefs))

    # ── 自己物語（narrative_versions）────────────────────────

    def add_narrative(self, text: str, basis: str = "", approved: str = "auto",
                      provenance: str = "experienced") -> Narrative:
        _check_provenance(provenance)
        text = text.strip()
        if not text:
            raise ValueError("narrative text is empty")
        row = self._conn.execute(
            "SELECT MAX(version) AS v FROM narrative_versions"
        ).fetchone()
        version = (row["v"] or 0) + 1
        now = time.time()
        cur = self._conn.execute(
            "INSERT INTO narrative_versions (version, text, basis, approved,"
            " provenance, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (version, text, basis, approved, provenance, now),
        )
        self._conn.commit()
        logger.info("MindStore: narrative saved", version=version,
                    chars=len(text))
        return Narrative(id=cur.lastrowid, version=version, text=text,
                         basis=basis, approved=approved,
                         provenance=provenance, created_at=now)

    def latest_narrative(self) -> Narrative | None:
        row = self._conn.execute(
            "SELECT * FROM narrative_versions ORDER BY version DESC LIMIT 1"
        ).fetchone()
        if row is None:
            return None
        return Narrative(id=row["id"], version=row["version"], text=row["text"],
                         basis=row["basis"], approved=row["approved"],
                         provenance=row["provenance"],
                         created_at=row["created_at"])

    def list_narratives(self) -> list[Narrative]:
        rows = self._conn.execute(
            "SELECT * FROM narrative_versions ORDER BY version DESC"
        ).fetchall()
        return [
            Narrative(id=r["id"], version=r["version"], text=r["text"],
                      basis=r["basis"], approved=r["approved"],
                      provenance=r["provenance"], created_at=r["created_at"])
            for r in rows
        ]

    # ── 思考ジャーナル ────────────────────────────────────────

    def add_journal(self, kind: str, output_text: str, input_refs: str = "",
                    applied_changes: dict | None = None,
                    provenance: str = "inferred") -> int:
        _check_provenance(provenance)
        if kind not in JOURNAL_KINDS:
            raise ValueError(f"invalid journal kind: {kind!r}")
        cur = self._conn.execute(
            "INSERT INTO thought_journal (ts, kind, input_refs, output_text,"
            " applied_changes, provenance) VALUES (?, ?, ?, ?, ?, ?)",
            (time.time(), kind, input_refs, output_text,
             json.dumps(applied_changes or {}, ensure_ascii=False), provenance),
        )
        self._conn.commit()
        return cur.lastrowid

    def list_journal(self, limit: int = 20) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM thought_journal ORDER BY ts DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

    def archive_journal(self, older_than_days: int = 30,
                        now: float | None = None) -> dict:
        """古い思考ジャーナルを月次サマリへ圧縮して退避する（世代管理）。

        情報を消すのではなく、原文は thought_journal_archive へ移し、
        本体には（月×kind）ごとのサマリを1件ずつ残す。サマリ行の ts は
        圧縮した時刻になるため、再実行しても二重に圧縮されない（冪等）。
        """
        now = now or time.time()
        cutoff = now - older_than_days * 86400
        rows = self._conn.execute(
            "SELECT * FROM thought_journal WHERE ts < ? ORDER BY ts",
            (cutoff,),
        ).fetchall()
        if not rows:
            return {"archived": 0, "summaries": 0, "cutoff": cutoff}
        batch = time.strftime("%Y%m%d%H%M%S", time.gmtime(now))
        jst = timezone(timedelta(hours=9))
        groups: dict[tuple[str, str], list] = {}
        for r in rows:
            month = datetime.fromtimestamp(r["ts"], jst).strftime("%Y-%m")
            groups.setdefault((month, r["kind"]), []).append(r)
        summaries = 0
        for (month, kind), grows in groups.items():
            for r in grows:
                self._conn.execute(
                    "INSERT INTO thought_journal_archive (orig_id, ts, kind,"
                    " input_refs, output_text, applied_changes, provenance,"
                    " archived_at, batch) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (r["id"], r["ts"], r["kind"], r["input_refs"],
                     r["output_text"], r["applied_changes"], r["provenance"],
                     now, batch),
                )
            self._conn.executemany(
                "DELETE FROM thought_journal WHERE id = ?",
                [(r["id"],) for r in grows],
            )
            snippets = " / ".join(
                (r["output_text"] or "")[:60] for r in grows[:3])
            self._conn.execute(
                "INSERT INTO thought_journal (ts, kind, input_refs,"
                " output_text, applied_changes, provenance)"
                " VALUES (?, 'archive', ?, ?, ?, 'inferred')",
                (now, f"archive:{month}:{kind}",
                 f"{month}の{kind}記録 {len(grows)}件（アーカイブ済み）: "
                 f"{snippets}",
                 json.dumps({"archived_count": len(grows), "month": month,
                             "original_kind": kind,
                             "orig_ids": [r["id"] for r in grows]},
                            ensure_ascii=False)),
            )
            summaries += 1
        self._conn.commit()
        logger.info("MindStore: journal archived", archived=len(rows),
                    summaries=summaries)
        return {"archived": len(rows), "summaries": summaries,
                "cutoff": cutoff}

    # ── スナップショット・統計 ────────────────────────────────

    def build_snapshot(self) -> MindSnapshot:
        """プロンプトの L0b〜L2 に載せる自己のスナップショットを作る"""
        narrative = self.latest_narrative()
        energy = self.get_state("energy")
        return MindSnapshot(
            beliefs=list(self.get_state("beliefs", []) or []),
            narrative=narrative.text if narrative else "",
            mood=str(self.get_state("mood", "") or ""),
            energy=float(energy) if isinstance(energy, (int, float)) else None,
            interests=list(self.get_state("current_interests", []) or []),
            focus_topic=str(self.get_state("focus_topic", "") or ""),
            intentions=[i.title for i in self.list_intentions(limit=5)],
            opinions=[o.summary_line() for o in self.list_opinions(limit=5)],
        )

    def stats(self) -> dict:
        def count(table: str, where: str = "") -> int:
            return self._conn.execute(
                f"SELECT COUNT(*) FROM {table} {where}"
            ).fetchone()[0]

        narrative = self.latest_narrative()
        return {
            "opinions_active": count("opinions", "WHERE status = 'active'"),
            "intentions_active": count("intentions", "WHERE state = 'active'"),
            "preferences": count("preferences"),
            "narrative_version": narrative.version if narrative else 0,
            "journal_entries": count("thought_journal"),
            "journal_archived": count("thought_journal_archive"),
            "path": str(self._path),
        }

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass
