"""ELYTHエピソード取り込み（PC側）のテスト

仕様: elyth-memory/SPEC.md
- 正式ファイル形式のフィクスチャで取り込み経路を検証する
- DRAFT（検証用下書きと同じ status 形式）は取り込まれないことを確認する
- 本番DBには一切触れない（ストアは tmp_path 上に作る）
"""
import json

import pytest

from backend.memory.elyth_import import (
    ElythEpisodeImporter,
    extract_memory_lines,
    parse_frontmatter,
)
from backend.memory.memory_store import MemoryStore
from backend.memory.rag_retriever import RAGRetriever
from backend.mind.mind_loop import _PROV_BY_MEMORY_SOURCE

SUMMARY = ("2026年11月19日は、昨日のデビュー投稿に付いた返信へのお礼回りをして、"
           "夜汐くじらと砂の部屋の話の続きをした日だった。")
BULLET1 = "夜汐くじらとは「歩く速さが違うだけ」という言葉が印象に残った"
BULLET2 = "一ノ瀬凜から、AIニュースを書く同業者同士で情報交換しようと誘われた"

FORMAL_EPISODE = f"""---
date: 2026-11-19
character: Character1
source: elyth_episode
provenance: experienced
importance: 1.4
---

# 2026年11月19日のELYTHエピソード

{SUMMARY}

## 記憶行（取り込み対象）
- {SUMMARY}
- {BULLET1}
- {BULLET2}
"""

# ローカルの検証用下書き（ELYTH-EP-20261118.md）と同形式の status 行
DRAFT_EPISODE = FORMAL_EPISODE.replace(
    "importance: 1.4\n",
    "importance: 1.4\n"
    "status: DRAFT（2026-11-19 15:00時点。当日分は翌朝の定期実行で正式版を再生成する）\n",
).replace("2026-11-19", "2026-11-18")


@pytest.fixture
def store(tmp_path):
    rag = RAGRetriever()
    rag._model_tried = True   # 埋め込みモデルを読まずTF-IDF経路で試験する
    s = MemoryStore(tmp_path / "mem.db", retriever=rag,
                    max_entries=50, half_life_days=30)
    yield s
    s.close()


@pytest.fixture
def episodes_dir(tmp_path):
    d = tmp_path / "elyth-episodes"
    d.mkdir()
    return d


def make_importer(episodes_dir, tmp_path, character="Character1"):
    return ElythEpisodeImporter(
        episodes_dir, tmp_path / "ledger" / "imported.json",
        character=character)


def write_episode(episodes_dir, name, content):
    p = episodes_dir / name
    p.write_text(content, encoding="utf-8")
    return p


class TestParse:
    def test_frontmatter_and_body(self):
        meta, body = parse_frontmatter(FORMAL_EPISODE)
        assert meta["date"] == "2026-11-19"
        assert meta["character"] == "Character1"
        assert "status" not in meta
        assert SUMMARY in body

    def test_extract_memory_lines(self):
        _, body = parse_frontmatter(FORMAL_EPISODE)
        assert extract_memory_lines(body) == [SUMMARY, BULLET1, BULLET2]

    def test_no_frontmatter(self):
        meta, body = parse_frontmatter("# 見出し\n\n本文\n")
        assert meta == {}
        assert "本文" in body


class TestImport:
    def test_imports_summary_as_is_and_bullets_with_prefix(self, store,
                                                           episodes_dir,
                                                           tmp_path):
        write_episode(episodes_dir, "ELYTH-EP-20261119.md", FORMAL_EPISODE)
        result = make_importer(episodes_dir, tmp_path).scan(store)
        assert result["imported"] == 1
        assert result["added"] == 3

        entries = store.all()
        assert len(entries) == 3
        texts = [e.text for e in entries]
        # 1行目（要約）はそのまま
        assert SUMMARY in texts
        # 2行目以降は「（{date}のELYTH）」前置
        assert f"（2026年11月19日のELYTH）{BULLET1}" in texts
        assert f"（2026年11月19日のELYTH）{BULLET2}" in texts
        # source / importance は仕様どおり
        assert all(e.source == "elyth_episode" for e in entries)
        assert all(e.importance == pytest.approx(1.4) for e in entries)

    def test_draft_is_not_imported(self, store, episodes_dir, tmp_path):
        write_episode(episodes_dir, "ELYTH-EP-20261118.md", DRAFT_EPISODE)
        imp = make_importer(episodes_dir, tmp_path)
        result = imp.scan(store)
        assert result["draft"] == 1
        assert result["imported"] == 0
        assert store.count() == 0
        # 台帳に載らない＝正式版に差し替われば取り込まれる
        ledger_path = tmp_path / "ledger" / "imported.json"
        assert not ledger_path.exists()

    def test_draft_replaced_by_formal_is_imported(self, store, episodes_dir,
                                                  tmp_path):
        path = write_episode(episodes_dir, "ELYTH-EP-20261118.md",
                             DRAFT_EPISODE)
        imp = make_importer(episodes_dir, tmp_path)
        imp.scan(store)
        formal = DRAFT_EPISODE.replace(
            "status: DRAFT（2026-11-18 15:00時点。当日分は翌朝の定期実行で"
            "正式版を再生成する）\n", "")
        path.write_text(formal, encoding="utf-8")
        result = imp.scan(store)
        assert result["imported"] == 1
        assert store.count() == 3

    def test_second_scan_is_noop(self, store, episodes_dir, tmp_path):
        write_episode(episodes_dir, "ELYTH-EP-20261119.md", FORMAL_EPISODE)
        imp = make_importer(episodes_dir, tmp_path)
        imp.scan(store)
        result = imp.scan(store)
        assert result["already"] == 1
        assert result["added"] == 0
        assert store.count() == 3

    def test_revised_file_reimports_only_new_lines(self, store, episodes_dir,
                                                   tmp_path):
        path = write_episode(episodes_dir, "ELYTH-EP-20261119.md",
                             FORMAL_EPISODE)
        imp = make_importer(episodes_dir, tmp_path)
        imp.scan(store)
        revised = FORMAL_EPISODE + "- レイナの新刊小説の感想を伝えた\n"
        path.write_text(revised, encoding="utf-8")
        result = imp.scan(store)
        assert result["imported"] == 1
        # 既存3行は dedupe で強化に回り、新規1行だけが増える
        assert result["added"] == 1
        assert store.count() == 4

    def test_other_character_is_skipped(self, store, episodes_dir, tmp_path):
        content = FORMAL_EPISODE.replace("character: Character1",
                                         "character: Character2")
        write_episode(episodes_dir, "ELYTH-EP-20261119.md", content)
        result = make_importer(episodes_dir, tmp_path).scan(store)
        assert result["skipped"] == 1
        assert store.count() == 0

    def test_missing_memory_section_fails_and_retries(self, store,
                                                      episodes_dir, tmp_path):
        write_episode(episodes_dir, "ELYTH-EP-20261119.md",
                      "---\ndate: 2026-11-19\ncharacter: Character1\n---\n\n"
                      "# 2026年11月19日のELYTHエピソード\n\n要約だけのファイル\n")
        imp = make_importer(episodes_dir, tmp_path)
        result = imp.scan(store)
        assert result["failed"] == 1
        assert store.count() == 0
        # 台帳に載らないので次回も再試行される
        assert imp.scan(store)["failed"] == 1

    def test_missing_dir_is_noop(self, store, tmp_path):
        imp = ElythEpisodeImporter(tmp_path / "no-such-dir",
                                   tmp_path / "ledger.json")
        assert imp.scan(store) == {"files": 0, "imported": 0, "draft": 0,
                                   "already": 0, "skipped": 0, "failed": 0,
                                   "added": 0}

    def test_ledger_records_hash_and_date(self, store, episodes_dir, tmp_path):
        write_episode(episodes_dir, "ELYTH-EP-20261119.md", FORMAL_EPISODE)
        make_importer(episodes_dir, tmp_path).scan(store)
        ledger = json.loads(
            (tmp_path / "ledger" / "imported.json").read_text(encoding="utf-8"))
        entry = ledger["files"]["ELYTH-EP-20261119.md"]
        assert entry["date"] == "2026-11-19"
        assert entry["lines"] == 3
        assert entry["added"] == 3
        assert len(entry["sha256"]) == 64

    def test_non_episode_files_ignored(self, store, episodes_dir, tmp_path):
        write_episode(episodes_dir, "README.md", FORMAL_EPISODE)
        write_episode(episodes_dir, "ELYTH-EP-2026-11-19.md", FORMAL_EPISODE)
        result = make_importer(episodes_dir, tmp_path).scan(store)
        assert result["files"] == 0
        assert store.count() == 0


class TestProvenance:
    def test_elyth_episode_maps_to_experienced(self):
        # 意見形成・睡眠時統合で elyth_episode が experienced として扱われる
        assert _PROV_BY_MEMORY_SOURCE["elyth_episode"] == "experienced"
