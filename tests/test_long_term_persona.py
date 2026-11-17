"""長期記憶＆人格維持機能（ViewerProfileStore, PersonaValidator, ReflectionEngine, MemoryStore）の単体テスト"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from backend.memory.viewer_store import ViewerProfileStore
from backend.llm.persona_validator import PersonaValidator
from backend.memory.reflection_engine import ReflectionEngine
from backend.memory.memory_store import MemoryStore
from backend.memory.long_term_memory import LongTermMemory


class TestLongTermPersona(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.base_path = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    # ── 1. ViewerProfileStore テスト ──────────────────────────────
    def test_viewer_profile_store(self):
        db_path = self.base_path / "test_viewers.db"
        store = ViewerProfileStore(db_path)

        # 初回来訪
        p1 = store.record_visit("user1", "アリス")
        self.assertEqual(p1.display_name, "アリス")
        self.assertEqual(p1.visit_count, 1)
        self.assertEqual(p1.familiarity_level, 1)
        self.assertEqual(p1.level_name, "初見")

        # 複数回来訪で常連化 (visit_count = 5 -> Lv3 常連)
        for _ in range(4):
            p2 = store.record_visit("user1", "アリス")
        self.assertEqual(p2.visit_count, 5)
        self.assertEqual(p2.familiarity_level, 3)
        self.assertEqual(p2.level_name, "常連")

        # メモの追加
        store.add_note("user1", "猫が好きと言っていた")
        store.add_note("user1", "新しいキーボードを買った")
        # 重複メモは追加されない
        store.add_note("user1", "猫が好きと言っていた")

        p3 = store.get_profile("user1")
        self.assertIn("猫が好きと言っていた", p3.notes)
        self.assertIn("新しいキーボードを買った", p3.notes)

        # プロンプト生成
        prompt = store.build_prompt("user1")
        self.assertIn("アリス", prompt)
        self.assertIn("常連", prompt)
        self.assertIn("親密度Lv3", prompt)
        self.assertIn("猫が好きと言っていた", prompt)

        store.close()

    # ── 2. PersonaValidator テスト ───────────────────────────────
    def test_persona_validator(self):
        validator = PersonaValidator()

        # Character1の一人称逸脱補正 (私 -> わらわ)
        text1 = "私は九尾の狐ですよ。今日は良い天気ですね。"
        repaired1, modified1 = validator.validate_and_repair(text1, character_name="Character1")
        self.assertTrue(modified1)
        self.assertIn("わらわは", repaired1)
        self.assertNotIn("私は", repaired1)
        self.assertIn("じゃぞ", repaired1)
        self.assertIn("じゃな", repaired1)

        # 僕 -> わらわ
        text2 = "僕の言うことを聞きなさい。"
        repaired2, modified2 = validator.validate_and_repair(text2, character_name="Character1")
        self.assertTrue(modified2)
        self.assertIn("わらわの", repaired2)
        self.assertNotIn("僕の", repaired2)

        # すでに正しい口調の場合は変更フラグが立たない
        text3 = "わらわの力を見るがよい！"
        repaired3, modified3 = validator.validate_and_repair(text3, character_name="Character1")
        self.assertFalse(modified3)
        self.assertEqual(repaired3, text3)

        # 任意の一人称指定テスト
        text4 = "俺がルールだ。"
        repaired4, modified4 = validator.validate_and_repair(
            text4, character_name="一般キャラ", expected_first_person="余"
        )
        self.assertTrue(modified4)
        self.assertIn("余が", repaired4)

    # ── 3. MemoryStore viewer_name テスト ────────────────────────
    def test_memory_store_viewer_name(self):
        db_path = self.base_path / "test_memories.db"
        store = MemoryStore(db_path)

        store.add("タナカさんはカレーが好き", source="auto", viewer_name="タナカ")
        store.add("サトウさんは犬を飼っている", source="auto", viewer_name="サトウ")
        store.add("タナカさんは辛いのが得意", source="auto", viewer_name="タナカ")

        # タナカさんのエピソードのみ取得
        tanaka_mems = store.by_viewer("タナカ")
        self.assertEqual(len(tanaka_mems), 2)
        texts = [m.text for m in tanaka_mems]
        self.assertIn("タナカさんはカレーが好き", texts)
        self.assertIn("タナカさんは辛いのが得意", texts)

        satou_mems = store.by_viewer("サトウ")
        self.assertEqual(len(satou_mems), 1)

    # ── 4. ReflectionEngine コア記憶更新テスト ────────────────────
    def test_reflection_engine_diary_append(self):
        engine = ReflectionEngine(max_diary_entries=3)
        prompts_dir = self.base_path / "prompts"
        prompts_dir.mkdir(parents=True)
        ltm = LongTermMemory("Character1", prompts_dir=prompts_dir)

        # 1回目の日記追加
        entry1 = "### [2026-09-25 10:00] 配信の思い出\nリスナーのみんなと昔話をして楽しかったのじゃ。"
        engine._append_to_core_memory(ltm, entry1)

        raw1 = ltm.get_raw()
        self.assertIn("## 最近の思い出・日記", raw1)
        self.assertIn("リスナーのみんなと昔話をして楽しかったのじゃ。", raw1)

        # 2回目・3回目・4回目の日記追加（上限3件でローテーションされるか）
        engine._append_to_core_memory(ltm, "### [2026-09-26 10:00] 配信の思い出\n日記2")
        engine._append_to_core_memory(ltm, "### [2026-09-27 10:00] 配信の思い出\n日記3")
        engine._append_to_core_memory(ltm, "### [2026-09-28 10:00] 配信の思い出\n日記4")

        raw_final = ltm.get_raw()
        self.assertIn("日記4", raw_final)
        self.assertIn("日記3", raw_final)
        self.assertIn("日記2", raw_final)
        # 最も古い日記1はローテーションで押し出されている
        self.assertNotIn("リスナーのみんなと昔話をして楽しかったのじゃ。", raw_final)


if __name__ == "__main__":
    unittest.main()
