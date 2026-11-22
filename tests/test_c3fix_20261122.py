# -*- coding: utf-8 -*-
"""c3fix（2026-11-22）: §3-C3 フィルタの配線是正のテスト。

経緯: CharacterAgent.build_idle_memory の §3-C3 フィルタ（話題化した
ばかりの記憶を L3 常駐注入から外す）は、プランナーを
getattr(orch, "_planner", None) で引いていた。orch は OrchestratorConfig
（設定値だけのオブジェクト）で _planner を持たないため、Phase 1 の
実装当初から本番では常に None になり、フィルタは無効だった。
既存テスト（test_phase1_incorporation）は疑似 config に _planner を
生やしていたため、死んだ配線のまま合格し続けていた。

是正後の配線は、視聴者台帳の共有（F2）と同じ方式:
Orchestrator が保持する会話単位のプランナーを、_build_cast で
set_topic_planner により全出演者へ差し込む。

ここでは:
- 差し込みでのみフィルタが効くこと（config 側の属性は二度と
  参照されないこと）を固定する
- 実物の Orchestrator（knowledge 有効）で、全出演者に同一の
  プランナーが差し込まれ、commit → build_idle_memory の除外が
  端から端まで効くこと
- knowledge 無効時はプランナーも差し込みも None で従来動作のこと
"""
from __future__ import annotations

import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import yaml

from backend.config import AppConfig, CharacterConfig, EnsembleConfig
from backend.ensemble.agent import CharacterAgent
from backend.knowledge.models import WebDocument
from backend.knowledge.topic_planner import Topic, TopicPlanner
from backend.orchestrator import Orchestrator
from helpers import FakeHub


# ── bare agent（Phase 1 テストと同じ形の最小構成） ──────────

class _Entry:
    def __init__(self, text):
        self.text = text


class _Store:
    def __init__(self, texts):
        self._texts = texts

    def sample_recent(self, limit=5):
        return [_Entry(t) for t in self._texts[:limit]]


TEXTS = ["お菓子作りの思い出の本文です", "別の記憶Aです",
         "別の記憶Bです", "別の記憶Cです"]


def _planner_with_topic_memory() -> TopicPlanner:
    p = TopicPlanner(state_path="")
    topic = Topic(kind="memory", label="記憶からの話題", documents=[
        WebDocument(title="タイトル", snippet=TEXTS[0], source="memory")])
    p.commit(topic)
    return p


def _bare_agent(store, orch_extra=None) -> CharacterAgent:
    agent = CharacterAgent.__new__(CharacterAgent)
    agent.long_term_memory = None
    agent.memory_store = store
    agent._topic_planner = None
    orch = SimpleNamespace(idle_episodic_entries=2,
                           idle_core_memory_chars=4000, **(orch_extra or {}))
    agent.app = SimpleNamespace(orchestrator=orch)
    return agent


class TestSetterWiring:
    def test_excluded_when_injected(self):
        agent = _bare_agent(_Store(TEXTS))
        agent.set_topic_planner(_planner_with_topic_memory())
        text = agent.build_idle_memory()
        assert "お菓子作りの思い出" not in text
        # 除外で空いた枠は別の記憶で埋まる
        assert "別の記憶Aです" in text and "別の記憶Bです" in text

    def test_config_attribute_is_never_consulted(self):
        # 旧配線の再発防止: config 側に _planner があっても、差し込みが
        # 無ければフィルタは効かない（本番の OrchestratorConfig には
        # 元々 _planner が無く、旧コードは常にこの状態だった）
        planner = _planner_with_topic_memory()
        agent = _bare_agent(_Store(TEXTS), orch_extra={"_planner": planner})
        text = agent.build_idle_memory()
        assert "お菓子作りの思い出" in text

    def test_setter_none_disables(self):
        agent = _bare_agent(_Store(TEXTS))
        agent.set_topic_planner(_planner_with_topic_memory())
        agent.set_topic_planner(None)
        text = agent.build_idle_memory()
        assert "お菓子作りの思い出" in text

    def test_cooldown_expired_not_excluded(self):
        planner = _planner_with_topic_memory()
        # クールダウン（30分）を過ぎた使用記録は除外対象にしない
        planner._topic_memory_used_at = {
            k: time.time() - 1801 for k in planner._topic_memory_used_at}
        agent = _bare_agent(_Store(TEXTS))
        agent.set_topic_planner(planner)
        text = agent.build_idle_memory()
        assert "お菓子作りの思い出" in text


# ── 実物 Orchestrator での配線 ─────────────────────────────

def _write_char(dirpath: Path, name: str) -> str:
    path = dirpath / f"{name}.yaml"
    path.write_text(yaml.safe_dump(
        {"name": name, "description": f"{name}の説明",
         "personality": f"{name}の口調"}, allow_unicode=True),
        encoding="utf-8")
    return str(path)


def _make_orchestrator(knowledge_enabled: bool) -> Orchestrator:
    tmp = Path(tempfile.mkdtemp(prefix="aivt_c3fix_"))
    cfg = AppConfig()
    cfg.stt.enabled = False
    cfg.ollama.auto_start = False
    cfg.ollama.preload_on_start = False
    cfg.tts.irodori_server.auto_start = False
    cfg.knowledge.enabled = knowledge_enabled
    cfg.characters = [
        CharacterConfig(name="クロエ", id="chloe", data_dir=str(tmp),
                        prompt_file=_write_char(tmp, "クロエ")),
        CharacterConfig(name="ミライ", id="mirai", data_dir=str(tmp),
                        prompt_file=_write_char(tmp, "ミライ")),
    ]
    cfg.ensemble = EnsembleConfig()
    return Orchestrator(cfg, FakeHub())


class TestOrchestratorWiring:
    def test_all_agents_share_orchestrator_planner(self):
        orch = _make_orchestrator(knowledge_enabled=True)
        assert orch._planner is not None
        assert len(orch.agents) == 2
        for agent in orch.agents:
            assert agent._topic_planner is orch._planner

    def test_knowledge_disabled_injects_none(self):
        orch = _make_orchestrator(knowledge_enabled=False)
        assert orch._planner is None
        for agent in orch.agents:
            assert agent._topic_planner is None

    def test_end_to_end_exclusion(self):
        orch = _make_orchestrator(knowledge_enabled=True)
        planner = orch._planner
        # 本番の planner_state.json を汚さないよう、永続化を切り、
        # 読み込まれた使用台帳は空にしてから始める（メモリ内のみ）
        planner._state_path = None
        planner._topic_memory_used_at.clear()
        lead = orch.agents[0]
        lead.memory_store = _Store(TEXTS)
        planner.commit(Topic(kind="memory", label="記憶からの話題",
                             documents=[WebDocument(
                                 title="タイトル", snippet=TEXTS[0],
                                 source="memory")]))
        text = lead.build_idle_memory()
        assert "お菓子作りの思い出" not in text
        assert "別の記憶Aです" in text
