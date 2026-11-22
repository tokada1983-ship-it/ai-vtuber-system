"""設計監査（design-audit-20261122）所見 F1〜F10 の修正に対する回帰テスト"""
from __future__ import annotations

import sys
import tempfile
import types
from pathlib import Path

import pytest
import yaml

from backend.config import AppConfig, CharacterConfig, EnsembleConfig
from backend.orchestrator import Orchestrator
from helpers import FakeHub, NoAudio, StubLLM, StubTTS

RAW_A = '{"emotion":"happy","motion":"wave","speech":"わらわが答えてやろう。"}'
RAW_B = '{"emotion":"surprised","motion":"nod","speech":"えっ、そうなの？"}'


def _write_char(dirpath: Path, name: str, description: str) -> str:
    path = dirpath / f"{name}.yaml"
    path.write_text(yaml.safe_dump({"name": name, "description": description,
                                    "personality": f"{name}の口調"}, allow_unicode=True),
                    encoding="utf-8")
    return str(path)


def make_duo() -> tuple[Orchestrator, FakeHub]:
    """test_ensemble.make_duo と同じ組み立ての2体オーケストレータ"""
    tmp = Path(tempfile.mkdtemp(prefix="aivt_audit_duo_"))
    cfg = AppConfig()
    cfg.stt.enabled = False
    cfg.ollama.auto_start = False
    cfg.ollama.preload_on_start = False
    cfg.tts.irodori_server.auto_start = False
    cfg.knowledge.enabled = False
    cfg.characters = [
        CharacterConfig(name="クロエ", id="chloe", data_dir=str(tmp),
                        prompt_file=_write_char(tmp, "クロエ", "大妖狐のAI"),
                        aliases=["くろえ"]),
        CharacterConfig(name="ミライ", id="mirai", data_dir=str(tmp),
                        prompt_file=_write_char(tmp, "ミライ", "未来から来た相方AI")),
    ]
    cfg.ensemble = EnsembleConfig(turn_gap_sec=0)
    hub = FakeHub()
    orch = Orchestrator(cfg, hub)
    a, b = orch.agents
    a.llm, a.tts = StubLLM(RAW_A), StubTTS()
    b.llm, b.tts = StubLLM(RAW_B), StubTTS()
    orch._audio = NoAudio()
    orch._audio_cache = Path(tempfile.mkdtemp(prefix="aivt_audio_"))
    return orch, hub


# ── F1: WS 記憶操作の適用範囲 ─────────────────────────────────

def _load_ws_routes():
    """backend.runtime をスタブに差し替えてから ws_routes を import する。

    本物の runtime は import 時点で本番 config の Orchestrator を構築して
    しまうため、テストでは名前だけを満たすスタブを先に差し込む。
    ws_routes は import 時に名前を束縛するので、import 後にスタブを
    sys.modules から外しても参照は保たれる。
    """
    inserted = False
    if "backend.runtime" not in sys.modules:
        stub = types.ModuleType("backend.runtime")
        stub.AUDIO_CACHE = Path(".")
        stub.FRONTEND = Path(".")
        stub.MODELS_DIR = Path(".")
        stub._avatar_calibration = {}
        stub._preset_manager = None
        stub.config = AppConfig()
        stub.hub = None
        stub.monitor = None
        stub.orchestrator = None
        sys.modules["backend.runtime"] = stub
        inserted = True
    try:
        from backend.api import ws_routes
    finally:
        if inserted:
            del sys.modules["backend.runtime"]
    return ws_routes


class FakeWS:
    def __init__(self):
        self.sent: list[dict] = []

    async def send_json(self, data):
        self.sent.append(data)


@pytest.fixture()
def ws_env(monkeypatch):
    ws_routes = _load_ws_routes()
    orch, hub = make_duo()
    monkeypatch.setattr(ws_routes, "orchestrator", orch)
    monkeypatch.setattr(ws_routes, "hub", hub)
    return ws_routes, orch, hub


class TestF1WsMemoryScope:
    async def test_clear_episodic_applies_to_all_characters(self, ws_env):
        ws_routes, orch, hub = ws_env
        a, b = orch.agents
        a.memory_store.add("猫カフェでモンブランを食べた日のこと", source="manual")
        a.memory_store.add("真冬の屋外で流れ星を数えた夜のこと", source="manual")
        b.memory_store.add("ミライが初めて自転車に乗れた日のこと", source="manual")
        assert a.memory_store.count() == 2 and b.memory_store.count() == 1

        ws = FakeWS()
        await ws_routes._handle_ws_cmd("clear_episodic_memory", {}, ws=ws)

        assert a.memory_store.count() == 0
        assert b.memory_store.count() == 0
        resp = [d for d in ws.sent if d.get("type") == "episodic_memory_cleared"]
        assert resp and resp[0]["removed"] == 3
        assert resp[0]["per_character"] == {a.key: 2, b.key: 1}
        logs = [d["message"] for ch, d in hub.events if ch == "log"]
        assert any("全員" in m and "クロエ" in m and "ミライ" in m for m in logs)

    async def test_clear_episodic_single_character(self, ws_env):
        ws_routes, orch, hub = ws_env
        a, b = orch.agents
        a.memory_store.add("クロエだけの思い出", source="manual")
        b.memory_store.add("ミライだけの思い出", source="manual")

        ws = FakeWS()
        await ws_routes._handle_ws_cmd(
            "clear_episodic_memory", {"character": "mirai"}, ws=ws)

        assert a.memory_store.count() == 1
        assert b.memory_store.count() == 0
        resp = [d for d in ws.sent if d.get("type") == "episodic_memory_cleared"]
        assert resp and resp[0]["removed"] == 1
        logs = [d["message"] for ch, d in hub.events if ch == "log"]
        assert any("ミライのエピソード記憶を全消去" in m for m in logs)

    async def test_get_episodic_memory_labels_character(self, ws_env):
        ws_routes, orch, hub = ws_env
        a, b = orch.agents
        b.memory_store.add("ミライの思い出", source="manual")

        ws = FakeWS()
        await ws_routes._handle_ws_cmd(
            "get_episodic_memory", {"character": "ミライ"}, ws=ws)
        resp = [d for d in ws.sent if d.get("type") == "episodic_memory"]
        assert resp and resp[0]["character"] == b.key
        assert len(resp[0]["entries"]) == 1

        ws2 = FakeWS()
        await ws_routes._handle_ws_cmd("get_episodic_memory", {}, ws=ws2)
        resp2 = [d for d in ws2.sent if d.get("type") == "episodic_memory"]
        assert resp2 and resp2[0]["character"] == a.key
        assert resp2[0]["entries"] == []

    async def test_clear_short_term_applies_to_all(self, ws_env):
        ws_routes, orch, hub = ws_env
        a, b = orch.agents
        a.log.add_idle("クロエの発話")
        b.log.add_idle("ミライの発話")
        assert a.log.history_size > 0 and b.log.history_size > 0

        ws = FakeWS()
        await ws_routes._handle_ws_cmd("clear_short_term_memory", {}, ws=ws)

        assert a.log.history_size == 0
        assert b.log.history_size == 0
        resp = [d for d in ws.sent if d.get("type") == "short_term_memory_cleared"]
        assert resp and resp[0]["characters"] == [a.key, b.key]

    async def test_unknown_character_is_an_error_not_silent_lead(self, ws_env):
        ws_routes, orch, hub = ws_env
        a, b = orch.agents
        a.memory_store.add("消えてはいけない思い出", source="manual")

        ws = FakeWS()
        await ws_routes._handle_ws_cmd(
            "clear_episodic_memory", {"character": "存在しない"}, ws=ws)

        # 黙ってリーダーを消さず、エラー応答になり実データは無傷
        assert a.memory_store.count() == 1
        errs = [d for d in ws.sent if d.get("type") == "error"]
        assert errs and "存在しない" in errs[0]["error"]


# ── F2: 視聴者ポイント台帳の単一化 ─────────────────────────────

def _make_legacy_store(path: Path, cfg) -> "ViewerProfileStore":
    from backend.memory.viewer_store import ViewerProfileStore
    path.parent.mkdir(parents=True, exist_ok=True)
    return ViewerProfileStore(db_path=path, points_cfg=cfg)


class TestF2ViewerStoreMerge:
    def test_merge_sums_points_and_unions_profiles(self, tmp_path):
        from backend.memory.viewer_store import (
            ViewerProfileStore, merge_legacy_viewer_dbs)
        points_cfg = AppConfig().viewer_points
        p_a = tmp_path / "a" / "viewer_profiles.db"
        p_b = tmp_path / "b" / "viewer_profiles.db"
        sa = _make_legacy_store(p_a, points_cfg)
        sa.record_visit("共通さん", "共通さん")          # visit +10
        sa.award_points("共通さん", "comment")            # +2
        sa.record_visit("Aだけさん", "Aだけさん")         # visit +10
        sa.close()
        sb = _make_legacy_store(p_b, points_cfg)
        sb.record_visit("共通さん", "共通さん")          # visit +10
        sb.award_points("共通さん", "picked")             # +3
        sb.record_visit("Bだけさん", "Bだけさん")         # visit +10
        sb.close()

        shared = ViewerProfileStore(db_path=tmp_path / "viewer_profiles.db",
                                    points_cfg=points_cfg)
        summary = merge_legacy_viewer_dbs(shared, [p_a, p_b])
        try:
            assert summary["points_before"] == 45
            assert summary["points_after"] == 45
            assert summary["profiles_after"] == 3
            common = shared.get_profile("共通さん")
            assert common is not None
            assert common.points == 25          # (10+2) + (10+3)
            assert common.visit_count == 2      # 両 DB の来訪を合算
            assert shared.get_profile("Aだけさん").points == 10
            assert shared.get_profile("Bだけさん").points == 10
            assert summary["events_copied"] == 6
            # 二重マージ防止: もう一度呼んでも何も足されない
            again = merge_legacy_viewer_dbs(shared, [p_a, p_b])
            assert again["files"] == []
            assert shared.get_profile("共通さん").points == 25
        finally:
            shared.close()

    def test_merge_handles_legacy_schema_without_points(self, tmp_path):
        import sqlite3
        from backend.memory.viewer_store import (
            ViewerProfileStore, merge_legacy_viewer_dbs)
        points_cfg = AppConfig().viewer_points
        p_old = tmp_path / "old" / "viewer_profiles.db"
        p_old.parent.mkdir(parents=True)
        conn = sqlite3.connect(str(p_old))
        conn.execute(
            "CREATE TABLE viewer_profiles ("
            "viewer_id TEXT PRIMARY KEY, display_name TEXT NOT NULL,"
            "visit_count INTEGER NOT NULL DEFAULT 1,"
            "first_seen_at REAL NOT NULL, last_seen_at REAL NOT NULL,"
            "familiarity_level INTEGER NOT NULL DEFAULT 1,"
            "notes TEXT NOT NULL DEFAULT '', custom_nickname TEXT)")
        conn.execute(
            "INSERT INTO viewer_profiles VALUES (?,?,?,?,?,?,?,?)",
            ("古参さん", "古参さん", 12, 1000.0, 2000.0, 3, "", None))
        conn.commit()
        conn.close()

        shared = ViewerProfileStore(db_path=tmp_path / "viewer_profiles.db",
                                    points_cfg=points_cfg)
        try:
            summary = merge_legacy_viewer_dbs(shared, [p_old])
            prof = shared.get_profile("古参さん")
            assert prof is not None
            # 来訪12回・Lv3 の旧行は開始ポイント200へ換算されて引き継がれる
            assert prof.points == 200
            assert summary["points_after"] == summary["points_before"] == 200
        finally:
            shared.close()


class TestF2SharedStoreWiring:
    def test_both_agents_share_one_store(self):
        orch, hub = make_duo()
        a, b = orch.agents
        assert a.viewer_store is not None
        assert a.viewer_store is b.viewer_store
        # 片方への加算がもう片方から見える（同一台帳）
        a.viewer_store.record_visit("視聴者X", "視聴者X")
        a.viewer_store.award_points("視聴者X", "comment")
        prof = b.viewer_store.get_profile("視聴者X")
        assert prof is not None and prof.points == 12

    def test_unanswered_and_answered_use_same_ledger(self):
        orch, hub = make_duo()
        from helpers import comment as make_comment
        orch._record_unanswered_comment(make_comment("やあ", author="常連さん"),
                                        triage_category="probability_miss")
        a, b = orch.agents
        pa = a.viewer_store.get_profile("常連さん")
        pb = b.viewer_store.get_profile("常連さん")
        assert pa is not None and pb is not None
        assert pa.points == pb.points == 12   # 来訪10＋コメント2


# ── F3: タグ解析の失敗帰属（生成単位） ─────────────────────────

class TagStubLLM(StubLLM):
    """タグ方式バックエンドの generate を模す（parse_tag_output 経由）"""

    async def generate(self, system_prompt, messages, *, route="respond",
                       temperature=None):
        self.generate_calls += 1
        from backend.llm.parser import parse_tag_output
        return parse_tag_output(self._raw)


class TestF3PerGenerationAttribution:
    async def test_prefetch_report_counts_only_own_failure(self):
        orch, hub = make_duo()
        a, b = orch.agents
        a.llm = TagStubLLM("[emo:happy][mot:")       # fatal（ヘッダ途中で切断）
        b.llm = TagStubLLM(RAW_A.replace("emotion", "emo").replace("motion", "mot")
                           if False else "[emo:happy][mot:wave]やっほー！")
        report_a: dict = {}
        resp_a, _ = await orch._generate_idle_response("sys", a,
                                                       gen_report=report_a)
        assert resp_a is None
        assert report_a.get("tag_failures") == 1
        assert report_a.get("tag_outcome") == "fatal"

        report_b: dict = {}
        resp_b, _ = await orch._generate_idle_response("sys", b,
                                                       gen_report=report_b)
        assert resp_b is not None
        # 成功した生成の報告先に、別生成の失敗は混ざらない
        assert report_b.get("tag_failures", 0) == 0
        assert report_b.get("tag_outcome") == "ok"


# ── F4: 飽和リセットのクールダウン一本化 ───────────────────────

class TestF4SaturationCooldown:
    def test_saturation_escalates_after_shared_cooldown(self):
        import time as _time
        from helpers import make_orchestrator
        orch, _hub = make_orchestrator(StubLLM(RAW_A), StubTTS())
        # 別駆動元（LoopGuard 等）が300秒前にリセットした状態。
        # 旧600秒ゲートでは飽和エスカレーションが無言で不発だった
        orch._last_context_reset_at = _time.time() - 300
        for _ in range(4):
            orch._note_saturation("吸着ワード")
        assert orch._last_context_reset_at > _time.time() - 10

    def test_saturation_suppressed_within_shared_cooldown(self):
        import time as _time
        from helpers import make_orchestrator
        orch, _hub = make_orchestrator(StubLLM(RAW_A), StubTTS())
        just_now = _time.time()
        orch._last_context_reset_at = just_now
        for _ in range(5):
            orch._note_saturation("吸着ワード")
        # 共有クールダウン（120秒）内は抑止され、上書きされない
        assert orch._last_context_reset_at == just_now


# ── F5: provenance 写像の網羅 ─────────────────────────────────

class TestF5ProvenanceMapping:
    def test_stream_digest_and_imported_are_experienced(self):
        from backend.mind.mind_loop import provenance_for_memory_source
        assert provenance_for_memory_source("stream_digest") == "experienced"
        assert provenance_for_memory_source("imported") == "experienced"
        assert provenance_for_memory_source("stream_episode") == "experienced"
        assert provenance_for_memory_source("web") == "researched"
        assert provenance_for_memory_source("manual") == "told"

    def test_unknown_source_falls_back_to_inferred_explicitly(self):
        from backend.mind.mind_loop import provenance_for_memory_source
        assert provenance_for_memory_source("no_such_source") == "inferred"
        assert provenance_for_memory_source("") == "inferred"

    def test_all_memory_store_sources_are_mapped(self):
        """MemoryStore.add に渡される source リテラル全量が写像に載っていること。

        新しい source を足して写像を更新し忘れると、このテストが落ちる。
        """
        import re
        from pathlib import Path as _P
        from backend.memory.elyth_import import SOURCE as ELYTH_SOURCE
        from backend.memory.mind_store import PROVENANCE_VALUES
        from backend.mind.mind_loop import _PROV_BY_MEMORY_SOURCE

        root = _P(__file__).resolve().parent.parent
        files = list((root / "backend" / "memory").glob("*.py")) + [
            root / "backend" / "orchestrator.py",
            root / "backend" / "api" / "ws_routes.py",
        ]
        found = {ELYTH_SOURCE}
        for f in files:
            found |= set(re.findall(r'source="([a-z_]+)"',
                                    f.read_text(encoding="utf-8")))
        unmapped = {s for s in found if s not in _PROV_BY_MEMORY_SOURCE}
        assert not unmapped, f"provenance 写像に無い source: {unmapped}"
        for prov in _PROV_BY_MEMORY_SOURCE.values():
            assert prov in PROVENANCE_VALUES


# ── F6: 陳腐化した複製モジュールの除去 ─────────────────────────

class TestF6NoDuplicateModules:
    def test_no_duplicate_module_basenames(self):
        """同名モジュールの複製が増えていないこと（旧 backend/topic_planner.py
        の轍: 正本が knowledge 側にあるのに旧版が backend 直下に残った）。

        既存の同名は別ドメインのパッケージ内 base/models で、完全修飾
        import されるため混同しない。ここでは「既知の許可分を超える
        新しい重複」を検出する。
        """
        from collections import Counter
        from pathlib import Path as _P
        allowed = {"base.py": 4, "models.py": 2}
        root = _P(__file__).resolve().parent.parent / "backend"
        names = [p.name for p in root.rglob("*.py")
                 if p.name != "__init__.py" and "__pycache__" not in p.parts]
        over = {n: c for n, c in Counter(names).items()
                if c > allowed.get(n, 1)}
        assert not over, f"同名モジュールが増えています: {over}"

    def test_stale_topic_planner_removed(self):
        from pathlib import Path as _P
        root = _P(__file__).resolve().parent.parent / "backend"
        assert not (root / "topic_planner.py").exists()
        # 正本は knowledge 側
        assert (root / "knowledge" / "topic_planner.py").exists()


# ── F7: 保留回答の永続化 ───────────────────────────────────────

def make_stateful_planner(state_path, **kw):
    from backend.knowledge.topic_planner import TopicPlanner
    args = dict(
        news=None, search=None,
        keyword_provider=lambda: [],
        theme_provider=lambda: "",
        memory_provider=lambda: None,
        state_path=str(state_path),
    )
    args.update(kw)
    return TopicPlanner(**args)


class TestF7PendingAnswerPersistence:
    def test_ready_answer_survives_restart(self, tmp_path):
        from backend.knowledge.models import WebDocument
        state = tmp_path / "planner_state.json"
        p1 = make_stateful_planner(state)
        pid = p1.enqueue_answer("視聴者さん", "富士山の高さは？")
        assert pid
        doc = WebDocument(title="富士山の標高", url="https://example.com/fuji",
                          snippet="標高は3776メートル", source="google_cse",
                          site="example.com")
        assert p1.attach_answer_documents(pid, [doc])

        p2 = make_stateful_planner(state)   # 再起動に相当
        answers = p2.pending_answers()
        assert len(answers) == 1
        assert answers[0]["id"] == pid
        assert answers[0]["state"] == "ready"
        assert answers[0]["question"] == "富士山の高さは？"
        assert len(answers[0]["documents"]) == 1
        assert answers[0]["documents"][0].title == "富士山の標高"
        assert p2.has_deliverable_answer()

    def test_expired_answer_is_not_restored(self, tmp_path):
        import time as _time
        state = tmp_path / "planner_state.json"
        p1 = make_stateful_planner(state)
        pid = p1.enqueue_answer("視聴者さん", "期限切れの質問", ttl_sec=60)
        assert pid
        # 投入時刻を過去へずらして保存し直す（TTL 切れを再現）
        for e in p1._pending_answers:
            e["enqueued_at"] = _time.time() - 3600
        p1._save_state()

        p2 = make_stateful_planner(state)
        assert p2.pending_answers() == []

    def test_waiting_answer_restored_and_claimable(self, tmp_path):
        state = tmp_path / "planner_state.json"
        p1 = make_stateful_planner(state)
        pid = p1.enqueue_answer("視聴者さん", "調査中の質問")
        assert pid

        p2 = make_stateful_planner(state)
        work = p2.claim_answer_work()
        assert work is not None and work["id"] == pid
        # ID の連番が復元分と衝突しない
        pid2 = p2.enqueue_answer("別の人", "別の質問です")
        assert pid2 and pid2 != pid


# ── F9: arc_stats の理由別集計 ─────────────────────────────────

class TestF9ArcStatsReasons:
    def test_abandoned_split_by_reason(self, tmp_path):
        import time as _time
        state = tmp_path / "planner_state.json"
        p = make_stateful_planner(state)
        p._thread = {"kind": "news", "at": _time.time()}
        p.reset_thread()                                # 文脈リセット経路
        p._thread = {"kind": "news", "at": _time.time()}
        p.reset_thread(reason="force_switch")           # 強制切替経路
        stats = p._arc_stats
        assert stats["abandoned"] == 2
        assert stats["abandoned_reset"] == 1
        assert stats["abandoned_force_switch"] == 1

        p2 = make_stateful_planner(state)               # 永続化の往復
        assert p2._arc_stats["abandoned"] == 2
        assert p2._arc_stats["abandoned_reset"] == 1
        assert p2._arc_stats["abandoned_force_switch"] == 1

    def test_no_thread_no_count(self, tmp_path):
        state = tmp_path / "planner_state.json"
        p = make_stateful_planner(state)
        p._thread = None
        p.reset_thread()
        assert p._arc_stats["abandoned"] == 0


# ── F8: config 表面への露出 ────────────────────────────────────

class TestF8MindConfig:
    def test_defaults_match_legacy_code_defaults(self):
        from backend.config import MindConfig
        m = MindConfig()
        assert m.enabled is True
        assert m.idle_sec == 600.0
        assert m.max_per_hour == 3
        assert m.min_interval_sec is None     # None = 3600/max_per_hour
        assert m.poll_sec == 20.0
        assert m.sleep_enabled is True
        assert m.sleep_poll_sec == 60.0

    def test_mindloop_reads_config(self, monkeypatch):
        from types import SimpleNamespace
        from backend.config import AppConfig
        from backend.mind.mind_loop import MindLoop
        for var in ("AIVT_MIND_ENABLED", "AIVT_MIND_IDLE_SEC",
                    "AIVT_MIND_MAX_PER_HOUR", "AIVT_MIND_MIN_INTERVAL_SEC",
                    "AIVT_MIND_POLL_SEC"):
            monkeypatch.delenv(var, raising=False)
        cfg = AppConfig()
        cfg.mind.idle_sec = 123.0
        cfg.mind.max_per_hour = 5
        loop = MindLoop(SimpleNamespace(_cfg=cfg))
        assert loop.idle_sec == 123.0
        assert loop.max_per_hour == 5
        assert loop.min_interval_sec == 3600.0 / 5   # None は導出される
        assert loop.enabled is True

    def test_env_still_overrides_for_migration(self, monkeypatch):
        from types import SimpleNamespace
        from backend.config import AppConfig
        from backend.mind.mind_loop import MindLoop
        monkeypatch.setenv("AIVT_MIND_IDLE_SEC", "999")
        loop = MindLoop(SimpleNamespace(_cfg=AppConfig()))
        assert loop.idle_sec == 999.0

    def test_sleep_manager_reads_config(self, monkeypatch):
        from types import SimpleNamespace
        from backend.config import AppConfig
        from backend.mind.sleep_cycle import SleepManager
        monkeypatch.delenv("AIVT_SLEEP_ENABLED", raising=False)
        monkeypatch.delenv("AIVT_SLEEP_POLL_SEC", raising=False)
        cfg = AppConfig()
        cfg.mind.sleep_poll_sec = 45.0
        mgr = SleepManager(SimpleNamespace(_cfg=cfg))
        assert mgr.poll_sec == 45.0
        assert mgr.enabled is True


class TestF8ConsolidationIntervalConfig:
    def test_interval_initialized_from_config(self):
        from helpers import make_orchestrator
        orch, _hub = make_orchestrator(
            StubLLM(RAW_A), StubTTS(), consolidation_interval_min=25)
        assert orch._consolidation_interval_min == 25

    def test_interval_default_is_disabled(self):
        from helpers import make_orchestrator
        orch, _hub = make_orchestrator(StubLLM(RAW_A), StubTTS())
        assert orch._consolidation_interval_min == 0


class TestF8UnknownKeyWarning:
    def test_unknown_keys_warn_by_name(self, tmp_path, caplog):
        import logging
        import yaml as _yaml
        from backend.config import load_config
        path = tmp_path / "config.yaml"
        path.write_text(_yaml.safe_dump({
            "bogus_top": 1,
            "orchestrator": {"idle_threshold_sec": 30, "bogus_nested": 2},
        }), encoding="utf-8")
        with caplog.at_level(logging.WARNING):
            cfg = load_config(path)
        msgs = " ".join(r.getMessage() for r in caplog.records)
        assert "bogus_top" in msgs
        assert "orchestrator.bogus_nested" in msgs
        # 既知キーは警告されない・値は読める
        assert "idle_threshold_sec" not in msgs
        assert cfg.orchestrator.idle_threshold_sec == 30


# ── F10: 観測死角と dead code ──────────────────────────────────

class TestF10AvatarHealthExposed:
    async def test_api_status_includes_avatar_health(self, monkeypatch):
        _load_ws_routes()   # runtime スタブ経由で http_routes を import する
        import sys as _sys
        http_routes = _sys.modules["backend.api.http_routes"]
        from types import SimpleNamespace

        class _Hub:
            def avatar_health(self):
                return {"chloe": {"state": "alive", "age_sec": 3.0}}

        orch = SimpleNamespace(_status=SimpleNamespace(
            video_id="v1", llm_backend="stub", tts_backend="stub",
            uptime_sec=12.0))
        monkeypatch.setattr(http_routes, "orchestrator", orch)
        monkeypatch.setattr(http_routes, "hub", _Hub())
        result = await http_routes.api_status()
        assert result["avatar_health"] == {"chloe": {"state": "alive",
                                                     "age_sec": 3.0}}


class TestF10DeadCodeRemoved:
    def test_removed_methods_stay_removed(self):
        from backend.knowledge.news import NewsFeed
        from backend.knowledge.web_search import WebSearchClient
        from backend.memory.memory_consolidator import MemoryConsolidator
        from backend.memory.memory_store import MemoryStore
        from backend.memory.mind_store import MindStore
        from backend.orchestrator import Orchestrator
        assert not hasattr(NewsFeed, "pick")
        assert not hasattr(NewsFeed, "is_used")
        assert not hasattr(NewsFeed, "reset_used")
        assert not hasattr(WebSearchClient, "clear_cache")
        assert not hasattr(MemoryStore, "by_viewer")
        assert not hasattr(MindStore, "list_narratives")
        assert not hasattr(MindStore, "search_opinions")
        assert not hasattr(MemoryConsolidator, "last_attempt")
        assert not hasattr(MemoryConsolidator, "last_success")
        assert not hasattr(Orchestrator, "_prefetch_passes_recheck")
        assert not hasattr(Orchestrator, "_recall")
        # 現行APIは残っている
        assert hasattr(NewsFeed, "unused_items")
        assert hasattr(NewsFeed, "mark_used")
        assert hasattr(MindStore, "latest_narrative")
        assert hasattr(MindStore, "list_opinions")
