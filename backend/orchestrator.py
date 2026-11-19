"""Orchestrator: 配信の進行役（メインの asyncio イベントループ制御）

キャラクター1体ぶんの LLM・TTS・記憶・会話ログは CharacterAgent（backend/ensemble）が持つ。
ここが受け持つのは「舞台」の側だけ:

  - コメントの取得・選別・キュー
  - 誰が答えるか／誰が話題を振るか（TurnDirector）と、キャラ同士の掛け合い
  - 発話の排他（同時に喋るのは常に1体）とTTSの直列化・ハング対策
  - 知識の仕入れ・BGM・音声入力・状態配信

出演キャラが1体なら、従来の1体運用とまったく同じに動く。
"""
from __future__ import annotations

import asyncio
import dataclasses
import difflib
import itertools
import random
import re
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from backend import factory
from backend.avatar.actions import dispatch_actions, extract_actions
from backend.avatar.ws_hub import WebSocketHub
from backend.bgm.bgm_player import BGMPlayer
from backend.effects import (EffectsDispatcher, ThoughtsDispatcher,
                             catalog_summary, resolve_effects)
from backend.config import AppConfig
from backend.ensemble import CharacterAgent, TurnDirector
from backend.knowledge.models import WebDocument
from backend.knowledge.news import NewsFeed
from backend.knowledge.topic_planner import Topic, TopicPlanner
from backend.knowledge.web_search import WebSearchClient
from backend.llm import ollama_service
from backend.llm.base import LLMBackend
from backend.llm.prompt_builder import (
    build_answer_pending_prompt,
    build_banter_prompt,
    build_idle_prompt,
    build_research_prompt,
    build_system_prompt,
    build_topic_prompt,
    _speech_limit,
)
from backend.llm.streaming import SpeechStreamParser, TagSpeechStreamParser
from backend.llm.parser import (TAG_OUTCOME_FATAL, TAG_OUTCOME_NO_SPEECH,
                                contains_meta_leak,
                                dedupe_repeated_sentences,
                                parse_outcome_stats,
                                strip_greeting_template,
                                tag_parse_outcome_stats)
from backend.logger import get_logger
from backend.memory.reflection_engine import ReflectionEngine
from backend.mind.mind_loop import (MindLoop, find_relevant_opinions,
                                    format_opinion_context,
                                    intention_context_for, push_curiosity)
from backend.mind.sleep_cycle import SleepManager
from backend.memory.stream_archive import StreamArchiver
from backend.memory.elyth_import import ElythEpisodeImporter
from backend.models import Comment, LLMResponse, SystemStatus, TTSResult
from backend.safety.response_filter import ResponseFilter
from backend.selection.comment_selector import CommentSelector
from backend.selection.comment_triage import CommentTriage
from backend.sources.base import CommentSource
from backend.sources.local_chat_source import LocalChatSource
from backend.sources.video_id import extract_youtube_id
from backend.stt.base import STTBackend
from backend.tts.audio_router import AudioRouter
from backend.tts.base import TTSBackend

logger = get_logger(__name__)

# 制御タグの断片（生成崩れの兆候）。発話文に混ざっていたら読み上げない
_TAG_FRAGMENT_RE = re.compile(r"<\s*(emotion|motion|speech)\s*:|<\|[^>]*\|>")


@dataclasses.dataclass
class _StreamSpeakResult:
    """ストリーミング発話の結果"""
    response: LLMResponse | None = None
    spoke: bool = False                  # 1文でも実際に発話したか
    blocked: str = ""                    # 応答後フィルタで中断した理由
    spoken: list[str] = dataclasses.field(default_factory=list)  # 実際に喋った文
    tag_no_speech: bool = False          # タグ方式で本文なしに終わったか
    gen_meta: dict = dataclasses.field(default_factory=dict)  # 生成メタ（done_reason等）


@dataclasses.dataclass(order=True)
class _QueueItem:
    priority: float
    seq: int  # 同点優先度のFIFO保証用（投入順）
    comment: Comment = dataclasses.field(compare=False)
    enqueued_at: float = dataclasses.field(default=0.0, compare=False)


class Orchestrator:
    """
    AIVtuberのメイン制御クラス。
    - comment_ingest_loop: コメント取得 → キューへ
    - response_loop: キューから取り出し → 担当キャラを決めて LLM → TTS → WebSocket
    - idle_loop: 一定時間無発話なら独り言（掛け合いなら相方が受ける）
    - health_check_loop: バックエンド死活監視
    """

    _TTS_FAIL_LIMIT = 2       # この回数連続でTTSが応答しなければ字幕のみモードへ
    _TTS_COOLDOWN_SEC = 60.0  # 字幕のみモードを維持する時間
    # 序盤TTS保護（P3）: セッション開始直後は合成が冷えていて初回が
    # タイムアウトしやすい（2026-11-17 再実施で開始直後2件が無音化）。
    # 開始からこの秒数の間は、タイムアウトを長めに取り1回だけ再試行する
    _TTS_EARLY_WINDOW_SEC = 180.0   # 序盤とみなす時間（秒）
    _TTS_EARLY_TIMEOUT_SEC = 90.0   # 序盤の合成タイムアウト（通常より長く）
    # エンジン開始時の声ウォームアップ（P3）: 声ごとの初回合成を先に済ませる
    _TTS_WARMUP_LINE = "あー、テスト。"
    _TTS_WARMUP_TIMEOUT_SEC = 120.0
    # 反復ブロック回復フロー（Phase 2）の定数
    _BLOCK_RESET_THRESHOLD = 6     # 合算の連続ブロックがこの回数で文脈リセットへ
    _SPEAKER_COOLDOWN_SEC = 180.0  # 話者交代時にブロック側を選ばない時間
    # 反復ブロックの段階化（P1-B）: 短い汎用句（相づち・感謝など）は
    # 単発の再演では差し止めず、同一句が短時間に繰り返された場合だけ
    # ブロック対象にする。汎用句のブロックが回復サイクルを消費して
    # 無音を広げるのを防ぐ（2026-11-17 再実施: ブロック19回の多くが短句）
    _SHORT_PHRASE_MAX_LEN = 15        # 正規化後この字数以下を「短い汎用句」とする
    _SHORT_PHRASE_WINDOW_SEC = 120.0  # 同一句の出現を数える時間窓（秒）
    _SHORT_PHRASE_BLOCK_COUNT = 3     # 窓内でこの回数以上 出たらブロック対象
    _BLOCKED_FORMULAIC = "formulaic opening"  # 文型反復の差し止め理由（ブロック扱いしない）
    # 定型相づちのクールダウン（P2-A）: Character2が再演しやすい
    # 「…どういうことなの？わたしにも教えて？」型の文型。正規化テキストへ
    # の正規表現で検出し、1回出たらしばらく再演を差し止める
    _AIZUCHI_PATTERNS: tuple[tuple[str, "re.Pattern[str]"], ...] = (
        ("douiu_koto", re.compile(r"どういうことなの.{0,10}教えて")),
        # 一人称はペルソナ補正で書き換わり得る（わたし→わらわ等）ため
        # 人称を含まない形で拾う
        ("watashi_mo", re.compile(r"にも教えて")),
        ("douiu_kaka", re.compile(r"どういうことか.{0,10}教えて")),
        # 文単位で届く場合は前半だけでも拾う（「それはどういうことなの？」）
        ("douiu_stem", re.compile(r"どういうことなの")),
    )
    _AIZUCHI_COOLDOWN_SEC = 600.0  # 定型相づちを差し止める時間（秒）
    # Phase 4: 回復リトライ1回目だけ生成温度を少し上げて再生成条件を変える
    _RECOVERY_TEMP_RAISE = 0.1     # 温度の上乗せ幅（例: 0.8 → 0.9）
    # 救済時のプロンプト縮小（P1-D）: 回復リトライでは会話文脈を直近
    # この件数に切り詰めて再生成し、生成時間そのものを縮める
    _COMPACT_RETRY_MESSAGES = 4
    # 先行生成（P1-C）: 先行生成した発話を使い回せる時間（秒）
    _PREFETCH_TTL_SEC = 120.0
    _RECOVERY_TEMP_MAX = 1.2       # 上乗せ後の上限
    _TEMP_OVERRIDE_TTL_SEC = 120.0  # 温度上書きが消費されなかった場合の失効
    # no-speech 対策（2026-11-19）: ヘッダのみで終わる生成への再試行は
    # 同一条件にせず温度を下げ、回復時の温度上昇は no-speech 系の失敗
    # では使わない。idle 1ターンあたりの生成回数にも上限を設ける
    _NOSPEECH_TEMP_DROP = 0.3      # フォールバック再試行での温度の下げ幅
    _NOSPEECH_TEMP_FLOOR = 0.2     # 下げた温度の下限
    _IDLE_TURN_GEN_CAP = 6         # idle 1ターンあたりの生成回数上限
    _NOSPEECH_TURN_LIMIT = 2       # idle 1ターンで許す no-speech 系失敗数
    # Phase 4: failback はプローブの連続成功がこの回数に達したら行う
    _FAILBACK_PROBE_SUCCESSES = 2

    def __init__(self, config: AppConfig, hub: WebSocketHub):
        self._cfg = config
        self._hub = hub
        ollama_service.configure(config.ollama)
        self._status = SystemStatus()
        self._start_time = time.time()
        # 配信アーカイバ（全発話の記録と終了時のエピソード化）
        try:
            self._archiver: StreamArchiver | None = StreamArchiver(
                Path(__file__).resolve().parent.parent / "data" / "streams")
        except Exception:
            self._archiver = None

        # ELYTHエピソード取り込み（Driveの日次エピソード→対象キャラの記憶。
        # 台帳は PC ローカルの data/elyth/imported.json。無効・未設定なら None）
        self._elyth_importer: ElythEpisodeImporter | None = None
        try:
            _elyth_cfg = config.elyth_memory
            if _elyth_cfg.enabled and _elyth_cfg.episodes_dir:
                self._elyth_importer = ElythEpisodeImporter(
                    _elyth_cfg.episodes_dir,
                    Path(__file__).resolve().parent.parent
                    / "data" / "elyth" / "imported.json",
                    character=_elyth_cfg.character)
        except Exception:
            self._elyth_importer = None

        # コメントキュー（高優先度を先に処理）
        self._comment_queue: asyncio.PriorityQueue[_QueueItem] = asyncio.PriorityQueue()
        self._queue_seq = itertools.count()   # 同点優先度のFIFO保証

        # コメントソース
        self._source: CommentSource | None = None
        self._source_fallback: CommentSource | None = None
        # ローカル1対1チャットモードのときだけ入る（それ以外は None）
        self._local_chat: LocalChatSource | None = None

        # 応答後フィルタ（NGワード・個人情報の配信前チェック）。全キャラ共通
        self._filter = ResponseFilter(
            ng_words=config.safety.ng_words,
            enabled=config.safety.enabled,
        )

        # 出演キャラクター（先頭がリーダー）と、話者の決め方
        self._agents: list[CharacterAgent] = []
        self._director = TurnDirector(config.ensemble)
        self._build_cast()

        self._audio = AudioRouter(device=config.audio_output.device)
        self._audio_cache = Path("audio_cache")
        self._audio_cache.mkdir(exist_ok=True)
        self._bgm = BGMPlayer()
        # 演出エフェクト（カットイン・SE・フリップ）の配送係。
        # 音声は TTS と同じ AudioRouter を、その時点の self._audio で引く
        # （テスト等で _audio を差し替えても追随する）
        self._effects = EffectsDispatcher(
            config, hub, lambda wav: self._audio.play(wav))
        # 内心（thought）の表示配送係。TTS には流さず画面だけに出す
        self._thoughts = ThoughtsDispatcher(config, hub)

        self._last_speech_time = time.time()
        self._is_speaking = False
        self._speaking_character = ""
        self._tasks: list[asyncio.Task] = []
        self._running = False
        # 停止後の整理処理（背景タスク。Phase 4.5 で stop() から分離）
        self._post_stop_task: asyncio.Task | None = None

        # 発話排他制御（掛け合いでも同時に喋るのは1体だけ）
        self._speak_lock = asyncio.Lock()
        # TTS合成の直列化。ローカルTTSサーバは同時リクエストで詰まりやすいため、
        # 先読み合成は行いつつ実際のリクエストは1本ずつ流す（キャラが違っても同じサーバ）
        self._tts_lock = asyncio.Lock()
        self._tts_timeouts = 0        # 連続タイムアウト回数（復帰したら0に戻す）
        # 序盤TTS保護の起点（monotonic）。start() で設定されるまでは
        # None（＝序盤扱いしない。単体テストや未開始状態の挙動を変えない）
        self._session_start_mono: float | None = None
        self._tts_skip_until = 0.0    # この時刻までは合成を試みない（字幕のみモード）
        self._tts_degraded_turn = False  # この発話ではもうTTSを待たない
        # 即時フェイルオーバーの最終試行時刻（キャラkey→monotonic時刻）。
        # 短時間の往復（flapping）防止のクールダウンに使う
        self._immediate_failover_at: dict[str, float] = {}

        # 音声認識（マイク入力）。設定で無効なら生成しない
        self._stt: STTBackend | None = factory.build_stt(config)
        self._consolidation_interval_min: int = 0  # 0=無効、N=N分ごとに自動実行
        self._consolidating: set[str] = set()      # 記憶整理中のキャラ（二重起動防止）
        self._reflection_engine = ReflectionEngine(
            max_diary_entries=config.orchestrator.max_diary_entries)

        # 知識の仕入れ（Web検索・ニュース）と自律トークの話題決め
        self._search: WebSearchClient | None = None
        self._news: NewsFeed | None = None
        self._planner: TopicPlanner | None = None
        self._last_web_topic_time = 0.0   # 直近でWebネタを話した時刻
        self._last_search_time = 0.0      # 直近でコメント起点の検索をした時刻
        self._recent_speech_norms: list[str] = []  # 直近発話の正規化テキスト（反復ループ断ち用）
        # 短い汎用句の出現履歴（P1-B の段階判定用。(時刻, 正規化文)）
        self._short_phrase_events: list[tuple[float, str]] = []
        # テーマ反復の検知（Phase 4.6）: 直近発話のテーマバイグラム（両キャラ
        # 合算）と、次回発話へ1回だけ付ける話題切り替えヒントの状態
        self._recent_theme_grams: list[set[str]] = []
        self._theme_hint_pending: str = ""
        self._theme_hint_count: int = 0
        self._theme_hint_fired: dict[str, float] = {}  # テーマ名 → 最後にヒントを出した時刻
        self._theme_hint_gram: dict[str, str] = {}  # テーマ名 → 代表gram（テーマban用）
        self._force_switch_pending: bool = False  # 次回idleの話題を強制切替（案C）
        # 反復ブロックの回復フロー（Phase 2）: 連続ブロック数・話者クールダウン・
        # 回復ヒント・フォールバック台詞のローテーション位置
        self._block_streak: dict[str, int] = {}
        self._total_block_streak: int = 0
        self._speaker_cooldown_until: dict[str, float] = {}
        self._recovery_hint_pending: str = ""  # 回復ヒントを付けるキャラkey（""=なし）
        # 定型相づちのクールダウン（P2-A）: パターンID → 差し止め終了時刻
        self._aizuchi_cooldown_until: dict[str, float] = {}
        # 先行生成（P1-C）: 生成済みの次の自律発話と、その生成タスク
        self._prefetched: dict | None = None
        self._prefetch_task: asyncio.Task | None = None
        self._fallback_line_cursor: dict[str, int] = {}
        # テーマのクラスタ化（Phase 4・案D）: 表示ラベルが割れても代表gram
        # 単位で発火・回数を数え、別ラベルでも同一テーマとしてエスカレーション
        # させるための状態。_theme_hint_fired（ラベル単位）は従来どおり併用する
        self._theme_gram_fired: dict[str, float] = {}      # gram → 最終発火時刻
        self._theme_gram_fire_count: dict[str, int] = {}   # gram → 累計発火回数
        self._theme_hint_consumed_gram: str = ""  # ヒント消費直後の検証用（gram）
        # 回復リトライ1回目の温度上書き（Phase 4）: キャラkey → (温度, 失効時刻)
        self._temp_override: dict[str, tuple[float, float]] = {}
        # no-speech 対策（2026-11-19）: idle ターン内の生成予算（None=無制限）
        self._gen_budget: int | None = None
        # failback 用プローブの連続成功回数（キャラkey → 回数。Phase 4）
        self._failback_probe_success: dict[str, int] = {}
        self._saturation_events: list[tuple[float, str]] = []  # (時刻, 語) 吸着の発火履歴
        self._last_context_reset_at: float = 0.0
        self._materialless_idle_count: int = 0  # 材料なし自律トークの連続回数
        self._build_knowledge()

        # 「個」を育てる自律思考ループ（Phase 2）。状態は duck typing で読ませる
        self._mind_loop = MindLoop(self)

        # 睡眠時統合のスケジューラ（Phase 3）。アプリ常駐で回すため、
        # 起動は main の lifespan から行う（run_sleep_scheduler）
        self._sleep_manager = SleepManager(self)

    # ── 出演キャラクター ──────────────────────────────────────

    def _build_cast(self) -> None:
        """config の characters から出演者を組み立てる"""
        for agent in self._agents:
            agent.close()
        cast = self._cfg.cast
        self._agents = [
            CharacterAgent(self._cfg, ch, is_lead=(i == 0), response_filter=self._filter)
            for i, ch in enumerate(cast)
        ]
        self._refresh_cast_links()

    def _refresh_cast_links(self) -> None:
        """共演者の紹介とコメント選別の呼びかけ名を出演者に合わせる"""
        cast = [a.cfg for a in self._agents]
        for agent in self._agents:
            agent.set_partners(cast)
        names = [n for a in self._agents for n in a.cfg.call_names()]
        self._selector = CommentSelector(self._cfg.orchestrator, names)
        self._triage = CommentTriage(
            self._cfg.comment_filter, names,
            known_author=self._is_known_viewer)

    async def reload_cast(self) -> list[dict]:
        """
        config.yaml の characters を変えたあとに出演者を組み直す。
        発話の途中で差し替えないよう、発話ロックを取ってから行う。
        """
        async with self._speak_lock:
            for agent in self._agents:
                if agent.llm and agent.log.get_pending():
                    await self._consolidate_agent(agent, triggered_by="cast_reload")
            old_llms = [b for a in self._agents for b in (a.llm, a.llm_fallback) if b]
            self._build_cast()
            for b in old_llms:
                try:
                    await b.aclose()
                except Exception:
                    pass
            await self.rebuild_llm()
            self.rebuild_tts()
            self._restore_chat_context()
        await self._hub.send_log("info", "出演者: " + " / ".join(a.name for a in self._agents))
        return self.characters_status()

    @property
    def agents(self) -> list[CharacterAgent]:
        return list(self._agents)

    @property
    def lead(self) -> CharacterAgent:
        return self._agents[0]

    @property
    def _lead(self) -> CharacterAgent:
        return self._agents[0]

    @property
    def is_ensemble(self) -> bool:
        """2体以上で掛け合い中か"""
        return len(self._agents) > 1

    def get_agent(self, key: str | None) -> CharacterAgent | None:
        """id か名前でキャラを引く。空ならリーダー"""
        if not key:
            return self._lead
        for a in self._agents:
            if key in (a.key, a.name):
                return a
        return None

    def characters_status(self) -> list[dict]:
        return [{**a.status(), "speaking": a.key == self._speaking_character}
                for a in self._agents]

    # ── リーダーへの近道（1体運用の頃の属性名。ルート・テストが使う）────

    def _lead_attr(name: str, doc: str):  # type: ignore[misc]
        def getter(self):
            return getattr(self._agents[0], name)

        def setter(self, value):
            setattr(self._agents[0], name, value)
        return property(getter, setter, doc=doc)

    _llm = _lead_attr("llm", "リーダーのLLM")
    _llm_fallback = _lead_attr("llm_fallback", "リーダーのフォールバックLLM")
    _tts = _lead_attr("tts", "リーダーのTTS")
    _log = _lead_attr("log", "リーダーの会話ログ")
    _long_term_memory = _lead_attr("long_term_memory", "リーダーのコア記憶")
    _memory_store = _lead_attr("memory_store", "リーダーのエピソード記憶")
    _viewer_store = _lead_attr("viewer_store", "リーダーの視聴者プロファイル")
    _chat_history = _lead_attr("chat_history", "リーダーのチャット履歴")
    _rag = _lead_attr("rag", "リーダーのコア記憶検索")
    _episodic_rag = _lead_attr("episodic_rag", "リーダーのエピソード記憶検索")
    _consolidator = _lead_attr("consolidator", "リーダーの記憶整理器")
    _persona_validator = _lead_attr("persona_validator", "リーダーの人格補正")
    del _lead_attr

    @property
    def persona_model_active(self) -> bool:
        """リーダーが人格QLoRAの専用モデルで応答しているか（フォールバック中は False）"""
        return self._lead.persona_model_active

    @property
    def slim_persona_prompt(self) -> bool:
        return self._lead.slim_persona_prompt

    async def switch_character(self, name: str) -> None:
        """
        リーダーを別のキャラクターに切り替える。記憶系を一括で差し替え、
        前のキャラの会話が新しいキャラに引き継がれないよう履歴も畳む。
        """
        lead = self._lead
        if lead.llm and lead.log.get_pending():
            await self._consolidate_agent(lead, triggered_by="character_switch")
        lead.cfg.name = name
        lead.log.clear()
        lead.load_memory()
        self._refresh_cast_links()
        # 切替先のキャラクターの過去ログを文脈に読み戻す（1対1チャット時のみ）
        self._restore_chat_context()
        await self._hub.send_log("info", f"キャラクター切替: {name}")

    # ── ローカル1対1チャットモード ────────────────────────────

    def _restore_chat_context(self) -> int:
        """
        保存済みのチャット履歴を、LLMに渡す文脈として読み戻す。

        これが無いと、再起動直後は「さっき話したこと」が通じない。
        エピソード記憶（要約された事実）とは別に、直前の会話そのものを
        数ターンぶんだけ戻すことで会話が地続きになる。
        """
        cfg = self._cfg.local_chat
        if not (self.is_local_chat and cfg.persist_history):
            return 0
        return sum(a.restore_chat_context(cfg.restore_context_turns) for a in self._agents)

    def _record_chat(self, role: str, text: str, author: str = "", emotion: str = "",
                     agent: CharacterAgent | None = None) -> None:
        """
        ローカルチャットの発言を保存する（他モードでは何もしない）。

        各キャラは自分視点の履歴を持つ。ユーザーの発言は全員に、
        キャラの発言は本人には assistant、相方には partner として残す。
        """
        if not (self.is_local_chat and self._cfg.local_chat.persist_history):
            return
        if role == "user" or agent is None:
            for a in self._agents:
                a.record_chat(role, text, author=author, emotion=emotion)
            return
        agent.record_chat("assistant", text, emotion=emotion)
        for other in self._agents:
            if other is not agent:
                other.record_chat("partner", text, author=agent.name, emotion=emotion)

    def chat_history(self, limit: int = 100) -> list[dict]:
        """チャット画面に表示する過去ログ（古い順・リーダー視点）"""
        lead = self._lead
        if lead.chat_history is None:
            return []
        items = lead.chat_history.recent(limit)
        for it in items:
            if it.get("role") == "assistant":
                it.setdefault("author", lead.name)
        return items

    def clear_chat_history(self) -> int:
        """過去ログを消す（エピソード記憶は消さない）。戻り値はリーダーの削除件数"""
        removed = 0
        for a in self._agents:
            if a.chat_history is not None:
                n = a.chat_history.clear()
                if a.is_lead:
                    removed = n
        return removed

    def chat_memory_status(self) -> dict:
        """チャット画面に出す記憶まわりの状態"""
        lead = self._lead
        return {
            "persist_history": self._cfg.local_chat.persist_history,
            "history_count": lead.chat_history.count() if lead.chat_history else 0,
            "history_path": str(lead.chat_history.path) if lead.chat_history else "",
            "episodic_count": lead.memory_store.count() if lead.memory_store else 0,
            "turns_since_consolidation": lead.log.turns_since_consolidation,
            "consolidate_every_turns": self._cfg.local_chat.consolidate_every_turns,
        }

    @property
    def is_local_chat(self) -> bool:
        """いまローカル1対1チャットモードで動いているか"""
        return self._local_chat is not None

    @property
    def local_user_name(self) -> str:
        return self._cfg.local_chat.user_name

    async def send_local_message(self, text: str, author: str | None = None) -> dict:
        """
        ローカルユーザーの発言を受け取り、応答キューへ流す。

        未起動なら（local_chat.autostart が真のとき）ローカルモードで自動起動する。
        別モードで動いている場合は切り替えずに断る——配信中に勝手にソースを
        差し替えると、視聴者コメントの取得が止まってしまうため。
        """
        text = (text or "").strip()
        if not text:
            return {"ok": False, "error": "メッセージが空です"}

        if not self._running:
            if not self._cfg.local_chat.autostart:
                return {"ok": False, "error": "ローカルチャットモードを開始してください"}
            await self.start("local")
        elif not self.is_local_chat:
            return {"ok": False,
                    "error": f"現在は別モード（{self._status.video_id}）で動作中です。"
                             "停止してからローカルチャットを開始してください"}

        assert self._local_chat is not None
        comment = await self._local_chat.push(text, author=author)
        if comment is None:
            return {"ok": False, "error": "発言を受け付けられませんでした（混雑中）"}

        self._record_chat("user", comment.message, author=comment.author)
        await self._hub.send_chat("user", comment.message, author=comment.author)
        return {"ok": True, "author": comment.author, "message": comment.message}

    # ── 知識の仕入れ（Web検索・ニュース）────────────────────────

    def _build_knowledge(self) -> None:
        """config.yaml の knowledge セクションから検索・ニュース・話題決めを組み立てる"""
        k = self._cfg.knowledge
        if not k.enabled:
            self._search = self._news = self._planner = None
            logger.info("Knowledge disabled by config")
            return

        self._search = WebSearchClient(
            provider=k.search.provider,
            api_key_env=k.search.api_key_env,
            cse_id_env=k.search.cse_id_env,
            brave_api_key_env=k.search.brave_api_key_env,
            max_results=k.search.max_results,
            timeout_sec=k.search.timeout_sec,
            cache_ttl_sec=k.search.cache_ttl_sec,
            blocked_domains=k.blocked_domains,
            language=k.search.language,
            region=k.search.region,
            min_request_interval_sec=k.search.min_request_interval_sec,
            cooldown_base_sec=k.search.cooldown_base_sec,
            cooldown_max_sec=k.search.cooldown_max_sec,
        )
        self._news = NewsFeed(
            feeds=k.news.feeds,
            max_items=k.news.max_items,
            max_age_hours=k.news.max_age_hours,
            timeout_sec=k.search.timeout_sec,
            blocked_domains=k.blocked_domains,
            keywords=k.news.keywords,
            retry_base_sec=k.news.retry_base_sec,
        ) if k.news.enabled else None

        self._planner = TopicPlanner(
            news=self._news,
            search=self._search,
            keyword_provider=self._topic_keywords,
            memory_provider=self._memory_topic_seed,
            theme_provider=lambda: self._cfg.character.stream_theme or "",
            weights=k.autonomous_topic.weights,
            max_documents=k.autonomous_topic.max_documents,
            intention_provider=self._intention_seeds,
            memory_cap_ratio=k.autonomous_topic.memory_share_cap,
            memory_cap_window=k.autonomous_topic.memory_share_window,
            filler_enabled=k.autonomous_topic.filler_enabled,
            planner_enabled=k.autonomous_topic.planner_enabled,
            arc_max_speeches=k.autonomous_topic.arc_max_speeches,
            arc_max_continuations=k.autonomous_topic.arc_max_continuations,
            arc_max_sec=k.autonomous_topic.arc_max_sec,
            arc_ask_probability=k.autonomous_topic.arc_ask_probability,
        )
        logger.info("Knowledge enabled",
                    search=self._search.available_providers(),
                    news_feeds=len(self._news.feeds) if self._news else 0)

    def _topic_keywords(self) -> list[str]:
        """検索クエリの種にする、出演者全員の記憶由来のキーワード"""
        keywords: list[str] = []
        for agent in self._agents:
            keywords.extend(agent.topic_keywords())
        return list(dict.fromkeys(keywords))

    # 思い出話題の種にしない記憶（挨拶・定型文系は会話の核にならない）
    _MEMORY_SEED_NG = re.compile(r"元気な挨拶|定型的なやり取り|自己紹介のテーマ|挨拶を交わ|元気なことをアピール")

    def _memory_topic_seed(self) -> dict | None:
        """自律トークの「思い出」話題用に、記憶から1件選ぶ。

        配信エピソード（stream_episode）を優先しつつ、全員の
        MemoryStore の最近の記憶からランダムに選ぶ。
        2026-11-14の劣化対策：直近30分で使った種の再利用と、
        挨拶・定型文系エピソードの種化を禁止する。
        """
        try:
            now = time.time()
            used = getattr(self, "_recent_memory_seed_keys", None)
            if used is None:
                used = self._recent_memory_seed_keys = {}
            for k in [k for k, at in used.items() if now - at > 1800]:
                del used[k]
            pool = []
            for agent in self._agents:
                if agent.memory_store is not None:
                    # プールを広げて偏りを薄める（Phase 4・案E: 20 → 50）
                    pool.extend(agent.memory_store.sample_recent(limit=50))
            banned = set()
            if self._planner is not None:
                try:
                    banned = self._planner.banned_words()
                except Exception:
                    banned = set()
            pool = [e for e in pool
                    if len((e.text or "").strip()) >= 10
                    and (e.text or "").strip()[:80] not in used
                    and not self._MEMORY_SEED_NG.search(e.text or "")
                    and not any(w in (e.text or "") for w in banned)]
            if not pool:
                return None
            # 同一セッション由来がプールを支配しないよう上限を設け、
            # 支配テーマを含むエピソードは抽選重みを下げて選ぶ（案E）。
            # 一様ランダムのままだと、偏ったプールの偏りがそのまま
            # 次の話題の偏りとして再生産されるため。
            pool = self._cap_seed_pool_by_session(pool)
            weights = self._seed_pool_weights(pool)
            eps_idx = [i for i, e in enumerate(pool)
                       if e.source == "stream_episode"]
            if eps_idx and random.random() < 0.7:
                entry = random.choices(
                    [pool[i] for i in eps_idx],
                    weights=[weights[i] for i in eps_idx], k=1)[0]
            else:
                entry = random.choices(pool, weights=weights, k=1)[0]
            used[(entry.text or "").strip()[:80]] = now
            return {"text": entry.text.strip(), "source": entry.source}
        except Exception:
            return None

    @staticmethod
    def _seed_session_key(entry) -> str:
        """エピソードの由来セッションを表すキー（同一セッション上限用・案E）。

        配信エピソードは本文冒頭の日付（YYYY年M月D日）を、それ以外は
        作成日（ローカル日付）を使う。どちらも取れなければ source でまとめる。
        """
        if getattr(entry, "source", "") == "stream_episode":
            m = re.search(r"\d{4}年\d{1,2}月\d{1,2}日",
                          (getattr(entry, "text", "") or "")[:40])
            if m:
                return f"ep:{m.group(0)}"
        created = getattr(entry, "created_at", None)
        if created:
            try:
                return "day:" + time.strftime("%Y-%m-%d",
                                              time.localtime(created))
            except Exception:
                pass
        return f"src:{getattr(entry, 'source', '')}"

    def _cap_seed_pool_by_session(self, pool: list) -> list:
        """同一セッション由来がプールの1/3を超えないよう間引く（案E）"""
        if len(pool) < 3:
            return pool
        cap = max(1, len(pool) // 3)
        counts: dict[str, int] = {}
        kept: list = []
        for e in pool:
            key = self._seed_session_key(e)
            if counts.get(key, 0) >= cap:
                continue
            counts[key] = counts.get(key, 0) + 1
            kept.append(e)
        return kept or pool

    def _seed_pool_weights(self, pool: list) -> list[float]:
        """支配 gram を含むエピソードの抽選重みを下げる（案E）。

        プールの過半のエピソードに現れるテーマ gram を「支配 gram」とし、
        それを含むエピソードの重みを 0.25 にする。プール自体の偏りを
        抽選で増幅しないための補正で、除外まではしない（材料枯渇の回避）。
        """
        weights = [1.0] * len(pool)
        if len(pool) < 4:
            return weights
        gram_sets = [self._theme_grams(e.text or "") for e in pool]
        from collections import Counter
        coverage: Counter = Counter()
        for gs in gram_sets:
            for g in gs:
                coverage[g] += 1
        dominant = {g for g, n in coverage.items() if n * 2 > len(pool)}
        if not dominant:
            return weights
        for i, gs in enumerate(gram_sets):
            if gs & dominant:
                weights[i] = 0.25
        return weights

    def _intention_seeds(self) -> list[dict]:
        """TopicPlanner 用: 出演者全員の active な意図を種として渡す"""
        seeds: list[dict] = []
        for agent in self._agents:
            if agent.mind_store is None:
                continue
            try:
                for it in agent.mind_store.list_intentions(limit=5):
                    seeds.append({"title": it.title,
                                  "next_action": it.next_action,
                                  "character": agent.name})
            except Exception:
                continue
        return seeds

    def mind_status(self) -> dict:
        """ダッシュボード/API 用: 自律思考ループの状態"""
        status = self._mind_loop.status() if getattr(self, "_mind_loop", None) \
            else {"enabled": False}
        status["sleep"] = self.sleep_status()
        return status

    def sleep_status(self) -> dict:
        """ダッシュボード/API 用: 睡眠時統合スケジューラの状態"""
        return self._sleep_manager.status() \
            if getattr(self, "_sleep_manager", None) else {"enabled": False}

    async def run_sleep_scheduler(self) -> None:
        """アプリ常駐の睡眠スケジューラ（main の lifespan から起動する）"""
        if getattr(self, "_sleep_manager", None) is not None:
            await self._sleep_manager.run()

    async def run_sleep_cycle(self, char: str = "") -> dict:
        """睡眠時統合を手動で1回実行する（POST /api/mind/sleep 用）"""
        if getattr(self, "_sleep_manager", None) is None:
            return {"error": "sleep manager not available"}
        return await self._sleep_manager.run_manual(char)

    def knowledge_status(self) -> dict:
        """ダッシュボード用の状態まとめ"""
        return {
            "enabled": bool(self._cfg.knowledge.enabled),
            "search": self._search.status() if self._search else None,
            "news": self._news.status() if self._news else None,
            "planner": self._planner.status() if self._planner else None,
            "parse_outcomes": parse_outcome_stats(self._cfg.llm.protocol),
            "llm_context": {
                agent.key: agent.llm.context_status()
                for agent in self._agents if agent.llm is not None
            },
            "autonomous_topic": {
                "enabled": self._cfg.knowledge.autonomous_topic.enabled,
                "weights": self._cfg.knowledge.autonomous_topic.weights,
                "min_web_interval_sec": self._cfg.knowledge.autonomous_topic.min_web_interval_sec,
                "last_web_topic": self._last_web_topic_time,
            },
        }

    async def refresh_news(self) -> list[WebDocument]:
        """ニュースを取り直してダッシュボードへ配る"""
        if self._news is None:
            return []
        items = await self._news.refresh()
        await self._hub.broadcast("dashboard", {
            "type": "news",
            "count": len(items),                       # 在庫の総数（itemsは表示用に間引く）
            "items": [d.to_dict() for d in items[:15]],
            "last_error": self._news.last_error,
        })
        if self._news.last_error:
            await self._hub.send_log("warn", f"ニュース取得エラー: {self._news.last_error}")
        return items

    async def search_web(self, query: str, limit: int | None = None) -> list[WebDocument]:
        """Web検索する（ダッシュボードからの手動検索にも使う）"""
        if self._search is None:
            return []
        docs = await self._search.search(query, limit=limit)
        if not docs and self._search.last_error:
            logger.info("web search empty", query=query[:40], error=self._search.last_error)
        return docs

    async def _knowledge_loop(self) -> None:
        """RSSを定期的に取り直して、話題の在庫を切らさないようにする"""
        while self._running:
            try:
                interval = max(60, self._cfg.knowledge.news.refresh_interval_sec)
                if self._news is not None and self._news.is_stale(interval):
                    await self.refresh_news()
                await asyncio.sleep(30.0)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("knowledge_loop error", error=str(e))
                await asyncio.sleep(60.0)

    # ── コメント起点のWeb検索 ─────────────────────────────────

    # 数値そのものを尋ねる疑問詞（疑問形であり、事実シグナルも兼ねる）
    _NUMERIC_QUESTION_WORDS = ("いくら", "何円", "何位", "何人", "何時",
                               "何分", "何日", "何年", "何月", "何％",
                               "何パーセント")

    # 「調べる」宣言のパターン（宣言と実行の一致の見張り用。Phase 3）
    _DECLARATION_RE = re.compile(
        r"調べ(る|てくる|ます|ておく)|検索(する|してくる|します)|"
        r"確認してくる|ググ(る|ってくる)|後で教える|あとで教える")

    def _classify_factual_question(self, message: str) -> tuple[bool, str]:
        """コメントが「調べて答えるべき事実質問」かを判定する（純粋な分類）。

        戻り値は (事実質問か, 理由)。判定は2段構え:
          1. trigger_words（明示的な依頼語）があれば即・事実質問
          2. 意見マーカーがあれば事実質問にしない（意見ルールと一致）
          3. 疑問形 ×（事実カテゴリ語 or 日時語 or 数値疑問詞）で拾う
        2026-11-17 の実例「昨日の日経平均株価の終値って、いくらだったか
        分かりますか？」は trigger_words を1語も含まず無回答に落ちた。
        その反省で、キーワード一致必須をやめて意味の組み合わせで拾う。
        間隔ゲートはここには含めない（同期するか保留キューへ回すかは
        呼び出し側が決める。ゲートで質問を捨てないため）。
        """
        cfg = self._cfg.knowledge.answer_with_search
        text = (message or "").strip()
        if not text:
            return False, "empty"
        for w in cfg.trigger_words:
            if w and w in text:
                return True, f"trigger:{w}"
        for w in cfg.opinion_markers:
            if w and w in text:
                return False, f"opinion:{w}"
        has_question_form = (
            "?" in text or "？" in text
            or "ますか" in text or "ですか" in text or "でしょうか" in text
            or text.rstrip("。！!").endswith("か")
            or any(w in text for w in cfg.question_words))
        if not has_question_form:
            return False, "not-question"
        for w in cfg.fact_categories:
            if w and w in text:
                return True, f"category:{w}"
        for w in cfg.date_words:
            if w and w in text:
                return True, f"date:{w}"
        for w in self._NUMERIC_QUESTION_WORDS:
            if w in text:
                return True, f"numeric:{w}"
        return False, "no-fact-signal"

    def _needs_search(self, message: str) -> bool:
        """このコメントは「調べて答える」対象か（分類のみ。間隔は見ない）"""
        cfg = self._cfg.knowledge
        if not (cfg.enabled and cfg.answer_with_search.enabled and self._search):
            return False
        text = (message or "").strip()
        if len(text) < 3:
            return False
        factual, _reason = self._classify_factual_question(text)
        return factual

    def _enqueue_pending_answer(self, comment: Comment) -> bool:
        """事実質問を保留回答キューへ積む。積めたら True"""
        if self._planner is None:
            return False
        try:
            pid = self._planner.enqueue_answer(
                comment.author, comment.message,
                query=self._build_search_query(comment.message),
                ttl_sec=float(self._cfg.knowledge.answer_with_search
                              .pending_ttl_sec))
            return bool(pid)
        except Exception as e:
            logger.warning("enqueue answer failed", error=str(e))
            return False

    async def _research_worker_loop(self) -> None:
        """保留中の事実質問を裏で調べ、答えを用意する常駐タスク（Phase 3）"""
        while self._running:
            try:
                worked = await self._research_one_pending()
                await asyncio.sleep(2.0 if worked else 5.0)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("research worker error", error=str(e))
                await asyncio.sleep(10.0)

    def _answer_query_variants(self, entry: dict) -> list[str]:
        """保留エントリの検索クエリ候補を返す。

        エントリに保存されたクエリを先頭に置き、質問文から整形した
        バリエーションを後ろに続ける（重複は除く）。ワーカーの
        再試行ごとに次の候補へ進むための並び。
        """
        variants: list[str] = []
        saved = (entry.get("query") or "").strip()
        if saved:
            variants.append(saved)
        for q in self._build_search_queries(entry.get("question") or ""):
            if q not in variants:
                variants.append(q)
        if not variants:
            q = (entry.get("question") or "").strip()
            if q:
                variants.append(q)
        return variants

    async def _research_one_pending(self) -> bool:
        """保留エントリを1件調べて、結果をキューへ反映する。処理したら True"""
        planner = self._planner
        if planner is None or self._search is None:
            return False
        entry = planner.claim_answer_work()
        if entry is None:
            return False
        cfg = self._cfg.knowledge.answer_with_search
        question = entry.get("question") or ""
        attempts = int(entry.get("attempts") or 1)   # claim で加算済み
        # 再試行では整形クエリのバリエーションを順に試す（Phase 3.5）
        variants = self._answer_query_variants(entry)
        query = variants[(attempts - 1) % len(variants)] \
            if variants else question
        try:
            docs = await self.search_web(query, limit=cfg.max_results)
        except Exception as e:
            logger.warning("pending research failed", error=str(e))
            docs = []
        if docs and self._expects_numeric_answer(question) \
                and not self._docs_have_numeric_token(docs):
            # 資料はあるが答えの数値が無い＝不十分。空振り扱いにして
            # 次のバリエーションでの再試行に回す（Phase 3.5）
            logger.info("pending research insufficient (no numeric token)",
                        question=question[:40], query=query[:40])
            docs = []
        if docs:
            planner.attach_answer_documents(entry["id"], docs)
            await self._hub.send_log(
                "info", f"質問の回答準備ができました: {(entry.get('question') or '')[:30]}")
            return True
        # 空振り：間隔を空けて再試行し、尽きたら「分からなかった」締めへ
        delays = list(cfg.answer_retry_delays_sec) or [30.0, 120.0, 300.0]
        if attempts <= len(delays):
            planner.defer_answer(entry["id"], delays[attempts - 1])
            logger.info("pending research deferred",
                        question=(entry.get("question") or "")[:40],
                        attempt=attempts,
                        delay_sec=delays[attempts - 1])
        else:
            planner.fail_answer(entry["id"])
        return True

    # ── 検索クエリの整形と資料の十分性判定（Phase 3.5）──────
    # 質問文の丸投げでは「終値そのもの」を含む資料を引けなかった
    # （2026-11-17 ローカル検証）。決定的な抽出で要点だけのクエリを
    # 組み立てる：事実カテゴリ語＋JST で解決した日付を本命にし、
    # 広めのバリエーションを後ろに用意する。LLM には依存しない。

    _JST = timezone(timedelta(hours=9))
    # 相対日付（日単位）→ 今日からの差分。「一昨日」は「昨日」を
    # 含むので、走査は長い語から先に行う
    _RELATIVE_DAY_OFFSETS = {"一昨日": -2, "昨日": -1, "今日": 0,
                             "明日": 1, "明後日": 2}
    _EXPLICIT_DATE_RE = re.compile(r"\d{1,2}月\d{1,2}日|\d{4}年")
    _NUMBER_TOKEN_RE = re.compile(r"[0-9０-９][0-9０-９,，]*(?:\.[0-9０-９]+)?")
    # クエリ整形で落とす依頼・疑問・フィラー表現
    _QUERY_FILLER_RE = re.compile(
        r"調べて|検索して|ググって|教えて|くださいな?|ちょうだい|"
        r"分かりますか|わかりますか|知ってますか|でしょうか|ですか|ますか|"
        r"だったか|かな|って|[?？！!。]")

    def _resolve_date_tokens(self, text: str,
                             now: float | None = None) -> list[str]:
        """相対日付語を JST の具体表記へ解決し、クエリ用トークンを返す。

        日単位（昨日／今日／明日…）は「11月16日」のような日付へ、
        年・月単位も具体表記へ寄せる。週単位（今週／先週…）は
        語のまま残す。本文に明示日付があればそれも拾う。
        """
        cfg = self._cfg.knowledge.answer_with_search
        base = datetime.fromtimestamp(
            now if now is not None else time.time(), tz=self._JST).date()
        tokens: list[str] = []
        spans: list[tuple[int, int]] = []

        def add(token: str) -> None:
            if token and token not in tokens:
                tokens.append(token)

        def overlaps(s: int, e: int) -> bool:
            return any(s < b and a < e for a, b in spans)

        for m in self._EXPLICIT_DATE_RE.finditer(text):
            add(m.group(0))
            spans.append(m.span())
        for w in sorted(self._RELATIVE_DAY_OFFSETS, key=len, reverse=True):
            idx = text.find(w)
            if idx >= 0 and not overlaps(idx, idx + len(w)):
                d = base + timedelta(days=self._RELATIVE_DAY_OFFSETS[w])
                add(f"{d.month}月{d.day}日")
                spans.append((idx, idx + len(w)))
        if "去年" in text:
            add(f"{base.year - 1}年")
        if "今年" in text:
            add(f"{base.year}年")
        if "先月" in text:
            prev = base.replace(day=1) - timedelta(days=1)
            add(f"{prev.month}月")
        if "今月" in text:
            add(f"{base.month}月")
        handled = set(self._RELATIVE_DAY_OFFSETS) | {"去年", "今年",
                                                     "先月", "今月"}
        for w in cfg.date_words:
            if w and w not in handled and w in text:
                add(w)
        return tokens

    def _extract_category_words(self, text: str) -> list[str]:
        """本文に含まれる事実カテゴリ語を出現順に抽出する（最大4語）。

        複合語の一部として重なる場合は、内包される側を落として
        長い語を残す（例: 「発売日」の中の「発売」）。
        """
        cfg = self._cfg.knowledge.answer_with_search
        hits: list[tuple[int, int, str]] = []
        for w in cfg.fact_categories:
            if not w:
                continue
            idx = text.find(w)
            if idx >= 0:
                hits.append((idx, idx + len(w), w))
        hits.sort(key=lambda h: (h[0], -(h[1] - h[0])))
        out: list[str] = []
        spans: list[tuple[int, int]] = []
        for s, e, w in hits:
            if w in out or any(s >= a and e <= b for a, b in spans):
                continue
            out.append(w)
            spans.append((s, e))
            if len(out) >= 4:
                break
        return out

    def _clean_query_core(self, message: str) -> str:
        """カテゴリ語だけでは足りない質問向けに、本文の要点らしい残りを抜く"""
        cfg = self._cfg.knowledge.answer_with_search
        q = re.sub(r"[@＠][^\s]+", " ", message or "")
        q = self._QUERY_FILLER_RE.sub(" ", q)
        for w in list(cfg.question_words) \
                + list(self._NUMERIC_QUESTION_WORDS) + list(cfg.date_words):
            if w:
                q = q.replace(w, " ")
        q = re.sub(r"\s+", " ", q).strip()
        return q.strip("のにはがをへとや、, ")

    def _build_search_queries(self, message: str,
                              now: float | None = None) -> list[str]:
        """検索クエリのバリエーション（最大3本、先頭が本命）。

        第1＝カテゴリ語＋解決済み日付（要点に絞った本命）、
        第2＝カテゴリ語のみ（日付表記が資料と合わない場合の広め）、
        第3＝整形した本文＋日付（カテゴリで拾えない要素の回収）。
        同期検索は第1バリエーションを使い、保留キューの再試行で
        後続を順に試す。
        """
        text = (message or "").strip()
        if not text:
            return []
        variants: list[str] = []

        def add(q: str) -> None:
            q = re.sub(r"\s+", " ", q or "").strip()
            if len(q) >= 2 and q not in variants:
                variants.append(q)

        cats = self._extract_category_words(text)
        dates = self._resolve_date_tokens(text, now=now)
        if cats:
            add(" ".join(cats + dates))
            add(" ".join(cats))
        core = self._clean_query_core(text)
        if core:
            add(" ".join([core] + dates) if dates else core)
        if not variants:
            add(text)
        return variants[:3]

    def _build_search_query(self, message: str) -> str:
        """検索クエリの本命（バリエーションの先頭）"""
        queries = self._build_search_queries(message)
        return queries[0] if queries else (message or "").strip()

    def _expects_numeric_answer(self, message: str) -> bool:
        """質問が数値そのものの回答を期待しているか（数値疑問詞で判定）"""
        text = message or ""
        return any(w in text for w in self._NUMERIC_QUESTION_WORDS)

    @classmethod
    def _docs_have_numeric_token(cls, docs: list[WebDocument]) -> bool:
        """資料のタイトル／スニペットに「答えらしい数値」が含まれるか。

        単なる日付や「3時間前」のような経過表現を答えと誤認しない
        よう、3桁以上の数字か小数を含む数値トークンを条件にする。
        """
        for d in docs:
            for m in cls._NUMBER_TOKEN_RE.finditer(f"{d.title} {d.snippet}"):
                token = m.group(0)
                digits = re.sub(r"[^0-9０-９]", "", token)
                if len(digits) >= 3 or "." in token:
                    return True
        return False

    async def _research_for_comment(self, comment: Comment) -> tuple[str, list[WebDocument]]:
        """コメントの内容をWeb検索する。戻り値は (クエリ, 資料)。"""
        query = self._build_search_query(comment.message)
        self._last_search_time = time.time()
        await self._hub.send_log("info", f"検索中: {query}")
        docs = await self.search_web(
            query, limit=self._cfg.knowledge.answer_with_search.max_results
        )
        if docs:
            await self._hub.broadcast("dashboard", {
                "type": "search_result",
                "query": query,
                "items": [d.to_dict() for d in docs],
            })
        return query, docs

    # ── 自律トークの話題 ──────────────────────────────────────

    async def plan_topic(self, kind: str = "", query: str = "") -> Topic | None:
        """次に話す話題を決める。Webネタは間隔制限つき。"""
        if self._planner is None:
            return None
        cfg = self._cfg.knowledge.autonomous_topic
        forced = bool(kind or query)
        allow_web = forced or (
            cfg.enabled
            and (time.time() - self._last_web_topic_time) >= cfg.min_web_interval_sec
        )
        context = {"silence_sec": time.time() - self._last_speech_time}
        topic = await self._planner.plan(kind=kind, query=query, allow_web=allow_web,
                                         context=context)
        # 間隔の消費は「実際に発話できたとき」だけにする（_speak_idle 側）。
        # ここで消費すると、生成に失敗した回がニュース間隔を食い潰してしまう。
        return topic

    def _remember_topic(self, topic: Topic, agent: CharacterAgent | None = None) -> None:
        """話したWebネタを、話したキャラのエピソード記憶に残す（config で無効化できる）"""
        if not self._cfg.knowledge.autonomous_topic.remember:
            return
        store = (agent or self._lead).memory_store
        if store is None or not topic.has_material:
            return
        doc = topic.documents[0]
        label = "ニュース" if topic.kind == "news" else "調べたこと"
        site = f"（{doc.site}）" if doc.site else ""
        try:
            store.add(f"{label}: {doc.title}{site}", source="web", importance=0.8)
        except Exception as e:
            logger.warning("failed to remember topic", error=str(e))

    # ── Unityアクション ───────────────────────────────────────

    @staticmethod
    def _extract_unity_actions(text: str) -> tuple[str, list[dict]]:
        return extract_actions(text)

    async def _handle_unity_actions(self, actions: list[dict]) -> None:
        await dispatch_actions(self._hub, actions)

    # ── 部品の組み立て（実体は backend/factory.py）───────────────

    def _build_sources(self, video_id: str):
        primary, fallback = factory.build_sources(self._cfg, video_id)
        self._local_chat = primary if isinstance(primary, LocalChatSource) else None
        return primary, fallback

    async def rebuild_llm(self) -> str:
        """
        設定変更後に全キャラのLLMを作り直す。戻り値はリーダーのprimaryの名前。
        """
        for agent in self._agents:
            await agent.rebuild_llm()
        self._status.llm_backend = self._lead.llm.name if self._lead.llm else "none"
        self.warmup_llm()
        return self._status.llm_backend

    def warmup_llm(self) -> list[asyncio.Task]:
        """
        応答に使うLLMがOllamaなら、サーバを起動してモデルをVRAMへ読み込んでおく。
        ロードには数十秒かかるので裏で走らせる（その間に来た発話はOllama側で待たされる）。
        掛け合いで2体が同じモデルなら1回だけ読み込む。
        """
        if not self._cfg.ollama.preload_on_start:
            return []
        tasks: list[asyncio.Task] = []
        seen: set[tuple[str, str]] = set()
        for agent in self._agents:
            llm = agent.llm
            warmup = getattr(llm, "warmup", None)
            if warmup is None:
                continue
            cfg = getattr(llm, "_cfg", None)
            ident = (getattr(cfg, "base_url", ""), getattr(cfg, "model", ""))
            if ident in seen:
                continue
            seen.add(ident)
            tasks.append(asyncio.create_task(self._warmup_one(llm, warmup),
                                             name=f"llm_warmup_{agent.key}"))
        self._warmup_tasks = tasks
        return tasks

    async def _warmup_one(self, llm: LLMBackend, warmup) -> None:
        model = getattr(getattr(llm, "_cfg", None), "model", "")
        await self._hub.send_log("info", f"Ollama: {model} を読み込み中…")
        if await warmup():
            await self._hub.send_log("info", f"Ollama: {model} の準備ができました")
        else:
            err = ollama_service.get_service(llm._cfg.base_url).last_error
            await self._hub.send_log("warn", f"Ollama の準備に失敗: {err}")

    def rebuild_tts(self) -> str:
        """全キャラのTTS（声）を作り直す。戻り値はリーダーのTTS名"""
        for agent in self._agents:
            agent.rebuild_tts()
        self._status.tts_backend = self._lead.tts.name
        self.warmup_tts()
        return self._status.tts_backend

    def warmup_tts(self) -> asyncio.Task | None:
        """
        TTSサーバ（Irodori-TTS）が起動していなければ裏で立ち上げる。
        モデルのロードに数十秒〜数分かかるので待たずに戻る（その間の発話は字幕のみで進む）。
        """
        from backend.tts import irodori_service
        service = irodori_service.get_service(self._cfg)
        if service is None:
            return None
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return None

        async def run() -> None:
            if await service.is_healthy():
                return
            await self._hub.send_log("info", "Irodori-TTS サーバを起動しています（初回はモデルの読み込みに数分かかります）…")
            if await service.ensure_running():
                self._tts_timeouts = 0
                self._tts_skip_until = 0.0
                # 1回目の合成はGPUの準備で30秒以上かかるので、短い文で先に済ませておく
                try:
                    # 開始ボタンを押す前は TTS がまだ無いので、ウォームアップ用に作る
                    tts = self._lead.tts or factory.build_tts(self._cfg, self._lead.cfg)
                    await asyncio.wait_for(tts.synthesize("あー、テスト。", "neutral"), 180)
                except Exception as e:
                    logger.info("TTS warmup synthesis failed", error=str(e))
                await self._hub.send_log("info", "Irodori-TTS サーバの準備ができました")
            else:
                await self._hub.send_log("warn", f"Irodori-TTS を起動できませんでした: {service.last_error}")

        self._tts_warmup_task = asyncio.create_task(run(), name="tts_warmup")
        return self._tts_warmup_task

    async def _warmup_tts_voices(self) -> None:
        """エンジン開始時に、全キャラの声で短いダミー文を先に合成しておく。

        Irodori-TTS はサーバが既に起動していても、声ごとの初回合成では
        参照音声の読み込みなどで時間がかかり、開始直後の実発話が
        タイムアウトして音声なしになることがある（2026-11-17 再実施で
        序盤2件が無音化）。lifespan 側の warmup_tts はサーバを新規に
        立ち上げたときしかダミー合成しないため、開始経路でも両キャラぶんを
        明示的に温める。失敗しても開始は止めず、ログのみ残す（P3-A）。
        """
        targets = [a for a in self._agents if a.tts is not None]
        if not targets:
            return

        async def warm_one(agent: CharacterAgent) -> None:
            try:
                result = await asyncio.wait_for(
                    agent.tts.synthesize(self._TTS_WARMUP_LINE, "neutral"),
                    timeout=self._TTS_WARMUP_TIMEOUT_SEC)
                if result.audio_wav:
                    logger.info("TTS voice warmup done", character=agent.name,
                                duration_ms=result.duration_ms)
                else:
                    logger.warning("TTS voice warmup returned no audio",
                                   character=agent.name)
            except Exception as e:
                logger.warning("TTS voice warmup failed (continuing)",
                               character=agent.name, error=str(e))

        await asyncio.gather(*(warm_one(a) for a in targets))

    def _in_tts_early_window(self) -> bool:
        """セッション開始直後（TTSが冷えている時間帯）かどうか（P3-B）"""
        if self._session_start_mono is None:
            return False
        return (time.monotonic() - self._session_start_mono) \
            < self._TTS_EARLY_WINDOW_SEC

    # ── 音声入力（マイクで話しかける） ────────────────────────

    async def transcribe_and_enqueue(self, pcm16: bytes, sample_rate: int = 16000) -> dict:
        """
        マイクから届いた発話区間を認識し、コメントとしてキューに投入する。
        （掛け合い中は、呼ばれた名前で担当キャラが決まる）
        """
        cfg = self._cfg.stt
        if not self._stt:
            return {"ok": False, "error": "音声認識が無効です"}

        duration_ms = int(len(pcm16) / 2 / sample_rate * 1000)
        if duration_ms < cfg.min_duration_ms:
            return {"ok": False, "error": "短すぎます", "duration_ms": duration_ms}

        result = await self._stt.transcribe(pcm16, sample_rate)
        if not result.ok:
            return {"ok": False, "error": "聞き取れませんでした", "duration_ms": duration_ms}

        comment = Comment(
            id=f"voice_{int(time.time() * 1000)}",
            author=cfg.speaker_name,
            author_channel_id="local_voice",
            message=result.text,
            timestamp=datetime.now(),
            is_moderator=True,   # オーナーの発話として扱う
        )
        await self.enqueue_comment(comment, score=cfg.priority)
        await self._hub.send_comment({
            "author": cfg.speaker_name,
            "message": result.text,
            "priority": cfg.priority,
            "is_superchat": False,
            "is_voice": True,
        })
        logger.info("Voice input enqueued", text=result.text[:40],
                    latency_ms=result.latency_ms)
        return {
            "ok": True,
            "text": result.text,
            "duration_ms": result.duration_ms,
            "latency_ms": result.latency_ms,
        }

    @property
    def tts_health(self) -> dict:
        """
        実際の合成が通っているかを返す。
        「HTTPには応答するが合成が返ってこない」ハング状態は疎通確認では分からないため、
        実行時に観測した連続タイムアウト回数を診断に混ぜる。
        """
        skipping = bool(self._tts_skip_until and time.monotonic() < self._tts_skip_until)
        tts = self._lead.tts
        return {
            "backend": tts.name if tts else "",
            "consecutive_timeouts": self._tts_timeouts,
            "subtitle_only": skipping,
            "retry_in_sec": max(0, int(self._tts_skip_until - time.monotonic())) if skipping else 0,
        }

    @property
    def stt_status(self) -> dict:
        return {
            "enabled": self._stt is not None,
            "name": self._stt.name if self._stt else "",
            "ready": self._stt.is_ready if self._stt else False,
            "speaker_name": self._cfg.stt.speaker_name,
        }

    # ── 開始・停止 ────────────────────────────────────────────

    async def _cancel_tasks(self) -> None:
        """実行中のループを止めて完全に終了するまで待つ"""
        if not self._tasks:
            return
        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks = []

    async def start(self, video_id: str | None = None) -> None:
        # 前セッションの post-stop 処理（背景）が残っていたら先に終わらせる
        # （新しいセッションと記憶書き込みが競合しないように。Phase 4.5）
        if self._post_stop_task is not None and not self._post_stop_task.done():
            logger.info("Waiting for post-stop work before start")
            await self.await_post_stop(timeout=600.0)
        # video_idの指定が無いときの既定モード。primary_source: local なら
        # Mockではなくローカルチャットで立ち上げる
        default_mode = "local" if self._cfg.youtube.primary_source == "local" else "mock"
        vid = extract_youtube_id(video_id or self._cfg.youtube.video_id or default_mode)
        # 二重start対策：既存のループとソースを畳んでから作り直す
        # （ダッシュボードで「開始」を連打してもループが多重化しない）
        if self._tasks:
            logger.info("Restarting orchestrator (cancelling previous loops)")
            self._running = False
            await self._cancel_tasks()
            if self._source:
                try:
                    await self._source.stop()
                except Exception as e:
                    logger.warning("Failed to stop previous source", error=str(e))
        self._status.video_id = vid
        # 反復判定の履歴は配信セッション単位で持ち直す（前セッションの
        # 挨拶文が残っていると、開始直後の通常の挨拶まで反復扱いで
        # ブロックされてしまう。2026-11-14の2時間テストで発生）
        self._recent_speech_norms.clear()
        self._short_phrase_events.clear()
        self._prefetched = None
        logger.info("Orchestrator starting", video_id=vid,
                    cast=[a.name for a in self._agents])
        if self._archiver is not None:
            try:
                self._archiver.start_run(vid)
            except Exception:
                pass
            asyncio.create_task(self._process_pending_episodes())
        if getattr(self, "_mind_loop", None) is not None:
            self._mind_loop.notify_activity()

        self._source, self._source_fallback = self._build_sources(vid)
        await self.rebuild_llm()
        self.rebuild_tts()
        # 序盤TTS保護の起点をこのセッションの開始に合わせ、全キャラの声で
        # ダミー合成を済ませてから配信を始める（P3。失敗しても開始は止めない）
        self._session_start_mono = time.monotonic()
        await self._warmup_tts_voices()

        await self._source.start(vid)
        self._running = True
        self._restore_chat_context()

        self._tasks = [
            asyncio.create_task(self._comment_ingest_loop(), name="ingest"),
            asyncio.create_task(self._response_loop(), name="response"),
            asyncio.create_task(self._idle_loop(), name="idle"),
            asyncio.create_task(self._health_check_loop(), name="health"),
            asyncio.create_task(self._status_broadcast_loop(), name="status"),
            asyncio.create_task(self._consolidation_loop(), name="consolidation"),
            asyncio.create_task(self._knowledge_loop(), name="knowledge"),
            asyncio.create_task(self._research_worker_loop(), name="research"),
            asyncio.create_task(self._mind_loop.run(), name="mind"),
        ]
        # Whisperモデルの初回ロードは数秒かかるので、裏で先に済ませておく
        if self._stt and self._cfg.stt.warmup_on_start:
            asyncio.create_task(self._stt.warmup(), name="stt_warmup")
        logger.info("Orchestrator started", tasks=len(self._tasks))

    async def stop(self) -> None:
        self._running = False
        # 先行生成タスクは _tasks に入っていないので個別に止める（P1-C）
        if self._prefetch_task is not None and not self._prefetch_task.done():
            self._prefetch_task.cancel()
        self._prefetched = None
        await self._cancel_tasks()
        if self._source:
            await self._source.stop()
        try:
            await self._bgm.stop()
        except Exception:
            pass
        logger.info("Orchestrator stopped")
        # 停止後の重い処理（記憶整理・日記・エピソード化・マインドの
        # post-stop サイクル）は、応答をブロックしないよう背景で実行する
        # （Phase 4.5: /api/chat/stop の応答が35〜185秒掛かっていた対処）。
        # セッション終了（teardown）自体はここまでで完了している。
        self._schedule_post_stop()

    def _schedule_post_stop(self) -> None:
        if self._post_stop_task is not None and not self._post_stop_task.done():
            return
        task = asyncio.create_task(self._post_stop_work(), name="post_stop")
        self._post_stop_task = task
        task.add_done_callback(self._on_post_stop_done)

    @staticmethod
    def _on_post_stop_done(task: asyncio.Task) -> None:
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            logger.warning("Post-stop work failed", error=str(exc))

    async def await_post_stop(self, timeout: float | None = None) -> None:
        """背景の post-stop 処理の完了を待つ（アプリ終了時・再開時用）"""
        task = self._post_stop_task
        if task is None or task.done():
            return
        try:
            if timeout:
                await asyncio.wait_for(asyncio.shield(task), timeout)
            else:
                await task
        except Exception:
            pass

    async def _post_stop_work(self) -> None:
        """停止後の整理一式（旧 stop() の後半。背景タスクとして実行する）"""
        for agent in self._agents:
            # 停止時に、まだ整理していない会話をエピソード記憶に落とす
            if agent.llm and agent.log.get_pending():
                await self._hub.send_log("info", f"配信終了：{agent.name}の短期記憶を整理しています...")
                await self._consolidate_agent(agent, triggered_by="stop")

            # 配信終了時に自己内省（思い出日記）を生成し、コア記憶を更新する
            if agent.llm and self._cfg.orchestrator.enable_reflection:
                try:
                    await self._hub.send_log("info", f"配信終了：{agent.name}の今日の思い出・日記を記録しています...")
                    diary = await self._reflection_engine.reflect_and_update(
                        conversation_log=agent.log,
                        memory_store=agent.memory_store,
                        long_term_memory=agent.long_term_memory,
                        llm=agent.llm,
                        character_name=agent.name,
                    )
                    if diary:
                        await self._hub.send_log("info", f"{agent.name}の日記を保存しました: {diary[:40]}…")
                except Exception as e:
                    logger.warning("Reflection on stop failed", character=agent.name, error=str(e))

        # 配信の全発話からエピソード記憶を作る（台本→記憶の反映）
        if self._archiver is not None and self._archiver.has_run:
            try:
                lead = self._agents[0] if self._agents else None
                if lead is not None and lead.llm:
                    await self._hub.send_log(
                        "info", "配信終了：配信エピソードを記憶に記録しています...")
                    stores = {a.name: a.memory_store for a in self._agents
                              if a.memory_store is not None}
                    topics = []
                    if self._planner is not None:
                        try:
                            topics = self._planner.status().get("ledger", [])
                        except Exception:
                            topics = []
                    res = await self._archiver.build_episode(
                        lead.llm, stores, topics=topics)
                    if res:
                        await self._hub.send_log(
                            "info",
                            f"配信エピソードを記録しました: {res['summary'][:40]}…")
            except Exception as e:
                logger.warning("Stream episode build failed", error=str(e))

        # 終了後の内省（Phase 2）: 各キャラ1回、振り返り＋意見形成を回す。
        # エピソード化の後なので、最新の配信エピソードも材料に入る
        if getattr(self, "_mind_loop", None) is not None:
            for agent in self._agents:
                if not agent.llm:
                    continue
                try:
                    await self._hub.send_log(
                        "info", f"配信終了：{agent.name}が今日のことを考えています...")
                    await self._mind_loop.run_post_stop_cycle(agent)
                except Exception as e:
                    logger.warning("Mind post-stop cycle failed",
                                   character=agent.name, error=str(e))

        logger.info("Post-stop work completed")

    async def _process_pending_episodes(self) -> None:
        """過去の未処理ジャーナル（バックフィル等）をエピソード化する"""
        if self._archiver is None:
            return
        try:
            pending = self._archiver.pending_journals()
            if not pending:
                return
            lead = self._agents[0] if self._agents else None
            if lead is None or not lead.llm:
                return
            stores = {a.name: a.memory_store for a in self._agents
                      if a.memory_store is not None}
            for path in pending:
                try:
                    await self._archiver.build_episode_from_journal(
                        path, lead.llm, stores)
                except Exception as e:
                    logger.warning("Pending episode build failed",
                                   path=str(path), error=str(e))
        except Exception as e:
            logger.warning("Pending episode scan failed", error=str(e))

    # ── ELYTHエピソード取り込み ───────────────────────────────

    async def import_elyth_episodes(self) -> dict:
        """ELYTHの日次エピソードを対象キャラの MemoryStore へ取り込む。

        失敗してもアプリ本体・配信には影響しない（全例外をここで隔離する）。
        取り込み先は config.elyth_memory.character（既定 Character1）のみ。
        """
        result: dict = {}
        if self._elyth_importer is None:
            return result
        try:
            agent = self.get_agent(self._cfg.elyth_memory.character)
            if agent is None or agent.memory_store is None:
                return result
            # ファイル走査とDB投入はブロッキング処理なので別スレッドで回す
            result = await asyncio.to_thread(
                self._elyth_importer.scan, agent.memory_store)
            if result.get("imported"):
                await self._hub.send_log(
                    "info",
                    f"ELYTHエピソードを取り込みました: "
                    f"{result['imported']}ファイル（記憶 {result['added']} 件）")
        except Exception as e:
            logger.warning("ELYTH episode import scan failed", error=str(e))
        return result

    async def run_elyth_import_scheduler(self) -> None:
        """アプリ常駐のELYTH取り込みスケジューラ（main の lifespan から起動する）。

        起動時に1回走査し、その後は scan_interval_sec ごとに繰り返す。
        セッションの有無によらず回る（睡眠スケジューラと同じ扱い）。
        """
        if self._elyth_importer is None:
            return
        interval = max(300.0, float(self._cfg.elyth_memory.scan_interval_sec))
        while True:
            try:
                await self.import_elyth_episodes()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning("ELYTH import scheduler error", error=str(e))
            await asyncio.sleep(interval)

    # ── コメントキュー投入 ─────────────────────────────────────

    async def enqueue_comment(self, comment: Comment, score: float) -> bool:
        """
        コメントをキューに投入する。
        キューが一杯のときは、キュー内の最低優先度と比べて優先度が高ければ
        そちらを追い出して入れる（スパチャや音声入力が雑談で埋もれないようにする）。
        """
        if getattr(self, "_mind_loop", None) is not None:
            self._mind_loop.notify_activity()
        comment.priority = score
        # PriorityQueueは小さい値が先なので負にする
        item = _QueueItem(
            priority=-score,
            seq=next(self._queue_seq),
            comment=comment,
            enqueued_at=time.monotonic(),
        )

        if self._comment_queue.qsize() >= self._cfg.orchestrator.max_queue_size:
            victim = self._evict_lowest_priority(min_priority=item.priority)
            if victim is None:
                logger.info("Comment queue full, dropping incoming comment",
                            author=comment.author, score=score)
                return False
            logger.info("Comment queue full, evicted lower-priority comment",
                        evicted=victim.comment.author, incoming=comment.author)

        await self._comment_queue.put(item)
        return True

    def _evict_lowest_priority(self, min_priority: float) -> _QueueItem | None:
        """
        キュー内で最も優先度の低い項目を、min_priority より低い場合に限り取り除く。
        （priorityは負値で小さいほど高優先度なので、最大値が最低優先度）
        """
        pending: list[_QueueItem] = []
        while not self._comment_queue.empty():
            pending.append(self._comment_queue.get_nowait())
            self._comment_queue.task_done()
        if not pending:
            return None

        worst = max(pending, key=lambda x: x.priority)
        evicted = None
        if worst.priority > min_priority:
            pending.remove(worst)
            evicted = worst
        for it in pending:
            self._comment_queue.put_nowait(it)
        return evicted

    # ── コメント選別（トリアージ）補助 ─────────────────────────

    def _is_known_viewer(self, comment: Comment) -> bool:
        """いずれかの出演者の視聴者プロフィールに記録があるか（初コメ判定用）"""
        if not comment.author:
            return False
        for agent in self._agents:
            store = getattr(agent, "viewer_store", None)
            if store is None:
                continue
            try:
                if store.get_profile(comment.author) is not None:
                    return True
            except Exception:
                continue
        return False

    def _record_unanswered_comment(self, comment: Comment,
                                   triage_category: str | None = None) -> None:
        """応答しないコメントの記録と視聴者プロフィール更新。

        選別で落としたコメントも、配信アーカイブへの記録と視聴者
        プロフィールの来訪記録は従来どおり行う（記録と応答の分離）。
        来訪記録はリーダーのストアへ入れる（応答したコメントは
        _process_comment が担当キャラのストアへ従来どおり入れる）。

        ポイント（Phase 4）の扱いは選別カテゴリで分ける:
          - ng_word / duplicate / spam_repeat（悪用・連打系）は0点。
            来訪ポイントだけは record_visit 経由で通常どおり付く。
          - reaction_only は「リアクション」として小さく加算する。
          - probability_miss / saturated_low_score（混雑で落とした通常の
            コメント）は「コメント」として参加分を加算する。
            ただし「拾われた」上乗せは付かない。
        """
        if self._archiver is not None:
            try:
                self._archiver.record_comment(comment.author, comment.message)
            except Exception:
                pass
        if self._cfg.orchestrator.enable_viewer_profiles and comment.author:
            try:
                store = self._viewer_store
            except Exception:
                store = None
            if store is not None:
                try:
                    store.record_visit(comment.author, comment.author)
                except Exception:
                    pass
                try:
                    if triage_category == "reaction_only":
                        store.award_points(comment.author, "reaction")
                    elif triage_category in (
                            "probability_miss", "saturated_low_score"):
                        store.award_points(comment.author, "comment")
                except Exception:
                    pass

    # ── Loop A: コメント取得 ───────────────────────────────────

    async def _comment_ingest_loop(self) -> None:
        assert self._source is not None
        backoff = 5.0
        while self._running:
            try:
                async for comment in self._source.comments():
                    if not self._running:
                        break
                    backoff = 5.0  # 成功したらリセット
                    if self.is_local_chat and comment.author_channel_id == "local_user":
                        # 1対1の会話では選別・間引きをせず、必ず順番に応答する
                        score = self._cfg.local_chat.priority
                    else:
                        score = self._selector.score(comment)
                        if score < 0:
                            continue  # フィルタされたコメント
                        # コメント選別（トリアージ）: 応答しないコメントは
                        # キューへ入れない。ただし記録と視聴者プロフィールは
                        # 応答の有無と分離して従来どおり更新する
                        triage = getattr(self, "_triage", None)
                        if triage is not None:
                            result = triage.evaluate(
                                comment, score=score,
                                queue_size=self._comment_queue.qsize())
                            if not result.picked:
                                self._record_unanswered_comment(
                                    comment, result.category)
                                continue

                    if not await self.enqueue_comment(comment, score):
                        continue  # キュー満杯で破棄

                    await self._hub.send_comment({
                        "author": comment.author,
                        "message": comment.message,
                        "priority": score,
                        "is_superchat": comment.is_superchat,
                    })
                break  # 正常終了
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("comment_ingest_loop error (retrying)",
                             error=str(e), backoff_sec=backoff)
                await self._hub.send_log("warn", f"コメント取得エラー: {e}（{backoff}秒後にリトライ）")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60.0)  # 指数バックオフ、最大60秒

    # ── Loop B: 応答生成 ──────────────────────────────────────

    async def _response_loop(self) -> None:
        while self._running:
            try:
                item = await asyncio.wait_for(self._comment_queue.get(), timeout=1.0)
                # キュー内で古くなりすぎたコメントは破棄（TTL）
                ttl = self._cfg.orchestrator.comment_ttl_sec
                if item.enqueued_at and ttl > 0 and (time.monotonic() - item.enqueued_at) > ttl:
                    logger.info("Comment expired in queue, skipping",
                                author=item.comment.author,
                                age_sec=round(time.monotonic() - item.enqueued_at))
                    self._comment_queue.task_done()
                    continue
                await self._process_comment(item.comment)
                self._comment_queue.task_done()
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                import traceback
                logger.error("response_loop error",
                             error_type=type(e).__name__,
                             error=str(e),
                             traceback=traceback.format_exc())
                await asyncio.sleep(1.0)

    def _recall(self, query: str) -> str:
        """リーダーの記憶を引く（1体運用の頃のAPI）"""
        return self._lead.recall(query)

    async def _process_comment(self, comment: Comment) -> None:
        responder = self._director.pick_responder(self._agents, comment)
        if not responder.llm:
            return

        # 事実質問はWeb検索してから答える（発話ロックの外で検索する）。
        # 同期検索が空振りした場合と、間隔ゲートで同期を見送る場合は、
        # 質問を捨てずに保留回答キューへ積み、裏の調査ワーカーが調べて
        # 後から自律トークで答える（Phase 3。ゲートで質問が消える
        # 2026-11-17 の無回答事故への対処）
        research: tuple[str, list[WebDocument]] | None = None
        answer_queued = False
        if self._needs_search(comment.message):
            _factual, reason = self._classify_factual_question(comment.message)
            interval_ok = (time.time() - self._last_search_time) \
                >= self._cfg.knowledge.answer_with_search.min_interval_sec
            logger.info("comment search decision", factual=True, reason=reason,
                        sync=interval_ok, author=comment.author)
            if interval_ok:
                try:
                    research = await self._research_for_comment(comment)
                except Exception as e:
                    logger.warning("comment research failed", error=str(e))
            else:
                answer_queued = self._enqueue_pending_answer(comment)
                logger.info("comment search deferred (interval gate)",
                            author=comment.author, queued=answer_queued)
        # 調べたのに資料が無かった質問は保留キューで粘る。資料が
        # あっても、数値を聞かれているのに資料内に答えらしい数値が
        # 1つも無い場合は「不十分」として同じくキューで粘る
        # （Phase 3.5: 2026-11-17 のローカル検証で、資料は返ったのに
        # 終値が含まれておらず「分からない」に落ちた実測への対処）。
        # あわせて従来どおり「分からないまま抱える」ために回答担当の
        # 好奇心キューへも積む（Phase 2: 好奇心の発酵。視聴者への
        # 回答責任は保留キュー側が負い、発酵の結果は長期的な
        # 好奇心として記憶へ行く）
        sync_insufficient = False
        if research is not None and research[1]:
            sync_insufficient = (
                self._expects_numeric_answer(comment.message)
                and not self._docs_have_numeric_token(research[1]))
            if sync_insufficient:
                logger.info("comment research insufficient "
                            "(no numeric token in docs)",
                            author=comment.author)
        if research is not None and (not research[1] or sync_insufficient):
            answer_queued = self._enqueue_pending_answer(comment) or answer_queued
            if responder.mind_store is not None:
                try:
                    push_curiosity(responder.mind_store, comment.message,
                                   comment.author)
                except Exception as e:
                    logger.warning("curiosity enqueue failed", error=str(e))

        async with self._speak_lock:
            self._is_speaking = True
            self._status.last_comment = f"{comment.author}: {comment.message}"
            if self._archiver is not None:
                try:
                    self._archiver.record_comment(comment.author, comment.message)
                except Exception:
                    pass

            # 返事をするのは1体でも、コメントは出演者全員が聞いている
            for agent in self._agents:
                agent.hear_comment(comment)

            # 視聴者プロファイルの来訪記録とコンテキスト取得（返事をするキャラとの関係）
            viewer_context = ""
            store = responder.viewer_store
            if store and self._cfg.orchestrator.enable_viewer_profiles and comment.author:
                store.record_visit(comment.author, comment.author)
                # ポイント（Phase 4）: 応答したコメントは「コメント＋
                # 拾われた」分を、来訪記録と同じストアへ加算する
                # （二重計上は record_visit と同じ場所に寄せて防ぐ）。
                # スパチャは金額に応じた上乗せも足す。
                try:
                    store.award_points(comment.author, "comment")
                    if comment.is_superchat:
                        store.award_points(
                            comment.author, "superchat",
                            superchat_amount=comment.superchat_amount)
                    store.award_points(comment.author, "picked")
                except Exception:
                    pass
                viewer_context = store.build_prompt(comment.author)

            recall_query = f"{comment.author} {comment.message}".strip()
            memory_prompt = await asyncio.to_thread(responder.recall, recall_query)
            # 主題に合う「自分の意見」があれば、状況（L3）として添える（Phase 2）
            try:
                matched = find_relevant_opinions(
                    responder.mind_store, comment.message, limit=2)
                opinion_ctx = format_opinion_context(matched)
            except Exception as e:
                logger.warning("opinion match failed", error=str(e))
                opinion_ctx = ""
            if opinion_ctx:
                memory_prompt = (memory_prompt + "\n\n" + opinion_ctx) \
                    if memory_prompt else opinion_ctx
            # 予定・取り組みを尋ねるコメントには「自分の意図」も同格に
            # 添える（Phase 4.5: 意図タイトルが回答に反映されなかった対処）
            try:
                intention_ctx = intention_context_for(
                    responder.mind_store, comment.message)
            except Exception as e:
                logger.warning("intention context failed", error=str(e))
                intention_ctx = ""
            if intention_ctx:
                memory_prompt = (memory_prompt + "\n\n" + intention_ctx) \
                    if memory_prompt else intention_ctx
            if research is not None and research[1] \
                    and not sync_insufficient:
                _query, docs = research
                system_prompt = build_research_prompt(
                    self._cfg,
                    question=comment.message,
                    documents=docs,
                    long_term_memory_text=memory_prompt,
                    **responder.prompt_kwargs(),
                )
            elif answer_queued:
                # 検索は保留キューへ依頼済み。この分岐でのみ「調べるね」
                # 系の宣言を許可する（宣言と実行の一致。Phase 3）
                system_prompt = build_answer_pending_prompt(
                    self._cfg,
                    question=comment.message,
                    long_term_memory_text=memory_prompt,
                    **responder.prompt_kwargs(),
                )
            elif research is not None:
                # 同期検索が空振りで、保留キューも無い（プランナ無効時）。
                # 従来どおり「調べたけど分からなかった」と正直に答える
                _query, docs = research
                system_prompt = build_research_prompt(
                    self._cfg,
                    question=comment.message,
                    documents=docs,
                    long_term_memory_text=memory_prompt,
                    **responder.prompt_kwargs(),
                )
            else:
                system_prompt = build_system_prompt(
                    self._cfg,
                    long_term_memory_text=memory_prompt,
                    viewer_context=viewer_context,
                    **responder.prompt_kwargs(),
                )

            try:
                response = await self._generate_and_speak(system_prompt, responder)
            finally:
                self._is_speaking = False

            if response is None:
                return
            # 宣言と実行の一致の見張り（Phase 3）: 検索も保留も無いのに
            # 「調べるね」と言ってしまったら、言った約束をシステム側が
            # 引き受けて保留キューへ遡及投入する
            if research is None and not answer_queued and response.speech \
                    and self._DECLARATION_RE.search(response.speech):
                logger.warning("unfulfilled research promise",
                               speech=response.speech[:40],
                               author=comment.author)
                self._enqueue_pending_answer(comment)
            self._after_speech(responder, response)
            self._selector.mark_responded(comment)
            # 調査経路（同期・キュー）に入ったコメントは note_comment の
            # 対象から外す。答えの無い生の質問が comment hook で自律
            # トピック化され、雑談に溶けて消える経路を塞ぐ（Phase 3）
            if self._planner is not None and research is None \
                    and not answer_queued:
                try:
                    self._planner.note_comment(comment.author, comment.message)
                except Exception:
                    pass

        # 掛け合い中なら、相方が一言かぶせることがある
        await self._continue_banter(responder, after="comment")
        self._schedule_consolidation()
        self._maybe_start_prefetch()

    @staticmethod
    def _normalize_speech(text: str) -> str:
        """反復判定用に、句読点・記号・空白を落として正規化する"""
        return re.sub(r"[\s。、！？!?…♪・「」『』（）()\"“”'’]", "", text)

    @staticmethod
    def _repeat_match(norm: str, prev: str) -> bool:
        """正規化済みテキスト同士の反復一致（完全一致・包含・類似度）。"""
        if norm == prev:
            return True
        shorter, longer = (norm, prev) if len(norm) <= len(prev) else (prev, norm)
        if shorter and shorter in longer:
            return True
        ratio = difflib.SequenceMatcher(None, norm, prev).ratio()
        if ratio > 0.85:
            return True
        # 挨拶文などの「出だしが同じで末尾だけ違う」反復も拾う
        prefix = 0
        for a, b in zip(norm, prev):
            if a != b:
                break
            prefix += 1
        if prefix >= 10 and ratio > 0.70:
            return True
        return False

    def _is_repeat_speech(self, text: str) -> bool:
        """直近の発話と（ほぼ）同じ内容かどうか"""
        norm = self._normalize_speech(text)
        if len(norm) < 8:
            return False
        return any(self._repeat_match(norm, prev)
                   for prev in self._recent_speech_norms)

    def _should_block_repeat(self, text: str) -> bool:
        """反復を実際に差し止めるか（P1-B の段階化を適用した判定）。

        長い発話の再演は従来どおり即ブロックする。15字以下の短い汎用句
        （相づち・感謝など）は、単発の再演では止めず、同一句が短時間
        （120秒）の窓に3回以上 出た場合だけブロック対象にする。
        """
        if not self._is_repeat_speech(text):
            return False
        norm = self._normalize_speech(text)
        if len(norm) > self._SHORT_PHRASE_MAX_LEN:
            return True
        # 段階化は「短い発話そのものの再演」にだけ適用する。長い発話の
        # 一部（短い文）が包含一致で反復扱いになった場合は即ブロック。
        short_match = any(
            self._repeat_match(norm, prev)
            for prev in self._recent_speech_norms
            if len(prev) <= self._SHORT_PHRASE_MAX_LEN)
        if not short_match:
            return True
        now = time.time()
        count = sum(1 for at, n in self._short_phrase_events
                    if n == norm
                    and now - at < self._SHORT_PHRASE_WINDOW_SEC)
        if count + 1 >= self._SHORT_PHRASE_BLOCK_COUNT:
            return True
        logger.info("short phrase repeat allowed (tiered block)",
                    text=text[:30], occurrences=count + 1)
        return False

    # 文頭の相づちフレームと定型賛辞（文型レベルの反復検出用）
    _FORMULAIC_OPENINGS = ("はいはい", "あらあら", "うんうん", "そうそう",
                           "なるほど", "ほほう", "おお")
    _FORMULAIC_PRAISE = ("すごいですね", "すごいね", "素敵ですね", "素敵だね",
                         "さすがですね", "さすがだね", "立派ですね",
                         "頑張りましたね", "えらいですね")

    def _is_formulaic_opening(self, text: str) -> bool:
        """「文頭の相づち＋定型賛辞のみ・新規の内容語ゼロ」の発話か。

        相づちアトラクタ（「はいはい、すごいですね。」等）は正規化8字未満で
        _is_repeat_speech の対象外になることがあり、言い回しの亜種は
        包含・類似度でも拾いきれない。ここでは文型として検出する。
        判定に使うのは「相づちと賛辞を取り除いた残りにテーマ語
        （_theme_grams）が無いこと」で、内容が続けば定型文型ではない。
        """
        norm = self._normalize_speech(text)
        if not norm:
            return False
        rest = norm
        for op in self._FORMULAIC_OPENINGS:
            if rest.startswith(op):
                rest = rest[len(op):]
                break
        else:
            return False
        for pr in self._FORMULAIC_PRAISE:
            if rest.startswith(pr):
                rest = rest[len(pr):]
                break
        else:
            return False
        return not self._theme_grams(rest)

    def _aizuchi_pattern_of(self, text: str) -> str:
        """定型相づち型（P2-A）に一致したらパターンID、しなければ空文字。

        Character2の「どういうことなの？わたしにも教えて？」型は文頭ではなく
        文末・文中に現れるため、文型（formulaic opening）検出とは別に
        正規化テキスト全体へのパターン一致で拾う。
        """
        norm = self._normalize_speech(text)
        if not norm:
            return ""
        for pid, pat in self._AIZUCHI_PATTERNS:
            if pat.search(norm):
                return pid
        return ""

    def _aizuchi_in_cooldown(self, text: str) -> str:
        """クールダウン中の定型相づち型に一致したらパターンIDを返す"""
        pid = self._aizuchi_pattern_of(text)
        if pid and time.time() < self._aizuchi_cooldown_until.get(pid, 0.0):
            return pid
        return ""

    def _note_blocked(self, agent: CharacterAgent) -> None:
        """発話ブロックを数える（回復フローの状態管理。Phase 2）。

        キャラ別と合算の連続ブロック数を進め、合算が上限に達したら
        既存の文脈リセットへエスカレーションする。リセット自体は
        発話ではないので、カウンタは発火時点で戻す（リセット後も
        詰まるようなら、また数え直して再発火する）。
        """
        self._block_streak[agent.key] = self._block_streak.get(agent.key, 0) + 1
        self._total_block_streak += 1
        logger.info("block streak updated", character=agent.name,
                    streak=self._block_streak[agent.key],
                    total=self._total_block_streak)
        if self._total_block_streak >= self._BLOCK_RESET_THRESHOLD:
            self._total_block_streak = 0
            self._block_streak.clear()
            self._reset_saturated_context(self._saturation_word())

    # 話題飽和の判定から外す3-gram（名前・定型の言い回し）
    _SATURATION_STOP_GRAMS = {
        "くださ", "ださい", "ありが", "りがと", "がとう", "うござ", "ござい",
        "でしょ", "ました", "ですね", "ますね", "なこち", "Character1", "Character2",
        "んこち", "ボクき", "クきな", "わたし",
    }

    def _note_saturation(self, word: str) -> None:
        """吸着ガードの発火を記録し、同じ語で繰り返すようなら文脈リセットへ"""
        if not word:
            return
        now = time.time()
        self._saturation_events.append((now, word))
        self._saturation_events = [(at, w) for at, w in self._saturation_events
                                   if now - at < 900]
        same = sum(1 for _, w in self._saturation_events if w == word)
        if same >= 4 and now - self._last_context_reset_at > 600:
            self._reset_saturated_context(word)

    def _reset_saturated_context(self, word: str) -> None:
        """吸着した文脈を捨てる：会話ログ全消去＋反復履歴クリア＋語の禁止

        2026-11-14の2時間テストで、記憶の種を除去しても会話ログ内の
        文脈だけで吸着が再形成されたため、エスカレーションとして追加。
        """
        now = time.time()
        self._last_context_reset_at = now
        for agent in self._agents:
            try:
                agent.log.clear()
            except Exception as e:
                logger.warning("context reset: log clear failed", error=str(e))
        self._recent_speech_norms.clear()
        self._short_phrase_events.clear()
        self._recent_theme_grams.clear()
        self._theme_hint_pending = ""
        if self._planner is not None:
            try:
                self._planner.reset_thread()
                self._planner.ban_word(word, seconds=1800)
            except Exception as e:
                logger.warning("context reset: planner reset failed", error=str(e))
        self._materialless_idle_count = 0
        self._last_speech_time = now   # 少し間を空けてから再開する
        logger.info("context reset (saturation)", word=word)

    def _character_name_forms(self) -> list[str]:
        """飽和検出から外す出演者名の表記（名前・愛称・「〜ちゃん」形）。

        長い形から順に返す（「Character1ちゃん」を「Character1」より先に消すため）。
        正規化済み発話と比べるので、名前側も同じ正規化を通した形で持つ。
        """
        forms: set[str] = set()
        for agent in self._agents:
            names = [agent.name] + list(getattr(agent.cfg, "aliases", []) or [])
            for n in names:
                if n:
                    forms.add(n)
                    forms.add(n + "ちゃん")
        normalized = {self._normalize_speech(f) for f in forms}
        return sorted((f for f in normalized if f), key=len, reverse=True)

    def _saturation_word(self) -> str:
        """直近の発話に同じ内容語が偏っていないか（話題の飽和検出）。

        正規化済み発話の直近8件について、カタカナ・漢字の3-gramが
        何件の発話に現れたかを数え、5件以上ならその語を返す。
        モデル側の「好きな話題」へ会話が吸着するのを断つための信号で、
        記憶DBに残っていない吸着にも効く（2026-11-14追加）。

        3-gram 化の前に出演者名を取り除く（Phase 4・案F）。名前そのものは
        停止グラムで落とせても、「Character2も」のような名前＋助詞の断片が
        最頻 gram に勝ってしまい、本来の内容語を拾えなかったため。
        """
        norms = self._recent_speech_norms[-8:]
        if len(norms) < 6:
            return ""
        name_forms = self._character_name_forms()
        from collections import Counter
        coverage: Counter = Counter()
        for norm in norms:
            for form in name_forms:
                norm = norm.replace(form, "")
            grams = set()
            for i in range(len(norm) - 2):
                g = norm[i:i + 3]
                if all(("\u3040" <= c <= "\u30ff") or ("\u4e00" <= c <= "\u9fff")
                       for c in g):
                    grams.add(g)
            for g in grams - self._SATURATION_STOP_GRAMS:
                coverage[g] += 1
        if not coverage:
            return ""
        word, n = coverage.most_common(1)[0]
        return word if n >= 5 else ""

    # ── テーマ反復の検知と話題切り替えヒント（Phase 4.6）──────
    # 飽和検出（_saturation_word）が「掛け合いを切る／文脈を捨てる」のに
    # 対し、こちらは一段軽い誘導: 同じテーマが続いたら、次の発話の
    # プロンプトに「別の話題・別の切り口へ」と1回だけ添える。特定語の
    # 禁止ではなくテーマ一般の反復への仕組みで、計数は両キャラ合算
    # （片方が持ち出したテーマへの相方の追随増幅も数える）。

    _THEME_WINDOW = 6            # 反復を見る直近発話数（両キャラ合算）
    _THEME_REPEAT_MIN = 3        # 窓の中でこの件数以上に出たら反復とみなす
    _THEME_HINT_SUPPRESS_SEC = 600.0  # 同じテーマでは再発させない間隔
    _THEME_STOP_GRAMS = {"ボク", "アン", "ンコ"}  # 一人称・相方のカタカナ名

    @staticmethod
    def _theme_grams(text: str) -> set[str]:
        """発話のテーマを表す文字バイグラム集合。

        数えるのは「漢字・カタカナ同士の対」と「漢字＋ひらがなの
        送り仮名対（調べ・整え等）」だけ。ひらがなだけの助詞・語尾や
        「クは」のようなカタカナ＋ひらがなの対は話題を表さないので
        落とす。名前はひらがな表記なら自然に落ち、カタカナ表記は
        止め語で落とす。
        """
        def is_ck(c: str) -> bool:
            return ("\u4e00" <= c <= "\u9fff") or ("\u30a0" <= c <= "\u30ff")

        norm = Orchestrator._normalize_speech(text or "")
        grams: set[str] = set()
        for i in range(len(norm) - 1):
            a, b = norm[i], norm[i + 1]
            pair = a + b
            if pair in Orchestrator._THEME_STOP_GRAMS:
                continue
            if (is_ck(a) and is_ck(b)) or \
                    ("\u4e00" <= a <= "\u9fff" and "\u3040" <= b <= "\u309f"):
                grams.add(pair)
        return grams

    @staticmethod
    def _theme_label(text: str, gram: str) -> str:
        """ヒント文に出すテーマの呼び名（gram を含む漢字・カタカナ連続を優先）"""
        runs = re.findall(r"[\u4e00-\u9fff\u30a0-\u30ff]+", text or "")
        hits = [r for r in runs if gram in r]
        if hits:
            return max(hits, key=len)[:8]
        return gram

    def _note_theme_repetition(self, text: str) -> None:
        """発話のテーマを記録し、反復が続くようなら切り替えヒントを積む"""
        grams = self._theme_grams(text)
        # 案D（Phase 4）: ヒントを付けた発話の検証。消費直後の発話に同じ
        # gram がまだ残っていたら、窓の再判定を待たずその場でエスカレーション
        # する（ban 第2段階＋強制切替）。残っていなければヒントが効いたと
        # みなして検証を終える。マーカーは消費後の最初の発話だけに効く。
        consumed = self._theme_hint_consumed_gram
        if consumed:
            self._theme_hint_consumed_gram = ""
            if consumed in grams:
                logger.info("theme escalation: hint ignored", gram=consumed)
                if self._planner is not None:
                    try:
                        self._planner.ban_word(consumed, seconds=600)
                        logger.info("theme ban applied", word=consumed,
                                    tier=2, seconds=600)
                    except Exception as e:
                        logger.warning("theme ban failed", error=str(e))
                self._force_switch_pending = True
        self._recent_theme_grams.append(grams)
        del self._recent_theme_grams[:-12]
        if not grams:
            return
        window = self._recent_theme_grams[-self._THEME_WINDOW:]
        if len(window) < self._THEME_REPEAT_MIN:
            return
        scored = [(g, sum(1 for s in window if g in s)) for g in grams]
        scored = [(g, n) for g, n in scored if n >= self._THEME_REPEAT_MIN]
        if not scored:
            return

        def rank(item):
            # 件数優先、同数なら漢字・カタカナだけの対（綺麗なテーマ名に
            # なる）を優先し、送り仮名対（調べ等）は次点にする
            g, n = item
            clean = all(("\u4e00" <= c <= "\u9fff") or ("\u30a0" <= c <= "\u30ff")
                        for c in g)
            return (n, 1 if clean else 0)

        gram, n = max(scored, key=rank)
        label = self._theme_label(text, gram)
        now = time.time()
        if now - self._theme_hint_fired.get(label, 0.0) \
                < self._THEME_HINT_SUPPRESS_SEC:
            return
        # 案D（Phase 4）: エスカレーションの蓄積は代表gram単位で共有する。
        # ラベルが「調査結果」「調査スキル」のように割れても、同じ gram の
        # 別ラベルで再検知されたら初回扱いにせずエスカレーションとする
        label_fired_before = label in self._theme_hint_fired
        cluster_fired_before = gram in self._theme_gram_fired \
            and not label_fired_before
        escalated = label_fired_before or cluster_fired_before
        self._theme_hint_fired[label] = now
        self._theme_hint_gram[label] = gram
        self._theme_gram_fired[gram] = now
        self._theme_gram_fire_count[gram] = \
            self._theme_gram_fire_count.get(gram, 0) + 1
        if not self._theme_hint_pending:
            self._theme_hint_pending = label
            logger.info("theme repetition detected", theme=label, count=n,
                        gram=gram, cluster=cluster_fired_before)
        # 案A: ヒント（助言）だけでなく、材料選択そのものからテーマを外す。
        # ban のフィルタはプランナの材料選択経路に配線済みで、コメント応答や
        # ニュース記事そのものには効かない。第1段階はラベル（表層フレーズ）
        # を、再検知＝エスカレーション時は代表gramを禁止する。
        if self._planner is not None:
            try:
                self._planner.ban_word(label, seconds=600)
                logger.info("theme ban applied", word=label, tier=1, seconds=600)
                if escalated:
                    self._planner.ban_word(gram, seconds=600)
                    logger.info("theme ban applied", word=gram, tier=2,
                                seconds=600)
            except Exception as e:
                logger.warning("theme ban failed", error=str(e))
        if escalated:
            # 案C: 次の自律トークは話題種別を強制して旧テーマから離す
            self._force_switch_pending = True
            logger.info("theme escalation: force switch armed", theme=label)

    def _consume_theme_hint(self, system_prompt: str) -> str:
        """積まれている話題切り替えヒントを次の発話のプロンプトへ1回だけ付ける"""
        theme = self._theme_hint_pending
        if not theme:
            return system_prompt
        self._theme_hint_pending = ""
        self._theme_hint_count += 1
        logger.info("theme shift hint injected", theme=theme,
                    count=self._theme_hint_count)
        # 案D（Phase 4）: 消費したヒントの代表gramを記録し、次の発話で
        # まだ同じテーマが残っていたら即エスカレーションできるようにする
        gram = self._theme_hint_gram.get(theme, "")
        if gram:
            self._theme_hint_consumed_gram = gram
        extra = ""
        if gram and self._theme_gram_fire_count.get(gram, 0) >= 2:
            # 同じテーマの検知が繰り返されている場合は、禁止対象を
            # ヒント文でも明示する（強制力の本体は ban と強制切替側）
            extra = f"特に「{gram}」という言葉や話題は、今回は避けてください。"
        return system_prompt + (
            "\n\n## 話題の切り替え\n"
            f"直近の会話では「{theme}」の話題が続いています。"
            "今回はその話題には触れず、別の話題や別の切り口で話してください。"
            + extra
        )

    def _consume_recovery_hint(self, system_prompt: str,
                               agent: CharacterAgent) -> str:
        """反復ブロック・文型反復の直後の再生成にだけ付ける回復ヒントを消費する。

        テーマヒントと同じ方式で、対象キャラの次の生成1回だけ
        システムプロンプト末尾に「言い回しを変える」指示を足す。
        """
        if self._recovery_hint_pending != agent.key:
            return system_prompt
        self._recovery_hint_pending = ""
        logger.info("recovery hint injected", character=agent.name)
        return system_prompt + (
            "\n\n## 言い回しの変更\n"
            "直前の生成は、最近話した内容と重なって読み上げられませんでした。"
            "相づちや定型の感想だけで終えず、具体的な内容を含む別の切り口で、"
            "新鮮な一言を話してください。"
        )

    def _after_speech(self, agent: CharacterAgent, response: LLMResponse,
                      idle: bool = False, topic_kind: str = "") -> bool:
        """発話を記録する（本人のログ・相方のログ・チャット履歴・状態表示）

        topic_kind: その発話を生んだ話題の種別（自律トーク経由のみ渡る）。
        "answer"（調査結果の受け渡し）はテーマ反復の計数から外す（Phase 4・案D）。

        Returns: 直近発話の反復だったら True（呼び出し側は掛け合いを打ち切る）
        """
        # 発話できたらブロック連続数はリセット（回復フローの状態管理）
        self._block_streak[agent.key] = 0
        self._total_block_streak = 0
        # 反復ループ対策：直近発話とほぼ同じ内容は、LLMの文脈（本人ログ・
        # 相方の聞き取り）へ戻さない。自己参照で同じ文を再生産し続ける
        # 縮退を断つため。チャット履歴と状態表示は通常どおり残す。
        repeat = self._is_repeat_speech(response.speech)
        speech_norm = self._normalize_speech(response.speech)
        self._recent_speech_norms.append(speech_norm)
        del self._recent_speech_norms[:-12]
        # 短い汎用句は出現時刻も記録する（P1-B: 窓内の回数で段階判定する）
        if speech_norm and len(speech_norm) <= self._SHORT_PHRASE_MAX_LEN:
            now = time.time()
            self._short_phrase_events.append((now, speech_norm))
            self._short_phrase_events = [
                (at, n) for at, n in self._short_phrase_events
                if now - at < 900]
        # 定型相づちが実際に発話されたら、その型にクールダウンを設定する
        # （P2-A: 以後の生成で同じ型が出たら再生成・話者交代の対象にする）
        aizuchi = self._aizuchi_pattern_of(response.speech)
        if aizuchi:
            # 型はファミリー単位で差し止める（文単位で届く前半・後半の
            # どちらかだけの再演も止めるため、一致した型だけでなく
            # 全パターンを同時にクールダウンへ入れる）
            until = time.time() + self._AIZUCHI_COOLDOWN_SEC
            for pid, _pat in self._AIZUCHI_PATTERNS:
                self._aizuchi_cooldown_until[pid] = until
            logger.info("formulaic aizuchi cooldown armed", pattern=aizuchi,
                        character=agent.name)
        # テーマ反復の計数（両キャラ合算。反復として弾かれた発話も
        # 実際には発話されているので数える。Phase 4.6）
        if topic_kind == "answer":
            # 調査結果の受け渡しは正当な資料伝達であり、同じ語が続くのは
            # 当然なので、張り付きと誤認しないよう計数から外す（案D）。
            # ヒント検証用マーカーも、この発話には帰属させられないため解除する
            self._theme_hint_consumed_gram = ""
        else:
            self._note_theme_repetition(response.speech)
        if repeat:
            logger.warning("repeat speech: history feedback suppressed",
                           character=agent.name, speech=response.speech[:40])
        else:
            if idle:
                agent.log.add_idle(response.speech)
            else:
                agent.log.add_response(response)
            for other in self._agents:
                if other is not agent:
                    other.hear_partner(agent.name, response.speech)
        self._record_chat("assistant", response.speech,
                          emotion=response.emotion.value, agent=agent)
        agent.emotion = response.emotion
        if not repeat:
            # 反復時は相方の反応材料（last_speech）を更新しない。
            # 同じ文への反応が次の反復を呼ぶ増幅を断つため。
            agent.last_speech = response.speech
        agent.last_spoke_at = time.time()
        agent.speak_count += 1
        self._status.last_speech = (f"{agent.name}: {response.speech}" if self.is_ensemble
                                    else response.speech)
        self._status.current_emotion = response.emotion
        self._last_speech_time = time.time()
        if self._planner is not None:
            try:
                # 発話テーマも渡す（Phase 4.5: 直近発話テーマとの重複を
                # 次の話題選択でペナルティ化するため）
                self._planner.note_speech(idle=idle, text=response.speech)
            except Exception:
                pass
        if self._archiver is not None:
            try:
                self._archiver.record_speech(
                    agent.name, response.speech, "idle" if idle else "talk")
            except Exception:
                pass
        return repeat

    def _schedule_consolidation(self) -> None:
        """ターン数が溜まったキャラの記憶整理を裏で走らせる（Lock外で実行）"""
        trigger = self._consolidation_trigger_turns()
        for agent in self._agents:
            if (agent.llm and agent.key not in self._consolidating
                    and agent.log.should_consolidate(trigger)):
                asyncio.create_task(self._consolidate_agent(agent, triggered_by="turn_count"))

    # ── 掛け合い ──────────────────────────────────────────────

    async def _continue_banter(self, speaker: CharacterAgent, after: str) -> int:
        """
        直前に喋ったキャラに相方が反応し、短い掛け合いを続ける。
        コメントが待っていたら視聴者を優先して切り上げる。戻り値は続いた回数。
        """
        if not self.is_ensemble:
            return 0
        cfg = self._cfg.ensemble
        turn = 0
        current = speaker
        while self._director.wants_banter(after if turn == 0 else "banter", turn):
            if cfg.yield_to_comments and not self._comment_queue.empty():
                break
            saturated = self._saturation_word()
            if saturated:
                logger.info("banter cut (topic saturation)", word=saturated)
                self._note_saturation(saturated)
                break
            reactor = self._director.pick_reactor(self._agents, current)
            if reactor is None or not reactor.llm:
                break
            if cfg.turn_gap_sec > 0:
                await asyncio.sleep(cfg.turn_gap_sec)
            last_turn = (turn + 1) >= cfg.max_banter_turns
            if not await self._speak_banter(reactor, current, last_turn=last_turn):
                break
            current = reactor
            turn += 1
        return turn

    async def _speak_banter(self, agent: CharacterAgent, partner: CharacterAgent,
                            last_turn: bool = False) -> LLMResponse | None:
        """partner の直前の発言に agent が返す"""
        async with self._speak_lock:
            self._is_speaking = True
            try:
                query = f"{partner.name} {partner.last_speech}".strip()
                memory_prompt = await asyncio.to_thread(agent.recall, query)
                system_prompt = build_banter_prompt(
                    self._cfg,
                    partner_name=partner.name,
                    partner_speech=partner.last_speech,
                    long_term_memory_text=memory_prompt,
                    last_turn=last_turn,
                    **agent.prompt_kwargs(),
                )
                response = await self._generate_and_speak(system_prompt, agent,
                                                          route="idle")
            finally:
                self._is_speaking = False
            if response is not None:
                if self._after_speech(agent, response, idle=True):
                    return None   # 反復したら掛け合いをここで打ち切る
            return response

    # ── Loop C: 独り言 ────────────────────────────────────────

    def _effective_idle_threshold(self) -> float:
        """独り言までの待ち時間を状況で変える（AI Director式ペーシング）

        固定閾値だと配信の流れに合わないため、基本値に係数を掛ける：
        開幕直後は詰める／コメント応答の直後は少し譲る／長い沈黙では詰める。
        """
        base = float(self._cfg.orchestrator.idle_threshold_sec)
        now = time.time()
        factor = 1.0
        if now - self._start_time < 300:
            factor *= 0.7
        if self._planner is not None \
                and now - self._planner.last_comment_speech_at < 90:
            factor *= 1.35
        if now - self._last_speech_time > 600:
            factor *= 0.6
        return max(15.0, min(base * 2.0, base * factor))

    async def _idle_loop(self) -> None:
        while self._running:
            try:
                await asyncio.sleep(5.0)
                if self.is_local_chat and not self._cfg.local_chat.idle_talk:
                    continue   # 1対1の会話中は、こちらから話しかけない設定
                silence = time.time() - self._last_speech_time
                if silence >= self._effective_idle_threshold() and not self._is_speaking:
                    await self._speak_idle()
                    # 材料なし等で喋れず無音が長引く時間帯は、システム発火の
                    # 軽い演出で間を持つ（Phase 2）。実発話からの経過で判定
                    # する（_last_speech_time は試行でも更新されるため使わない）
                    sys_cfg = self._cfg.effects.system
                    if sys_cfg.enabled and not self._is_speaking:
                        last_spoke = max(
                            [self._start_time]
                            + [getattr(a, "last_spoke_at", 0.0)
                               for a in self._agents])
                        if time.time() - last_spoke >= sys_cfg.silence_threshold_sec:
                            await self._maybe_system_effect("long_silence")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("idle_loop error", error=str(e))
                await asyncio.sleep(5.0)

    def _build_idle_memory(self, agent: CharacterAgent | None = None) -> str:
        return (agent or self._lead).build_idle_memory(self._bgm.build_bgm_prompt())

    def _other_agent(self, agent: CharacterAgent) -> CharacterAgent | None:
        """相方を返す（1体運用なら None）"""
        for a in self._agents:
            if a.key != agent.key:
                return a
        return None

    async def _speak_fallback_line(self, agent: CharacterAgent) -> LLMResponse | None:
        """生成が詰まったときの最終手段：キャラ別の定型台詞をそのまま喋る。

        config.yaml の characters[].fallback_lines（話題に依存しない
        繋ぎ台詞）を順番に使い、直近発話と重なるものは飛ばす。
        呼び出し側が _after_speech で通常の発話として記録する。
        """
        pool = [line.strip() for line in (agent.cfg.fallback_lines or [])
                if line and line.strip()]
        if not pool:
            return None
        start = self._fallback_line_cursor.get(agent.key, 0) % len(pool)
        for i in range(len(pool)):
            line = pool[(start + i) % len(pool)]
            if self._is_repeat_speech(line):
                continue
            self._fallback_line_cursor[agent.key] = (start + i + 1) % len(pool)
            response = LLMResponse(speech=line)
            await self._speak(response, agent)
            logger.info("fallback line spoken", character=agent.name,
                        speech=line[:40])
            return response
        return None

    async def _maybe_system_effect(self, reason: str) -> None:
        """無音・材料なしの時間帯に、軽い演出をシステムから発火する。

        演出は追加経路なので、失敗しても配信フローは止めない。
        発火先のキャラは出演者からランダムに選ぶ。
        """
        try:
            agent = random.choice(self._agents) if self._agents else self._lead
            await self._effects.dispatch_system(agent=agent, reason=reason)
        except Exception as e:
            logger.warning("system effect failed", error=str(e))

    async def _speak_idle(self, kind: str = "", query: str = "",
                          agent: CharacterAgent | None = None) -> None:
        """
        コメントが無いときに自分から話す。

        話題は TopicPlanner が決める。ニュース/検索ネタが取れればそれを、
        取れなければ従来どおり記憶のキーワードから話す（ネット不要のフォールバック）。
        掛け合い中は、しばらく喋っていない方が話題を振り、相方が受ける。
        """
        self._gen_budget = None  # 前ターンの予算が残っていたら掃除する
        speaker = agent or self._director.pick_idle_speaker(self._agents)
        if agent is None and len(self._agents) > 1:
            # 回復フローでクールダウン中の話者は選ばない（director は
            # 無変更とし、選択結果の受け渡し側で制御する。Phase 2）
            until = self._speaker_cooldown_until.get(speaker.key, 0.0)
            if time.time() < until:
                partner = self._other_agent(speaker)
                if partner is not None and partner.llm:
                    logger.info("idle speaker in cooldown, using partner",
                                character=speaker.name, partner=partner.name)
                    speaker = partner
        if not speaker.llm:
            return
        # 成功/失敗に関わらず再試行間隔を確保（LLM連打防止）
        self._last_speech_time = time.time()

        # 先行生成が使える状況なら、生成済みの発話をそのまま喋る（P1-C）。
        # 生成と回復リトライは前の発話の再生・待機中に済んでいるため、
        # ここでの待ちは発話だけになり、無音区間が縮む
        if not kind and not query:
            pf = self._take_prefetched(speaker)
            if pf is not None:
                pf_topic = pf["topic"]
                async with self._speak_lock:
                    self._is_speaking = True
                    try:
                        await self._speak(pf["response"], speaker)
                        if pf["actions"]:
                            asyncio.create_task(
                                self._handle_unity_actions(pf["actions"]))
                    finally:
                        self._is_speaking = False
                pf_repeat = self._after_speech(
                    speaker, pf["response"], idle=True,
                    topic_kind=pf_topic.kind if pf_topic is not None else "")
                if pf_topic is not None and self._planner is not None:
                    try:
                        self._planner.commit(pf_topic)
                    except Exception as e:
                        logger.warning("topic commit failed", error=str(e))
                if pf_topic is not None and pf_topic.has_material \
                        and pf_topic.kind in ("news", "curiosity"):
                    self._last_web_topic_time = time.time()
                    self._remember_topic(pf_topic, speaker)
                logger.info("idle speech from prefetch",
                            character=speaker.name)
                if not pf_repeat:
                    await self._continue_banter(speaker, after="idle")
                    self._schedule_consolidation()
                    self._maybe_start_prefetch()
                return

        # 話題決め（検索を伴うので発話ロックの外でやる）
        if not kind and not query:
            # 保留中の回答が最優先（Phase 3 の優先順の取り決め）：
            # 差し込み可能な回答（ready／failed）がある間は、吸着対策の
            # ニュース強制もテーマ反復の強制切替も効かせず、plan() の
            # 自動経路に回答トピックを返させる。視聴者が回答待ちのため。
            # waiting（調査中）は対象外で、その間は通常の話題進行を止めない
            answer_deliverable = False
            if self._planner is not None:
                try:
                    answer_deliverable = \
                        self._planner.has_deliverable_answer()
                except Exception:
                    answer_deliverable = False
            if answer_deliverable:
                logger.info("idle: pending answer takes priority")
            else:
                saturated = self._saturation_word()
                if saturated:
                    logger.info("idle topic forced to news (topic saturation)",
                                word=saturated)
                    self._note_saturation(saturated)
                    kind = "news"
            if not answer_deliverable and self._force_switch_pending:
                # テーマ反復のエスカレーション（案C）: 継続スレッドを捨て、
                # 次の話題をWeb系に強制して旧テーマへの引き戻しを断つ。
                # plan() は kind 指定時に継続・復帰の自動経路をスキップする
                self._force_switch_pending = False
                if self._planner is not None:
                    try:
                        self._planner.reset_thread()
                    except Exception as e:
                        logger.warning("force switch reset failed", error=str(e))
                if not kind:
                    try:
                        news_stock = self._news is not None \
                            and bool(self._news.unused_items())
                    except Exception:
                        news_stock = False
                    if news_stock:
                        kind = "news"
                    elif self._search is not None:
                        kind = "curiosity"
                    if kind:
                        logger.info("idle topic forced (theme escalation)",
                                    kind=kind)
        topic: Topic | None = None
        try:
            topic = await self.plan_topic(kind=kind, query=query)
        except Exception as e:
            logger.warning("topic planning failed, falling back to memory", error=str(e))

        # 材料のない話題では喋らない（文脈からの自由連想が吸着を再形成する。
        # 2026-11-14の2時間テストで「強制ニュース→在庫切れ→材料なし」が
        # 約100分続いた反省）。ニュース強制だった場合は再取得して1回引き直す。
        if self._planner is not None \
                and not (topic is not None and topic.has_material):
            if kind == "news":
                try:
                    await self.refresh_news()
                    topic = await self.plan_topic(kind="news", query=query)
                except Exception as e:
                    logger.warning("news refresh retry failed", error=str(e))
            if not (topic is not None and topic.has_material):
                self._materialless_idle_count += 1
                logger.info("idle talk skipped (no material)",
                            count=self._materialless_idle_count)
                if self._materialless_idle_count >= 3:
                    self._reset_saturated_context(self._saturation_word())
                # 喋れない時間帯はシステム発火の演出で間を持つ（Phase 2）
                await self._maybe_system_effect("no_material")
                return
        if topic is not None and topic.has_material:
            self._materialless_idle_count = 0

        # 生成と発話。反復ブロック等で喋れなかったら、条件を変えて
        # 最大2回まで再試行する（回復フロー。Phase 2）:
        #   1回目 … 回復ヒントを次回生成に付け、話題を引き直して同じ話者で再生成
        #   2回目 … 話者を相方へ交代し、詰まった側にクールダウンを設定
        # それでも喋れなければ、最後にフォールバック台詞で無言を回避する。
        response: LLMResponse | None = None
        idle_repeat = False
        current = speaker
        attempt = 0
        # no-speech 対策（2026-11-19）: このターンの生成予算と、
        # no-speech 系失敗の累計を初期化する
        self._gen_budget = self._IDLE_TURN_GEN_CAP
        no_speech_failures = 0
        while response is None and attempt < 3:
            attempt += 1
            if attempt > 1:
                # 前の試行の話題は解放済み（下の commit/release 部）なので
                # 引き直す。引き直しても材料がなければ再試行を打ち切る
                topic = None
                try:
                    topic = await self.plan_topic(kind=kind, query=query)
                except Exception as e:
                    logger.warning("topic re-planning failed", error=str(e))
                if self._planner is not None \
                        and not (topic is not None and topic.has_material):
                    break
            async with self._speak_lock:
                self._is_speaking = True
                try:
                    idle_mem = self._build_idle_memory(current)
                    if topic is not None and topic.has_material:
                        system_prompt = build_topic_prompt(
                            self._cfg,
                            documents=topic.documents,
                            long_term_memory_text=idle_mem,
                            kind=topic.kind,
                            query=topic.query,
                            question_author=topic.author,
                            resumed=topic.resumed,
                            arc_stage=topic.arc_stage,
                            prev_speech=topic.prev_speech,
                            intent=topic.intent,
                            **current.prompt_kwargs(),
                        )
                        await self._hub.send_log("info", f"自律トーク: {topic.label}")
                    else:
                        system_prompt = build_idle_prompt(
                            self._cfg, long_term_memory_text=idle_mem,
                            **current.prompt_kwargs())

                    failures_before = self._tag_failure_count()
                    response = await self._generate_and_speak(
                        system_prompt, current, route="idle",
                        compact_retry=attempt >= 2)
                    if response is not None:
                        idle_repeat = self._after_speech(
                            current, response, idle=True,
                            topic_kind=topic.kind if topic is not None else "")
                        if topic is not None and topic.has_material \
                                and topic.kind in ("news", "curiosity"):
                            self._last_web_topic_time = time.time()
                            self._remember_topic(topic, current)
                finally:
                    self._is_speaking = False

            # 話題の消費確定は発話できたときだけ（失敗時は予約を解放する）
            if topic is not None and self._planner is not None:
                try:
                    if response is not None:
                        self._planner.commit(topic)
                    else:
                        self._planner.release(topic)
                except Exception as e:
                    logger.warning("topic commit/release failed", error=str(e))

            if response is not None:
                break
            # この試行の失敗が no-speech 系（ヘッダのみ・本文なし）
            # だったかを数える
            no_speech_failures += max(
                0, self._tag_failure_count() - failures_before)
            if no_speech_failures >= self._NOSPEECH_TURN_LIMIT:
                # ヘッダのみの失敗を繰り返しても条件は好転しない。
                # 生成を打ち切ってフォールバック台詞に委ねる
                logger.info("idle speech recovery: stopping retries "
                            "(no-speech cap)", character=current.name,
                            no_speech_failures=no_speech_failures)
                break
            if attempt == 1:
                # 1回目の失敗: 回復ヒントを積んで同じ話者で再生成する。
                # 通常は再生成の温度も少し上げて生成条件自体を変える
                # （Phase 4）が、no-speech 系の失敗では温度上昇が
                # 逆効果になるため、ヒントのみにする（2026-11-19）
                self._recovery_hint_pending = current.key
                if no_speech_failures:
                    logger.info("idle speech recovery: retry with hint "
                                "(no temperature raise; no-speech failure)",
                                character=current.name)
                else:
                    self._arm_recovery_temperature(current)
                    logger.info("idle speech recovery: retry with hint",
                                character=current.name)
            elif attempt == 2:
                # 2回目の失敗: 話者を交代し、詰まった側をしばらく選ばない
                partner = self._other_agent(current)
                if partner is None or not partner.llm:
                    break
                self._speaker_cooldown_until[current.key] = (
                    time.time() + self._SPEAKER_COOLDOWN_SEC)
                logger.info("idle speech recovery: switching speaker",
                            from_character=current.name,
                            to_character=partner.name)
                current = partner

        self._gen_budget = None

        if response is None:
            # それでも喋れなければ、定型の繋ぎ台詞で無言を必ず回避する
            async with self._speak_lock:
                self._is_speaking = True
                try:
                    response = await self._speak_fallback_line(current)
                    if response is not None:
                        idle_repeat = self._after_speech(current, response,
                                                         idle=True)
                finally:
                    self._is_speaking = False

        if response is not None and not idle_repeat:
            await self._continue_banter(current, after="idle")
            self._schedule_consolidation()
            self._maybe_start_prefetch()

    def _backend_base_temperature(self, agent: CharacterAgent) -> float:
        """そのキャラの待機系バックエンドのベース温度（取れなければ 0.8）"""
        base = 0.8
        llm = agent.llm
        inner = getattr(llm, "idle_backend", llm) if llm is not None else None
        cfg = getattr(inner, "_cfg", None) if inner is not None else None
        if cfg is not None:
            try:
                base = float(getattr(cfg, "temperature", base))
            except (TypeError, ValueError):
                pass
        return base

    def _tag_failure_count(self) -> int:
        """タグ方式の失敗系（fatal + no_speech）の累計。json 方式では 0。

        生成の前後で差分を取ることで、その生成の失敗が no-speech 系
        （ヘッダのみ・本文なし）だったかを判定する。
        """
        if str(self._cfg.llm.protocol or "json").lower() != "tag":
            return 0
        stats = tag_parse_outcome_stats()
        return stats.get(TAG_OUTCOME_FATAL, 0) \
            + stats.get(TAG_OUTCOME_NO_SPEECH, 0)

    def _consume_gen_budget(self) -> bool:
        """idle ターンの生成予算を1つ消費する。予算外（None）なら常に True"""
        if self._gen_budget is None:
            return True
        if self._gen_budget <= 0:
            return False
        self._gen_budget -= 1
        return True

    def _arm_recovery_temperature(self, agent: CharacterAgent) -> None:
        """回復リトライ1回目の再生成だけ、生成温度を少し上げて条件を変える。

        ベース温度はそのキャラの待機系バックエンド設定から読む（取れなければ
        既定の 0.8）。上書きは _generate_and_speak の idle 経路で1回だけ
        消費され、消費されなくても短時間で失効する（Phase 4）。
        なお no-speech 系の失敗では呼び出し側が装填自体を止める
        （温度上昇は「ヘッダ直後で止まる」生成には逆効果。2026-11-19）。
        """
        base = self._backend_base_temperature(agent)
        raised = min(base + self._RECOVERY_TEMP_RAISE, self._RECOVERY_TEMP_MAX)
        self._temp_override[agent.key] = (
            raised, time.monotonic() + self._TEMP_OVERRIDE_TTL_SEC)
        logger.info("recovery temperature armed", character=agent.name,
                    base=base, temperature=raised)

    # ── 応答生成 + 発話 ───────────────────────────────────────

    # ── 先行生成（P1-C） ─────────────────────────────────────

    def _maybe_start_prefetch(self) -> None:
        """次の自律発話の先行生成を裏で始める（P1-C）。

        生成には時間がかかり、失敗すれば回復リトライも要る。その時間を
        現在の発話の再生・待機に重ねるため、発話が終わった時点で次の
        一言を先に作り始める。コメント対応が優先なので、キューに
        コメントがあるときや発話中は始めない。
        """
        if not self._cfg.orchestrator.prefetch_idle:
            return
        if not self._running or self._is_speaking:
            return
        if self._prefetched is not None:
            return
        if self._prefetch_task is not None and not self._prefetch_task.done():
            return
        if not self._comment_queue.empty():
            return
        self._prefetch_task = asyncio.create_task(self._prefetch_idle(),
                                                  name="idle_prefetch")

    async def _prefetch_idle(self) -> None:
        try:
            await self._prefetch_idle_inner()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning("idle prefetch failed", error=str(e))

    async def _prefetch_idle_inner(self) -> None:
        # 話者の選び方は _speak_idle と同じ（クールダウン中の話者は避ける）
        speaker = self._director.pick_idle_speaker(self._agents)
        if len(self._agents) > 1:
            until = self._speaker_cooldown_until.get(speaker.key, 0.0)
            if time.time() < until:
                partner = self._other_agent(speaker)
                if partner is not None and partner.llm:
                    speaker = partner
        if not speaker.llm:
            return
        topic: Topic | None = None
        try:
            topic = await self.plan_topic()
        except Exception as e:
            logger.warning("prefetch topic planning failed", error=str(e))
        if self._planner is not None \
                and not (topic is not None and topic.has_material):
            # 材料のない話題は先行生成しない（本経路と同じ判断）
            if topic is not None:
                try:
                    self._planner.release(topic)
                except Exception:
                    pass
            return
        # 生成の直前に、コメント対応が始まっていないか再確認する
        if self._is_speaking or not self._comment_queue.empty() \
                or not self._running:
            if topic is not None and self._planner is not None:
                try:
                    self._planner.release(topic)
                except Exception:
                    pass
            return
        idle_mem = self._build_idle_memory(speaker)
        if topic is not None and topic.has_material:
            system_prompt = build_topic_prompt(
                self._cfg,
                documents=topic.documents,
                long_term_memory_text=idle_mem,
                kind=topic.kind,
                query=topic.query,
                question_author=topic.author,
                resumed=topic.resumed,
                arc_stage=topic.arc_stage,
                prev_speech=topic.prev_speech,
                intent=topic.intent,
                **speaker.prompt_kwargs(),
            )
        else:
            system_prompt = build_idle_prompt(
                self._cfg, long_term_memory_text=idle_mem,
                **speaker.prompt_kwargs())
        # 生成は最大2試行。1回目がダメなら回復ヒント＋温度を変えて
        # 作り直す（本経路の回復フローの先行版）。それでもダメなら
        # 先行生成は諦め、本経路の通常フローに任せる
        response: LLMResponse | None = None
        actions: list = []
        for attempt in range(2):
            failures_before = self._tag_failure_count()
            response, actions = await self._generate_idle_response(
                system_prompt, speaker)
            if response is not None:
                break
            if attempt == 0:
                self._recovery_hint_pending = speaker.key
                if self._tag_failure_count() > failures_before:
                    # no-speech 系の失敗では温度を上げない（本経路と同方針）
                    logger.info("prefetch recovery: hint only "
                                "(no temperature raise; no-speech failure)",
                                character=speaker.name)
                else:
                    self._arm_recovery_temperature(speaker)
        if response is None:
            if topic is not None and self._planner is not None:
                try:
                    self._planner.release(topic)
                except Exception:
                    pass
            return
        # 計画の前提になった最終発話を記録する。消費時に変わっていたら
        # （コメント応答などの割り込みがあったら）計画は破棄する
        premise = ""
        if self._planner is not None:
            try:
                premise = self._planner.last_speech_text
            except Exception:
                premise = ""
        self._prefetched = {
            "at": time.time(),
            "agent_key": speaker.key,
            "topic": topic,
            "response": response,
            "actions": actions,
            "premise_speech": premise,
        }
        logger.info("idle speech prefetched", character=speaker.name,
                    speech=response.speech[:40])

    async def _generate_idle_response(self, system_prompt: str,
                                      agent: CharacterAgent,
                                      ) -> tuple[LLMResponse | None, list]:
        """発話せずに生成だけを行う（P1-C の先行生成用）。

        検査チェーン（挨拶テンプレ除去・メタリーク・反復/文型/定型相づち
        の差し止め・発話上限・応答フィルタ）は _generate_and_speak の
        一括生成経路と同じものを通す。喋るのは消費側（_speak_idle）。
        """
        llm = agent.llm
        if not llm:
            return None, []
        # 先行生成は「次のターン」の生成なので、ここもターン境界として
        # 適応 num_ctx の段変更を適用する
        llm.begin_turn()
        temperature: float | None = None
        armed = self._temp_override.pop(agent.key, None)
        if armed is not None and time.monotonic() <= armed[1]:
            temperature = armed[0]
        agent.speech_route = "idle"
        system_prompt = self._consume_theme_hint(system_prompt)
        system_prompt = self._consume_recovery_hint(system_prompt, agent)
        allow_intro = agent.speak_count == 0
        messages = agent.log.get_messages()
        gen_extra = {"temperature": temperature} \
            if temperature is not None else {}
        response = await llm.generate(system_prompt, messages, route="idle",
                                      **gen_extra)
        if response.skip or not response.speech:
            return None, []
        clean_speech, unity_actions = self._extract_unity_actions(
            response.speech)
        response.speech = agent.repair_speech(clean_speech)
        if not response.speech.strip():
            return None, []
        if not allow_intro:
            stripped = strip_greeting_template(response.speech)
            if stripped != response.speech:
                logger.warning("Prefetch greeting template stripped",
                               character=agent.name,
                               speech=response.speech[:60])
                response.speech = stripped
                if not response.speech.strip():
                    return None, []
        if contains_meta_leak(response.speech):
            logger.warning("Prefetch speech blocked (meta leak)",
                           character=agent.name, speech=response.speech[:60])
            return None, []
        response.speech = dedupe_repeated_sentences(response.speech)
        if not response.speech.strip():
            return None, []
        if len(re.sub(r"[\s、。，．！？!?'\"“”‘’]", "",
                      response.speech)) <= 3:
            logger.warning("Prefetch speech discarded (sole fragment)",
                           character=agent.name, speech=response.speech[:60])
            return None, []
        limit = _speech_limit(agent.cfg)
        truncated = self._truncate_to_speech_limit(response.speech, limit)
        if truncated != response.speech:
            response.speech = truncated
            if not response.speech.strip():
                return None, []
        verdict = self._filter.check(response.speech)
        if not verdict.ok:
            self._note_blocked(agent)
            return None, []
        if self._should_block_repeat(response.speech):
            logger.warning("Prefetch speech blocked (repeat)",
                           character=agent.name, speech=response.speech[:60])
            self._note_blocked(agent)
            return None, []
        if self._is_formulaic_opening(response.speech) \
                or self._is_formulaic_opening(clean_speech):
            self._recovery_hint_pending = agent.key
            return None, []
        if self._aizuchi_in_cooldown(response.speech):
            self._recovery_hint_pending = agent.key
            return None, []
        return response, unity_actions

    def _take_prefetched(self, speaker: CharacterAgent) -> dict | None:
        """先行生成を取り出す（P1-C）。使えない場合は破棄して None。

        保留中の回答・話題飽和の強制切替など、通常経路が優先すべき
        状況では先行生成を使わない（予約した話題は解放する）。
        """
        pf = self._prefetched
        if pf is None:
            return None
        self._prefetched = None
        usable = (
            time.time() - pf["at"] < self._PREFETCH_TTL_SEC
            and pf["agent_key"] == speaker.key
            and not self._force_switch_pending
            and not self._saturation_word()
        )
        if usable and self._planner is not None:
            try:
                usable = not self._planner.has_deliverable_answer()
            except Exception:
                usable = False
        if usable and self._planner is not None:
            # 前提発話の同一性チェック（STUDY §6.4-3）：計画後に別の
            # 発話（コメント応答など）が挟まっていたら、この先行生成は
            # 文脈がずれているので使わない
            premise = pf.get("premise_speech") or ""
            if premise and self._planner.last_speech_text != premise:
                logger.info("prefetch discarded (premise speech changed)")
                usable = False
        if not usable:
            topic = pf.get("topic")
            if topic is not None and self._planner is not None:
                try:
                    self._planner.release(topic)
                except Exception:
                    pass
            return None
        return pf

    async def _generate_and_speak(self, system_prompt: str,
                                  agent: CharacterAgent | None = None,
                                  route: str = "respond",
                                  compact_retry: bool = False) -> LLMResponse | None:
        """
        agent（省略時はリーダー）として応答を生成して発話する。
        ストリーミング対応バックエンドなら文単位で喋り、
        非対応・失敗時は従来の一括生成にフォールバックする。

        Returns:
            発話したLLMResponse（Unityアクションタグ除去済み）。
            何も喋らなかった／フィルタでブロックされた場合は None。
        """
        agent = agent or self._lead
        llm = agent.llm
        if not llm:
            return None
        # ターン境界: 適応 num_ctx の保留中の段変更はここでだけ適用する
        # （生成の途中で段が変わらないようにするため）
        llm.begin_turn()
        # 回復リトライ用に温度の上書きが積まれていれば、この生成（idle
        # 経路のみ）で消費する。期限切れの上書きは捨てる（Phase 4）
        temperature: float | None = None
        if route == "idle":
            armed = self._temp_override.pop(agent.key, None)
            if armed is not None and time.monotonic() <= armed[1]:
                temperature = armed[0]
        # ハイブリッド振り分け用に、この発話の種別をエージェントへ伝える
        # （ルーター以外のバックエンドでは単なる記録で、挙動は変わらない）
        agent.speech_route = route
        # テーマ反復の切り替えヒントが積まれていればこの発話に付ける
        # （コメント応答・掛け合い・自律トークの全経路がここを通る。Phase 4.6）
        system_prompt = self._consume_theme_hint(system_prompt)
        # 反復ブロック・文型反復の直後なら、回復ヒントも付ける（Phase 2）
        system_prompt = self._consume_recovery_hint(system_prompt, agent)
        # セッション開始からそのキャラの最初の発話だけ、挨拶・自己紹介
        # テンプレを正規の開始挨拶として許容する（2026-11-16 P0改修）。
        # speak_count は _after_speech で発話完了時に加算されるため、
        # 生成時点では「このセッションでまだ喋っていない」判定に使える。
        allow_intro = agent.speak_count == 0
        messages = agent.log.get_messages()
        if compact_retry:
            # 回復リトライ（P1-D）: 文脈を直近数件に切り詰め、要点を短く
            # 話す指示を足して、救済サイクルの生成時間そのものを縮める。
            # 人格・話題資料などの必須要素はシステムプロンプト側に残る
            messages = messages[-self._COMPACT_RETRY_MESSAGES:]
            system_prompt = system_prompt + (
                "\n\n## 再生成（短縮）\n"
                "直前の生成はうまくいきませんでした。直近の文脈だけで、"
                "話題の要点を短く、言い回しを変えて話してください。")
        self._tts_degraded_turn = False   # 発話ごとにTTS待ちの権利を1回ぶん戻す

        use_stream = (
            self._cfg.orchestrator.streaming_tts
            and llm.supports_streaming
            and agent.tts is not None
        )
        if route == "idle" and not self._consume_gen_budget():
            # idle ターンの生成予算を使い切った（_speak_idle が設定する）
            logger.info("idle generation budget exhausted",
                        character=agent.name)
            return None
        if use_stream:
            result = await self._speak_streaming(system_prompt, messages, agent,
                                                 allow_intro=allow_intro,
                                                 temperature=temperature)
            if result.blocked:
                await self._hub.send_log("warn", f"発話を中断しました: {result.blocked}")
                if result.blocked == self._BLOCKED_FORMULAIC:
                    # 文型レベルの反復はブロックとして数えず、回復ヒントで
                    # 再生成を促す（Phase 2 文型対策）
                    self._recovery_hint_pending = agent.key
                else:
                    self._note_blocked(agent)
                return None
            if result.spoke:
                return result.response
            # 1文も喋れていないので、一括生成でやり直す（二重発話にはならない）
            logger.info("Streaming produced no speech, falling back to non-streaming")

        if use_stream and route == "idle" \
                and not self._consume_gen_budget():
            # ストリーミングで予算を使い切った場合は一括再試行もしない
            logger.info("idle generation budget exhausted before fallback",
                        character=agent.name)
            return None
        # 温度の上書きは指定があるときだけ渡す（未指定時の呼び出し形状は
        # 従来どおりに保ち、温度引数を持たない実装とも共存する）
        gen_extra = {"temperature": temperature} if temperature is not None else {}
        if use_stream and result.tag_no_speech:
            # ヘッダのみで終わった生成の一括再試行は同一条件にしない。
            # 温度を下げて「ヘッダ直後で止まる」サンプリングから外しにいく
            # （回復の温度上書きが積まれていても、こちらを優先する）
            effective = temperature if temperature is not None \
                else self._backend_base_temperature(agent)
            lowered = max(effective - self._NOSPEECH_TEMP_DROP,
                          self._NOSPEECH_TEMP_FLOOR)
            gen_extra = {"temperature": lowered}
            if result.gen_meta.get("done_reason") == "length":
                # コンテキスト溢れ（プロンプトが num_ctx に達して生成枠が
                # 数トークンしか残っていない）で切れた生成は、温度を変えても
                # 決定的に再び切れる。文脈を直近だけに切り詰めて生成枠を
                # 確保する（P1-D の compact retry と同じ考え方）
                messages = messages[-self._COMPACT_RETRY_MESSAGES:]
                logger.info("non-streaming retry with compact context "
                            "(tag no-speech, length-truncated)",
                            character=agent.name, temperature=lowered,
                            prompt_eval_count=result.gen_meta.get(
                                "prompt_eval_count"))
            else:
                logger.info("non-streaming retry with lowered temperature "
                            "(tag no-speech)", character=agent.name,
                            temperature=lowered)
        response = await llm.generate(system_prompt, messages, route=route,
                                      **gen_extra)
        if response.skip or not response.speech:
            # ストリーミングでも一括でも喋れなかった。バックエンド自体が
            # 落ちている場合（二段とも失敗し、不健全を自己申告している）に
            # 限り、90秒周期のヘルスチェックを待たずに切替を試す
            if use_stream:
                await self._maybe_immediate_failover(agent, llm)
            return None

        clean_speech, unity_actions = self._extract_unity_actions(response.speech)
        response.speech = agent.repair_speech(clean_speech)
        if not response.speech.strip():
            return None

        if not allow_intro:
            stripped = strip_greeting_template(response.speech)
            if stripped != response.speech:
                logger.warning("Speech greeting template stripped",
                               character=agent.name,
                               speech=response.speech[:60])
                response.speech = stripped
                if not response.speech.strip():
                    return None

        if contains_meta_leak(response.speech):
            logger.warning("Speech blocked (meta leak)",
                           character=agent.name, speech=response.speech[:60])
            return None
        response.speech = dedupe_repeated_sentences(response.speech)
        if not response.speech.strip():
            return None

        # 断片だけの応答（「あ、」等）は喋らせない。ここでは応答全体を
        # 見るので、複数文の中にある短い文は対象にならない
        if len(re.sub(r"[\s、。，．！？!?'\"“”‘’]", "", response.speech)) <= 3:
            logger.warning("Speech discarded (sole fragment)",
                           character=agent.name, speech=response.speech[:60])
            return None

        # 発話上限の実行時検査（プロンプト指定は破られることがあるための
        # 保険。文境界で切り捨て、1文だけなら丸ごと落とす）
        limit = _speech_limit(agent.cfg)
        truncated = self._truncate_to_speech_limit(response.speech, limit)
        if truncated != response.speech:
            logger.warning("Speech truncated at limit",
                           character=agent.name, limit=limit,
                           speech=response.speech[:60])
            response.speech = truncated
            if not response.speech.strip():
                return None

        verdict = self._filter.check(response.speech)
        if not verdict.ok:
            await self._hub.send_log("warn", f"発話をブロックしました: {verdict.reason}")
            self._note_blocked(agent)
            return None

        if route == "idle" and self._should_block_repeat(response.speech):
            logger.warning("Speech blocked (repeat, non-streaming)",
                           character=agent.name, speech=response.speech[:60])
            self._note_blocked(agent)
            return None

        if route == "idle" and (self._is_formulaic_opening(response.speech)
                                or self._is_formulaic_opening(clean_speech)):
            # 文型レベルの反復はブロックとして数えず、回復ヒントで再生成を促す
            logger.info("Speech diverted (formulaic opening, non-streaming)",
                        character=agent.name, speech=response.speech[:60])
            self._recovery_hint_pending = agent.key
            return None

        if route == "idle" and self._aizuchi_in_cooldown(response.speech):
            # 定型相づちの再演もブロックとして数えず、回復ヒントで再生成を促す
            logger.info("Speech diverted (formulaic aizuchi, non-streaming)",
                        character=agent.name, speech=response.speech[:60])
            self._recovery_hint_pending = agent.key
            return None

        await self._speak(response, agent)
        if unity_actions:
            asyncio.create_task(self._handle_unity_actions(unity_actions))
        return response

    async def _maybe_immediate_failover(self, agent: CharacterAgent,
                                        llm: LLMBackend | None) -> None:
        """二段失敗時の即時フェイルオーバー。

        ストリーミングが1文も喋れず、一括生成も空だった場合に、バックエンドが
        不健全を自己申告している（is_healthy=False）ときに限り、90秒周期の
        ヘルスチェックを待たずに failover_llm() を試す。モデルが skip を
        返しただけの通常ケースでは is_healthy が True のままなので発動しない。
        短時間の往復（flapping）を防ぐため、キャラごとに180秒の
        クールダウンを置く（クールダウンは試行時点で記録する）。
        """
        if llm is None or llm.is_healthy:
            return
        now = time.monotonic()
        if now - self._immediate_failover_at.get(agent.key, 0.0) < 180.0:
            return
        self._immediate_failover_at[agent.key] = now
        if agent.failover_llm():
            label = f"{agent.name}: " if self.is_ensemble else ""
            await self._hub.send_log("warn", f"LLM切替（即時）: {label}{agent.llm.name}")
            self._status.llm_backend = agent.llm.name if agent.llm else "none"

    @staticmethod
    def _truncate_to_speech_limit(speech: str, limit: int) -> str:
        """発話上限に合わせて文境界で切り捨てる。

        文が1つしかなく、それ自体が上限を超える場合は、文の途中で切らずに
        丸ごと無効（空文字）にする。
        """
        if len(speech) <= limit:
            return speech
        parts = [p for p in re.split(r"(?<=[。！？!?])", speech) if p]
        out: list[str] = []
        total = 0
        for part in parts:
            if total + len(part) > limit:
                break
            out.append(part)
            total += len(part)
        return "".join(out).strip()

    def _voice_kwargs(self, agent: CharacterAgent) -> dict:
        """アバター・字幕へのメッセージに付ける「誰の発話か」"""
        return {"character": agent.key, "speaker": agent.name if self.is_ensemble else None}

    async def _speak_streaming(self, system_prompt: str, messages: list,
                               agent: CharacterAgent | None = None,
                               allow_intro: bool = False,
                               temperature: float | None = None) -> _StreamSpeakResult:
        """
        LLMのストリーム出力を文単位でTTSに流し込みながら発話する。

        1文目を喋っている間に2文目を合成するので、全文の生成完了を待つ従来方式に比べて
        発話開始までの待ち時間が大幅に短くなる。
        """
        agent = agent or self._lead
        cfg = self._cfg.orchestrator
        if str(self._cfg.llm.protocol or "json").lower() == "tag":
            # タグ方式（Phase 5）: flip の定型文の名前解決のため、
            # そのキャラのフリップカタログ名をパーサへ渡す
            flip_names = set(resolve_effects(self._cfg, agent.cfg).flip)
            parser = TagSpeechStreamParser(
                min_chunk_chars=cfg.stream_min_chunk_chars,
                max_chunk_chars=cfg.stream_max_chunk_chars,
                flip_names=flip_names,
            )
        else:
            parser = SpeechStreamParser(
                min_chunk_chars=cfg.stream_min_chunk_chars,
                max_chunk_chars=cfg.stream_max_chunk_chars,
            )
        # (文, 合成タスク) を順に渡すキュー。maxsizeで先読み合成を2文までに抑える
        synth_queue: asyncio.Queue = asyncio.Queue(maxsize=2)
        unity_actions: list[dict] = []
        # 演出エフェクト: ストリームから抽出し、1文目の発話後に1回だけ配送
        effect_box: dict = {"raw": None, "done": False}

        def capture_effect() -> None:
            if effect_box["raw"] is None:
                effect_box["raw"] = parser.extract_effect()

        # 内心: effect と同方式で抽出し、発話が成立した後に1回だけ表示する
        # （TTS キューには一切入れない。画面の吹き出し表示だけの別チャネル）
        thought_box: dict = {"text": "", "done": False}

        def capture_thought() -> None:
            if not thought_box["text"]:
                thought_box["text"] = parser.extract_thought()
        state = _StreamSpeakResult()
        voice = self._voice_kwargs(agent)

        pushed_count = 0

        seen_sentences: set[str] = set()  # この応答内で喋った文（繰り返し除去用）
        # 発話上限の実行時検査（保険）用の累計と打ち切りフラグ
        speech_limit = _speech_limit(agent.cfg)
        spoken_chars = 0
        over_limit = False

        async def push(sentence: str) -> bool:
            """文を検査してから合成キューへ。中断すべきときは False を返す。"""
            nonlocal pushed_count, spoken_chars, over_limit
            clean, actions = self._extract_unity_actions(sentence)
            unity_actions.extend(actions)
            if not clean.strip():
                return True
            pre_repair = clean
            clean = agent.repair_speech(clean)
            # 発話冒頭の挨拶・自己紹介テンプレは、セッション最初の発話
            # 以外では落とす（会話中盤の自己紹介割り込み対策。2026-11-16 P0）
            if pushed_count == 0 and not allow_intro:
                stripped = strip_greeting_template(clean)
                if stripped != clean:
                    logger.warning("Streaming greeting template stripped",
                                   character=agent.name, text=clean[:60])
                    clean = stripped
                    if not clean.strip():
                        return True
            # タグ断片を含む文は読み上げずに捨てる（2026-11-14対策）
            if _TAG_FRAGMENT_RE.search(clean):
                logger.warning("Streaming sentence dropped (tag fragments)",
                               character=agent.name, text=clean[:60])
                return True
            if contains_meta_leak(clean):
                logger.warning("Streaming sentence dropped (meta leak)",
                               character=agent.name, text=clean[:60])
                return True
            _norm = re.sub(r"[\s、。，．！？!?'\"“”‘’]", "", clean)
            if len(_norm) >= 6:
                if _norm in seen_sentences:
                    logger.warning("Streaming sentence dropped (repeat in response)",
                                   character=agent.name, text=clean[:60])
                    return True
                seen_sentences.add(_norm)
            # 自律発話（idle経路）の1文目が直近発話の反復なら発話自体を止める
            if pushed_count == 0 and agent.speech_route == "idle" \
                    and self._should_block_repeat(clean):
                state.blocked = "repeat speech"
                logger.warning("Streaming speech blocked (repeat)",
                               character=agent.name, text=clean[:60])
                return False
            # 文型レベルの反復（相づち＋定型賛辞のみ・内容語なし）は
            # 差し止めて回復リトライに回す。ブロックとしては数えない
            if pushed_count == 0 and agent.speech_route == "idle" \
                    and (self._is_formulaic_opening(clean)
                         or self._is_formulaic_opening(pre_repair)):
                state.blocked = self._BLOCKED_FORMULAIC
                logger.info("Streaming speech diverted (formulaic opening)",
                            character=agent.name, text=clean[:60])
                return False
            # クールダウン中の定型相づち型の文は読み上げずに捨てる（P2-A）。
            # 文中の出現も拾うため1文目以外にも適用する。全て捨てられた
            # 場合は一括生成→回復リトライへ流れる
            if agent.speech_route == "idle" \
                    and self._aizuchi_in_cooldown(clean):
                logger.info("Streaming sentence dropped (formulaic aizuchi)",
                            character=agent.name, text=clean[:60])
                return True
            # 発話上限の実行時検査（保険）。上限を超える文は喋らず、以降の文も
            # 切り捨てて応答を文境界で終わらせる。1文目自体が上限超えの場合も
            # 同様に落とし、一括生成へのフォールバックに委ねる
            if over_limit or spoken_chars + len(clean) > speech_limit:
                if not over_limit:
                    logger.warning("Streaming speech truncated at limit",
                                   character=agent.name, limit=speech_limit,
                                   text=clean[:60])
                over_limit = True
                return True
            spoken_chars += len(clean)
            pushed_count += 1
            verdict = self._filter.check(clean)
            if not verdict.ok:
                state.blocked = verdict.reason
                logger.warning("Streaming speech blocked", reason=verdict.reason)
                return False
            emotion = parser.peek("emotion") or "neutral"
            task = asyncio.create_task(self._synthesize(clean, emotion, agent.tts))
            await synth_queue.put((clean, emotion, task))
            return True

        async def produce() -> None:
            # 温度の上書きは指定があるときだけ渡す（_generate_and_speak と同じ方針）
            stream_extra = {"temperature": temperature} \
                if temperature is not None else {}
            try:
                async for delta in agent.llm.generate_stream(
                        system_prompt, messages, route=agent.speech_route,
                        **stream_extra):
                    if parser.is_skip:
                        return
                    for sentence in parser.feed(delta):
                        if not await push(sentence):
                            return
                    capture_effect()
                    capture_thought()
                for sentence in parser.flush():
                    if not await push(sentence):
                        return
                capture_effect()
                capture_thought()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning("Streaming producer error",
                               error_type=type(e).__name__, error=str(e))
            finally:
                await synth_queue.put(None)

        async def dispatch_effect_once() -> None:
            """抽出済みエフェクトを1回だけ配送する（発話が成立した後のみ）"""
            if effect_box["done"] or not state.spoke:
                return
            capture_effect()
            if effect_box["raw"] is not None:
                effect_box["done"] = True
                await self._effects.dispatch(effect_box["raw"], agent=agent)

        async def dispatch_thought_once() -> None:
            """抽出済みの内心を1回だけ表示する（発話が成立した後のみ）"""
            if thought_box["done"] or not state.spoke:
                return
            capture_thought()
            if thought_box["text"]:
                thought_box["done"] = True
                await self._thoughts.dispatch(
                    thought_box["text"], agent=agent,
                    speech="".join(state.spoken))

        async def consume() -> None:
            while True:
                item = await synth_queue.get()
                if item is None:
                    break
                sentence, emotion, task = item
                await self._hub.send_chat("assistant", sentence, emotion=emotion,
                                          **self._chat_kwargs(agent))
                announced = False
                if not state.spoke:
                    # 1文目の再生直前にアバターへ表情・モーションを送る
                    self._speaking_character = agent.key
                    await self._hub.send_speech_start(sentence, emotion, **voice)
                    await self._hub.send_motion(parser.peek("motion") or "idle",
                                                character=agent.key)
                    announced = True
                await self._play_sentence(sentence, emotion, task, announced=announced,
                                          agent=agent)
                state.spoke = True
                state.spoken.append(sentence)
                await dispatch_effect_once()
                await dispatch_thought_once()

        producer = asyncio.create_task(produce())
        try:
            await consume()
        finally:
            await producer
            # 中断した場合、合成途中のタスクを片付ける
            while not synth_queue.empty():
                leftover = synth_queue.get_nowait()
                if leftover:
                    leftover[2].cancel()

        if not state.spoke and isinstance(parser, TagSpeechStreamParser):
            # タグ方式で1文も生まなかったストリームは、計測と診断ログに
            # 残す（従来はフォールバック先の成否に埋もれて内訳が見えず、
            # 成功率が実態より良く見えていた。2026-11-19 観測追加）
            meta: dict = {}
            try:
                if agent.llm is not None:
                    meta = agent.llm.last_generation_meta()
            except Exception:
                meta = {}
            state.gen_meta = meta
            if parser.report_no_speech(meta):
                state.tag_no_speech = True

        if state.spoke:
            # エフェクトはJSON末尾にあるため、生成完了が発話消費より遅れた
            # 場合はここで最終試行する（配送漏れ防止）。内心も同じ
            await dispatch_effect_once()
            await dispatch_thought_once()
            await self._hub.send_speech_end(character=agent.key)
            self._speaking_character = ""
            state.response = parser.build_response()
            # 履歴には実際に喋った文（アクションタグ除去済み）だけを残す
            state.response.speech = "".join(state.spoken)
            logger.info("Spoke (streaming)",
                        character=agent.name,
                        speech=state.response.speech[:40],
                        emotion=state.response.emotion.value,
                        motion=state.response.motion.value)
            if unity_actions:
                asyncio.create_task(self._handle_unity_actions(unity_actions))
        return state

    def _chat_kwargs(self, agent: CharacterAgent) -> dict:
        """チャット画面に流す発話の名札（1体運用でも付けておく）"""
        return {"author": agent.name, "character": agent.key}

    async def _synthesize(self, text: str, emotion: str, tts: TTSBackend | None = None):
        """
        TTS合成を1本ずつ実行する（ローカルTTSサーバの詰まり対策）。

        ローカルTTSサーバはハングすることがある。1文ごとにタイムアウトを待つと
        字幕がその都度止まってしまうので、連続で失敗したらしばらく合成を丸ごと
        飛ばし（字幕のみモード）、クールダウン後に1回だけ試して復帰を確認する。
        """
        tts = tts or self._lead.tts
        empty = TTSResult(audio_wav=b"", duration_ms=0)
        if tts is None:
            return empty

        # 字幕のみモード中は、クールダウンが明けるまで合成を試みない
        if self._tts_skip_until and time.monotonic() < self._tts_skip_until:
            return empty

        timeout = self._cfg.tts.timeout_sec
        # 序盤保護（P3-B）: 開始直後は合成が冷えていて初回が間に合わない
        # ことがあるため、タイムアウトを長めに取り、1回だけ再試行してから
        # 音声なしに落とす。通常時（1回で諦める）の挙動は変えない
        early = self._in_tts_early_window()
        if early:
            timeout = max(timeout, self._TTS_EARLY_TIMEOUT_SEC)
        attempts = 2 if early else 1
        async with self._tts_lock:
            # 同じ発話内で既にタイムアウトしているなら、残りの文で待ち直さない
            # （1発話あたりのTTS待ちを最大1回に抑える）
            if self._tts_degraded_turn:
                return empty
            for attempt in range(attempts):
                try:
                    return await asyncio.wait_for(
                        tts.synthesize(text, emotion), timeout=timeout)
                except asyncio.TimeoutError:
                    if attempt + 1 < attempts:
                        logger.warning(
                            "TTS timed out in early window, retrying once",
                            timeout_sec=timeout, text=text[:20])
                        continue
                    self._tts_timeouts += 1
                    self._tts_degraded_turn = True
                    logger.warning("TTS timed out, continuing without audio",
                                   timeout_sec=timeout, consecutive=self._tts_timeouts,
                                   text=text[:20])
                    if self._tts_timeouts >= self._TTS_FAIL_LIMIT:
                        self._tts_skip_until = time.monotonic() + self._TTS_COOLDOWN_SEC
                        await self._hub.send_log(
                            "error",
                            f"TTSが{self._tts_timeouts}回連続で応答しません。"
                            f"{int(self._TTS_COOLDOWN_SEC)}秒間は字幕のみで進行します"
                            f"（TTSサーバの再起動を検討してください）")
                    return empty
            return empty

    def _note_tts_success(self) -> None:
        """合成が成功したら字幕のみモードを解除する"""
        if self._tts_timeouts:
            logger.info("TTS recovered", after_timeouts=self._tts_timeouts)
            asyncio.create_task(self._hub.send_log("info", "TTSが復帰しました"))
        self._tts_timeouts = 0
        self._tts_skip_until = 0.0
        self._tts_degraded_turn = False

    async def _play_sentence(self, sentence: str, emotion: str, task,
                             announced: bool = False,
                             agent: CharacterAgent | None = None) -> None:
        """
        1文ぶんのTTS結果を待って再生する。
        announced=True なら、この文の字幕は既に送信済みなので重複送信しない。
        """
        voice = self._voice_kwargs(agent or self._lead)
        if task is None:
            await asyncio.sleep(max(len(sentence) * 0.12, 0.5))
            return
        try:
            tts_result = await task
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning("TTS synthesize failed", error=str(e))
            await asyncio.sleep(max(len(sentence) * 0.12, 0.5))
            return

        if not tts_result.audio_wav:
            # TTS失敗時は字幕だけ出して、読み上げ相当の時間ぶん待つ
            if not announced:
                await self._hub.send_speech_start(sentence, emotion, **voice)
            await asyncio.sleep(max(len(sentence) * 0.12, 0.5))
            return

        self._note_tts_success()
        await self._hub.send_speech_start(sentence, emotion,
                                          audio_url=self._cache_audio(tts_result.audio_wav),
                                          **voice)
        await self._audio.play(tts_result.audio_wav)

    # ── 共通発話処理 ──────────────────────────────────────────

    async def _speak(self, response: LLMResponse, agent: CharacterAgent | None = None) -> None:
        agent = agent or self._lead
        voice = self._voice_kwargs(agent)
        emotion_str = response.emotion.value
        motion_str = response.motion.value
        self._speaking_character = agent.key

        # 演出エフェクト（任意）: 発話開始に合わせて配送する。
        # SE は TTS 合成より前に背景再生が始まり、カットイン/フリップは
        # 画面へ speech_start とほぼ同時に届く
        if response.effect:
            await self._effects.dispatch(response.effect, agent=agent)

        # 内心（任意）: 声には出さず、画面の吹き出し表示だけに配送する
        if response.thought:
            await self._thoughts.dispatch(response.thought, agent=agent,
                                          speech=response.speech)

        await self._hub.send_chat("assistant", response.speech, emotion=emotion_str,
                                  **self._chat_kwargs(agent))
        # 先にアバターに感情を伝える
        await self._hub.send_speech_start(response.speech, emotion_str, **voice)
        await self._hub.send_motion(motion_str, character=agent.key)

        # TTS 合成 → 音声出力
        if agent.tts:
            tts_result = await self._synthesize(response.speech, emotion_str, agent.tts)
            if tts_result.audio_wav:
                self._note_tts_success()
                await self._hub.send_speech_start(
                    response.speech, emotion_str,
                    audio_url=self._cache_audio(tts_result.audio_wav),
                    **voice,
                )
                await self._audio.play(tts_result.audio_wav)
                await asyncio.sleep(0.1)
            else:
                # TTS失敗時は推定時間で待機
                await asyncio.sleep(max(len(response.speech) * 0.12, 1.0))
        else:
            await asyncio.sleep(max(len(response.speech) * 0.12, 1.0))

        await self._hub.send_speech_end(character=agent.key)
        self._speaking_character = ""
        logger.info("Spoke", character=agent.name, speech=response.speech[:40],
                    emotion=emotion_str, motion=motion_str)

    def _cache_audio(self, wav: bytes, keep: int = 20) -> str:
        """
        WAVをキャッシュに保存してブラウザ用のURLを返す。
        ファイル名はタイムスタンプ+乱数（同時発話でも上書きしない）。
        """
        fname = f"{int(time.time() * 1000)}_{uuid.uuid4().hex[:6]}.wav"
        (self._audio_cache / fname).write_bytes(wav)
        self._cleanup_audio_cache(keep=keep)
        return f"/audio/{fname}"

    def _cleanup_audio_cache(self, keep: int = 10) -> None:
        """古い音声キャッシュファイルを削除する"""
        try:
            wav_files = sorted(self._audio_cache.glob("*.wav"), key=lambda p: p.stat().st_mtime)
            for f in wav_files[:-keep]:
                f.unlink(missing_ok=True)
        except Exception:
            pass  # 掃除失敗は無視

    async def _probe_primary(self, agent: CharacterAgent) -> bool:
        """fallback スロットにいる primary の回復を能動的に確認する（Phase 4）。

        probe() を持つバックエンド（Ollama／HybridRouter）では実際に
        生成できるかまで確認し、健全フラグも実結果で更新される。
        probe を持たない系統では能動確認ができないため、健全フラグの
        自己申告だけを根拠にする（フラグは実リクエストの成功でしか
        立たないので、未使用のバックエンドでは通常 False のままで、
        結果として failback しない側に倒れる）。
        """
        backend = agent.llm_fallback
        if backend is None:
            return False
        probe = getattr(backend, "probe", None)
        if probe is None:
            return backend.is_healthy
        try:
            return await probe()
        except Exception as e:
            logger.warning("failback probe error", character=agent.name,
                           error=str(e))
            return False

    # ── Loop D: ヘルスチェック ────────────────────────────────

    async def _health_check_loop(self) -> None:
        while self._running:
            await asyncio.sleep(90.0)
            await self._health_check_cycle()

    async def _health_check_cycle(self) -> None:
        """ヘルスチェック1回分（フェイルオーバー／フェイルバック／ソース切替）"""
        # LLM フォールバック（キャラごと）
        for agent in self._agents:
            if agent.failover_llm():
                label = f"{agent.name}: " if self.is_ensemble else ""
                await self._hub.send_log("warn", f"LLM切替: {label}{agent.llm.name}")

        # LLM フェイルバック（キャラごと。Phase 4）: fallback へ退避中の
        # キャラは、退避先スロットにいる primary へ能動プローブを投げて
        # 回復を確認する。プローブが連続で成功し（ヒステリシス）、
        # 現行側も健全なときだけ failback_llm() で戻す。単発の成功や
        # 無条件の自動復帰では戻さない。
        for agent in self._agents:
            if not agent.on_fallback:
                self._failback_probe_success.pop(agent.key, None)
                continue
            if not await self._probe_primary(agent):
                self._failback_probe_success[agent.key] = 0
                continue
            successes = self._failback_probe_success.get(agent.key, 0) + 1
            self._failback_probe_success[agent.key] = successes
            logger.info("failback probe succeeded",
                        character=agent.name, consecutive=successes,
                        required=self._FAILBACK_PROBE_SUCCESSES)
            if successes < self._FAILBACK_PROBE_SUCCESSES:
                continue
            if agent.failback_llm():
                self._failback_probe_success.pop(agent.key, None)
                label = f"{agent.name}: " if self.is_ensemble else ""
                await self._hub.send_log("info", f"LLM復帰: {label}{agent.llm.name}")
        self._status.llm_backend = self._lead.llm.name if self._lead.llm else "none"

        # Source フォールバック
        if self._source and not self._source.is_healthy and self._source_fallback:
            logger.warning("Comment source unhealthy, switching to fallback",
                           primary=self._source.name, fallback=self._source_fallback.name)
            await self._source.stop()
            self._source, self._source_fallback = self._source_fallback, self._source
            self._status.comment_source = self._source.name
            await self._hub.send_log("warn", f"コメントソース切替: {self._source.name}")
            await self._source.start(self._status.video_id)
            self._restart_ingest()

    def _restart_ingest(self) -> None:
        for t in self._tasks:
            if t.get_name() == "ingest":
                t.cancel()
        self._tasks = [t for t in self._tasks if not t.done() and t.get_name() != "ingest"]
        self._tasks.append(asyncio.create_task(self._comment_ingest_loop(), name="ingest"))

    # ── Loop E: ステータス配信 ────────────────────────────────

    async def _status_broadcast_loop(self) -> None:
        while self._running:
            await asyncio.sleep(2.0)
            self._status.uptime_sec = time.time() - self._start_time
            self._status.queue_size = self._comment_queue.qsize()
            self._status.is_speaking = self._is_speaking
            self._status.comment_source = self._source.name if self._source else "none"
            await self._hub.send_status(self.status_snapshot())

    def status_snapshot(self) -> dict:
        return {
            "comment_source": self._status.comment_source,
            "llm_backend": self._status.llm_backend,
            "tts_backend": self._status.tts_backend,
            "is_speaking": self._status.is_speaking,
            "current_emotion": self._status.current_emotion.value,
            "queue_size": self._status.queue_size,
            "last_comment": self._status.last_comment,
            "last_speech": self._status.last_speech,
            "video_id": self._status.video_id,
            "uptime_sec": round(self._status.uptime_sec, 1),
            "speaking_character": self._speaking_character,
            "theme_hint_count": self._theme_hint_count,
            "characters": self.characters_status(),
        }

    # ── 記憶整理 ──────────────────────────────────────────────

    def _consolidation_trigger_turns(self) -> int:
        """記憶整理を発火させるターン数。ローカルチャット中は専用の間隔を使う。"""
        if self.is_local_chat and self._cfg.local_chat.consolidate_every_turns > 0:
            return self._cfg.local_chat.consolidate_every_turns
        return self._cfg.orchestrator.consolidation_trigger_turns

    async def _run_consolidation(self, triggered_by: str = "manual",
                                 agent: CharacterAgent | None = None):
        """
        記憶整理を実行する。agent を省略すると出演者全員ぶんを順に整理し、
        リーダーの結果を返す（ダッシュボード・チャット画面の表示用）。
        """
        if agent is not None:
            return await self._consolidate_agent(agent, triggered_by)
        lead_result = None
        for a in self._agents:
            result = await self._consolidate_agent(a, triggered_by)
            if a.is_lead:
                lead_result = result
        return lead_result

    async def _consolidate_agent(self, agent: CharacterAgent, triggered_by: str = "manual"):
        """
        1体ぶんの記憶整理。

        整理に成功したときだけ整理待ちバッファとターンカウンタをリセットする。
        以前は失敗時もリセットしていたため、LLMエラーで整理できなかった会話が
        そのまま捨てられ、次の整理まで更にNターン待つことになっていた。
        """
        log = agent.log
        if not agent.llm or not log.get_pending() or agent.key in self._consolidating:
            return None
        label = f"{agent.name}: " if self.is_ensemble else ""
        self._consolidating.add(agent.key)
        try:
            logger.info("Running memory consolidation", character=agent.name,
                        trigger=triggered_by, turns=log.turns_since_consolidation,
                        pending=log.pending_size)
            await self._hub.send_log("info",
                f"{label}記憶整理を実行中（{log.turns_since_consolidation}ターン経過）...")
            try:
                result = await asyncio.wait_for(
                    agent.consolidator.consolidate(log, agent.llm), timeout=120.0)
            except asyncio.TimeoutError:
                logger.warning("Memory consolidation timed out", trigger=triggered_by)
                await self._hub.send_log("warn", f"{label}記憶整理がタイムアウトしました（次回に持ち越します）")
                return None
            except Exception as e:
                logger.warning("Memory consolidation failed",
                               error_type=type(e).__name__, error=repr(e))
                await self._hub.send_log("warn", f"{label}記憶整理に失敗しました: {e}")
                return None
        finally:
            self._consolidating.discard(agent.key)

        if result.failed:
            # 会話は保持したまま次の機会に再挑戦する
            await self._hub.send_log("warn", f"{label}記憶整理に失敗: {result.reason}")
            return result

        log.mark_consolidated()
        if result.added > 0:
            msg = f"{label}記憶を {result.added} 件おぼえました"
            if result.skipped_duplicates:
                msg += f"（既知 {result.skipped_duplicates} 件）"
            if result.forgotten:
                msg += f"／古い記憶 {result.forgotten} 件を忘れました"
            await self._hub.send_log("info", msg)
        elif result.skipped_duplicates:
            await self._hub.send_log("info",
                f"{label}新しい記憶はありませんでした（既知 {result.skipped_duplicates} 件）")
        else:
            await self._hub.send_log("info", f"{label}記憶整理完了（{result.reason}）")
        if agent.is_lead:
            await self._broadcast_memory_stats()
        return result

    async def _broadcast_memory_stats(self) -> None:
        lead = self._lead
        if lead.memory_store is None:
            return
        await self._hub.broadcast("dashboard", {
            "type": "memory_stats",
            **lead.memory_store.stats(),
            "pending_turns": lead.log.turns_since_consolidation,
            "character": lead.name,
        })

    async def _consolidation_loop(self) -> None:
        """
        定期的に短期記憶をエピソード記憶へ整理する。

        起点は「前回の試行時刻」。以前は「前回成功した時刻」を見ており、
          - 一度も成功していないと永久に発火しない
          - 一度成功した後に「記憶なし」が続くと時刻が更新されず毎分発火し続ける
        という2つの不具合があった。
        """
        last_run = time.monotonic()
        while self._running:
            try:
                await asyncio.sleep(30.0)
                if not self._running:
                    break
                interval = self._consolidation_interval_min
                if interval <= 0:
                    continue
                if (time.monotonic() - last_run) / 60.0 < interval:
                    continue
                last_run = time.monotonic()
                for agent in self._agents:
                    if agent.llm and agent.log.get_pending():
                        await self._consolidate_agent(agent, triggered_by="schedule")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("consolidation_loop error", error=str(e))
                await asyncio.sleep(60.0)

    async def set_video_id(self, video_id: str) -> None:
        """Dashboardからvideo_idを変更する"""
        video_id = extract_youtube_id(video_id)
        if self._source:
            await self._source.stop()
        self._status.video_id = video_id
        self._source, self._source_fallback = self._build_sources(video_id)
        await self._source.start(video_id)
        self._restart_ingest()
        logger.info("video_id updated", video_id=video_id)

    # ── 演出エフェクト ────────────────────────────────────

    async def trigger_effect(self, effect_type: str, name: str = "",
                             text: str = "", character: str = "") -> dict:
        """エフェクトの手動発火（動作確認・ダッシュボードからの演出用）。

        character はキャラの id（省略時はキャラなし＝全体設定で解決し、
        全アバターページへ届く）。自発エフェクトと同じ検証・配送経路を
        通るが、クールダウンは無視する。
        """
        agent = self.get_agent(character) if character else None
        raw = {"type": effect_type, "name": name, "text": text}
        return await self._effects.dispatch(raw, agent=agent, manual=True)

    def effects_catalog(self, character: str = "") -> dict:
        """そのキャラが使えるエフェクトのカタログ（確認用）"""
        agent = self.get_agent(character) if character else None
        resolved = resolve_effects(self._cfg, agent.cfg if agent else None)
        return catalog_summary(resolved)
