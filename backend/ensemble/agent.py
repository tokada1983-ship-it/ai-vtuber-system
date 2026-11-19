"""CharacterAgent: 出演キャラクター1体ぶんの「頭」と「声」

以前は Orchestrator がキャラ1体ぶんの LLM・TTS・記憶・会話ログを直接持っていたため、
2体目を出すには Orchestrator ごと複製するしかなかった。キャラに属するものは
ここにまとめ、Orchestrator は「誰に喋らせるか」と「発話の順番」だけを受け持つ。

キャラごとに持つもの:
  - LLM（primary / fallback）と TTS（声）
  - 会話ログ（自分視点。相方の発言は <partner_speech> として入る）
  - コア記憶・エピソード記憶・視聴者との関係・チャット履歴
  - 記憶整理器・人格の補正
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

from backend import factory
from backend.config import AppConfig, AvatarConfig, CharacterConfig
from backend.llm.base import LLMBackend
from backend.llm.persona_validator import PersonaValidator
from backend.llm.prompt_builder import build_cast_note, extract_memory_keywords
from backend.logger import get_logger
from backend.memory.chat_history import ChatHistory
from backend.memory.conversation_log import ConversationLog
from backend.memory.long_term_memory import LongTermMemory
from backend.memory.memory_consolidator import MemoryConsolidator
from backend.memory.memory_store import MemoryStore, migrate_legacy_section
from backend.memory.mind_store import MindSnapshot, MindStore
from backend.memory.rag_retriever import RAGRetriever
from backend.memory.viewer_store import ViewerProfileStore
from backend.models import Comment, Emotion
from backend.safety.response_filter import ResponseFilter
from backend.tts.base import TTSBackend

logger = get_logger(__name__)


class CharacterAgent:
    """出演キャラクター1体"""

    def __init__(
        self,
        config: AppConfig,
        character: CharacterConfig,
        *,
        is_lead: bool,
        response_filter: ResponseFilter,
    ):
        self.app = config
        self.cfg = character
        self.is_lead = is_lead
        self.partners: list[CharacterConfig] = []

        self.llm: LLMBackend | None = None
        self.llm_fallback: LLMBackend | None = None
        # failover で fallback 側に退避中かどうか（failback 判定用。Phase 4）。
        # failover_llm() の入れ替えで True、failback_llm() と作り直しで False。
        self._on_fallback: bool = False
        # 現在の発話がどちらの経路か（"idle" / "respond"）。
        # Orchestrator が発話直前にセットし、ハイブリッド振り分けが読む。
        self.speech_route: str = "respond"
        self.tts: TTSBackend | None = None

        self.log = ConversationLog(max_turns=config.orchestrator.response_history_turns)
        self.persona_validator = PersonaValidator()
        # 記憶は2層構成
        #   コア記憶     : memory.md（人が書く設定。自動追記しない）
        #   エピソード記憶: MemoryStore（配信中に自動で貯まる。上限・忘却つき）
        self.rag = RAGRetriever()            # コア記憶（Markdown）
        self.episodic_rag = RAGRetriever()   # エピソード記憶（1件1チャンク）
        self.long_term_memory: LongTermMemory | None = None
        self.memory_store: MemoryStore | None = None
        # 「個」のストア（意見・意図・選好・自己物語）。memory_store と同じ
        # memories.db に同居するが、担当するテーブルは分離している。
        self.mind_store: MindStore | None = None
        # プロンプトに載せる自己のスナップショット。発話のたびにDBを引かず、
        # load_memory と refresh_mind() のときだけ差し替える（発話間のみ更新）。
        self._mind_snapshot: MindSnapshot | None = None
        self.viewer_store: ViewerProfileStore | None = None
        self.chat_history: ChatHistory | None = None
        self.consolidator = MemoryConsolidator(memory_store=None, response_filter=response_filter)

        # 画面表示・話者選びに使う状態
        self.emotion: Emotion = Emotion.NEUTRAL
        self.last_speech: str = ""
        self.last_spoke_at: float = 0.0
        self.speak_count: int = 0

        self.load_memory()

    # ── 基本情報 ──────────────────────────────────────────────

    @property
    def name(self) -> str:
        return self.cfg.name

    @property
    def key(self) -> str:
        """アバター画面・WebSocketでの識別子"""
        return self.cfg.key

    @property
    def avatar(self) -> AvatarConfig:
        """このキャラのアバター設定（専用の指定が無ければ全体設定）"""
        return self.cfg.avatar or self.app.avatar

    @property
    def cast_note(self) -> str:
        return build_cast_note(self.cfg, self.partners)

    def set_partners(self, partners: list[CharacterConfig]) -> None:
        self.partners = [p for p in partners if p.key != self.key]

    def is_addressed(self, text: str) -> bool:
        """このキャラが名指しされているか"""
        return self.mention_position(text) >= 0

    def mention_position(self, text: str) -> int:
        """名前（別名を含む）が最初に出てくる位置。出てこなければ -1"""
        text = text or ""
        hits = [text.find(n) for n in self.cfg.call_names() if n in text]
        return min(hits) if hits else -1

    # ── 記憶 ──────────────────────────────────────────────────

    def load_memory(self) -> None:
        """
        このキャラのコア記憶・エピソード記憶・関係性・チャット履歴を読み込み、
        記憶整理の書き込み先もまとめて差し替える。

        以前は長期記憶だけを差し替えていたため、整理器が前のキャラのファイルを
        掴んだままになり、キャラを切り替えると別キャラの記憶に書き込まれていた。
        """
        self.close()
        name = self.cfg.name
        data_dir = self.cfg.data_dir
        orch = self.app.orchestrator
        self.long_term_memory = LongTermMemory(name, prompts_dir=data_dir)
        self.episodic_rag = RAGRetriever()
        char_dir = Path(data_dir) / "characters" / name
        self.memory_store = MemoryStore(
            db_path=char_dir / "memories.db",
            retriever=self.episodic_rag,
            max_entries=orch.max_memory_entries,
            half_life_days=orch.memory_half_life_days,
        )
        self.mind_store = MindStore(db_path=char_dir / "memories.db")
        self.refresh_mind()
        self.viewer_store = ViewerProfileStore(
            db_path=char_dir / "viewer_profiles.db",
            points_cfg=self.app.viewer_points)
        self.chat_history = ChatHistory(
            name, prompts_dir=data_dir,
            max_entries=self.app.local_chat.history_max_entries,
        )
        self.consolidator.bind_store(self.memory_store, viewer_store=self.viewer_store)
        # 旧方式でMDに溜まっていた自動記憶を一度だけ引っ越す（初回のみ）
        if self.memory_store.count() == 0:
            migrate_legacy_section(self.long_term_memory, self.memory_store)
        logger.info("Character memory loaded", character=name,
                    episodic=self.memory_store.count(),
                    core_chars=len(self.long_term_memory.get_raw()))

    def close(self) -> None:
        for store in (self.memory_store, self.viewer_store, self.mind_store):
            if store is not None:
                try:
                    store.close()
                except Exception:
                    pass

    def recall(self, query: str) -> str:
        """
        発言に関連する記憶を集めてプロンプト断片にする（同期・別スレッド実行前提）。

        クエリには発言者名も含める。「名前おぼえてる？」のような本文だけでは
        誰の記憶を引けばよいか分からず、実測でも発言者名を足すと
        該当記憶のスコアが 0.23 → 0.55 に上がった。
        """
        orch = self.app.orchestrator
        parts: list[str] = []
        core_text = self.long_term_memory.get_raw() if self.long_term_memory else ""
        if core_text:
            self.rag.index(core_text)
            core = self.rag.build_rag_prompt(query, top_k=orch.memory_top_k)
            if core:
                parts.append(core[:400])
        if self.memory_store is not None:
            # 日付指定の配信質問（「11月14日の配信で話した話題は？」等）は
            # 意味検索が当該エピソードを取りこぼすため、日付ラベルで直接
            # 引いた記録を優先し、意味検索はフォールバックにする（Phase 4.5）
            dated = self._dated_episode_prompt(query)
            if dated:
                parts.append(dated)
            else:
                episodic = self.memory_store.build_prompt(
                    query, top_k=orch.memory_top_k, max_chars=400,
                    min_score=orch.memory_min_score,
                )
                if episodic:
                    parts.append(episodic)
        return "\n\n".join(parts)

    _EPISODE_DATE_RE = re.compile(r"(\d{1,2})月(\d{1,2})日|(\d{1,2})/(\d{1,2})")

    def _episode_query_dates(self, query: str) -> list[tuple[int, int]]:
        """質問文から配信日（月, 日）を抽出する。

        明示日付（11月14日 / 11/14）があればそれを使い、無ければ
        配信への言及＋相対日付（昨日・一昨日）を JST 基準で解決する
        （PC時計は JST と約15時間ずれているため UTC+9 で明示計算する）。
        """
        dates: list[tuple[int, int]] = []
        for m in self._EPISODE_DATE_RE.finditer(query or ""):
            mo = int(m.group(1) or m.group(3))
            da = int(m.group(2) or m.group(4))
            if 1 <= mo <= 12 and 1 <= da <= 31 and (mo, da) not in dates:
                dates.append((mo, da))
        if dates:
            return dates
        if not any(k in (query or "") for k in ("配信", "放送", "ライブ")):
            return []
        jst = datetime.now(timezone.utc) + timedelta(hours=9)
        if "一昨日" in query or "おととい" in query:
            d = jst - timedelta(days=2)
            return [(d.month, d.day)]
        if "昨日" in query:
            d = jst - timedelta(days=1)
            return [(d.month, d.day)]
        if "今日" in query:
            return [(jst.month, jst.day)]
        return []

    def _dated_episode_prompt(self, query: str) -> str:
        """日付で直接引いた配信エピソードのプロンプト断片（無ければ空）"""
        if self.memory_store is None:
            return ""
        dates = self._episode_query_dates(query)
        if not dates:
            return ""
        lines: list[str] = []
        seen: set[int] = set()
        for mo, da in dates[:2]:
            try:
                entries = self.memory_store.find_episodes_on(mo, da, limit=5)
            except Exception:
                return ""
            for e in entries:
                if e.id in seen:
                    continue
                seen.add(e.id)
                lines.append(f"- {e.text}")
        if not lines:
            return ""
        text = "## その日の配信の記録（日付で引いた実記録）\n" + "\n".join(lines)
        return text[:700]

    def build_idle_memory(self, extra: str = "") -> str:
        """
        独り言のお題に使う記憶。コア記憶は上限つきで切り、エピソード記憶は最近のぶんだけ。
        （全文を入れていた頃は、記憶が育つとコンテキスト超過で空応答になっていた）
        """
        orch = self.app.orchestrator
        parts: list[str] = []
        core = self.long_term_memory.build_prompt() if self.long_term_memory else ""
        if core:
            limit = orch.idle_core_memory_chars
            parts.append(core if len(core) <= limit else core[:limit] + "\n…")
        if self.memory_store is not None:
            limit = orch.idle_episodic_entries
            recent = self.memory_store.sample_recent(limit=limit)
            # 話題化したばかりの記憶は L3 にも出さない（Phase 1 / STUDY
            # §3-C3 の最小形）。プランナーのクールダウン台帳を共有し、
            # 「話題として話した記憶」と「常駐注入される記憶」の二経路で
            # 同じエピソードが繰り返される（お菓子ループの温床）を断つ。
            # 除外で枠が空く場合は、少し広く引いて別の記憶で埋める
            planner = getattr(orch, "_planner", None)
            if planner is not None and recent:
                try:
                    fresh = [e for e in recent
                             if not planner.is_topic_memory_recent(e.text)]
                    if len(fresh) < limit:
                        wider = self.memory_store.sample_recent(limit=limit * 3)
                        fresh = [e for e in wider
                                 if not planner.is_topic_memory_recent(e.text)]
                    recent = fresh[:limit]
                except Exception:
                    pass
            if recent:
                parts.append("\n".join(["## 最近おぼえたこと"] + [f"- {e.text}" for e in recent]))
        if extra:
            parts.append(extra)
        return "\n\n".join(parts).strip()

    def topic_keywords(self) -> list[str]:
        """自律トークの検索クエリの種にする、記憶由来のキーワード"""
        try:
            core = self.long_term_memory.get_raw() if self.long_term_memory else ""
        except Exception:
            core = ""
        keywords = extract_memory_keywords(core, max_keywords=6)
        if self.memory_store is not None:
            for entry in self.memory_store.sample_recent(limit=4):
                text = entry.text.strip()
                if 2 <= len(text) <= 20:
                    keywords.append(text)
        return keywords

    # ── 会話ログ ──────────────────────────────────────────────

    def hear_comment(self, comment: Comment) -> None:
        """視聴者コメントを聞く（返事をするのが自分でなくても文脈には入れる）"""
        self.log.add_comment(comment)

    def hear_partner(self, name: str, text: str) -> None:
        self.log.add_partner(name, text)

    def record_chat(self, role: str, text: str, author: str = "", emotion: str = "") -> None:
        """ローカルチャットの発言をこのキャラの履歴ファイルに残す"""
        if self.chat_history is not None:
            self.chat_history.append(role, text, author=author, emotion=emotion)

    def restore_chat_context(self, turns: int) -> int:
        """保存済みのチャット履歴を、LLMに渡す文脈として読み戻す（会話中なら何もしない）"""
        if turns <= 0 or self.chat_history is None or self.log.history_size > 0:
            return 0
        restored = self.log.restore(self.chat_history.to_messages(turns=turns))
        if restored:
            logger.info("Restored chat context", messages=restored, character=self.name)
        return restored

    # ── LLM / TTS ─────────────────────────────────────────────

    @property
    def persona_model_active(self) -> bool:
        """いま応答しているのが人格QLoRAの専用モデルか（フォールバック中は False）"""
        return bool(self.llm is not None and getattr(self.llm, "is_persona", False))

    @property
    def slim_persona_prompt(self) -> bool:
        return self.persona_model_active and self.app.persona.prompt_mode == "slim"

    def refresh_mind(self) -> None:
        """MindStore から自己のスナップショットを取り直す。

        プロンプトの L0b〜L2 はこのスナップショットから組み立てる。
        発話中には差し替えない（自己は発話と発話の間でのみ更新される）。
        """
        old_snapshot = self._mind_snapshot
        self._mind_snapshot = (
            self.mind_store.build_snapshot() if self.mind_store is not None else None
        )
        # 焦点ローテーションの提示履歴はスナップショット差し替えを跨いで
        # 引き継ぐ（差し替えるたびに履歴が消えるとローテーションが
        # 成立しない。Phase 4.6）
        if old_snapshot is not None and self._mind_snapshot is not None:
            self._mind_snapshot.focus_history = list(old_snapshot.focus_history)

    def recent_own_speeches(self, limit: int = 3) -> list[str]:
        """自分の直近の発話を時系列順に返す（繰り返し回避の提示用。P2-B）。

        会話ログの assistant メッセージが自分の発話（add_response /
        add_idle で積まれる）。相方の発言は user 側の partner_speech に
        入るため混ざらない。
        """
        own = [m.content for m in self.log.get_messages()
               if m.role == "assistant" and (m.content or "").strip()]
        return own[-limit:]

    def prompt_kwargs(self) -> dict:
        """prompt_builder の各関数に渡す、このキャラ固有の引数"""
        return {
            "character": self.cfg,
            "cast_note": self.cast_note,
            "slim_persona": self.slim_persona_prompt,
            "mind": self._mind_snapshot,
            "recent_own_speeches": self.recent_own_speeches(),
        }

    async def rebuild_llm(self) -> str:
        """LLMを作り直す。古い接続は閉じてから差し替える。"""
        old = [b for b in (self.llm, self.llm_fallback) if b is not None]
        self.llm, self.llm_fallback = factory.build_character_llm(
            self.app, self.cfg, is_lead=self.is_lead)
        self._on_fallback = False
        for b in old:
            if b is self.llm or b is self.llm_fallback:
                continue
            try:
                await b.aclose()
            except Exception:
                pass
        return self.llm.name if self.llm else "none"

    def rebuild_tts(self) -> str:
        self.tts = factory.build_tts(self.app, self.cfg)
        return self.tts.name

    def failover_llm(self) -> bool:
        """primary が落ちていたら fallback と入れ替える。入れ替えたら True"""
        # fallback 自体が不健全なときは入れ替えない（死んだ同士の往復を防ぐ）
        if self.llm and not self.llm.is_healthy and self.llm_fallback \
                and self.llm_fallback.is_healthy:
            logger.warning("LLM primary unhealthy, switching to fallback",
                           character=self.name, primary=self.llm.name,
                           fallback=self.llm_fallback.name)
            self.llm, self.llm_fallback = self.llm_fallback, self.llm
            self._on_fallback = True
            return True
        return False

    @property
    def on_fallback(self) -> bool:
        """いま fallback 側で喋っているか（failover で退避中か）"""
        return self._on_fallback

    def failback_llm(self) -> bool:
        """退避中だったら primary へ戻す。戻したら True（Phase 4）。

        呼び出し側（Orchestrator のヘルスチェック）が、fallback スロットに
        いる primary への能動プローブで回復を確認済みのときだけ呼ぶ前提。
        ここでは「退避中であること」「現行・復帰先の双方が健全なこと」だけを
        条件にし、無条件の自動復帰はしない。
        """
        if not self._on_fallback:
            return False
        if self.llm and self.llm.is_healthy and self.llm_fallback \
                and self.llm_fallback.is_healthy:
            logger.warning("LLM primary recovered, switching back",
                           character=self.name, primary=self.llm_fallback.name,
                           fallback=self.llm.name)
            self.llm, self.llm_fallback = self.llm_fallback, self.llm
            self._on_fallback = False
            return True
        return False

    def repair_speech(self, text: str) -> str:
        """一人称・口調の崩れを直す（設定で無効なら素通し）"""
        if not self.app.orchestrator.enable_persona_validation:
            return text
        fixed, _ = self.persona_validator.validate_and_repair(text, character_name=self.name)
        return fixed

    def status(self) -> dict:
        voice = self.cfg.voice
        return {
            "id": self.key,
            "name": self.name,
            "is_lead": self.is_lead,
            "llm_backend": self.llm.name if self.llm else "none",
            "llm_model": getattr(getattr(self.llm, "_cfg", None), "model", ""),
            "tts_backend": self.tts.name if self.tts else "",
            "voice": voice.voice or (str(voice.speaker_id) if voice.speaker_id is not None else ""),
            "avatar_type": self.avatar.type,
            "emotion": self.emotion.value,
            "last_speech": self.last_speech,
            "speak_count": self.speak_count,
            "pending_turns": self.log.turns_since_consolidation,
            "episodic_count": self.memory_store.count() if self.memory_store else 0,
            "mind": self.mind_store.stats() if self.mind_store else {},
        }
