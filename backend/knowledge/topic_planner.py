"""自律トークの話題を決めるプランナー

コメントが無いときに何を話すかを決める。話題の出どころは3つ:

  memory    … 長期記憶のキーワード（従来の独り言。ネット接続不要）
  news      … RSSで仕入れた最新ニュース
  curiosity … 記憶のキーワードや配信テーマをWeb検索して仕入れた話
  theme     … 配信テーマ（stream_theme）に沿った話題。テーマ設定中は
              話題シェア目標（theme_share_target）に応じて定期的に差し込む

【2026-11-14 改修：LLMに依らない決定層】
話題の選択はLLMではなく、以下の決定的な仕組みで行う（設計メモ
DESIGN-2026-11-14-autonomous-speech.md 参照）。

  - 種類（kind）の選択は固定の重み抽選ではなく Utility スコア制。
    ニュースは「未話の新鮮な記事があるか」「最後にニュースを話してから
    どれだけ経ったか（干上がり係数）」でスコアが上下するので、
    新鮮なネタが埋もれず、かつ連投にもならない。
  - ニュース記事の選択もスコア制（鮮度×新規性×媒体の多様性）。
    新規性は台帳にある過去の話題タイトルとの類似度で測る。
  - 消費は commit/release 方式。plan() 時点では記事を「予約」するだけ
    で、実際に発話できたときに commit() が呼ばれて初めて使用済みに
    なる。生成が失敗してもネタは消えない（予約は10分で自動失効）。
  - 話した話題は台帳（ledger）に残り、次回以降のスコア計算と
    繰り返し回避に使う。
  - 内面状態として energy（0..1）を持ち、コメント応答で上がり
    時間で減衰する。energy が低い（＝退屈）ときは curiosity の
    スコアが上がり、自分で調べに行く方向に傾く。

ネットが死んでいてもmemoryに落ちるので、配信が止まることはない。
"""
from __future__ import annotations

import asyncio
import json
import math
import random
import re
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from backend.knowledge.models import WebDocument
from backend.knowledge.news import NewsFeed
from backend.knowledge.web_search import WebSearchClient
from backend.logger import get_logger

logger = get_logger(__name__)

# 好奇心検索のクエリ雛形
_CURIOSITY_TEMPLATES = [
    "{kw} 最新",
    "{kw} 話題",
    "{kw} とは",
    "{kw} おすすめ",
    "{kw} 豆知識",
    "{kw} 最近のニュース",
]

# 好奇心クエリの種にしない語（Phase 3）。MindLoop が生成する自己改善系の
# 意図タイトル（「調査の品質向上とスピードの維持」等）がそのまま種になり、
# 自分自身についての検索が繰り返された実測への対処。記憶種の除外
# （Orchestrator._MEMORY_SEED_NG）と同流儀で、種の全系統に適用する。
_CURIOSITY_SEED_NG = re.compile(
    r"調査|検索|調べ|情報収集|情報の整理|パフォーマンス|スピード|品質|"
    r"安定|維持|改善|効率|スキル|さんのため")

# 保留回答（視聴者の事実質問への後追い回答）の有効期限（秒）。
# comment hook の窓（10分）と揃える
_PENDING_ANSWER_TTL_SEC = 600.0

# 予約の有効期限（秒）。発話に至らなかった話題はこの時間で解放される
_RESERVATION_TTL_SEC = 600.0
# 直近の同種トピックへのクールダウン（秒）
_KIND_COOLDOWN_SEC = 150.0
# 話題の種（意図タイトル・記憶キーワード）の再使用クールダウン（秒）。
# 最有力の種（conf の高い意見に紐づく意図など）へ自律話題が恒常的に
# 引き寄せられるのを断つ（Phase 4.5: 35分ソークで「調査・整理」テーマに
# 29発話が張り付いた実測への対処。fix6 の台帳＋クールダウン方式の拡張）。
_SEED_COOLDOWN_SEC = 1800.0
# 直近発話テーマと候補話題の重なり（containment）がこの値以上なら候補を避ける
_THEME_OVERLAP_SEED_MAX = 0.35
_THEME_OVERLAP_MEMORY_MAX = 0.5

# memory 話題シェアハードキャップ（Phase 1 / STUDY §3-C1）。台帳がこの件数
# たまるまでは誤爆を避けるためキャップを効かせない
_MEMORY_CAP_MIN_SAMPLES = 10

# force_topic 種まきキューの保持上限と有効期限（2026-11-22 themefix）
_FORCED_TOPIC_MAX = 8
_FORCED_TOPIC_TTL_SEC = 3600.0
# 話題化した記憶を L3 常駐注入（build_idle_memory）へ再浮上させない
# クールダウン（秒。STUDY §3-C3 の最小形：プランナー側の台帳を共有する）
_TOPIC_MEMORY_COOLDOWN_SEC = 1800.0

# 次発言プランニング（Phase 3 / STUDY §6-(a1)）：展開の発話で「問いかけ」
# の意図を選ぶための energy の下限。これより退屈なときは問いかけず、
# 深掘りに専念させる
_ARC_ASK_MIN_ENERGY = 0.35
# 弧が完遂してからこの秒数以内に始めた新規話題は「転換」の意図にする
_ARC_TRANSITION_WINDOW_SEC = 300.0

# 定番小噺プール（Phase 1 / STUDY §3-C2）：ニュース・検索・記憶の材料が
# すべて尽きたときのフォールバック終端。材料として数える軽い話題で、
# 特定の事実を主張しない、季節・日常・配信まわりの無難なお題に限る
_FILLER_TOPICS: list[tuple[str, str]] = [
    ("季節の変わり目", "いまの季節の空気や、最近の天気の移り変わりについて軽く触れる。季節らしい食べ物や景色の話でもよい"),
    ("今日の調子", "今日の自分の調子や気分について短く話す。見てくれている人の今日の様子も気にかける"),
    ("カフェの近況", "カフェの店内の様子や、最近淹れた飲み物の話など、身の回りの小さな近況を話す"),
    ("視聴者への呼びかけ", "見てくれている人への感謝と、コメントや感想を気軽に書いてほしいという呼びかけをする"),
    ("最近のマイブーム", "最近なんとなく気に入っている小さなこと（飲み物・道具・習慣など）を1つ挙げて話す"),
    ("一息のすすめ", "見ている人へ、温かい飲み物や深呼吸など、小さな休憩のすすめを話す"),
    ("近いうちの楽しみ", "近いうちに楽しみにしている小さなことを話す。実在しない予定を断定はしない"),
    ("配信の裏側", "配信の準備や機材まわりで最近あった小さな出来事を話す。具体的な固有名詞は出さない"),
]


def _title_tokens(text: str) -> set[str]:
    """タイトル類似度用の素朴なトークン化（日本語は文字バイグラム併用）"""
    text = (text or "").lower()
    words = set(re.findall(r"[a-z0-9]{2,}", text))
    kana = re.sub(r"[^ぁ-んァ-ヶ一-龥]", "", text)
    bigrams = {kana[i:i + 2] for i in range(len(kana) - 1)}
    return words | bigrams


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


@dataclass
class Topic:
    """次に話す話題1つぶん"""
    kind: str                                   # "memory" | "news" | "curiosity" | "answer" | "theme"
    label: str = ""                             # ダッシュボード表示用の短い説明
    query: str = ""                             # 検索に使ったクエリ
    documents: list[WebDocument] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    topic_id: str = ""                          # プランナーが発行する一意ID
    resumed: bool = False                       # 中断した話題の続報なら True
    pending_id: str = ""                        # 保留回答キュー由来ならそのID（Phase 3）
    author: str = ""                            # kind="answer" のときの質問者名（Phase 3）
    arc_stage: str = ""                         # 話題の弧の段階（"" | introduce | deepen | close）。Phase 1 では状態と既定意図まで
    prev_speech: str = ""                       # 継続時に、直前の自分の発話（実文・切り詰め）
    intent: str = ""                            # 次発言の意図（Phase 3: "" | introduce | deepen | ask | react | close | transition）
    arc_key: str = ""                           # 弧の同一判定キー（Phase 3）

    @property
    def has_material(self) -> bool:
        return bool(self.documents)

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "label": self.label,
            "query": self.query,
            "documents": [d.to_dict() for d in self.documents],
            "created_at": self.created_at,
            "topic_id": self.topic_id,
            "resumed": self.resumed,
            "pending_id": self.pending_id,
            "author": self.author,
            "arc_stage": self.arc_stage,
            "prev_speech": self.prev_speech,
            "intent": self.intent,
            "arc_key": self.arc_key,
        }


class TopicPlanner:
    """話題の決定（Utilityスコア制）と、その材料集めを担当する"""

    def __init__(
        self,
        news: NewsFeed | None = None,
        search: WebSearchClient | None = None,
        keyword_provider: Optional[Callable[[], list[str]]] = None,
        theme_provider: Optional[Callable[[], str]] = None,
        memory_provider: Optional[Callable[[], dict | None]] = None,
        intention_provider: Optional[Callable[[], list[dict]]] = None,
        weights: Optional[dict[str, float]] = None,
        max_documents: int = 3,
        history_size: int = 12,
        state_path: str | None = None,
        memory_cap_ratio: float = 0.45,
        memory_cap_window: int = 20,
        filler_enabled: bool = False,
        planner_enabled: bool = True,
        arc_max_speeches: int = 4,
        arc_max_continuations: int = 2,
        arc_max_sec: float = 900.0,
        arc_ask_probability: float = 0.25,
        theme_share_target: float = 0.34,
        theme_share_window: int = 12,
        topic_tenure_enabled: bool = True,
        topic_tenure_speeches: int = 6,
        topic_tenure_sec: float = 1200.0,
        topic_tenure_cooldown_sec: float = 1800.0,
        topic_tenure_window_sec: float = 1800.0,
    ):
        self._news = news
        self._search = search
        self._keyword_provider = keyword_provider or (lambda: [])
        self._theme_provider = theme_provider or (lambda: "")
        self._memory_provider = memory_provider
        # 「取り組んでいること」（意図）の種。{"title", "next_action"} のリストを返す
        self._intention_provider = intention_provider
        self.weights = {"memory": 3.0, "news": 2.0, "curiosity": 1.0}
        if weights:
            self.weights.update({k: float(v) for k, v in weights.items()})
        self.max_documents = max_documents
        self._history: deque[str] = deque(maxlen=history_size)
        self.last_topic: Topic | None = None
        # Phase 1 の穴埋め設定（STUDY §3-C1/C2）
        self._memory_cap_ratio = float(memory_cap_ratio)
        self._memory_cap_window = max(1, int(memory_cap_window))
        self._filler_enabled = bool(filler_enabled)
        # 次発言プランニング（Phase 3 / STUDY §6-(a1)）：弧の状態と意図の
        # ルール決定・完遂判定。無効化すると意図は付かず、弧の段階付けは
        # Phase 1 までの規則（継続上限で締め）に戻る
        self._planner_enabled = bool(planner_enabled)
        self._arc_max_speeches = max(1, int(arc_max_speeches))
        self._arc_max_continuations = max(0, int(arc_max_continuations))
        self._arc_max_sec = float(arc_max_sec)
        self._arc_ask_probability = min(1.0, max(0.0,
                                                 float(arc_ask_probability)))
        # 配信テーマ（stream_theme）の話題シェア目標（2026-11-22 themefix）。
        # テーマ設定中、話題決定のうちこの割合をテーマ話題にする。
        # 0 以下で無効（従来動作）
        self._theme_share_target = min(1.0, max(0.0,
                                                 float(theme_share_target)))
        self._theme_share_window = max(1, int(theme_share_window))
        # 話題の任期（topic tenure、2026-11-22 longstream）：同じ主題の
        # 連なりが長くなりすぎたら、決定境界で弧を「完遂」として閉じ、
        # その主題を一定期間、次候補から外す（詳細は _try_tenure_transition）
        self._tenure_enabled = bool(topic_tenure_enabled)
        self._tenure_speeches = max(1, int(topic_tenure_speeches))
        self._tenure_sec = float(topic_tenure_sec)
        self._tenure_cooldown_sec = float(topic_tenure_cooldown_sec)
        self._tenure_window_sec = float(topic_tenure_window_sec)
        # 経過時間だけで tenure 到達とみなすのに必要な最小発話数
        # （2026-11-22 verifyfix）。発話1〜3件のチェーンが、セッション
        # 跨ぎの壁時計経過だけで移行対象になるのを防ぐための下限で、
        # 発話数そのものの閾値（_tenure_speeches）より保守的に小さく取る
        self._tenure_min_speeches_for_time = 3
        # tenure 除外中の主題（arc_key → {until, title}）。永続化する
        self._tenure_excluded: dict[str, dict] = {}
        # tenure 強制クローズを出した弧のキー（commit の完遂処理で参照）
        self._tenure_closing_key: str = ""
        # force_topic の種まきキュー（2026-11-22 themefix）。投入順に
        # 次回以降の自動決定で消費される。プロセス内のみ（永続化しない）
        self._forced_topics: deque[dict] = deque()
        # テーマ言及率の計測（2026-11-22 themefix）。テーマ文字列ごとに
        # 発話数と言及数を数え、テーマが変わったらリセットする
        self._theme_stats_theme: str = ""
        self._theme_stat_speeches: int = 0
        self._theme_stat_mentions: int = 0
        # 弧の計測（完遂／期限切れ／放棄の回数）と、直近の完遂時刻
        # （完遂直後の新規話題を「転換」の意図にするために使う）。
        # abandoned は合計に加え、理由別（abandoned_reset＝文脈リセット／
        # abandoned_force_switch＝テーマ反復の強制切替）にも分けて数える
        # （F9: 経路で意味が混ざると比率の解釈を誤るため）
        self._arc_stats: dict[str, int] = {
            "completed": 0, "expired": 0, "abandoned": 0, "tenure": 0,
            "abandoned_reset": 0, "abandoned_force_switch": 0}
        self._last_arc_end_at: float = 0.0
        # 話題化した記憶の使用台帳（正規化キー → 使用時刻）。L3 常駐注入の
        # サンプリングと共有し、同じ記憶の二経路での繰り返しを断つ（§3-C3）
        self._topic_memory_used_at: dict[str, float] = {}
        # 直前の自分の発話（継続フレーミングで「さっきの話」として渡す）
        self._last_speech_text: str = ""

        # ── 決定層の状態（2026-11-14 追加） ──
        self._born_at = time.time()
        self._seq = 0
        self._ledger: deque[dict] = deque(maxlen=60)        # 話した話題の台帳
        self._reserved: dict[str, float] = {}               # doc.key → 予約時刻
        self._last_kind_at: dict[str, float] = {}           # kind → 最後に話した時刻
        self._last_news_site: str = ""
        self._search_failures: deque[float] = deque(maxlen=10)  # 空振り検索の時刻
        self._last_utilities: dict[str, float] = {}
        # 内面状態：energy（高いほど活発）。コメント応答で上がり、時間で減衰
        self._energy = 0.5
        self._energy_at = time.time()
        self.last_comment_speech_at: float = 0.0
        self._stack: list[dict] = []                # 中断された話題（復帰用）
        # 保留回答キュー（Phase 3）: その場で答えられなかった事実質問を
        # 裏で調べ、答えを自律トークとして差し込むための待ち行列。
        # {id, author, question, query, key, enqueued_at, ttl, attempts,
        #  state: waiting|ready|failed, documents, next_try_at}
        self._pending_answers: deque[dict] = deque()
        self._pending_seq = 0
        # 会話の継続用：直近の話題スレッドと、拾い直し待ちのコメント
        self._thread: dict | None = None            # {kind,title,query,at,continuations}
        self._banned: dict[str, float] = {}         # 吸着した語 → 禁止期限（文脈リセット時）
        self._comment_hook: dict | None = None      # {author,text,at}
        self._last_hook_woven_at: float = 0.0
        # 話題の種の使用履歴（正規化した種 → 最後に使った時刻）と、
        # 直近発話のテーマ（トークン集合）。話題の多様化用（Phase 4.5）
        self._seed_used_at: dict[str, float] = {}
        self._recent_speech_tokens: deque[set] = deque(maxlen=4)
        # 状態の永続化先。None指定で既定位置、""で無効（テスト用）
        if state_path is None:
            self._state_path: Path | None = \
                Path(__file__).resolve().parent.parent.parent / "data" / "planner_state.json"
        elif state_path == "":
            self._state_path = None
        else:
            self._state_path = Path(state_path)
        self._load_state()

    # ── 内面状態 ────────────────────────────────────────────

    def _decay_energy(self, now: float) -> float:
        dt = max(0.0, now - self._energy_at)
        if dt > 0:
            self._energy *= math.exp(-dt / 1800.0)   # 時定数30分で減衰
            self._energy_at = now
        return self._energy

    @property
    def energy(self) -> float:
        return self._decay_energy(time.time())

    @property
    def last_speech_text(self) -> str:
        """直前の自分の発話の実文（継続フレーミング／先行生成の前提照合用）"""
        return self._last_speech_text

    def note_speech(self, idle: bool = False, text: str = "") -> None:
        """発話イベントの通知（Orchestrator._after_speech から呼ばれる）"""
        now = time.time()
        self._decay_energy(now)
        if text:
            # 継続フレーミング用に、直前の発話の実文を短く保持する
            self._last_speech_text = text.strip()[:120]
            # 直近発話のテーマを覚える（次の話題選択で重複を避けるため）
            toks = _title_tokens(text)
            if toks:
                self._recent_speech_tokens.append(toks)
        # テーマ言及率の計測（2026-11-22 themefix）：テーマが変わったら
        # カウンタをリセットし、設定中は発話数と言及数を数える
        theme = (self._theme_provider() or "").strip()
        if theme != self._theme_stats_theme:
            self._theme_stats_theme = theme
            self._theme_stat_speeches = 0
            self._theme_stat_mentions = 0
        if theme and text:
            self._theme_stat_speeches += 1
            if theme in text:
                self._theme_stat_mentions += 1
                logger.info(
                    "stream theme mentioned",
                    theme=theme,
                    mentions=self._theme_stat_mentions,
                    speeches=self._theme_stat_speeches,
                    rate=round(self._theme_stat_mentions
                               / self._theme_stat_speeches, 3))
        if idle:
            self._energy = min(1.0, self._energy + 0.04)
        else:
            self._energy = min(1.0, self._energy + 0.15)
            self.last_comment_speech_at = now
            self._suspend_current(now)

    # ── 話題の種のクールダウンと発話テーマの重なり（Phase 4.5）──

    @staticmethod
    def _seed_key(seed: str) -> str:
        return re.sub(r"\s+", "", (seed or "")).lower()

    def _seed_on_cooldown(self, seed: str) -> bool:
        at = self._seed_used_at.get(self._seed_key(seed))
        return at is not None and time.time() - at < _SEED_COOLDOWN_SEC

    def _mark_seed_used(self, seed: str) -> None:
        if seed:
            self._seed_used_at[self._seed_key(seed)] = time.time()

    def _theme_overlap(self, text: str) -> float:
        """候補テキストと直近発話テーマとの最大の重なり（0..1）。

        発話は短く候補（エピソード文など）は長いため、Jaccard では希釈
        されて効かない。意見マージと同じ containment（小さい側の集合に
        対する共通部分の割合）で測る。
        """
        toks = _title_tokens(text or "")
        if not toks or not self._recent_speech_tokens:
            return 0.0
        def containment(prev: set) -> float:
            if not prev:
                return 0.0
            return len(toks & prev) / min(len(toks), len(prev))
        return max((containment(prev)
                    for prev in self._recent_speech_tokens), default=0.0)

    def _suspend_current(self, now: float) -> None:
        """直近の話題がコメントで中断されたら、復帰用にスタックへ積む"""
        if not self._ledger:
            return
        last = self._ledger[-1]
        if last.get("kind") not in ("news", "curiosity"):
            return
        if now - float(last.get("at") or 0) > 180:
            return
        if any(e.get("topic_id") == last.get("topic_id") for e in self._stack):
            return
        entry = {**last, "suspended_at": now}
        th = self._thread
        if th is not None and th.get("kind") == last.get("kind") \
                and (not th.get("topic_id")
                     or th.get("topic_id") == last.get("topic_id")):
            # 弧の状態も一緒に退避する（Phase 3）。コメント応答は寄り道で、
            # 復帰時は弧の途中から再開し、締めまでの残りを数え直さない
            entry["arc_speeches"] = int(th.get("speeches") or 0)
            entry["arc_started_at"] = float(th.get("started_at") or 0)
            entry["arc_stage"] = th.get("stage") or ""
            entry["continuations"] = int(th.get("continuations") or 0)
            entry["arc_key"] = th.get("arc_key") or ""
        self._stack.append(entry)
        del self._stack[:-3]
        self._save_state()
        logger.info("topic suspended", kind=last.get("kind"),
                    title=(last.get("title") or last.get("query") or "")[:40])

    # ── 状態の永続化 ────────────────────────────────────────

    @staticmethod
    def _pending_answer_to_dict(e: dict) -> dict:
        """保留回答1件を JSON 化できる形にする（F7）"""
        return {
            "id": e.get("id") or "",
            "author": e.get("author") or "",
            "question": e.get("question") or "",
            "query": e.get("query") or "",
            "key": e.get("key") or "",
            "enqueued_at": float(e.get("enqueued_at") or 0),
            "ttl": float(e.get("ttl") or _PENDING_ANSWER_TTL_SEC),
            "attempts": int(e.get("attempts") or 0),
            "state": e.get("state") or "waiting",
            "documents": [d.to_dict() for d in (e.get("documents") or [])
                          if hasattr(d, "to_dict")],
            "next_try_at": float(e.get("next_try_at") or 0),
        }

    def _save_state(self) -> None:
        """台帳・最終話題時刻・energy・スタックをJSONに保存する（commit時に呼ぶ）"""
        if self._state_path is None:
            return
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "version": 1,
                "ledger": list(self._ledger),
                "last_kind_at": self._last_kind_at,
                "last_news_site": self._last_news_site,
                "energy": round(self._energy, 3),
                "seq": self._seq,
                "stack": self._stack,
                "thread": self._thread,
                "seed_used_at": self._seed_used_at,
                "topic_memory_used_at": self._topic_memory_used_at,
                "arc_stats": self._arc_stats,
                # 保留回答も永続化する（F7）。以前は再起動のたびに
                # 調査済みの回答ごと消え、質問が無言で失われていた
                "pending_answers": [
                    self._pending_answer_to_dict(e)
                    for e in self._pending_answers],
                "tenure_excluded": {
                    k: {"until": float(v.get("until") or 0),
                        "title": v.get("title") or ""}
                    for k, v in self._tenure_excluded.items()},
            }
            tmp = self._state_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            tmp.replace(self._state_path)
        except Exception as e:
            logger.warning("planner state save failed", error=str(e))

    def _load_state(self) -> None:
        """保存済みの状態を復元する。壊れていても起動は止めない"""
        if self._state_path is None:
            return
        try:
            if not self._state_path.exists():
                return
            data = json.loads(self._state_path.read_text(encoding="utf-8"))
            for e in data.get("ledger", [])[-60:]:
                if isinstance(e, dict):
                    self._ledger.append(e)
            lka = data.get("last_kind_at") or {}
            self._last_kind_at = {k: float(v) for k, v in lka.items()
                                  if isinstance(v, (int, float))}
            self._last_news_site = str(data.get("last_news_site") or "")
            self._energy = min(1.0, max(0.0, float(data.get("energy", 0.5))))
            self._energy_at = time.time()
            self._seq = int(data.get("seq") or 0)
            st = data.get("stack") or []
            self._stack = [e for e in st if isinstance(e, dict)][-3:]
            su = data.get("seed_used_at") or {}
            now = time.time()
            self._seed_used_at = {
                str(k): float(v) for k, v in su.items()
                if isinstance(v, (int, float))
                and now - float(v) < _SEED_COOLDOWN_SEC * 4}
            tmu = data.get("topic_memory_used_at") or {}
            self._topic_memory_used_at = {
                str(k): float(v) for k, v in tmu.items()
                if isinstance(v, (int, float))
                and now - float(v) < _TOPIC_MEMORY_COOLDOWN_SEC * 4}
            th = data.get("thread")
            if isinstance(th, dict) and time.time() - float(th.get("at") or 0) < 600:
                self._thread = th
            ast = data.get("arc_stats") or {}
            if isinstance(ast, dict):
                for k in ("completed", "expired", "abandoned", "tenure",
                          "abandoned_reset", "abandoned_force_switch"):
                    if isinstance(ast.get(k), (int, float)):
                        self._arc_stats[k] = int(ast[k])
            te = data.get("tenure_excluded") or {}
            if isinstance(te, dict):
                for k, v in te.items():
                    if isinstance(v, dict) \
                            and float(v.get("until") or 0) > now:
                        self._tenure_excluded[str(k)] = {
                            "until": float(v.get("until") or 0),
                            "title": str(v.get("title") or "")}
            # 保留回答の復元（F7）。TTL を過ぎたものは復元しない
            # （期限の意味を再起動で延ばさないため）。ID の連番は
            # 復元分と衝突しないよう最大値に合わせて進める
            pa = data.get("pending_answers") or []
            if isinstance(pa, list):
                for e in pa:
                    if not isinstance(e, dict) or not e.get("id"):
                        continue
                    enqueued_at = float(e.get("enqueued_at") or 0)
                    ttl = float(e.get("ttl") or _PENDING_ANSWER_TTL_SEC)
                    if now - enqueued_at > ttl:
                        continue
                    state = str(e.get("state") or "waiting")
                    if state not in ("waiting", "ready", "failed"):
                        continue
                    docs = []
                    for d in e.get("documents") or []:
                        if isinstance(d, dict) and d.get("title"):
                            try:
                                docs.append(WebDocument(
                                    title=str(d.get("title") or ""),
                                    url=str(d.get("url") or ""),
                                    snippet=str(d.get("snippet") or ""),
                                    source=str(d.get("source") or ""),
                                    site=str(d.get("site") or ""),
                                    published_at=d.get("published_at"),
                                    fetched_at=float(d.get("fetched_at")
                                                     or now),
                                ))
                            except Exception:
                                continue
                    self._pending_answers.append({
                        "id": str(e["id"]),
                        "author": str(e.get("author") or ""),
                        "question": str(e.get("question") or ""),
                        "query": str(e.get("query") or ""),
                        "key": str(e.get("key") or ""),
                        "enqueued_at": enqueued_at,
                        "ttl": ttl,
                        "attempts": int(e.get("attempts") or 0),
                        "state": state,
                        "documents": docs,
                        "next_try_at": float(e.get("next_try_at") or 0),
                    })
                    try:
                        seq = int(str(e["id"]).rsplit("-", 1)[1])
                        self._pending_seq = max(self._pending_seq, seq)
                    except (IndexError, ValueError):
                        pass
            logger.info("planner state loaded", ledger=len(self._ledger),
                        pending_answers=len(self._pending_answers))
        except Exception as e:
            logger.warning("planner state load failed", error=str(e))

    # ── 台帳と予約 ──────────────────────────────────────────

    def _recent_titles(self, n: int = 8) -> list[str]:
        out = []
        for e in list(self._ledger)[-n:]:
            t = e.get("title") or e.get("query") or ""
            if t:
                out.append(t)
        return out

    def _expire_reservations(self, now: float) -> None:
        expired = [k for k, at in self._reserved.items()
                   if now - at > _RESERVATION_TTL_SEC]
        for k in expired:
            del self._reserved[k]

    def _is_theme_word(self, word: str) -> bool:
        """その語が配信テーマ自体（またはその一部／全体を含む）か。

        テーマ語を反復抑制（ban）の対象にすると、テーマ会話が始まった
        途端にテーマ自体が禁止され維持できなくなる（2026-11-21 BFラン
        分析の主因5）。テーマ語は ban から外し、切り口の反復は類似度
        ベースの抑制（LoopGuard・話題飽和）に任せる（2026-11-22 themefix）
        """
        theme = (self._theme_provider() or "").strip()
        w = (word or "").strip()
        if not theme or not w:
            return False
        return w == theme or theme in w or (len(w) >= 2 and w in theme)

    def ban_word(self, word: str, seconds: float = 1800) -> None:
        """吸着した語をしばらく話題の種にしない（文脈リセットとセットで呼ぶ）"""
        if word:
            if self._is_theme_word(word):
                logger.info("ban skipped for stream theme word", word=word)
                return
            self._banned[word] = time.time() + seconds

    def banned_words(self) -> set[str]:
        now = time.time()
        return {w for w, until in self._banned.items() if until > now}

    def reset_thread(self, reason: str = "reset") -> None:
        """継続中スレッドを破棄する。

        reason は放棄の経路（F9）: "reset"＝文脈リセット（飽和・ブロック
        エスカレーション）、"force_switch"＝テーマ反復の強制切替。
        abandoned の合計と理由別の両方を進める（合計だけだと、強制切替の
        多発が文脈リセットの多発に見えて比率の解釈を誤る）。
        """
        self._tenure_closing_key = ""
        if self._thread is not None and self._planner_enabled:
            # 締めに届かず捨てられた弧として数える（Phase 3 の計測）
            self._arc_stats["abandoned"] = \
                int(self._arc_stats.get("abandoned") or 0) + 1
            bucket = ("abandoned_force_switch" if reason == "force_switch"
                      else "abandoned_reset")
            self._arc_stats[bucket] = \
                int(self._arc_stats.get(bucket) or 0) + 1
        self._thread = None
        self._save_state()

    # ── 弧（アーク）の状態管理（Phase 3） ─────────────────────
    # 1つの話題は「導入→展開→締め」の弧として扱う。弧の解放は、
    # 締めの発話が commit された時点（完遂）か、上限時間の超過
    # （期限切れ）のときだけ。コメント応答などの割り込みでは解放しない

    @staticmethod
    def _arc_key_for(title: str, query: str) -> str:
        """弧の同一判定キー（タイトル優先、空白を落として先頭80字）"""
        base = (title or query or "").strip()
        return re.sub(r"\s+", "", base)[:80]

    def _fresh_intent(self) -> str:
        """新規話題の意図。直前に弧が完遂していれば「転換」、通常は「導入」"""
        if not self._planner_enabled:
            return ""
        if self._last_arc_end_at \
                and time.time() - self._last_arc_end_at \
                < _ARC_TRANSITION_WINDOW_SEC:
            return "transition"
        return "introduce"

    def _decide_arc(self, speeches: int, continuations: int,
                    now: float) -> tuple[str, str]:
        """継続発話の弧段階（arc_stage）と意図（intent）をルールで決める。

        speeches は弧の現在の発話数（これから話す分を含まない）、
        continuations はこれまでの継続回数。次の発話が上限（継続回数／
        発話数）に達する回なら「締め」。展開の回は、energy が低くない
        ときに確率で「問いかけ」を混ぜる。プランナー無効時は Phase 1
        までの規則（継続上限に達する回が締め、意図は付けない）に倒す。
        """
        if not self._planner_enabled:
            stage = "close" \
                if continuations + 1 >= self._arc_max_continuations \
                else "deepen"
            return stage, ""
        if continuations + 1 >= self._arc_max_continuations \
                or speeches + 1 >= self._arc_max_speeches:
            return "close", "close"
        if random.random() < self._arc_ask_probability \
                and self._decay_energy(now) >= _ARC_ASK_MIN_ENERGY:
            return "deepen", "ask"
        return "deepen", "deepen"

    def _advance_thread(self, th: dict, now: float, stage: str,
                        arc_key: str, started_at: float,
                        speeches: int) -> None:
        """継続が決まったスレッドの弧状態を1歩進める（計画時点の更新）"""
        th["continuations"] = int(th.get("continuations") or 0) + 1
        th["speeches"] = speeches + 1
        th["at"] = now
        th["stage"] = stage
        th["started_at"] = started_at
        th["arc_key"] = arc_key
        th["closing"] = stage == "close"

    def _restore_thread(self, entry: dict, now: float, stage: str,
                        arc_key: str, started_at: float, speeches: int,
                        continuations: int) -> None:
        """復帰した話題の弧状態をスレッドへ戻す。

        別の弧が進行中ならそちらを優先し、スレッドは触らない（復帰の
        発話自体は一発の続報として成立させる）。
        """
        th = self._thread
        if th is not None and th.get("arc_key") and arc_key \
                and th.get("arc_key") != arc_key:
            return
        self._thread = {
            "kind": entry.get("kind"),
            "title": entry.get("title") or "",
            "query": entry.get("query") or "",
            "at": now,
            "continuations": continuations + 1,
            "stage": stage,
            "speeches": speeches + 1,
            "started_at": started_at,
            "closing": stage == "close",
            "arc_key": arc_key,
            "topic_id": "",
        }
        self._save_state()

    def _expire_thread(self, reason: str) -> None:
        """進行中の弧を、締めに届かないまま解放する（計測に残す）"""
        th = self._thread
        if th is None:
            return
        self._thread = None
        if self._planner_enabled:
            self._arc_stats["expired"] = \
                int(self._arc_stats.get("expired") or 0) + 1
            logger.info("topic arc expired", reason=reason,
                        kind=th.get("kind"), speeches=th.get("speeches"))
        self._save_state()

    def _complete_arc(self, topic: Topic, now: float) -> None:
        """弧の完遂を記録し、スレッドと残った復帰予約を解放する。

        commit() から、プランナー有効かつ「締め」の発話が実際に喋れた
        ときだけ呼ばれる（発話失敗時は release() で締め予定を取り消し、
        弧は開いたままになる）。
        """
        th = self._thread
        speeches = int((th or {}).get("speeches") or 0)
        if self._ledger and self._ledger[-1].get("topic_id") == topic.topic_id:
            self._ledger[-1]["arc_completed"] = True
            if speeches:
                self._ledger[-1]["arc_speeches"] = speeches
        self._arc_stats["completed"] = \
            int(self._arc_stats.get("completed") or 0) + 1
        self._last_arc_end_at = now
        if th is not None and (not topic.arc_key or not th.get("arc_key")
                               or topic.arc_key == th.get("arc_key")):
            self._thread = None
        # 完遂した弧の復帰予約がスタックに残っていたら一緒に捨てる
        if topic.arc_key:
            self._stack = [e for e in self._stack
                           if e.get("arc_key") != topic.arc_key]
        logger.info("topic arc completed", kind=topic.kind,
                    speeches=speeches, label=(topic.label or "")[:40])
        # tenure 強制クローズによる完遂は、移行として数えて主題を除外する
        if self._tenure_closing_key \
                and (not topic.arc_key
                     or topic.arc_key == self._tenure_closing_key):
            self._arc_stats["tenure"] = \
                int(self._arc_stats.get("tenure") or 0) + 1
            title = (th or {}).get("title") \
                or (topic.documents[0].title if topic.documents else "") \
                or topic.label or ""
            self._register_tenure_exclusion(
                self._tenure_closing_key, title, now)
            logger.info("topic tenure transition completed",
                        reason="topic_tenure",
                        arc_key=self._tenure_closing_key[:40])
            self._tenure_closing_key = ""


    # ── 話題の任期（topic tenure、2026-11-22 longstream） ─────
    # 同じ主題（弧）が長くなりすぎた場合の「通常経路」の移行機構。
    # 弧の上限（arc_max_*）は1本の弧の中での締め判定だが、弧が置換・
    # 失効で静かに消えて同じ主題が別弧として再登場する連なりは数え
    # られていなかった（BFラン実測：正式な完遂は全ランで1件のみ、
    # 同じニュース主題が台帳60件中に4回出現）。ここでは台帳ベースで
    # 主題の連なり（チェーン）を測り、閾値を超えたら決定境界で弧を
    # 「完遂」として閉じ、その主題を一定期間、次候補から外す。
    # 役割分担：tenure は「長さ」への通常経路（締め発話で自然に畳む）。
    # LoopGuard は「言い換えレベルの反復」への緊急経路（発話後検出→
    # ヒント・強制切替・文脈リセット）で、飽和リセットも緊急側。
    # 通常は tenure が先に働き、緊急経路の発動自体を減らす想定。

    def _purge_tenure_exclusions(self, now: float) -> None:
        expired = [k for k, v in self._tenure_excluded.items()
                   if float(v.get("until") or 0) <= now]
        for k in expired:
            del self._tenure_excluded[k]

    def _register_tenure_exclusion(self, arc_key: str, title: str,
                                   now: float,
                                   chain: dict | None = None) -> None:
        """移行した主題をクールダウンの間、次候補から外す"""
        if not arc_key:
            return
        self._tenure_excluded[arc_key] = {
            "until": now + self._tenure_cooldown_sec,
            "title": title or "",
        }
        logger.info("topic tenure exclusion registered",
                    reason="topic_tenure", arc_key=arc_key[:40],
                    title=(title or "")[:40],
                    cooldown_sec=self._tenure_cooldown_sec,
                    speeches=(chain or {}).get("speeches"),
                    elapsed_sec=round(float((chain or {}).get("elapsed")
                                            or 0), 1))
        self._save_state()

    def _is_tenure_excluded(self, title: str, query: str) -> bool:
        """タイトル／クエリが tenure 除外中の主題に当たるか。

        弧キーの完全一致か、タイトル同士の Jaccard が 0.6 以上で判定する。
        """
        now = time.time()
        key = self._arc_key_for(title or "", query or "")
        ent = self._tenure_excluded.get(key)
        if ent and float(ent.get("until") or 0) > now:
            return True
        toks = _title_tokens(title or query or "")
        if not toks:
            return False
        for ent in self._tenure_excluded.values():
            if float(ent.get("until") or 0) <= now:
                continue
            etoks = _title_tokens(ent.get("title") or "")
            if etoks and _jaccard(toks, etoks) >= 0.6:
                return True
        return False

    def _matches_tenure_exclusion(self, text: str) -> bool:
        """長いテキスト（記憶文など）が除外中の主題を含むか。

        記憶文はタイトルよりずっと長いため Jaccard では希釈される。
        _theme_overlap と同じ containment（小さい側に対する共通割合）
        で測る。
        """
        if not self._tenure_excluded:
            return False
        toks = _title_tokens(text or "")
        if not toks:
            return False
        now = time.time()
        for ent in self._tenure_excluded.values():
            if float(ent.get("until") or 0) <= now:
                continue
            etoks = _title_tokens(ent.get("title") or "")
            if etoks and len(toks & etoks) / min(len(toks), len(etoks)) >= 0.5:
                return True
        return False

    def _subject_chain(self, now: float) -> dict | None:
        """現在の主題チェーン（同一 arc_key の連なり）を台帳から測る。

        対象はスレッド（弧）を開く kind（news/curiosity/theme）のみ。
        memory 話題はタイトルが常に「覚えていること」で弧キーが潰れる
        ため対象外（そちらの張り付きは話題シェアキャップと発話テーマの
        重なり判定が担当する）。
        """
        th = self._thread
        thread_key = ""
        thread_speeches = 0
        thread_started = 0.0
        if th is not None:
            thread_key = th.get("arc_key") or self._arc_key_for(
                th.get("title") or "", th.get("query") or "")
            thread_speeches = int(th.get("speeches") or 0)
            thread_started = float(th.get("started_at")
                                   or th.get("at") or 0)
        cutoff = now - self._tenure_window_sec
        counts: dict[str, dict] = {}
        for e in self._ledger:
            if e.get("kind") not in ("news", "curiosity", "theme"):
                continue
            # テーマ×記憶の話題は材料の出所が memory で、タイトルが
            # 「覚えていること」に潰れて別主題と混ざるため対象外にする
            # （kind ではなく source で判定する。2026-11-22 verifyfix）
            if e.get("source") == "memory":
                continue
            at = float(e.get("at") or 0)
            if at < cutoff:
                continue
            k = e.get("arc_key") or self._arc_key_for(
                e.get("title") or "", e.get("query") or "")
            if not k:
                continue
            c = counts.setdefault(k, {"speeches": 0, "first_at": at,
                                      "title": e.get("title")
                                      or e.get("query") or ""})
            c["speeches"] += 1
            c["first_at"] = min(c["first_at"], at)
        key = thread_key
        if not key and self._ledger:
            last = self._ledger[-1]
            if last.get("kind") in ("news", "curiosity", "theme"):
                key = last.get("arc_key") or self._arc_key_for(
                    last.get("title") or "", last.get("query") or "")
        if not key:
            return None
        c = counts.get(key) or {"speeches": 0, "first_at": 0.0, "title": ""}
        speeches = max(int(c["speeches"]), thread_speeches)
        started = thread_started or float(c["first_at"] or 0)
        if thread_started and c["first_at"]:
            started = min(thread_started, float(c["first_at"]))
        # 経過時間はこのセッション（プランナ生成時点）以降でのみ測る。
        # 永続台帳から復元した前セッション分の開始時刻をそのまま使うと、
        # エンジン停止中の壁時計まで「経過」に含まれてしまう
        # （2026-11-22 5時間検証：開始5秒後に elapsed 1742秒で到達）
        if started:
            started = max(started, float(getattr(self, "_born_at", 0.0)
                                         or 0.0))
        title = ""
        if th is not None:
            title = th.get("title") or th.get("query") or ""
        title = title or c["title"]
        return {"arc_key": key, "speeches": speeches,
                "started_at": started,
                "elapsed": max(0.0, now - started) if started else 0.0,
                "title": title, "thread_open": th is not None}

    def _tenure_exceeded(self, now: float) -> dict | None:
        """主題チェーンが tenure 閾値を超えていたらチェーン情報を返す"""
        if not self._tenure_enabled or not self._planner_enabled:
            return None
        chain = self._subject_chain(now)
        if chain is None:
            return None
        ent = self._tenure_excluded.get(chain["arc_key"])
        if ent and float(ent.get("until") or 0) > now:
            return None   # 移行済みの主題では再発火しない
        if chain["speeches"] >= self._tenure_speeches:
            return chain
        # 経過時間だけでの到達は、最低限の発話数が伴う場合に限る。
        # 従来は OR 条件で発話1件でも時間だけで発火していた
        if chain["speeches"] >= self._tenure_min_speeches_for_time \
                and chain["elapsed"] >= self._tenure_sec:
            return chain
        return None

    def _finish_arc_tenure(self, chain: dict, now: float) -> None:
        """締め材料が無いまま tenure に達した弧を、完遂として閉じる"""
        th = self._thread
        speeches = int((th or {}).get("speeches") or chain["speeches"] or 0)
        if self._ledger:
            last = self._ledger[-1]
            lk = last.get("arc_key") or self._arc_key_for(
                last.get("title") or "", last.get("query") or "")
            if lk == chain["arc_key"]:
                last["arc_completed"] = True
                if speeches:
                    last["arc_speeches"] = speeches
        self._arc_stats["completed"] = \
            int(self._arc_stats.get("completed") or 0) + 1
        self._arc_stats["tenure"] = \
            int(self._arc_stats.get("tenure") or 0) + 1
        self._last_arc_end_at = now
        kind = (th or {}).get("kind")
        self._thread = None
        if chain["arc_key"]:
            self._stack = [e for e in self._stack
                           if e.get("arc_key") != chain["arc_key"]]
        logger.info("topic arc completed", reason="topic_tenure",
                    kind=kind, speeches=speeches,
                    label=(chain["title"] or "")[:40])
        self._register_tenure_exclusion(chain["arc_key"], chain["title"],
                                        now, chain)
        self._save_state()

    async def _try_tenure_transition(self, allow_web: bool) -> Topic | None:
        """tenure 到達時の移行を試みる（plan の決定境界でのみ呼ばれる）。

        弧が開いていれば、まず「締め」の発話で自然に畳む（強制クローズ）。
        材料が取れなければ完遂として閉じ、通常の新規選択へ委ねる。
        弧が開いていない（別弧として再登場を繰り返した）場合は、主題の
        除外だけを登録する。いずれも発話の途中では割り込まない。
        """
        now = time.time()
        self._purge_tenure_exclusions(now)
        chain = self._tenure_exceeded(now)
        if chain is None:
            return None
        logger.info("topic tenure reached", reason="topic_tenure",
                    arc_key=chain["arc_key"][:40],
                    title=(chain["title"] or "")[:40],
                    speeches=chain["speeches"],
                    elapsed_sec=round(chain["elapsed"], 1),
                    thread_open=chain["thread_open"])
        th = self._thread
        if th is not None and th.get("closing"):
            # 既に締めに入っている弧は自然な完遂を待つ。除外だけ先に
            # 登録し、締め発話の commit では tenure 完遂として数える
            self._tenure_closing_key = chain["arc_key"]
            self._register_tenure_exclusion(chain["arc_key"],
                                            chain["title"], now, chain)
            return None
        if th is not None:
            topic = await self._try_continue(allow_web, force_close=True)
            if topic is not None:
                self._tenure_closing_key = chain["arc_key"]
                return topic
            self._finish_arc_tenure(chain, now)
            return None
        # 弧が開いていない到達：閉じるべき弧が無いため、従来は除外
        # 登録だけで終わり完遂の記録が残らなかった。到達と完遂を
        # 対応付けられるよう、完遂として記録してから除外する
        self._record_tenure_completion(chain, now)
        return None

    def _record_tenure_completion(self, chain: dict, now: float) -> None:
        """弧が開いていないチェーンの tenure 到達を完遂として記録する"""
        self._arc_stats["tenure"] = \
            int(self._arc_stats.get("tenure") or 0) + 1
        self._last_arc_end_at = now
        logger.info("topic arc completed", reason="topic_tenure",
                    kind=None, speeches=chain["speeches"],
                    label=(chain["title"] or "")[:40], thread_open=False)
        self._register_tenure_exclusion(chain["arc_key"], chain["title"],
                                        now, chain)
        self._save_state()

    # ── 保留回答キュー（Phase 3）──────────────────────────────
    # その場で答えられなかった事実質問（同期検索の空振り／間隔ゲート）を
    # ここに積み、Orchestrator の調査ワーカーが裏で調べて資料を付け、
    # plan() の自動経路で最優先に回答トピックとして差し込む。
    # 消費は通常の話題と同じ commit/release 方式：発話成功（commit）で
    # エントリを消し、失敗（release）では残す（回答が生成ブロック等で
    # 潰れても質問が消えないように）。

    @staticmethod
    def _question_key(text: str) -> str:
        """質問の重複判定キー（空白を落として比較する）"""
        return re.sub(r"\s+", "", (text or "")).strip()

    def enqueue_answer(self, author: str, question: str, query: str = "",
                       ttl_sec: float = _PENDING_ANSWER_TTL_SEC) -> str:
        """事実質問を保留キューへ積む。積めたらエントリID、重複・空なら ""。

        同じ質問が保留中なら新規に積まず、既存エントリのIDを返す。
        """
        q = (question or "").strip()
        if not q:
            return ""
        key = self._question_key(q)
        now = time.time()
        self._purge_expired_answers(now)
        for e in self._pending_answers:
            if e["key"] == key:
                return e["id"]
        self._pending_seq += 1
        pid = f"pa{int(now)}-{self._pending_seq}"
        self._pending_answers.append({
            "id": pid,
            "author": author or "",
            "question": q[:200],
            "query": (query or q)[:200],
            "key": key,
            "enqueued_at": now,
            "ttl": float(ttl_sec),
            "attempts": 0,
            "state": "waiting",
            "documents": [],
            "next_try_at": now,
        })
        logger.info("answer queued", author=author or "", question=q[:40])
        self._save_state()
        return pid

    def _purge_expired_answers(self, now: float) -> None:
        """期限切れの保留エントリをログを残して破棄する"""
        kept: deque[dict] = deque()
        dropped = 0
        for e in self._pending_answers:
            if now - float(e.get("enqueued_at") or 0) > float(e.get("ttl")
                                                              or _PENDING_ANSWER_TTL_SEC):
                logger.info("pending answer expired",
                            question=(e.get("question") or "")[:40],
                            state=e.get("state"))
                dropped += 1
                continue
            kept.append(e)
        self._pending_answers = kept
        if dropped:
            # 破棄も状態変化として残す（復元で期限切れが蘇らないように）
            self._save_state()

    def claim_answer_work(self) -> dict | None:
        """調査ワーカー用：再試行時刻を過ぎた waiting エントリを1件返す。

        返すときに attempts を加算する（試行回数の管理はワーカー側の
        再試行間隔リストと対応させる）。エントリ自体はキューに残る。
        """
        now = time.time()
        self._purge_expired_answers(now)
        for e in self._pending_answers:
            if e["state"] == "waiting" and now >= float(e.get("next_try_at") or 0):
                e["attempts"] = int(e.get("attempts") or 0) + 1
                return dict(e)
        return None

    def attach_answer_documents(self, pending_id: str,
                                documents: list[WebDocument]) -> bool:
        """ワーカーが調べた資料をエントリに付けて ready にする"""
        for e in self._pending_answers:
            if e["id"] == pending_id:
                e["documents"] = list(documents)[:5]
                e["state"] = "ready"
                logger.info("answer ready",
                            question=(e.get("question") or "")[:40],
                            docs=len(e["documents"]))
                self._save_state()
                return True
        return False

    def defer_answer(self, pending_id: str, delay_sec: float) -> bool:
        """検索が空振りだったエントリの次回試行を遅らせる"""
        for e in self._pending_answers:
            if e["id"] == pending_id and e["state"] == "waiting":
                e["next_try_at"] = time.time() + max(0.0, delay_sec)
                self._save_state()
                return True
        return False

    def fail_answer(self, pending_id: str) -> bool:
        """再試行を尽くしたエントリを failed にする（締め発話の対象）"""
        for e in self._pending_answers:
            if e["id"] == pending_id:
                e["state"] = "failed"
                logger.info("answer research exhausted",
                            question=(e.get("question") or "")[:40])
                self._save_state()
                return True
        return False

    def has_deliverable_answer(self) -> bool:
        """差し込み可能な回答（ready／failed）があるか。

        Orchestrator が話題の強制切替より回答を優先するかの判定に使う。
        waiting（調査中）は含めない：まだ渡せる答えが無いため、
        その間は通常の話題進行を止めない。
        """
        now = time.time()
        self._purge_expired_answers(now)
        return any(e["state"] in ("ready", "failed")
                   for e in self._pending_answers)

    def pending_answers(self) -> list[dict]:
        """保留エントリのスナップショット（状態表示・テスト用）"""
        return [dict(e) for e in self._pending_answers]

    def _is_pending_question(self, text: str) -> bool:
        """このテキストが保留中の質問と同一か（comment hook の二重処理防止）"""
        key = self._question_key(text)
        if not key:
            return False
        now = time.time()
        self._purge_expired_answers(now)
        return any(e["key"] == key for e in self._pending_answers)

    def _try_pending_answer(self) -> Topic | None:
        """保留回答をトピック化する（plan() 自動経路の最優先チェック用）。

        ready なら資料付き、failed なら「分からなかった」締め用の
        プレースホルダ資料付きで返す。エントリの削除は commit() で行う
        （ここで消すと、発話失敗時に質問ごと消えてしまうため）。
        """
        now = time.time()
        self._purge_expired_answers(now)
        # 視聴者への反応として差し込む発話なので意図は react（Phase 3）
        react = "react" if self._planner_enabled else ""
        for e in self._pending_answers:
            if e["state"] == "ready":
                logger.info("pending answer planned",
                            question=(e.get("question") or "")[:40])
                return Topic(kind="answer",
                             label=f"質問への回答: {(e.get('question') or '')[:24]}",
                             query=e.get("question") or "",
                             documents=list(e.get("documents") or []),
                             pending_id=e["id"],
                             author=e.get("author") or "",
                             intent=react)
            if e["state"] == "failed":
                logger.info("pending answer planned (not found)",
                            question=(e.get("question") or "")[:40])
                doc = WebDocument(title="調べた質問",
                                  snippet=e.get("question") or "",
                                  source="answer_failed", site="")
                return Topic(kind="answer",
                             label=f"質問への回答（不明）: {(e.get('question') or '')[:24]}",
                             query=e.get("question") or "",
                             documents=[doc],
                             pending_id=e["id"],
                             author=e.get("author") or "",
                             intent=react)
        return None

    def commit(self, topic: Topic) -> None:
        """話題が実際に発話されたことを記録する（消費の確定）"""
        now = time.time()
        if topic.kind == "answer" and topic.pending_id:
            # 保留回答は発話できた時点で消費する（Phase 3）
            self._pending_answers = deque(
                e for e in self._pending_answers
                if e["id"] != topic.pending_id)
        if topic.kind == "news" and topic.documents:
            doc = topic.documents[0]
            if self._news is not None:
                self._news.mark_used(doc)
            self._reserved.pop(doc.key, None)
            self._last_news_site = doc.site or ""
        self._last_kind_at[topic.kind] = now
        self._ledger.append({
            "topic_id": topic.topic_id,
            "kind": topic.kind,
            "label": topic.label,
            "query": topic.query,
            "title": topic.documents[0].title if topic.documents else "",
            "at": now,
            "intent": topic.intent,
            "arc_stage": topic.arc_stage,
            # 弧キーを台帳にも保存する（2026-11-22 verifyfix）。従来は
            # 保存されず、チェーン集計がタイトル由来キーで再計算する
            # ため弧キー（クエリ由来）と食い違い、弧の継続中はチェーン
            # 履歴が評価対象から外れていた
            "arc_key": topic.arc_key or self._arc_key_for(
                topic.documents[0].title if topic.documents else "",
                topic.query),
            # 材料の出所。テーマ×記憶の話題をチェーン集計から外す判定に
            # 使う（タイトル「覚えていること」で潰れる問題の対策）
            "source": topic.documents[0].source if topic.documents else "",
        })
        if topic.documents and (topic.kind == "memory"
                                or (topic.kind == "theme"
                                    and topic.documents[0].source == "memory")):
            # 話題化した記憶を記録する。L3 常駐注入のサンプリング側が
            # この台帳を見て、同じ記憶の再浮上をクールダウン中は避ける
            # （STUDY §3-C3 の最小形：二経路の重複を断つ）
            key = self._memory_text_key(topic.documents[0].snippet)
            if key:
                self._topic_memory_used_at[key] = now
                self._topic_memory_used_at = {
                    k: at for k, at in self._topic_memory_used_at.items()
                    if now - at < _TOPIC_MEMORY_COOLDOWN_SEC * 4}
        # 会話の流れ管理：新しいWeb話題はスレッド（弧）を開き、それ以外は
        # 閉じる。続報・続き（resumed）の commit では弧を前進させる。
        # 質問への回答（answer）は割り込みとして扱い、スレッドは開かず
        # 進行中のものも消さない（回答で会話を閉じるのが役割。Phase 3）
        if topic.kind == "answer":
            pass
        elif topic.kind in ("news", "curiosity", "theme") and not topic.resumed:
            self._thread = {
                "kind": topic.kind,
                "title": topic.documents[0].title if topic.documents else "",
                "query": topic.query,
                "at": now,
                "continuations": 0,
                # 弧の状態（Phase 3）：導入を1発話目として数え始める
                "stage": "introduce",
                "speeches": 1,
                "started_at": now,
                "closing": False,
                "arc_key": topic.arc_key or self._arc_key_for(
                    topic.documents[0].title if topic.documents else "",
                    topic.query),
                "topic_id": topic.topic_id,
            }
        elif topic.kind in ("news", "curiosity", "theme"):
            # 継続・続報の commit：弧の所在を同期する。「締め」の発話が
            # 実際に喋れたら、ここで初めて弧を解放する（ADDENDUM：
            # 話題の解放は締め到達時か上限超過のときだけ）
            th = self._thread
            if th is not None and (not topic.arc_key
                                   or not th.get("arc_key")
                                   or topic.arc_key == th.get("arc_key")):
                th["topic_id"] = topic.topic_id
                th["at"] = now
            if self._planner_enabled and topic.arc_stage == "close":
                self._complete_arc(topic, now)
        elif topic.kind not in ("news", "curiosity", "theme"):
            self._thread = None
        self._save_state()

    def release(self, topic: Topic) -> None:
        """発話に至らなかった話題の予約を解放する（ネタは消えない）"""
        # kind="answer" は意図的に何もしない：保留エントリを残し、
        # 次回の plan() で再び差し込まれるようにする（Phase 3）
        if topic.kind == "news" and topic.documents:
            self._reserved.pop(topic.documents[0].key, None)
        # 「締め」として計画した発話が失敗したら、締め予定を取り消して
        # 弧は開いたままにする（完遂は commit された締め発話でのみ成立）
        if topic.arc_stage == "close" and self._tenure_closing_key \
                and (not topic.arc_key
                     or topic.arc_key == self._tenure_closing_key):
            self._tenure_closing_key = ""
        th = self._thread
        if th is not None and topic.arc_stage == "close" \
                and th.get("closing") \
                and (not topic.arc_key or not th.get("arc_key")
                     or topic.arc_key == th.get("arc_key")):
            th["closing"] = False

    # ── 話題化した記憶の台帳（L3注入との共有）と話題シェア ──────

    @staticmethod
    def _memory_text_key(text: str) -> str:
        """記憶テキストの同一判定キー（Orchestrator の種キーと同じ先頭80字）"""
        return (text or "").strip()[:80]

    def is_topic_memory_recent(self, text: str) -> bool:
        """この記憶が最近（クールダウン内）話題として使われたか。

        L3 常駐注入（CharacterAgent.build_idle_memory）のサンプリング側が、
        話題化したばかりの記憶をプロンプトへ再浮上させないために使う。
        """
        key = self._memory_text_key(text)
        if not key:
            return False
        now = time.time()
        for used_key, at in self._topic_memory_used_at.items():
            if now - at >= _TOPIC_MEMORY_COOLDOWN_SEC:
                continue
            if used_key == key or key.startswith(used_key) \
                    or used_key.startswith(key):
                return True
        return False

    def memory_share(self, window: int | None = None) -> float:
        """直近ウィンドウでの memory 話題の話題シェア（0..1）。台帳が空なら0"""
        n = window or self._memory_cap_window
        entries = list(self._ledger)[-n:]
        if not entries:
            return 0.0
        return sum(1 for e in entries if e.get("kind") == "memory") / len(entries)

    def theme_share(self, window: int | None = None) -> float:
        """直近ウィンドウでの theme 話題の話題シェア（0..1）。台帳が空なら0"""
        n = window or self._theme_share_window
        entries = list(self._ledger)[-n:]
        if not entries:
            return 0.0
        return sum(1 for e in entries if e.get("kind") == "theme") / len(entries)

    def _theme_due(self) -> bool:
        """テーマ話題を差し込むべき局面か（話題シェア目標ベース、2026-11-22）"""
        if self._theme_share_target <= 0:
            return False
        theme = (self._theme_provider() or "").strip()
        if not theme or not self._seed_ok(theme):
            return False
        entries = list(self._ledger)[-self._theme_share_window:]
        if entries and entries[-1].get("kind") == "theme":
            return False   # テーマ話題の直後は1回空ける（連投防止）
        return self.theme_share() < self._theme_share_target

    # ── force_topic の種まきキュー（2026-11-22 themefix） ────

    def enqueue_forced_topic(self, kind: str = "", query: str = "",
                             label: str = "") -> int:
        """話題の種まきを受け付ける。次回以降の自動決定で消費される。

        戻り値は投入後のキュー長。旧 force_topic は無ログだったため、
        投入・消費の両方を必ずログに残す。
        """
        kind = (kind or "").strip()
        if kind == "auto":
            kind = ""
        entry = {
            "kind": kind,
            "query": (query or "").strip(),
            "label": (label or "").strip(),
            "enqueued_at": time.time(),
        }
        self._forced_topics.append(entry)
        while len(self._forced_topics) > _FORCED_TOPIC_MAX:
            dropped = self._forced_topics.popleft()
            logger.info("forced topic dropped (queue full)",
                        kind=dropped.get("kind") or "auto",
                        query=(dropped.get("query") or "")[:40])
        logger.info("forced topic enqueued",
                    kind=kind or "auto", query=entry["query"][:40],
                    queue=len(self._forced_topics))
        return len(self._forced_topics)

    def _purge_forced_topics(self, now: float) -> None:
        kept: deque[dict] = deque()
        for e in self._forced_topics:
            if now - float(e.get("enqueued_at") or 0) > _FORCED_TOPIC_TTL_SEC:
                logger.info("forced topic expired",
                            kind=e.get("kind") or "auto",
                            query=(e.get("query") or "")[:40])
            else:
                kept.append(e)
        self._forced_topics = kept

    async def _try_forced_topic(self, allow_web: bool,
                                allow_web_theme: bool | None = None
                                ) -> Topic | None:
        # テーマ話題の検索は、全体の Web 間隔ゲートとは別に許可され
        # うる（allow_web_theme。plan() の説明を参照）
        theme_web = allow_web or bool(allow_web_theme)
        """種まきキューを消費する。材料が取れなければ通常フローへ委ねる"""
        self._purge_forced_topics(time.time())
        if not self._forced_topics:
            return None
        entry = self._forced_topics.popleft()
        kind = entry.get("kind") or ""
        query = entry.get("query") or ""
        logger.info("forced topic consumed",
                    kind=kind or "auto", query=query[:40],
                    waited_sec=round(time.time()
                                     - float(entry.get("enqueued_at") or 0), 1))
        topic: Topic | None = None
        if kind == "news":
            topic = self._plan_news_filtered(query)
            if topic is None:
                logger.info("forced news had no matching article; "
                            "falling back to any news", query=query[:40])
                topic = self._plan_news()
        elif kind == "theme":
            topic = await self._plan_theme(query, theme_web)
        elif kind == "curiosity":
            topic = await self._plan_curiosity(query)
        elif kind == "memory":
            topic = self._plan_memory()
        elif query:
            # kind 未指定＋クエリ指定は、そのクエリをテーマ的お題として扱う
            topic = await self._plan_theme(query, theme_web)
        else:
            # kind・クエリ未指定：配信テーマがあればテーマ話題、無ければ
            # 通常抽選に委ねる（None を返して通常フローを進める）
            theme = (self._theme_provider() or "").strip()
            if theme:
                topic = await self._plan_theme("", theme_web)
        if topic is None:
            logger.info("forced topic fizzled (no material)",
                        kind=kind or "auto", query=query[:40])
            return None
        if entry.get("label"):
            topic.label = entry["label"]
        return topic

    # ── 種類のUtilityスコア ─────────────────────────────────

    def _score_kinds(self, allow_web: bool, context: dict | None) -> dict[str, float]:
        now = time.time()
        ctx = context or {}
        silence = float(ctx.get("silence_sec") or 0.0)
        energy = self._decay_energy(now)
        w = self.weights
        scores: dict[str, float] = {}

        # memory：常時利用可能な床。ニュースが強いときは少し譲る。
        # 定数床のままだと重み抽選で選ばれ続ける（実測で話題シェア59%）ため、
        # news/curiosity と同様に同種クールダウンと鮮度減衰を効かせる
        # （Phase 2 案B）。最後に memory を話した時刻は commit() が
        # _last_kind_at に記録済みなので、新しい状態は持たない。
        u_memory = max(0.0, w.get("memory", 0.0))
        since_memory = now - self._last_kind_at.get("memory", self._born_at)
        u_memory *= 0.45 + 0.55 * min(1.0, since_memory / 480.0)
        if self._ledger and self._ledger[-1].get("kind") == "memory" \
                and since_memory < _KIND_COOLDOWN_SEC:
            u_memory *= 0.4
        scores["memory"] = u_memory

        # news：未話の新鮮な記事があるほど・ニュースから遠ざかるほど上がる
        u_news = 0.0
        if allow_web and self._news is not None:
            self._expire_reservations(now)
            pool = [d for d in self._news.unused_items()
                    if d.key not in self._reserved]
            if pool:
                ages = [d.age_hours(now) for d in pool]
                ages = [a for a in ages if a is not None]
                freshest = min(ages) if ages else 3.0
                freshness = 1.3 if freshest < 1 else 1.0 if freshest < 6 else 0.75
                since_news = now - self._last_kind_at.get("news", self._born_at)
                drought = min(2.2, 0.45 + since_news / 480.0)
                pool_factor = min(1.2, 0.7 + 0.1 * len(pool))
                u_news = w.get("news", 0.0) * freshness * drought * pool_factor
                if self._ledger and self._ledger[-1].get("kind") == "news" \
                        and since_news < _KIND_COOLDOWN_SEC:
                    u_news *= 0.35
        scores["news"] = max(0.0, u_news)

        # curiosity：空振りが続くと下がり、沈黙が長く退屈なほど上がる
        u_cur = 0.0
        if allow_web and self._search is not None:
            recent_fails = sum(1 for t in self._search_failures if now - t < 900)
            fail_factor = max(0.25, 1.0 - 0.25 * recent_fails)
            silence_boost = 1.0 + min(0.8, silence / 600.0)
            boredom_boost = 1.2 - 0.4 * energy
            u_cur = w.get("curiosity", 0.0) * fail_factor * silence_boost * boredom_boost
            since_cur = now - self._last_kind_at.get("curiosity", self._born_at)
            if self._ledger and self._ledger[-1].get("kind") == "curiosity" \
                    and since_cur < _KIND_COOLDOWN_SEC:
                u_cur *= 0.4
            # 実行待ちの意図があると、調べもの話題を少しだけ優先する
            # （Phase 2: 意図の持続を自律トークに繋ぐ。ブーストは控えめに）
            if u_cur > 0 and self._intention_items():
                u_cur *= 1.3
        scores["curiosity"] = max(0.0, u_cur)

        if scores["news"] > scores["memory"] > 0:
            scores["memory"] *= 0.8
        return scores

    def _pick_kind(self, allow_web: bool, context: dict | None = None,
                   exclude: set[str] | None = None) -> str:
        scores = self._score_kinds(allow_web, context)
        for k in (exclude or set()):
            scores[k] = 0.0
        # 話題シェアハードキャップ（STUDY §3-C1）：直近ウィンドウで memory が
        # 支配的なら、材料のある他候補が残っている間は memory を候補から
        # 外す。重み調整ではなく quota として効かせるのが要点で、台帳の
        # サンプルが少ないうちは誤爆を避けるため効かせない
        if scores.get("memory", 0.0) > 0 \
                and any(v > 0 for k, v in scores.items() if k != "memory"):
            window_entries = list(self._ledger)[-self._memory_cap_window:]
            if len(window_entries) >= _MEMORY_CAP_MIN_SAMPLES:
                share = sum(1 for e in window_entries
                            if e.get("kind") == "memory") / len(window_entries)
                if share >= self._memory_cap_ratio:
                    scores["memory"] = 0.0
                    logger.info("memory share cap applied",
                                share=round(share, 2),
                                cap=self._memory_cap_ratio,
                                window=len(window_entries))
        self._last_utilities = scores
        kinds = [k for k, v in scores.items() if v > 0]
        if not kinds:
            return "memory"
        picked = random.choices(kinds, weights=[scores[k] for k in kinds], k=1)[0]
        logger.info("topic decision",
                    picked=picked,
                    utilities={k: round(v, 2) for k, v in scores.items()},
                    energy=round(self._energy, 2))
        return picked

    # ── コメントの拾い直し ──────────────────────────────────

    def note_comment(self, author: str, text: str) -> None:
        """応答したコメントを記録する（後の自律トークで適度に拾い直すため）"""
        text = (text or "").strip()
        if len(text) < 4:
            return
        # 調査経路に入った質問は hook に残さない（Phase 3）。答えの無い
        # 生の質問が後で雑談に織り込まれて消える経路を塞ぐための保険で、
        # 主な分岐は Orchestrator._process_comment 側にある
        if self._is_pending_question(text):
            logger.info("comment hook skipped (answer pending)",
                        text=text[:30])
            return
        self._comment_hook = {"author": author or "", "text": text[:120],
                              "at": time.time()}

    def _try_comment_hook(self) -> Topic | None:
        """直近のコメントの話題を自律トークで拾い直す。

        応答した直後（45秒以内）は拾わない。10分を過ぎたら捨てる。
        拾うのは最大で4分に1回まで、確率55%（＝適度に）。
        """
        hook = self._comment_hook
        if not hook:
            return None
        now = time.time()
        age = now - float(hook.get("at") or 0)
        if age > 600:
            self._comment_hook = None
            return None
        if age < 45:
            return None
        if now - self._last_hook_woven_at < 240:
            return None
        if random.random() > 0.55:
            return None
        self._comment_hook = None
        self._last_hook_woven_at = now
        author = hook.get("author") or "視聴者"
        text = hook.get("text") or ""
        doc = WebDocument(title=f"{author}さんのコメント", snippet=text,
                          source="comment", site="ライブチャット")
        logger.info("comment woven into talk", author=author, text=text[:30])
        return Topic(kind="comment", label=f"コメントから: {author}",
                     query=text, documents=[doc],
                     intent="react" if self._planner_enabled else "")

    # ── 話題の継続（スレッド） ──────────────────────────────

    async def _try_continue(self, allow_web: bool,
                            force_close: bool = False) -> Topic | None:
        """直近の話題の続きを話し、会話に流れを作る（最大2回まで・確率60%）。

        force_close=True は tenure 到達時の強制クローズ用：確率と継続
        上限を無視し、「締め」の発話として弧を進める（材料の取得と
        期限切れ・ban の扱いは通常経路と同じ）。
        """
        th = self._thread
        if not th or not allow_web:
            return None
        banned = self.banned_words()
        if banned and any(w in (th.get("title") or "") or w in (th.get("query") or "")
                          for w in banned):
            self._thread = None
            return None
        now = time.time()
        if now - float(th.get("at") or 0) > 600:
            # 最終発話から10分空いたら弧は自然消滅（期限切れとして数える）
            self._expire_thread("inactive")
            return None
        started_at = float(th.get("started_at") or th.get("at") or now)
        if self._planner_enabled and now - started_at > self._arc_max_sec:
            # 弧の開始からの上限時間を超えたら、締め切れなくても解放する
            # （弧が永遠に続かないための上限。ADDENDUM の完遂条件の片方）
            self._expire_thread("arc_max_sec")
            return None
        if not force_close \
                and int(th.get("continuations") or 0) \
                >= self._arc_max_continuations:
            return None
        if not force_close and random.random() > 0.6:
            return None
        kind = th.get("kind")
        # 弧の段階と次発言の意図はルールで決める（Phase 3 / STUDY §6-(a1)）。
        # 継続上限・発話数上限に達する回が「締め」、展開の一部は「問いかけ」
        speeches = int(th.get("speeches")
                       or int(th.get("continuations") or 0) + 1)
        if force_close:
            stage, intent = "close", "close"
        else:
            stage, intent = self._decide_arc(
                speeches, int(th.get("continuations") or 0), now)
        arc_key = th.get("arc_key") or self._arc_key_for(
            th.get("title") or "", th.get("query") or "")
        if kind == "news":
            doc = self._pick_followup_doc(th)
            if doc is None:
                return None
            self._advance_thread(th, now, stage, arc_key, started_at,
                                 speeches)
            logger.info("topic continue", kind="news", title=doc.title[:40])
            return Topic(kind="news", label=f"続き: {doc.title[:30]}",
                         query=doc.title, documents=[doc], resumed=True,
                         arc_stage=stage, prev_speech=self._last_speech_text,
                         intent=intent, arc_key=arc_key)
        if kind == "curiosity":
            seed = self._seed_from_query(th.get("query") or "")
            if not seed or self._search is None:
                return None
            # クエリは種そのまま（2026-11-22 themefix のクエリ単純化）
            q = seed
            if q == th.get("query"):
                return None
            docs = await self._search_with_cap(q)
            if not docs:
                return None
            self._advance_thread(th, now, stage, arc_key, started_at,
                                 speeches)
            logger.info("topic continue", kind="curiosity", query=q[:40])
            return Topic(kind="curiosity", label=f"続き: {q[:30]}",
                         query=q, documents=docs[: self.max_documents],
                         resumed=True, arc_stage=stage,
                         prev_speech=self._last_speech_text,
                         intent=intent, arc_key=arc_key)
        if kind == "theme":
            # テーマ話題の継続（2026-11-22 themefix）：テーマ単体／
            # テーマ＋キーワード1つで調べ直し、弧を進める
            if self._search is None:
                return None
            theme = (self._theme_provider() or "").strip()
            head = theme or self._seed_from_query(th.get("query") or "")
            if not head:
                return None
            candidates = self._theme_query_candidates(head) \
                if theme else [head]
            docs = []
            q = ""
            for cand in candidates:
                if cand == th.get("query"):
                    continue
                docs = await self._search.search(
                    cand, limit=self.max_documents)
                if docs:
                    q = cand
                    break
            if not docs:
                return None
            self._advance_thread(th, now, stage, arc_key, started_at,
                                 speeches)
            logger.info("topic continue", kind="theme", query=q[:40])
            return Topic(kind="theme", label=f"続き: {q[:30]}",
                         query=q, documents=docs[: self.max_documents],
                         resumed=True, arc_stage=stage,
                         prev_speech=self._last_speech_text,
                         intent=intent, arc_key=arc_key)
        return None

    @staticmethod
    def _seed_from_query(query: str) -> str:
        """好奇心クエリから種（キーワード部分）を取り出す"""
        q = (query or "").strip()
        for tpl in _CURIOSITY_TEMPLATES:
            suffix = tpl.replace("{kw}", "").strip()
            if suffix and q.endswith(suffix):
                return q[: -len(suffix)].strip()
        return q

    # ── 話題スタック（中断と復帰） ──────────────────────────

    async def _try_resume(self, allow_web: bool) -> Topic | None:
        """中断された話題の続きを試みる（スタックの pop）"""
        if not allow_web or not self._stack:
            return None
        now = time.time()
        top = self._stack[-1]
        age = now - float(top.get("suspended_at") or 0)
        if age > 600:
            self._stack.pop()
            return None
        if self._tenure_enabled \
                and self._is_tenure_excluded(top.get("title") or "",
                                             top.get("query") or ""):
            # tenure で移行済みの主題は復帰させない
            self._stack.pop()
            logger.info("topic resume skipped (tenure exclusion)",
                        kind=top.get("kind"),
                        title=(top.get("title") or "")[:40])
            return None
        if age > 240 and random.random() < 0.5:
            return None   # 時間が経っていたら半分は見送り（スタックは残す）
        # 弧の状態（_suspend_current が退避したスナップショット）を取り出す。
        # 弧はコメント割り込みではリセットされず、復帰は弧の続きとして
        # 数える（ADDENDUM）。スナップショットが無い旧形式のエントリは
        # 導入済みの1発話として扱う
        speeches = int(top.get("arc_speeches") or 1)
        continuations = int(top.get("continuations") or 0)
        started_at = float(top.get("arc_started_at")
                           or top.get("suspended_at") or now)
        arc_key = top.get("arc_key") or self._arc_key_for(
            top.get("title") or "", top.get("query") or "")
        if self._planner_enabled and now - started_at > self._arc_max_sec:
            # 中断が長引いて弧の上限時間を超えたら、復帰させず解放する。
            # 同じ弧がスレッド側にも残っていれば一緒に捨て、二重に
            # 期限切れを数えないようにする
            self._stack.pop()
            self._arc_stats["expired"] = \
                int(self._arc_stats.get("expired") or 0) + 1
            th = self._thread
            if th is not None and arc_key and th.get("arc_key") == arc_key:
                self._thread = None
            logger.info("topic arc expired on resume", kind=top.get("kind"))
            return None
        stage, intent = self._decide_arc(speeches, continuations, now)
        kind = top.get("kind")
        if kind == "news":
            doc = self._pick_followup_doc(top)
            if doc is None:
                self._stack.pop()
                return None
            self._stack.pop()
            if self._planner_enabled:
                self._restore_thread(top, now, stage, arc_key, started_at,
                                     speeches, continuations)
            logger.info("topic resume", kind="news", title=doc.title[:40])
            return Topic(kind="news", label=f"続報: {doc.title[:30]}",
                         query=doc.title, documents=[doc], resumed=True,
                         arc_stage=stage,
                         prev_speech=self._last_speech_text,
                         intent=intent, arc_key=arc_key)
        if kind in ("curiosity", "theme"):
            q = (top.get("query") or "").strip()
            if not q or self._search is None:
                self._stack.pop()
                return None
            docs = await self._search.search(q, limit=self.max_documents)
            if not docs:
                self._stack.pop()
                return None
            self._stack.pop()
            if self._planner_enabled:
                self._restore_thread(top, now, stage, arc_key, started_at,
                                     speeches, continuations)
            logger.info("topic resume", kind=kind, query=q[:40])
            return Topic(kind=kind, label=f"続報: {q[:30]}",
                         query=q, documents=docs[: self.max_documents],
                         resumed=True, arc_stage=stage,
                         prev_speech=self._last_speech_text,
                         intent=intent, arc_key=arc_key)
        self._stack.pop()
        return None

    def _pick_followup_doc(self, suspended: dict) -> WebDocument | None:
        """中断されたニュースと似た（＝同じ事件の続報らしい）未話の記事を選ぶ"""
        if self._news is None:
            return None
        now = time.time()
        self._expire_reservations(now)
        pool = [d for d in self._news.unused_items()
                if d.key not in self._reserved]
        if not pool:
            return None
        base = _title_tokens(suspended.get("title") or suspended.get("query") or "")
        if not base:
            return None
        best: WebDocument | None = None
        best_score = 0.0
        for d in pool:
            sim = _jaccard(_title_tokens(d.title), base)
            age = d.age_hours(now)
            fresh = math.exp(-age / 6.0) if age is not None else 0.45
            score = sim * (0.5 + 0.5 * fresh)
            if score > best_score:
                best, best_score = d, score
        if best is None or best_score < 0.06:
            return None
        if self._tenure_enabled \
                and self._is_tenure_excluded(best.title, best.title):
            return None   # tenure で移行済みの主題の続報は追わない
        self._reserved[best.key] = now
        return best

    # ── ニュース記事のスコア選択 ────────────────────────────

    def _pick_news_doc(self) -> WebDocument | None:
        if self._news is None:
            return None
        now = time.time()
        self._expire_reservations(now)
        pool = [d for d in self._news.unused_items()
                if d.key not in self._reserved]
        if not pool:
            return None
        if self._tenure_enabled:
            # tenure で移行済みの主題は候補から外す（全滅したら従来どおり
            # 選ぶ。話題を枯らさないことを優先する）
            fresh = [d for d in pool
                     if not self._is_tenure_excluded(d.title, d.title)]
            if fresh:
                pool = fresh
        recent_tokens = [_title_tokens(t) for t in self._recent_titles(8)]
        speech_tokens = list(self._recent_speech_tokens)
        scored: list[tuple[WebDocument, float]] = []
        for d in pool:
            age = d.age_hours(now)
            fresh = math.exp(-age / 6.0) if age is not None else 0.45
            toks = _title_tokens(d.title)
            novelty = 1.0 - max((_jaccard(toks, rt) for rt in recent_tokens),
                                default=0.0)
            # 直近発話のテーマと重なる記事には追加のペナルティ（Phase 4.5）。
            # 発話側が長いので containment で測り、係数は控えめにする
            if speech_tokens and toks:
                speech_con = max(
                    (len(toks & st) / min(len(toks), len(st))
                     for st in speech_tokens if st), default=0.0)
                novelty *= (1.0 - 0.5 * speech_con)
            site_pen = 0.55 if d.site and d.site == self._last_news_site else 1.0
            score = (0.25 + fresh) * (0.5 + 0.5 * novelty) * site_pen \
                + random.random() * 0.05
            scored.append((d, score))
        doc = random.choices([d for d, _ in scored],
                             weights=[s for _, s in scored], k=1)[0]
        self._reserved[doc.key] = now
        return doc

    # ── 好奇心クエリ ────────────────────────────────────────

    def _intention_items(self) -> list[dict]:
        """実行可能な next_action を持つ意図の一覧（provider が無ければ空）"""
        if self._intention_provider is None:
            return []
        try:
            items = self._intention_provider() or []
        except Exception:
            return []
        return [it for it in items
                if isinstance(it, dict) and (it.get("next_action") or "").strip()]

    @staticmethod
    def _seed_ok(seed: str) -> bool:
        """好奇心クエリの種として妥当か（Phase 3 の衛生フィルタ）。

        自己言及・メタな種（_CURIOSITY_SEED_NG）と、文の形をした種
        （自己説明文など）を弾く。意図そのものを消す話ではないので、
        フィルタは種の採用可否だけに効かせる。
        """
        s = (seed or "").strip()
        if not s or len(s) > 60:
            return False
        if _CURIOSITY_SEED_NG.search(s):
            return False
        if s.endswith(("。", "です", "ます")):
            return False
        return True

    def _build_query(self) -> str:
        """好奇心検索のクエリを作る。配信テーマ優先、無ければ記憶のキーワード。

        クエリは「種そのまま」か「テーマ＋キーワード1つ」までに単純化した
        （2026-11-22 themefix）。旧方式は種にテンプレート（最新／おすすめ等）
        を付け、テーマ時は記憶キーワードまで混ぜた複合クエリになり、検索が
        連続で空振りした（BFラン分析の問題7）。テーマ語は ban の対象外
        （ban_word 側で免除）なので、テーマを ban で落とす分岐は持たない。
        台帳にある直近のクエリと同じ種は避ける（最大3回引き直す）。
        """
        theme = (self._theme_provider() or "").strip()
        if theme and not self._seed_ok(theme):
            theme = ""
        banned = self.banned_words()
        # 直近で種に使ったキーワードはクールダウン中は避ける（Phase 4.5）
        keywords = [k for k in self._keyword_provider()
                    if k.strip() and not any(w in k for w in banned)
                    and not self._seed_on_cooldown(k)
                    and not self._is_tenure_excluded(k, k)
                    and self._seed_ok(k)]
        ledger_queries = {e.get("query", "") for e in self._ledger}
        # 取り組んでいる意図があれば、その主題を種にすることがある
        # （半分程度の確率に留め、通常の種選びを壊さない）。
        # ただしクールダウン中の種と、直近発話テーマと重なる種は避ける
        # （最有力の意図へ恒常的に引き寄せられないように。Phase 4.5）
        intention_seeds = []
        for it in self._intention_items():
            seed = (it.get("title") or "").strip()
            if seed and not any(w in seed for w in banned) \
                    and not self._seed_on_cooldown(seed) \
                    and not self._is_tenure_excluded(seed, seed) \
                    and self._seed_ok(seed) \
                    and self._theme_overlap(seed) < _THEME_OVERLAP_SEED_MAX:
                intention_seeds.append(seed)
        if intention_seeds and random.random() < 0.5:
            seed = random.choice(intention_seeds)
            if seed not in ledger_queries and seed not in self._history:
                self._mark_seed_used(seed)
                return seed
        best = ""
        best_mark = ""
        for _ in range(3):
            q = ""
            mark = ""
            if theme and keywords and random.random() < 0.5:
                kw = random.choice(keywords)
                q = f"{theme} {kw}"
                mark = kw
            elif theme:
                q = theme
            elif keywords:
                q = random.choice(keywords)
                mark = q
            if not q:
                return ""
            best = q
            best_mark = mark
            if q not in ledger_queries and q not in self._history:
                if mark:
                    self._mark_seed_used(mark)
                return q
        if best_mark:
            self._mark_seed_used(best_mark)
        return best

    # ── メイン ────────────────────────────────────────────────

    async def plan(self, kind: str = "", query: str = "",
                   allow_web: bool = True,
                   allow_web_theme: bool | None = None,
                   context: dict | None = None) -> Topic:
        """
        次の話題を決めて材料を集める。

        kind を指定すればその種類を強制する（ダッシュボードの手動実行用）。
        query を指定した場合は curiosity（そのクエリで検索）になる。
        材料が集まらなければ memory にフォールバックする。
        自動決定では、保留回答→force_topic の種まき→コメント拾い→
        復帰→継続の順に割り込みを確認し、どれも無ければ配信テーマの
        話題シェア目標（_theme_due）を確認してから通常の抽選に入る。
        消費はここでは確定しない。発話できたら commit()、できなければ
        release() を呼ぶこと（ニュース記事は予約だけされる）。
        """
        # テーマ話題の検索可否は全体の allow_web とは独立に扱える
        # （2026-11-22 verifyfix）。テーマ設定中、テーマ話題まで間隔
        # ゲートで検索を止めると材料なしお題に劣化するため、呼び出し
        # 側は allow_web_theme でテーマ検索だけを許可できる
        theme_web = allow_web or bool(allow_web_theme)
        if query:
            kind = "curiosity"
        if not kind and not query:
            # 保留回答が最優先（Phase 3）：視聴者が回答を待っているため、
            # comment hook・復帰・継続のどれよりも前に差し込む
            auto: Topic | None = self._try_pending_answer()
            if auto is None:
                # force_topic の種まき（2026-11-22 themefix）：運用が
                # 明示的に積んだ話題は、保留回答の次に優先して消費する
                auto = await self._try_forced_topic(allow_web,
                                                    allow_web_theme)
            if auto is None:
                auto = self._try_comment_hook()
            if auto is None:
                # 話題の任期（2026-11-22 longstream）：同じ主題が長く
                # なりすぎていたら、復帰・継続より前に移行を試みる
                auto = await self._try_tenure_transition(allow_web)
            if auto is None:
                auto = await self._try_resume(allow_web)
            if auto is None:
                auto = await self._try_continue(allow_web)
            if auto is not None:
                self._seq += 1
                auto.topic_id = f"t{int(time.time())}-{self._seq}"
                self.last_topic = auto
                if auto.query:
                    self._history.append(auto.query)
                return auto
        topic: Topic | None = None
        if not kind and not query and self._theme_due():
            # 配信テーマの話題シェア目標（2026-11-22 themefix）：テーマ設定中は
            # 話題決定の一定割合をテーマ話題にする。保留回答・種まき・
            # コメント拾い・弧の継続より後、新規話題の抽選より前に差し込む
            topic = await self._plan_theme("", theme_web)
            if topic is not None:
                kind = "theme"
                logger.info("theme topic selected (quota)",
                            share=round(self.theme_share(), 2),
                            target=self._theme_share_target)
        if topic is None:
            kind = kind or self._pick_kind(allow_web, context)
            if kind == "news":
                topic = self._plan_news()
                if topic is None:
                    # ニュース在庫が尽きたら、材料なしの記憶話題へ黙って落ちず、
                    # ニュースを除いて選び直す（2026-11-14の2時間テストで、
                    # 在庫切れのまま「強制ニュース→材料なし」が約100分続いた）
                    logger.warning("news pool exhausted, re-picking kind without news")
                    kind = self._pick_kind(allow_web, context, exclude={"news"})
                    if kind == "curiosity":
                        topic = await self._plan_curiosity(query)
            elif kind == "curiosity":
                topic = await self._plan_curiosity(query)
            elif kind == "theme":
                topic = await self._plan_theme(query, theme_web)

        if topic is None:
            topic = self._plan_memory()
            if topic is None and self._filler_enabled:
                # フォールバック終端の多様化（STUDY §3-C2）：記憶の材料も
                # 無いときは、材料なしの memory 話題で黙り込まず定番小噺へ
                # 落とす。全経路の終端が memory になる構造を断つ
                topic = self._plan_filler()
                logger.info("fallback to filler topic", label=topic.label)
            if topic is None:
                topic = Topic(kind="memory", label="記憶からの話題")

        if self._planner_enabled and not topic.intent:
            # 記憶・小噺など単発の話題は「導入」の意図で記録する
            topic.intent = "introduce"
        self._seq += 1
        topic.topic_id = f"t{int(time.time())}-{self._seq}"
        self.last_topic = topic
        if topic.query:
            self._history.append(topic.query)
        return topic

    def _plan_memory(self) -> Topic | None:
        """記憶から具体的な1件を話題化する（配信エピソード等を自律トークに繋ぐ）

        直近発話のテーマと重なる記憶は最大5回まで引き直す（Phase 4.5 で
        導入し、Phase 4 で 3→5 に拡大）。記憶プール自体が同じテーマに
        偏っていると、種の30分再利用禁止だけでは別の同テーマエピソードが
        次々に選ばれて張り付くため。
        引き直しても全て重なる場合は最後の候補を採用する（話題を枯らさない）。
        ただし ban 中の語を含む候補だけは例外で絶対に採用せず、引き直し
        回数に数えたうえで、全滅したら材料なしとして None を返す（Phase 4）。
        """
        if self._memory_provider is None:
            return None
        banned = self.banned_words()
        item = None
        item_banned = False
        for _ in range(5):
            try:
                item = self._memory_provider()
            except Exception:
                return None
            if not item:
                return None
            text = (item.get("text") or "").strip()
            if len(text) < 10:
                return None
            item_banned = any(w in text for w in banned)
            if item_banned:
                logger.info("memory topic redrawn (banned word)",
                            text=text[:40])
                continue
            if self._tenure_enabled \
                    and self._matches_tenure_exclusion(text):
                # tenure で移行済みの主題に触れる記憶は話題化しない
                item_banned = True
                logger.info("memory topic redrawn (tenure exclusion)",
                            text=text[:40])
                continue
            if self._theme_overlap(text) < _THEME_OVERLAP_MEMORY_MAX:
                break
            logger.info("memory topic redrawn (theme overlap)",
                        overlap=round(self._theme_overlap(text), 2),
                        text=text[:40])
        if item is None or item_banned:
            return None
        text = (item.get("text") or "").strip()
        doc = WebDocument(title="覚えていること", snippet=text[:300],
                          source="memory", site="思い出")
        return Topic(kind="memory", label=f"思い出: {text[:24]}",
                     documents=[doc])

    def _plan_filler(self) -> Topic:
        """定番小噺から1つ選ぶ。直近に使った小噺は避ける（台帳で判定）"""
        recent_labels = [e.get("label") or ""
                         for e in list(self._ledger)[-4:]]
        pool = [(t, s) for t, s in _FILLER_TOPICS
                if not any(t in lbl for lbl in recent_labels)]
        if not pool:
            pool = list(_FILLER_TOPICS)
        title, snippet = random.choice(pool)
        doc = WebDocument(title=title, snippet=snippet,
                          source="filler", site="定番の小噺")
        return Topic(kind="filler", label=f"小噺: {title}", documents=[doc])

    def _plan_news(self) -> Topic | None:
        doc = self._pick_news_doc()
        if doc is None:
            return None
        return Topic(
            kind="news",
            label=f"ニュース: {doc.title[:30]}",
            query=doc.title,
            documents=[doc],
            arc_stage="introduce",
            intent=self._fresh_intent(),
            arc_key=self._arc_key_for(doc.title, doc.title),
        )

    def _plan_news_filtered(self, query: str = "") -> Topic | None:
        """クエリ（無ければ配信テーマ）に合うニュース記事だけから選ぶ。

        force_topic(kind=news) の絞り込み用（2026-11-22 themefix）。
        タイトルのトークンとクエリのトークンの重なりで判定し、最も
        重なりの大きい記事を採る。合う記事が無ければ None。
        """
        needle = (query or (self._theme_provider() or "")).strip()
        if not needle or self._news is None:
            return None
        base = _title_tokens(needle)
        if not base:
            return None
        now = time.time()
        self._expire_reservations(now)
        pool = [d for d in self._news.unused_items()
                if d.key not in self._reserved]
        best: WebDocument | None = None
        best_score = 0.0
        for d in pool:
            toks = _title_tokens(d.title)
            if not toks:
                continue
            shared = len(toks & base)
            if shared < 2:
                continue
            score = shared / min(len(toks), len(base))
            if score > best_score:
                best, best_score = d, score
        if best is None:
            return None
        self._reserved[best.key] = now
        return Topic(
            kind="news",
            label=f"ニュース: {best.title[:30]}",
            query=best.title,
            documents=[best],
            arc_stage="introduce",
            intent=self._fresh_intent(),
            arc_key=self._arc_key_for(best.title, best.title),
        )

    def _theme_query_candidates(self, theme: str) -> list[str]:
        """テーマ検索クエリの候補（テーマ単体→テーマ＋キーワード1つ）"""
        queries = [theme]
        keywords = [k.strip() for k in self._keyword_provider()
                    if k.strip() and self._seed_ok(k)
                    and not self._seed_on_cooldown(k)
                    and k.strip() not in theme]
        if keywords:
            queries.append(f"{theme} {random.choice(keywords)}")
        return queries

    async def _plan_theme(self, query: str = "",
                          allow_web: bool = True) -> Topic | None:
        """配信テーマに沿った話題を作る（2026-11-22 themefix で新設）。

        材料は (1) テーマ単体／テーマ＋キーワード1つの Web 検索、
        (2) テーマ関連の記憶、(3) 材料なしのお題（検索が死んでいても
        テーマ発話自体は成立させるための終端）の順に探す。
        query が指定された場合は、そのクエリをテーマ的お題として扱う。
        """
        theme = (self._theme_provider() or "").strip()
        head = (query or theme).strip()
        if not head:
            return None
        if allow_web and self._search is not None:
            candidates = [head] if query \
                else self._theme_query_candidates(theme)
            for q in candidates:
                docs = await self._search.search(q, limit=self.max_documents)
                if docs:
                    return Topic(
                        kind="theme",
                        label=f"テーマ: {q[:30]}",
                        query=q,
                        documents=docs[: self.max_documents],
                        arc_stage="introduce",
                        intent=self._fresh_intent(),
                        arc_key=self._arc_key_for("", q),
                    )
                logger.info("theme search returned nothing", query=q)
                self._search_failures.append(time.time())
        topic = self._plan_theme_memory(theme or head)
        if topic is not None:
            return topic
        # 材料なしのお題。お題ドキュメント（source="theme"）を1件だけ
        # 付け、プロンプト側で「資料」ではなく「お題」として描画させる
        doc = WebDocument(
            title=f"配信テーマ: {head}",
            snippet=("自分の感想や最近の出来事との接点、視聴者への"
                     "問いかけを交えて話す。具体的な事実（価格・日程・"
                     "数値など）は断定しない。"),
            source="theme", site="配信テーマ")
        return Topic(
            kind="theme",
            label=f"テーマ: {head[:30]}",
            query=head,
            documents=[doc],
            arc_stage="introduce",
            intent=self._fresh_intent(),
            arc_key=self._arc_key_for("", head),
        )

    def _plan_theme_memory(self, theme: str) -> Topic | None:
        """記憶からテーマ関連のエピソードを探してテーマ話題にする"""
        if self._memory_provider is None or not theme:
            return None
        base = _title_tokens(theme)
        if not base:
            return None
        banned = self.banned_words()
        for _ in range(5):
            try:
                item = self._memory_provider()
            except Exception:
                return None
            if not item:
                return None
            text = (item.get("text") or "").strip()
            if len(text) < 10:
                continue
            if any(w in text for w in banned):
                continue
            toks = _title_tokens(text)
            if len(toks & base) < 2:
                continue
            doc = WebDocument(title="覚えていること", snippet=text[:300],
                              source="memory", site="思い出")
            return Topic(
                kind="theme",
                label=f"テーマ×思い出: {text[:24]}",
                query=theme,
                documents=[doc],
                arc_stage="introduce",
                intent=self._fresh_intent(),
                arc_key=self._arc_key_for("", theme),
            )
        return None

    # curiosity 系の検索1回あたりの時間上限（秒）。プロバイダ巡回
    # （各8秒）＋最小間隔＋クールダウン待ちで、空振りでも数十秒の
    # サイクル時間を消費した（2026-11-21 BFラン: 182秒間隔の一因）。
    # 上限を超えたら検索なしとして次の話題へ進む
    _CURIOSITY_SEARCH_TIMEOUT_SEC = 30.0

    async def _search_with_cap(self, q: str) -> list | None:
        """時間上限付きの検索。タイムアウト時は None（空振りは []）"""
        try:
            return await asyncio.wait_for(
                self._search.search(q, limit=self.max_documents),
                timeout=self._CURIOSITY_SEARCH_TIMEOUT_SEC)
        except asyncio.TimeoutError:
            logger.info("curiosity search timed out", query=q,
                        timeout_sec=self._CURIOSITY_SEARCH_TIMEOUT_SEC)
            return None

    async def _plan_curiosity(self, query: str = "") -> Topic | None:
        if self._search is None:
            return None
        q = (query or self._build_query()).strip()
        if not q:
            return None
        if not query and q in self._history:
            # 直近で使ったクエリなら1回だけ引き直す
            q = (self._build_query() or q).strip()

        docs = await self._search_with_cap(q)
        if docs is None:
            self._search_failures.append(time.time())
            return None
        if not docs:
            logger.info("curiosity search returned nothing", query=q)
            self._search_failures.append(time.time())
            return None
        return Topic(
            kind="curiosity",
            label=f"検索: {q[:30]}",
            query=q,
            documents=docs[: self.max_documents],
            arc_stage="introduce",
            intent=self._fresh_intent(),
            arc_key=self._arc_key_for("", q),
        )

    def status(self) -> dict:
        chain = self._subject_chain(time.time()) \
            if self._tenure_enabled else None
        return {
            "weights": self.weights,
            "recent_queries": list(self._history),
            "last_topic": self.last_topic.to_dict() if self.last_topic else None,
            "utilities": {k: round(v, 2) for k, v in self._last_utilities.items()},
            "energy": round(self.energy, 2),
            "reserved": len(self._reserved),
            "thread": ({"kind": self._thread.get("kind"),
                        "title": (self._thread.get("title")
                                  or self._thread.get("query") or "")[:40],
                        "continuations": self._thread.get("continuations")}
                       if self._thread else None),
            "comment_hook": bool(self._comment_hook),
            "pending_answers": [
                {"question": (e.get("question") or "")[:30],
                 "state": e.get("state"),
                 "attempts": e.get("attempts")}
                for e in self._pending_answers],
            "stack": [{"kind": e.get("kind"),
                        "title": (e.get("title") or e.get("query") or "")[:40],
                        "suspended_at": e.get("suspended_at")} for e in self._stack],
            "state_path": str(self._state_path) if self._state_path else None,
            "ledger": list(self._ledger)[-10:],
            "kind_counts": {
                k: sum(1 for e in self._ledger if e.get("kind") == k)
                for k in ("memory", "news", "curiosity", "filler", "theme")
            },
            "stream_theme": {
                "theme": (self._theme_provider() or "").strip(),
                "share_target": self._theme_share_target,
                "theme_share": round(self.theme_share(), 2),
                "speeches": self._theme_stat_speeches,
                "mentions": self._theme_stat_mentions,
                "mention_rate": round(
                    self._theme_stat_mentions / self._theme_stat_speeches, 3)
                if self._theme_stat_speeches else 0.0,
                "forced_pending": len(self._forced_topics),
            },
            "memory_share": round(self.memory_share(), 2),
            "memory_cap": {"ratio": self._memory_cap_ratio,
                           "window": self._memory_cap_window},
            "filler_enabled": self._filler_enabled,
            "planner_enabled": self._planner_enabled,
            "arc": ({"kind": self._thread.get("kind"),
                     "stage": self._thread.get("stage"),
                     "speeches": self._thread.get("speeches"),
                     "closing": bool(self._thread.get("closing"))}
                    if self._thread else None),
            "arc_stats": dict(self._arc_stats),
            "tenure": {
                "enabled": self._tenure_enabled,
                "speeches": self._tenure_speeches,
                "sec": self._tenure_sec,
                "cooldown_sec": self._tenure_cooldown_sec,
                "excluded": [
                    {"arc_key": k[:40],
                     "title": (v.get("title") or "")[:40],
                     "remaining_sec": round(
                         float(v.get("until") or 0) - time.time(), 1)}
                    for k, v in self._tenure_excluded.items()
                    if float(v.get("until") or 0) > time.time()],
                "chain": ({"arc_key": chain["arc_key"][:40],
                           "speeches": chain["speeches"],
                           "elapsed_sec": round(chain["elapsed"], 1)}
                          if chain else None),
            },
        }
