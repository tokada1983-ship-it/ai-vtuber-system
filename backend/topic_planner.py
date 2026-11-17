"""自律トークの話題を決めるプランナー

コメントが無いときに何を話すかを決める。話題の出どころは3つ:

  memory    … 長期記憶のキーワード（従来の独り言。ネット接続不要）
  news      … RSSで仕入れた最新ニュース
  curiosity … 記憶のキーワードや配信テーマをWeb検索して仕入れた話

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

# 予約の有効期限（秒）。発話に至らなかった話題はこの時間で解放される
_RESERVATION_TTL_SEC = 600.0
# 直近の同種トピックへのクールダウン（秒）
_KIND_COOLDOWN_SEC = 150.0


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
    kind: str                                   # "memory" | "news" | "curiosity"
    label: str = ""                             # ダッシュボード表示用の短い説明
    query: str = ""                             # 検索に使ったクエリ
    documents: list[WebDocument] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    topic_id: str = ""                          # プランナーが発行する一意ID
    resumed: bool = False                       # 中断した話題の続報なら True

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
        weights: Optional[dict[str, float]] = None,
        max_documents: int = 3,
        history_size: int = 12,
        state_path: str | None = None,
    ):
        self._news = news
        self._search = search
        self._keyword_provider = keyword_provider or (lambda: [])
        self._theme_provider = theme_provider or (lambda: "")
        self._memory_provider = memory_provider
        self.weights = {"memory": 3.0, "news": 2.0, "curiosity": 1.0}
        if weights:
            self.weights.update({k: float(v) for k, v in weights.items()})
        self.max_documents = max_documents
        self._history: deque[str] = deque(maxlen=history_size)
        self.last_topic: Topic | None = None

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
        # 会話の継続用：直近の話題スレッドと、拾い直し待ちのコメント
        self._thread: dict | None = None            # {kind,title,query,at,continuations}
        self._banned: dict[str, float] = {}         # 吸着した語 → 禁止期限（文脈リセット時）
        self._comment_hook: dict | None = None      # {author,text,at}
        self._last_hook_woven_at: float = 0.0
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

    def note_speech(self, idle: bool = False) -> None:
        """発話イベントの通知（Orchestrator._after_speech から呼ばれる）"""
        now = time.time()
        self._decay_energy(now)
        if idle:
            self._energy = min(1.0, self._energy + 0.04)
        else:
            self._energy = min(1.0, self._energy + 0.15)
            self.last_comment_speech_at = now
            self._suspend_current(now)

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
        self._stack.append({**last, "suspended_at": now})
        del self._stack[:-3]
        self._save_state()
        logger.info("topic suspended", kind=last.get("kind"),
                    title=(last.get("title") or last.get("query") or "")[:40])

    # ── 状態の永続化 ────────────────────────────────────────

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
            th = data.get("thread")
            if isinstance(th, dict) and time.time() - float(th.get("at") or 0) < 600:
                self._thread = th
            logger.info("planner state loaded", ledger=len(self._ledger))
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

    def ban_word(self, word: str, seconds: float = 1800) -> None:
        """吸着した語をしばらく話題の種にしない（文脈リセットとセットで呼ぶ）"""
        if word:
            self._banned[word] = time.time() + seconds

    def banned_words(self) -> set[str]:
        now = time.time()
        return {w for w, until in self._banned.items() if until > now}

    def reset_thread(self) -> None:
        """継続中スレッドを破棄する（文脈リセット用）"""
        self._thread = None
        self._save_state()

    def commit(self, topic: Topic) -> None:
        """話題が実際に発話されたことを記録する（消費の確定）"""
        now = time.time()
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
        })
        # 会話の流れ管理：新しいWeb話題はスレッドを開き、それ以外は閉じる。
        # 続報・続き（resumed）の commit ではスレッドを維持する
        if topic.kind in ("news", "curiosity") and not topic.resumed:
            self._thread = {
                "kind": topic.kind,
                "title": topic.documents[0].title if topic.documents else "",
                "query": topic.query,
                "at": now,
                "continuations": 0,
            }
        elif topic.kind not in ("news", "curiosity"):
            self._thread = None
        self._save_state()

    def release(self, topic: Topic) -> None:
        """発話に至らなかった話題の予約を解放する（ネタは消えない）"""
        if topic.kind == "news" and topic.documents:
            self._reserved.pop(topic.documents[0].key, None)

    # ── 種類のUtilityスコア ─────────────────────────────────

    def _score_kinds(self, allow_web: bool, context: dict | None) -> dict[str, float]:
        now = time.time()
        ctx = context or {}
        silence = float(ctx.get("silence_sec") or 0.0)
        energy = self._decay_energy(now)
        w = self.weights
        scores: dict[str, float] = {}

        # memory：常時利用可能な床。ニュースが強いときは少し譲る
        scores["memory"] = max(0.0, w.get("memory", 0.0))

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
        scores["curiosity"] = max(0.0, u_cur)

        if scores["news"] > scores["memory"] > 0:
            scores["memory"] *= 0.8
        return scores

    def _pick_kind(self, allow_web: bool, context: dict | None = None,
                   exclude: set[str] | None = None) -> str:
        scores = self._score_kinds(allow_web, context)
        for k in (exclude or set()):
            scores[k] = 0.0
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
                     query=text, documents=[doc])

    # ── 話題の継続（スレッド） ──────────────────────────────

    async def _try_continue(self, allow_web: bool) -> Topic | None:
        """直近の話題の続きを話し、会話に流れを作る（最大2回まで・確率60%）"""
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
            self._thread = None
            return None
        if int(th.get("continuations") or 0) >= 2:
            return None
        if random.random() > 0.6:
            return None
        kind = th.get("kind")
        if kind == "news":
            doc = self._pick_followup_doc(th)
            if doc is None:
                return None
            th["continuations"] = int(th.get("continuations") or 0) + 1
            th["at"] = now
            logger.info("topic continue", kind="news", title=doc.title[:40])
            return Topic(kind="news", label=f"続き: {doc.title[:30]}",
                         query=doc.title, documents=[doc], resumed=True)
        if kind == "curiosity":
            seed = self._seed_from_query(th.get("query") or "")
            if not seed or self._search is None:
                return None
            q = random.choice(_CURIOSITY_TEMPLATES).format(kw=seed)
            if q == th.get("query"):
                return None
            docs = await self._search.search(q, limit=self.max_documents)
            if not docs:
                return None
            th["continuations"] = int(th.get("continuations") or 0) + 1
            th["at"] = now
            logger.info("topic continue", kind="curiosity", query=q[:40])
            return Topic(kind="curiosity", label=f"続き: {q[:30]}",
                         query=q, documents=docs[: self.max_documents],
                         resumed=True)
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
        if age > 240 and random.random() < 0.5:
            return None   # 時間が経っていたら半分は見送り（スタックは残す）
        kind = top.get("kind")
        if kind == "news":
            doc = self._pick_followup_doc(top)
            if doc is None:
                self._stack.pop()
                return None
            self._stack.pop()
            logger.info("topic resume", kind="news", title=doc.title[:40])
            return Topic(kind="news", label=f"続報: {doc.title[:30]}",
                         query=doc.title, documents=[doc], resumed=True)
        if kind == "curiosity":
            q = (top.get("query") or "").strip()
            if not q or self._search is None:
                self._stack.pop()
                return None
            docs = await self._search.search(q, limit=self.max_documents)
            if not docs:
                self._stack.pop()
                return None
            self._stack.pop()
            logger.info("topic resume", kind="curiosity", query=q[:40])
            return Topic(kind="curiosity", label=f"続報: {q[:30]}",
                         query=q, documents=docs[: self.max_documents],
                         resumed=True)
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
        recent_tokens = [_title_tokens(t) for t in self._recent_titles(8)]
        scored: list[tuple[WebDocument, float]] = []
        for d in pool:
            age = d.age_hours(now)
            fresh = math.exp(-age / 6.0) if age is not None else 0.45
            toks = _title_tokens(d.title)
            novelty = 1.0 - max((_jaccard(toks, rt) for rt in recent_tokens),
                                default=0.0)
            site_pen = 0.55 if d.site and d.site == self._last_news_site else 1.0
            score = (0.25 + fresh) * (0.5 + 0.5 * novelty) * site_pen \
                + random.random() * 0.05
            scored.append((d, score))
        doc = random.choices([d for d, _ in scored],
                             weights=[s for _, s in scored], k=1)[0]
        self._reserved[doc.key] = now
        return doc

    # ── 好奇心クエリ ────────────────────────────────────────

    def _build_query(self) -> str:
        """好奇心検索のクエリを作る。配信テーマ優先、無ければ記憶のキーワード。

        台帳にある直近のクエリと同じ種は避ける（最大3回引き直す）。
        """
        theme = (self._theme_provider() or "").strip()
        banned = self.banned_words()
        if banned and any(w in theme for w in banned):
            theme = ""
        keywords = [k for k in self._keyword_provider()
                    if k.strip() and not any(w in k for w in banned)]
        ledger_queries = {e.get("query", "") for e in self._ledger}
        best = ""
        for _ in range(3):
            seed = ""
            if theme and keywords and random.random() < 0.5:
                seed = f"{theme} {random.choice(keywords)}"
            elif theme:
                seed = theme
            elif keywords:
                seed = random.choice(keywords)
            if not seed:
                return ""
            q = random.choice(_CURIOSITY_TEMPLATES).format(kw=seed)
            best = q
            if q not in ledger_queries and q not in self._history:
                return q
        return best

    # ── メイン ────────────────────────────────────────────────

    async def plan(self, kind: str = "", query: str = "",
                   allow_web: bool = True,
                   context: dict | None = None) -> Topic:
        """
        次の話題を決めて材料を集める。

        kind を指定すればその種類を強制する（ダッシュボードの手動実行用）。
        query を指定した場合は curiosity（そのクエリで検索）になる。
        材料が集まらなければ memory にフォールバックする。
        消費はここでは確定しない。発話できたら commit()、できなければ
        release() を呼ぶこと（ニュース記事は予約だけされる）。
        """
        if query:
            kind = "curiosity"
        if not kind and not query:
            auto: Topic | None = self._try_comment_hook()
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
        kind = kind or self._pick_kind(allow_web, context)

        topic: Topic | None = None
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

        if topic is None:
            topic = self._plan_memory() or Topic(kind="memory",
                                                 label="記憶からの話題")

        self._seq += 1
        topic.topic_id = f"t{int(time.time())}-{self._seq}"
        self.last_topic = topic
        if topic.query:
            self._history.append(topic.query)
        return topic

    def _plan_memory(self) -> Topic | None:
        """記憶から具体的な1件を話題化する（配信エピソード等を自律トークに繋ぐ）"""
        if self._memory_provider is None:
            return None
        try:
            item = self._memory_provider()
        except Exception:
            return None
        if not item:
            return None
        text = (item.get("text") or "").strip()
        if len(text) < 10:
            return None
        doc = WebDocument(title="覚えていること", snippet=text[:300],
                          source="memory", site="思い出")
        return Topic(kind="memory", label=f"思い出: {text[:24]}",
                     documents=[doc])

    def _plan_news(self) -> Topic | None:
        doc = self._pick_news_doc()
        if doc is None:
            return None
        return Topic(
            kind="news",
            label=f"ニュース: {doc.title[:30]}",
            query=doc.title,
            documents=[doc],
        )

    async def _plan_curiosity(self, query: str = "") -> Topic | None:
        if self._search is None:
            return None
        q = (query or self._build_query()).strip()
        if not q:
            return None
        if not query and q in self._history:
            # 直近で使ったクエリなら1回だけ引き直す
            q = (self._build_query() or q).strip()

        docs = await self._search.search(q, limit=self.max_documents)
        if not docs:
            logger.info("curiosity search returned nothing", query=q)
            self._search_failures.append(time.time())
            return None
        return Topic(
            kind="curiosity",
            label=f"検索: {q[:30]}",
            query=q,
            documents=docs[: self.max_documents],
        )

    def status(self) -> dict:
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
            "stack": [{"kind": e.get("kind"),
                        "title": (e.get("title") or e.get("query") or "")[:40],
                        "suspended_at": e.get("suspended_at")} for e in self._stack],
            "state_path": str(self._state_path) if self._state_path else None,
            "ledger": list(self._ledger)[-10:],
            "kind_counts": {
                k: sum(1 for e in self._ledger if e.get("kind") == k)
                for k in ("memory", "news", "curiosity")
            },
        }
