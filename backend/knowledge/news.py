"""ニュース取得（RSS/Atom）

APIキー無しで最新ニュースを仕入れるための最小実装。
標準ライブラリの xml.etree でRSS 2.0 / Atom の両方を読む（feedparser不要）。

デフォルトのフィードは Google ニュース（日本版）のトップ。
config.yaml の knowledge.news.feeds で好きなフィードに差し替えられる。
"""
from __future__ import annotations

import asyncio
import random
import time
from email.utils import parsedate_to_datetime
from xml.etree import ElementTree as ET

import httpx

from backend.knowledge.models import WebDocument
from backend.knowledge.sanitize import clean_text, domain_of, is_blocked, is_safe_url
from backend.logger import get_logger

logger = get_logger(__name__)

_UA = "Mozilla/5.0 (compatible; AIVtuberBot/1.0; +local)"
# フィード取得用のブラウザUA（web_search 側の教訓：ボットUAは不利。Phase 3）。
# 定数はこのファイルに置く（web_search は news.parse_rss を遅延 import する
# 関係で、逆向きの import を作らないため）
BROWSER_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")

DEFAULT_FEEDS = [
    "https://news.google.com/rss?hl=ja&gl=JP&ceid=JP:ja",
]

_ATOM_NS = "{http://www.w3.org/2005/Atom}"


def _parse_date(text: str | None) -> float | None:
    """RSSのpubDate / AtomのupdatedをUNIX時刻へ。読めなければNone。"""
    if not text:
        return None
    text = text.strip()
    try:
        return parsedate_to_datetime(text).timestamp()
    except (TypeError, ValueError, IndexError):
        pass
    # Atom: 2026-09-01T12:34:56Z
    try:
        from datetime import datetime
        return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def parse_rss(xml_text: str, source: str = "rss") -> list[WebDocument]:
    """RSS 2.0 / Atom のXMLを WebDocument のリストにする。壊れていたら空リスト。"""
    if not xml_text or not xml_text.strip():
        return []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        logger.warning("RSS parse error", error=str(e))
        return []

    docs: list[WebDocument] = []

    # RSS 2.0
    for item in root.iter("item"):
        title = clean_text(item.findtext("title") or "", 160)
        link = (item.findtext("link") or "").strip()
        desc = clean_text(item.findtext("description") or "", 300)
        pub = _parse_date(item.findtext("pubDate"))
        site = ""
        src_el = item.find("source")
        if src_el is not None and (src_el.text or "").strip():
            site = clean_text(src_el.text, 60)
        docs.append(WebDocument(
            title=title, url=link, snippet=desc, source=source,
            site=site or domain_of(link), published_at=pub,
        ))

    # Atom
    if not docs:
        for entry in root.iter(f"{_ATOM_NS}entry"):
            title_el = entry.find(f"{_ATOM_NS}title")
            link_el = entry.find(f"{_ATOM_NS}link")
            summary_el = entry.find(f"{_ATOM_NS}summary")
            if summary_el is None:
                summary_el = entry.find(f"{_ATOM_NS}content")
            updated_el = entry.find(f"{_ATOM_NS}updated")
            link = (link_el.get("href", "") if link_el is not None else "").strip()
            docs.append(WebDocument(
                title=clean_text(title_el.text if title_el is not None else "", 160),
                url=link,
                snippet=clean_text(summary_el.text if summary_el is not None else "", 300),
                source=source,
                site=domain_of(link),
                published_at=_parse_date(updated_el.text if updated_el is not None else None),
            ))

    return [d for d in docs if d.title]


class NewsFeed:
    """
    複数のRSSフィードをまとめて取得し、新しい順のニュース一覧を保持する。

    - 取得は refresh_interval_sec ごと（配信中は Orchestrator のループが呼ぶ）
    - 一度話題にした記事は mark_used() で記録し、同じネタを繰り返さない
    """

    def __init__(
        self,
        feeds: list[str] | None = None,
        max_items: int = 30,
        max_age_hours: float = 24.0,
        timeout_sec: float = 8.0,
        blocked_domains: list[str] | None = None,
        keywords: list[str] | None = None,
        retry_base_sec: float = 120.0,
    ):
        self.feeds = [f for f in (feeds or DEFAULT_FEEDS) if f.strip()]
        self.max_items = max_items
        self.max_age_hours = max_age_hours
        self.timeout_sec = timeout_sec
        self.blocked_domains = blocked_domains or []
        self.keywords = [k for k in (keywords or []) if k.strip()]
        self.retry_base_sec = max(10.0, float(retry_base_sec))

        self._items: list[WebDocument] = []
        self._used_keys: set[str] = set()
        self._last_fetch: float = 0.0      # 最終試行時刻
        self._last_success: float = 0.0    # 最終成功時刻（Phase 3 で分離）
        self._fail_streak: int = 0         # 連続失敗回数（バックオフ用）
        self._next_retry_at: float = 0.0   # 失敗中の次回試行時刻
        self.last_error: str = ""
        self._lock = asyncio.Lock()

    # ── 状態 ──────────────────────────────────────────────────

    @property
    def items(self) -> list[WebDocument]:
        return list(self._items)

    @property
    def last_fetch(self) -> float:
        return self._last_fetch

    @property
    def fail_streak(self) -> int:
        return self._fail_streak

    def status(self) -> dict:
        now = time.time()
        return {
            "feeds": self.feeds,
            "count": len(self._items),
            "used": len(self._used_keys),
            "last_fetch": self._last_fetch,
            "last_success": self._last_success,
            "fail_streak": self._fail_streak,
            "next_retry_in_sec": (round(self._next_retry_at - now, 1)
                                  if self._fail_streak else 0.0),
            "last_error": self.last_error,
        }

    def is_stale(self, interval_sec: float) -> bool:
        now = time.time()
        if self._fail_streak > 0:
            # 失敗中は成功周期（interval_sec）ではなくバックオフ間隔で粘る。
            # 現行の「失敗すると丸々 interval_sec 沈黙する」挙動への対処（Phase 3）
            return now >= self._next_retry_at
        return (now - self._last_fetch) >= interval_sec

    # ── 取得 ──────────────────────────────────────────────────

    async def refresh(self) -> list[WebDocument]:
        """全フィードを取得して内部リストを更新する。失敗しても例外は投げない。"""
        async with self._lock:
            results = await asyncio.gather(
                *[self._fetch_feed(url) for url in self.feeds],
                return_exceptions=True,
            )
            docs: list[WebDocument] = []
            errors: list[str] = []
            succeeded = False
            for url, res in zip(self.feeds, results):
                if isinstance(res, Exception):
                    errors.append(f"{domain_of(url) or url}: {res}")
                    continue
                succeeded = True
                docs.extend(res)

            merged = self._merge(docs)
            now = time.time()
            self._last_fetch = now
            if succeeded:
                self._last_success = now
                self._fail_streak = 0
                self._next_retry_at = 0.0
            else:
                # 全フィード失敗：指数バックオフで次回試行を決める（Phase 3）。
                # 上限は通常の取得周期（本番900秒）を超えない
                self._fail_streak += 1
                backoff = min(900.0,
                              self.retry_base_sec * (2 ** (self._fail_streak - 1)))
                backoff += random.uniform(0.0, backoff * 0.1)   # ジッタ
                self._next_retry_at = now + backoff
                logger.warning("news refresh failed, backing off",
                               streak=self._fail_streak,
                               retry_in_sec=round(backoff, 1))
            self.last_error = "; ".join(errors) if errors and not merged else ""
            if merged:
                self._items = merged
            logger.info("news refreshed", count=len(self._items), errors=len(errors))
            return self.items

    async def _fetch_feed(self, url: str) -> list[WebDocument]:
        headers = {"User-Agent": BROWSER_UA}
        async with httpx.AsyncClient(timeout=self.timeout_sec, headers=headers,
                                     follow_redirects=True) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            return parse_rss(resp.text, source="rss")

    def _merge(self, docs: list[WebDocument]) -> list[WebDocument]:
        """新しい順に整列しつつ、重複・古すぎる記事・NGドメインを落とす"""
        now = time.time()
        seen: set[str] = set()
        out: list[WebDocument] = []
        for d in docs:
            if d.url and not is_safe_url(d.url):
                continue
            if is_blocked(d.url, self.blocked_domains):
                continue
            if d.key in seen:
                continue
            age = d.age_hours(now)
            if self.max_age_hours > 0 and age is not None and age > self.max_age_hours:
                continue
            if self.keywords and not any(k in (d.title + d.snippet) for k in self.keywords):
                continue
            seen.add(d.key)
            out.append(d)

        out.sort(key=lambda d: d.published_at or 0.0, reverse=True)
        return out[: self.max_items]

    # ── 話題の取り出し ────────────────────────────────────────

    def pick(self, avoid_used: bool = True) -> WebDocument | None:
        """
        まだ話していないニュースを1本選ぶ。
        新しいものほど選ばれやすいよう、上位のほうに重みを寄せる。
        """
        pool = [d for d in self._items if not (avoid_used and d.key in self._used_keys)]
        if not pool:
            return None
        # 先頭（＝新しい記事）ほど重い線形の重み
        weights = [len(pool) - i for i in range(len(pool))]
        return random.choices(pool, weights=weights, k=1)[0]

    def is_used(self, doc: WebDocument) -> bool:
        return doc.key in self._used_keys

    def unused_items(self) -> list[WebDocument]:
        """まだ話題にしていない記事の一覧（新しい順）"""
        return [d for d in self._items if d.key not in self._used_keys]

    def mark_used(self, doc: WebDocument) -> None:
        self._used_keys.add(doc.key)
        # 使用済みキーが増えすぎたら、古いぶんは忘れて再度話題にできるようにする
        if len(self._used_keys) > self.max_items * 4:
            self._used_keys = set(list(self._used_keys)[-self.max_items * 2:])

    def reset_used(self) -> None:
        self._used_keys.clear()
