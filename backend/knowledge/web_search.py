"""Web検索クライアント

APIキーがあれば Google Custom Search / Brave を使い、無ければキー不要の
フォールバック（DuckDuckGo HTML / Google News RSS）で検索する。
どのプロバイダでも WebDocument のリストを返すので、呼び出し側は違いを意識しない。

取得した文章は「外部の他人が書いたテキスト」なので、必ず sanitize を通してから返す。
"""
from __future__ import annotations

import asyncio
import os
import random
import re
import time
from urllib.parse import quote_plus, unquote

import httpx

from backend.knowledge.models import WebDocument
from backend.knowledge.sanitize import clean_text, domain_of, is_blocked, is_safe_url
from backend.logger import get_logger

logger = get_logger(__name__)

_UA = "Mozilla/5.0 (compatible; AIVtuberBot/1.0; +local)"
# DuckDuckGoのHTML版はボット名のUAだと結果を返さない（202のチャレンジページになる）ため、
# フォールバック検索では通常のブラウザUAを使う
_BROWSER_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
               "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")

GOOGLE_CSE_ENDPOINT = "https://www.googleapis.com/customsearch/v1"
BRAVE_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"
DDG_ENDPOINT = "https://html.duckduckgo.com/html/"
GOOGLE_NEWS_SEARCH = "https://news.google.com/rss/search"

# DuckDuckGo HTML の結果ブロック（タイトルリンク + スニペット）
_DDG_RESULT_RE = re.compile(
    r'<a[^>]+class="result__a"[^>]+href="(?P<url>[^"]+)"[^>]*>(?P<title>.*?)</a>'
    r'(?:.*?class="result__snippet"[^>]*>(?P<snippet>.*?)</a>)?',
    re.S | re.I,
)
_DDG_REDIRECT_RE = re.compile(r"[?&]uddg=([^&]+)")


class SearchError(RuntimeError):
    """検索に失敗した（全プロバイダがダメだった）"""


class WebSearchClient:
    """
    Web検索の入り口。

    provider:
        auto        … google_cse → brave → duckduckgo → google_news の順に試す
        google_cse  … Google Custom Search JSON API（APIキー + 検索エンジンID）
        brave       … Brave Search API（APIキー）
        duckduckgo  … キー不要（HTMLスクレイピング。ベストエフォート）
        google_news … キー不要（ニュースRSS検索。時事ネタに強い）
    """

    def __init__(
        self,
        provider: str = "auto",
        api_key_env: str = "GOOGLE_SEARCH_API_KEY",
        cse_id_env: str = "GOOGLE_CSE_ID",
        brave_api_key_env: str = "BRAVE_SEARCH_API_KEY",
        max_results: int = 5,
        timeout_sec: float = 8.0,
        cache_ttl_sec: int = 900,
        blocked_domains: list[str] | None = None,
        language: str = "ja",
        region: str = "JP",
        min_request_interval_sec: float = 0.0,
        cooldown_base_sec: float = 60.0,
        cooldown_max_sec: float = 900.0,
    ):
        self.provider = provider or "auto"
        self._api_key_env = api_key_env
        self._cse_id_env = cse_id_env
        self._brave_key_env = brave_api_key_env
        self.max_results = max_results
        self.timeout_sec = timeout_sec
        self.cache_ttl_sec = cache_ttl_sec
        self.blocked_domains = blocked_domains or []
        self.language = language
        self.region = region
        # レート制限対策（Phase 3）。既定では最小間隔は無効（0）で、
        # 本番は config（knowledge.search）から有効値が渡される
        self.min_request_interval_sec = max(0.0, float(min_request_interval_sec))
        self.cooldown_base_sec = max(1.0, float(cooldown_base_sec))
        self.cooldown_max_sec = max(self.cooldown_base_sec,
                                    float(cooldown_max_sec))
        # クエリ → (取得時刻, 結果)
        self._cache: dict[str, tuple[float, list[WebDocument]]] = {}
        self._lock = asyncio.Lock()
        self.last_provider = ""
        self.last_error = ""
        # プロバイダ別の 429 連続失敗数とクールダウン期限、前回リクエスト時刻
        self._provider_fails: dict[str, int] = {}
        self._provider_cooldown_until: dict[str, float] = {}
        self._last_request_at: float = 0.0

    # ── 設定 ──────────────────────────────────────────────────

    @property
    def google_api_key(self) -> str:
        return os.getenv(self._api_key_env, "").strip()

    @property
    def google_cse_id(self) -> str:
        return os.getenv(self._cse_id_env, "").strip()

    @property
    def brave_api_key(self) -> str:
        return os.getenv(self._brave_key_env, "").strip()

    def available_providers(self) -> list[str]:
        """今の環境で実際に使えるプロバイダ一覧（優先順）"""
        if self.provider != "auto":
            return [self.provider]
        out = []
        if self.google_api_key and self.google_cse_id:
            out.append("google_cse")
        if self.brave_api_key:
            out.append("brave")
        out.extend(["duckduckgo", "google_news"])
        return out

    def status(self) -> dict:
        now = time.time()
        return {
            "provider": self.provider,
            "available": self.available_providers(),
            "google_cse_configured": bool(self.google_api_key and self.google_cse_id),
            "brave_configured": bool(self.brave_api_key),
            "last_provider": self.last_provider,
            "last_error": self.last_error,
            "cached_queries": len(self._cache),
            "provider_cooldowns": {
                name: round(until - now, 1)
                for name, until in self._provider_cooldown_until.items()
                if until > now},
            "min_request_interval_sec": self.min_request_interval_sec,
        }

    # ── レート制限対策（Phase 3）──────────────────────────────

    def _cooldown_remaining(self, name: str) -> float:
        """プロバイダのクールダウン残り秒数（冷却中でなければ0）"""
        return max(0.0, self._provider_cooldown_until.get(name, 0.0)
                   - time.time())

    def _in_cooldown(self, name: str) -> bool:
        return self._cooldown_remaining(name) > 0

    @staticmethod
    def _retry_after_sec(exc: Exception) -> float:
        """429 応答の Retry-After ヘッダ（秒）。読めなければ0"""
        resp = getattr(exc, "response", None)
        headers = getattr(resp, "headers", None) if resp is not None else None
        if not headers:
            return 0.0
        raw = headers.get("Retry-After") or headers.get("retry-after")
        if not raw:
            return 0.0
        try:
            return max(0.0, float(raw))
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _is_rate_limited(exc: Exception) -> bool:
        """例外がレート制限（429）かどうか"""
        resp = getattr(exc, "response", None)
        if resp is not None and getattr(resp, "status_code", None) == 429:
            return True
        if getattr(exc, "status_code", None) == 429:
            return True
        return "429" in str(exc)

    def _note_provider_failure(self, name: str, exc: Exception) -> None:
        """プロバイダの失敗を記録する。429 なら指数バックオフで冷却する"""
        if not self._is_rate_limited(exc):
            return
        fails = self._provider_fails.get(name, 0) + 1
        self._provider_fails[name] = fails
        backoff = min(self.cooldown_max_sec,
                      self.cooldown_base_sec * (2 ** (fails - 1)))
        backoff = max(backoff, self._retry_after_sec(exc))
        backoff += random.uniform(0.0, backoff * 0.1)   # ジッタ
        self._provider_cooldown_until[name] = time.time() + backoff
        logger.warning("web search provider cooldown",
                       provider=name, fails=fails,
                       cooldown_sec=round(backoff, 1))

    def _note_provider_success(self, name: str) -> None:
        """成功したら連続失敗とクールダウンをリセットする"""
        self._provider_fails.pop(name, None)
        self._provider_cooldown_until.pop(name, None)

    async def _respect_min_interval(self) -> None:
        """検索リクエスト全体の最小間隔を守る（直列化ロック内で呼ぶ）"""
        if self.min_request_interval_sec <= 0:
            self._last_request_at = time.time()
            return
        wait = self.min_request_interval_sec - (time.time()
                                                - self._last_request_at)
        if wait > 0:
            await asyncio.sleep(wait)
        self._last_request_at = time.time()

    # ── 検索 ──────────────────────────────────────────────────

    async def search(self, query: str, limit: int | None = None,
                     use_cache: bool = True) -> list[WebDocument]:
        """
        クエリで検索して結果を返す。全プロバイダが失敗したら空リスト。
        （配信を止めないため、例外は投げずに空を返す方針）
        """
        query = (query or "").strip()
        if not query:
            return []
        limit = limit or self.max_results

        cache_key = f"{query}::{limit}"
        if use_cache:
            hit = self._cache.get(cache_key)
            if hit and (time.time() - hit[0]) < self.cache_ttl_sec:
                self.last_provider = "cache"
                return hit[1][:limit]

        async with self._lock:
            # ロック待ちの間に別タスクが同じクエリを埋めているかもしれない
            if use_cache:
                hit = self._cache.get(cache_key)
                if hit and (time.time() - hit[0]) < self.cache_ttl_sec:
                    return hit[1][:limit]

            docs: list[WebDocument] = []
            errors: list[str] = []
            tried: list[str] = []
            for name in self.available_providers():
                if self._in_cooldown(name):
                    # 冷却中のプロバイダにはリクエスト自体を送らない
                    errors.append(f"{name}: cooldown "
                                  f"{self._cooldown_remaining(name):.0f}s")
                    logger.info("web search provider skipped (cooldown)",
                                provider=name,
                                remaining=round(self._cooldown_remaining(name), 1))
                    continue
                await self._respect_min_interval()
                tried.append(name)
                try:
                    docs = await self._search_with(name, query, limit)
                except Exception as e:  # プロバイダ単位で握りつぶして次へ
                    errors.append(f"{name}: {e}")
                    logger.warning("web search provider failed",
                                   provider=name, query=query[:40], error=str(e))
                    self._note_provider_failure(name, e)
                    continue
                if docs:
                    self._note_provider_success(name)
                    self.last_provider = name
                    break
                # 例外なし・結果0件（DDG の空振り等）も観測できるように残す
                logger.debug("web search provider empty",
                             provider=name, query=query[:40])

            docs = self._postprocess(docs, limit)
            if docs:
                self.last_error = ""
                self._cache[cache_key] = (time.time(), docs)
                self._prune_cache()
            else:
                self.last_error = "; ".join(errors) or "no results"
            logger.debug("web search done", query=query[:40],
                         tried=tried, results=len(docs))
            return docs

    async def _search_with(self, name: str, query: str, limit: int) -> list[WebDocument]:
        if name == "google_cse":
            return await self._search_google_cse(query, limit)
        if name == "brave":
            return await self._search_brave(query, limit)
        if name == "duckduckgo":
            return await self._search_duckduckgo(query, limit)
        if name == "google_news":
            return await self._search_google_news(query, limit)
        raise SearchError(f"unknown provider: {name}")

    def _postprocess(self, docs: list[WebDocument], limit: int) -> list[WebDocument]:
        """URLの正当性チェック・ブロックドメイン除去・重複排除"""
        seen: set[str] = set()
        out: list[WebDocument] = []
        for d in docs:
            if d.url and not is_safe_url(d.url):
                continue
            if is_blocked(d.url, self.blocked_domains):
                continue
            if not d.title and not d.snippet:
                continue
            if d.key in seen:
                continue
            seen.add(d.key)
            out.append(d)
            if len(out) >= limit:
                break
        return out

    def _prune_cache(self, max_entries: int = 64) -> None:
        if len(self._cache) <= max_entries:
            return
        oldest = sorted(self._cache, key=lambda k: self._cache[k][0])
        for key in oldest[: len(self._cache) - max_entries]:
            self._cache.pop(key, None)

    # ── 各プロバイダ ──────────────────────────────────────────

    async def _search_google_cse(self, query: str, limit: int) -> list[WebDocument]:
        key, cx = self.google_api_key, self.google_cse_id
        if not key or not cx:
            raise SearchError("GOOGLE_SEARCH_API_KEY / GOOGLE_CSE_ID が未設定です")
        params = {
            "key": key,
            "cx": cx,
            "q": query,
            "num": min(max(limit, 1), 10),
            "hl": self.language,
            "gl": self.region.lower(),
            "safe": "active",
        }
        async with httpx.AsyncClient(timeout=self.timeout_sec) as client:
            resp = await client.get(GOOGLE_CSE_ENDPOINT, params=params)
            if resp.status_code == 429:
                raise SearchError("Google Custom Search のクォータ超過")
            resp.raise_for_status()
            data = resp.json()

        docs = []
        for item in data.get("items", []):
            url = item.get("link", "")
            docs.append(WebDocument(
                title=clean_text(item.get("title", ""), 120),
                url=url,
                snippet=clean_text(item.get("snippet", ""), 300),
                source="google_cse",
                site=item.get("displayLink", "") or domain_of(url),
            ))
        return docs

    async def _search_brave(self, query: str, limit: int) -> list[WebDocument]:
        key = self.brave_api_key
        if not key:
            raise SearchError("BRAVE_SEARCH_API_KEY が未設定です")
        headers = {"X-Subscription-Token": key, "Accept": "application/json"}
        params = {"q": query, "count": min(max(limit, 1), 20),
                  "country": self.region, "search_lang": self.language}
        async with httpx.AsyncClient(timeout=self.timeout_sec, headers=headers) as client:
            resp = await client.get(BRAVE_ENDPOINT, params=params)
            resp.raise_for_status()
            data = resp.json()

        docs = []
        for item in (data.get("web", {}) or {}).get("results", []):
            url = item.get("url", "")
            docs.append(WebDocument(
                title=clean_text(item.get("title", ""), 120),
                url=url,
                snippet=clean_text(item.get("description", ""), 300),
                source="brave",
                site=domain_of(url),
            ))
        return docs

    async def _search_duckduckgo(self, query: str, limit: int) -> list[WebDocument]:
        """APIキー不要のフォールバック。HTMLの構造変更で壊れうるのでベストエフォート。"""
        headers = {"User-Agent": _BROWSER_UA,
                   "Accept-Language": f"{self.language},en;q=0.5"}
        async with httpx.AsyncClient(timeout=self.timeout_sec, headers=headers,
                                     follow_redirects=True) as client:
            resp = await client.post(DDG_ENDPOINT, data={"q": query, "kl": "jp-jp"})
            resp.raise_for_status()
            html_text = resp.text

        docs = []
        for m in _DDG_RESULT_RE.finditer(html_text):
            url = self._unwrap_ddg_url(m.group("url"))
            docs.append(WebDocument(
                title=clean_text(m.group("title") or "", 120),
                url=url,
                snippet=clean_text(m.group("snippet") or "", 300),
                source="duckduckgo",
                site=domain_of(url),
            ))
            if len(docs) >= limit * 2:   # ブロック除去ぶんを見込んで多めに拾う
                break
        return docs

    @staticmethod
    def _unwrap_ddg_url(raw: str) -> str:
        """DuckDuckGoのリダイレクトURL（//duckduckgo.com/l/?uddg=...）を実URLに戻す"""
        m = _DDG_REDIRECT_RE.search(raw or "")
        if m:
            return unquote(m.group(1))
        if raw.startswith("//"):
            return "https:" + raw
        return raw

    async def _search_google_news(self, query: str, limit: int) -> list[WebDocument]:
        """キー不要の最終フォールバック。ニュース記事に限られるが時事ネタには十分。"""
        from backend.knowledge.news import parse_rss   # 循環importを避けて遅延import
        url = (f"{GOOGLE_NEWS_SEARCH}?q={quote_plus(query)}"
               f"&hl={self.language}&gl={self.region}&ceid={self.region}:{self.language}")
        # DDG と同じ教訓：ボット名義のUAは不利なのでブラウザUAを使う（Phase 3）
        headers = {"User-Agent": _BROWSER_UA}
        async with httpx.AsyncClient(timeout=self.timeout_sec, headers=headers,
                                     follow_redirects=True) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            xml_text = resp.text
        return parse_rss(xml_text, source="google_news")[:limit]
