"""Web検索・ニュース取得で扱う共通データモデル"""
from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class WebDocument:
    """
    Web検索の1件、またはニュース記事1本。

    外部から取ってきた「信用できないテキスト」であることを忘れないため、
    プロンプトへ入れるときは必ず as_prompt_line() を通す（タグで囲んで無害化する）。
    """
    title: str
    url: str = ""
    snippet: str = ""
    source: str = ""          # 取得元（"google_cse" / "google_news" など）
    site: str = ""            # ドメインまたは媒体名
    published_at: float | None = None   # UNIX時刻（分かる場合のみ）
    fetched_at: float = field(default_factory=time.time)

    @property
    def key(self) -> str:
        """重複判定用のキー。URLが無い場合はタイトルで代用する。"""
        return (self.url or self.title).strip().lower()

    def age_hours(self, now: float | None = None) -> float | None:
        if self.published_at is None:
            return None
        return max(0.0, ((now or time.time()) - self.published_at) / 3600.0)

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "source": self.source,
            "site": self.site,
            "published_at": self.published_at,
            "fetched_at": self.fetched_at,
        }

    def as_prompt_line(self, max_snippet: int = 160) -> str:
        """プロンプトに入れる1行表現（本文は切り詰める）"""
        snippet = self.snippet[:max_snippet].strip()
        parts = [f"- {self.title.strip()}"]
        if self.site:
            parts.append(f"（{self.site}）")
        if snippet:
            parts.append(f": {snippet}")
        return "".join(parts)
