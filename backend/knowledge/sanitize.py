"""外部Webテキストの無害化

検索結果やRSSの本文は「他人が書いた文章」であり、
そのままプロンプトに入れるとプロンプトインジェクションの入口になる。
ここでは表示・プロンプト投入の前に必ず通す掃除処理をまとめる。
"""
from __future__ import annotations

import html
import re
from urllib.parse import urlparse

# <script>/<style> はタグごと中身を落とす
_SCRIPT_RE = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.I | re.S)
_TAG_RE = re.compile(r"<[^>]+>")
_CTRL_RE = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f\u200b-\u200f\u2028\u2029]")
_WS_RE = re.compile(r"[ \t\r\n　]+")

# こちらのプロンプト構造を模倣・脱出しようとする文字列を潰す。
# 「```」でコードブロックを閉じたり、<web_result> を閉じて指示を書く手口を防ぐ。
_STRUCTURE_RE = re.compile(
    r"(```+|</?\s*(?:web_result|web_results|viewer_comment|system|assistant|user)\s*>)",
    re.I,
)

# 「以前の指示を無視して」系の定番フレーズは、話題としても不要なので落とす
_INJECTION_PHRASES = [
    "ignore previous instructions",
    "ignore all previous",
    "disregard previous",
    "system prompt",
    "these instructions",
    "以前の指示を無視",
    "これまでの指示を無視",
    "上記の指示を無視",
    "システムプロンプト",
    "命令を無視",
]


def strip_html(text: str) -> str:
    """HTMLタグと実体参照を落として素のテキストにする"""
    if not text:
        return ""
    text = _SCRIPT_RE.sub(" ", text)
    text = _TAG_RE.sub(" ", text)
    text = html.unescape(text)
    return text


def clean_text(text: str, max_chars: int = 300) -> str:
    """
    外部テキストをプロンプト/表示に使える形に整える。
    - HTML除去、制御文字除去、空白圧縮
    - プロンプト構造を模倣する記号の除去
    - 既知のインジェクション定番フレーズの除去
    - 長さ制限
    """
    if not text:
        return ""
    text = strip_html(text)
    text = _CTRL_RE.sub(" ", text)
    text = _STRUCTURE_RE.sub(" ", text)

    lowered = text.lower()
    for phrase in _INJECTION_PHRASES:
        if phrase in lowered:
            # 大文字小文字を無視して消す
            text = re.sub(re.escape(phrase), " ", text, flags=re.I)
            lowered = text.lower()

    text = _WS_RE.sub(" ", text).strip()
    if len(text) > max_chars:
        text = text[:max_chars].rstrip() + "…"
    return text


def domain_of(url: str) -> str:
    """URLからドメイン（www.除去）を取り出す。失敗したら空文字。"""
    if not url:
        return ""
    try:
        host = urlparse(url).netloc.lower()
    except ValueError:
        return ""
    if host.startswith("www."):
        host = host[4:]
    return host.split(":")[0]


def is_safe_url(url: str) -> bool:
    """http(s) の外部URLだけを通す（file: や javascript: を弾く）"""
    if not url:
        return False
    try:
        parsed = urlparse(url)
    except ValueError:
        return False
    return parsed.scheme in ("http", "https") and bool(parsed.netloc)


def is_blocked(url: str, blocked_domains: list[str]) -> bool:
    """ブロックリストに含まれるドメイン（およびそのサブドメイン）か"""
    if not blocked_domains:
        return False
    host = domain_of(url)
    if not host:
        return False
    for blocked in blocked_domains:
        b = blocked.strip().lower().lstrip(".")
        if not b:
            continue
        if host == b or host.endswith("." + b):
            return True
    return False
