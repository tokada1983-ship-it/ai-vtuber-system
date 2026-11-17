"""ResponseFilter: LLM応答の配信前チェック（応答後フィルタ）

README §10 のプロンプトインジェクション対策の最終ライン。
プロンプト内の指示だけでは守りきれないため、TTS/配信に乗る直前の
テキストを機械的に検査し、問題があれば発話をスキップさせる。

検出対象:
  - 電話番号・メールアドレス・郵便番号（個人情報の読み上げ防止）
  - APIキーらしき文字列（sk-.../AKIA.../ghp_... や長いhex）
  - システムプロンプト漏洩の兆候（内部タグ・「システムプロンプト」等）
  - 設定可能なNGワード（config.yaml の safety.ng_words）
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from backend.logger import get_logger

logger = get_logger(__name__)

# 個人情報・秘密情報のパターン
_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("電話番号", re.compile(r"0\d{1,4}[-‐－ー]\d{1,4}[-‐－ー]\d{3,4}|0\d{9,10}")),
    ("メールアドレス", re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")),
    ("郵便番号", re.compile(r"〒\s*\d{3}[-‐－ー]?\d{4}")),
    ("APIキー", re.compile(
        r"sk-[A-Za-z0-9]{16,}"          # OpenAI系
        r"|AKIA[0-9A-Z]{16}"            # AWS
        r"|ghp_[A-Za-z0-9]{20,}"        # GitHub
        r"|xox[baprs]-[A-Za-z0-9-]{10,}"  # Slack
        r"|\b[0-9a-fA-F]{32,}\b"        # 長いhex文字列
    )),
    ("内部タグ漏洩", re.compile(r"<\s*/?\s*viewer_comment|</?think>|/no_think")),
    ("プロンプト漏洩", re.compile(r"システムプロンプト|system\s*prompt|出力フォーマット定義", re.IGNORECASE)),
]


@dataclass
class FilterResult:
    ok: bool
    reason: str = ""


class ResponseFilter:
    """発話テキストの配信前チェック"""

    def __init__(self, ng_words: list[str] | None = None, enabled: bool = True):
        self._enabled = enabled
        self._ng_words = [w for w in (ng_words or []) if w.strip()]

    def check(self, text: str) -> FilterResult:
        """問題なければ ok=True。問題があれば ok=False と理由を返す。"""
        if not self._enabled or not text:
            return FilterResult(ok=True)

        for label, pattern in _PATTERNS:
            m = pattern.search(text)
            if m:
                logger.warning("ResponseFilter: blocked",
                               reason=label, matched=m.group(0)[:30])
                return FilterResult(ok=False, reason=label)

        for word in self._ng_words:
            if word in text:
                logger.warning("ResponseFilter: NG word", word=word)
                return FilterResult(ok=False, reason=f"NGワード「{word}」")

        return FilterResult(ok=True)
