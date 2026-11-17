"""YouTube の配信URLから動画ID（11文字）を取り出す

ダッシュボードの入力欄に URL をまるごと貼っても動くようにするため。
対応する形:
  https://www.youtube.com/watch?v=ID   https://youtu.be/ID
  https://www.youtube.com/live/ID      https://www.youtube.com/shorts/ID
  https://www.youtube.com/embed/ID     https://studio.youtube.com/video/ID/livestreaming
YouTube の URL でないもの（ツイキャス・Twitch の入力など）はそのまま返す。
"""
from __future__ import annotations

import re

_ID = r"[A-Za-z0-9_-]{11}"
_PATTERNS = [
    re.compile(r"[?&]v=(" + _ID + r")(?![A-Za-z0-9_-])"),
    re.compile(r"youtu\.be/(" + _ID + r")(?![A-Za-z0-9_-])"),
    re.compile(r"youtube\.com/(?:live|shorts|embed|v)/(" + _ID + r")(?![A-Za-z0-9_-])"),
    re.compile(r"studio\.youtube\.com/video/(" + _ID + r")(?![A-Za-z0-9_-])"),
]


def extract_youtube_id(text: str) -> str:
    t = (text or "").strip()
    if "youtu" not in t:
        return t
    for pattern in _PATTERNS:
        m = pattern.search(t)
        if m:
            return m.group(1)
    return t