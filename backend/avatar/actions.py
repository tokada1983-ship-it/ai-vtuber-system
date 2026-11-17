"""LLM応答に埋め込まれたアバター（Unity）アクションタグの解析と送信

例: 「椅子に座るね [action: sit, target: chair]」
    → 発話テキスト「椅子に座るね」と [{"action": "sit", "target": "chair"}]
"""
from __future__ import annotations

import re

_PATTERN_EN = re.compile(r"\[action:\s*([^,\]]+)(?:,\s*target:\s*([^\]]+))?\]", re.IGNORECASE)
_PATTERN_JA = re.compile(r"\[アクション:\s*([^,\]]+)(?:,\s*(?:対象|ターゲット|対象物):\s*([^\]]+))?\]")


def extract_actions(text: str) -> tuple[str, list[dict]]:
    """タグを取り除いたテキストと、アクションのリストを返す"""
    if not text:
        return "", []
    actions: list[dict] = []
    for pattern in (_PATTERN_EN, _PATTERN_JA):
        for m in pattern.finditer(text):
            actions.append({
                "action": m.group(1).strip(),
                "target": m.group(2).strip() if m.group(2) else None,
            })
    clean = _PATTERN_JA.sub("", _PATTERN_EN.sub("", text))
    return re.sub(r"\s+", " ", clean).strip(), actions


async def dispatch_actions(hub, actions: list[dict]) -> None:
    """抽出したアクションをWebSocket経由でアバターへ送る"""
    for act in actions:
        name, target = act["action"], act["target"]
        # 座標指定の移動（"x,y,z"）だけは専用メッセージにする
        if name in ("walk_to", "move_to") and target:
            try:
                coords = [float(x) for x in target.split(",")]
            except ValueError:
                coords = []
            if len(coords) == 3:
                await hub.send_unity_move(*coords)
                continue
        await hub.send_unity_action(name, target)