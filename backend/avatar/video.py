"""動画アバター（モーション動画再生）のカタログ読み取り

透過WebMなどのモーション動画を「待機（ループ）」と「アクション（1回再生）」に
分けて再生する描画方式のための、カタログJSONの解決を担う。

カタログは models/ 以下に置く（/models で静的配信される）。
ファイルが存在しないエントリは黙って落とす（画面を落とさないことを優先）。
"""
from __future__ import annotations

import json
from pathlib import Path


def _to_url(base_dir_url: str, entry: str) -> str:
    """カタログ内エントリ（ファイル名 or URL）をブラウザ用URLに直す"""
    text = (entry or "").strip()
    if not text:
        return ""
    if text.startswith(("http://", "https://", "/")):
        return text
    return base_dir_url.rstrip("/") + "/" + text.replace("\\", "/")


def _resolve_entries(base_dir: Path, base_url: str, raw) -> list[str]:
    """文字列 or 文字列リストを、実在するファイルのURLリストにする"""
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list):
        return []
    urls: list[str] = []
    for entry in raw:
        if not isinstance(entry, str) or not entry.strip():
            continue
        if entry.startswith(("http://", "https://", "/")):
            urls.append(entry)
            continue
        if (base_dir / entry).is_file():
            urls.append(_to_url(base_url, entry))
    return urls


def load_catalog(catalog_path: str | Path, models_root: str | Path = "models") -> dict:
    """
    カタログJSONを読み、再生に使えるURLの一覧に解決して返す。

    戻り値: {
        "ok": bool, "character": str,
        "idle": [url, ...],
        "motion_map": {モーション名: [url, ...]},
        "emotion_map": {感情名: [url, ...]},
    }
    読めない・存在しない場合は ok=False で空の一覧を返す。
    """
    empty = {"ok": False, "character": "", "idle": [], "motion_map": {}, "emotion_map": {}}
    path = Path(str(catalog_path or "").replace("\\", "/"))
    if not str(catalog_path or "").strip():
        return empty
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return empty
    if not isinstance(raw, dict):
        return empty

    base_dir = path.parent
    # カタログの親フォルダを /models 基準のURLに直す
    try:
        rel = base_dir.resolve().relative_to(Path(models_root).resolve())
        base_url = "/models/" + rel.as_posix()
    except (ValueError, OSError):
        base_url = "/models/" + base_dir.name

    def resolve_map(raw_map) -> dict[str, list[str]]:
        resolved: dict[str, list[str]] = {}
        if isinstance(raw_map, dict):
            for key, value in raw_map.items():
                urls = _resolve_entries(base_dir, base_url, value)
                if urls:
                    resolved[str(key)] = urls
        return resolved

    return {
        "ok": True,
        "character": str(raw.get("character", "")),
        "idle": _resolve_entries(base_dir, base_url, raw.get("idle")),
        "motion_map": resolve_map(raw.get("motion_map")),
        "emotion_map": resolve_map(raw.get("emotion_map")),
    }


def payload(video_cfg, models_root: str | Path = "models") -> dict:
    """アバター画面（/api/avatar_config）へ渡す video セクション"""
    catalog = load_catalog(getattr(video_cfg, "catalog_path", ""), models_root)
    return {
        "catalog_path": getattr(video_cfg, "catalog_path", ""),
        "catalog_ok": catalog["ok"],
        "idle": catalog["idle"],
        "motion_map": catalog["motion_map"],
        "emotion_map": catalog["emotion_map"],
    }
