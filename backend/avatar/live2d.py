"""Live2D（Cubism）モデルの探索とメタ情報の読み取り

Live2Dのモデルは「1ファイル」ではなくフォルダ一式（.model3.json + テクスチャ +
motions/ + expressions/）で配布される。ここでは models/ 以下を走査して
ダッシュボードに出せる一覧を作り、モデル定義から使えるモーション名・表情名を取り出す。

Cubism 4（*.model3.json）を主対象とし、古いCubism 2（*.model.json）も読める範囲で扱う。
"""
from __future__ import annotations

import json
from pathlib import Path

# 走査から除外するフォルダ（テクスチャの中間生成物やSDK同梱ファイルが多い）
_SKIP_DIRS = {".git", "__pycache__", "node_modules"}


def is_model_file(path: str | Path) -> bool:
    name = Path(path).name.lower()
    return name.endswith(".model3.json") or name.endswith(".model.json")


def cubism_version(path: str | Path) -> int:
    """3 or 4系なら4、旧Cubism 2系なら2を返す"""
    return 2 if Path(path).name.lower().endswith(".model.json") else 4


def to_url(model_path: str | Path) -> str:
    """
    config.yaml のパス（models/live2d/x/x.model3.json）をブラウザ用URLに直す。

    Windowsの \ 区切りと先頭スラッシュの有無を吸収する。http(s) はそのまま返す。
    """
    text = str(model_path or "").strip()
    if not text:
        return ""
    if text.startswith(("http://", "https://")):
        return text
    normalized = text.replace("\\", "/")
    return normalized if normalized.startswith("/") else "/" + normalized


def scan_models(root: str | Path = "models") -> list[dict]:
    """
    models/ 以下からLive2Dモデル定義を探して一覧を返す。

    戻り値: [{"name": 表示名, "path": "models/live2d/x/x.model3.json",
              "url": "/models/...", "version": 4}, ...]
    """
    root = Path(root)
    if not root.is_dir():
        return []

    found: list[dict] = []
    for path in sorted(root.rglob("*.json")):
        if not is_model_file(path):
            continue
        if any(part in _SKIP_DIRS for part in path.parts):
            continue
        rel = path.as_posix()
        found.append({
            "name": path.parent.name or path.stem,
            "path": rel,
            "url": to_url(rel),
            "version": cubism_version(path),
        })
    return found


def describe_model(model_path: str | Path) -> dict:
    """
    モデル定義から、使えるモーショングループと表情名を取り出す。

    ダッシュボードで「このモデルにはどんなモーションがあるか」を見せて、
    motion_map（AIのモーション名 → モデルのグループ名）を設定するために使う。
    読めない・壊れている場合は空の一覧を返す（画面を落とさない）。
    """
    path = Path(model_path)
    empty = {"motions": {}, "expressions": [], "version": cubism_version(path)}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return empty
    if not isinstance(raw, dict):
        return empty

    if cubism_version(path) == 4:
        refs = raw.get("FileReferences") or {}
        motions_raw = refs.get("Motions") or {}
        expressions_raw = refs.get("Expressions") or []
        motions = {
            group: len(entries) if isinstance(entries, list) else 0
            for group, entries in motions_raw.items()
        } if isinstance(motions_raw, dict) else {}
        expressions = [
            e.get("Name", "") for e in expressions_raw
            if isinstance(e, dict) and e.get("Name")
        ] if isinstance(expressions_raw, list) else []
    else:
        motions_raw = raw.get("motions") or {}
        expressions_raw = raw.get("expressions") or []
        motions = {
            group: len(entries) if isinstance(entries, list) else 0
            for group, entries in motions_raw.items()
        } if isinstance(motions_raw, dict) else {}
        expressions = [
            e.get("name", "") for e in expressions_raw
            if isinstance(e, dict) and e.get("name")
        ] if isinstance(expressions_raw, list) else []

    return {"motions": motions, "expressions": expressions, "version": cubism_version(path)}


def local_cores(root: str | Path = "models") -> dict:
    """
    ローカルに置かれたCubism Coreを探す。

    Coreはライセンス上リポジトリに同梱できないので、既定では公式CDNから読む。
    models/live2d/core/ に置いてあればそちらを使う（オフラインのPCでも動く）。
    """
    core_dir = Path(root) / "live2d" / "core"
    result = {"cubism4": "", "cubism2": ""}
    for key, filename in (("cubism4", "live2dcubismcore.min.js"), ("cubism2", "live2d.min.js")):
        path = core_dir / filename
        if path.is_file():
            result[key] = to_url(path.as_posix())
    return result


def resolve_model_path(configured: str, root: str | Path = "models") -> str:
    """
    設定されたモデルパスを確定する。

    未設定、または存在しないパスなら models/ 以下で最初に見つかったモデルを使う。
    1つも無ければ空文字（アバター画面側で「モデル未設置」と案内する）。
    """
    configured = (configured or "").strip()
    if configured:
        if configured.startswith(("http://", "https://")):
            return configured
        if Path(configured.replace("\\", "/")).is_file():
            return configured.replace("\\", "/")
    found = scan_models(root)
    return found[0]["path"] if found else ""


def is_inside(path: str | Path, root: str | Path) -> bool:
    """path が root 配下かどうか（../ でmodels外を読ませないためのガード）"""
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
    except (ValueError, OSError):
        return False
    return True
