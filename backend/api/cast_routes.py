"""出演キャラクターと掛け合いのHTTP API

  GET  /api/characters           出演者の一覧と状態
  POST /api/characters/{key}     キャラの設定を変更（enabled / aliases / voice / llm / avatar）
  POST /api/characters           キャラを追加（name 必須）
  GET  /api/ensemble             掛け合いの設定
  POST /api/ensemble             掛け合いの設定を変更
  POST /api/ensemble/banter      いますぐ掛け合いを始めさせる（話題振り＋相方の返し）

変更は config.yaml に保存し、出演者を組み直す（配信中でも次の発話から反映）。
"""
from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException

from backend.config import (
    AvatarConfig,
    CharacterConfig,
    CharacterLLMConfig,
    CharacterVoiceConfig,
    EnsembleConfig,
    save_config,
)
from backend.logger import get_logger
from backend.runtime import config, orchestrator

logger = get_logger(__name__)
router = APIRouter()


def _find_config(key: str) -> CharacterConfig:
    for ch in config.characters:
        if key in (ch.key, ch.name):
            return ch
    raise HTTPException(status_code=404, detail=f"キャラクターが見つかりません: {key}")


def _character_dict(ch: CharacterConfig) -> dict:
    return {
        "id": ch.key,
        "name": ch.name,
        "enabled": ch.enabled,
        "is_lead": ch is config.character,
        "aliases": list(ch.aliases),
        "prompt_file": str(ch.resolved_prompt_file()),
        "voice": ch.voice.model_dump(),
        "llm": ch.llm.model_dump(),
        "avatar": ch.avatar.model_dump() if ch.avatar else None,
    }


@router.get("/api/characters")
async def api_characters():
    return {
        "characters": [_character_dict(ch) for ch in config.characters],
        "cast": orchestrator.characters_status(),
        "ensemble": config.ensemble.model_dump(),
    }


@router.post("/api/characters")
async def api_add_character(body: dict):
    name = str(body.get("name") or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="name を指定してください")
    if any(name in (c.name, c.key) for c in config.characters):
        raise HTTPException(status_code=400, detail=f"同じ名前のキャラクターがいます: {name}")
    ch = CharacterConfig(
        name=name,
        id=str(body.get("id") or ""),
        prompt_file=str(body.get("prompt_file") or ""),
        data_dir=config.character.data_dir,
        stream_theme=config.character.stream_theme,
        enabled=bool(body.get("enabled", True)),
    )
    config.characters.append(ch)
    save_config(config)
    cast = await orchestrator.reload_cast()
    return {"ok": True, "character": _character_dict(ch), "cast": cast}


@router.post("/api/characters/{key}")
async def api_update_character(key: str, body: dict):
    ch = _find_config(key)
    try:
        if "enabled" in body:
            if ch is config.character and not body["enabled"]:
                raise HTTPException(status_code=400, detail="リーダーは無効にできません")
            ch.enabled = bool(body["enabled"])
        if "aliases" in body:
            ch.aliases = [str(a).strip() for a in body["aliases"] if str(a).strip()]
        if "prompt_file" in body:
            ch.prompt_file = str(body["prompt_file"] or "")
        if "voice" in body:
            ch.voice = CharacterVoiceConfig.model_validate({**ch.voice.model_dump(), **body["voice"]})
        if "llm" in body:
            ch.llm = CharacterLLMConfig.model_validate({**ch.llm.model_dump(), **body["llm"]})
        if "avatar" in body:
            ch.avatar = AvatarConfig.model_validate(body["avatar"]) if body["avatar"] else None
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    save_config(config)
    cast = await orchestrator.reload_cast()
    return {"ok": True, "character": _character_dict(ch), "cast": cast}


@router.get("/api/ensemble")
async def api_get_ensemble():
    return config.ensemble.model_dump()


@router.post("/api/ensemble")
async def api_set_ensemble(body: dict):
    try:
        updated = EnsembleConfig.model_validate({**config.ensemble.model_dump(), **body})
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    # Orchestrator と TurnDirector は同じオブジェクトを見ているので、中身を書き換える
    for field, value in updated.model_dump().items():
        setattr(config.ensemble, field, value)
    save_config(config)
    return {"ok": True, "ensemble": config.ensemble.model_dump()}


@router.post("/api/ensemble/banter")
async def api_force_banter(body: dict | None = None):
    """話題振りから掛け合いを始めさせる（character で振る側を指定できる）"""
    body = body or {}
    if not orchestrator.is_ensemble:
        return {"ok": False, "error": "出演者が1人のため掛け合いできません"}
    if orchestrator.lead.llm is None:
        return {"ok": False, "error": "配信が開始されていません"}
    agent = orchestrator.get_agent(body.get("character", "")) if body.get("character") else None
    asyncio.create_task(orchestrator._speak_idle(
        kind=(body.get("kind") or "").strip(),
        query=(body.get("query") or "").strip(),
        agent=agent,
    ))
    return {"ok": True}
