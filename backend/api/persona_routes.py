"""人格QLoRAのHTTP API（ダッシュボードの /persona 画面から使う）"""
from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse

from backend.config import save_config
from backend.llm import ollama_service
from backend.logger import get_logger
from backend.runtime import FRONTEND, config, persona

logger = get_logger(__name__)
router = APIRouter()


def _bad_request(e: Exception) -> HTTPException:
    return HTTPException(status_code=400, detail=str(e))


@router.get("/persona", response_class=HTMLResponse)
async def persona_page():
    return (FRONTEND / "persona.html").read_text(encoding="utf-8")


@router.get("/api/persona")
async def api_persona_status():
    return persona.status()


@router.post("/api/persona/environment")
async def api_persona_environment():
    return await persona.check_environment()


@router.post("/api/persona/settings")
async def api_persona_settings(body: dict):
    try:
        return {"ok": True, "config": persona.update_settings(body)}
    except (TypeError, ValueError) as e:
        raise _bad_request(e)


@router.post("/api/persona/dataset")
async def api_persona_dataset(body: dict):
    """データセット作成は数分かかるので裏で走らせ、進捗はWebSocketで流す"""
    if persona.busy:
        raise HTTPException(status_code=409, detail="別の処理を実行中です")
    synthetic = body.get("synthetic_samples")
    task = asyncio.create_task(persona.build_dataset(synthetic=synthetic))
    task.add_done_callback(lambda t: t.exception() and logger.warning(
        "persona dataset task failed", error=str(t.exception())))
    return {"ok": True, "started": True}


@router.post("/api/persona/train")
async def api_persona_train(body: dict):
    if persona.busy:
        raise HTTPException(status_code=409, detail="別の処理を実行中です")
    try:
        return {"ok": True, **await persona.start_training(body)}
    except (RuntimeError, ValueError) as e:
        raise _bad_request(e)


@router.post("/api/persona/cancel")
async def api_persona_cancel():
    return {"ok": await persona.cancel()}


@router.post("/api/persona/register")
async def api_persona_register(body: dict):
    if persona.busy:
        raise HTTPException(status_code=409, detail="別の処理を実行中です")
    try:
        return {"ok": True, **await persona.register(
            run=body.get("run", ""),
            model_name=body.get("model_name", ""),
            mode=body.get("mode", "gguf"),
            ollama_base=body.get("ollama_base", ""),
            quantize=body.get("quantize", ""),
        )}
    except (RuntimeError, ValueError, FileNotFoundError) as e:
        raise _bad_request(e)


@router.post("/api/persona/activate")
async def api_persona_activate(body: dict):
    try:
        return await persona.activate(
            run=body.get("run", ""),
            runtime_model=body.get("runtime_model", ""),
            backend=body.get("backend", ""),
            prompt_mode=body.get("prompt_mode", ""),
        )
    except ValueError as e:
        raise _bad_request(e)


@router.post("/api/persona/deactivate")
async def api_persona_deactivate():
    return await persona.deactivate()

# ── Ollama（人格モデルの実行環境）の自動起動設定 ───────────────

_OLLAMA_KEYS = ("auto_start", "preload_on_start", "keep_alive", "auto_pull", "startup_timeout_sec")


@router.get("/api/ollama")
async def api_ollama_status():
    be = config.llm.get_backend("ollama")
    status = await ollama_service.get_service(be.base_url if be else "").status()
    return {"settings": config.ollama.model_dump(), **status}


@router.post("/api/ollama/settings")
async def api_ollama_settings(body: dict):
    o = config.ollama
    try:
        for key in _OLLAMA_KEYS:
            if key not in body:
                continue
            current = getattr(o, key)
            value = body[key]
            if isinstance(current, bool):
                value = value if isinstance(value, bool) else str(value).lower() in ("1", "true", "on")
            else:
                value = type(current)(value)
            setattr(o, key, value)
    except (TypeError, ValueError) as e:
        raise _bad_request(e)
    save_config(config)
    ollama_service.configure(o)
    return {"ok": True, "settings": o.model_dump()}
