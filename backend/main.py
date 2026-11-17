"""FastAPI エントリポイント

組み立ては backend/runtime.py、エンドポイントは backend/api/ 以下にある。
ここはアプリの生成・起動/終了処理・ルータの登録だけを受け持つ。
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from backend.api import cast_routes, http_routes, persona_routes, ws_routes
from backend.llm import ollama_service
from backend.tts import irodori_service
from backend.logger import get_logger
from backend.runtime import AUDIO_CACHE, FRONTEND, MODELS_DIR, config, monitor, orchestrator, persona

logger = get_logger(__name__)


@asynccontextmanager
async def _lifespan(app: FastAPI):
    logger.info("AI Vtuber backend started", port=config.server.port)
    # video_idが明示的に設定されている場合のみ自動スタート（Mockモードは自動起動しない）
    if config.youtube.video_id and config.youtube.video_id != "mock":
        logger.info("Auto-starting with configured video_id", video_id=config.youtube.video_id)
        await orchestrator.start()
    else:
        logger.info("Waiting for manual start from dashboard")
        # 開始ボタンを押す前に、Ollamaの起動とモデルのロードを済ませておく
        await orchestrator.rebuild_llm()
    # TTSサーバ（Irodori-TTS）も同時に立ち上げる
    orchestrator.warmup_tts()
    monitor.start()
    # 睡眠時統合のスケジューラ（Phase 3）はセッションの有無によらず
    # アプリ常駐で回す（夜間枠＋キャッチアップをアイドル時に実行する）
    sleep_task = asyncio.create_task(orchestrator.run_sleep_scheduler(),
                                     name="sleep")
    yield
    sleep_task.cancel()
    await monitor.stop()
    await orchestrator.stop()
    # stop() は後処理を背景化したので、アプリ終了時は完了まで待つ
    # （終了で post-stop の記録が失われないように。Phase 4.5）
    await orchestrator.await_post_stop()
    await persona.shutdown()
    ollama_service.stop_all()
    irodori_service.stop()


app = FastAPI(title="AI Vtuber Backend", lifespan=_lifespan)

AUDIO_CACHE.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
app.mount("/frontend", StaticFiles(directory=str(FRONTEND)), name="frontend")
app.mount("/models", StaticFiles(directory=str(MODELS_DIR)), name="models")
app.mount("/audio", StaticFiles(directory=str(AUDIO_CACHE)), name="audio")

app.include_router(http_routes.router)
app.include_router(persona_routes.router)
app.include_router(cast_routes.router)
# 注意: /ws/voice は /ws/{channel} より先に登録される必要がある（ws_routes内の定義順で担保）
app.include_router(ws_routes.router)