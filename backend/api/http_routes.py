"""HTTPエンドポイント（ページ配信・設定・記憶・知識・ローカルチャット）"""
from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from backend.avatar import live2d as live2d_util
from backend.avatar import video as video_util
from backend.config import save_config
from backend.logger import get_logger
from backend.orchestrator import PreviousLoopsStuckError
from backend.memory.character_preset import AvatarPreset as AvatarPresetModel
from backend.memory.character_preset import CharacterPreset
from backend.runtime import (
    AUDIO_CACHE,
    FRONTEND,
    MODELS_DIR,
    _avatar_calibration,
    _preset_manager,
    config,
    hub,
    monitor,
    orchestrator,
)
from backend.stream_state import (
    clear_stream_state,
    mark_stream_stopping,
    write_stream_state,
)
from backend.tts.audio_router import AudioRouter

logger = get_logger(__name__)
router = APIRouter()


# ── HTTP エンドポイント ────────────────────────────────────────

@router.get("/", response_class=HTMLResponse)
async def root():
    return (FRONTEND / "index.html").read_text(encoding="utf-8")

@router.get("/avatar", response_class=HTMLResponse)
async def avatar_page(char: str = ""):
    """
    OBSに貼るアバター画面。アバターの種類（vrm / live2d）に応じて描画方式を切り替える。
    掛け合い配信では /avatar?char=<キャラのid> をキャラの数だけ貼る（省略時はリーダー）。
    """
    agent = orchestrator.get_agent(char) or orchestrator.lead
    page = {
        "live2d": "live2d.html",
        "video": "video.html",
    }.get(agent.avatar.type, "avatar.html")
    return (FRONTEND / page).read_text(encoding="utf-8")

@router.get("/video", response_class=HTMLResponse)
async def video_page():
    """動画アバター画面（type設定に関係なく直接開きたいとき用）"""
    return (FRONTEND / "video.html").read_text(encoding="utf-8")

@router.get("/live2d", response_class=HTMLResponse)
async def live2d_page():
    """Live2Dアバター画面（type設定に関係なく直接開きたいとき用）"""
    return (FRONTEND / "live2d.html").read_text(encoding="utf-8")

@router.get("/vrm", response_class=HTMLResponse)
async def vrm_page():
    """VRMアバター画面（type設定に関係なく直接開きたいとき用）"""
    return (FRONTEND / "avatar.html").read_text(encoding="utf-8")

@router.get("/subtitle", response_class=HTMLResponse)
async def subtitle_page():
    return (FRONTEND / "subtitle.html").read_text(encoding="utf-8")

@router.get("/chat", response_class=HTMLResponse)
async def chat_page():
    """ローカルテストモード: ローカルユーザーと1対1でチャットする画面"""
    return (FRONTEND / "chat.html").read_text(encoding="utf-8")

@router.get("/speechlog", response_class=HTMLResponse)
async def speechlog_page():
    """配信画面右端の発言ログ欄（OBSブラウザソース用、?demo=1 で表示確認）"""
    return (FRONTEND / "speech_log.html").read_text(encoding="utf-8")

@router.get("/api/status")
async def api_status():
    return {
        "video_id": orchestrator._status.video_id,
        "llm_backend": orchestrator._status.llm_backend,
        "tts_backend": orchestrator._status.tts_backend,
        "uptime_sec": orchestrator._status.uptime_sec,
        # アバター画面のハートビート状態（F10: 観測口が無く死角だった）
        "avatar_health": hub.avatar_health(),
    }

@router.get("/api/devices")
async def api_devices():
    return {"devices": AudioRouter.list_devices()}

@router.get("/api/diagnose")
async def api_diagnose():
    """ローカルAIサービスの検出結果と設定ミスの一覧を返す"""
    from backend.discovery import diagnose
    result = await diagnose(orchestrator._cfg)
    result["stt"] = orchestrator.stt_status
    result["tts_health"] = orchestrator.tts_health
    result["avatar_health"] = hub.avatar_health()
    # 疎通はするが合成が返らないハング状態は、実行時の観測でしか分からない
    health = result["tts_health"]
    if health["consecutive_timeouts"] > 0:
        result["issues"].append({
            "severity": "error",
            "target": "tts.runtime",
            "message": (f"TTSに接続はできますが合成が{health['consecutive_timeouts']}回連続で"
                        f"タイムアウトしています"),
            "hint": ("字幕のみで進行中です。TTSサーバを再起動してください"
                     if health["subtitle_only"] else "TTSサーバの状態を確認してください"),
        })
        result["ok"] = False
    return result

@router.get("/api/avatar_calibration")
async def api_get_calibration():
    return _avatar_calibration

@router.post("/api/avatar_calibration")
async def api_set_calibration(body: dict):
    _avatar_calibration.update({k: float(v) for k, v in body.items() if k in ("armDownZ", "foreArmZ", "spineX")})
    await hub.broadcast("avatar", {"type": "calibration_update", "calibration": _avatar_calibration})
    return {"ok": True, "calibration": _avatar_calibration}

def _avatar_payload(char: str = "") -> dict:
    """アバター画面（VRM/Live2D共通）が必要とする設定一式（char 省略時はリーダー）"""
    agent = orchestrator.get_agent(char) or orchestrator.lead
    cfg = agent.avatar
    l2d = cfg.live2d
    return {
        # この画面が担当するキャラ。WebSocketのメッセージはこの id で振り分ける
        "character": agent.key,
        "character_name": agent.name,
        "is_lead": agent.is_lead,
        "cast": [a.key for a in orchestrator.agents],
        "type": cfg.type,
        # VRM画面は従来どおり model_path / idle_motion をトップレベルで見る
        "model_path": cfg.vrm.model_path,
        "idle_motion": cfg.vrm.idle_motion,
        "live2d": {
            "model_path": live2d_util.resolve_model_path(l2d.model_path, MODELS_DIR),
            "scale": l2d.scale,
            "position_x": l2d.position_x,
            "position_y": l2d.position_y,
            "idle_motion_group": l2d.idle_motion_group,
            "lipsync_gain": l2d.lipsync_gain,
            "mouth_param": l2d.mouth_param,
            "expression_map": dict(l2d.expression_map),
            "motion_map": dict(l2d.motion_map),
            # ローカルにCubism Coreがあればそのパス（無ければ画面側がCDNを使う）
            "core": live2d_util.local_cores(MODELS_DIR),
        },
        # 動画アバター: カタログを解決した再生リスト（待機・モーション・感情）
        "video": video_util.payload(cfg.video, MODELS_DIR),
    }


@router.get("/api/avatar_config")
async def api_avatar_config(char: str = ""):
    """アバター画面が起動時に読み込む設定を返す（?char= でキャラを指定）"""
    return _avatar_payload(char)

@router.get("/api/live2d_models")
async def api_live2d_models():
    """models/ 以下に置かれたLive2Dモデル（*.model3.json）の一覧"""
    return {"models": live2d_util.scan_models(MODELS_DIR)}

@router.get("/api/live2d_model_info")
async def api_live2d_model_info(path: str = ""):
    """指定モデルが持つモーショングループ名・表情名を返す（対応表の設定用）"""
    target = live2d_util.resolve_model_path(path, MODELS_DIR)
    if not target:
        return {"ok": False, "error": "Live2Dモデルが見つかりません",
                "motions": {}, "expressions": []}
    if not live2d_util.is_inside(target, MODELS_DIR):
        return {"ok": False, "error": "models/ フォルダの外は読み込めません",
                "motions": {}, "expressions": []}
    info = live2d_util.describe_model(target)
    return {"ok": True, "path": target, **info}

# /api/start の直列化ロック。開始処理（数秒〜数十秒かかる）の途中に
# 2回目の start が割り込むと、開始途中の状態のまま再起動経路へ入って
# しまうため、稼働チェックと開始を不可分にする。
_start_lock = asyncio.Lock()


@router.post("/api/start")
async def api_start(body: dict):
    video_id = body.get("video_id", "")
    async with _start_lock:
        # 冪等化：このプロセスで配信セッションが既に開始済みの場合は、
        # 再起動せず現状を返す（2026-11-21 の二重投入で、再起動時の
        # 残タスクが新セッションを無言で塞ぐ事象が発生したため）。
        # ウォッチドッグの復旧は新プロセスへの /api/start で行われ、
        # その時点ではセッション未開始なので通常どおり開始できる。
        if orchestrator.is_running:
            current_video_id = orchestrator._status.video_id
            if video_id and video_id != current_video_id:
                # 開始済みセッションと違う video_id が来ても、黙って
                # 差し替えず現行セッションを維持する（video_id の
                # 変更は /api/video_id の役割）
                logger.warning(
                    "/api/start ignored: session already running "
                    "with a different video_id",
                    requested_video_id=video_id,
                    current_video_id=current_video_id)
            else:
                logger.info(
                    "/api/start ignored: session already running",
                    video_id=current_video_id)
            return {"ok": True, "already_running": True,
                    "video_id": current_video_id}
        try:
            await orchestrator.start(video_id)
        except PreviousLoopsStuckError as e:
            # 前セッションのループが止め切れていない。開始成功を
            # 装わず、呼び出し元へ失敗を返す（復旧はプロセス再起動）
            logger.error("/api/start failed: previous loops stuck",
                         error=str(e))
            raise HTTPException(status_code=409, detail=str(e))
        # 配信中であることを外部ウォッチドッグへ伝える状態ファイルを作成
        write_stream_state(video_id)
        return {"ok": True, "video_id": video_id}

# /api/stop の応答上限（秒）。停止処理は段階別タイムアウト付きだが、
# それでも終わらない場合は応答だけ先に返し、処理は裏で継続させる。
# 2026-11-19 にこのエンドポイントが120秒応答しなかった事象への対策。
API_STOP_TIMEOUT_SEC = 25.0


@router.post("/api/stop")
async def api_stop():
    # 停止の意図を最初に状態ファイルへ記録する。停止処理が長引いたり
    # 途中でアプリが落とされても、ウォッチドッグが「意図した停止」を
    # 落下と誤認して自動復旧しないようにするため（stop-fix）。
    mark_stream_stopping()
    stop_task = asyncio.create_task(orchestrator.stop())

    def _log_stop_failure(task: asyncio.Task) -> None:
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            logger.error("Orchestrator stop failed", error=str(exc))

    stop_task.add_done_callback(_log_stop_failure)
    in_background = False
    try:
        # shield: タイムアウトしても停止処理そのものは裏で継続する
        await asyncio.wait_for(asyncio.shield(stop_task),
                               timeout=API_STOP_TIMEOUT_SEC)
    except asyncio.TimeoutError:
        in_background = True
        logger.warning(
            "/api/stop: stop still in progress; responding anyway",
            timeout_sec=API_STOP_TIMEOUT_SEC)
    finally:
        # 停止の成否によらず、監視を止めるために状態ファイルを消す
        clear_stream_state()
    return {"ok": True, "stop_in_background": in_background}

@router.post("/api/effects/trigger")
async def api_effects_trigger(body: dict):
    """演出エフェクト（カットイン・効果音・フリップ）の手動発火。

    動作確認とダッシュボードからの手動演出用。自発エフェクトと同じ
    検証・配送経路を通る（クールダウンだけ無視する）。
    body: {"type": "cutin"|"se"|"flip", "name": ..., "text": ...,
           "character": <キャラid, 任意>}
    """
    return await orchestrator.trigger_effect(
        str(body.get("type", "")), name=str(body.get("name", "")),
        text=str(body.get("text", "")), character=str(body.get("character", "")))


@router.get("/api/effects/catalog")
async def api_effects_catalog(character: str = ""):
    """使えるエフェクトのカタログ（?character=<id> でキャラ別に解決）"""
    return orchestrator.effects_catalog(character)


@router.post("/api/video_id")
async def api_set_video_id(body: dict):
    video_id = body.get("video_id", "")
    await orchestrator.set_video_id(video_id)
    return {"ok": True}

# ── 設定API ────────────────────────────────────────────────────

@router.get("/api/config")
async def api_get_config():
    cfg = orchestrator._cfg
    return {
        "llm": {
            "primary": cfg.llm.primary,
            "fallback": cfg.llm.fallback,
            "hybrid": {
                "enabled": cfg.llm.hybrid.enabled,
                "idle_backend": cfg.llm.hybrid.idle_backend,
                "fast_backend": cfg.llm.hybrid.fast_backend,
            },
            "backends": {
                name: {
                    "base_url": be.base_url,
                    "model": be.model,
                    "temperature": be.temperature,
                    "max_tokens": be.max_tokens,
                }
                for name, be in cfg.llm.backends.items()
            },
        },
        "tts": {
            "primary": cfg.tts.primary,
            "backends": {
                name: {
                    "base_url": be.base_url,
                    "speaker_id": be.speaker_id,
                    "speed": be.speed,
                }
                for name, be in cfg.tts.backends.items()
            },
        },
        "youtube": {
            "primary_source": cfg.youtube.primary_source,
            "video_id": cfg.youtube.video_id,
        },
        "character": {"name": cfg.character.name},
        "characters": [{"id": c.key, "name": c.name, "enabled": c.enabled}
                       for c in cfg.characters],
        "orchestrator": {
            "idle_threshold_sec": cfg.orchestrator.idle_threshold_sec,
            "response_history_turns": cfg.orchestrator.response_history_turns,
            "comment_dedupe_window_sec": cfg.orchestrator.comment_dedupe_window_sec,
        },
        "audio_output": {"device": cfg.audio_output.device},
    }

@router.post("/api/config/llm")
async def api_set_llm(body: dict):
    if "primary" in body:
        orchestrator._cfg.llm.primary = body["primary"]
    if "fallback" in body:
        orchestrator._cfg.llm.fallback = body["fallback"]
    hybrid = body.get("hybrid")
    if isinstance(hybrid, dict):
        h = orchestrator._cfg.llm.hybrid
        if "enabled" in hybrid:
            h.enabled = bool(hybrid["enabled"])
        if "idle_backend" in hybrid:
            h.idle_backend = str(hybrid["idle_backend"] or "")
        if "fast_backend" in hybrid:
            h.fast_backend = str(hybrid["fast_backend"] or "")
    await orchestrator.rebuild_llm()
    await hub.send_log("info", f"LLM切替: {orchestrator._cfg.llm.primary}")
    return {"ok": True, "primary": orchestrator._cfg.llm.primary}

@router.post("/api/config/llm/backend")
async def api_set_llm_backend(body: dict):
    name = body.get("name")
    if not name or name not in orchestrator._cfg.llm.backends:
        return {"ok": False, "error": "backend not found"}
    be = orchestrator._cfg.llm.backends[name]
    if "base_url" in body: be.base_url = body["base_url"]
    if "model" in body: be.model = body["model"]
    if "temperature" in body: be.temperature = float(body["temperature"])
    if "max_tokens" in body: be.max_tokens = int(body["max_tokens"])
    await orchestrator.rebuild_llm()
    await hub.send_log("info", f"LLMパラメータ更新: {name}")
    return {"ok": True}

@router.post("/api/config/tts")
async def api_set_tts(body: dict):
    if "primary" in body:
        orchestrator._cfg.tts.primary = body["primary"]
    name = orchestrator._cfg.tts.primary
    if name in orchestrator._cfg.tts.backends:
        be = orchestrator._cfg.tts.backends[name]
        if "speaker_id" in body: be.speaker_id = int(body["speaker_id"])
        if "speed" in body: be.speed = float(body["speed"])
        if "base_url" in body: be.base_url = body["base_url"]
    orchestrator.rebuild_tts()
    await hub.send_log("info", f"TTS切替: {orchestrator._cfg.tts.primary}")
    return {"ok": True}

@router.post("/api/config/source")
async def api_set_source(body: dict):
    if "primary_source" in body:
        orchestrator._cfg.youtube.primary_source = body["primary_source"]
    await hub.send_log("info", f"コメントソース変更: {body.get('primary_source')}（次回start時から反映）")
    return {"ok": True, "note": "次回start時から反映されます"}

@router.post("/api/config/orchestrator")
async def api_set_orchestrator(body: dict):
    cfg = orchestrator._cfg.orchestrator
    if "idle_threshold_sec" in body:
        cfg.idle_threshold_sec = int(body["idle_threshold_sec"])
    if "response_history_turns" in body:
        cfg.response_history_turns = int(body["response_history_turns"])
        for agent in orchestrator.agents:
            agent.log.set_max_turns(cfg.response_history_turns)
    if "comment_dedupe_window_sec" in body:
        cfg.comment_dedupe_window_sec = int(body["comment_dedupe_window_sec"])
    await hub.send_log("info", "Orchestratorパラメータ更新")
    return {"ok": True}


def _memory_of(char: str = ""):
    agent = orchestrator.get_agent(char)
    return agent.long_term_memory if agent else None


@router.get("/api/memory")
async def api_get_memory(char: str = ""):
    """長期記憶（コア記憶）の内容を返す（?char= でキャラ指定。省略時はリーダー）"""
    mem = _memory_of(char)
    if mem is None:
        return {"ok": False, "error": f"キャラクターが見つかりません: {char}"}
    return {
        "character": mem.character_name,
        "content": mem.get_raw(),
        "path": str(mem.path),
        "is_loaded": mem.is_loaded,
    }


@router.post("/api/memory")
async def api_save_memory(body: dict):
    """長期記憶の内容を上書き保存する（body.character でキャラ指定）"""
    mem = _memory_of(body.get("character", ""))
    if mem is None:
        return {"ok": False, "error": "キャラクターが見つかりません"}
    content = body.get("content", "")
    mem.save(content)
    mem.reload()
    await hub.send_log("info", f"{mem.character_name}の長期記憶を更新しました ({len(content)}文字)")
    return {"ok": True, "chars": len(content)}


@router.post("/api/memory/reload")
async def api_reload_memory(body: dict | None = None):
    """長期記憶ファイルを再読み込みする"""
    mem = _memory_of((body or {}).get("character", ""))
    if mem is None:
        return {"ok": False, "error": "キャラクターが見つかりません"}
    mem.reload()
    return {"ok": True, "is_loaded": mem.is_loaded}


def _apply_knowledge_settings(body: dict) -> None:
    """ダッシュボードから来た knowledge 設定を反映する（未指定のキーは触らない）"""
    k = orchestrator._cfg.knowledge
    if "enabled" in body:
        k.enabled = bool(body["enabled"])
    if "blocked_domains" in body:
        k.blocked_domains = [d.strip() for d in body["blocked_domains"] if str(d).strip()]

    search = body.get("search") or {}
    if "provider" in search:
        k.search.provider = str(search["provider"])
    if "max_results" in search:
        k.search.max_results = max(1, min(10, int(search["max_results"])))
    if "cache_ttl_sec" in search:
        k.search.cache_ttl_sec = max(0, int(search["cache_ttl_sec"]))

    news = body.get("news") or {}
    if "enabled" in news:
        k.news.enabled = bool(news["enabled"])
    if "feeds" in news:
        feeds = news["feeds"]
        if isinstance(feeds, str):
            feeds = [line for line in feeds.splitlines()]
        k.news.feeds = [f.strip() for f in feeds if str(f).strip()]
    if "refresh_interval_sec" in news:
        k.news.refresh_interval_sec = max(60, int(news["refresh_interval_sec"]))
    if "max_age_hours" in news:
        k.news.max_age_hours = float(news["max_age_hours"])
    if "keywords" in news:
        kws = news["keywords"]
        if isinstance(kws, str):
            kws = kws.split(",")
        k.news.keywords = [w.strip() for w in kws if str(w).strip()]

    topic = body.get("autonomous_topic") or {}
    if "enabled" in topic:
        k.autonomous_topic.enabled = bool(topic["enabled"])
    if "weights" in topic and isinstance(topic["weights"], dict):
        k.autonomous_topic.weights.update(
            {str(key): float(val) for key, val in topic["weights"].items()}
        )
    if "min_web_interval_sec" in topic:
        k.autonomous_topic.min_web_interval_sec = max(0, int(topic["min_web_interval_sec"]))
    if "remember" in topic:
        k.autonomous_topic.remember = bool(topic["remember"])

    answer = body.get("answer_with_search") or {}
    if "enabled" in answer:
        k.answer_with_search.enabled = bool(answer["enabled"])
    if "trigger_words" in answer:
        words = answer["trigger_words"]
        if isinstance(words, str):
            words = words.split(",")
        k.answer_with_search.trigger_words = [w.strip() for w in words if str(w).strip()]
    if "max_results" in answer:
        k.answer_with_search.max_results = max(1, min(10, int(answer["max_results"])))

    # 設定を反映したクライアントを作り直す
    orchestrator._build_knowledge()


# ── 「個」（マインド）まわり ──────────────────────────────────────

def _mind_payload(agent) -> dict:
    """1体ぶんの mind 状態を API 応答用にまとめる"""
    import dataclasses as _dc

    ms = agent.mind_store
    if ms is None:
        return {"name": agent.name, "available": False}
    state = ms.all_state()
    return {
        "name": agent.name,
        "available": True,
        "stats": ms.stats(),
        "beliefs": ms.get_state("beliefs", []),
        "state": {k: v for k, v in state.items() if k != "beliefs"},
        "opinions": [_dc.asdict(o) for o in ms.list_opinions(limit=20)],
        "intentions": [_dc.asdict(i) for i in ms.list_intentions(limit=10)],
        "preferences": [_dc.asdict(p) for p in
                        ms.list_preferences(min_evidence=1, limit=20)],
        "curiosity": state.get("curiosity", []),
        "journal": ms.list_journal(limit=5),
    }


@router.get("/api/mind")
async def api_mind(char: str = ""):
    """「個」の状態（意見・意図・選好・自己状態・思考ジャーナル）を返す。

    Phase 1 で「稼働中アプリの mind 状態を直接確認できない」という留保が
    あったことへの対処。char でキャラ key を指定でき、省略時は全員ぶん。
    """
    agents = []
    if char:
        found = orchestrator.get_agent(char)
        if found is not None:
            agents = [found]
    if not agents:
        agents = list(orchestrator.agents)
    return {
        "characters": {a.key: _mind_payload(a) for a in agents},
        "loop": orchestrator.mind_status(),
    }


@router.post("/api/mind/curiosity")
async def api_mind_curiosity(body: dict):
    """好奇心キューへ疑問を手動で積む（body: {char, question, author}）"""
    from backend.mind.mind_loop import get_curiosity, push_curiosity

    question = (body.get("question") or "").strip()
    if not question:
        return {"ok": False, "error": "question が空です"}
    agent = orchestrator.get_agent(body.get("char") or "") or orchestrator.lead
    if agent is None or agent.mind_store is None:
        return {"ok": False, "error": "対象キャラの MindStore がありません"}
    added = push_curiosity(agent.mind_store, question,
                           body.get("author") or "")
    return {"ok": True, "added": added, "character": agent.name,
            "curiosity": get_curiosity(agent.mind_store)}


@router.post("/api/mind/sleep")
async def api_mind_sleep(body: dict):
    """睡眠時統合を手動で1回実行する（body: {char} でキャラ key を任意指定）。

    統合にはLLM生成を含むため、応答まで時間がかかることがある。
    """
    result = await orchestrator.run_sleep_cycle((body or {}).get("char") or "")
    return {"ok": True, **result}


@router.get("/api/knowledge")
async def api_knowledge_status():
    """Web検索・ニュース・自律トークの状態と現在の設定"""
    k = orchestrator._cfg.knowledge
    status = orchestrator.knowledge_status()
    status["config"] = {
        "enabled": k.enabled,
        "blocked_domains": k.blocked_domains,
        "search": {
            "provider": k.search.provider,
            "max_results": k.search.max_results,
            "cache_ttl_sec": k.search.cache_ttl_sec,
        },
        "news": {
            "enabled": k.news.enabled,
            "feeds": k.news.feeds,
            "refresh_interval_sec": k.news.refresh_interval_sec,
            "max_age_hours": k.news.max_age_hours,
            "keywords": k.news.keywords,
        },
        "autonomous_topic": {
            "enabled": k.autonomous_topic.enabled,
            "weights": k.autonomous_topic.weights,
            "min_web_interval_sec": k.autonomous_topic.min_web_interval_sec,
            "remember": k.autonomous_topic.remember,
        },
        "answer_with_search": {
            "enabled": k.answer_with_search.enabled,
            "trigger_words": k.answer_with_search.trigger_words,
            "max_results": k.answer_with_search.max_results,
        },
    }
    return status


@router.post("/api/config/knowledge")
async def api_set_knowledge(body: dict):
    """知識まわりの設定を更新して config.yaml に保存する"""
    _apply_knowledge_settings(body)
    save_config(orchestrator._cfg)
    await hub.send_log("info", "知識設定を更新しました（保存済）")
    return {"ok": True, "status": orchestrator.knowledge_status()}


@router.get("/api/knowledge/news")
async def api_get_news(refresh: bool = False):
    """仕入れ済みのニュース一覧を返す。refresh=true で取り直す。"""
    if orchestrator._news is None:
        return {"enabled": False, "items": []}
    if refresh or not orchestrator._news.items:
        await orchestrator.refresh_news()
    return {
        "enabled": True,
        "items": [d.to_dict() for d in orchestrator._news.items],
        "status": orchestrator._news.status(),
    }


@router.post("/api/knowledge/search")
async def api_knowledge_search(body: dict):
    """Web検索だけを実行して結果を返す（発話はしない）"""
    query = (body.get("query") or "").strip()
    if not query:
        return {"ok": False, "error": "query is empty", "items": []}
    if orchestrator._search is None:
        return {"ok": False, "error": "knowledge disabled", "items": []}
    limit = int(body.get("limit") or orchestrator._cfg.knowledge.search.max_results)
    docs = await orchestrator.search_web(query, limit=limit)
    return {
        "ok": bool(docs),
        "query": query,
        "provider": orchestrator._search.last_provider,
        "error": orchestrator._search.last_error if not docs else "",
        "items": [d.to_dict() for d in docs],
    }


@router.post("/api/knowledge/topic")
async def api_force_topic(body: dict):
    """
    話題の種まき（2026-11-22 themefix で意味を変更）。

    旧実装はその場で _speak_idle を1回叩く fire-and-forget で、テーマを
    運ばずログも残らなかった。新実装はプランナの優先キューへ積み、
    次回以降の話題決定で最優先に消費させる（投入・消費ともログに残る）。
    kind: "" (自動) / "memory" / "news" / "curiosity" / "theme"。
    news は query（無ければ配信テーマ）で絞り込み、query 指定は
    テーマ的お題として扱う。レスポンス形式は従来と互換（queued を追加）。
    """
    if orchestrator._llm is None:
        return {"ok": False, "error": "配信が開始されていません"}
    kind = (body.get("kind") or "").strip()
    query = (body.get("query") or "").strip()
    # character を指定するとそのキャラが話題を振る（省略時は順番）
    agent = orchestrator.get_agent(body.get("character", "")) if body.get("character") else None
    queued = orchestrator.seed_topic(kind=kind, query=query)
    return {"ok": True, "kind": kind or "auto", "query": query,
            "character": agent.name if agent else "auto",
            "queued": queued}


def _chat_status() -> dict:
    """チャット画面に返すモード情報"""
    return {
        "running": orchestrator._running,
        "is_local_chat": orchestrator.is_local_chat,
        "mode": orchestrator._status.video_id or "",
        "user_name": orchestrator.local_user_name,
        "character": " & ".join(a.name for a in orchestrator.agents),
        "characters": [{"id": a.key, "name": a.name} for a in orchestrator.agents],
        "llm_backend": orchestrator._status.llm_backend,
        "tts_backend": orchestrator._status.tts_backend,
        "idle_talk": orchestrator._cfg.local_chat.idle_talk,
        "is_speaking": orchestrator._is_speaking,
        "memory": orchestrator.chat_memory_status(),
    }


@router.get("/api/chat")
async def api_chat_status():
    return _chat_status()


@router.post("/api/chat")
async def api_chat_send(body: dict):
    """ローカルユーザーの発言を送る（未起動なら自動でローカルモードを開始する）"""
    result = await orchestrator.send_local_message(
        body.get("message", ""), author=body.get("author") or None
    )
    result["status"] = _chat_status()
    return result


@router.get("/api/chat/history")
async def api_chat_history(limit: int = 100):
    """保存済みのチャット履歴（古い順）を返す"""
    limit = max(1, min(500, int(limit)))
    return {
        "items": orchestrator.chat_history(limit),
        "status": _chat_status(),
    }


@router.post("/api/chat/history/clear")
async def api_chat_history_clear():
    """チャット履歴を削除する（エピソード記憶は消さない）"""
    removed = orchestrator.clear_chat_history()
    await hub.send_log("info", f"チャット履歴を削除しました（{removed}件）")
    return {"ok": True, "removed": removed, "status": _chat_status()}


@router.post("/api/chat/remember")
async def api_chat_remember():
    """いまの会話をエピソード記憶へ整理する（手動トリガー）"""
    result = await orchestrator._run_consolidation(triggered_by="chat_manual")
    return {
        "ok": result is not None,
        "added": getattr(result, "added", 0),
        "status": _chat_status(),
    }


@router.post("/api/chat/start")
async def api_chat_start():
    """ローカルチャットモードで開始する"""
    await orchestrator.start("local")
    await hub.send_log("info", "ローカルチャットモードを開始しました")
    return {"ok": True, "status": _chat_status()}


@router.post("/api/chat/stop")
async def api_chat_stop():
    await orchestrator.stop()
    return {"ok": True, "status": _chat_status()}


@router.post("/api/config/audio")
async def api_set_audio(body: dict):
    device = body.get("device")
    orchestrator._cfg.audio_output.device = device if device else None
    orchestrator._audio = AudioRouter(device=orchestrator._cfg.audio_output.device)
    await hub.send_log("info", f"オーディオデバイス変更: {device or 'デフォルト'}")
    return {"ok": True}


@router.get("/api/system")
async def api_system():
    """CPU・メモリ・VRAM の現在値と直近5分の履歴"""
    if not monitor.snapshot().get("ts"):
        await monitor.sample()
    return monitor.snapshot()
