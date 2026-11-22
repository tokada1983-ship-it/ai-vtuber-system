"""WebSocketエンドポイントとダッシュボードからのコマンド処理"""
from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from backend.avatar import live2d as live2d_util
from backend.config import save_config
from backend.logger import get_logger
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
    orchestrator,
)
from backend.tts.audio_router import AudioRouter

logger = get_logger(__name__)
router = APIRouter()
from backend.api.http_routes import _apply_knowledge_settings, _avatar_payload, _chat_status


# ── WebSocket ─────────────────────────────────────────────────

# 注意: /ws/voice は /ws/{channel} より先に登録すること（先勝ちで解決されるため）
@router.websocket("/ws/voice")
async def websocket_voice(ws: WebSocket):
    """
    マイク音声の受け口。

    プロトコル:
      Client → Server: バイナリ = 発話区間1つぶんの PCM16LE / 16kHz / mono
                       JSON     = {"cmd": "status"} など
      Server → Client: {"type":"transcript","text":...,"latency_ms":...}
                       {"type":"transcript_empty","reason":...}
    """
    await ws.accept()
    logger.info("Voice WS connected")
    try:
        while True:
            message = await ws.receive()
            if message.get("type") == "websocket.disconnect":
                break

            if (payload := message.get("bytes")) is not None:
                result = await orchestrator.transcribe_and_enqueue(payload)
                if result.get("ok"):
                    await ws.send_json({"type": "transcript", **result})
                else:
                    await ws.send_json({"type": "transcript_empty",
                                        "reason": result.get("error", "")})
            elif (text := message.get("text")) is not None:
                try:
                    data = json.loads(text)
                except json.JSONDecodeError:
                    continue
                if data.get("cmd") == "status":
                    await ws.send_json({"type": "stt_status", **orchestrator.stt_status})
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.warning("Voice WS error", error=str(e))
    finally:
        logger.info("Voice WS disconnected")


@router.websocket("/ws/{channel}")
async def websocket_endpoint(ws: WebSocket, channel: str):
    await hub.connect(ws, channel)
    try:
        while True:
            data = await ws.receive_json()
            cmd = data.get("cmd")
            if cmd:
                await _handle_ws_cmd(cmd, data, ws=ws)
            elif data.get("type") in ("heartbeat", "page_event"):
                # アバターページの生死判定（ハートビート）と異常通知。
                # テレメトリ処理の失敗で本筋の接続を落とさないよう、ここで例外を隔離する
                try:
                    await hub.handle_client_message(channel, data)
                except Exception as e:
                    logger.warning("WS client message error", channel=channel, error=str(e))
    except WebSocketDisconnect:
        await hub.disconnect(ws, channel)
    except Exception as e:
        logger.warning("WS error", channel=channel, error=str(e))
        await hub.disconnect(ws, channel)


async def _handle_ws_cmd(cmd: str, data: dict, ws=None) -> None:
    """WebSocket経由の設定コマンドを処理する（例外ガード付き）"""
    try:
        await _handle_ws_cmd_inner(cmd, data, ws)
    except Exception as e:
        logger.error("WS command error", cmd=cmd, error=str(e))
        if ws:
            try:
                await ws.send_json({"type": "error", "cmd": cmd, "error": str(e)})
            except Exception:
                pass


def _ws_target_agent(data: dict):
    """WSコマンドの対象キャラを data["character"]（id か名前）で解決する。

    省略時は None を返し、呼び出し側が既定スコープを決める
    （表示・編集系はリーダー、破壊的なクリア系は全員）。
    指定されたのに見つからない場合は ValueError（呼び出し側の例外ガードが
    エラー応答に変換する）。黙ってリーダーに落とすと、操作の表示範囲と
    実際の適用範囲が食い違うため、解決できない指定は必ず失敗させる。
    """
    key = (data.get("character") or "").strip()
    if not key:
        return None
    agent = orchestrator.get_agent(key)
    if agent is None:
        raise ValueError(f"キャラクターが見つかりません: {key}")
    return agent


async def _handle_ws_cmd_inner(cmd: str, data: dict, ws=None) -> None:
    """WebSocket経由の設定コマンドを処理する"""
    if cmd == "unity_move":
        x = float(data.get("x", 0.0))
        y = float(data.get("y", 0.0))
        z = float(data.get("z", 0.0))
        await hub.send_unity_move(x, y, z)
        await hub.send_log("info", f"Unity移動指示: x={x}, y={y}, z={z}")

    elif cmd == "unity_gimmick":
        gimmick_id = data.get("gimmick_id", "")
        await hub.send_unity_gimmick(gimmick_id)
        await hub.send_log("info", f"Unityギミック使用指示: {gimmick_id}")

    elif cmd == "unity_action":
        action = data.get("action", "")
        target = data.get("target")
        await hub.send_unity_action(action, target)
        await hub.send_log("info", f"Unityアクション指示: {action} (target={target})")

    elif cmd == "start":
        await orchestrator.start(data.get("video_id", ""))

    elif cmd == "stop":
        await orchestrator.stop()

    elif cmd == "set_video_id":
        await orchestrator.set_video_id(data.get("video_id", ""))

    elif cmd == "set_llm":
        if "primary" in data:
            orchestrator._cfg.llm.primary = data["primary"]
        if "fallback" in data:
            orchestrator._cfg.llm.fallback = data["fallback"]
        await orchestrator.rebuild_llm()
        save_config(orchestrator._cfg)
        await hub.send_log("info", f"LLM切替: {orchestrator._cfg.llm.primary} (保存済)")
        await hub.broadcast("dashboard", {"type": "config_updated", "section": "llm",
                                           "primary": orchestrator._cfg.llm.primary})

    elif cmd == "set_llm_backend":
        name = data.get("name")
        if name and name in orchestrator._cfg.llm.backends:
            be = orchestrator._cfg.llm.backends[name]
            if "base_url"    in data: be.base_url    = data["base_url"]
            if "model"       in data: be.model        = data["model"]
            if "temperature" in data: be.temperature  = float(data["temperature"])
            if "max_tokens"  in data: be.max_tokens   = int(data["max_tokens"])
            await orchestrator.rebuild_llm()
            save_config(orchestrator._cfg)
            await hub.send_log("info", f"LLMパラメータ更新: {name} (保存済)")
            await hub.broadcast("dashboard", {"type": "config_updated", "section": "llm_backend", "name": name})

    elif cmd == "set_tts":
        if "primary" in data:
            orchestrator._cfg.tts.primary = data["primary"]
        name = orchestrator._cfg.tts.primary
        if name in orchestrator._cfg.tts.backends:
            be = orchestrator._cfg.tts.backends[name]
            if "base_url"              in data: be.base_url              = data["base_url"]
            if "speaker_id"            in data: be.speaker_id            = int(data["speaker_id"])
            if "speed"                 in data: be.speed                 = float(data["speed"])
            if "voice"                 in data: be.voice                 = data["voice"]
            if "use_emotion_voice"     in data: be.use_emotion_voice     = bool(data["use_emotion_voice"])
            if "emotion_voice_prefix"  in data: be.emotion_voice_prefix  = data["emotion_voice_prefix"]
        orchestrator.rebuild_tts()
        save_config(orchestrator._cfg)
        await hub.send_log("info", f"TTS切替: {orchestrator._cfg.tts.primary} (保存済)")
        await hub.broadcast("dashboard", {"type": "config_updated", "section": "tts",
                                           "primary": orchestrator._cfg.tts.primary})

    elif cmd == "set_source":
        if "primary_source" in data:
            orchestrator._cfg.youtube.primary_source = data["primary_source"]
        save_config(orchestrator._cfg)
        await hub.send_log("info", f"コメントソース変更: {data.get('primary_source')}（次回start時から反映・保存済）")

    elif cmd == "set_twitcasting":
        tc = orchestrator._cfg.twitcasting
        if "access_token" in data and data["access_token"]:
            tc.access_token = data["access_token"]
        if "user_id" in data:
            tc.user_id = data["user_id"]
        if "polling_interval_ms" in data:
            tc.polling_interval_ms = int(data["polling_interval_ms"])
        save_config(orchestrator._cfg)
        await hub.send_log("info", f"ツイキャス設定保存: user={tc.user_id}")

    elif cmd == "set_twitch":
        tw = orchestrator._cfg.twitch
        if "token" in data and data["token"]:
            tw.token = data["token"]
        if "nick" in data:
            tw.nick = data["nick"]
        if "channel" in data:
            tw.channel = data["channel"]
        save_config(orchestrator._cfg)
        await hub.send_log("info", f"Twitch設定保存: channel={tw.channel}")

    elif cmd == "chat_send":
        result = await orchestrator.send_local_message(
            data.get("message", ""), author=data.get("author") or None
        )
        if not result.get("ok"):
            await hub.broadcast("chat", {"type": "chat_error", "error": result.get("error", "")})
            await hub.send_log("warn", f"ローカルチャット: {result.get('error', '')}")
        await hub.broadcast("chat", {"type": "chat_status", **_chat_status()})

    elif cmd == "chat_start":
        await orchestrator.start("local")
        await hub.send_log("info", "ローカルチャットモードを開始しました")
        await hub.broadcast("chat", {"type": "chat_status", **_chat_status()})

    elif cmd == "chat_stop":
        await orchestrator.stop()
        await hub.broadcast("chat", {"type": "chat_status", **_chat_status()})

    elif cmd == "chat_history_clear":
        removed = orchestrator.clear_chat_history()
        await hub.send_log("info", f"チャット履歴を削除しました（{removed}件）")
        await hub.broadcast("chat", {"type": "chat_history_cleared", "removed": removed})
        await hub.broadcast("chat", {"type": "chat_status", **_chat_status()})

    elif cmd == "chat_remember":
        result = await orchestrator._run_consolidation(triggered_by="chat_manual")
        added = getattr(result, "added", 0)
        await hub.broadcast("chat", {
            "type": "chat_remembered",
            "added": added,
            "ok": result is not None,
        })
        await hub.broadcast("chat", {"type": "chat_status", **_chat_status()})

    elif cmd == "chat_ping":
        await hub.broadcast("chat", {"type": "chat_status", **_chat_status()})

    elif cmd == "set_knowledge":
        _apply_knowledge_settings(data)
        save_config(orchestrator._cfg)
        await hub.send_log("info", "知識設定を更新しました（保存済）")
        await hub.broadcast("dashboard", {"type": "config_updated", "section": "knowledge"})

    elif cmd == "refresh_news":
        await orchestrator.refresh_news()

    elif cmd == "force_topic":
        if orchestrator._llm is None:
            await hub.send_log("warn", "配信が開始されていないため話題を振れません")
        else:
            agent = (orchestrator.get_agent(data.get("character", ""))
                     if data.get("character") else None)
            asyncio.create_task(orchestrator._speak_idle(
                kind=(data.get("kind") or "").strip(),
                query=(data.get("query") or "").strip(),
                agent=agent,
            ))

    elif cmd == "web_search":
        query = (data.get("query") or "").strip()
        docs = await orchestrator.search_web(query) if query else []
        await hub.broadcast("dashboard", {
            "type": "search_result",
            "query": query,
            "items": [d.to_dict() for d in docs],
        })
        if not docs:
            await hub.send_log("warn", f"検索結果が見つかりませんでした: {query}")

    elif cmd == "set_orchestrator":
        cfg = orchestrator._cfg.orchestrator
        if "idle_threshold_sec"        in data: cfg.idle_threshold_sec        = int(data["idle_threshold_sec"])
        if "response_history_turns"    in data:
            cfg.response_history_turns = int(data["response_history_turns"])
            for agent in orchestrator.agents:
                agent.log.set_max_turns(cfg.response_history_turns)
        if "comment_dedupe_window_sec" in data: cfg.comment_dedupe_window_sec = int(data["comment_dedupe_window_sec"])
        if "streaming_tts"             in data: cfg.streaming_tts             = bool(data["streaming_tts"])
        save_config(orchestrator._cfg)
        await hub.send_log("info", "Orchestratorパラメータ更新 (保存済)")
        await hub.broadcast("dashboard", {"type": "config_updated", "section": "orchestrator"})

    elif cmd == "set_audio":
        device = data.get("device") or None
        orchestrator._cfg.audio_output.device = device
        orchestrator._audio = AudioRouter(device=device)
        save_config(orchestrator._cfg)
        await hub.send_log("info", f"オーディオデバイス変更: {device or 'デフォルト'} (保存済)")

    elif cmd == "save_memory":
        agent = _ws_target_agent(data) or orchestrator.lead
        content = data.get("content", "")
        agent.long_term_memory.save(content)
        agent.long_term_memory.reload()
        await hub.send_log("info", f"{agent.name}の長期記憶を更新しました ({len(content)}文字)")
        await hub.broadcast("dashboard", {"type": "config_updated", "section": "memory"})

    elif cmd == "reload_memory":
        agent = _ws_target_agent(data) or orchestrator.lead
        agent.long_term_memory.reload()
        await hub.send_log("info", f"{agent.name}の長期記憶を再読込しました")

    elif cmd == "reload_avatar":
        payload = _avatar_payload()
        model_path = (payload["live2d"]["model_path"] if payload["type"] == "live2d"
                      else payload["model_path"])
        logger.info("Sending reload to avatar", type=payload["type"], model_path=model_path)
        await hub.broadcast("avatar", {
            "type": "config_update",
            # ダッシュボードで変えるのは全体のアバター設定。専用アバターを持つキャラの画面は無視する
            "characters": [a.key for a in orchestrator.agents if a.cfg.avatar is None],
            "avatar_type": payload["type"],
            "model_path": payload["model_path"],
            "idle_motion": payload["idle_motion"],
            "live2d": payload["live2d"],
            "video": payload["video"],
            "calibration": _avatar_calibration,
        })
        await hub.send_log("info", f"アバター画面にリロード要求: {model_path}")

    elif cmd == "set_calibration":
        for k in ("armDownZ", "foreArmZ", "spineX"):
            if k in data:
                _avatar_calibration[k] = float(data[k])
        logger.info("Calibration updated", calibration=_avatar_calibration)
        await hub.broadcast("avatar", {"type": "calibration_update", "calibration": _avatar_calibration})
        await hub.send_log("info", f"キャリブレーション更新 → armDownZ={_avatar_calibration['armDownZ']} foreArmZ={_avatar_calibration['foreArmZ']}")

    elif cmd == "test_motion":
        motion = data.get("motion", "wave")
        await hub.broadcast("avatar", {"type": "test_motion", "motion": motion})
        logger.info("Test motion sent to avatar channel", motion=motion)
        await hub.send_log("info", f"テストモーション送信: {motion}")
        # 送信者にも確認を返す
        if ws:
            try:
                await ws.send_json({"type": "test_motion_ok", "motion": motion})
            except Exception:
                pass

    elif cmd == "consolidate_memory":
        """短期記憶を手動でエピソード記憶に整理する（character 省略時は全員）"""
        target = _ws_target_agent(data)
        targets = [target] if target else list(orchestrator.agents)
        if not any(a.llm for a in targets):
            await ws.send_json({"type": "consolidation_result",
                                "request_id": data.get("request_id"),
                                "ok": False, "error": "LLMが動いていません"})
        elif not any(a.log.get_pending() for a in targets):
            await ws.send_json({"type": "consolidation_result",
                                "request_id": data.get("request_id"),
                                "ok": False, "error": "整理する会話がありません"})
        else:
            result = await orchestrator._run_consolidation(triggered_by="manual",
                                                           agent=target)
            if result is None:
                await ws.send_json({"type": "consolidation_result",
                                    "request_id": data.get("request_id"),
                                    "ok": False, "error": "記憶整理に失敗しました"})
            else:
                await ws.send_json({
                    "type": "consolidation_result",
                    "request_id": data.get("request_id"),
                    "ok": result.ok,
                    **result.to_dict(),
                })

    elif cmd == "get_episodic_memory":
        """蓄積されたエピソード記憶の一覧を返す（character 省略時はリーダー）"""
        agent = _ws_target_agent(data) or orchestrator.lead
        store = agent.memory_store
        entries = store.all() if store else []
        await ws.send_json({
            "type": "episodic_memory",
            "request_id": data.get("request_id"),
            "character": agent.key,
            "character_name": agent.name,
            "entries": [e.to_dict() for e in entries],
            "stats": store.stats() if store else {},
            "pending_turns": agent.log.turns_since_consolidation,
            "trigger_turns": orchestrator._cfg.orchestrator.consolidation_trigger_turns,
        })

    elif cmd == "delete_episodic_memory":
        agent = _ws_target_agent(data) or orchestrator.lead
        entry_id = int(data.get("id", 0))
        ok = agent.memory_store.delete(entry_id) if agent.memory_store else False
        await ws.send_json({"type": "episodic_memory_deleted",
                            "request_id": data.get("request_id"),
                            "character": agent.key,
                            "ok": ok, "id": entry_id})

    elif cmd == "pin_episodic_memory":
        """ピン留めした記憶は忘却の対象外になる"""
        agent = _ws_target_agent(data) or orchestrator.lead
        entry_id = int(data.get("id", 0))
        pinned = bool(data.get("pinned", True))
        ok = (agent.memory_store.set_pinned(entry_id, pinned)
              if agent.memory_store else False)
        await ws.send_json({"type": "episodic_memory_pinned",
                            "request_id": data.get("request_id"),
                            "character": agent.key,
                            "ok": ok, "id": entry_id, "pinned": pinned})

    elif cmd == "add_episodic_memory":
        """手動で記憶を追加する（source=manual は忘却されない）"""
        agent = _ws_target_agent(data) or orchestrator.lead
        text = (data.get("text") or "").strip()
        store = agent.memory_store
        entry = store.add(text, source="manual") if (store and text) else None
        await ws.send_json({"type": "episodic_memory_added",
                            "request_id": data.get("request_id"),
                            "character": agent.key,
                            "ok": entry is not None,
                            "entry": entry.to_dict() if entry else None,
                            "error": "" if entry else "空、または既に似た記憶があります"})

    elif cmd == "clear_episodic_memory":
        """エピソード記憶の全消去。character 省略時は全員に適用する
        （以前はリーダーだけを消して「全消去」と表示していた）"""
        target = _ws_target_agent(data)
        targets = [target] if target else list(orchestrator.agents)
        per_character = {
            a.key: (a.memory_store.clear() if a.memory_store else 0) for a in targets
        }
        n = sum(per_character.values())
        if target:
            msg = f"{target.name}のエピソード記憶を全消去しました（{n}件）"
        else:
            detail = "／".join(
                f"{a.name} {per_character[a.key]}件" for a in targets)
            msg = f"全員のエピソード記憶を全消去しました（{detail}、計{n}件）"
        await hub.send_log("warn", msg)
        await ws.send_json({"type": "episodic_memory_cleared",
                            "request_id": data.get("request_id"), "ok": True,
                            "removed": n, "per_character": per_character,
                            "character": target.key if target else ""})

    elif cmd == "set_consolidation_interval":
        """定期整理の間隔を設定する（0=無効）。config へ保存し永続化する（F8）"""
        minutes = int(data.get("minutes", 0))
        orchestrator._consolidation_interval_min = minutes
        orchestrator._cfg.orchestrator.consolidation_interval_min = minutes
        save_config(orchestrator._cfg)
        msg = f"定期記憶整理: {'無効' if minutes == 0 else str(minutes) + '分ごと'}（保存済）"
        await hub.send_log("info", msg)
        await ws.send_json({"type": "consolidation_interval_set",
                            "request_id": data.get("request_id"),
                            "ok": True, "minutes": minutes})

    elif cmd == "bgm_load":
        """プレイリストを読み込む（フォルダパス or M3Uファイル）"""
        source = data.get("source", "bgm")
        count = await asyncio.to_thread(orchestrator._bgm.load_playlist, source)
        await hub.send_log("info", f"BGM: {count}曲読み込み完了 ({source})")
        await ws.send_json({
            "type": "bgm_status",
            "request_id": data.get("request_id"),
            **orchestrator._bgm.current_track_info(),
            "playlist": orchestrator._bgm.get_playlist(),
        })

    elif cmd == "bgm_play":
        ok = await orchestrator._bgm.play()
        info = orchestrator._bgm.current_track_info()
        await hub.send_log("info", f"BGM再生: {info.get('name', '')}")
        await hub.broadcast("dashboard", {"type": "bgm_status", **info})

    elif cmd == "bgm_pause":
        await orchestrator._bgm.pause()
        await hub.send_log("info", "BGM一時停止")
        await hub.broadcast("dashboard", {"type": "bgm_status", **orchestrator._bgm.current_track_info()})

    elif cmd == "bgm_resume":
        await orchestrator._bgm.resume()
        await hub.send_log("info", "BGM再開")
        await hub.broadcast("dashboard", {"type": "bgm_status", **orchestrator._bgm.current_track_info()})

    elif cmd == "bgm_stop":
        await orchestrator._bgm.stop()
        await hub.send_log("info", "BGM停止")
        await hub.broadcast("dashboard", {"type": "bgm_status", "is_playing": False, "name": ""})

    elif cmd == "bgm_next":
        await orchestrator._bgm.next_track()
        info = orchestrator._bgm.current_track_info()
        await hub.send_log("info", f"BGM次の曲: {info.get('name', '')}")
        await hub.broadcast("dashboard", {"type": "bgm_status", **info})

    elif cmd == "bgm_prev":
        await orchestrator._bgm.prev_track()
        info = orchestrator._bgm.current_track_info()
        await hub.broadcast("dashboard", {"type": "bgm_status", **info})

    elif cmd == "bgm_play_at":
        index = int(data.get("index", 0))
        await orchestrator._bgm.play_at(index)
        info = orchestrator._bgm.current_track_info()
        await hub.send_log("info", f"BGM再生: {info.get('name', '')}")
        await hub.broadcast("dashboard", {"type": "bgm_status", **info})

    elif cmd == "bgm_volume":
        vol = float(data.get("volume", 0.5))
        orchestrator._bgm.set_volume(vol)
        await hub.broadcast("dashboard", {"type": "bgm_status", **orchestrator._bgm.current_track_info()})

    elif cmd == "bgm_shuffle":
        orchestrator._bgm.set_shuffle(bool(data.get("shuffle", False)))
        await hub.broadcast("dashboard", {"type": "bgm_status", **orchestrator._bgm.current_track_info()})

    elif cmd == "bgm_status":
        await ws.send_json({
            "type": "bgm_status",
            "request_id": data.get("request_id"),
            **orchestrator._bgm.current_track_info(),
            "playlist": orchestrator._bgm.get_playlist(),
        })

    elif cmd == "set_stream_theme":
        theme = data.get("theme", "").strip()
        # 配信テーマは出演者全員で共有する
        for ch in orchestrator._cfg.characters:
            ch.stream_theme = theme
        save_config(orchestrator._cfg)
        msg = f"配信テーマ設定: 「{theme}」" if theme else "配信テーマ: なし（フリー）"
        await hub.send_log("info", msg)
        await ws.send_json({"type": "stream_theme_set",
                            "request_id": data.get("request_id"),
                            "ok": True, "theme": theme})

    elif cmd == "inject_comment":
        """ダッシュボードからテストコメントを注入する"""
        message = data.get("message", "").strip()
        author  = data.get("author", "テスト").strip() or "テスト"
        if not message:
            await ws.send_json({"type": "inject_error", "error": "message required"})
        else:
            from backend.models import Comment
            from datetime import datetime
            comment = Comment(
                id=f"inject_{datetime.now().timestamp()}",
                author=author,
                author_channel_id="inject_channel",
                message=message,
                timestamp=datetime.now(),
                is_superchat=data.get("is_superchat", False),
                superchat_amount=int(data["amount"]) if data.get("is_superchat") and data.get("amount") else None,
                is_member=data.get("is_member", False),
                is_moderator=data.get("is_moderator", False),
                priority=10.0,  # 最優先
            )
            # コメントキューに直接投入（上限チェック付き）
            await orchestrator.enqueue_comment(comment, score=10.0)
            # ダッシュボードのコメント一覧にも表示
            await hub.send_comment({
                "author": author,
                "message": message,
                "priority": 10.0,
                "is_superchat": comment.is_superchat,
                "is_member": comment.is_member,
                "injected": True,
            })
            await hub.send_log("info", f"テストコメント注入: [{author}] {message}")

    elif cmd == "list_presets":
        names = _preset_manager.list_presets()
        summaries = [_preset_manager.get_summary(n) for n in names]
        await ws.send_json({
            "type": "presets_list",
            "request_id": data.get("request_id"),
            "presets": summaries,
        })

    elif cmd == "load_preset":
        name = data.get("name", "")
        preset = _preset_manager.load(name)
        if not preset:
            await ws.send_json({"type": "preset_error", "request_id": data.get("request_id"), "error": "not found"})
        else:
            import yaml as _yaml
            from pathlib import Path as _Path

            # character.yamlに書き込む
            char_path = _Path(orchestrator._cfg.character.prompt_file)
            char_path.parent.mkdir(parents=True, exist_ok=True)
            char_data = {
                "name": preset.name,
                "description": preset.description,
                "personality": preset.personality,
                "rules": preset.rules,
            }
            with open(char_path, "w", encoding="utf-8") as _f:
                _yaml.dump(char_data, _f, allow_unicode=True, default_flow_style=False)

            # 記憶系をまとめて切り替える（コア記憶・エピソード記憶・整理先・会話履歴）
            await orchestrator.switch_character(preset.name)
            if preset.memory:
                orchestrator._long_term_memory.save(preset.memory)

            # キャリブレーションを更新
            _avatar_calibration["armDownZ"] = preset.avatar.arm_down_z
            _avatar_calibration["foreArmZ"] = preset.avatar.fore_arm_z
            _avatar_calibration["spineX"]   = preset.avatar.spine_x

            # アバターパスを更新
            orchestrator._cfg.avatar.vrm.model_path = preset.avatar.model_path
            orchestrator._cfg.avatar.vrm.idle_motion = preset.avatar.idle_motion
            orchestrator._cfg.character.name = preset.name
            save_config(orchestrator._cfg)

            # アバター画面にも通知
            await hub.broadcast("avatar", {
                "type": "config_update",
                "model_path": preset.avatar.model_path,
                "idle_motion": preset.avatar.idle_motion,
                "calibration": _avatar_calibration,
            })
            await hub.send_log("info", f"プリセット読み込み: {preset.name}")
            await ws.send_json({
                "type": "preset_loaded",
                "request_id": data.get("request_id"),
                "name": preset.name,
                "character": char_data,
                "memory": preset.memory,
                "avatar": {
                    "model_path": preset.avatar.model_path,
                    "idle_motion": preset.avatar.idle_motion,
                },
                "calibration": _avatar_calibration,
            })

    elif cmd == "save_preset":
        name = data.get("name", "").strip()
        if not name:
            await ws.send_json({"type": "preset_error", "request_id": data.get("request_id"), "error": "name required"})
        else:
            import yaml as _yaml
            from pathlib import Path as _Path

            # 現在の設定を収集
            char_path = _Path(orchestrator._cfg.character.prompt_file)
            char_data = {}
            if char_path.exists():
                with open(char_path, encoding="utf-8") as _f:
                    char_data = _yaml.safe_load(_f) or {}

            memory_text = orchestrator._long_term_memory.get_raw()

            preset = CharacterPreset(
                name=name,
                description=data.get("description") or char_data.get("description", ""),
                personality=data.get("personality") or char_data.get("personality", ""),
                rules=data.get("rules") or char_data.get("rules", ""),
                memory=data.get("memory") or memory_text,
                avatar=AvatarPresetModel(
                    model_path=orchestrator._cfg.avatar.vrm.model_path,
                    idle_motion=orchestrator._cfg.avatar.vrm.idle_motion,
                    arm_down_z=_avatar_calibration.get("armDownZ", 0.1),
                    fore_arm_z=_avatar_calibration.get("foreArmZ", 0.05),
                    spine_x=_avatar_calibration.get("spineX", 0.0),
                )
            )
            _preset_manager.save(preset)
            await hub.send_log("info", f"プリセット保存: {name}")
            await ws.send_json({
                "type": "preset_saved",
                "request_id": data.get("request_id"),
                "ok": True,
                "name": name,
            })

    elif cmd == "delete_preset":
        name = data.get("name", "")
        ok = _preset_manager.delete(name)
        await hub.send_log("info" if ok else "warn", f"プリセット削除: {name}")
        await ws.send_json({"type": "preset_deleted", "request_id": data.get("request_id"), "ok": ok})

    elif cmd == "get_character":
        """キャラクター設定（character.yaml）の内容を返す"""
        import yaml as _yaml
        from pathlib import Path as _Path
        char_path = _Path(orchestrator._cfg.character.prompt_file)
        char_data = {}
        if char_path.exists():
            with open(char_path, encoding="utf-8") as _f:
                char_data = _yaml.safe_load(_f) or {}
        await ws.send_json({
            "type": "character_response",
            "request_id": data.get("request_id"),
            "name":        char_data.get("name", ""),
            "description": char_data.get("description", ""),
            "personality": char_data.get("personality", ""),
            "rules":       char_data.get("rules", ""),
            "path":        str(char_path),
        })

    elif cmd == "save_character":
        """キャラクター設定をcharacter.yamlに保存する"""
        import yaml as _yaml
        from pathlib import Path as _Path
        char_path = _Path(orchestrator._cfg.character.prompt_file)
        char_path.parent.mkdir(parents=True, exist_ok=True)
        char_data = {
            "name":        data.get("name", ""),
            "description": data.get("description", ""),
            "personality": data.get("personality", ""),
            "rules":       data.get("rules", ""),
        }
        with open(char_path, "w", encoding="utf-8") as _f:
            _yaml.dump(char_data, _f, allow_unicode=True, default_flow_style=False)
        # キャラ名をconfigにも反映
        if char_data["name"]:
            orchestrator._cfg.character.name = char_data["name"]
            save_config(orchestrator._cfg)
        await hub.send_log("info", f"キャラクター設定保存: {char_data['name']}")
        await ws.send_json({"type": "character_saved", "request_id": data.get("request_id"), "ok": True})

    elif cmd == "get_short_term_memory":
        """短期記憶（会話履歴）の内容を返す（character 省略時はリーダー）"""
        agent = _ws_target_agent(data) or orchestrator.lead
        messages = agent.log.get_messages()
        entries = []
        for m in messages:
            entry = {"role": m.role, "content": m.content}
            if m.comment_id:
                entry["comment_id"] = m.comment_id
            entries.append(entry)
        await ws.send_json({
            "type": "short_term_memory_response",
            "request_id": data.get("request_id"),
            "character": agent.key,
            "character_name": agent.name,
            "messages": entries,
            "count": len(entries),
            "max_turns": orchestrator._cfg.orchestrator.response_history_turns,
            "total_turns": agent.log.total_turns,
            "turns_since_consolidation": agent.log.turns_since_consolidation,
            "consolidation_trigger": orchestrator._cfg.orchestrator.consolidation_trigger_turns,
        })

    elif cmd == "clear_short_term_memory":
        """短期記憶をクリアする。character 省略時は全員に適用する
        （以前はリーダーだけを消していたため、二人の文脈が非対称になった）"""
        target = _ws_target_agent(data)
        targets = [target] if target else list(orchestrator.agents)
        for a in targets:
            a.log.clear()
        if target:
            msg = f"{target.name}の短期記憶をクリアしました"
        else:
            names = "・".join(a.name for a in targets)
            msg = f"全員（{names}）の短期記憶をクリアしました"
        await hub.send_log("info", msg)
        await ws.send_json({"type": "short_term_memory_cleared",
                            "request_id": data.get("request_id"), "ok": True,
                            "characters": [a.key for a in targets]})

    elif cmd == "rag_search":
        """RAG検索テスト（ダッシュボードから検索できる）"""
        agent = _ws_target_agent(data) or orchestrator.lead
        query = data.get("query", "")
        top_k = int(data.get("top_k", 3))
        if not query:
            await ws.send_json({"type": "rag_result", "request_id": data.get("request_id"),
                                "character": agent.key, "results": []})
        else:
            # インデックスを最新化
            memory_text = agent.long_term_memory.get_raw()
            agent.rag.index(memory_text)
            results = agent.rag.search(query, top_k=top_k)
            await ws.send_json({
                "type": "rag_result",
                "request_id": data.get("request_id"),
                "character": agent.key,
                "character_name": agent.name,
                "query": query,
                "mode": agent.rag.mode,
                "chunks": agent.rag.chunk_count,
                "results": [
                    {"text": r.text, "section": r.section, "score": round(r.score, 3)}
                    for r in results
                ],
            })

    elif cmd == "get_rag_status":
        """RAGの状態を返す"""
        agent = _ws_target_agent(data) or orchestrator.lead
        memory_text = agent.long_term_memory.get_raw()
        agent.rag.index(memory_text)
        await ws.send_json({
            "type": "rag_status",
            "request_id": data.get("request_id"),
            "character": agent.key,
            "character_name": agent.name,
            "mode": agent.rag.mode,
            "chunk_count": agent.rag.chunk_count,
            "indexed": len(memory_text) > 0,
        })

    elif cmd == "set_avatar":
        av_type = data.get("type")
        if av_type in ("vrm", "live2d"):
            orchestrator._cfg.avatar.type = av_type
        model_path = data.get("model_path")
        if model_path:
            orchestrator._cfg.avatar.vrm.model_path = model_path
        idle_motion = data.get("idle_motion")
        if idle_motion:
            orchestrator._cfg.avatar.vrm.idle_motion = idle_motion
        # Live2D（送られてきた項目だけ更新する）
        l2d = data.get("live2d") or {}
        cfg_l2d = orchestrator._cfg.avatar.live2d
        if "model_path" in l2d:
            cfg_l2d.model_path = str(l2d["model_path"] or "").replace("\\", "/")
        for key in ("scale", "position_x", "position_y", "lipsync_gain"):
            if key in l2d:
                try:
                    setattr(cfg_l2d, key, float(l2d[key]))
                except (TypeError, ValueError):
                    pass
        for key in ("idle_motion_group", "mouth_param"):
            if key in l2d:
                setattr(cfg_l2d, key, str(l2d[key] or ""))
        for key in ("expression_map", "motion_map"):
            value = l2d.get(key)
            if isinstance(value, dict):
                setattr(cfg_l2d, key, {str(k): str(v) for k, v in value.items() if v})
        save_config(orchestrator._cfg)
        payload = _avatar_payload()
        shown = (payload["live2d"]["model_path"] if payload["type"] == "live2d"
                 else payload["model_path"])
        logger.info("Avatar config saved", type=payload["type"], model_path=shown)
        await hub.send_log("info", "アバター設定保存: " + payload["type"] + " / " + shown)
        await hub.broadcast("avatar", {
            "type": "config_update",
            "avatar_type": payload["type"],
            "model_path": payload["model_path"],
            "idle_motion": payload["idle_motion"],
            "live2d": payload["live2d"],
            "video": payload["video"],
            "calibration": _avatar_calibration,
        })

    elif cmd == "get_config":
        cfg = orchestrator._cfg
        payload = {
            "type": "config_response",
            "request_id": data.get("request_id"),
            "llm": {
                "primary": cfg.llm.primary,
                "fallback": cfg.llm.fallback,
                "backends": {
                    name: {"base_url": be.base_url, "model": be.model,
                           "temperature": be.temperature, "max_tokens": be.max_tokens}
                    for name, be in cfg.llm.backends.items()
                },
            },
            "tts": {
                "primary": cfg.tts.primary,
                "backends": {
                    name: {"base_url": be.base_url, "speaker_id": be.speaker_id,
                           "speed": be.speed, "voice": be.voice,
                           "use_emotion_voice": be.use_emotion_voice,
                           "emotion_voice_prefix": be.emotion_voice_prefix}
                    for name, be in cfg.tts.backends.items()
                },
            },
            "youtube": {"primary_source": cfg.youtube.primary_source},
            "character": {"name": cfg.character.name, "stream_theme": cfg.character.stream_theme},
            "characters": orchestrator.characters_status(),
            "ensemble": cfg.ensemble.model_dump(),
            "orchestrator": {
                "idle_threshold_sec": cfg.orchestrator.idle_threshold_sec,
                "response_history_turns": cfg.orchestrator.response_history_turns,
                "consolidation_trigger_turns": cfg.orchestrator.consolidation_trigger_turns,
                "comment_dedupe_window_sec": cfg.orchestrator.comment_dedupe_window_sec,
                "streaming_tts": cfg.orchestrator.streaming_tts,
            },
            "audio_output": {"device": cfg.audio_output.device},
            "twitcasting": {
                "user_id": cfg.twitcasting.user_id,
                "polling_interval_ms": cfg.twitcasting.polling_interval_ms,
                "access_token_env": cfg.twitcasting.access_token_env,
            },
            "twitch": {
                "nick": cfg.twitch.nick,
                "channel": cfg.twitch.channel,
                "token_env": cfg.twitch.token_env,
            },
            "avatar": {
                "type": cfg.avatar.type,
                "vrm": {
                    "model_path": cfg.avatar.vrm.model_path,
                    "idle_motion": cfg.avatar.vrm.idle_motion,
                },
            },
        }
        await ws.send_json(payload)

    elif cmd == "get_devices":
        from backend.tts.audio_router import AudioRouter
        devices = AudioRouter.list_devices()
        await ws.send_json({
            "type": "devices_response",
            "request_id": data.get("request_id"),
            "devices": devices,
        })

    elif cmd == "get_memory":
        agent = _ws_target_agent(data) or orchestrator.lead
        mem = agent.long_term_memory
        await ws.send_json({
            "type": "memory_response",
            "request_id": data.get("request_id"),
            "character": mem.character_name,
            "character_key": agent.key,
            "content": mem.get_raw(),
            "path": str(mem.path),
            "is_loaded": mem.is_loaded,
        })
