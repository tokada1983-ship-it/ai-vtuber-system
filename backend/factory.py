"""コンポーネントの組み立て（config から実体を作る）

以前は Orchestrator の中に LLM / TTS / STT / コメントソースの生成処理が
直書きされており、バックエンドを1つ足すたびに本体を触る必要があった。
生成はここに集め、Orchestrator は「組み立て済みの部品を回す」ことに専念する。

新しいLLMバックエンドを足すときは LLM_BACKENDS に1行追加するだけでよい。
"""
from __future__ import annotations

from typing import Callable

from backend.config import AppConfig, CharacterConfig, LLMBackendConfig, TTSBackendConfig
from backend.llm.base import LLMBackend
from backend.logger import get_logger
from backend.sources.base import CommentSource
from backend.stt.base import STTBackend
from backend.tts.base import TTSBackend

logger = get_logger(__name__)


# ── LLM ─────────────────────────────────────────────────────

def _lmstudio(cfg: LLMBackendConfig) -> LLMBackend:
    from backend.llm.lmstudio_backend import LMStudioBackend
    return LMStudioBackend(cfg)


def _ollama(cfg: LLMBackendConfig) -> LLMBackend:
    from backend.llm.ollama_backend import OllamaBackend
    return OllamaBackend(cfg)


def _claude(cfg: LLMBackendConfig) -> LLMBackend:
    from backend.llm.claude_backend import ClaudeBackend
    return ClaudeBackend(cfg)


def _openai(cfg: LLMBackendConfig) -> LLMBackend:
    from backend.llm.openai_backend import OpenAIBackend
    return OpenAIBackend(cfg)


def _gemini(cfg: LLMBackendConfig) -> LLMBackend:
    from backend.llm.gemini_backend import GeminiBackend
    return GeminiBackend(cfg)


def _openai_compat(cfg: LLMBackendConfig) -> LLMBackend:
    # llama.cpp server / vLLM / text-generation-webui など。
    # QLoRAアダプタを llama-server --lora で載せたモデルもここから呼べる
    from backend.llm.openai_compat import OpenAICompatBackend
    return OpenAICompatBackend(cfg)


LLM_BACKENDS: dict[str, Callable[[LLMBackendConfig], LLMBackend]] = {
    "lmstudio": _lmstudio,
    "ollama": _ollama,
    "claude": _claude,
    "openai": _openai,
    "gemini": _gemini,
    "openai_compat": _openai_compat,
    "llamacpp": _openai_compat,
    "vllm": _openai_compat,
}


def _apply_adaptive_ctx(backend: LLMBackend | None,
                        config: AppConfig) -> LLMBackend | None:
    """config.llm の num_ctx 適応制御設定をバックエンドへ伝える。

    対応するバックエンド（Ollama）だけが configure_adaptive_ctx を持ち、
    それ以外は素通りする。HybridRouter の内側にも行き渡らせる
    （ルーター生成前に内側が印付けされる経路が主だが、念のため再帰する）。
    """
    if backend is None:
        return None
    from backend.llm.hybrid_router import HybridRouter
    if isinstance(backend, HybridRouter):
        _apply_adaptive_ctx(backend.idle_backend, config)
        _apply_adaptive_ctx(backend.fast_backend, config)
        return backend
    setter = getattr(backend, "configure_adaptive_ctx", None)
    if setter is not None:
        setter(config.llm)
    return backend


def _apply_protocol(backend: LLMBackend | None,
                    config: AppConfig) -> LLMBackend | None:
    """config.llm.protocol をバックエンドへ伝える（Phase 5 Stage D）。

    バックエンドはこれを見てパーサ（JSON / タグ）を切り替え、Ollama は
    format:json の送信を抑える。is_persona と同じ「生成時に印を付ける」
    方式で、config 側の単一の切替点（llm.protocol）から全体に行き渡る。
    あわせて num_ctx 適応制御の設定（llm.adaptive_ctx 等）もここから
    行き渡らせる（生成時の印付けはこの1箇所に集約する）。
    """
    if backend is not None:
        backend.protocol = str(config.llm.protocol or "json").lower()
        _apply_adaptive_ctx(backend, config)
    return backend


def create_llm(name: str, cfg: LLMBackendConfig | None) -> LLMBackend | None:
    """バックエンド名と設定からLLMを作る。未知の名前・設定なしなら None。"""
    factory = LLM_BACKENDS.get(name)
    if factory is None or cfg is None:
        return None
    return factory(cfg)


def build_llm(config: AppConfig) -> tuple[LLMBackend, LLMBackend | None]:
    """
    (primary, fallback) を返す。

    persona.enabled なら、学習済みキャラモデル（QLoRA）を primary にし、
    通常の primary を fallback に回す。専用モデルが落ちていても
    ヘルスチェックで通常モデルに切り替わるので配信は止まらない。
    """
    llm_cfg = config.llm
    primary = _apply_protocol(
        create_llm(llm_cfg.primary, llm_cfg.get_backend(llm_cfg.primary)), config)
    if primary is None:
        from backend.llm.ollama_backend import OllamaBackend
        primary = _apply_protocol(
            OllamaBackend(llm_cfg.get_backend("ollama") or LLMBackendConfig()), config)
    fallback = _apply_protocol(
        create_llm(llm_cfg.fallback, llm_cfg.get_backend(llm_cfg.fallback)), config)

    persona_llm = build_persona_llm(config)
    if persona_llm is not None:
        base = primary if config.persona.fallback_to_base else fallback
        return persona_llm, base
    return wrap_hybrid(config, primary), fallback


def build_character_llm(
    config: AppConfig, character: CharacterConfig, *, is_lead: bool,
) -> tuple[LLMBackend, LLMBackend | None]:
    """
    出演キャラ1体ぶんの (primary, fallback) を返す。

    - リーダーは従来どおり（人格QLoRAが有効ならその専用モデル）
    - 相方は人格QLoRAを使わない。QLoRAはリーダーの口調で学習してあるため、
      相方に載せると相方までリーダーの口調で喋ってしまう
    - character.llm で、キャラごとにバックエンドやモデル（相方専用モデル等）を差し替えられる
    """
    over = character.llm
    if not (over.backend or over.model):
        if is_lead:
            return build_llm(config)
        return build_llm(config.model_copy(update={
            "persona": config.persona.model_copy(update={"enabled": False}),
        }))

    name = over.backend or config.llm.primary
    base_cfg = config.llm.get_backend(name)
    if base_cfg is None:
        logger.warning("character llm backend not configured",
                       character=character.name, backend=name)
        return build_character_llm(
            config, character.model_copy(update={"llm": type(over)()}), is_lead=is_lead)
    cfg = base_cfg.model_copy(update={"model": over.model}) if over.model else base_cfg
    primary = _apply_protocol(create_llm(name, cfg), config)
    if primary is None:
        logger.warning("unknown character llm backend", character=character.name, backend=name)
        return build_llm(config)
    # 専用モデルが落ちていたら全体の primary に戻す
    fallback = _apply_protocol(
        create_llm(config.llm.primary, config.llm.get_backend(config.llm.primary)),
        config)
    return wrap_hybrid(config, primary), fallback


def wrap_hybrid(config: AppConfig, base: LLMBackend) -> LLMBackend:
    """
    llm.hybrid が有効なら、base（従来の primary）を待機系に据えたまま
    即応系バックエンドを被せた HybridRouter を返す。無効なら base のまま。

    待機系は base そのもの（キャラ別LLMや人格QLoRAの解決が済んだ実体）を
    そのまま使うので、ハイブリッドを切れば従来の挙動に完全に戻る。
    """
    hybrid = config.llm.hybrid
    if not hybrid.enabled or not hybrid.fast_backend:
        return base

    def build_named(name: str) -> LLMBackend | None:
        cfg = config.llm.get_backend(name)
        if cfg is None:
            logger.warning("hybrid backend not configured", backend=name)
            return None
        backend = _apply_protocol(create_llm(name, cfg), config)
        if backend is None:
            logger.warning("hybrid backend unknown", backend=name)
        return backend

    idle = base
    if hybrid.idle_backend:
        named_idle = build_named(hybrid.idle_backend)
        if named_idle is not None:
            idle = named_idle
    fast = build_named(hybrid.fast_backend)
    if fast is None:
        return base
    if idle is fast:
        return base
    from backend.llm.hybrid_router import HybridRouter
    logger.info("hybrid LLM routing enabled",
                idle=idle.name, fast=fast.name)
    return HybridRouter(idle, fast)


def build_persona_llm(config: AppConfig) -> LLMBackend | None:
    """人格QLoRAを焼き込んだ専用モデルのLLMバックエンド（無効なら None）"""
    p = config.persona
    if not (p.enabled and p.runtime_model):
        return None
    base_cfg = config.llm.get_backend(p.backend)
    if base_cfg is None:
        logger.warning("persona backend not configured", backend=p.backend)
        return None
    # 接続先（base_url 等）は流用し、モデル名だけ専用モデルに差し替える
    cfg = base_cfg.model_copy(update={"model": p.runtime_model})
    llm = _apply_protocol(create_llm(p.backend, cfg), config)
    if llm is None:
        logger.warning("unknown persona backend", backend=p.backend)
        return None
    # Orchestrator はこの印を見て、軽量プロンプト（slim）に切り替える。
    # ヘルスチェックで通常モデルへフォールバックしたら自動で全文プロンプトに戻る
    llm.is_persona = True
    return llm


# ── TTS ─────────────────────────────────────────────────────

def build_tts(config: AppConfig, character: CharacterConfig | None = None) -> TTSBackend:
    """
    TTSを作る。character を渡すと、そのキャラの声（character.voice）で上書きする。
    掛け合いでは同じTTSサーバを使い回し、話者（voice / speaker_id）だけ変える。
    """
    tts_cfg = config.tts
    voice = character.voice if character is not None else None
    primary = (voice.backend if voice and voice.backend else "") or tts_cfg.primary
    raw = tts_cfg.backends.get(primary)
    if voice is not None:
        update = {k: v for k, v in {
            "voice": voice.voice or None,
            "speaker_id": voice.speaker_id,
            "speed": voice.speed,
            "pitch": voice.pitch,
        }.items() if v is not None}
        if update:
            raw = (raw or TTSBackendConfig()).model_copy(update=update)

    if primary == "irodori":
        from backend.tts.irodori_backend import IrodoriTTSBackend, IrodoriTTSConfig
        cfg_dict = {}
        if raw:
            cfg_dict = {
                "base_url": raw.base_url,
                "model": raw.model,
                "voice": raw.voice,
                "use_emotion_voice": raw.use_emotion_voice,
                "emotion_voice_prefix": raw.emotion_voice_prefix,
                "response_format": raw.response_format,
                "api_key": raw.api_key,
            }
        return IrodoriTTSBackend(IrodoriTTSConfig.from_dict(cfg_dict))

    from backend.tts.voicevox_backend import VoicevoxBackend
    if primary == "voicevox" and raw:
        return VoicevoxBackend(raw)
    # フォールバック: VOICEVOXデフォルト
    return VoicevoxBackend(TTSBackendConfig())


# ── STT ─────────────────────────────────────────────────────

def build_stt(config: AppConfig) -> STTBackend | None:
    cfg = config.stt
    if not cfg.enabled:
        return None
    if cfg.backend != "faster_whisper":
        logger.warning("Unknown STT backend, disabling voice input", backend=cfg.backend)
        return None
    from backend.stt.whisper_backend import FasterWhisperBackend
    return FasterWhisperBackend(
        model_size=cfg.model,
        device=cfg.device,
        language=cfg.language,
        compute_type=cfg.compute_type,
    )


# ── コメントソース ──────────────────────────────────────────

def is_local_mode(config: AppConfig, video_id: str) -> bool:
    """ローカル1対1チャットで起動すべきか"""
    primary = config.youtube.primary_source
    # Mockを明示的に押したときはMockを優先する（設定がlocalでも上書きしない）
    return video_id == "local" or (primary == "local" and video_id != "mock")


def build_sources(config: AppConfig, video_id: str) -> tuple[CommentSource, CommentSource | None]:
    """(primary, fallback) のコメントソースを返す"""
    from backend.sources.local_chat_source import LocalChatSource
    from backend.sources.mock_source import MockSource

    cfg = config.youtube

    # ── ローカル1対1チャット（テストモード）───────────────
    # 配信サイトにはつながらず、/chat 画面とマイク入力だけを受け付ける
    if is_local_mode(config, video_id):
        logger.info("Using LocalChatSource (local 1:1 chat mode)")
        return LocalChatSource(user_name=config.local_chat.user_name), None

    if video_id == "mock":
        logger.info("Using MockSource (explicit mock mode)")
        return MockSource(interval_sec=4.0), None
    if not video_id:
        logger.warning("No video_id provided — using MockSource as fallback")
        return MockSource(interval_sec=4.0), None

    primary_name = cfg.primary_source

    # ── ツイキャス ──────────────────────────────────────────
    if primary_name == "twitcasting":
        from backend.sources.twitcasting_source import TwitcastingSource
        tc_cfg = config.twitcasting
        token = tc_cfg.get_token()
        if not token:
            logger.error("TWITCASTING_ACCESS_TOKEN not set")
            return MockSource(interval_sec=4.0), None
        return TwitcastingSource(
            access_token=token,
            user_id=tc_cfg.user_id,
            polling_interval_ms=tc_cfg.polling_interval_ms,
        ), None

    # ── Twitch ─────────────────────────────────────────────
    if primary_name == "twitch":
        from backend.sources.twitch_source import TwitchSource
        tw_cfg = config.twitch
        token = tw_cfg.get_token()
        if not token or not tw_cfg.nick:
            logger.error("Twitch token or nick not set")
            return MockSource(interval_sec=4.0), None
        return TwitchSource(token=token, nick=tw_cfg.nick,
                            channel=tw_cfg.channel or video_id), None

    # ── YouTube ────────────────────────────────────────────
    from backend.sources.pytchat_source import PytchatSource
    from backend.sources.youtube_api_source import YouTubeDataAPISource

    api_key = cfg.youtube_api.api_key
    fallback: CommentSource | None = None
    if primary_name == "pytchat":
        primary: CommentSource = PytchatSource()
        if api_key:
            fallback = YouTubeDataAPISource(api_key)
        else:
            logger.info("No YOUTUBE_API_KEY — fallback source disabled")
    elif not api_key:
        logger.warning("YOUTUBE_API_KEY not set, falling back to pytchat")
        primary = PytchatSource()
    else:
        primary = YouTubeDataAPISource(api_key)
        fallback = PytchatSource()
    return primary, fallback