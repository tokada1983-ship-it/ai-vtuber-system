"""共通データモデル"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class Emotion(str, Enum):
    NEUTRAL = "neutral"
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    SURPRISED = "surprised"
    RELAXED = "relaxed"


class Motion(str, Enum):
    IDLE = "idle"
    NOD = "nod"
    WAVE = "wave"
    TILT = "tilt"
    JUMP = "jump"
    BOW = "bow"
    SHAKE_HEAD = "shake_head"
    THINK = "think"
    EXCITED = "excited"
    SHY = "shy"
    POINT = "point"
    CLAP = "clap"
    STRETCH = "stretch"
    LEAN_FORWARD = "lean_forward"
    DANCE = "dance"
    NONE = "none"


@dataclass
class Comment:
    id: str
    author: str
    author_channel_id: str
    message: str
    timestamp: datetime
    is_superchat: bool = False
    superchat_amount: int | None = None
    is_member: bool = False
    is_moderator: bool = False
    priority: float = 0.0  # CommentSelectorが付与


@dataclass
class ChatMessage:
    """LLMへの入力用会話履歴エントリ"""
    role: str  # "user" | "assistant"
    content: str
    comment_id: str | None = None


@dataclass
class LLMResponse:
    speech: str
    emotion: Emotion = Emotion.NEUTRAL
    motion: Motion = Motion.IDLE
    target_comment_id: str | None = None
    skip: bool = False
    raw: dict = field(default_factory=dict)


@dataclass
class TTSResult:
    audio_wav: bytes
    duration_ms: int
    phonemes: list | None = None  # 将来のリップシンク高精度化用


@dataclass
class SystemStatus:
    """Dashboardに流すシステム状態"""
    comment_source: str = "disconnected"
    llm_backend: str = "unknown"
    tts_backend: str = "unknown"
    is_speaking: bool = False
    current_emotion: Emotion = Emotion.NEUTRAL
    queue_size: int = 0
    last_comment: str = ""
    last_speech: str = ""
    video_id: str = ""
    viewer_count: int = 0
    uptime_sec: float = 0.0
