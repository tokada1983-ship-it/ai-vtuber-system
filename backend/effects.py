"""演出エフェクト（カットイン動画・効果音・フリップ表示）

キャラクターの発話JSONに任意で付く "effect" を検証し、配信画面と
アプリ側音声へ配送する。エフェクトは発話フローの追加経路であり、
検証に失敗したもの・クールダウン中のものは無害に捨てる（発話自体は
通常どおり流れる）。

出力形式（LLMが付ける場合、JSONの末尾に最大1つ）:
    "effect": {"type": "cutin", "name": "<カタログ名>"}
    "effect": {"type": "se",    "name": "<カタログ名>"}
    "effect": {"type": "flip",  "text": "<短い一言>"}      # 自由文
    "effect": {"type": "flip",  "name": "<カタログ名>"}    # カタログ定型文

カタログは config.yaml の effects セクション（キャラごとに上書き可）。
cutin のファイルは models/ 以下に置くと /models/ で静的配信される。
se のファイルはアプリ側の音声出力（AudioRouter、TTSと同じ出力先）で鳴らす。

内心（inner thoughts）:
    発話JSONの末尾に任意で付く "thought"（文字列）を検証し、画面の
    小さい吹き出し表示にだけ配送する。声には絶対に出さない（TTS の
    入力は speech のみで、thought は別チャネル）。

システム発火:
    無音・材料なしの時間帯に、EffectsDispatcher.dispatch_system が
    カタログから軽い演出（SE/フリップ中心）を自動発火する。全体
    クールダウンは自発発火と共有し、さらにシステム発火どうしの
    間隔（system.cooldown_sec）も空ける。
"""
from __future__ import annotations

import asyncio
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable

from backend.config import (AppConfig, CharacterConfig, EffectsConfig,
                            ThoughtsConfig)
from backend.llm.parser import contains_meta_leak
from backend.logger import get_logger

logger = get_logger(__name__)

EFFECT_TYPES = ("cutin", "se", "flip")


@dataclass
class EffectSpec:
    """検証済みエフェクト（配送に必要な情報を解決済み）"""
    type: str                 # cutin / se / flip
    name: str = ""            # カタログ名（flip自由文のときは空）
    text: str = ""            # flip の表示文（解決済み）
    label: str = ""           # カタログの説明（画面ラベル・ログ用）
    file: str = ""            # cutin / se のファイルパス
    url: str = ""             # cutin の配信URL（file から解決）


def resolve_effects(config: AppConfig,
                    character: CharacterConfig | None) -> EffectsConfig:
    """全体設定にキャラごとの上書きを重ねた実効エフェクト設定を返す"""
    base = config.effects
    if character is None:
        return base
    override = character.effects
    if override is None:
        return base
    merged = base.model_copy(deep=True)
    if override.enabled is not None:
        merged.enabled = override.enabled
    if override.cooldown_sec is not None:
        merged.cooldown_sec = override.cooldown_sec
    if override.flip_duration_sec is not None:
        merged.flip_duration_sec = override.flip_duration_sec
    if override.flip_max_chars is not None:
        merged.flip_max_chars = override.flip_max_chars
    merged.cutin.update(override.cutin)
    merged.se.update(override.se)
    merged.flip.update(override.flip)
    return merged


def file_to_url(file: str) -> str:
    """アプリ作業フォルダ相対のファイルパスをブラウザ用URLに直す。

    models/ 以下は /models で静的配信される。それ以外は配信できないので
    空文字を返す（呼び出し側は空なら配送しない）。
    """
    text = (file or "").strip().replace("\\", "/")
    if not text:
        return ""
    if text.startswith(("http://", "https://", "/")):
        return text
    if text.startswith("models/"):
        return "/" + text
    return ""


def validate_effect(raw, resolved: EffectsConfig) -> EffectSpec | None:
    """LLM出力などの effect を検証する。不正なら None（無害に捨てる）。

    - type は cutin / se / flip のいずれか
    - cutin / se は name がカタログに実在し、ファイルが設定されていること
      （ファイル自体の実在は配送時に確認する）
    - flip は text（上限文字数で切り詰め）か、カタログ名による定型文
    """
    if not isinstance(raw, dict):
        return None
    etype = str(raw.get("type", "")).strip().lower()
    if etype not in EFFECT_TYPES:
        return None
    name = str(raw.get("name", "")).strip()

    if etype in ("cutin", "se"):
        catalog = resolved.cutin if etype == "cutin" else resolved.se
        entry = catalog.get(name)
        if entry is None or not entry.file:
            return None
        return EffectSpec(type=etype, name=name, label=entry.label,
                          file=entry.file,
                          url=file_to_url(entry.file) if etype == "cutin" else "")

    # flip
    text = str(raw.get("text", "")).strip()
    entry = resolved.flip.get(name) if name else None
    if not text and entry is not None:
        text = entry.text.strip()
    if not text:
        return None
    limit = max(1, int(resolved.flip_max_chars))
    if len(text) > limit:
        text = text[:limit]
    return EffectSpec(type="flip", name=name if entry else "", text=text,
                      label=entry.label if entry else "")


def catalog_lines(resolved: EffectsConfig) -> list[str]:
    """プロンプトに載せるカタログの説明行（種類・名前・説明）"""
    lines: list[str] = []
    for kind, catalog in (("cutin", resolved.cutin), ("se", resolved.se),
                          ("flip", resolved.flip)):
        for name, entry in catalog.items():
            label = entry.label or (entry.text if kind == "flip" else "")
            lines.append(f"  - {kind}: {name}" + (f"（{label}）" if label else ""))
    return lines


def catalog_summary(resolved: EffectsConfig) -> dict:
    """確認用API向けにカタログを要約する"""
    def entries(kind: str, catalog) -> list[dict]:
        out = []
        for name, entry in catalog.items():
            item = {"name": name, "label": entry.label}
            if kind == "cutin":
                item["url"] = file_to_url(entry.file)
            if kind == "flip":
                item["text"] = entry.text
            out.append(item)
        return out
    return {
        "enabled": resolved.enabled,
        "cooldown_sec": resolved.cooldown_sec,
        "flip_duration_sec": resolved.flip_duration_sec,
        "flip_max_chars": resolved.flip_max_chars,
        "cutin": entries("cutin", resolved.cutin),
        "se": entries("se", resolved.se),
        "flip": entries("flip", resolved.flip),
    }


class EffectsDispatcher:
    """検証済みエフェクトの配送（WebSocket表示＋アプリ側SE再生）とクールダウン。

    クールダウンは自発エフェクト全体で共有する（キャラが交互に連発しない）。
    手動トリガー（manual=True）はクールダウンを無視し、時刻も更新しない
    （動作確認の連打で自発側の間隔を汚さないため）。
    """

    def __init__(self, config: AppConfig, hub,
                 play_wav: Callable[[bytes], Awaitable[None]]):
        self._cfg = config
        self._hub = hub
        self._play_wav = play_wav
        self._last_fired_at = 0.0   # monotonic（自発エフェクトの最終発火）
        self._last_system_fired_at = 0.0  # monotonic（システム発火の最終発火）
        self._se_tasks: set[asyncio.Task] = set()

    async def dispatch(self, raw, agent=None, *, manual: bool = False) -> dict:
        """エフェクトを1件配送する。戻り値は結果の要約（API応答用）。

        agent は CharacterAgent（キャラ別カタログと名札の解決に使う）。
        失敗しても例外は投げず、{"ok": False, "reason": ...} を返す。
        """
        char_cfg = getattr(agent, "cfg", None)
        resolved = resolve_effects(self._cfg, char_cfg)
        if not resolved.enabled:
            return {"ok": False, "reason": "effects disabled"}
        spec = validate_effect(raw, resolved)
        if spec is None:
            logger.info("Effect dropped (invalid)", raw=str(raw)[:120])
            return {"ok": False, "reason": "invalid effect"}

        if not manual and resolved.cooldown_sec > 0:
            elapsed = time.monotonic() - self._last_fired_at
            if elapsed < resolved.cooldown_sec:
                logger.info("Effect dropped (cooldown)", type=spec.type,
                            name=spec.name)
                return {"ok": False, "reason": "cooldown"}

        if spec.type == "cutin":
            if not spec.url or not Path(spec.file).is_file():
                logger.warning("Effect dropped (cutin file missing)",
                               name=spec.name, file=spec.file)
                return {"ok": False, "reason": "cutin file missing"}
            msg = {"type": "effect", "effect_type": "cutin", "name": spec.name,
                   "url": spec.url, "label": spec.label}
        elif spec.type == "flip":
            msg = {"type": "effect", "effect_type": "flip", "name": spec.name,
                   "text": spec.text, "label": spec.label,
                   "duration_sec": float(resolved.flip_duration_sec)}
        else:  # se: 音はアプリ側。画面側には通知だけ（表示は何もしない）
            msg = {"type": "effect", "effect_type": "se", "name": spec.name,
                   "label": spec.label}

        character = getattr(agent, "key", "") or ""
        speaker = getattr(agent, "name", "") or ""
        if character:
            msg["character"] = character
        if speaker:
            msg["speaker"] = speaker

        if not manual:
            self._last_fired_at = time.monotonic()
        await self._hub.broadcast("avatar", msg)
        if spec.type == "se":
            self._play_se(spec)

        logger.info("Effect dispatched", type=spec.type, name=spec.name,
                    character=character or "-", manual=manual)
        return {"ok": True, "type": spec.type, "name": spec.name,
                "text": spec.text}

    async def dispatch_system(self, agent=None, *, reason: str = "") -> dict:
        """システム条件での自動発火（無音・材料なし時間帯の軽い演出）。

        モデルの自発発火とは別経路だが、配送自体は dispatch() を通すため
        全体クールダウン（cooldown_sec）を自発発火と共有し、自発側の
        間隔を破らない（自発が優先）。そのうえで、システム発火どうしの
        間隔（system.cooldown_sec）も空けて連発を防ぐ。

        種別はカタログから選ぶ：カットインは低確率（system.
        cutin_probability）で混ぜ、基本は SE かフリップ定型文。
        """
        char_cfg = getattr(agent, "cfg", None)
        resolved = resolve_effects(self._cfg, char_cfg)
        if not resolved.enabled or not resolved.system.enabled:
            return {"ok": False, "reason": "system effects disabled"}
        if resolved.system.cooldown_sec > 0:
            elapsed = time.monotonic() - self._last_system_fired_at
            if elapsed < resolved.system.cooldown_sec:
                return {"ok": False, "reason": "system cooldown"}
        raw = self._pick_system_effect(resolved)
        if raw is None:
            return {"ok": False, "reason": "no system effect available"}
        result = await self.dispatch(raw, agent=agent)
        if result.get("ok"):
            self._last_system_fired_at = time.monotonic()
            logger.info("System effect fired", reason=reason or "-",
                        type=result.get("type"), name=result.get("name"))
        return result

    def _pick_system_effect(self, resolved: EffectsConfig) -> dict | None:
        """システム発火の候補をカタログから1つ選ぶ（実在エントリのみ）"""
        cutins = [n for n, e in resolved.cutin.items() if e.file]
        ses = [n for n, e in resolved.se.items() if e.file]
        flips = [n for n, e in resolved.flip.items() if e.text]
        prob = max(0.0, min(1.0, float(resolved.system.cutin_probability)))
        if cutins and random.random() < prob:
            return {"type": "cutin", "name": random.choice(cutins)}
        pool = ([{"type": "se", "name": n} for n in ses]
                + [{"type": "flip", "name": n} for n in flips])
        if pool:
            return random.choice(pool)
        if cutins:
            return {"type": "cutin", "name": random.choice(cutins)}
        return None

    def _play_se(self, spec: EffectSpec) -> None:
        """SEをTTSと同じ音声出力で再生する（発話を待たせないよう背景実行）"""
        path = Path(spec.file)
        if not path.is_file():
            logger.warning("Effect SE file missing", name=spec.name,
                           file=spec.file)
            return

        async def run() -> None:
            try:
                data = await asyncio.to_thread(path.read_bytes)
                await self._play_wav(data)
            except Exception as e:
                logger.warning("Effect SE play failed", error=str(e))

        task = asyncio.create_task(run())
        self._se_tasks.add(task)
        task.add_done_callback(self._se_tasks.discard)
# ── 内心（inner thoughts） ────────────────────────────────────

_THOUGHT_TAG_RE = re.compile(
    r"<\s*(emotion|motion|speech)\s*:|<\/?(think|thought|reasoning)>")


def resolve_thoughts(config: AppConfig,
                     character: CharacterConfig | None) -> ThoughtsConfig:
    """全体設定にキャラごとの上書きを重ねた実効の内心設定を返す"""
    base = config.thoughts
    if character is None:
        return base
    override = character.thoughts
    if override is None:
        return base
    merged = base.model_copy(deep=True)
    if override.enabled is not None:
        merged.enabled = override.enabled
    if override.max_chars is not None:
        merged.max_chars = override.max_chars
    if override.cooldown_sec is not None:
        merged.cooldown_sec = override.cooldown_sec
    return merged


def validate_thought(raw, resolved: ThoughtsConfig) -> str:
    """内心テキストを検証する。不正なら ""（無害に捨てる）。

    - 文字列でない・空白だけ・制御タグや推論漏れを含むものは捨てる
    - 改行は空白に畳み、上限文字数で切り詰める
    """
    if not isinstance(raw, str):
        return ""
    text = re.sub(r"\s+", " ", raw).strip()
    if not text:
        return ""
    if _THOUGHT_TAG_RE.search(text) or contains_meta_leak(text):
        return ""
    limit = max(1, int(resolved.max_chars))
    if len(text) > limit:
        text = text[:limit]
    return text


class ThoughtsDispatcher:
    """内心（thought）の表示配送とクールダウン。

    thought は発話とは別チャネルで、TTS には一切流さない。画面側の
    小さい吹き出し表示にだけ WebSocket で届ける。クールダウンは
    thought 専用（effect の全体クールダウンとは別時計）で、キャラが
    交互に出す場合も共有する。手動（manual=True）はクールダウンを
    無視し、時刻も更新しない。
    """

    def __init__(self, config: AppConfig, hub):
        self._cfg = config
        self._hub = hub
        self._last_shown_at = 0.0   # monotonic（内心の最終表示）

    async def dispatch(self, raw, agent=None, *, speech: str = "",
                       manual: bool = False) -> dict:
        """内心を1件表示する。戻り値は結果の要約。

        speech が渡された場合、それと実質同じ文は「発話の繰り返し」
        として捨てる（内心は本音・感想であるべき、という設計）。
        失敗しても例外は投げず、{"ok": False, "reason": ...} を返す。
        """
        char_cfg = getattr(agent, "cfg", None)
        resolved = resolve_thoughts(self._cfg, char_cfg)
        if not resolved.enabled:
            return {"ok": False, "reason": "thoughts disabled"}
        text = validate_thought(raw, resolved)
        if not text:
            logger.info("Thought dropped (invalid)", raw=str(raw)[:80])
            return {"ok": False, "reason": "invalid thought"}
        if speech:
            def norm(s: str) -> str:
                return re.sub(r"[\s、。，．！？!?]", "", s or "")
            if norm(text) and norm(text) in norm(speech):
                logger.info("Thought dropped (duplicates speech)",
                            text=text[:40])
                return {"ok": False, "reason": "duplicates speech"}

        if not manual and resolved.cooldown_sec > 0:
            elapsed = time.monotonic() - self._last_shown_at
            if elapsed < resolved.cooldown_sec:
                return {"ok": False, "reason": "cooldown"}

        msg = {"type": "thought", "text": text}
        character = getattr(agent, "key", "") or ""
        speaker = getattr(agent, "name", "") or ""
        if character:
            msg["character"] = character
        if speaker:
            msg["speaker"] = speaker

        if not manual:
            self._last_shown_at = time.monotonic()
        await self._hub.broadcast("avatar", msg)
        logger.info("Thought shown", text=text[:40],
                    character=character or "-", manual=manual)
        return {"ok": True, "text": text}
