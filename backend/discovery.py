"""ローカルAIサービスの自動検出と設定の突き合わせ

ローカル運用では「LM Studioを起動し忘れた」「config.yamlのvoice名がサーバに無い」
といった理由で無言になることが多く、しかもログを見ないと原因が分からない。

このモジュールは既知のローカルサービスを一斉に叩いて、
  - 起動しているか
  - どんなモデル/ボイスが使えるか
  - 今のconfig.yamlの指定がその中に実在するか
を返す。ダッシュボードから一目で分かるようにするのが目的。
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

import httpx

from backend.config import AppConfig
from backend.logger import get_logger

logger = get_logger(__name__)

_PROBE_TIMEOUT = 2.5


@dataclass
class ProviderStatus:
    key: str                       # "lmstudio" / "voicevox" ...
    kind: str                      # "llm" | "tts" | "stt"
    label: str                     # 画面表示名
    base_url: str
    running: bool = False
    items: list[str] = field(default_factory=list)   # 利用可能なモデル/ボイス
    detail: str = ""

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "kind": self.kind,
            "label": self.label,
            "base_url": self.base_url,
            "running": self.running,
            "items": self.items,
            "detail": self.detail,
        }


@dataclass
class ConfigIssue:
    severity: str   # "error" | "warn"
    target: str     # "llm.primary" など
    message: str
    hint: str = ""

    def to_dict(self) -> dict:
        return {
            "severity": self.severity,
            "target": self.target,
            "message": self.message,
            "hint": self.hint,
        }


# ── 個別プローブ ──────────────────────────────────────────────

async def _probe_openai_models(client: httpx.AsyncClient, base_url: str) -> list[str]:
    """OpenAI互換 /models からモデルID一覧を取る（LM Studio・Irodori等）"""
    resp = await client.get(f"{base_url.rstrip('/')}/models")
    resp.raise_for_status()
    data = resp.json().get("data", [])
    return [m.get("id", "") for m in data if m.get("id")]


async def _probe_ollama(client: httpx.AsyncClient, base_url: str) -> list[str]:
    resp = await client.get(f"{base_url.rstrip('/')}/api/tags")
    resp.raise_for_status()
    models = resp.json().get("models", [])
    return [m.get("name", "") for m in models if m.get("name")]


async def _probe_voicevox(client: httpx.AsyncClient, base_url: str) -> list[str]:
    """VOICEVOX / AivisSpeech の話者一覧（"名前:スタイル (id)" 形式）"""
    resp = await client.get(f"{base_url.rstrip('/')}/speakers")
    resp.raise_for_status()
    out: list[str] = []
    for spk in resp.json():
        name = spk.get("name", "?")
        for style in spk.get("styles", []):
            out.append(f"{name}:{style.get('name', '')} ({style.get('id')})")
    return out


async def _probe_irodori_voices(client: httpx.AsyncClient, base_url: str) -> list[str]:
    resp = await client.get(f"{base_url.rstrip('/')}/audio/voices")
    resp.raise_for_status()
    data = resp.json().get("data", [])
    return [v.get("id", "") for v in data if v.get("id")]


async def _safe_probe(
    key: str, kind: str, label: str, base_url: str, fn
) -> ProviderStatus:
    """プローブ1件。落ちているだけなら running=False を返し、例外は投げない。"""
    status = ProviderStatus(key=key, kind=kind, label=label, base_url=base_url)
    if not base_url:
        status.detail = "base_url未設定"
        return status
    try:
        async with httpx.AsyncClient(timeout=_PROBE_TIMEOUT) as client:
            status.items = await fn(client, base_url)
            status.running = True
    except (httpx.ConnectError, httpx.ConnectTimeout):
        status.detail = "未起動（接続できません）"
    except httpx.HTTPStatusError as e:
        # 起動はしているがエンドポイントが違う
        status.running = True
        status.detail = f"応答あり（HTTP {e.response.status_code}）"
    except Exception as e:
        status.detail = f"{type(e).__name__}: {e}"
    return status


# ── まとめて検出 ──────────────────────────────────────────────

async def discover_providers(cfg: AppConfig) -> list[ProviderStatus]:
    """設定に書かれているローカルサービスを並列にプローブする"""
    llm = cfg.llm.backends
    tts = cfg.tts.backends

    def url_of(table: dict, name: str, default: str) -> str:
        be = table.get(name)
        return (be.base_url if be and be.base_url else default)

    probes = [
        _safe_probe("lmstudio", "llm", "LM Studio",
                    url_of(llm, "lmstudio", "http://localhost:1234/v1"),
                    _probe_openai_models),
        _safe_probe("ollama", "llm", "Ollama",
                    url_of(llm, "ollama", "http://localhost:11434"),
                    _probe_ollama),
        _safe_probe("voicevox", "tts", "VOICEVOX",
                    url_of(tts, "voicevox", "http://localhost:50021"),
                    _probe_voicevox),
        _safe_probe("aivisspeech", "tts", "AivisSpeech",
                    "http://localhost:10101",
                    _probe_voicevox),
        _safe_probe("irodori", "tts", "Irodori TTS",
                    url_of(tts, "irodori", "http://localhost:8088/v1"),
                    _probe_irodori_voices),
    ]
    return list(await asyncio.gather(*probes))


def check_config(cfg: AppConfig, providers: list[ProviderStatus]) -> list[ConfigIssue]:
    """
    検出結果と config.yaml を突き合わせて、設定ミスを洗い出す。
    「起動しているのに設定値が実在しない」ケースを重点的に見る。
    """
    issues: list[ConfigIssue] = []
    by_key = {p.key: p for p in providers}

    # ── LLM ────────────────────────────────────────────────
    primary = cfg.llm.primary
    llm_provider = by_key.get(primary)
    if llm_provider is None:
        issues.append(ConfigIssue(
            "warn", "llm.primary",
            f"LLMバックエンド「{primary}」は自動検出の対象外です",
        ))
    elif not llm_provider.running:
        fallback = cfg.llm.fallback
        fb = by_key.get(fallback)
        hint = (f"フォールバック先の {fallback} も未起動です"
                if fb and not fb.running else
                f"{fallback} にフォールバックします")
        issues.append(ConfigIssue(
            "error", "llm.primary",
            f"{llm_provider.label} が起動していません（{llm_provider.base_url}）",
            hint,
        ))
    else:
        model = cfg.llm.backends.get(primary).model if cfg.llm.backends.get(primary) else ""
        if model and model != "auto" and llm_provider.items and model not in llm_provider.items:
            issues.append(ConfigIssue(
                "error", f"llm.backends.{primary}.model",
                f"モデル「{model}」が {llm_provider.label} に見つかりません",
                f"利用可能: {', '.join(llm_provider.items[:5])}",
            ))

    # ── TTS ────────────────────────────────────────────────
    tts_primary = cfg.tts.primary
    tts_provider = by_key.get(tts_primary)
    tts_cfg = cfg.tts.backends.get(tts_primary)
    if tts_provider is None:
        issues.append(ConfigIssue(
            "warn", "tts.primary",
            f"TTSバックエンド「{tts_primary}」は自動検出の対象外です",
        ))
    elif not tts_provider.running:
        issues.append(ConfigIssue(
            "error", "tts.primary",
            f"{tts_provider.label} が起動していません（{tts_provider.base_url}）",
            "音声なしで字幕のみになります",
        ))
    elif tts_primary == "irodori" and tts_cfg and tts_provider.items:
        voice = tts_cfg.voice
        if voice and voice not in tts_provider.items:
            issues.append(ConfigIssue(
                "error", "tts.backends.irodori.voice",
                f"ボイス「{voice}」が Irodori TTS に登録されていません",
                f"利用可能: {', '.join(tts_provider.items)}",
            ))

    # ── STT ────────────────────────────────────────────────
    if cfg.stt.enabled:
        try:
            import faster_whisper  # noqa: F401
        except ImportError:
            issues.append(ConfigIssue(
                "error", "stt.enabled",
                "音声認識が有効ですが faster-whisper がインストールされていません",
                "pip install faster-whisper",
            ))

    return issues


async def diagnose(cfg: AppConfig) -> dict:
    """ダッシュボード用のまとめ結果を返す"""
    providers = await discover_providers(cfg)
    issues = check_config(cfg, providers)
    return {
        "providers": [p.to_dict() for p in providers],
        "issues": [i.to_dict() for i in issues],
        "ok": not any(i.severity == "error" for i in issues),
    }
