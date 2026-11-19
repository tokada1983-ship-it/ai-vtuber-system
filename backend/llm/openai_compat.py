"""OpenAI互換 Chat Completions API を話すLLMバックエンドの共通実装

LM Studio / OpenAI / llama.cpp server / vLLM は同じAPI形式なので、
リクエスト組み立て・エラー処理・ストリーム読み取りをここに集約する。

以前はバックエンドごとに毎リクエスト httpx.AsyncClient を作り直しており、
1発話ごとにTCP接続を張り直していた。インスタンスごとに1本の接続を使い回すことで
ローカルLLMでも数十ms、クラウドAPIではTLSハンドシェイクぶんの待ちが消える。
"""
from __future__ import annotations

import os
from typing import AsyncIterator

import httpx

from backend.config import LLMBackendConfig
from backend.llm.base import LLMBackend
from backend.llm.parser import parse_llm_output, parse_tag_output
from backend.llm.sse import iter_openai_sse
from backend.logger import get_logger
from backend.models import ChatMessage, LLMResponse

logger = get_logger(__name__)


class OpenAICompatBackend(LLMBackend):
    """OpenAI互換APIの汎用バックエンド（llama.cpp server / vLLM などはこれをそのまま使う）"""

    backend_name = "openai_compat"
    default_base_url = "http://localhost:8080/v1"
    default_model = ""
    api_key_env = ""            # 空なら認証ヘッダを付けない
    requires_api_key = False
    timeout_sec = 120.0

    def __init__(self, cfg: LLMBackendConfig):
        self._cfg = cfg
        self._healthy = True
        self._client: httpx.AsyncClient | None = None

    # ── 接続 ──────────────────────────────────────────────────

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout_sec, connect=10.0),
                limits=httpx.Limits(max_keepalive_connections=4, keepalive_expiry=60.0),
            )
        return self._client

    async def aclose(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
        self._client = None

    def _api_key(self) -> str:
        env = self._cfg.api_key_env or self.api_key_env
        return os.getenv(env, "") if env else ""

    def _headers(self) -> dict:
        key = self._api_key()
        return {"Authorization": f"Bearer {key}"} if key else {}

    @property
    def _url(self) -> str:
        base = (self._cfg.base_url or self.default_base_url).rstrip("/")
        return f"{base}/chat/completions"

    # ── リクエスト組み立て（サブクラスで調整する） ────────────────

    def _build_payload(self, system_prompt: str, messages: list[ChatMessage],
                       stream: bool, temperature: float | None = None) -> dict:
        api_messages = [{"role": "system", "content": system_prompt}]
        api_messages += [{"role": m.role, "content": m.content} for m in messages]
        payload: dict = {
            "messages": api_messages,
            "temperature": self._cfg.temperature
            if temperature is None else temperature,
            "max_tokens": self._cfg.max_tokens,
            "stream": stream,
        }
        model = self._cfg.model or self.default_model
        if model and model != "auto":
            payload["model"] = model
        # API固有の追加パラメータ（config の extra_payload が最優先で上書きする）
        for key, value in (self._cfg.extra_payload or {}).items():
            payload[key] = value
        return payload

    def _extract_text(self, data: dict) -> str:
        return data["choices"][0]["message"].get("content") or ""

    # ── 一括生成 ──────────────────────────────────────────────

    async def generate(self, system_prompt: str, messages: list[ChatMessage],
                       *, route: str = "respond",
                       temperature: float | None = None) -> LLMResponse:
        if self.requires_api_key and not self._api_key():
            logger.error(f"{self.name}: API key not set", env=self.api_key_env)
            return LLMResponse(speech="", skip=True)

        payload = self._build_payload(system_prompt, messages, stream=False,
                                      temperature=temperature)
        try:
            resp = await self._get_client().post(self._url, json=payload, headers=self._headers())
            resp.raise_for_status()
            data = resp.json()
            raw_text = self._extract_text(data)
            self._healthy = True
            if not raw_text.strip():
                # ローカルLLMの空応答はコンテキスト超過など一時的なことが多い
                usage = data.get("usage", {})
                logger.warning(f"{self.name}: empty response",
                               prompt_tokens=usage.get("prompt_tokens", "?"),
                               finish_reason=data["choices"][0].get("finish_reason", "?"))
                return LLMResponse(speech="", skip=True)
            result = parse_tag_output(raw_text) \
                if self.protocol == "tag" else parse_llm_output(raw_text)
            logger.info(f"{self.name} response",
                        speech=result.speech[:40] if result.speech else "skip",
                        emotion=result.emotion.value if not result.skip else "-")
            return result
        except httpx.ConnectError as e:
            logger.warning(f"{self.name}: connection refused — is the server running?",
                           url=self._url, detail=str(e))
            self._healthy = False
        except httpx.TimeoutException:
            # 一時的な遅延（モデルのロード中など）の可能性があるので healthy は下げない
            logger.warning(f"{self.name}: request timed out", timeout_sec=self.timeout_sec)
        except httpx.HTTPStatusError as e:
            logger.warning(f"{self.name} HTTP error", status=e.response.status_code,
                           body=e.response.text[:300])
            self._healthy = False
        except (KeyError, IndexError) as e:
            logger.warning(f"{self.name}: unexpected response format", missing_key=str(e))
            self._healthy = False
        except Exception as e:
            logger.warning(f"{self.name} error", error_type=type(e).__name__, error=repr(e))
            self._healthy = False
        return LLMResponse(speech="", skip=True)

    # ── ストリーミング生成 ────────────────────────────────────

    async def generate_stream(self, system_prompt: str,
                              messages: list[ChatMessage],
                              *, route: str = "respond",
                              temperature: float | None = None) -> AsyncIterator[str]:
        if self.requires_api_key and not self._api_key():
            logger.error(f"{self.name}: API key not set", env=self.api_key_env)
            return
        payload = self._build_payload(system_prompt, messages, stream=True,
                                      temperature=temperature)
        try:
            async with self._get_client().stream("POST", self._url, json=payload,
                                                 headers=self._headers()) as resp:
                resp.raise_for_status()
                self._healthy = True
                async for delta in iter_openai_sse(resp):
                    yield delta
        except httpx.ConnectError:
            logger.warning(f"{self.name} stream: connection refused", url=self._url)
            self._healthy = False
        except httpx.TimeoutException:
            logger.warning(f"{self.name} stream: timed out")
        except httpx.HTTPStatusError as e:
            logger.warning(f"{self.name} stream HTTP error", status=e.response.status_code)
            self._healthy = False
        except Exception as e:
            logger.warning(f"{self.name} stream error", error_type=type(e).__name__, error=repr(e))
            self._healthy = False

    @property
    def supports_streaming(self) -> bool:
        return True

    @property
    def name(self) -> str:
        return self.backend_name

    @property
    def is_healthy(self) -> bool:
        return self._healthy
