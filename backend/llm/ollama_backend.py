"""OllamaBackend: Ollama を使ったLLMバックエンド"""
from __future__ import annotations

from typing import AsyncIterator

import httpx

from backend.config import LLMBackendConfig
from backend.llm import ollama_service
from backend.llm.base import LLMBackend
from backend.llm.parser import parse_llm_output
from backend.llm.sse import iter_ndjson
from backend.logger import get_logger
from backend.models import ChatMessage, LLMResponse

logger = get_logger(__name__)


class OllamaBackend(LLMBackend):

    def __init__(self, cfg: LLMBackendConfig):
        self._cfg = cfg
        self._healthy = True
        self._client: httpx.AsyncClient | None = None

    @property
    def service(self) -> ollama_service.OllamaService:
        return ollama_service.get_service(self._cfg.base_url)

    async def _revive(self) -> bool:
        """接続できなかったとき、Ollama を起動してやり直せるようにする"""
        ok = await self.service.ensure_running()
        if ok:
            logger.info("Ollama was not running — started it, retrying")
        return ok

    async def warmup(self) -> bool:
        """モデルを先にVRAMへ読み込んでおく（1発話目のロード待ちをなくす）"""
        return await self.service.ensure_model(self._cfg.model)

    async def probe(self) -> bool:
        """回復確認用の最小生成プローブ（failback 判定用。Phase 4）。

        設定中モデルへ num_predict=1 の短い生成を1回だけ投げ、応答が
        返れば健全とみなして _healthy を戻す。サーバ生存確認（/api/tags）
        だけでは「モデルが実際に生成できるか」までは分からないため、
        実生成で確認する。通常の発話経路には影響しない（結果は捨てる）。
        """
        payload: dict = {
            "model": self._cfg.model,
            "messages": [{"role": "user", "content": "ping"}],
            "stream": False,
            "keep_alive": self.service.keep_alive,
            "options": {"temperature": 0.0, "num_predict": 1},
        }
        extra = self._cfg.extra_payload or {}
        if "think" in extra:
            payload["think"] = extra["think"]
        try:
            resp = await self._get_client().post(
                self._url, json=payload,
                timeout=httpx.Timeout(20.0, connect=5.0))
            resp.raise_for_status()
            data = resp.json()
            ok = isinstance(data.get("message"), dict)
        except Exception as e:
            logger.debug("Ollama probe failed", error=str(e))
            ok = False
        self._healthy = ok
        return ok

    def _get_client(self) -> httpx.AsyncClient:
        # 接続を使い回す（毎リクエストでTCPを張り直さない）
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=10.0))
        return self._client

    async def aclose(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
        self._client = None

    def _build_payload(
        self,
        system_prompt: str,
        messages: list[ChatMessage],
        stream: bool,
        temperature: float | None = None,
    ) -> dict:
        api_messages = [{"role": "system", "content": system_prompt}]
        for m in messages:
            api_messages.append({"role": m.role, "content": m.content})
        payload = {
            "model": self._cfg.model,
            "messages": api_messages,
            "stream": stream,
            # 配信中にモデルがアンロードされて毎回ロード待ちになるのを防ぐ
            "keep_alive": self.service.keep_alive,
            "options": {
                "temperature": self._cfg.temperature
                if temperature is None else temperature,
                "num_predict": self._cfg.max_tokens,
            },
        }
        # Gemma 4 など思考モード既定ONのモデルは、付けっぱなしだと発話までの
        # 遅延とJSON崩れの原因になる（2026-11-11 実測: 思考ONで約7.9秒）。
        # extra_payload に "think": false が書かれたバックエンドだけトップレベルへ移す。
        extra = self._cfg.extra_payload or {}
        if "think" in extra:
            payload["think"] = extra["think"]
        # JSONモード（文法制約でJSON以外の出力を抑える）。think と同じ
        # extra_payload 伝搬で、config に "format": "json" と書かれた
        # バックエンドだけ有効になる（既定は無指定＝OFF。Phase 4）
        if "format" in extra:
            payload["format"] = extra["format"]
        return payload

    @property
    def _url(self) -> str:
        return f"{self._cfg.base_url}/api/chat"

    async def generate(
        self,
        system_prompt: str,
        messages: list[ChatMessage],
        *,
        route: str = "respond",
        temperature: float | None = None,
    ) -> LLMResponse:
        payload = self._build_payload(system_prompt, messages, stream=False,
                                      temperature=temperature)
        url = self._url

        try:
            try:
                resp = await self._get_client().post(url, json=payload)
            except httpx.ConnectError:
                if not await self._revive():
                    raise
                resp = await self._get_client().post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
            raw_text = data["message"]["content"]
            self._healthy = True
            result = parse_llm_output(raw_text)
            logger.debug("Ollama response", speech=result.speech[:40] if result.speech else "skip")
            return result

        except httpx.ConnectError:
            logger.warning("Ollama: connection refused — is Ollama running?")
            self._healthy = False
            return LLMResponse(speech="", skip=True)
        except httpx.TimeoutException:
            logger.warning("Ollama: request timed out — model may be loading")
            return LLMResponse(speech="", skip=True)
        except Exception as e:
            logger.warning("Ollama error", error=str(e))
            self._healthy = False
            return LLMResponse(speech="", skip=True)

    async def generate_stream(
        self,
        system_prompt: str,
        messages: list[ChatMessage],
        *,
        route: str = "respond",
        temperature: float | None = None,
    ) -> AsyncIterator[str]:
        payload = self._build_payload(system_prompt, messages, stream=True,
                                      temperature=temperature)

        for attempt in range(2):
            try:
                async with self._get_client().stream("POST", self._url, json=payload) as resp:
                    resp.raise_for_status()
                    self._healthy = True
                    async for delta in iter_ndjson(resp, key_path=("message", "content")):
                        yield delta
                return
            except httpx.ConnectError:
                # 1文字も返す前の接続失敗なら、起動してもう1回だけ試す
                if attempt == 0 and await self._revive():
                    continue
                logger.warning("Ollama stream: connection refused — is Ollama running?")
                self._healthy = False
                return
            except Exception as e:
                logger.warning("Ollama stream error", error=str(e))
                self._healthy = False
                return

    @property
    def supports_streaming(self) -> bool:
        return True

    @property
    def name(self) -> str:
        return "ollama"

    @property
    def is_healthy(self) -> bool:
        return self._healthy
