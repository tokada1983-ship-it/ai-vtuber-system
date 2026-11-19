"""ClaudeBackend: Anthropic Claude API を使ったLLMバックエンド (オプション)"""
from __future__ import annotations

import os

from backend.config import LLMBackendConfig
from backend.llm.base import LLMBackend
from backend.llm.parser import parse_llm_output, parse_tag_output
from backend.logger import get_logger
from backend.models import ChatMessage, LLMResponse

logger = get_logger(__name__)


class ClaudeBackend(LLMBackend):

    def __init__(self, cfg: LLMBackendConfig):
        self._cfg = cfg
        self._healthy = True

    async def generate(
        self,
        system_prompt: str,
        messages: list[ChatMessage],
        *,
        route: str = "respond",
        temperature: float | None = None,
    ) -> LLMResponse:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            logger.error("ANTHROPIC_API_KEY not set")
            return LLMResponse(speech="", skip=True)

        try:
            from anthropic import AsyncAnthropic
            client = AsyncAnthropic(api_key=api_key)

            api_messages = [
                {"role": m.role, "content": m.content}
                for m in messages
            ]

            create_kwargs: dict = {}
            if temperature is not None:
                create_kwargs["temperature"] = temperature
            response = await client.messages.create(
                model=self._cfg.model or "claude-sonnet-4-20250514",
                max_tokens=self._cfg.max_tokens,
                system=system_prompt,
                messages=api_messages,
                **create_kwargs,
            )
            raw_text = response.content[0].text
            self._healthy = True
            return parse_tag_output(raw_text) \
                if self.protocol == "tag" else parse_llm_output(raw_text)

        except Exception as e:
            logger.warning("Claude backend error", error=str(e))
            self._healthy = False
            return LLMResponse(speech="", skip=True)

    @property
    def name(self) -> str:
        return "claude"

    @property
    def is_healthy(self) -> bool:
        return self._healthy
