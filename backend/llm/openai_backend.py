"""OpenAIBackend: OpenAI Chat Completions API を使ったLLMバックエンド"""
from __future__ import annotations

from backend.llm.openai_compat import OpenAICompatBackend


class OpenAIBackend(OpenAICompatBackend):

    backend_name = "openai"
    default_base_url = "https://api.openai.com/v1"
    default_model = "gpt-4o-mini"
    api_key_env = "OPENAI_API_KEY"
    requires_api_key = True
    timeout_sec = 60.0
