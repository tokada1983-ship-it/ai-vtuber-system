"""LMStudioBackend: LM Studio の OpenAI互換エンドポイントを使ったLLMバックエンド"""
from __future__ import annotations

from backend.llm.openai_compat import OpenAICompatBackend
from backend.logger import get_logger
from backend.models import ChatMessage

logger = get_logger(__name__)


class LMStudioBackend(OpenAICompatBackend):

    backend_name = "lmstudio"
    default_base_url = "http://localhost:1234/v1"

    def _build_payload(self, system_prompt: str, messages: list[ChatMessage],
                       stream: bool, temperature: float | None = None) -> dict:
        payload = super()._build_payload(system_prompt, messages, stream,
                                         temperature=temperature)
        api_messages = payload["messages"]
        # 履歴に user が1件も無いターン（起動直後の独り言など）は、
        # bonsai のチャットテンプレートが "No user query found" で落ちる。
        # 合成の user メッセージを末尾に足して回避する。
        if not any(m.get("role") == "user" for m in api_messages):
            api_messages.append({
                "role": "user",
                "content": "（いま視聴者コメントは無い。自分から話題を振って、自由に話して）",
            })
        # qwen3系は思考モードをオフにしないとJSONが崩れる
        # /no_think を最後のユーザーメッセージに付与する
        for i in range(len(api_messages) - 1, -1, -1):
            if api_messages[i]["role"] == "user":
                if "/no_think" not in api_messages[i]["content"]:
                    api_messages[i]["content"] += " /no_think"
                break
        # Reasoningモデルの思考モードを無効化 (LM Studio 0.3.6+)
        payload["reasoning_effort"] = "none"
        return payload

    def _extract_text(self, data: dict) -> str:
        message = data["choices"][0]["message"]
        # content が空の場合は reasoning_content を使う
        raw_text = message.get("content") or ""
        if not raw_text.strip():
            raw_text = message.get("reasoning_content") or ""
            if raw_text:
                logger.info("LMStudio: using reasoning_content (content was empty)")
        return raw_text
