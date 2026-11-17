"""GeminiBackend: Google Gemini API の OpenAI互換エンドポイントを使うLLMバックエンド

Gemini は OpenAI 互換の Chat Completions API を公開しているので、既存の
OpenAICompatBackend をそのまま使い、差分だけをここに閉じ込める:

  - 思考（thinking）はデフォルトでオフにする。発話は40字以内の短いJSONで、
    思考トークンは「出力」として課金されるため、付けるとコストが数倍になる。
    config の llm.backends.gemini.extra_payload で上書きできる。
    指定は thinking_config だけで行うこと。reasoning_effort と同時に入れると
    Gemini が 400（"Expected one of either..."）を返す（2026-11-11 実測）。
    さらに実測では、gemini-3.5-flash-lite は thinking_budget=0 も
    reasoning_effort="none" も単独で 400（invalid argument）になり、
    受理されるのは thinking_level="minimal" のみだった。そのため既定は minimal。
    thinking_budget を受理するモデル（2.5系など）では extra_payload の
    thinking_config で上書きできる（例: {"thinking_budget": 0}）。
  - 互換レイヤー経由だと Gemini 側の思考フィールド（reasoning_content 等）が
    delta に混ざる場合があるため、テキスト抽出は content を優先する
    （親クラスの _extract_text / SSE パーサの既定動作で足りる）。

課金プロジェクトの API キーを .env の GEMINI_API_KEY に置く想定。
無料枠プロジェクトのキーと混ぜないこと（有料Tier化して無料枠運用が崩れる）。
"""
from __future__ import annotations

from backend.llm.openai_compat import OpenAICompatBackend
from backend.models import ChatMessage


class GeminiBackend(OpenAICompatBackend):

    backend_name = "gemini"
    default_base_url = "https://generativelanguage.googleapis.com/v1beta/openai"
    default_model = "gemini-3.5-flash-lite"
    api_key_env = "GEMINI_API_KEY"
    requires_api_key = True
    timeout_sec = 60.0

    def _build_payload(self, system_prompt: str, messages: list[ChatMessage],
                       stream: bool, temperature: float | None = None) -> dict:
        payload = super()._build_payload(system_prompt, messages, stream,
                                         temperature=temperature)
        # 思考の抑制は thinking_config だけで指定する。
        # reasoning_effort と両方入れると Gemini が 400 を返す
        # （"Expected one of either reasoning_effort or custom thinking_config"）。
        # また 3.5 系は thinking_budget=0 を単独でも受理しない（400）ため、
        # 既定は受理される最小値 thinking_level="minimal" にする。
        # extra_payload で reasoning_effort が明示された場合はそちらを優先し、
        # thinking_config は付けない。thinking_config 自体の上書きも可能。
        extra = dict(payload.get("extra_body") or {})
        gcfg = dict(extra.get("google") or {})
        if "reasoning_effort" in payload:
            gcfg.pop("thinking_config", None)
        else:
            gcfg.setdefault("thinking_config", {"thinking_level": "minimal"})
        if gcfg:
            extra["google"] = gcfg
        else:
            extra.pop("google", None)
        if extra:
            payload["extra_body"] = extra
        else:
            payload.pop("extra_body", None)
        return payload
