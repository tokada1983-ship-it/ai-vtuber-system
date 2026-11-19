"""LLMBackend インターフェース"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import AsyncIterator

from backend.llm.parser import render_tag_output
from backend.models import ChatMessage, LLMResponse


class LLMBackend(ABC):
    """
    route: 応答の種別（"respond"=即応が要る応答 / "idle"=待機系）。
    ハイブリッド振り分け（HybridRouter）だけが使う。他のバックエンドは無視してよい。
    """

    # 出力プロトコル（"json" | "tag"）。factory が config.llm.protocol を
    # スタンプする。バックエンド側のパーサ選択と、Ollama の format:json
    # 抑制に使う（Phase 5 Stage D）
    protocol: str = "json"

    @abstractmethod
    async def generate(
        self,
        system_prompt: str,
        messages: list[ChatMessage],
        *,
        route: str = "respond",
        temperature: float | None = None,
    ) -> LLMResponse:
        """応答生成。必ずLLMResponseを返す。失敗時はskip=Trueで返す。

        temperature: 生成温度の呼び出し単位の上書き（None＝バックエンド設定値
        のまま）。反復ブロック回復リトライなど、条件を変えて再生成したい
        場面だけが渡す（Phase 4）。
        """

    async def generate_stream(
        self,
        system_prompt: str,
        messages: list[ChatMessage],
        *,
        route: str = "respond",
        temperature: float | None = None,
    ) -> AsyncIterator[str]:
        """
        生成中のテキスト断片を順にyieldする。
        SpeechStreamParser に食わせることで、全文完成を待たずに発話を始められる。

        ストリーミング非対応のバックエンドはこのデフォルト実装が使われ、
        generate() の結果を一括で1回だけyieldする（呼び出し側は同じコードで動く）。
        """
        result = await self.generate(system_prompt, messages, route=route,
                                     temperature=temperature)
        if result.raw:
            if isinstance(result.raw, dict) \
                    and result.raw.get("protocol") == "tag":
                # タグ方式では下流の TagSpeechStreamParser が読める形で流す
                yield render_tag_output(result)
            else:
                yield json.dumps(result.raw, ensure_ascii=False)
        elif result.speech:
            yield json.dumps({
                "speech": result.speech,
                "emotion": result.emotion.value,
                "motion": result.motion.value,
                "skip": False,
            }, ensure_ascii=False)

    async def aclose(self) -> None:
        """保持している接続を閉じる（バックエンドを作り直す前に呼ばれる）"""

    @property
    def supports_streaming(self) -> bool:
        """True を返すバックエンドは generate_stream() が本物のストリームを返す"""
        return False

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def is_healthy(self) -> bool: ...
