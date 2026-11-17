"""HybridRouter: 応答の種別でLLMを振り分けるルーター

待機系（独り言・掛け合い・記憶整理など、急がない生成）はローカルLLMへ、
即応が要る応答（コメント返答・呼びかけへの返事）はクラウドAPIへ流す。

ルーター自体も LLMBackend なので、Orchestrator や CharacterAgent からは
「いつものバックエンド1個」に見える。切り替えは generate / generate_stream に
渡る route 引数（"respond" / "idle"）だけで決まる。

フォールバック規則:
  - クラウド側が不調（is_healthy=False）の間は、即応もローカルで受ける
    （クラウド障害で配信が止まらないように）。
  - クラウド側が空応答（skip）を返したら、その場でローカルでもう一度だけ生成する
    （APIのレート制限や一時エラーを、視聴者に見せないための保険）。
  - ローカル側が不調なら従来どおり agent 側の fallback 機構に任せる
    （ルーターの is_healthy はローカル側の状態をそのまま返す）。
"""
from __future__ import annotations

from typing import AsyncIterator

from backend.llm.base import LLMBackend
from backend.logger import get_logger
from backend.models import ChatMessage, LLMResponse

logger = get_logger(__name__)


class HybridRouter(LLMBackend):
    """idle ルートと fast ルートを持つ振り分けバックエンド"""

    def __init__(self, idle_backend: LLMBackend, fast_backend: LLMBackend):
        self._idle = idle_backend
        self._fast = fast_backend

    # ── 振り分け ──────────────────────────────────────────────

    def _pick(self, route: str) -> LLMBackend:
        if route == "idle":
            return self._idle
        # 即応ルート: クラウドが不調ならローカルで受ける
        if not self._fast.is_healthy:
            return self._idle
        return self._fast

    # ── 生成 ──────────────────────────────────────────────────

    async def generate(
        self,
        system_prompt: str,
        messages: list[ChatMessage],
        *,
        route: str = "respond",
        temperature: float | None = None,
    ) -> LLMResponse:
        backend = self._pick(route)
        # 温度の上書きは指定されたときだけ内側へ渡す（未指定時の呼び出し
        # 形状は従来どおりに保ち、温度引数を持たない実装とも共存する）
        extra = {"temperature": temperature} if temperature is not None else {}
        result = await backend.generate(system_prompt, messages, **extra)
        if route != "idle" and backend is self._fast and (result.skip or not result.speech):
            # クラウドの空振りはローカルで拾い直す（1回だけ）
            logger.warning("HybridRouter: fast backend returned empty, retrying on idle backend")
            retry = await self._idle.generate(system_prompt, messages, **extra)
            if not retry.skip and retry.speech:
                return retry
        return result

    async def generate_stream(
        self,
        system_prompt: str,
        messages: list[ChatMessage],
        *,
        route: str = "respond",
        temperature: float | None = None,
    ) -> AsyncIterator[str]:
        backend = self._pick(route)
        extra = {"temperature": temperature} if temperature is not None else {}
        produced = False
        async for delta in backend.generate_stream(system_prompt, messages, **extra):
            produced = True
            yield delta
        if not produced and route != "idle" and backend is self._fast:
            # 1文字も出なかったときだけローカルで流し直す（途中失敗の二重発話は防ぐ）
            logger.warning("HybridRouter: fast stream produced nothing, retrying on idle backend")
            async for delta in self._idle.generate_stream(system_prompt, messages, **extra):
                yield delta

    async def probe(self) -> bool:
        """プライマリ（待機系＝ローカル）の回復確認を内側へ委譲する。

        failback 判定用（Phase 4）。内側が probe を持たなければ、
        現在の健全フラグをそのまま返す（能動確認できない系統では
        failback しない側に倒すため、フラグが立っていない限り False）。
        """
        inner = getattr(self._idle, "probe", None)
        if inner is None:
            return self._idle.is_healthy
        return await inner()

    async def aclose(self) -> None:
        for backend in (self._idle, self._fast):
            try:
                await backend.aclose()
            except Exception:
                pass

    # ── 状態 ──────────────────────────────────────────────────

    @property
    def supports_streaming(self) -> bool:
        # どちらもストリーミング前提（片方だけ非対応でも、非対応側は
        # 基底クラスのデフォルト実装が一括yieldするので揃う）
        return self._idle.supports_streaming or self._fast.supports_streaming

    @property
    def name(self) -> str:
        return f"hybrid({self._idle.name}+{self._fast.name})"

    @property
    def is_healthy(self) -> bool:
        # agent.failover は「プライマリ全体が死んだか」で見る。待機系＝常駐側の
        # 状態を代表させる（クラウド不調は _pick 側で吸収する）
        return self._idle.is_healthy

    # ── 内部部品へのアクセス（Orchestrator の warmup 等が使う） ──

    @property
    def idle_backend(self) -> LLMBackend:
        return self._idle

    @property
    def fast_backend(self) -> LLMBackend:
        return self._fast
