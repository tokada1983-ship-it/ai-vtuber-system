"""OllamaBackend: Ollama を使ったLLMバックエンド"""
from __future__ import annotations

import json
import time
from typing import AsyncIterator

import httpx

from backend.config import LLMBackendConfig
from backend.llm import ollama_service
from backend.llm.adaptive_ctx import AdaptiveContextController
from backend.llm.base import LLMBackend
from backend.llm.parser import parse_llm_output, parse_tag_output
from backend.logger import get_logger
from backend.models import ChatMessage, LLMResponse

logger = get_logger(__name__)

# ストリーミング生成の読み取りタイムアウト（秒）。中途停滞（数十トークン
# 進んでから無音になる型）は、この時間だけ無音が続いたら失敗として
# 切り上げる。プリフィル（数千トークン）とモデルの再読込（実測約15秒）
# には十分な余裕を持たせてある（2026-11-21 BFラン: 120秒待ちが
# 346秒間隔の一因になったことへの対策）
_STREAM_READ_TIMEOUT_SEC = 60.0


def _ns_to_ms(value) -> int | None:
    """Ollama がナノ秒で返す所要時間をミリ秒へ（無い場合は None）"""
    if not isinstance(value, (int, float)):
        return None
    return int(value / 1_000_000)


class OllamaBackend(LLMBackend):

    def __init__(self, cfg: LLMBackendConfig):
        self._cfg = cfg
        self._healthy = True
        self._client: httpx.AsyncClient | None = None
        # 直近の生成メタ（done_reason / eval_count / thinking 文字数）。
        # no-speech 失敗の切り分け用に保持し、呼び出し側が
        # last_generation_meta() で引く（base.LLMBackend 参照）
        self._last_gen_meta: dict = {}
        # num_ctx 適応制御（factory が configure_adaptive_ctx で有効化する。
        # 無効なら None で、従来どおり extra_payload の固定値が効く）
        self._adaptive: AdaptiveContextController | None = None
        # プロンプトの文字数→トークン数の較正（最上段での生成前
        # 切り詰めの推定に使う。実測 prompt_eval_count で逐次較正する）
        self._last_prompt_chars: int = 0
        self._tokens_per_char: float | None = None

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
        """回復確認用の生成プローブ（failback 判定用。Phase 4）。

        実生成に近い条件で判定する: ある程度の長さのストリーミング生成が
        最後まで流れることをもって健全とする。以前の num_predict=1 の
        一括生成では「短いリクエストは通るが、実生成（大きなプロンプト＋
        長いストリーム）は途中で止まる」状態を健全と誤判定し、primary へ
        戻しては再失敗する往復が起きた（2026-11-21 BFラン）。生成結果の
        テキストは捨てる（通常の発話経路には影響しない）。
        """
        payload: dict = {
            "model": self._cfg.model,
            "messages": [{"role": "user", "content":
                           "今日の配信の見どころを、3文くらいで紹介してください。"}],
            "stream": True,
            "keep_alive": self.service.keep_alive,
            "options": {"temperature": 0.0, "num_predict": 48},
        }
        extra = self._cfg.extra_payload or {}
        if "think" in extra:
            payload["think"] = extra["think"]
        # 実生成と同じコンテキスト長で投げる（段が違うと判定がずれる）
        if self._adaptive is not None:
            payload["options"]["num_ctx"] = self._adaptive.current_num_ctx
        elif "num_ctx" in extra:
            payload["options"]["num_ctx"] = extra["num_ctx"]
        started = time.monotonic()
        eval_count = 0
        done = False
        try:
            async with self._get_client().stream(
                    "POST", self._url, json=payload,
                    timeout=httpx.Timeout(45.0, connect=5.0)) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if chunk.get("done"):
                        eval_count = int(chunk.get("eval_count") or 0)
                        done = True
                        break
        except Exception as e:
            logger.debug("Ollama probe failed",
                         error_type=type(e).__name__, error=str(e))
        # done まで流れ、かつ相応のトークン数が出ていること（1トークンで
        # 即 done する応答は「生成できる」とはみなさない）
        ok = done and eval_count >= 16
        logger.info("Ollama probe", model=self._cfg.model, ok=ok,
                    eval_count=eval_count,
                    duration_ms=int((time.monotonic() - started) * 1000))
        self._healthy = ok
        return ok

    # ── num_ctx 適応制御 ──────────────────────────────────────

    def configure_adaptive_ctx(self, llm_cfg) -> None:
        """factory から config.llm の適応制御設定を受け取る。

        llm.adaptive_ctx が有効なときだけコントローラを持つ。
        無効なら従来動作（extra_payload の num_ctx 固定）に戻る。
        """
        if not getattr(llm_cfg, "adaptive_ctx", False):
            self._adaptive = None
            return
        self._adaptive = AdaptiveContextController(
            list(getattr(llm_cfg, "context_tiers",
                         [4096, 6144, 8192, 12288, 16384])),
            int(getattr(llm_cfg, "context_initial", 8192)),
            raise_ratio=float(getattr(llm_cfg, "context_raise_ratio", 0.85)),
            lower_ratio=float(getattr(llm_cfg, "context_lower_ratio", 0.45)),
            cooldown_sec=float(getattr(llm_cfg, "context_cooldown_sec", 300.0)),
            lower_turns=int(getattr(llm_cfg, "context_lower_turns", 50)),
            lower_window_sec=float(
                getattr(llm_cfg, "context_lower_window_sec", 1800.0)),
        )
        logger.info("Adaptive num_ctx enabled", model=self._cfg.model,
                    num_ctx=self._adaptive.current_num_ctx,
                    tiers=self._adaptive.tiers)

    def begin_turn(self) -> None:
        """ターン境界: 保留中の段変更を適用する（クールダウン判定込み）"""
        if self._adaptive is None:
            return
        change = self._adaptive.begin_turn()
        if change:
            logger.info(
                "Adaptive num_ctx changed", model=self._cfg.model,
                from_num_ctx=change["from_num_ctx"],
                to_num_ctx=change["to_num_ctx"],
                reason=change["reason"],
                prompt_eval_count=change["prompt_eval_count"],
                done_reason=change["done_reason"])

    def context_status(self) -> dict:
        if self._adaptive is not None:
            status = self._adaptive.status()
            status["adaptive"] = True
            status["model"] = self._cfg.model
            return status
        extra = self._cfg.extra_payload or {}
        return {
            "adaptive": False,
            "num_ctx": extra.get("num_ctx"),
            "last_prompt_eval_count":
                self._last_gen_meta.get("prompt_eval_count"),
            "last_done_reason": self._last_gen_meta.get("done_reason"),
            "model": self._cfg.model,
        }

    def _observe_ctx_meta(self) -> None:
        """生成メタを適応制御へ渡し、文字数→トークン数の較正も更新する"""
        meta = self._last_gen_meta
        pec = meta.get("prompt_eval_count")
        if pec and self._last_prompt_chars > 0:
            ratio = pec / self._last_prompt_chars
            self._tokens_per_char = min(2.5, max(0.2, ratio))
        if self._adaptive is not None:
            self._adaptive.observe(meta)

    def _truncate_api_messages(self, api_messages: list[dict]) -> list[dict]:
        """最上段でも収まらない見込みのとき、履歴の古い順に落とす。

        system（先頭）は保持し、最低でも直近2件は残す。落とす量は
        較正済みの文字数→トークン数比で推定し、生成上限と安全マージンを
        差し引いた予算に収まるまで落とす。呼び出し側の履歴そのものは
        変えず、このリクエストぶんだけを切り詰める（既存の length 時
        フォールバック＝直近4件への切り詰めとは別経路で、こちらは
        切断が起きる前に効く）。
        """
        top = self._adaptive.current_num_ctx
        budget = top - int(self._cfg.max_tokens) - 256
        ratio = self._tokens_per_char or 1.0

        def est_tokens(msgs: list[dict]) -> float:
            return sum(len(m.get("content") or "") for m in msgs) * ratio

        trimmed = list(api_messages)
        dropped = 0
        while len(trimmed) > 3 and est_tokens(trimmed) > budget:
            trimmed.pop(1)
            dropped += 1
        if dropped:
            logger.warning(
                "Adaptive num_ctx: pre-truncating history at top tier",
                model=self._cfg.model, num_ctx=top, dropped=dropped,
                kept=len(trimmed) - 1, est_prompt_tokens=int(est_tokens(trimmed)),
                budget=budget)
        return trimmed

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
        # 適応制御が最上段で溢れの兆候を見ているときだけ、生成前に
        # 履歴を切り詰める（切断されてから直近4件に落とす既存の
        # フォールバックより前に効かせる）
        if self._adaptive is not None and self._adaptive.needs_pre_truncate(
                self._cfg.max_tokens):
            api_messages = self._truncate_api_messages(api_messages)
        # 較正用に、このリクエストのプロンプト文字数を覚えておく
        self._last_prompt_chars = sum(
            len(m.get("content") or "") for m in api_messages)
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
        # コンテキスト長（num_ctx）も同じ伝搬に乗せる。既定（Modelfile の
        # 4096）のままだと、履歴や資料でプロンプトが膨らんだときに生成の
        # 残り枠が数トークンしか残らず、発話がヘッダ途中で length 切断
        # される（2026-11-19 実測: prompt_eval_count 4091 で eval_count 5）。
        # 適応制御が有効ならコントローラの現在段を送る（ターン境界でのみ
        # 段が動き、生成途中では変わらない）。無効なら従来どおり
        # extra_payload の固定値を送る
        if self._adaptive is not None:
            payload["options"]["num_ctx"] = self._adaptive.current_num_ctx
        elif "num_ctx" in extra:
            payload["options"]["num_ctx"] = extra["num_ctx"]
        # JSONモード（文法制約でJSON以外の出力を抑える）。think と同じ
        # extra_payload 伝搬で、config に "format": "json" と書かれた
        # バックエンドだけ有効になる（既定は無指定＝OFF。Phase 4）
        # タグ方式（protocol="tag"）ではプレーンテキスト生成が前提のため、
        # config に残っていても送らない（PROTOCOL §7。戻すときは
        # protocol を json にするだけで従来動作に復帰する）
        if "format" in extra and self.protocol != "tag":
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
        started = time.monotonic()

        try:
            try:
                resp = await self._get_client().post(url, json=payload)
            except httpx.ConnectError:
                if not await self._revive():
                    raise
                resp = await self._get_client().post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
            message = data.get("message") or {}
            raw_text = message.get("content") or ""
            # 生成メタを保持する（思考フィールドは content とは別に届く。
            # 長さだけを残し、中身はログにもメモリにも残さない）
            self._last_gen_meta = {
                "done_reason": data.get("done_reason"),
                "eval_count": data.get("eval_count"),
                "prompt_eval_count": data.get("prompt_eval_count"),
                "thinking_chars": len(message.get("thinking") or ""),
                "total_duration_ms": _ns_to_ms(data.get("total_duration")),
                "eval_duration_ms": _ns_to_ms(data.get("eval_duration")),
            }
            self._observe_ctx_meta()
            self._healthy = True
            # 成功時も含めて生成レイテンシとトークン数を残す（劣化の前兆＝
            # 緩やかなレイテンシ増加を後から追えるようにするため。
            # 2026-11-21 BFランでは失敗時の記録しかなく分布が取れなかった）
            logger.info(
                "Ollama generation", model=self._cfg.model, route=route,
                duration_ms=int((time.monotonic() - started) * 1000),
                eval_count=data.get("eval_count"),
                prompt_eval_count=data.get("prompt_eval_count"),
                total_duration_ms=self._last_gen_meta["total_duration_ms"],
                eval_duration_ms=self._last_gen_meta["eval_duration_ms"])
            result = parse_tag_output(raw_text, gen_meta=self._last_gen_meta) \
                if self.protocol == "tag" else parse_llm_output(raw_text)
            logger.debug("Ollama response", speech=result.speech[:40] if result.speech else "skip")
            return result

        except httpx.ConnectError:
            logger.warning("Ollama: connection refused — is Ollama running?")
            self._healthy = False
            return LLMResponse(speech="", skip=True)
        except httpx.TimeoutException:
            # 120秒の無応答はモデルのロード待ちでは説明がつかない
            # （実測の再読込は約15秒）。停滞として unhealthy を立て、
            # 呼び出し側のフェイルオーバー判定に乗せる
            logger.warning("Ollama: request timed out — model may be loading",
                           duration_ms=int((time.monotonic() - started) * 1000))
            self._healthy = False
            return LLMResponse(speech="", skip=True)
        except Exception as e:
            logger.warning("Ollama error", error_type=type(e).__name__, error=repr(e)[:300])
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
        started = time.monotonic()

        for attempt in range(2):
            try:
                async with self._get_client().stream(
                        "POST", self._url, json=payload,
                        timeout=httpx.Timeout(120.0, connect=10.0,
                                              read=_STREAM_READ_TIMEOUT_SEC)) as resp:
                    resp.raise_for_status()
                    self._healthy = True
                    # NDJSON を直接読む。content だけを下流へ流し、思考
                    # フィールドは文字数だけ数え、最終チャンクの生成メタ
                    # （done_reason / eval_count）を保持する（no-speech の
                    # 切り分け用。iter_ndjson ではどちらも捨てられていた）
                    thinking_chars = 0
                    async for line in resp.aiter_lines():
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            chunk = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        message = chunk.get("message") or {}
                        thinking = message.get("thinking")
                        if isinstance(thinking, str):
                            thinking_chars += len(thinking)
                        content = message.get("content")
                        if isinstance(content, str) and content:
                            yield content
                        if chunk.get("done"):
                            self._last_gen_meta = {
                                "done_reason": chunk.get("done_reason"),
                                "eval_count": chunk.get("eval_count"),
                                "prompt_eval_count": chunk.get(
                                    "prompt_eval_count"),
                                "thinking_chars": thinking_chars,
                                "total_duration_ms": _ns_to_ms(
                                    chunk.get("total_duration")),
                                "eval_duration_ms": _ns_to_ms(
                                    chunk.get("eval_duration")),
                            }
                            self._observe_ctx_meta()
                            # 成功時も含めて生成レイテンシとトークン数を残す
                            # （generate() 側と同じ観測。劣化の前兆検出用）
                            logger.info(
                                "Ollama generation (stream)",
                                model=self._cfg.model, route=route,
                                duration_ms=int(
                                    (time.monotonic() - started) * 1000),
                                eval_count=chunk.get("eval_count"),
                                prompt_eval_count=chunk.get(
                                    "prompt_eval_count"),
                                total_duration_ms=self._last_gen_meta[
                                    "total_duration_ms"],
                                eval_duration_ms=self._last_gen_meta[
                                    "eval_duration_ms"])
                            break
                return
            except httpx.ConnectError:
                # 1文字も返す前の接続失敗なら、起動してもう1回だけ試す
                if attempt == 0 and await self._revive():
                    continue
                logger.warning("Ollama stream: connection refused — is Ollama running?")
                self._healthy = False
                self._last_gen_meta = {**self._last_gen_meta,
                                       "stream_error": "ConnectError"}
                return
            except Exception as e:
                logger.warning("Ollama stream error", error_type=type(e).__name__, error=repr(e)[:300],
                               duration_ms=int((time.monotonic() - started) * 1000))
                self._healthy = False
                self._last_gen_meta = {**self._last_gen_meta,
                                       "stream_error": type(e).__name__}
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
