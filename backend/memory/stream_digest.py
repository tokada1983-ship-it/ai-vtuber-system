"""StreamDigest: 配信中の記憶整理（流れダイジェスト＋配信中エピソードの圧縮）

長時間配信では、配信中に自動で貯まる材料が2方向に膨らむ。

  1. エピソード記憶（MemoryStore）: 話題を話すたびに source="web" の
     記録が、記憶整理のたびに source="auto" の事実が増える。上限300件の
     忘却（compact）はあるが、想起プール（sample_recent や RAG 索引）は
     配信が進むほど「この配信で作ったばかりの記憶」で満たされ、自己参照
     と検索ノイズが増える（2026-11-21 BFランの反復構造の一因）。
  2. 会話の流れそのもの: LLM の履歴窓は response_history_turns（既定20
     ターン）しかなく、それより前は記憶整理が抽出した原子的な事実として
     しか残らない。整理入力も末尾6000字に切られるため、50ターンごとの
     ブロック前半の流れは事実抽出にも届かず、どこにも残らず消える。

このモジュールは、既存の記憶整理（MemoryConsolidator）の成功サイクルに
相乗りして、次の2つを行う（キャラごと）。

  - 流れダイジェスト: 整理待ちだった会話バッチを旧ダイジェストと統合し、
    「この配信のここまでの流れ」を更新する。ダイジェストは L3
    （CharacterAgent.build_idle_memory）に常駐注入され、履歴窓より前の
    流れを有界な文字数で保持する。
  - 配信中エピソードの圧縮: このセッションで増えた web/auto エピソード
    が上限を超えたら、古い側（直近保持と最近使用分を除く）を要約して
    source="stream_digest" の1件に置き換える。置き換えた原典の全文は
    配信アーカイブ（StreamArchiver の jsonl / episode.md）に残る。

安全条件:
  - LLM の失敗・空応答では何も変更しない（ダイジェストは旧版のまま、
    エピソードは削除しない）。次回サイクルで再試行する。
  - 進行中の弧・保留中の回答はプランナ側の状態であり、このモジュールは
    一切触らない。会話ログ（履歴窓・整理待ち）も読み取りのみ。
  - 圧縮対象から、ピン留め・手動追加・直近の保持件数・最近使われた
    エピソードは外す。
"""
from __future__ import annotations

import re
import time
from typing import TYPE_CHECKING

from backend.logger import get_logger
from backend.models import ChatMessage

if TYPE_CHECKING:
    from backend.llm.base import LLMBackend
    from backend.memory.memory_store import MemoryStore

logger = get_logger(__name__)

_DIGEST_SYSTEM = """あなたは配信の記録係です。VTuberの配信について、「これまでのダイジェスト」と「新しい会話記録」を統合し、配信の流れをまとめたダイジェストを更新してください。

- 出力はダイジェスト本文のみ（見出し・前置き・コードブロックは付けない）
- キャラクターの一人称は使わず、中立な文で書く（主語は名前で明示する）
- 話題の変遷が時系列で分かるように、箇条書き中心で書く
- 固有名詞（人名・作品名・製品名・数値など）は落とさない
- 視聴者との印象的なやり取り、約束したこと、決まったことがあれば残す
- 「楽しかった」など感想だけの行は書かない。何について話したかを書く
- 全体で {max_chars} 字以内に収める"""

_COMPACT_SYSTEM = """あなたは配信の記録係です。配信中に溜まった記憶の断片を、後で思い出すための1件の要約に統合してください。

- 出力は要約本文のみ（見出し・前置きは付けない）
- キャラクターの一人称は使わず、中立な文で書く
- 固有名詞（人名・作品名・製品名・数値など）は落とさない
- 断片同士で重なる内容は1つにまとめる
- 全体で300字以内に収める"""


def _fingerprint(message) -> str:
    """消化済み判定用のメッセージ指紋（role＋本文冒頭）"""
    content = re.sub(r"\s+", " ", getattr(message, "content", "") or "")
    return f"{getattr(message, 'role', '')}|{content[:80]}"


def _format_conversation(messages) -> str:
    """会話メッセージを記録テキストに整形する（MemoryConsolidator と同形式）"""
    lines = []
    for m in messages:
        content = m.content or ""
        vm = re.match(
            r'<viewer_comment author="([^"]*)"[^>]*>(.*?)</viewer_comment>',
            content, re.DOTALL)
        pm = re.match(
            r'<partner_speech name="([^"]*)"[^>]*>(.*?)</partner_speech>',
            content, re.DOTALL)
        if vm:
            lines.append(f"視聴者({vm.group(1)}): {vm.group(2).strip()}")
        elif pm:
            lines.append(f"共演者({pm.group(1)}): {pm.group(2).strip()}")
        else:
            lines.append(f"AI: {content}")
    return "\n".join(lines)


def _clean_generated(raw: str) -> str:
    """生成テキストから思考部とコードフェンスを落とす"""
    text = re.sub(r"<think>.*?</think>", "", raw or "", flags=re.DOTALL)
    text = re.sub(r"```(?:\w+)?", "", text)
    return text.strip()


def _cap_text(text: str, max_chars: int) -> str:
    """上限を超えたら文・行の境界で切る（境界が無ければそのまま切る）"""
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    best = -1
    for sep in ("。", "\n"):
        idx = cut.rfind(sep)
        if idx >= max_chars * 0.7:
            best = max(best, idx + (1 if sep == "。" else 0))
    if best > 0:
        return cut[:best].rstrip()
    return cut.rstrip()


class StreamDigest:
    """キャラ1体ぶんの配信中ダイジェストとエピソード圧縮を担う。

    呼び出しは Orchestrator の記憶整理サイクル（成功時）から。
    どの処理も失敗してよい設計で、例外は外へ投げない。
    """

    def __init__(
        self,
        *,
        max_chars: int = 900,
        input_chars: int = 9000,
        episode_cap: int = 40,
        episode_keep_recent: int = 15,
        episode_protect_sec: float = 600.0,
        session_started_at: float | None = None,
    ):
        self._max_chars = max(100, int(max_chars))
        self._input_chars = max(1000, int(input_chars))
        self._episode_cap = max(0, int(episode_cap))
        self._episode_keep_recent = max(0, int(episode_keep_recent))
        self._episode_protect_sec = max(0.0, float(episode_protect_sec))
        self._session_started_at = float(session_started_at or time.time())
        self._digest = ""
        # 消化済みバッチの末尾指紋。整理待ちが再提示された場合（整理失敗で
        # バッファが残った等）に同じ範囲を二重に取り込まないための番兵
        self._digested_tail = ""
        self._digest_count = 0
        self._compaction_count = 0
        self._last_compaction: dict = {}

    @property
    def text(self) -> str:
        """現在の流れダイジェスト（L3 注入用）"""
        return self._digest

    def stats(self) -> dict:
        return {
            "digest_chars": len(self._digest),
            "digest_count": self._digest_count,
            "compactions": self._compaction_count,
            "last_compaction": dict(self._last_compaction),
        }

    async def update(
        self,
        *,
        messages: list,
        llm: "LLMBackend",
        store: "MemoryStore | None",
        character_name: str = "",
    ) -> dict:
        """整理サイクルに相乗りする入口。結果の要約を返し、例外は投げない"""
        result = {"digest_updated": False, "compacted": 0}
        try:
            result["digest_updated"] = await self._update_digest(messages, llm)
        except Exception as e:
            logger.warning("stream digest update failed",
                           character=character_name,
                           error_type=type(e).__name__, error=repr(e))
        try:
            result["compacted"] = await self._compact_episodes(store, llm)
        except Exception as e:
            logger.warning("stream episode compaction failed",
                           character=character_name,
                           error_type=type(e).__name__, error=repr(e))
        return result

    # ── 流れダイジェスト ────────────────────────────────────

    async def _update_digest(self, messages, llm) -> bool:
        from backend.memory.memory_consolidator import MemoryConsolidator

        msgs = list(messages or [])
        if not msgs:
            return False
        start = 0
        if self._digested_tail:
            # 前回消化した末尾より後ろだけを今回のバッチにする
            for i in range(len(msgs) - 1, -1, -1):
                if _fingerprint(msgs[i]) == self._digested_tail:
                    start = i + 1
                    break
        batch = msgs[start:]
        if not batch:
            return False
        conv = _format_conversation(batch)
        if not conv.strip():
            # 中身のないバッチは消化済みにして次へ進める
            self._digested_tail = _fingerprint(msgs[-1])
            return False
        if len(conv) > self._input_chars:
            half = self._input_chars // 2
            conv = conv[:half] + "\n…（中略）…\n" + conv[-half:]
        user = (f"## これまでのダイジェスト\n{self._digest or '（まだ無い）'}"
                f"\n\n## 新しい会話記録\n{conv}")
        raw = await MemoryConsolidator._generate_raw(
            llm, _DIGEST_SYSTEM.format(max_chars=self._max_chars),
            [ChatMessage(role="user", content=user)])
        text = _clean_generated(raw)
        if not text:
            # 空応答では番兵を進めない（次回サイクルで同じバッチを再試行）
            logger.info("stream digest skipped (empty generation)")
            return False
        before = len(self._digest)
        self._digest = _cap_text(text, self._max_chars)
        self._digested_tail = _fingerprint(msgs[-1])
        self._digest_count += 1
        logger.info("stream digest updated",
                    before_chars=before, after_chars=len(self._digest),
                    messages=len(batch), digests=self._digest_count)
        return True

    # ── 配信中エピソードの圧縮 ──────────────────────────────

    async def _compact_episodes(self, store, llm) -> int:
        from backend.memory.memory_consolidator import MemoryConsolidator

        if store is None or self._episode_cap <= 0:
            return 0
        now = time.time()
        session = [e for e in store.all()
                   if e.created_at >= self._session_started_at
                   and e.source in ("web", "auto") and not e.pinned]
        if len(session) <= self._episode_cap:
            return 0
        session.sort(key=lambda e: e.created_at)
        keep_ids = ({e.id for e in session[-self._episode_keep_recent:]}
                    if self._episode_keep_recent > 0 else set())
        eligible = [e for e in session
                    if e.id not in keep_ids
                    and not (e.last_used_at
                             and now - e.last_used_at
                             < self._episode_protect_sec)]
        # 置換後の件数が上限に収まるだけの件数が必要（要約1件が増える分）
        need = len(session) - self._episode_cap + 1
        if len(eligible) < need:
            logger.info("stream episode compaction deferred (protected)",
                        session=len(session), eligible=len(eligible),
                        need=need)
            return 0
        batch = eligible[:need]
        body = "\n".join(f"- {e.text}" for e in batch)
        if len(body) > self._input_chars:
            body = body[: self._input_chars]
        raw = await MemoryConsolidator._generate_raw(
            llm, _COMPACT_SYSTEM,
            [ChatMessage(role="user",
                         content="次の記憶の断片を1件に統合してください。"
                                 f"\n\n{body}")])
        summary = _clean_generated(raw)
        if len(summary) < 20:
            # 要約として成立していない出力では原典を消さない
            logger.info("stream episode compaction skipped (bad summary)",
                        preview=summary[:40])
            return 0
        importance = min(1.6, max((e.importance for e in batch),
                                  default=1.0))
        entry = store.add(f"（配信中のまとめ）{summary}",
                          source="stream_digest", importance=importance,
                          dedupe=False)
        if entry is None:
            logger.warning("stream episode compaction aborted "
                           "(digest entry not added)")
            return 0
        deleted = 0
        for e in batch:
            if store.delete(e.id):
                deleted += 1
        self._compaction_count += 1
        self._last_compaction = {"at": now, "merged": deleted,
                                 "session_before": len(session)}
        logger.info("stream episodes compacted",
                    merged=deleted, session_before=len(session),
                    session_after=len(session) - deleted + 1,
                    total=store.count())
        return deleted
