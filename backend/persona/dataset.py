"""人格QLoRA用の学習データを作る

教材は3種類:
  sample    … character.yaml に書かれたセリフ例（「…」で囲まれた部分）
  history   … ローカルチャットで実際に交わした会話（chat_history.jsonl）
  synthetic … 今のLLMに「全文のキャラ設定」を渡して答えさせた応答（教師データ）

出力は1行1会話の JSONL（OpenAI/HF共通の messages 形式）。
システムプロンプトは実行時と同じ build_system_prompt(slim_persona=True) で作るので、
「学習したときと違うプロンプトで呼ばれて口調がぶれる」ことがない。
アシスタントの出力は実行時と同じ {"emotion","motion","speech"} のJSONにする。
これによりJSON形式の遵守もアダプタが覚える（パース失敗によるskipが減る）。
"""
from __future__ import annotations

import asyncio
import json
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Awaitable, Callable

import yaml

from backend.config import AppConfig
from backend.llm.persona_validator import PersonaValidator
from backend.llm.prompt_builder import build_system_prompt, idle_instruction
from backend.logger import get_logger
from backend.models import ChatMessage, Emotion, Motion
from backend.persona import seeds
from backend.safety.response_filter import ResponseFilter

logger = get_logger(__name__)

ProgressFn = Callable[[int, int, str], Awaitable[None] | None]

_EMOTIONS = {e.value for e in Emotion}
_MOTIONS = {m.value for m in Motion}
# 実会話ログには motion が残っていないので、感情から自然なものを当てる
_EMOTION_TO_MOTION = {
    "happy": "nod", "surprised": "jump", "sad": "shy",
    "angry": "shake_head", "relaxed": "idle", "neutral": "idle",
}
_MAX_SPEECH_CHARS = 80


@dataclass
class Example:
    messages: list[dict]             # system / (user) / assistant
    source: str                      # sample / history / synthetic
    category: str = ""

    @property
    def speech(self) -> str:
        try:
            return json.loads(self.messages[-1]["content"]).get("speech", "")
        except (json.JSONDecodeError, KeyError, IndexError):
            return ""

    def to_json(self) -> str:
        return json.dumps({"messages": self.messages, "source": self.source,
                           "category": self.category}, ensure_ascii=False)


@dataclass
class DatasetReport:
    path: str = ""
    eval_path: str = ""
    total: int = 0
    train: int = 0
    eval: int = 0
    by_source: dict = field(default_factory=dict)
    rejected: int = 0
    created_at: float = 0.0

    def to_dict(self) -> dict:
        return dict(self.__dict__)


# ── 小道具 ──────────────────────────────────────────────────

def viewer_turn(author: str, message: str) -> str:
    """実行時の ConversationLog.add_comment と同じ形の user 発言を作る"""
    def esc(t: str) -> str:
        return (t.replace("&", "&amp;").replace("<", "&lt;")
                 .replace(">", "&gt;").replace('"', "&quot;"))
    return f'<viewer_comment author="{esc(author)}">{esc(message)}</viewer_comment>'


def assistant_json(speech: str, emotion: str = "neutral", motion: str = "idle") -> str:
    emotion = emotion if emotion in _EMOTIONS else "neutral"
    motion = motion if motion in _MOTIONS else "idle"
    return json.dumps({"emotion": emotion, "motion": motion, "speech": speech},
                      ensure_ascii=False, separators=(",", ":"))


def shorten(text: str, limit: int = _MAX_SPEECH_CHARS) -> str:
    """長いセリフを文の切れ目で limit 文字以内に収める"""
    text = re.sub(r"\s+", " ", text or "").strip()
    if len(text) <= limit:
        return text
    out = ""
    for sent in re.split(r"(?<=[。！？!?…」])", text):
        if len(out) + len(sent) > limit:
            break
        out += sent
    return out.strip() or text[:limit]


def load_character(config: AppConfig) -> dict:
    """character.yaml（キャラ別フォルダ優先）を読む"""
    name = config.character.name
    candidates = [
        Path(config.character.data_dir) / "characters" / name / "character.yaml",
        Path(config.character.prompt_file),
    ]
    for path in candidates:
        if path.exists():
            with open(path, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            if not data.get("name") or data.get("name") == name:
                return data
    return {"name": name}


_HEADING = re.compile(r"^[ \t]*(?:\d+[.．、)]|[-・*])[ \t]*([^「\n]+?)[ \t]*$", re.MULTILINE)
_QUOTE = re.compile(r"「([^「」]{4,400})」", re.DOTALL)


def _is_line(text: str) -> bool:
    """「わらわ」「眠れる森の番人」のような語句・肩書きはセリフ扱いしない"""
    return len(text) >= 12 and bool(re.search(r"[、。！？!?…]", text))


def extract_sample_lines(char_data: dict) -> list[tuple[str, str]]:
    """
    キャラ設定文からセリフ例を (状況, セリフ) で抜き出す。
    「1. 配信開始の挨拶」のような見出しの後にある「…」を、その状況のセリフとみなす。
    """
    text = "\n".join(str(char_data.get(k) or "") for k in ("description", "personality", "rules"))
    headings = [(m.start(), m.group(1)[:40]) for m in _HEADING.finditer(text)]
    results: list[tuple[str, str]] = []
    for m in _QUOTE.finditer(text):
        line = re.sub(r"\s+", " ", m.group(1)).strip()
        if not _is_line(line):
            continue
        situation = "雑談"
        for pos, head in headings:
            if pos > m.start():
                break
            situation = head
        results.append((situation, line))
    return results


def guess_emotion(text: str) -> str:
    if re.search(r"ひぃ|無理|ダメ|怖|待って", text):
        return "surprised"
    if re.search(r"[！!]{1,}|さあ|よいぞ|楽し|嬉し|ありがと", text):
        return "happy"
    if re.search(r"エラー|ログアウト|うぅ|しょんぼり|悲し", text):
        return "sad"
    return "neutral"


# ── 本体 ────────────────────────────────────────────────────

class PersonaDatasetBuilder:

    def __init__(self, config: AppConfig, chat_history_path: Path | None = None,
                 seed: int | None = None):
        self._cfg = config
        self._rng = random.Random(seed)
        self._validator = PersonaValidator()
        self._filter = ResponseFilter(ng_words=config.safety.ng_words, enabled=True)
        self._chat_history_path = chat_history_path

    # システムプロンプト（学習用=slim / 教師用=full）
    def _system(self, slim: bool, extra: str = "") -> str:
        return build_system_prompt(self._cfg, slim_persona=slim) + extra

    def _accept(self, speech: str) -> str | None:
        """教材として使えるか検査し、使えるなら整えたセリフを返す"""
        speech = shorten(speech)
        if not speech or len(speech) < 2:
            return None
        if not self._filter.check(speech).ok:
            return None
        speech, _ = self._validator.validate_and_repair(
            speech, character_name=self._cfg.character.name)
        return speech

    # ① キャラ設定のセリフ例
    def from_samples(self) -> list[Example]:
        out: list[Example] = []
        for situation, line in extract_sample_lines(load_character(self._cfg)):
            speech = self._accept(line)
            if not speech:
                continue
            user = viewer_turn("配信", f"（状況: {situation}）")
            out.append(Example(
                messages=[
                    {"role": "system", "content": self._system(slim=True)},
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": assistant_json(
                        speech, guess_emotion(line), "idle")},
                ],
                source="sample", category=situation,
            ))
        return out

    # ② ローカルチャットの実会話
    def from_history(self, context_turns: int = 2) -> list[Example]:
        path = self._chat_history_path
        if not path or not path.exists():
            return []
        entries: list[dict] = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(e, dict) and e.get("text") and e.get("role") in ("user", "assistant"):
                    entries.append(e)

        out: list[Example] = []
        system = self._system(slim=True)
        for i, e in enumerate(entries):
            if e["role"] != "assistant" or i == 0 or entries[i - 1]["role"] != "user":
                continue
            speech = self._accept(e["text"])
            if not speech:
                continue
            # 直前の数ターンを文脈として残す（会話の流れに沿った口調を学ばせる）
            ctx: list[dict] = []
            for prev in entries[max(0, i - 1 - context_turns * 2): i - 1]:
                if prev["role"] == "user":
                    ctx.append({"role": "user",
                                "content": viewer_turn(prev.get("author") or "ユーザー", prev["text"])})
                else:
                    ctx.append({"role": "assistant", "content": prev["text"]})
            # user→user のような並びはテンプレートが嫌うので整える
            while ctx and ctx[0]["role"] != "user":
                ctx.pop(0)
            user = entries[i - 1]
            emotion = e.get("emotion") or "neutral"
            out.append(Example(
                messages=[{"role": "system", "content": system}, *ctx,
                          {"role": "user", "content": viewer_turn(user.get("author") or "ユーザー",
                                                                  user["text"])},
                          {"role": "assistant", "content": assistant_json(
                              speech, emotion, _EMOTION_TO_MOTION.get(emotion, "idle"))}],
                source="history", category="chat",
            ))
        return out

    # ③ 教師LLMによる合成
    async def synthesize(self, teacher, n: int, progress: ProgressFn | None = None,
                         concurrency: int = 2,
                         retry_wait: float = 3.0) -> tuple[list[Example], int]:
        """
        teacher（今使っているLLM）に全文のキャラ設定を渡して応答させ、
        その応答を slim プロンプトとの組で教材にする（知識の蒸留）。
        """
        if n <= 0 or teacher is None:
            return [], 0
        jobs: list[tuple[str, str, str | None]] = []   # (category, 指示 or コメント, author)
        idle_ratio = 0.2
        for _ in range(n):
            if self._rng.random() < idle_ratio:
                jobs.append(("idle", self._rng.choice(seeds.IDLE_TOPICS), None))
            else:
                cat, msg = self._rng.choice(seeds.COMMENTS)
                jobs.append((cat, msg, self._rng.choice(seeds.VIEWER_NAMES)))

        sem = asyncio.Semaphore(max(1, concurrency))
        results: list[Example] = []
        rejected = 0
        done = 0

        async def one(cat: str, text: str, author: str | None) -> None:
            nonlocal rejected, done
            async with sem:
                if author is None:
                    extra = idle_instruction(text)
                    # Qwen3系のチャットテンプレートは user 発言が無いとエラーになるので、
                    # 先生役にだけ合図を渡す（学習データ側は実行時と同じく user 無しのまま）
                    teacher_msgs: list[ChatMessage] = [
                        ChatMessage(role="user", content="（コメントが途切れました。独り言をどうぞ）")]
                    user_msg = None
                else:
                    extra = ""
                    user_msg = viewer_turn(author, text)
                    teacher_msgs = [ChatMessage(role="user", content=user_msg)]
                resp = None
                # モデルのロード中などは一時的に500/タイムアウトになるので1回だけやり直す
                for attempt in range(2):
                    try:
                        resp = await teacher.generate(self._system(slim=False, extra=extra),
                                                      teacher_msgs)
                    except Exception as e:
                        logger.warning("teacher generate failed", error=str(e))
                        resp = None
                    if resp is not None and not resp.skip:
                        break
                    if attempt == 0:
                        await asyncio.sleep(retry_wait)
            done += 1
            speech = None if resp is None or resp.skip else self._accept(resp.speech)
            if not speech:
                rejected += 1
            else:
                msgs = [{"role": "system", "content": self._system(slim=True, extra=extra)}]
                if user_msg:
                    msgs.append({"role": "user", "content": user_msg})
                msgs.append({"role": "assistant", "content": assistant_json(
                    speech, resp.emotion.value, resp.motion.value)})
                results.append(Example(messages=msgs, source="synthetic", category=cat))
            if progress:
                r = progress(done, len(jobs), speech or "")
                if asyncio.iscoroutine(r):
                    await r

        await asyncio.gather(*(one(*j) for j in jobs))
        return results, rejected

    # まとめて書き出す
    def write(self, examples: list[Example], out_dir: Path, eval_ratio: float = 0.05,
              rejected: int = 0) -> DatasetReport:
        # 同じセリフの重複を落とす（合成データは似た応答が出やすい）
        seen: set[str] = set()
        unique: list[Example] = []
        for ex in examples:
            key = ex.speech
            if key and key not in seen:
                seen.add(key)
                unique.append(ex)
        self._rng.shuffle(unique)

        n_eval = int(len(unique) * eval_ratio) if len(unique) >= 20 else 0
        eval_set, train_set = unique[:n_eval], unique[n_eval:]

        out_dir.mkdir(parents=True, exist_ok=True)
        train_path, eval_path = out_dir / "train.jsonl", out_dir / "eval.jsonl"
        train_path.write_text("".join(e.to_json() + "\n" for e in train_set), encoding="utf-8")
        eval_path.write_text("".join(e.to_json() + "\n" for e in eval_set), encoding="utf-8")

        by_source: dict[str, int] = {}
        for e in unique:
            by_source[e.source] = by_source.get(e.source, 0) + 1
        report = DatasetReport(
            path=str(train_path), eval_path=str(eval_path), total=len(unique),
            train=len(train_set), eval=len(eval_set), by_source=by_source,
            rejected=rejected, created_at=time.time(),
        )
        (out_dir / "meta.json").write_text(
            json.dumps({**report.to_dict(), "character": self._cfg.character.name},
                       ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("persona dataset written", **{k: v for k, v in report.to_dict().items()
                                                   if k in ("total", "train", "eval", "rejected")})
        return report