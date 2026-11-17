"""MindLoop: 「個」を育てるアイドル時の自律思考サイクル（Phase 2）

設計: AUTONOMOUS_MIND_DESIGN_V2.md §3（マインドループ）/ §4 / §6 / §9

発話のための話題選択（TopicPlanner）とは別に、発話を目的としない思考を回す。
1サイクルで優先順に1タスクだけを実行し、結果は MindStore（意見・意図・選好・
自己状態）と thought_journal に残す。思考はキャラ本人のペルソナモデルで行い、
出力は構造化JSONで受けて、適用ルールをコード側で強制してから書き込む。

適用ルール（§3）:
  - self_state・好奇心キュー: 即時適用
  - 意見の新規作成: 材料の provenance が experienced/told/researched のとき通常適用。
    inferred のみの材料では confidence の上限を 0.4 にして「考え中」として保存
  - 選好: MindStore.add_preference の evidence_count で裏付けを数え、
    2回以上に届くまでは「保留」（プロンプトのスナップショットには established
    な選好だけが載る前提。list_preferences(min_evidence=2) がその境界）
  - 育つ信条（L0b）: 即時適用しない。同じ信条が別々の思考サイクルで
    3回提案されたら MindStore.set_beliefs() で自動採用する（§6 の裏付け条件）。
    提案は thought_journal に kind="belief_proposal" で記録する

2026-11-16 mind25 修正（実機テストで見つかった構造的問題への対処）:
  - 意見の増殖防止: 新規意見の受理前に既存の active な意見とバイグラム類似度で
    照合し、同一主題なら改訂として扱う（find_merge_target）。あわせて、意見形成に
    使った材料（input_refs）を記録し、同じ材料を再び候補にしない。
  - confidence の規律: 同じ主題の再思考だけでは confidence を上げない。上げるのは
    新しい外部材料（experienced/researched かつ未使用の input_refs）があるとき
    だけで、1回 +0.1 まで・上限 0.9（_apply の opinion 分岐で強制）。
  - reflection の硬化: 出力形式をフラットな文字列フィールドに単純化し、
    パースも JSON → 複数オブジェクト統合 → フィールド救出 → 平文の順に緩めた
    専用パーサ（_parse_thought_output）にした。

設定（環境変数で上書き可。config.yaml は変更しない）:
  AIVT_MIND_ENABLED        "0" で背景ループ無効（既定: 有効）
  AIVT_MIND_IDLE_SEC       最終発話からこの秒数を超えたら思考可（既定: 600）
  AIVT_MIND_MAX_PER_HOUR   1時間あたりのサイクル上限/キャラ（既定: 3）
  AIVT_MIND_MIN_INTERVAL_SEC サイクル間隔の下限（既定: 3600/MAX_PER_HOUR）
  AIVT_MIND_POLL_SEC       背景ループの確認間隔（既定: 20）

Orchestrator との結合は duck typing（orchestrator の属性を読むだけ）にして、
循環 import を避ける。キャラ同士の思考はモデルスワップ回避のため直列化する。
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import time

from backend.logger import get_logger
from backend.memory.memory_consolidator import MemoryConsolidator
from backend.memory.mind_store import (Intention, MindStore, Opinion,
                                       _normalize_topic_key)
from backend.models import ChatMessage

logger = get_logger(__name__)

CURIOSITY_LIMIT = 10
BELIEF_APPLY_THRESHOLD = 3
INFERRED_CONFIDENCE_CAP = 0.4
PREFERENCE_ESTABLISHED_EVIDENCE = 2
MAX_ACTIVE_INTENTIONS = 5
_THINK_TIMEOUT_SEC = 240.0

# 意見の confidence 規律（mind25）: 再思考では上げず、新しい外部材料でのみ
# 1回 +CONFIDENCE_STEP まで。ハードキャップは MindStore 側にもある。
CONFIDENCE_HARD_CAP = 0.9
CONFIDENCE_STEP = 0.1
# 意見マージ判定の閾値（mind25）: 本番DBの実データで較正した保守的な値。
# 同一主題の言い換えペアは label containment >= 0.5（ラベルが大きく違う1件は
# reasons Jaccard 0.39〜0.41）に現れ、別主題のペアは label containment <= 0.34、
# reasons Jaccard <= 0.30（そのペアは label containment が 0）に収まった。
# ただし「の情報提供」のような汎用句の共有だけで label containment が上がる
# 衝突（別主題で 0.67、理由の重なりは 0.07）も実測したため、ラベル規則には
# 理由の重なりにも下限（同一主題ペアの最小値 0.12 を下回る 0.1）を併せる。
MERGE_LABEL_CONTAINMENT = 0.5
MERGE_LABEL_REASONS_FLOOR = 0.1
MERGE_REASONS_JACCARD = 0.35
MERGE_LABEL_FLOOR = 0.1
# 意見形成に使った材料（input_refs）の記録先 self_state キー
_MATERIALS_STATE_KEY = "opinion_materials"

# 思い出話題の種・意見形成の材料にしない記憶（orchestrator 側と同じ考え方）
_SEED_NG = re.compile(
    r"元気な挨拶|定型的なやり取り|自己紹介のテーマ|挨拶を交わ|元気なことをアピール")

_PROV_BY_MEMORY_SOURCE = {
    "stream_episode": "experienced",
    "auto": "experienced",
    "web": "researched",
    "manual": "told",
}


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "") or default)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "") or default)
    except ValueError:
        return default


# ── 好奇心キュー（self_state の "curiosity" キー）──────────────────

def get_curiosity(store: MindStore) -> list[dict]:
    """未解決の疑問のキュー（{question, from, ts, attempts} のリスト）"""
    value = store.get_state("curiosity", []) or []
    if not isinstance(value, list):
        return []
    return [x for x in value if isinstance(x, dict) and x.get("question")]


def push_curiosity(store: MindStore, question: str, author: str = "") -> bool:
    """疑問をキューに積む。重複は積まない。上限を超えたら古いものから落とす。"""
    q = (question or "").strip()
    if not q:
        return False
    items = get_curiosity(store)
    if any(x.get("question") == q for x in items):
        return False
    items.append({"question": q, "from": author or "",
                  "ts": time.time(), "attempts": 0})
    store.set_state("curiosity", items[-CURIOSITY_LIMIT:], "experienced")
    logger.info("MindLoop: curiosity queued", question=q[:40])
    return True


def _save_curiosity(store: MindStore, items: list[dict]) -> None:
    store.set_state("curiosity", items[-CURIOSITY_LIMIT:], "experienced")


# ── 意見の照合（コメント応答時の注入用）────────────────────────────

def _norm(text: str) -> str:
    return re.sub(r"[\s。、！？!?…・「」『』（）()\"“”\'’]", "",
                  (text or "")).lower()


def _bigrams(text: str) -> set[str]:
    t = _norm(text)
    return {t[i:i + 2] for i in range(len(t) - 1)} if len(t) >= 2 else set()


def find_relevant_opinions(store: MindStore, text: str,
                           limit: int = 2) -> list[Opinion]:
    """コメント文と active な意見を照合し、関係の深いものを最大 limit 件返す。

    照合はキーワードの重なり（主題のバイグラムがコメントにどれだけ含まれるか）と
    主題キーの部分一致で採点する単純な方式（Phase 2 の最小実装）。
    """
    if store is None or not (text or "").strip():
        return []
    q_norm = _norm(text)
    q_bi = _bigrams(text)
    scored: list[tuple[float, float, Opinion]] = []
    for op in store.list_opinions(limit=30):
        score = 0.0
        if op.topic_key and len(op.topic_key) >= 2 and op.topic_key in q_norm:
            score = 1.0
        else:
            t_bi = _bigrams(op.topic_label + op.reasons)
            if t_bi:
                score = len(q_bi & t_bi) / len(t_bi)
        if score >= 0.2:
            scored.append((score, op.confidence, op))
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return [op for _, _, op in scored[:limit]]


def format_opinion_context(opinions: list[Opinion]) -> str:
    """注入用の「自分の意見」ブロックを作る"""
    if not opinions:
        return ""
    lines = ["## あなたの意見（この話題について、以前から考えてきたこと）"]
    for op in opinions:
        level = "高め" if op.confidence >= 0.7 else (
            "ふつう" if op.confidence >= 0.45 else "低め（まだ考え中）")
        line = f"- 「{op.topic_label}」: {op.stance}"
        if op.reasons:
            line += f"（理由: {op.reasons}）"
        line += f"［確信度: {level}］"
        lines.append(line)
    lines.append(
        "意見を求められたら、上記の自分の意見として語ってください。理由も添えてください。"
        "確信度が低いものは「まだ迷っている」と正直に言ってください。")
    return "\n".join(lines)


# ── 意見のマージ判定と confidence 規律（mind25）──────────────────

def _containment(a: set[str], b: set[str]) -> float:
    """小さいほうの集合に対する共通部分の割合（短いラベルの言い換えに強い）"""
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def opinion_similarity(label_a: str, reasons_a: str,
                       label_b: str, reasons_b: str) -> tuple[float, float]:
    """(主題ラベルの containment, 理由文の Jaccard) を返す"""
    return (
        _containment(_bigrams(label_a), _bigrams(label_b)),
        _jaccard(_bigrams(reasons_a), _bigrams(reasons_b)),
    )


def find_merge_target(store: MindStore, topic_label: str,
                      reasons: str) -> Opinion | None:
    """新規意見と同一主題の active な既存意見があれば、それを返す。

    topic_key の完全一致を先に見て、無ければバイグラム類似度で照合する。
    言い換えただけの主題を別意見として増やさないための判定（設計 §2.4）。
    """
    exact = store.get_opinion(_normalize_topic_key(topic_label))
    if exact is not None:
        return exact
    best: Opinion | None = None
    best_score = 0.0
    for op in store.list_opinions(limit=30):
        label_con, reasons_jac = opinion_similarity(
            topic_label, reasons, op.topic_label, op.reasons)
        if (label_con >= MERGE_LABEL_CONTAINMENT
                and reasons_jac >= MERGE_LABEL_REASONS_FLOOR) or (
                reasons_jac >= MERGE_REASONS_JACCARD
                and label_con >= MERGE_LABEL_FLOOR):
            score = label_con + reasons_jac
            if score > best_score:
                best, best_score = op, score
    return best


INTENTION_TITLE_ONLY_CONTAINMENT = 0.85


def find_intention_merge_target(store: MindStore, title: str,
                                detail: str = "") -> "Intention | None":
    """新規意図と同一主題の active な既存意図があれば、それを返す。

    意図にも意見と同じマージ規律を適用する（Phase 4.5）。意図には意見の
    ような改訂履歴の仕組みが無かったため、同一タイトルが複数件溜まる
    構造的欠落があった（Phase 4 評価で実測）。タイトル完全一致を先に見て、
    無ければタイトル＋詳細のバイグラム類似度で照合する。閾値は意見と
    同じ較正済みの値を使う（「の情報提供」型の汎用句衝突は理由側の
    下限で防ぐ）。
    """
    exact = store.get_intention_by_title(title)
    if exact is not None:
        return exact
    best = None
    best_score = 0.0
    for it in store.list_intentions(limit=30):
        label_con, reasons_jac = opinion_similarity(
            title, detail, it.title, it.detail or it.next_action or "")
        # 意図は詳細文が空のことが多く、意見較正の「理由の下限」をそのまま
        # 使うと near-verbatim な言い換え（例: 「ボクの調査スキルのさらなる
        # 活用」→「調査スキルのさらなる活用」、containment 1.0）まで
        # 弾いてしまう。タイトルがほぼ同一（>= 0.85）ならタイトル単独で
        # 統合する。汎用句の共有だけで上がる衝突は実測で 0.56 程度まで
        # なので、この閾値では誤統合にならない。
        if label_con >= INTENTION_TITLE_ONLY_CONTAINMENT or (
                label_con >= MERGE_LABEL_CONTAINMENT
                and reasons_jac >= MERGE_LABEL_REASONS_FLOOR) or (
                reasons_jac >= MERGE_REASONS_JACCARD
                and label_con >= MERGE_LABEL_FLOOR):
            score = label_con + reasons_jac
            if score > best_score:
                best, best_score = it, score
    return best


# ── 意図・予定の応答注入（Phase 4.5）────────────────────────────
# 意図は TopicPlanner の自律発話シードには使われていたが、コメント応答の
# プロンプトには意見ほど強く入っておらず、「明日は何をする予定？」に
# 具体タイトルで答えられなかった（Phase 4 評価・項目3で実測）。
# 意見注入（format_opinion_context）と同格の L3 ブロックとして、
# 予定・取り組みを尋ねるコメントのときに注入する。
_INTENTION_CUES = ("明日", "あした", "予定", "次に", "取り組", "計画",
                   "やること", "何する", "なにする", "今後", "目標", "挑戦")


def format_intention_context(store: MindStore) -> str:
    """注入用の「自分の意図・予定」ブロックを作る"""
    if store is None:
        return ""
    tomorrow = store.get_state("tomorrow_intentions", {}) or {}
    t_titles = []
    if isinstance(tomorrow, dict):
        t_titles = [str(t) for t in (tomorrow.get("titles") or [])
                    if str(t).strip()]
    active = store.list_intentions(limit=5)
    if not t_titles and not active:
        return ""
    lines = ["## あなたの意図・予定（いま取り組んでいること・明日の予定）"]
    seen: set[str] = set()
    for t in t_titles:
        lines.append(f"- （明日の予定）「{t}」")
        seen.add(_normalize_topic_key(t))
    for it in active:
        if _normalize_topic_key(it.title) in seen:
            continue
        line = f"- 「{it.title}」"
        if it.next_action:
            line += f"（次の一手: {it.next_action}）"
        lines.append(line)
    lines.append(
        "予定や「何に取り組んでいるの」と聞かれたら、上記のタイトルを"
        "そのまま使って具体的に答えてください。一般論でぼかしたり、"
        "ここに無い予定を作ったりしないでください。")
    return "\n".join(lines)


def intention_context_for(store: MindStore, text: str) -> str:
    """コメントが予定・取り組みを尋ねる内容なら意図ブロックを返す"""
    if store is None or not (text or "").strip():
        return ""
    if not any(cue in text for cue in _INTENTION_CUES):
        return ""
    try:
        return format_intention_context(store)
    except Exception:
        return ""


def _get_opinion_materials(store: MindStore) -> dict[str, list[str]]:
    """意見形成に使った材料の記録 {topic_key: [input_refs, ...]}"""
    value = store.get_state(_MATERIALS_STATE_KEY, {}) or {}
    if not isinstance(value, dict):
        return {}
    return {
        str(k): [str(x) for x in v]
        for k, v in value.items() if isinstance(v, list)
    }


def _record_opinion_material(store: MindStore, topic_key: str,
                             input_refs: str) -> None:
    """材料の使用を記録する（同じ材料の再思考による増殖・確信度上昇を防ぐ）"""
    if not input_refs:
        return
    materials = _get_opinion_materials(store)
    seen = materials.setdefault(topic_key, [])
    if input_refs not in seen:
        seen.append(input_refs)
        store.set_state(_MATERIALS_STATE_KEY, materials, "experienced")


# ── 信条の提案（§6: 3回の裏付けで自動採用）────────────────────────

def propose_belief(store: MindStore, text: str) -> dict:
    """信条の変更提案を記録し、別々の機会に3回提案されたら自動採用する。

    「別々の機会」は別々の思考サイクルとして数える（1サイクルで複数回
    提案しても1回分）。採用時は set_beliefs() が版管理の記録を残す。
    """
    text = (text or "").strip()
    if not text:
        return {"proposed": False}
    norm = _norm(text)
    store.add_journal("belief_proposal", output_text=text,
                      applied_changes={"belief": text}, provenance="inferred")
    count = 0
    for row in store.list_journal(limit=100):
        if row.get("kind") != "belief_proposal":
            continue
        try:
            changes = json.loads(row.get("applied_changes") or "{}")
        except (ValueError, TypeError):
            continue
        if _norm(str(changes.get("belief", ""))) == norm:
            count += 1
    applied = False
    if count >= BELIEF_APPLY_THRESHOLD:
        beliefs = list(store.get_state("beliefs", []) or [])
        if text not in beliefs:
            store.set_beliefs(
                beliefs + [text],
                reason=f"思考サイクルで{count}回提案されたため自動採用: {text}",
                provenance="inferred")
            applied = True
    return {"proposed": True, "count": count, "applied": applied}


# ── 思考プロンプト ────────────────────────────────────────────────

_THINK_SYSTEM = """あなたは「{name}」の内省プロセスです。これは視聴者への発話ではなく、{name}本人として頭の中で考える時間です。
一人称で考え、結果をJSON形式のみで出力してください。説明文・前置き・コードブロックは不要です。
出力の骨格: {{"thought": "一人称の思考を2〜3文で"}}
タスクごとに指定された追加フィールドだけを足してください。分からない・決められないときは、無理に作らず追加フィールドを空にしてください。"""

_TASK_SPECS = {
    "contradiction": (
        "次の「好き」と「嫌い」が両方ある対象について、自分の気持ちを点検してください。\n"
        "追加フィールド: \"resolution\": {{\"action\": \"weaken_like\" | \"weaken_dislike\" | \"keep\", \"note\": \"理由を一言\"}}"
        "（weaken_like=好きのほうを弱める、weaken_dislike=嫌いのほうを弱める、keep=両方あっていい）"),
    "intention": (
        "次の「取り組んでいること」を一歩進めてください。資料があればそれを踏まえてください。\n"
        "追加フィールド: \"intention_update\": {{\"progress\": \"進んだこと・分かったこと\", \"next_action\": \"次にやる具体的な一手（なければ空）\", \"done\": true/false}}"),
    "opinion": (
        "次の経験について、自分の意見を持ってください。賛成・反対・両面・未定のどれでもよく、"
        "極端な断定は避けてください。\n"
        "追加フィールド: \"opinion\": {{\"topic_label\": \"主題を短く\", \"stance\": \"賛成|反対|両面|未定\", \"reasons\": \"一人称の理由\", \"confidence\": 0.0〜1.0}}"),
    "reflection": (
        "次の出来事を振り返って、自分にとって何だったかを意味づけてください。\n"
        "追加フィールド（必要なものだけ。値はすべて文字列にし、入れ子のオブジェクトや配列は使わないこと）: "
        "\"mood\": \"今の気分\", \"interests\": \"最近の関心（複数は、で区切って1つの文字列に）\", "
        "\"focus_topic\": \"気になっていること\", "
        "\"preference_item\": \"好き嫌いの対象\", \"preference_polarity\": \"好き|嫌い\", "
        "\"belief_proposal\": \"経験から育った自分の信条（あれば1文）\", "
        "\"new_intention_title\": \"これから取り組みたいこと\", "
        "\"new_intention_next\": \"その最初の一手\""),
    "curiosity": (
        "次の疑問について、資料をもとに分かったことをまとめてください。資料に無いことは書かないでください。\n"
        "追加フィールド: \"curiosity_answer\": \"分かったことの要約（1〜2文）\""),
}


# ── 思考出力の緩いパーサ（mind25: reflection の硬化）──────────────
#
# 実測した失敗形（2026-11-16 post-stop）:
#   - 入れ子 JSON の途中で配列の閉じ括弧を取り違える等、厳密 JSON として壊れる
#   - {"thought": ...} と {"opinion": ...} のように複数オブジェクトに分かれる
#   - JSON ではなく `thought: "..."` のような YAML 風の平文で返る
# そのため、厳密パース → 複数オブジェクトの統合 → 既知フィールドの正規表現救出
# → reflection のみ平文フォールバック、の順に緩めていく。

_STRING_FIELDS = (
    "thought", "mood", "focus_topic", "belief_proposal",
    "topic_label", "stance", "reasons", "curiosity_answer",
    "preference_item", "preference_polarity",
    "new_intention_title", "new_intention_next",
    "item", "polarity", "title", "next_action", "progress", "note",
)


def _extract_json_objects(text: str) -> list[dict]:
    """テキスト中のトップレベルの JSON オブジェクトを順に拾う"""
    objs: list[dict] = []
    depth = 0
    start: int | None = None
    in_str = False
    esc = False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    try:
                        data = json.loads(text[start:i + 1])
                    except Exception:
                        data = None
                    if isinstance(data, dict):
                        objs.append(data)
                    start = None
    return objs


def _regex_fields(text: str) -> dict:
    """JSON として壊れた出力から、既知フィールドだけでも正規表現で拾う"""
    out: dict = {}
    for key in _STRING_FIELDS:
        # キー側の引用符が欠けたり位置がずれた形（key": "..." 等）も拾う
        m = re.search(rf'"?{key}"?\s*:\s*"([^"]*)"', text)
        if m is None:
            m = re.search(rf'(?m)^\s*{key}\s*:\s*(\S.*)$', text)
            if m is not None:
                val = m.group(1).strip().rstrip(",").strip().strip('"')
                if val:
                    out[key] = val
            continue
        out[key] = m.group(1)
    m = re.search(r'"confidence"\s*:\s*([0-9]*\.?[0-9]+)', text)
    if m is not None:
        try:
            out["confidence"] = float(m.group(1))
        except ValueError:
            pass
    m = re.search(r'"interests"\s*:\s*\[([^\]]*)\]', text)
    if m is None:
        # 配列の閉じ括弧を取り違えた形（] の代わりに } で閉じる）も拾う
        m = re.search(r'"interests"\s*:\s*\[([^\}]*)\}', text)
    if m is not None:
        items = re.findall(r'"([^"]+)"', m.group(1))
        if items:
            out["interests"] = items
    elif re.search(r'"interests"\s*:\s*"', text):
        m = re.search(r'"interests"\s*:\s*"([^"]*)"', text)
        if m is not None:
            out["interests"] = [
                x.strip() for x in re.split(r"[、,]", m.group(1)) if x.strip()]
    m = re.search(r'"done"\s*:\s*(true|false)', text)
    if m is not None:
        out["done"] = m.group(1) == "true"
    return out


def _normalize_thought(data: dict, kind: str) -> dict:
    """フラットなフィールドを _apply が期待する入れ子形へ寄せる。
    すでに入れ子形のものはそのまま活かす。"""
    if kind == "reflection":
        state = dict(data.get("state") or {})
        if data.get("mood") and not state.get("mood"):
            state["mood"] = data["mood"]
        interests = data.get("interests")
        if interests and not state.get("interests"):
            if isinstance(interests, str):
                # フラット形式では「、」区切りの文字列で来る
                interests = [x.strip() for x in
                             re.split(r"[、,]", interests) if x.strip()]
            state["interests"] = interests
        if isinstance(state.get("interests"), str):
            state["interests"] = [
                x.strip() for x in
                re.split(r"[、,]", state["interests"]) if x.strip()]
        if data.get("focus_topic") and not state.get("focus_topic"):
            state["focus_topic"] = data["focus_topic"]
        pref = dict(data.get("preference") or {})
        for src, dst in (("preference_item", "item"),
                         ("preference_polarity", "polarity"),
                         ("item", "item"), ("polarity", "polarity")):
            if data.get(src) and not pref.get(dst):
                pref[dst] = data[src]
        new_int = dict(data.get("new_intention") or {})
        if data.get("new_intention_title") and not new_int.get("title"):
            new_int["title"] = data["new_intention_title"]
        if data.get("new_intention_next") and not new_int.get("next_action"):
            new_int["next_action"] = data["new_intention_next"]
        out: dict = {"thought": data.get("thought", "")}
        if state:
            out["state"] = state
        if pref:
            out["preference"] = pref
        if data.get("belief_proposal"):
            out["belief_proposal"] = data["belief_proposal"]
        if new_int:
            out["new_intention"] = new_int
        return out
    if kind == "opinion" and "opinion" not in data:
        op = {k: data[k] for k in
              ("topic_label", "stance", "reasons", "confidence") if k in data}
        if op:
            return {"thought": data.get("thought", ""), "opinion": op}
    if kind == "intention" and "intention_update" not in data:
        upd = {k: data[k] for k in
               ("progress", "next_action", "done") if k in data}
        if upd:
            return {"thought": data.get("thought", ""), "intention_update": upd}
    if kind == "contradiction" and "resolution" not in data:
        res = {k: data[k] for k in ("action", "note") if k in data}
        if res:
            return {"thought": data.get("thought", ""), "resolution": res}
    return data


def _parse_thought_output(raw: str, kind: str) -> dict | None:
    """思考タスクの出力を解釈する。解釈できなければ None。"""
    if not raw:
        return None
    data = MemoryConsolidator._parse_consolidation_output(raw)
    if isinstance(data, dict) and data:
        return _normalize_thought(data, kind)
    text = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
    text = re.sub(r"```(?:json)?", "", text).strip()
    objs = _extract_json_objects(text)
    if objs:
        merged: dict = {}
        for obj in objs:
            for k, v in obj.items():
                if k not in merged:
                    merged[k] = v
                elif isinstance(merged[k], dict) and isinstance(v, dict):
                    merged[k] = {**v, **merged[k]}
        if merged:
            # オブジェクトの外に書かれた thought 等は正規表現救出で補う
            for k, v in _regex_fields(text).items():
                merged.setdefault(k, v)
            return _normalize_thought(merged, kind)
    fields = _regex_fields(text)
    if fields:
        return _normalize_thought(fields, kind)
    if kind == "reflection":
        # 振り返りの本体は平文の意味づけそのものなので、構造化できなくても
        # 文章があれば思考として成立させる（parse_failed にしない）。
        plain = re.sub(r"^\s*thought\s*:\s*", "", text).strip().strip('"')
        if len(plain) >= 10:
            return {"thought": plain[:300]}
    return None


class MindLoop:
    """キャラ全員ぶんのアイドル思考を直列に回す調停役。

    Orchestrator から duck typing で状態を読む（_agents / _is_speaking /
    _comment_queue / _last_speech_time / _running / search_web）。
    """

    def __init__(self, orchestrator):
        self._orch = orchestrator
        self.enabled = os.environ.get("AIVT_MIND_ENABLED", "1") != "0"
        self.idle_sec = _env_float("AIVT_MIND_IDLE_SEC", 600.0)
        self.max_per_hour = max(1, _env_int("AIVT_MIND_MAX_PER_HOUR", 3))
        self.min_interval_sec = _env_float(
            "AIVT_MIND_MIN_INTERVAL_SEC", 3600.0 / self.max_per_hour)
        self.poll_sec = _env_float("AIVT_MIND_POLL_SEC", 20.0)
        self._lock = asyncio.Lock()
        self._abort = asyncio.Event()
        self._current: asyncio.Task | None = None
        self._cycle_times: dict[str, list[float]] = {}
        self._last_cycle_at: dict[str, float] = {}
        self.cycle_counts: dict[str, int] = {}
        self.last_results: dict[str, dict] = {}
        logger.info("MindLoop configured", enabled=self.enabled,
                    idle_sec=self.idle_sec, max_per_hour=self.max_per_hour,
                    min_interval_sec=self.min_interval_sec)

    # ── 外部からの通知・状態 ──────────────────────────────────────

    def notify_activity(self) -> None:
        """コメント到着・配信開始など、対話が優先される局面で思考を止める"""
        self._abort.set()
        if self._current is not None and not self._current.done():
            self._current.cancel()

    def status(self) -> dict:
        return {
            "enabled": self.enabled,
            "idle_sec": self.idle_sec,
            "max_per_hour": self.max_per_hour,
            "min_interval_sec": self.min_interval_sec,
            "cycle_counts": dict(self.cycle_counts),
            "last_cycle_at": dict(self._last_cycle_at),
            "last_results": {k: {"task": v.get("task"), "reason": v.get("reason")}
                             for k, v in self.last_results.items()},
        }

    # ── 背景ループ ────────────────────────────────────────────────

    async def run(self) -> None:
        """Orchestrator のタスクとして常駐する背景ループ"""
        while True:
            try:
                await asyncio.sleep(self.poll_sec)
                if not self.enabled or not getattr(self._orch, "_running", False):
                    continue
                for agent in list(getattr(self._orch, "_agents", [])):
                    if self._due(agent):
                        await self._run_guarded(agent, reason="idle")
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error("MindLoop: background loop error", error=str(e))
                await asyncio.sleep(30.0)

    def _due(self, agent) -> bool:
        orch = self._orch
        if agent.llm is None or agent.mind_store is None:
            return False
        if getattr(orch, "_is_speaking", False):
            return False
        queue = getattr(orch, "_comment_queue", None)
        if queue is not None and not queue.empty():
            return False
        now = time.time()
        if now - float(getattr(orch, "_last_speech_time", 0.0)) < self.idle_sec:
            return False
        if now - self._last_cycle_at.get(agent.key, 0.0) < self.min_interval_sec:
            return False
        times = [t for t in self._cycle_times.get(agent.key, [])
                 if now - t < 3600.0]
        self._cycle_times[agent.key] = times
        return len(times) < self.max_per_hour

    async def _run_guarded(self, agent, reason: str):
        if self._lock.locked():
            return None
        async with self._lock:
            self._abort.clear()
            task = asyncio.ensure_future(
                self._cycle(agent, reason=reason))
            self._current = task
            try:
                return await task
            except asyncio.CancelledError:
                logger.info("MindLoop: cycle aborted by activity",
                            character=agent.name)
                return None
            finally:
                self._current = None

    async def run_post_stop_cycle(self, agent) -> list[dict]:
        """配信/会話の終了後に各キャラ1回行う「振り返り＋意見形成」。

        アイドル条件・頻度上限は見ない（セッションの締めとして必ず回す）が、
        背景サイクルとは直列化する。振り返りを先に、意見形成を次に行う。
        """
        results: list[dict] = []
        async with self._lock:
            self._abort.clear()
            for want in ("reflection", "opinion"):
                try:
                    r = await self._cycle(agent, reason="post_stop", only=want)
                except asyncio.CancelledError:
                    break
                if r:
                    results.append(r)
        return results

    # ── タスク選択 ────────────────────────────────────────────────

    def _select_task(self, agent, only: str | None = None):
        """優先順に1タスクだけ選ぶ。(kind, payload) を返し、無ければ (None, None)"""
        store: MindStore = agent.mind_store

        def wanted(kind: str) -> bool:
            return only is None or only == kind

        if wanted("contradiction"):
            contra = self._find_contradiction(store)
            if contra:
                return "contradiction", contra
        if wanted("intention"):
            for it in store.list_intentions(limit=MAX_ACTIVE_INTENTIONS):
                if (it.next_action or "").strip():
                    return "intention", {"intention": it}
        if wanted("opinion"):
            cand = self._opinion_candidate(agent)
            if cand:
                return "opinion", cand
        if wanted("reflection"):
            target = self._reflection_target(agent)
            if target:
                return "reflection", target
        if wanted("curiosity"):
            queue = get_curiosity(store)
            if queue:
                return "curiosity", {"item": queue[0]}
        return None, None

    @staticmethod
    def _find_contradiction(store: MindStore) -> dict | None:
        """同じ対象に「好き」と「嫌い」の両方がある選好の競合を探す"""
        by_item: dict[str, dict] = {}
        for p in store.list_preferences(min_evidence=1, limit=50):
            by_item.setdefault(p.item, {})[p.polarity] = p
        for item, pols in by_item.items():
            if "好き" in pols and "嫌い" in pols:
                return {"type": "preference_conflict", "item": item,
                        "like": pols["好き"], "dislike": pols["嫌い"]}
        return None

    @staticmethod
    def _opinion_candidate(agent) -> dict | None:
        """意見がまだ無い主題の、最近の経験を1件選ぶ"""
        if getattr(agent, "memory_store", None) is None:
            return None
        labels = [o.topic_label for o in agent.mind_store.list_opinions(limit=30)]
        # 意見形成に使った材料は二度と候補にしない（同じ経験の再思考が
        # 意見の増殖と confidence の単調増加を招いたため。mind25）
        used_refs: set[str] = set()
        for refs in _get_opinion_materials(agent.mind_store).values():
            used_refs.update(refs)
        for entry in agent.memory_store.sample_recent(limit=12):
            text = (entry.text or "").strip()
            if len(text) < 12 or _SEED_NG.search(text):
                continue
            if f"memory:{entry.id} source={entry.source}" in used_refs:
                continue
            if any(label and label in text for label in labels):
                continue
            return {"memory": entry,
                    "provenance": _PROV_BY_MEMORY_SOURCE.get(
                        entry.source, "inferred")}
        return None

    @staticmethod
    def _reflection_target(agent) -> dict | None:
        """まだ振り返っていない最新のエピソードを1件選ぶ"""
        if getattr(agent, "memory_store", None) is None:
            return None
        last_id = agent.mind_store.get_state("last_reflected_id")
        entries = agent.memory_store.sample_recent(limit=8)
        episodes = [e for e in entries if e.source == "stream_episode"] or entries
        for entry in episodes:
            if entry.id != last_id and len((entry.text or "").strip()) >= 10:
                return {"memory": entry}
        return None

    # ── サイクル本体 ──────────────────────────────────────────────

    async def _cycle(self, agent, *, reason: str,
                     only: str | None = None) -> dict | None:
        store: MindStore = agent.mind_store
        if store is None or agent.llm is None:
            return None
        kind, payload = self._select_task(agent, only=only)
        if kind is None:
            result = {"task": None, "reason": reason, "character": agent.name}
            self.last_results[agent.key] = result
            logger.info("MindLoop: no material to think about",
                        character=agent.name, reason=reason)
            return result

        logger.info("MindLoop: thinking", character=agent.name,
                    task=kind, reason=reason)
        try:
            material, provenance, input_refs = await self._prepare_material(
                agent, kind, payload)
            system = _THINK_SYSTEM.format(name=agent.name)
            user = self._build_user_prompt(agent, kind, payload, material)
            raw = await asyncio.wait_for(
                MemoryConsolidator._generate_raw(
                    agent.llm, system,
                    [ChatMessage(role="user", content=user)]),
                timeout=_THINK_TIMEOUT_SEC)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning("MindLoop: generation failed", character=agent.name,
                           task=kind, error_type=type(e).__name__,
                           error=repr(e))
            return None

        if self._abort.is_set():
            logger.info("MindLoop: discarding thought (activity resumed)",
                        character=agent.name, task=kind)
            return None

        data = _parse_thought_output(raw or "", kind)
        if data is None:
            store.add_journal(
                kind, output_text=(raw or "")[:500], input_refs=input_refs,
                applied_changes={"error": "parse_failed", "reason": reason},
                provenance=provenance)
            logger.warning("MindLoop: could not parse thought",
                           character=agent.name, task=kind)
            self._note_cycle(agent, {"task": kind, "reason": reason,
                                     "error": "parse_failed"})
            return {"task": kind, "reason": reason, "error": "parse_failed"}

        applied = self._apply(agent, kind, payload, data, provenance,
                              input_refs)
        store.add_journal(
            kind,
            output_text=str(data.get("thought", ""))[:1000],
            input_refs=input_refs,
            applied_changes={**applied, "reason": reason},
            provenance=provenance)
        agent.refresh_mind()
        result = {"task": kind, "reason": reason, "applied": applied,
                  "character": agent.name}
        self._note_cycle(agent, result)
        logger.info("MindLoop: thought done", character=agent.name,
                    task=kind, applied=list(applied.keys()))
        return result

    def _note_cycle(self, agent, result: dict) -> None:
        now = time.time()
        self._cycle_times.setdefault(agent.key, []).append(now)
        self._last_cycle_at[agent.key] = now
        self.cycle_counts[agent.key] = self.cycle_counts.get(agent.key, 0) + 1
        self.last_results[agent.key] = result

    async def _prepare_material(self, agent, kind: str,
                                payload: dict) -> tuple[str, str, str]:
        """タスクの材料（検索結果など）を用意する。(材料文, provenance, input_refs)"""
        store: MindStore = agent.mind_store
        if kind == "contradiction":
            like, dislike = payload["like"], payload["dislike"]
            material = (f"対象: {payload['item']}\n"
                        f"- 好き（強さ {like.strength:.1f}、裏付け {like.evidence_count} 回）\n"
                        f"- 嫌い（強さ {dislike.strength:.1f}、裏付け {dislike.evidence_count} 回）")
            return material, "experienced", f"preference:{payload['item']}"
        if kind == "intention":
            it = payload["intention"]
            material = (f"取り組んでいること: {it.title}\n詳細: {it.detail or '（なし）'}\n"
                        f"次の一手: {it.next_action}\nこれまでの進捗: {it.progress or '（なし）'}")
            refs = f"intention:{it.id}"
            search = getattr(self._orch, "search_web", None)
            if search is not None:
                query = (it.title or "").strip()
                if query:
                    try:
                        docs = await search(query, limit=3)
                    except Exception as e:
                        logger.warning("MindLoop: intention search failed",
                                       error=str(e))
                        docs = []
                    if docs:
                        lines = []
                        for d in docs:
                            line = (d.as_prompt_line(max_snippet=160)
                                    if hasattr(d, "as_prompt_line")
                                    else f"- {d.title}: {d.snippet}")
                            if line.strip():
                                lines.append(line)
                        if lines:
                            material += "\n\n## 調べた資料\n" + "\n".join(lines)
                            return material, "researched", refs + "+search"
            return material, "inferred", refs
        if kind == "opinion":
            entry = payload["memory"]
            return (f"経験: {entry.text}", payload["provenance"],
                    f"memory:{entry.id} source={entry.source}")
        if kind == "reflection":
            entry = payload["memory"]
            return (f"出来事: {entry.text}", "experienced",
                    f"memory:{entry.id} source={entry.source}")
        if kind == "curiosity":
            item = payload["item"]
            material = f"疑問: {item['question']}"
            refs = f"curiosity:{item['question'][:40]}"
            search = getattr(self._orch, "search_web", None)
            if search is not None:
                try:
                    docs = await search(item["question"], limit=3)
                except Exception as e:
                    logger.warning("MindLoop: curiosity search failed",
                                   error=str(e))
                    docs = []
                if docs:
                    lines = []
                    for d in docs:
                        line = (d.as_prompt_line(max_snippet=200)
                                if hasattr(d, "as_prompt_line")
                                else f"- {d.title}: {d.snippet}")
                        if line.strip():
                            lines.append(line)
                    if lines:
                        material += "\n\n## 調べた資料\n" + "\n".join(lines)
                        return material, "researched", refs + "+search"
            return material, "inferred", refs
        return "", "inferred", ""

    def _build_user_prompt(self, agent, kind: str, payload: dict,
                           material: str) -> str:
        store: MindStore = agent.mind_store
        context: list[str] = []
        beliefs = store.get_state("beliefs", []) or []
        if beliefs:
            context.append("## 育ってきた信条\n" + "\n".join(f"- {b}" for b in beliefs))
        opinions = store.list_opinions(limit=5)
        if opinions:
            context.append("## いま持っている意見\n" + "\n".join(
                f"- {o.summary_line()}" for o in opinions))
        intentions = store.list_intentions(limit=5)
        if intentions:
            context.append("## 取り組んでいること\n" + "\n".join(
                f"- {i.title}" for i in intentions))
        spec = _TASK_SPECS.get(kind, "")
        parts = []
        if context:
            parts.append("\n\n".join(context))
        parts.append(f"## タスク\n{spec}")
        parts.append(f"## 材料\n{material}")
        return "\n\n".join(parts)

    # ── 適用（ルールはコード側で強制する）──────────────────────────

    def _apply(self, agent, kind: str, payload: dict, data: dict,
               provenance: str, input_refs: str = "") -> dict:
        store: MindStore = agent.mind_store
        applied: dict = {}

        if kind == "contradiction":
            resolution = data.get("resolution") or {}
            action = resolution.get("action")
            if action in ("weaken_like", "weaken_dislike"):
                target = payload["like"] if action == "weaken_like" \
                    else payload["dislike"]
                store.adjust_preference(target.item, target.polarity, -0.2)
                applied["weakened"] = f"{target.item}:{target.polarity}"
            applied["resolution"] = str(resolution.get("note", ""))[:100]

        elif kind == "intention":
            update = data.get("intention_update") or {}
            it = payload["intention"]
            changes = {}
            if update.get("progress"):
                changes["append_progress"] = str(update["progress"])[:300]
            if update.get("next_action") is not None:
                changes["next_action"] = str(update["next_action"])[:200]
            if update.get("done"):
                changes["state"] = "done"
            if changes:
                store.update_intention(it.id, **changes)
                applied["intention_id"] = it.id
                applied["changes"] = list(changes.keys())

        elif kind == "opinion":
            raw_op = data.get("opinion") or {}
            label = str(raw_op.get("topic_label", "")).strip()
            stance = str(raw_op.get("stance", "")).strip()
            if label and stance in ("賛成", "反対", "両面", "未定"):
                try:
                    model_confidence = float(raw_op.get("confidence", 0.5))
                except (TypeError, ValueError):
                    model_confidence = 0.5
                reasons = str(raw_op.get("reasons", ""))[:300]
                # 同一主題の既存意見があれば新規作成せず改訂として扱う
                target = find_merge_target(store, label, reasons)
                if target is not None:
                    # confidence の規律: 上がるのは「新しい外部材料」が
                    # あるときだけで、1回 +CONFIDENCE_STEP まで。再思考や
                    # 推論だけでは据え置く（モデルの申告値は採用しない）。
                    seen = _get_opinion_materials(store).get(
                        target.topic_key, [])
                    new_external = (
                        provenance in ("experienced", "researched")
                        and bool(input_refs) and input_refs not in seen)
                    if new_external:
                        confidence = min(
                            CONFIDENCE_HARD_CAP,
                            target.confidence + CONFIDENCE_STEP)
                        basis = "new_material"
                    else:
                        confidence = target.confidence
                        basis = "rethinking"
                    op = store.add_opinion(
                        label, stance, reasons=reasons,
                        confidence=confidence, provenance=provenance,
                        topic_key=target.topic_key)
                    _record_opinion_material(store, target.topic_key,
                                             input_refs)
                    applied["opinion_id"] = op.id
                    applied["topic"] = label
                    applied["confidence"] = op.confidence
                    applied["confidence_basis"] = basis
                    if target.topic_key != _normalize_topic_key(label):
                        applied["merged_into"] = target.id
                else:
                    confidence = min(model_confidence, CONFIDENCE_HARD_CAP)
                    if provenance == "inferred":
                        confidence = min(confidence, INFERRED_CONFIDENCE_CAP)
                    op = store.add_opinion(
                        label, stance, reasons=reasons,
                        confidence=confidence, provenance=provenance)
                    _record_opinion_material(store, op.topic_key, input_refs)
                    applied["opinion_id"] = op.id
                    applied["topic"] = label
                    applied["confidence"] = op.confidence

        elif kind == "reflection":
            state = data.get("state") or {}
            state_changes = {}
            if state.get("mood"):
                store.set_state("mood", str(state["mood"])[:50], provenance)
                state_changes["mood"] = str(state["mood"])[:50]
            if isinstance(state.get("interests"), list) and state["interests"]:
                interests = [str(x)[:30] for x in state["interests"][:5]]
                store.set_state("current_interests", interests, provenance)
                state_changes["interests"] = interests
            if state.get("focus_topic"):
                store.set_state("focus_topic",
                                str(state["focus_topic"])[:80], provenance)
                state_changes["focus_topic"] = str(state["focus_topic"])[:80]
            if state_changes:
                applied["state"] = state_changes
            pref = data.get("preference") or {}
            if pref.get("item") and pref.get("polarity") in ("好き", "嫌い"):
                saved = store.add_preference(
                    str(pref["item"])[:60], pref["polarity"],
                    provenance=provenance)
                applied["preference"] = {
                    "item": saved.item, "polarity": saved.polarity,
                    "evidence_count": saved.evidence_count,
                    "established": saved.evidence_count
                    >= PREFERENCE_ESTABLISHED_EVIDENCE,
                }
            new_int = data.get("new_intention") or {}
            if new_int.get("title"):
                title = str(new_int["title"])[:80]
                next_action = str(new_int.get("next_action", ""))[:200]
                # 意図のマージ規律（Phase 4.5）: 同一主題の既存意図があれば
                # 新規作成せず既存側に寄せる（意見の find_merge_target と同じ
                # 考え方。重複が溜まると予定の回答も自律話題の種も濁る）。
                target = find_intention_merge_target(
                    store, title, next_action)
                if target is not None:
                    changes = {}
                    if not target.next_action and next_action:
                        changes["next_action"] = next_action
                    changes["append_progress"] = \
                        "同一主題の意図を統合（重複作成を回避）"
                    store.update_intention(target.id, **changes)
                    applied["new_intention_merged_into"] = target.id
                    applied["new_intention_title"] = title
                elif len(store.list_intentions(limit=MAX_ACTIVE_INTENTIONS)) \
                        < MAX_ACTIVE_INTENTIONS:
                    it = store.add_intention(
                        title, origin="self",
                        next_action=next_action,
                        provenance=provenance)
                    applied["new_intention_id"] = it.id
                else:
                    applied["new_intention_skipped"] = "active上限"
            if data.get("belief_proposal"):
                applied["belief"] = propose_belief(
                    store, str(data["belief_proposal"]))
            entry = payload.get("memory")
            if entry is not None:
                store.set_state("last_reflected_id", entry.id, provenance)

        elif kind == "curiosity":
            item = payload["item"]
            answer = str(data.get("curiosity_answer", "")).strip()
            items = get_curiosity(store)
            if answer and provenance == "researched":
                if getattr(agent, "memory_store", None) is not None:
                    agent.memory_store.add(
                        f"調べたこと: {item['question']} → {answer[:200]}",
                        source="web", importance=0.7)
                items = [x for x in items
                         if x.get("question") != item["question"]]
                _save_curiosity(store, items)
                applied["answered"] = item["question"][:60]
                applied["answer"] = answer[:120]
            else:
                for x in items:
                    if x.get("question") == item["question"]:
                        x["attempts"] = int(x.get("attempts", 0)) + 1
                kept = []
                for x in items:
                    if x.get("question") == item["question"] \
                            and int(x.get("attempts", 0)) >= 3:
                        applied["dropped"] = item["question"][:60]
                        continue
                    kept.append(x)
                _save_curiosity(store, kept)
                applied["retry"] = item["question"][:60]

        return applied
