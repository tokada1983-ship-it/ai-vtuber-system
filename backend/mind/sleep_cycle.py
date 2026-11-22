"""SleepCycle: 睡眠時統合（Phase 3）

設計: AUTONOMOUS_MIND_DESIGN_V2.md §7（睡眠時統合）/ §5（自己物語）/ §9

1日の終わりに、その日の経験を「個」へ固定化する重い統合処理。
夢（再結合）は作らない（2026-11-16 決定: 出所管理が完全に効く確信が
持てるまで見送り。dreamed の生成はこのモジュールにも無い）。

手順（§7 の固定順序）:
  1. 固定化: 前回睡眠以降のエピソードから重要なものを長期層へ昇格させ
     （MemoryStore の pinned = 忘却対象外、が既存の長期層）、軽いものは
     importance の上限を下げて減衰を早める。判定は規則ベースで決定的にし、
     何度実行しても同じ結果になる（冪等）ようにする。
  2. 意見の見直し: 新しい経験が言及している主題の意見を、本人のモデルに
     見直させる。改訂は MindStore.add_opinion の superseded で履歴を残し、
     confidence は mind25 の規律（再思考で上げない・1回の下げ幅に上限）を
     睡眠側でも守る。見直した材料は記録し、同じ材料では再見直ししない。
  3. 意図の棚卸し: 24時間以上触っていない意図の salience を少し下げ、
     active の上限（5件）を超えたら低いものから dropped にする。
     「明日の意図」は salience 上位2件を self_state に整える。
  4. 自己物語の小改訂: 最新の物語より後に新しい出来事がある場合のみ。
     全面改訂は週次（前回から7日以上）でのみ行い、日次は既存文を土台に
     新しい出来事を織り込む小改訂に留める。生成文は「新しい文が材料と
     照合できるか」を機械チェックし、材料に無い出来事が混ざったら
     改訂ごと棄却する（物語の幻覚が一番危ないため）。
     生成プロンプトにはゲート上限の9割を文字数予算として明示し、
     長さ超過による棄却を生成側で防ぐ。同一材料で棄却された改訂は、
     新しい材料が増えるまで再試行しない（クールダウン）。
  5. 思考ジャーナルの世代管理: 30日より古い journal を月次サマリへ
     圧縮してアーカイブテーブルへ退避する（MindStore.archive_journal）。

全ステップの結果は thought_journal に kind="sleep" で1件にまとめて残す。

スケジューリング:
  - SleepManager がアプリ常駐でポーリングし、JST 03:00〜04:00 の枠で
    1日1回（その日まだなら）＋キャッチアップ（前回完了から24時間以上
    経ち、未処理の材料がある場合）を、アイドル時に実行する。
  - JST は PC のローカル時計ではなく UTC+9 で明示的に計算する
    （PC 時計は JST とずれているため）。
  - 処理対象期間（watermark）は全ステップが完走したときだけ前進させる。
    途中で対話が再開したらステップ間で打ち切り、watermark は進めない
    （各ステップは冪等なので、次回にやり直しても二重に効かない）。
  - MindLoop とは同じロックで直列化する（モデルスワップ回避）。

設定（config.yaml の mind セクションが正。環境変数があれば移行互換のため優先）:
  AIVT_SLEEP_ENABLED    "0" で背景スケジューラ無効（既定: 有効）
  AIVT_SLEEP_POLL_SEC   ポーリング間隔（既定: 60）
"""
from __future__ import annotations

import asyncio
import hashlib
import os
import re
import time
from datetime import datetime, timedelta, timezone

from backend.logger import get_logger
from backend.memory.memory_consolidator import MemoryConsolidator
from backend.memory.mind_store import MindStore
from backend.mind.mind_loop import (MAX_ACTIVE_INTENTIONS,
                                    MERGE_REASONS_JACCARD,
                                    _TASK_SPECS, _THINK_SYSTEM, _bigrams,
                                    _containment, _get_opinion_materials,
                                    _norm, _parse_thought_output,
                                    _record_opinion_material, opinion_similarity,
                                    provenance_for_memory_source)
from backend.models import ChatMessage

logger = get_logger(__name__)

JST = timezone(timedelta(hours=9))

_SLEEP_STATE_KEY = "sleep_state"
_TOMORROW_KEY = "tomorrow_intentions"
_GEN_TIMEOUT_SEC = 240.0

# ステップ1（固定化）の規則
PROMOTE_IMPORTANCE = 1.6        # この重要度があり、かつ参照実績があれば昇格
PROMOTE_IMPORTANCE_SOLO = 2.5   # この重要度単独でも昇格（繰り返し強化された記憶）
DEMOTE_CEILING = 0.5            # 軽い記憶は importance をここまで下げて減衰を早める
# ステップ2（意見見直し）の上限と confidence の下げ幅上限
OPINION_REVIEW_LIMIT = 2
CONFIDENCE_DROP_MAX = 0.3
# ステップ3（意図の棚卸し）
STALLED_SEC = 24 * 3600
SALIENCE_STEP = 0.1
# ステップ4（自己物語）
NARRATIVE_FULL_INTERVAL_SEC = 7 * 86400
NARRATIVE_SENTENCE_OVERLAP_MIN = 0.3
NARRATIVE_BUDGET_RATIO = 0.9    # 生成プロンプトに示す文字数予算 = ゲート上限の9割
_NARRATIVE_ATTEMPT_KEY = "narrative_attempt"
# ジャーナル世代管理
JOURNAL_ARCHIVE_DAYS = 30


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "") or default)
    except ValueError:
        return default


def _jst_date(ts: float) -> str:
    return datetime.fromtimestamp(ts, JST).strftime("%Y-%m-%d")


def _jst_hour(ts: float) -> int:
    return datetime.fromtimestamp(ts, JST).hour


def _in_nightly_window(ts: float) -> bool:
    return _jst_hour(ts) in (3,)


# ── 睡眠の状態とキャッチアップ判定 ────────────────────────────────

def get_sleep_state(store: MindStore) -> dict:
    value = store.get_state(_SLEEP_STATE_KEY, {}) or {}
    return value if isinstance(value, dict) else {}


def _has_new_material(store: MindStore, memory_store, watermark: float) -> bool:
    """watermark より後に生まれた・更新された材料があるか"""
    if memory_store is not None:
        for e in memory_store.all():
            if e.created_at > watermark:
                return True
    for op in store.list_opinions(limit=30):
        if op.updated_at > watermark:
            return True
    for it in store.list_intentions(state="active", limit=10):
        if it.updated_at > watermark:
            return True
    for it in store.list_intentions(state="paused", limit=10):
        if it.updated_at > watermark:
            return True
    return False


def evaluate_sleep_due(store: MindStore, memory_store,
                       now: float) -> tuple[bool, str]:
    """睡眠サイクルを実行すべきかと、その理由を返す。

    理由: "nightly"（JST深夜枠でその日まだ）/ "catchup"（24時間以上
    未実行）/ "initial"（未実行で材料あり）。材料が無ければ常に False。
    """
    state = get_sleep_state(store)
    last_completed = float(state.get("last_completed_at") or 0)
    watermark = float(state.get("last_period_end") or 0)
    if not _has_new_material(store, memory_store, watermark):
        return False, ""
    if not last_completed:
        return True, "initial"
    if _in_nightly_window(now) \
            and state.get("last_sleep_date") != _jst_date(now):
        return True, "nightly"
    if now - last_completed >= 24 * 3600:
        return True, "catchup"
    return False, ""


# ── ステップ1: 固定化 ─────────────────────────────────────────────

def _step1_consolidate(agent, period_start: float,
                       period_end: float) -> dict:
    """期間内のエピソードを長期層へ昇格／減衰加速する（規則ベース・冪等）。

    昇格 = pinned 化（MemoryStore の忘却から外れる＝既存の長期層）。
    減衰加速 = importance を DEMOTE_CEILING まで下げる（上限設定なので
    再実行しても下がり続けない）。
    """
    mem = agent.memory_store
    promoted: list[int] = []
    demoted: list[int] = []
    for e in mem.all():
        if not (period_start < e.created_at <= period_end):
            continue
        if e.source == "manual" or e.pinned:
            continue
        if (e.importance >= PROMOTE_IMPORTANCE and e.use_count >= 1) \
                or e.importance >= PROMOTE_IMPORTANCE_SOLO:
            if mem.set_pinned(e.id, True):
                promoted.append(e.id)
        elif e.importance > DEMOTE_CEILING and e.use_count == 0:
            if mem.set_importance(e.id, DEMOTE_CEILING):
                demoted.append(e.id)
    return {"promoted": promoted, "demoted": demoted}


# ── ステップ2: 意見の見直し ───────────────────────────────────────

async def _step2_review_opinions(agent, period_start: float,
                                 period_end: float) -> dict:
    store: MindStore = agent.mind_store
    mem = agent.memory_store
    period_mems = [
        e for e in mem.all()
        if period_start < e.created_at <= period_end
        and len((e.text or "").strip()) >= 10
    ]
    reviewed: list[dict] = []
    if not period_mems:
        return {"reviewed": reviewed, "note": "期間内に新しい経験が無い"}

    materials = _get_opinion_materials(store)
    for op in store.list_opinions(limit=20):
        if len(reviewed) >= OPINION_REVIEW_LIMIT:
            break
        target = None
        target_ref = ""
        for e in period_mems:
            ref = f"memory:{e.id} source={e.source}"
            if ref in materials.get(op.topic_key, []):
                continue
            label_bi = _bigrams(op.topic_label)
            if label_bi and _containment(label_bi, _bigrams(e.text)) >= 0.5:
                target, target_ref = e, ref
                break
        if target is None:
            continue

        system = _THINK_SYSTEM.format(name=agent.name)
        user = (
            f"## タスク\n{_TASK_SPECS['opinion']}\n\n"
            f"## 見直す意見（いま持っているもの）\n"
            f"主題: {op.topic_label}\n立場: {op.stance}\n"
            f"理由: {op.reasons}\n確信度: {op.confidence:.2f}\n\n"
            f"## 新しい経験\n{target.text}\n\n"
            "この新しい経験を踏まえて意見を見直してください。"
            "考えが変わらないなら、立場も理由も今の意見と同じ内容を"
            "返してください。")
        try:
            raw = await asyncio.wait_for(
                MemoryConsolidator._generate_raw(
                    agent.llm, system,
                    [ChatMessage(role="user", content=user)]),
                timeout=_GEN_TIMEOUT_SEC)
        except Exception as e:
            logger.warning("SleepCycle: opinion review generation failed",
                           character=agent.name, topic=op.topic_label,
                           error=repr(e))
            reviewed.append({"topic": op.topic_label,
                             "action": "error", "error": repr(e)[:120]})
            continue

        data = _parse_thought_output(raw or "", "opinion")
        raw_op = (data or {}).get("opinion") or {}
        stance = str(raw_op.get("stance", "")).strip()
        if stance not in ("賛成", "反対", "両面", "未定"):
            reviewed.append({"topic": op.topic_label, "action": "unparsed"})
            _record_opinion_material(store, op.topic_key, target_ref)
            continue
        reasons = str(raw_op.get("reasons", ""))[:300]
        _, reasons_jac = opinion_similarity(
            op.topic_label, op.reasons, op.topic_label, reasons)
        provenance = provenance_for_memory_source(
            target.source, context="sleep_opinion_review")
        if stance == op.stance and reasons_jac >= MERGE_REASONS_JACCARD:
            _record_opinion_material(store, op.topic_key, target_ref)
            reviewed.append({"topic": op.topic_label, "action": "keep"})
            continue
        try:
            model_conf = float(raw_op.get("confidence", op.confidence))
        except (TypeError, ValueError):
            model_conf = op.confidence
        if stance == "未定" and op.stance != "未定":
            new_conf = max(0.1, op.confidence - 0.2)
        else:
            new_conf = min(op.confidence, model_conf)
            new_conf = max(op.confidence - CONFIDENCE_DROP_MAX, new_conf)
            new_conf = max(0.1, min(0.9, new_conf))
        saved = store.add_opinion(
            op.topic_label, stance, reasons=reasons, confidence=new_conf,
            provenance=provenance, topic_key=op.topic_key)
        _record_opinion_material(store, op.topic_key, target_ref)
        reviewed.append({
            "topic": op.topic_label, "action": "revised",
            "from": op.stance, "to": stance,
            "confidence": [op.confidence, saved.confidence],
            "revision": saved.revision,
        })
    return {"reviewed": reviewed}


# ── ステップ3: 意図の棚卸し ───────────────────────────────────────

def _step3_review_intentions(agent, now: float) -> dict:
    store: MindStore = agent.mind_store
    adjusted: list[dict] = []
    dropped: list[dict] = []
    for it in store.list_intentions(state="active", limit=50):
        if it.updated_at < now - STALLED_SEC and it.salience > 0.1:
            new_sal = max(0.1, round(it.salience - SALIENCE_STEP, 3))
            store.update_intention(
                it.id, salience=new_sal,
                append_progress="睡眠時の棚卸し: しばらく進んでいないため"
                                "優先度を少し下げた")
            adjusted.append({"id": it.id, "title": it.title,
                             "salience": [it.salience, new_sal]})
    active = store.list_intentions(state="active", limit=50)
    while len(active) > MAX_ACTIVE_INTENTIONS:
        victim = min(active, key=lambda i: (i.salience, i.updated_at, i.id))
        store.update_intention(
            victim.id, state="dropped",
            append_progress="睡眠時の棚卸し: active の上限を超えたため見送り")
        dropped.append({"id": victim.id, "title": victim.title})
        active = [a for a in active if a.id != victim.id]
    top = sorted(active,
                 key=lambda i: (-i.salience, -i.updated_at, i.id))[:2]
    tomorrow = {"date_jst": _jst_date(now + 86400),
                "ids": [i.id for i in top],
                "titles": [i.title for i in top]}
    store.set_state(_TOMORROW_KEY, tomorrow, "inferred")
    return {"adjusted": adjusted, "dropped": dropped, "tomorrow": tomorrow}


# ── ステップ4: 自己物語の小改訂 ───────────────────────────────────

def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[。！？])|\n+", text or "")
    return [p.strip() for p in parts if p.strip()]


def _verify_narrative(new_text: str, old_text: str, materials_text: str,
                      full: bool) -> tuple[bool, list[str]]:
    """改訂文の照合: 長さの範囲と、新しい文が材料と重なること。

    材料に無い出来事の混入（物語の幻覚）を機械的に弾くための関門。
    既存の文の使い回しは無条件に通し、新出の文だけを材料と照合する。
    """
    problems: list[str] = []
    n = len(new_text)
    if full:
        if not (350 <= n <= 950):
            problems.append(f"length {n} out of bounds (full: 350-950)")
    else:
        lo = int(len(old_text) * 0.7)
        hi = int(len(old_text) * 1.3)
        if not (lo <= n <= hi):
            problems.append(f"length {n} out of bounds (minor: {lo}-{hi})")
        if not (250 <= n <= 1100):
            problems.append(f"length {n} out of bounds (abs: 250-1100)")
    old_sents = {_norm(s) for s in _sentences(old_text)}
    mat_bi = _bigrams(materials_text)
    for s in _sentences(new_text):
        if len(s) < 8 or _norm(s) in old_sents:
            continue
        bi = _bigrams(s)
        if bi and _containment(bi, mat_bi) < NARRATIVE_SENTENCE_OVERLAP_MIN:
            problems.append(f"材料と照合できない文: {s[:40]}")
    return (not problems), problems


def _record_narrative_attempt(store: MindStore, fingerprint: str,
                              now: float) -> None:
    """物語改訂の棄却を、材料のフィンガープリントとともに記録する"""
    prev = store.get_state(_NARRATIVE_ATTEMPT_KEY, {}) or {}
    if not isinstance(prev, dict):
        prev = {}
    count = int(prev.get("rejected") or 0) \
        if prev.get("fingerprint") == fingerprint else 0
    store.set_state(_NARRATIVE_ATTEMPT_KEY, {
        "fingerprint": fingerprint, "rejected": count + 1,
        "last_at": now,
    }, "inferred")


async def _step4_revise_narrative(agent, now: float) -> dict:
    store: MindStore = agent.mind_store
    mem = agent.memory_store
    current = store.latest_narrative()
    if current is None:
        return {"revised": False, "note": "自己物語がまだ無い"}
    new_mems = [
        e for e in mem.all()
        if e.created_at > current.created_at
        and len((e.text or "").strip()) >= 10
    ]
    if not new_mems:
        return {"revised": False, "note": "物語より後の新しい出来事が無い"}
    full = (now - current.created_at) >= NARRATIVE_FULL_INTERVAL_SEC
    # 同一材料での再試行クールダウン: 棄却済みの材料セットでは、新しい
    # 材料が増えるまで生成し直さない（毎晩の再試行→棄却の繰り返し防止）
    fingerprint = hashlib.sha1(
        (("full" if full else "minor") + ":"
         + ",".join(str(i) for i in sorted(e.id for e in new_mems))
         ).encode("utf-8")).hexdigest()
    attempt = store.get_state(_NARRATIVE_ATTEMPT_KEY, {}) or {}
    if not isinstance(attempt, dict):
        attempt = {}
    if attempt.get("fingerprint") == fingerprint \
            and int(attempt.get("rejected") or 0) >= 1:
        return {"revised": False, "cooldown": True,
                "note": "同一材料での物語改訂は前回棄却済みのため見送り",
                "materials": len(new_mems)}
    materials_text = "\n".join(f"- {e.text}" for e in new_mems[:8])
    old_len = len(current.text)
    # 文字数予算: ゲート上限の9割をプロンプトに明示し、長さ超過の
    # 棄却を生成側で防ぐ（ゲート自体は緩めない）
    if full:
        gate_hi = min(950, int(old_len * 1.25))
        budget_hi = min(855, max(360, int(gate_hi * NARRATIVE_BUDGET_RATIO)))
        instruction = (
            "自己物語を書き直してください。一人称で、「何者か／これまで"
            "何があったか／何が好きで何を目指すか」の3部構成で、"
            f"全体を{budget_hi}字以内に収めてください"
            f"（目安は{old_len}字前後。長すぎる文章は採用されません）。"
            "材料に無い出来事・数字は書かないでください。")
    else:
        gate_hi = min(int(old_len * 1.3), 1100)
        budget_hi = int(gate_hi * NARRATIVE_BUDGET_RATIO)
        instruction = (
            "既存の文章を土台に、新しい出来事を1〜2文だけ織り込んで"
            "書き直してください。文章の大部分は今の物語のままにし、"
            f"全体の長さは今の物語（{old_len}字）とほぼ同じ程度で、"
            f"必ず{budget_hi}字以内に収めてください"
            "（長すぎる文章は採用されません）。一人称・3部構成を保ち、"
            "材料に無い出来事・数字は書かないでください。")
    system = (f"あなたは「{agent.name}」本人です。これは視聴者への発話では"
              "なく、自分の自己物語を見直す時間です。物語の本文だけを"
              "出力してください（前置き・説明・コードブロックは不要）。")
    user = (f"## 指示\n{instruction}\n\n"
            f"## 今の自己物語\n{current.text}\n\n"
            f"## 新しい出来事（材料）\n{materials_text}")
    try:
        raw = await asyncio.wait_for(
            MemoryConsolidator._generate_raw(
                agent.llm, system, [ChatMessage(role="user", content=user)]),
            timeout=_GEN_TIMEOUT_SEC)
    except Exception as e:
        logger.warning("SleepCycle: narrative generation failed",
                       character=agent.name, error=repr(e))
        return {"revised": False, "error": repr(e)[:120]}
    text = re.sub(r"<think>.*?</think>", "", raw or "", flags=re.DOTALL)
    text = re.sub(r"```(?:json|markdown)?", "", text).strip()
    if text.startswith("{"):
        _record_narrative_attempt(store, fingerprint, now)
        return {"revised": False, "note": "生成がJSON形式で返ったため棄却",
                "materials": len(new_mems)}
    ok, problems = _verify_narrative(text, current.text, materials_text,
                                     full=full)
    if not ok:
        logger.warning("SleepCycle: narrative revision rejected",
                       character=agent.name, problems=problems)
        _record_narrative_attempt(store, fingerprint, now)
        return {"revised": False, "rejected": problems,
                "materials": len(new_mems),
                "generated_chars": len(text), "budget": budget_hi}
    store.set_state(_NARRATIVE_ATTEMPT_KEY, {}, "inferred")
    saved = store.add_narrative(
        text,
        basis=(f"sleep {'full' if full else 'minor'} revision; "
               f"materials: memory ids {[e.id for e in new_mems[:8]]}"),
        approved="auto", provenance="experienced")
    agent.refresh_mind()
    return {"revised": True, "version": saved.version, "full": full,
            "materials": len(new_mems), "chars": len(text),
            "budget": budget_hi}


# ── 睡眠サイクル本体 ──────────────────────────────────────────────

async def run_sleep_cycle(agent, orch=None, reason: str = "manual",
                          now: float | None = None) -> dict:
    """キャラ1体ぶんの睡眠時統合を1回実行する。

    全ステップ完走したときだけ watermark を前進させる。中断・失敗時は
    journal にその旨を残し、watermark は進めない（各ステップ冪等）。
    """
    store: MindStore = agent.mind_store
    now = now or time.time()
    state = get_sleep_state(store)
    watermark = float(state.get("last_period_end") or 0)
    period_start = watermark if watermark else now - 7 * 86400
    period_end = now

    def aborted() -> bool:
        if orch is None:
            return False
        if getattr(orch, "_is_speaking", False):
            return True
        queue = getattr(orch, "_comment_queue", None)
        return queue is not None and not queue.empty()

    steps: dict = {}
    completed = True
    try:
        steps["consolidation"] = _step1_consolidate(
            agent, period_start, period_end)
    except Exception as e:
        steps["consolidation"] = {"error": repr(e)[:160]}
        completed = False
    if aborted():
        completed = False
    else:
        try:
            steps["opinion_review"] = await _step2_review_opinions(
                agent, period_start, period_end)
        except Exception as e:
            steps["opinion_review"] = {"error": repr(e)[:160]}
            completed = False
    if completed and not aborted():
        try:
            steps["intention_review"] = _step3_review_intentions(agent, now)
        except Exception as e:
            steps["intention_review"] = {"error": repr(e)[:160]}
            completed = False
    if completed and not aborted():
        try:
            steps["narrative"] = await _step4_revise_narrative(agent, now)
        except Exception as e:
            steps["narrative"] = {"error": repr(e)[:160]}
            completed = False
    if aborted():
        completed = False

    archive_result = {"archived": 0, "summaries": 0}
    try:
        archive_result = store.archive_journal(
            older_than_days=JOURNAL_ARCHIVE_DAYS, now=now)
    except Exception as e:
        logger.warning("SleepCycle: journal archive failed",
                       character=agent.name, error=repr(e))

    s1 = steps.get("consolidation") or {}
    s2 = steps.get("opinion_review") or {}
    s3 = steps.get("intention_review") or {}
    s4 = steps.get("narrative") or {}
    summary = (
        f"睡眠時統合（{reason}）: "
        f"固定化 昇格{len(s1.get('promoted', []))}件/"
        f"減衰{len(s1.get('demoted', []))}件、"
        f"意見見直し{len(s2.get('reviewed', []))}件、"
        f"意図調整{len(s3.get('adjusted', []))}件、"
        f"物語{'改訂あり v' + str(s4.get('version')) if s4.get('revised') else '変更なし'}、"
        f"ジャーナル圧縮{archive_result.get('archived', 0)}件")
    store.add_journal(
        "sleep", output_text=summary,
        input_refs=f"period:{int(period_start)}-{int(period_end)}",
        applied_changes={"reason": reason, "completed": completed,
                         "steps": steps, "archive": archive_result},
        provenance="experienced")
    if completed:
        store.set_state(_SLEEP_STATE_KEY, {
            "last_completed_at": now,
            "last_period_end": period_end,
            "last_sleep_date": _jst_date(now),
            "runs": int(state.get("runs") or 0) + 1,
        }, "experienced")
        agent.refresh_mind()
    logger.info("SleepCycle: done", character=agent.name, reason=reason,
                completed=completed, summary=summary)
    return {"character": agent.name, "reason": reason,
            "completed": completed, "steps": steps,
            "archive": archive_result,
            "period": [period_start, period_end], "summary": summary}


# ── スケジューラ ──────────────────────────────────────────────────

class SleepManager:
    """アプリ常駐の睡眠スケジューラ。Orchestrator から duck typing で読む。"""

    def __init__(self, orchestrator):
        self._orch = orchestrator
        # 調整値は config（mind セクション）を正とし、環境変数があれば
        # 移行互換のためそちらを優先する（F8）。既定値は従来どおり
        mind_cfg = getattr(getattr(orchestrator, "_cfg", None), "mind", None)
        cfg_enabled = True if mind_cfg is None else bool(mind_cfg.sleep_enabled)
        cfg_poll = 60.0 if mind_cfg is None else float(mind_cfg.sleep_poll_sec)
        self.enabled = os.environ.get(
            "AIVT_SLEEP_ENABLED", "1" if cfg_enabled else "0") != "0"
        self.poll_sec = _env_float("AIVT_SLEEP_POLL_SEC", cfg_poll)
        self._run_lock = asyncio.Lock()
        self.last_run_at: float = 0.0
        self.last_results: dict[str, dict] = {}
        logger.info("SleepManager configured", enabled=self.enabled,
                    poll_sec=self.poll_sec)

    def status(self) -> dict:
        return {
            "enabled": self.enabled,
            "poll_sec": self.poll_sec,
            "last_run_at": self.last_run_at,
            "last_results": {
                k: {"reason": v.get("reason"),
                    "completed": v.get("completed")}
                for k, v in self.last_results.items()
            },
        }

    async def run(self) -> None:
        """アプリの lifespan から常駐させる背景ループ"""
        await asyncio.sleep(min(15.0, self.poll_sec))
        while True:
            try:
                if self.enabled:
                    await self._tick()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error("SleepManager: tick failed", error=str(e))
            await asyncio.sleep(self.poll_sec)

    async def _tick(self) -> None:
        if not self._is_idle():
            return
        for agent in list(getattr(self._orch, "_agents", [])):
            if agent.llm is None or agent.mind_store is None \
                    or agent.memory_store is None:
                continue
            due, reason = evaluate_sleep_due(
                agent.mind_store, agent.memory_store, time.time())
            if due:
                logger.info("SleepManager: sleep due",
                            character=agent.name, reason=reason)
                await self.run_for(agent, reason=reason)

    def _is_idle(self) -> bool:
        orch = self._orch
        if getattr(orch, "_is_speaking", False):
            return False
        queue = getattr(orch, "_comment_queue", None)
        if queue is not None and not queue.empty():
            return False
        if getattr(orch, "_running", False):
            # セッション中は最後の発話から十分たってから統合する
            if time.time() - float(
                    getattr(orch, "_last_speech_time", 0.0)) < 300.0:
                return False
        return True

    async def run_for(self, agent, reason: str) -> dict:
        """MindLoop と同じロックで直列化してから睡眠サイクルを実行する"""
        mind_loop = getattr(self._orch, "_mind_loop", None)
        lock = getattr(mind_loop, "_lock", None) if mind_loop else None
        if lock is not None:
            async with lock:
                return await self._run_locked(agent, reason)
        return await self._run_locked(agent, reason)

    async def _run_locked(self, agent, reason: str) -> dict:
        async with self._run_lock:
            result = await run_sleep_cycle(
                agent, orch=self._orch, reason=reason)
            self.last_run_at = time.time()
            self.last_results[agent.key] = result
            return result

    async def run_manual(self, char: str = "") -> dict:
        """POST /api/mind/sleep から: 指定キャラ（省略時は全員）を1回実行"""
        orch = self._orch
        targets = []
        if char:
            found = orch.get_agent(char)
            if found is not None:
                targets = [found]
        if not targets:
            targets = list(orch.agents)
        results: dict[str, dict] = {}
        for agent in targets:
            if agent.llm is None or agent.mind_store is None \
                    or agent.memory_store is None:
                results[agent.key] = {"error": "not ready"}
                continue
            results[agent.key] = await self.run_for(agent, reason="manual")
        return {"characters": results}
