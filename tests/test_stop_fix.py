"""/api/stop の応答性と停止段階タイムアウトのテスト（stop-fix）

2026-11-19 に /api/stop が120秒応答しなかった事象への対策を検証する:
- /api/stop は停止処理が長引いても上限時間で応答を返し、停止処理は
  裏で継続する。停止の意図（mark_stream_stopping）は処理の冒頭で
  記録され、状態ファイルの削除は常に行われる。
- Orchestrator._cancel_tasks はキャンセル不能なタスクがあっても
  上限時間で待ちを打ち切り、未完了分を次回 start 用に保持する。
"""
from __future__ import annotations

import asyncio
import time

from backend.api import http_routes
from backend.orchestrator import Orchestrator


class _FakeOrchestrator:
    """stop() の所要時間を制御できるフェイク"""

    def __init__(self, stop_delay: float = 0.0):
        self.stop_delay = stop_delay
        self.stop_calls = 0
        self.stop_finished = False

    async def stop(self):
        self.stop_calls += 1
        if self.stop_delay:
            await asyncio.sleep(self.stop_delay)
        self.stop_finished = True


def _patch_route(monkeypatch, fake: _FakeOrchestrator) -> list[str]:
    """http_routes の依存をフェイクへ差し替え、呼び出し順を返す"""
    calls: list[str] = []
    monkeypatch.setattr(http_routes, "orchestrator", fake)
    monkeypatch.setattr(http_routes, "mark_stream_stopping",
                        lambda: calls.append("mark_stopping"))
    monkeypatch.setattr(http_routes, "clear_stream_state",
                        lambda: calls.append("clear_state"))
    return calls


async def test_api_stop_responds_immediately_when_stop_is_fast(
        monkeypatch):
    fake = _FakeOrchestrator(stop_delay=0.0)
    calls = _patch_route(monkeypatch, fake)
    resp = await http_routes.api_stop()
    assert resp == {"ok": True, "stop_in_background": False}
    assert fake.stop_finished is True
    # 意図の記録が最初、状態ファイル削除が最後
    assert calls == ["mark_stopping", "clear_state"]


async def test_api_stop_responds_within_bound_when_stop_hangs(
        monkeypatch):
    fake = _FakeOrchestrator(stop_delay=3.0)
    calls = _patch_route(monkeypatch, fake)
    monkeypatch.setattr(http_routes, "API_STOP_TIMEOUT_SEC", 0.5)
    start = time.monotonic()
    resp = await http_routes.api_stop()
    elapsed = time.monotonic() - start
    assert elapsed < 2.0  # 上限（0.5秒）＋余裕で応答が返る
    assert resp == {"ok": True, "stop_in_background": True}
    assert calls == ["mark_stopping", "clear_state"]
    # 停止処理は裏で継続し、最後まで完了する
    await asyncio.sleep(3.2)
    assert fake.stop_finished is True


def _bare_orchestrator() -> Orchestrator:
    """__init__ を経ずに _cancel_tasks だけ使える個体を作る"""
    orch = Orchestrator.__new__(Orchestrator)
    orch._tasks = []
    return orch


async def test_cancel_tasks_clears_finished_tasks():
    orch = _bare_orchestrator()

    async def _quick():
        await asyncio.sleep(0.05)

    orch._tasks = [asyncio.create_task(_quick(), name="quick")]
    await orch._cancel_tasks(timeout=2.0)
    assert orch._tasks == []


async def test_cancel_tasks_times_out_and_keeps_remaining():
    orch = _bare_orchestrator()

    async def _blocking():
        # キャンセルされても終了しないタスクの再現（例外処理の中で
        # 重い後処理を続けるタスク等。実事象ではこの種の待ちが
        # gather を無制限にブロックした）
        try:
            await asyncio.sleep(30.0)
        except asyncio.CancelledError:
            pass
        await asyncio.sleep(3.0)

    task = asyncio.create_task(_blocking(), name="blocking")
    orch._tasks = [task]
    await asyncio.sleep(0.1)  # タスクを sleep(30) の中まで進めておく
    start = time.monotonic()
    await orch._cancel_tasks(timeout=0.3)
    elapsed = time.monotonic() - start
    assert elapsed < 2.0  # 無界待ちにならない
    assert orch._tasks == [task]  # 未完了分は次回 start 用に残る
    # 後始末: スレッドの終了まで待って回収する
    await asyncio.gather(task, return_exceptions=True)
    assert task.done()
