"""リソースモニターのテスト（nvidia-smi・外部サービスには依存しない）"""
import asyncio

from backend.config import AppConfig
from backend import monitor as mon


class Hub:
    def __init__(self): self.sent = []
    def has_clients(self, ch): return True
    async def broadcast(self, ch, data): self.sent.append(data)


def _snap(**over):
    s = {"cpu": 10.0, "ram_percent": 50.0, "gpus": [
        {"vram_percent": 40.0, "vram_used_gb": 6.4, "vram_total_gb": 16.0, "temp": 50.0}]}
    s.update(over)
    return s


def test_warnings_thresholds():
    assert mon.ResourceMonitor._warnings(_snap()) == []
    w = mon.ResourceMonitor._warnings(_snap(ram_percent=93.0, cpu=97.0, gpus=[
        {"vram_percent": 95.0, "vram_used_gb": 15.2, "vram_total_gb": 16.0, "temp": 86.0}]))
    assert len(w) == 4 and any("VRAM" in x for x in w) and any("温度" in x for x in w)


def test_sample_keeps_history(monkeypatch):
    async def fake_gpus():
        return [{"index": 0, "name": "Fake GPU", "util": 30.0, "vram_used_gb": 8.0,
                 "vram_total_gb": 16.0, "vram_percent": 50.0, "temp": 60.0, "power_w": 100.0}]

    async def no_users(self):
        return [{"service": "Ollama", "model": "m", "vram_gb": 3.3}]

    monkeypatch.setattr(mon, "query_gpus", fake_gpus)
    monkeypatch.setattr(mon.ResourceMonitor, "vram_users", no_users)
    m = mon.ResourceMonitor(AppConfig(), Hub())
    s1 = asyncio.run(m.sample())
    asyncio.run(m.sample())
    assert s1["gpus"][0]["name"] == "Fake GPU" and s1["vram_users"][0]["vram_gb"] == 3.3
    snap = m.snapshot()
    assert len(snap["history"]) == 2 and snap["history"][-1]["vram"] == 50.0
    assert 0 <= snap["cpu"] <= 100 and snap["ram_total_gb"] > 0


def test_gpu_query_without_nvidia_smi(monkeypatch):
    monkeypatch.setattr(mon.shutil, "which", lambda name: None)
    assert asyncio.run(mon.query_gpus()) == []

def test_classify_known_services():
    c = mon.classify
    k, label, svc, kind = c(10, "llama-server.exe",
                            [r"C:\Users\x\.lmstudio\extensions\llama-server.exe", "--model", r"D:\LMStudio\a\Qwen3-8B-Q4_K_M.gguf"],
                            [], 0, 1)
    assert (label, svc, kind) == ("Qwen3-8B-Q4_K_M", "LM Studio", "ai")
    _, label, svc, _ = c(11, "llama-server.exe", [r"C:\x\Ollama\lib\ollama\llama-server.exe", "--model", "sha256-abc"],
                         ["persona-x:latest"], 1, 1)
    assert (label, svc) == ("persona-x:latest", "Ollama")
    assert c(12, "python.exe", ["python.exe", "-m", "irodori_openai_tts"], [], 0, 1)[2] == "音声合成"
    assert c(13, "python.exe", ["python", "-m", "backend.persona.trainer", "--job", "j"], [], 0, 1)[2] == "学習"
    assert c(14, "csrss.exe", [], [], 0, 1)[3] == "system"
    assert c(15, "chrome.exe", ["chrome.exe"], [], 0, 1)[1:] == ("chrome", "アプリ", "app")


def test_breakdown_groups_and_ignores_dwm(monkeypatch):
    m = mon.ResourceMonitor(AppConfig(), Hub())
    GB = mon.GB
    monkeypatch.setattr(m._gpu_mem, "read", lambda: {1: int(50 * GB), 2: int(3 * GB), 3: int(0.05 * GB)})

    class P:
        def __init__(self, pid): self.pid = pid
        def name(self): return {1: "dwm.exe", 2: "python.exe", 3: "steam.exe"}[self.pid]
        def cmdline(self):
            if self.pid == 1:
                raise mon.psutil.AccessDenied(1)
            return ["python", "-m", "irodori_openai_tts"] if self.pid == 2 else ["steam.exe"]

    monkeypatch.setattr(mon.psutil, "Process", P)
    items = m.vram_breakdown({"vram_used_gb": 4.0})
    labels = [i["label"] for i in items]
    assert labels[0].startswith("Irodori") and items[0]["gb"] == 3.0
    assert "その他のアプリ" in labels                      # 0.1GB 未満はまとめる
    assert not any("dwm" in l for l in labels)            # dwm の見かけの値は数えない
    assert items[-1]["key"] == "unknown" and items[-1]["gb"] == 0.95
