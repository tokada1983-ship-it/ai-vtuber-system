"""Ollama の自動起動とモデルの事前ロード

Ollama が起動していなければ `ollama serve` を裏で立ち上げ、使うモデルを
VRAM に読み込んでおく。配信開始後の1発話目で「モデルのロード待ち（数十秒）」や
「接続拒否で即フォールバック」が起きないようにするため。

- サーバ起動: 実行ファイルを探して `ollama serve` をウィンドウ無しで起動する
  （OLLAMA_HOST は config の base_url に合わせる）。自分で起動したものだけ終了時に止める
- モデルのロード: /api/generate に空プロンプト＋keep_alive を送ると、生成せずに読み込みだけ行う
- 未取得のモデルは既定では pull しない（数GBのダウンロードになるため auto_pull で明示的に許可）
"""
from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import httpx

from backend.logger import get_logger

logger = get_logger(__name__)

DEFAULT_BASE_URL = "http://localhost:11434"


def find_ollama(configured: str = "") -> str | None:
    """ollama 実行ファイルを探す（設定 → PATH → Windowsの既定インストール先）"""
    if configured and Path(configured).exists():
        return configured
    exe = shutil.which("ollama")
    if exe:
        return exe
    candidates = [
        Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe",
        Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Ollama" / "ollama.exe",
    ]
    for path in candidates:
        if path.exists():
            return str(path)
    return None


def normalize_model(name: str) -> str:
    """Ollama はタグ省略時に :latest を付けて保存するので、比較用にそろえる"""
    name = (name or "").strip()
    return name if not name or ":" in name else f"{name}:latest"


class OllamaService:
    """1つの Ollama サーバ（base_url）ぶんの起動・ロード管理"""

    def __init__(self, base_url: str = DEFAULT_BASE_URL, *, auto_start: bool = True,
                 executable: str = "", keep_alive: str = "30m", auto_pull: bool = False,
                 startup_timeout_sec: float = 30.0, stop_on_exit: bool = True):
        self.base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")
        self.auto_start = auto_start
        self.executable = executable
        self.keep_alive = keep_alive
        self.auto_pull = auto_pull
        self.startup_timeout_sec = startup_timeout_sec
        self.stop_on_exit = stop_on_exit
        self._proc: subprocess.Popen | None = None
        self._start_lock = asyncio.Lock()
        self._model_locks: dict[str, asyncio.Lock] = {}
        self.last_error = ""
        self.loaded: dict[str, float] = {}     # モデル名 → ロード完了時刻

    # ── サーバ ────────────────────────────────────────────────

    async def is_running(self, timeout: float = 2.0) -> bool:
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.get(f"{self.base_url}/api/version")
                return resp.status_code == 200
        except Exception:
            return False

    @property
    def started_by_us(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def _is_local(self) -> bool:
        host = urlparse(self.base_url).hostname or ""
        return host in ("localhost", "127.0.0.1", "::1", "0.0.0.0")

    async def ensure_running(self) -> bool:
        """起動していなければ起動して、応答するまで待つ。起動できたら True。"""
        if await self.is_running():
            return True
        if not self.auto_start:
            self.last_error = "Ollama が起動していません（自動起動は無効）"
            return False
        if not self._is_local():
            self.last_error = f"リモートの Ollama（{self.base_url}）は自動起動できません"
            return False

        async with self._start_lock:
            # 待っている間に別の呼び出しが起動済みにしていることがある
            if await self.is_running():
                return True
            exe = find_ollama(self.executable)
            if not exe:
                self.last_error = "ollama が見つかりません（https://ollama.com からインストールしてください）"
                logger.warning("Ollama executable not found")
                return False
            if not self.started_by_us:
                self._spawn(exe)
            deadline = time.monotonic() + self.startup_timeout_sec
            while time.monotonic() < deadline:
                if await self.is_running(timeout=1.0):
                    self.last_error = ""
                    logger.info("Ollama server is up", base_url=self.base_url)
                    return True
                if self._proc is not None and self._proc.poll() is not None:
                    self.last_error = f"ollama serve が終了しました（exit {self._proc.returncode}）"
                    break
                await asyncio.sleep(0.5)
            else:
                self.last_error = f"Ollama が {self.startup_timeout_sec:.0f} 秒以内に起動しませんでした"
            logger.warning("Ollama failed to start", error=self.last_error)
            return False

    def _spawn(self, exe: str) -> None:
        parsed = urlparse(self.base_url)
        env = dict(os.environ)
        env["OLLAMA_HOST"] = f"{parsed.hostname or '127.0.0.1'}:{parsed.port or 11434}"
        log_path = Path("audio_cache").parent / "ollama_serve.log"
        kwargs: dict = {"env": env, "stdin": subprocess.DEVNULL}
        try:
            kwargs["stdout"] = kwargs["stderr"] = open(log_path, "ab")
        except OSError:
            kwargs["stdout"] = kwargs["stderr"] = subprocess.DEVNULL
        if sys.platform == "win32":
            # コンソールウィンドウを出さない
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
        logger.info("Starting ollama serve", exe=exe, host=env["OLLAMA_HOST"])
        self._proc = subprocess.Popen([exe, "serve"], **kwargs)

    def stop(self) -> None:
        """自分で起動した ollama serve だけを止める（ユーザーが起動したものは触らない）"""
        if self.started_by_us and self.stop_on_exit:
            logger.info("Stopping ollama serve (started by us)")
            self._proc.terminate()
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self._proc = None

    # ── モデル ────────────────────────────────────────────────

    async def list_models(self) -> list[str]:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self.base_url}/api/tags")
                resp.raise_for_status()
                return [m.get("name", "") for m in resp.json().get("models", [])]
        except Exception:
            return []

    async def running_models(self) -> list[str]:
        """VRAMに載っているモデル"""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self.base_url}/api/ps")
                resp.raise_for_status()
                return [m.get("name", "") for m in resp.json().get("models", [])]
        except Exception:
            return []

    async def has_model(self, model: str) -> bool:
        return normalize_model(model) in {normalize_model(m) for m in await self.list_models()}

    async def pull(self, model: str) -> bool:
        logger.info("Pulling Ollama model", model=model)
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(None, connect=10.0)) as client:
                resp = await client.post(f"{self.base_url}/api/pull",
                                         json={"model": model, "stream": False})
                resp.raise_for_status()
            return True
        except Exception as e:
            self.last_error = f"{model} の取得に失敗: {e}"
            return False

    async def ensure_model(self, model: str) -> bool:
        """
        サーバを起動し、モデルを VRAM に読み込む。読み込めたら True。
        同じモデルへの同時呼び出しは1回にまとめる。
        """
        if not model:
            return False
        if not await self.ensure_running():
            return False
        key = normalize_model(model)
        lock = self._model_locks.setdefault(key, asyncio.Lock())
        async with lock:
            if key in {normalize_model(m) for m in await self.running_models()}:
                self.loaded[key] = time.time()
                return True
            if not await self.has_model(model):
                if not self.auto_pull:
                    self.last_error = (f"モデル {model} が Ollama にありません"
                                       f"（ollama pull {model} を実行するか、auto_pull を有効に）")
                    logger.warning("Ollama model not found", model=model)
                    return False
                if not await self.pull(model):
                    return False
            started = time.monotonic()
            try:
                # 空プロンプトは「読み込みだけ」の指示になる
                async with httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=10.0)) as client:
                    resp = await client.post(f"{self.base_url}/api/generate",
                                             json={"model": model, "keep_alive": self.keep_alive})
                    resp.raise_for_status()
            except Exception as e:
                self.last_error = f"{model} のロードに失敗: {e}"
                logger.warning("Ollama model preload failed", model=model, error=str(e))
                return False
            self.loaded[key] = time.time()
            logger.info("Ollama model loaded", model=model,
                        seconds=round(time.monotonic() - started, 1))
            return True

    async def status(self) -> dict:
        running = await self.is_running()
        return {
            "base_url": self.base_url,
            "installed": bool(find_ollama(self.executable)),
            "running": running,
            "started_by_us": self.started_by_us,
            "auto_start": self.auto_start,
            "models": await self.list_models() if running else [],
            "loaded": await self.running_models() if running else [],
            "last_error": self.last_error,
        }


# ── サーバごとのシングルトン ─────────────────────────────────

_settings: dict = {}
_services: dict[str, OllamaService] = {}


def configure(cfg) -> None:
    """config.yaml の ollama: セクションを反映する（既存のサービスにも適用）"""
    global _settings
    _settings = {
        "auto_start": cfg.auto_start,
        "executable": cfg.executable,
        "keep_alive": cfg.keep_alive,
        "auto_pull": cfg.auto_pull,
        "startup_timeout_sec": cfg.startup_timeout_sec,
        "stop_on_exit": cfg.stop_on_exit,
    }
    for svc in _services.values():
        for key, value in _settings.items():
            setattr(svc, key, value)


def get_service(base_url: str = "") -> OllamaService:
    key = (base_url or DEFAULT_BASE_URL).rstrip("/")
    if key not in _services:
        _services[key] = OllamaService(key, **_settings)
    return _services[key]


def stop_all() -> None:
    for svc in _services.values():
        svc.stop()