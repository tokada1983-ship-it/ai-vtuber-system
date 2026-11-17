"""Irodori-TTS サーバ（Irodori-TTS-Server）の自動起動

config.yaml:
  tts:
    primary: irodori
    irodori_server:
      auto_start: true
      dir: Irodori-TTS/Irodori-TTS-Server
      startup_timeout_sec: 300

起動コマンドはサーバのフォルダにある .venv の Python を優先し、無ければ
「uv run --no-sync」を使う（uv sync --extra cu128 で入れたGPU版torchを入れ替えないため）。
モデル・声・精度などはサーバ側の .env で設定する。
"""
from __future__ import annotations

import shutil
from pathlib import Path
from urllib.parse import urlparse

from backend.config import AppConfig
from backend.services.local_server import LocalServerProcess

_service: LocalServerProcess | None = None


def _origin(base_url: str) -> tuple[str, str, int]:
    u = urlparse(base_url or "http://localhost:8088/v1")
    host = u.hostname or "localhost"
    return f"{u.scheme or 'http'}://{host}:{u.port or 8088}", host, u.port or 8088


def build_command(server_dir: Path, port: int, custom: str = "") -> list[str]:
    if custom:
        return custom.split()
    host_args = ["--host", "127.0.0.1", "--port", str(port)]
    venv_py = server_dir / ".venv" / "Scripts" / "python.exe"
    if not venv_py.exists():
        venv_py = server_dir / ".venv" / "bin" / "python"
    if venv_py.exists():
        return [str(venv_py), "-m", "irodori_openai_tts", *host_args]
    uv = shutil.which("uv") or "uv"
    return [uv, "run", "--no-sync", "python", "-m", "irodori_openai_tts", *host_args]


def get_service(config: AppConfig) -> LocalServerProcess | None:
    """Irodori を使う設定で、自動起動が有効なときだけサービスを返す"""
    global _service
    cfg = config.tts.irodori_server
    be = config.tts.backends.get("irodori")
    if config.tts.primary != "irodori" or not cfg.auto_start or be is None:
        return None
    origin, host, port = _origin(be.base_url)
    if host not in ("localhost", "127.0.0.1", "::1"):
        return None   # 別のPCのサーバは起動できない
    server_dir = Path(cfg.dir)
    command = build_command(server_dir, port, cfg.command)
    if _service is None or _service.command != command or _service.env != cfg.env:
        if _service is not None:
            _service.stop()
        _service = LocalServerProcess(
            name="Irodori-TTS",
            health_url=f"{origin}/health",
            command=command,
            cwd=server_dir,
            env=dict(cfg.env),
            startup_timeout_sec=cfg.startup_timeout_sec,
            stop_on_exit=cfg.stop_on_exit,
            log_path=Path("logs") / "irodori_server.log",
        )
    return _service


def stop() -> None:
    if _service is not None:
        _service.stop()