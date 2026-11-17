"""アプリ全体で共有するシングルトン（コンポジションルート）

HTTPルート・WebSocketルート・起動処理はすべてここから同じ実体を import する。
以前は main.py 1ファイル（1400行）に全部が同居していた。
"""
from __future__ import annotations

from pathlib import Path

from backend.avatar.ws_hub import WebSocketHub
from backend.config import load_config, load_dotenv
from backend.logger import get_logger, setup_logging
from backend.memory.character_preset import CharacterPresetManager
from backend.monitor import ResourceMonitor
from backend.orchestrator import Orchestrator
from backend.persona.manager import PersonaManager

setup_logging("INFO")
logger = get_logger(__name__)

# APIキーは環境変数が最優先。無ければ .env から読む（Web検索キーなどをここに置ける）
_dotenv_count = load_dotenv(".env")
if _dotenv_count:
    logger.info(".env loaded", keys=_dotenv_count)

FRONTEND = Path("frontend")
MODELS_DIR = Path("models")
AUDIO_CACHE = Path("audio_cache")

config = load_config("config.yaml")
hub = WebSocketHub()
orchestrator = Orchestrator(config, hub)

# キャラクタープリセットマネージャー
_preset_manager = CharacterPresetManager()

# 人格QLoRA（データセット作成・学習ジョブ・専用モデルの切替）
persona = PersonaManager(config, hub, orchestrator)

# CPU・メモリ・VRAM の監視（ダッシュボードに表示）
monitor = ResourceMonitor(config, hub)

# アバターキャリブレーション設定（メモリ保持・再起動時はデフォルトに戻る）
_avatar_calibration: dict = {
    "armDownZ": 1.15,   # 腕を下ろす量（ラジアン）。0.1 だとほぼTポーズになる
    "foreArmZ": 0.05,
    "spineX": 0.0,
}