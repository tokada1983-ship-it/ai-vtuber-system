"""CharacterPresetManager: キャラクタープリセットの保存・読み込み・切り替えを管理する

ディレクトリ構造:
  backend/llm/prompts/characters/{キャラ名}/
    character.yaml  - 名前・説明・性格・ルール
    memory.md       - 長期記憶（Markdown）
    avatar.yaml     - VRMパス・キャリブレーション
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

from backend.logger import get_logger

logger = get_logger(__name__)

PRESETS_DIR = Path("backend/llm/prompts/characters")


@dataclass
class AvatarPreset:
    model_path: str = "models/default.vrm"
    idle_motion: str = "breathe"
    arm_down_z: float = 0.1
    fore_arm_z: float = 0.05
    spine_x: float = 0.0


@dataclass
class CharacterPreset:
    name: str = ""
    description: str = ""
    personality: str = ""
    rules: str = ""
    memory: str = ""
    avatar: AvatarPreset = field(default_factory=AvatarPreset)

    @property
    def dir_path(self) -> Path:
        return PRESETS_DIR / self.name

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "personality": self.personality,
            "rules": self.rules,
            "memory_chars": len(self.memory),
            "avatar": {
                "model_path": self.avatar.model_path,
                "idle_motion": self.avatar.idle_motion,
                "arm_down_z": self.avatar.arm_down_z,
                "fore_arm_z": self.avatar.fore_arm_z,
                "spine_x": self.avatar.spine_x,
            },
        }


class CharacterPresetManager:

    def __init__(self):
        PRESETS_DIR.mkdir(parents=True, exist_ok=True)

    def list_presets(self) -> list[str]:
        """保存済みプリセット名の一覧を返す"""
        return sorted([
            d.name for d in PRESETS_DIR.iterdir()
            if d.is_dir() and (d / "character.yaml").exists()
        ])

    def save(self, preset: CharacterPreset) -> None:
        """プリセットをディレクトリに保存する"""
        if not preset.name:
            raise ValueError("Character name is required")

        preset_dir = PRESETS_DIR / preset.name
        preset_dir.mkdir(parents=True, exist_ok=True)

        # character.yaml
        char_data = {
            "name": preset.name,
            "description": preset.description,
            "personality": preset.personality,
            "rules": preset.rules,
        }
        with open(preset_dir / "character.yaml", "w", encoding="utf-8") as f:
            yaml.dump(char_data, f, allow_unicode=True,
                      default_flow_style=False, sort_keys=False)

        # memory.md
        (preset_dir / "memory.md").write_text(preset.memory, encoding="utf-8")

        # avatar.yaml
        av_data = {
            "model_path": preset.avatar.model_path,
            "idle_motion": preset.avatar.idle_motion,
            "calibration": {
                "armDownZ": preset.avatar.arm_down_z,
                "foreArmZ": preset.avatar.fore_arm_z,
                "spineX":   preset.avatar.spine_x,
            },
        }
        with open(preset_dir / "avatar.yaml", "w", encoding="utf-8") as f:
            yaml.dump(av_data, f, allow_unicode=True,
                      default_flow_style=False, sort_keys=False)

        logger.info("CharacterPreset saved", name=preset.name)

    def load(self, name: str) -> Optional[CharacterPreset]:
        """プリセットを名前で読み込む"""
        preset_dir = PRESETS_DIR / name
        if not preset_dir.exists():
            logger.warning("CharacterPreset not found", name=name)
            return None

        preset = CharacterPreset(name=name)

        # character.yaml
        char_path = preset_dir / "character.yaml"
        if char_path.exists():
            with open(char_path, encoding="utf-8") as f:
                char_data = yaml.safe_load(f) or {}
            preset.description = char_data.get("description", "")
            preset.personality = char_data.get("personality", "")
            preset.rules       = char_data.get("rules", "")

        # memory.md
        mem_path = preset_dir / "memory.md"
        if mem_path.exists():
            preset.memory = mem_path.read_text(encoding="utf-8")

        # avatar.yaml
        av_path = preset_dir / "avatar.yaml"
        if av_path.exists():
            with open(av_path, encoding="utf-8") as f:
                av_data = yaml.safe_load(f) or {}
            calib = av_data.get("calibration", {})
            preset.avatar = AvatarPreset(
                model_path=av_data.get("model_path", "models/default.vrm"),
                idle_motion=av_data.get("idle_motion", "breathe"),
                arm_down_z=calib.get("armDownZ", 0.1),
                fore_arm_z=calib.get("foreArmZ", 0.05),
                spine_x=calib.get("spineX", 0.0),
            )

        logger.info("CharacterPreset loaded", name=name)
        return preset

    def delete(self, name: str) -> bool:
        """プリセットを削除する"""
        import shutil
        preset_dir = PRESETS_DIR / name
        if not preset_dir.exists():
            return False
        shutil.rmtree(preset_dir)
        logger.info("CharacterPreset deleted", name=name)
        return True

    def get_summary(self, name: str) -> dict:
        """プリセットのサマリーを返す（フル読み込みより軽い）"""
        preset_dir = PRESETS_DIR / name
        char_path = preset_dir / "character.yaml"
        av_path = preset_dir / "avatar.yaml"
        mem_path = preset_dir / "memory.md"

        char_data = {}
        if char_path.exists():
            with open(char_path, encoding="utf-8") as f:
                char_data = yaml.safe_load(f) or {}

        av_data = {}
        if av_path.exists():
            with open(av_path, encoding="utf-8") as f:
                av_data = yaml.safe_load(f) or {}

        mem_chars = len(mem_path.read_text(encoding="utf-8")) if mem_path.exists() else 0

        return {
            "name": name,
            "description": char_data.get("description", "")[:60],
            "model_path": av_data.get("model_path", ""),
            "memory_chars": mem_chars,
        }
