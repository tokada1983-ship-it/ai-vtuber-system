"""LongTermMemory: キャラクターごとの長期記憶MDファイルを管理する

ファイル配置:
  backend/llm/prompts/memory_{キャラ名}.md

例: backend/llm/prompts/memory_Character1.md

このファイルの内容はシステムプロンプトに「長期記憶」セクションとして挿入される。
記憶の内容はAIVtuber管理者が手動で編集・更新する。
配信中に内容を更新した場合は reload() を呼ぶか、次のターンから自動的に反映される。
"""
from __future__ import annotations

import time
from pathlib import Path

from backend.logger import get_logger

logger = get_logger(__name__)

_RELOAD_INTERVAL_SEC = 30.0  # ファイル変更を自動検出する間隔


class LongTermMemory:
    """
    キャラクター名に対応する memory_{name}.md を読み込み、
    build_prompt() でプロンプト断片として返す。

    特徴:
    - ファイルが存在しない場合は空文字を返す（エラーにしない）
    - 30秒ごとにファイルの更新を検出して自動リロード
    - 複数キャラクターへの対応（キャラ名を変えるだけ）
    """

    def __init__(self, character_name: str, prompts_dir: str | Path = "backend/llm/prompts"):
        self._name = character_name
        self._dir = Path(prompts_dir)

        # プリセットディレクトリ内の memory.md を優先して使用
        # フォールバック: 旧形式の memory_{name}.md
        # （パスは prompts_dir 基準。以前はここが固定パスで、prompts_dir を
        #   指定してもテスト用ディレクトリに切り替わらなかった）
        preset_path = self._dir / "characters" / character_name / "memory.md"
        legacy_path = self._dir / f"memory_{character_name}.md"

        if preset_path.exists():
            self._path = preset_path
        elif legacy_path.exists():
            # 旧形式が存在すればプリセットディレクトリに移行
            preset_path.parent.mkdir(parents=True, exist_ok=True)
            import shutil
            shutil.copy2(legacy_path, preset_path)
            self._path = preset_path
            logger.info("LongTermMemory migrated to preset dir",
                        from_=str(legacy_path), to=str(preset_path))
        else:
            # どちらもなければプリセットディレクトリに新規作成
            preset_path.parent.mkdir(parents=True, exist_ok=True)
            self._path = preset_path
        self._content: str = ""
        self._last_mtime: float = 0.0
        self._last_check: float = 0.0
        self._load()

    def _load(self) -> None:
        """ファイルを読み込む。存在しなければ空にする。"""
        if not self._path.exists():
            if self._content:
                logger.info("LongTermMemory file removed", path=str(self._path))
            self._content = ""
            self._last_mtime = 0.0
            return

        mtime = self._path.stat().st_mtime
        if mtime == self._last_mtime:
            return  # 変更なし

        try:
            self._content = self._path.read_text(encoding="utf-8").strip()
            self._last_mtime = mtime
            logger.info(
                "LongTermMemory loaded",
                character=self._name,
                chars=len(self._content),
                path=str(self._path),
            )
        except Exception as e:
            logger.warning("LongTermMemory load error", error=str(e))
            self._content = ""

    def _maybe_reload(self) -> None:
        """一定時間ごとにファイルの変更をチェックする"""
        now = time.time()
        if now - self._last_check >= _RELOAD_INTERVAL_SEC:
            self._last_check = now
            self._load()

    def reload(self) -> None:
        """強制リロード（API経由での手動更新用）"""
        self._last_check = 0.0
        self._load()

    def build_prompt(self) -> str:
        """
        プロンプトに挿入する文字列を返す。
        ファイルが空 or 存在しない場合は空文字。
        """
        self._maybe_reload()
        if not self._content:
            return ""
        return f"## {self._name}の長期記憶\n\n{self._content}"

    @property
    def path(self) -> Path:
        return self._path

    @property
    def character_name(self) -> str:
        return self._name

    @property
    def is_loaded(self) -> bool:
        return bool(self._content)

    def get_raw(self) -> str:
        """現在のメモリ内容をそのまま返す（API/UI用）"""
        self._maybe_reload()
        return self._content

    def save(self, content: str) -> None:
        """
        メモリ内容を上書き保存する。

        一時ファイルに書いてから置換することで、書き込み中に落ちても
        既存の記憶が壊れないようにしている。
        """
        # 書き込み先は _path であって _dir ではない（プリセット削除後に
        # 保存すると FileNotFoundError になっていた）
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        tmp.write_text(content, encoding="utf-8")
        tmp.replace(self._path)
        self._content = content.strip()
        self._last_mtime = self._path.stat().st_mtime
        logger.info("LongTermMemory saved", character=self._name, chars=len(content))
