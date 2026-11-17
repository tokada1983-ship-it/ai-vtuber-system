"""BGMPlayer: プレイリストからBGMを再生する

対応フォーマット: MP3, OGG, WAV, FLAC（pygameの対応範囲）
プレイリストフォルダ: config.yaml の bgm.playlist_dir で指定

注意: pygameのmixerはTTSの音声出力とは別チャンネルで動作するため干渉しない。
ただしWindowsでは同一デバイスに出力される。
仮想オーディオデバイスを使う場合は bgm.device を設定すること。
"""
from __future__ import annotations

import asyncio
import os
import random
from pathlib import Path
from typing import Callable

from backend.logger import get_logger

logger = get_logger(__name__)

SUPPORTED_EXTS = {".mp3", ".ogg", ".wav", ".flac", ".m4a"}


class BGMPlayer:
    """
    pygameを使ってBGMを再生するクラス。
    asyncioと共存するためファイルロード・再生制御は別スレッドで行う。
    """

    def __init__(self):
        self._playlist: list[Path] = []
        self._current_index: int = 0
        self._is_playing: bool = False
        self._volume: float = 0.5
        self._shuffle: bool = False
        self._repeat: bool = True
        self._initialized: bool = False
        self._monitor_task: asyncio.Task | None = None
        self._on_track_change: Callable | None = None  # 曲変更コールバック

    def _init_pygame(self) -> bool:
        if self._initialized:
            return True
        try:
            import pygame
            pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=2048)
            self._initialized = True
            logger.info("BGMPlayer: pygame mixer initialized")
            return True
        except Exception as e:
            logger.warning("BGMPlayer: pygame init failed", error=str(e))
            return False

    # ── プレイリスト管理 ──────────────────────────────────────

    def load_playlist(self, source: str | Path) -> int:
        """
        フォルダまたはM3Uファイルからプレイリストを読み込む。
        Returns: 読み込んだ曲数
        """
        source = Path(source)
        tracks: list[Path] = []

        if source.is_dir():
            for f in sorted(source.rglob("*")):
                if f.suffix.lower() in SUPPORTED_EXTS:
                    tracks.append(f)
        elif source.suffix.lower() == ".m3u":
            with open(source, encoding="utf-8", errors="ignore") as fp:
                for line in fp:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        p = Path(line)
                        if not p.is_absolute():
                            p = source.parent / p
                        if p.exists():
                            tracks.append(p)
        elif source.suffix.lower() in SUPPORTED_EXTS and source.exists():
            tracks.append(source)

        self._playlist = tracks
        self._current_index = 0
        if self._shuffle:
            random.shuffle(self._playlist)
        logger.info("BGMPlayer: playlist loaded", count=len(tracks), source=str(source))
        return len(tracks)

    def add_track(self, path: str | Path) -> None:
        p = Path(path)
        if p.exists() and p.suffix.lower() in SUPPORTED_EXTS:
            self._playlist.append(p)

    def clear_playlist(self) -> None:
        self.stop()
        self._playlist.clear()
        self._current_index = 0

    # ── 再生制御 ──────────────────────────────────────────────

    async def play(self) -> bool:
        """現在のトラックを再生する"""
        if not self._playlist:
            logger.warning("BGMPlayer: playlist is empty")
            return False
        if not self._init_pygame():
            return False

        track = self._playlist[self._current_index]
        try:
            import pygame
            await asyncio.to_thread(self._load_and_play, track)
            self._is_playing = True
            logger.info("BGMPlayer: playing", track=track.name, index=self._current_index)
            if self._on_track_change:
                await self._on_track_change(self.current_track_info())
            # 曲終了監視タスクを起動
            if self._monitor_task is None or self._monitor_task.done():
                self._monitor_task = asyncio.create_task(self._monitor_loop())
            return True
        except Exception as e:
            logger.warning("BGMPlayer: play error", error=str(e), track=str(track))
            return False

    def _load_and_play(self, track: Path) -> None:
        import pygame
        pygame.mixer.music.load(str(track))
        pygame.mixer.music.set_volume(self._volume)
        pygame.mixer.music.play()

    async def stop(self) -> None:
        if self._monitor_task:
            self._monitor_task.cancel()
            self._monitor_task = None
        if self._initialized:
            try:
                import pygame
                await asyncio.to_thread(pygame.mixer.music.stop)
            except Exception:
                pass
        self._is_playing = False
        logger.info("BGMPlayer: stopped")

    async def pause(self) -> None:
        if self._initialized and self._is_playing:
            import pygame
            await asyncio.to_thread(pygame.mixer.music.pause)
            self._is_playing = False

    async def resume(self) -> None:
        if self._initialized and not self._is_playing and self._playlist:
            import pygame
            await asyncio.to_thread(pygame.mixer.music.unpause)
            self._is_playing = True

    async def next_track(self) -> bool:
        if not self._playlist:
            return False
        self._current_index = (self._current_index + 1) % len(self._playlist)
        return await self.play()

    async def prev_track(self) -> bool:
        if not self._playlist:
            return False
        self._current_index = (self._current_index - 1) % len(self._playlist)
        return await self.play()

    async def play_at(self, index: int) -> bool:
        if 0 <= index < len(self._playlist):
            self._current_index = index
            return await self.play()
        return False

    def set_volume(self, volume: float) -> None:
        self._volume = max(0.0, min(1.0, volume))
        if self._initialized:
            try:
                import pygame
                pygame.mixer.music.set_volume(self._volume)
            except Exception:
                pass

    def set_shuffle(self, shuffle: bool) -> None:
        self._shuffle = shuffle
        if shuffle:
            random.shuffle(self._playlist)
            self._current_index = 0

    # ── 曲終了監視 ────────────────────────────────────────────

    async def _monitor_loop(self) -> None:
        """曲が終わったら次の曲に進む"""
        import pygame
        while self._is_playing and self._playlist:
            await asyncio.sleep(1.0)
            try:
                busy = await asyncio.to_thread(pygame.mixer.music.get_busy)
                if not busy and self._is_playing:
                    # 曲が終了した
                    if self._repeat or self._current_index < len(self._playlist) - 1:
                        self._current_index = (self._current_index + 1) % len(self._playlist)
                        await asyncio.sleep(0.5)
                        await self.play()
                    else:
                        self._is_playing = False
                        logger.info("BGMPlayer: playlist ended")
            except asyncio.CancelledError:
                break
            except Exception:
                break

    # ── 情報取得 ──────────────────────────────────────────────

    def current_track_info(self) -> dict:
        if not self._playlist:
            return {"name": "", "path": "", "index": 0, "total": 0}
        track = self._playlist[self._current_index]
        return {
            "name": track.stem,          # 拡張子なしのファイル名
            "filename": track.name,
            "path": str(track),
            "index": self._current_index,
            "total": len(self._playlist),
            "is_playing": self._is_playing,
            "volume": self._volume,
            "shuffle": self._shuffle,
            "repeat": self._repeat,
        }

    def get_playlist(self) -> list[dict]:
        return [
            {"index": i, "name": p.stem, "filename": p.name}
            for i, p in enumerate(self._playlist)
        ]

    def build_bgm_prompt(self) -> str:
        """現在再生中の曲情報をプロンプト用テキストに変換する"""
        if not self._is_playing or not self._playlist:
            return ""
        info = self.current_track_info()
        return (
            f"## 現在再生中のBGM\n"
            f"曲名: 「{info['name']}」\n"
            f"（{info['index'] + 1} / {info['total']} 曲目）\n\n"
            f"BGMについてコメントするときはこの曲名に言及してください。"
        )

    @property
    def is_playing(self) -> bool:
        return self._is_playing

    @property
    def playlist_count(self) -> int:
        return len(self._playlist)
