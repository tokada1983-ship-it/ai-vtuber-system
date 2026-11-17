"""動画アバター対応のテスト

カタログJSONの解決（実在ファイルだけをURL化する）と、
config.yaml の往復で video 設定が保持されることを検証する。
ブラウザ側の再生は対象外。
"""
import json

from backend.avatar import video
from backend.config import AppConfig, load_config, save_config


def _make_catalog(root, char="Character1"):
    folder = root / "motions" / char
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "idle_main.webm").write_bytes(b"webm")
    (folder / "idle_02.webm").write_bytes(b"webm")
    (folder / "custom_nod.webm").write_bytes(b"webm")
    (folder / "dance_01.webm").write_bytes(b"webm")
    catalog = {
        "character": char,
        "idle": ["idle_main.webm", "idle_02.webm", "missing.webm"],
        "motion_map": {
            "nod": "custom_nod.webm",
            "dance": ["dance_01.webm", "missing.webm"],
            "wave": "not_there.webm",
        },
        "emotion_map": {"happy": ["dance_01.webm"]},
    }
    path = folder / "catalog.json"
    path.write_text(json.dumps(catalog), encoding="utf-8")
    return path


def test_load_catalog_filters_missing_files(tmp_path):
    path = _make_catalog(tmp_path)
    result = video.load_catalog(path, tmp_path)

    assert result["ok"] is True
    assert result["character"] == "Character1"
    # 実在するファイルだけが /models 基準のURLになる
    assert result["idle"] == [
        "/models/motions/Character1/idle_main.webm",
        "/models/motions/Character1/idle_02.webm",
    ]
    assert result["motion_map"]["nod"] == ["/models/motions/Character1/custom_nod.webm"]
    assert result["motion_map"]["dance"] == ["/models/motions/Character1/dance_01.webm"]
    # 全滅したモーションはキーごと落ちる
    assert "wave" not in result["motion_map"]
    assert result["emotion_map"]["happy"] == ["/models/motions/Character1/dance_01.webm"]


def test_load_catalog_missing_or_empty(tmp_path):
    assert video.load_catalog("", tmp_path)["ok"] is False
    assert video.load_catalog(tmp_path / "nope.json", tmp_path)["ok"] is False


def test_payload_reports_catalog_state(tmp_path):
    path = _make_catalog(tmp_path)

    class Cfg:
        catalog_path = str(path)

    payload = video.payload(Cfg(), tmp_path)
    assert payload["catalog_ok"] is True
    assert payload["idle"]
    assert payload["motion_map"]


def test_config_roundtrip_keeps_video_avatar(tmp_path):
    cfg = AppConfig()
    cfg.avatar.type = "video"
    cfg.avatar.video.catalog_path = "models/motions/Character1/catalog.json"
    target = tmp_path / "config.yaml"
    save_config(cfg, target)

    loaded = load_config(target)
    assert loaded.avatar.type == "video"
    assert loaded.avatar.video.catalog_path == "models/motions/Character1/catalog.json"
    # キャラ専用アバターでも video が使える
    loaded.characters[0].avatar = loaded.avatar
    assert loaded.characters[0].avatar.video.catalog_path.endswith("catalog.json")
