"""Live2Dアバター対応のテスト

モデル探索・メタ情報の読み取り・設定の往復（config.yaml）を検証する。
ブラウザ側の描画は対象外。
"""
import json

import yaml

from backend.avatar import live2d
from backend.config import AppConfig, load_config, save_config

# Cubism 4 のモデル定義（必要な部分だけ）
MODEL3 = {
    "Version": 3,
    "FileReferences": {
        "Moc": "hiyori.moc3",
        "Textures": ["hiyori.4096/texture_00.png"],
        "Motions": {
            "Idle": [{"File": "motions/idle_01.motion3.json"},
                     {"File": "motions/idle_02.motion3.json"}],
            "TapBody": [{"File": "motions/tap_01.motion3.json"}],
        },
        "Expressions": [
            {"Name": "exp_smile", "File": "expressions/smile.exp3.json"},
            {"Name": "exp_angry", "File": "expressions/angry.exp3.json"},
        ],
    },
}

# 旧 Cubism 2 のモデル定義
MODEL2 = {
    "model": "kaori.moc",
    "textures": ["kaori.2048/texture_00.png"],
    "motions": {"idle": [{"file": "motions/idle.mtn"}]},
    "expressions": [{"name": "f01", "file": "expressions/f01.exp.json"}],
}


def _make_model(root, name="hiyori", data=MODEL3, suffix=".model3.json"):
    folder = root / "live2d" / name
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}{suffix}"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_scan_models_finds_cubism4_and_2(tmp_path):
    _make_model(tmp_path)
    _make_model(tmp_path, "kaori", MODEL2, ".model.json")
    # モデル定義ではないJSONは拾わない
    (tmp_path / "live2d" / "hiyori" / "hiyori.physics3.json").write_text("{}", encoding="utf-8")

    found = live2d.scan_models(tmp_path)
    by_name = {m["name"]: m for m in found}

    assert set(by_name) == {"hiyori", "kaori"}
    assert by_name["hiyori"]["version"] == 4
    assert by_name["kaori"]["version"] == 2
    assert by_name["hiyori"]["url"].startswith("/")
    assert by_name["hiyori"]["url"].endswith("/hiyori.model3.json")


def test_scan_models_on_missing_dir():
    assert live2d.scan_models("does/not/exist") == []


def test_to_url_normalizes_windows_paths():
    assert live2d.to_url(r"models\live2d\a\a.model3.json") == "/models/live2d/a/a.model3.json"
    assert live2d.to_url("/models/a.model3.json") == "/models/a.model3.json"
    assert live2d.to_url("https://example.com/a.model3.json") == "https://example.com/a.model3.json"
    assert live2d.to_url("") == ""


def test_describe_model_lists_motions_and_expressions(tmp_path):
    path = _make_model(tmp_path)
    info = live2d.describe_model(path)
    assert info["motions"] == {"Idle": 2, "TapBody": 1}
    assert info["expressions"] == ["exp_smile", "exp_angry"]
    assert info["version"] == 4


def test_describe_model_cubism2(tmp_path):
    path = _make_model(tmp_path, "kaori", MODEL2, ".model.json")
    info = live2d.describe_model(path)
    assert info["motions"] == {"idle": 1}
    assert info["expressions"] == ["f01"]
    assert info["version"] == 2


def test_describe_model_broken_file_is_safe(tmp_path):
    broken = tmp_path / "broken.model3.json"
    broken.write_text("{ これはJSONではない", encoding="utf-8")
    info = live2d.describe_model(broken)
    assert info["motions"] == {} and info["expressions"] == []
    # 存在しないファイルでも例外にしない
    assert live2d.describe_model(tmp_path / "none.model3.json")["motions"] == {}


def test_resolve_model_path_falls_back_to_scan(tmp_path, monkeypatch):
    path = _make_model(tmp_path)
    monkeypatch.chdir(tmp_path.parent)
    rel = path.relative_to(tmp_path.parent).as_posix()

    # 設定済みで実在するならそのまま
    assert live2d.resolve_model_path(rel, tmp_path) == rel
    # 空・存在しないパスなら自動検出に落とす
    assert live2d.resolve_model_path("", tmp_path).endswith("hiyori.model3.json")
    assert live2d.resolve_model_path("models/none.model3.json", tmp_path).endswith(
        "hiyori.model3.json")
    # 1つも無ければ空文字（アバター画面が案内を出す）
    assert live2d.resolve_model_path("", tmp_path / "empty") == ""


def test_is_inside_blocks_traversal(tmp_path):
    (tmp_path / "models").mkdir()
    inside = tmp_path / "models" / "a.model3.json"
    inside.write_text("{}", encoding="utf-8")
    assert live2d.is_inside(inside, tmp_path / "models")
    assert not live2d.is_inside(tmp_path / "secret.json", tmp_path / "models")
    assert not live2d.is_inside(tmp_path / "models" / ".." / "secret.json", tmp_path / "models")


def test_local_cores_detects_placed_files(tmp_path):
    core_dir = tmp_path / "live2d" / "core"
    core_dir.mkdir(parents=True)
    # 置いていなければ空（アバター画面はCDNへフォールバックする）
    assert live2d.local_cores(tmp_path) == {"cubism4": "", "cubism2": ""}

    (core_dir / "live2dcubismcore.min.js").write_text("// core", encoding="utf-8")
    found = live2d.local_cores(tmp_path)
    assert found["cubism4"].endswith("/live2d/core/live2dcubismcore.min.js")
    assert found["cubism4"].startswith("/")
    assert found["cubism2"] == ""

    (core_dir / "live2d.min.js").write_text("// core2", encoding="utf-8")
    assert live2d.local_cores(tmp_path)["cubism2"].endswith("/live2d/core/live2d.min.js")


def test_config_roundtrip_keeps_live2d_settings(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = AppConfig()
    cfg.avatar.type = "live2d"
    cfg.avatar.live2d.model_path = "models/live2d/hiyori/hiyori.model3.json"
    cfg.avatar.live2d.scale = 1.4
    cfg.avatar.live2d.position_y = 1.1
    cfg.avatar.live2d.expression_map = {"happy": "exp_smile"}
    cfg.avatar.live2d.motion_map = {"wave": "TapBody:1"}
    save_config(cfg, "config.yaml")

    loaded = load_config("config.yaml")
    assert loaded.avatar.type == "live2d"
    assert loaded.avatar.live2d.model_path == "models/live2d/hiyori/hiyori.model3.json"
    assert loaded.avatar.live2d.scale == 1.4
    assert loaded.avatar.live2d.position_y == 1.1
    assert loaded.avatar.live2d.expression_map == {"happy": "exp_smile"}
    assert loaded.avatar.live2d.motion_map == {"wave": "TapBody:1"}


def test_config_without_live2d_section_uses_defaults(tmp_path, monkeypatch):
    """Live2D追加前のconfig.yamlをそのまま読んでも落ちない（後方互換）"""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yaml").write_text(
        yaml.safe_dump({"avatar": {"type": "vrm", "vrm": {"model_path": "models/a.vrm"}}}),
        encoding="utf-8",
    )
    cfg = load_config("config.yaml")
    assert cfg.avatar.type == "vrm"
    assert cfg.avatar.live2d.model_path == ""
    assert cfg.avatar.live2d.scale == 0.9
