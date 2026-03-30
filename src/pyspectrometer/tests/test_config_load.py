"""Tests for load_config path resolution (explicit file missing vs implicit search)."""

from pathlib import Path

from pyspectrometer import config as cfgmod
from pyspectrometer.config import (
    Config,
    explicit_config_path_from_argv,
    load_config,
    parse_window_geometry,
    resolve_explicit_config_path,
)


def test_load_config_bang_missing_returns_resolved_save_path(monkeypatch, tmp_path: Path) -> None:
    base = tmp_path / "spectral"
    base.mkdir()
    monkeypatch.setattr(cfgmod, "app_config_dir", lambda: base)
    cfg, p = load_config(Path("!x.toml"))
    assert isinstance(cfg, Config)
    assert p == base / "x.toml"


def test_load_config_explicit_missing_returns_path_for_save(tmp_path: Path) -> None:
    missing = tmp_path / "garden.toml"
    cfg, path_for_save = load_config(missing)
    assert isinstance(cfg, Config)
    assert path_for_save == missing
    assert not missing.exists()


def test_resolve_explicit_config_path_bang_under_app_dir(monkeypatch, tmp_path) -> None:
    base = tmp_path / "spectral"
    base.mkdir()
    monkeypatch.setattr(cfgmod, "app_config_dir", lambda: base)
    assert resolve_explicit_config_path("!garden.toml") == base / "garden.toml"
    assert resolve_explicit_config_path("!sub/cal.toml") == base / "sub" / "cal.toml"


def test_resolve_explicit_config_path_bang_empty_is_default_main(monkeypatch, tmp_path) -> None:
    base = tmp_path / "spectral"
    base.mkdir()
    monkeypatch.setattr(cfgmod, "app_config_dir", lambda: base)
    assert resolve_explicit_config_path("!") == (base / "config.toml").resolve()


def test_explicit_config_path_from_argv_long_and_short() -> None:
    assert explicit_config_path_from_argv(
        ["fit_csv", "x.csv", "--save", "--config", "garden.toml"]
    ) == Path("garden.toml")
    assert explicit_config_path_from_argv(["calibrate", "-c", "a.toml", "1"]) == Path("a.toml")


def test_explicit_config_path_from_argv_none_when_absent() -> None:
    assert explicit_config_path_from_argv(["measure", "--gain", "5"]) is None


def test_parse_window_geometry_variants() -> None:
    assert parse_window_geometry("1280x720") == (1280, 720)
    assert parse_window_geometry(" 640 X 400 ") == (640, 400)
    assert parse_window_geometry("1024*768") == (1024, 768)
    assert parse_window_geometry("800×600") == (800, 600)
    assert parse_window_geometry("1920,1080") == (1920, 1080)


def test_parse_window_geometry_invalid() -> None:
    import pytest

    for bad in ("", "1280", "abc", "1280xx720", "-1x100"):
        with pytest.raises(ValueError):
            parse_window_geometry(bad)


def test_apply_window_geometry_stack_sums_to_height() -> None:
    c = Config()
    c.apply_window_geometry(1280, 720)
    d = c.display
    assert d.window_width == 1280
    assert d.graph_height + d.preview_height + d.message_height == d.stack_height
    assert d.stack_height <= 720


def test_apply_window_geometry_status_columns_ordered() -> None:
    c = Config()
    c.apply_window_geometry(1920, 1080)
    d = c.display
    assert d.status_col1_x < d.status_col2_x < d.window_width


def test_load_config_implicit_none_when_no_files(tmp_path: Path, monkeypatch) -> None:
    """No env, no cwd file, no user config dir file → second value is None."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("PYSPECTROMETER_CONFIG", raising=False)
    from pyspectrometer import config as cfgmod

    monkeypatch.setattr(cfgmod, "app_config_dir", lambda: tmp_path / "cfg_home")
    cfg, path_for_save = load_config(None)
    assert isinstance(cfg, Config)
    assert path_for_save is None
