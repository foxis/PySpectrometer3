"""Tests for lazy-import GPIO LED helpers (no gpiozero required on Windows)."""

import sys
from pathlib import Path

import pytest

from pyspectrometer.hardware.led import (
    DEFAULT_PIN,
    _normalize_duty,
    apply_led_config_from_values,
    effective_frequency_hz,
    effective_pin,
    gpio_led_supported,
    reset_gpio_led_support_cache,
    reset_led_runtime,
    turn_off,
    turn_on,
)


@pytest.fixture(autouse=True)
def _reset_led_support_cache() -> None:
    reset_gpio_led_support_cache()
    reset_led_runtime()
    yield
    reset_gpio_led_support_cache()
    reset_led_runtime()


def test_default_pin() -> None:
    assert DEFAULT_PIN == 22


def test_apply_led_config_from_values_updates_effective() -> None:
    apply_led_config_from_values(17, 1500)
    assert effective_pin() == 17
    assert effective_frequency_hz() == 1500


def test_apply_led_config_clamps_pin_and_frequency() -> None:
    apply_led_config_from_values(99, 50_000)
    assert effective_pin() == 40
    assert effective_frequency_hz() == 20_000


def test_load_config_hardware_led_pin(tmp_path: Path) -> None:
    p = tmp_path / "cfg.toml"
    p.write_text(
        "[hardware]\nled_pin = 18\nled_pwm_frequency_hz = 2000\n",
        encoding="utf-8",
    )
    from pyspectrometer.config import load_config

    cfg, loaded = load_config(p)
    assert loaded == p
    assert cfg.hardware.led_pin == 18
    assert cfg.hardware.led_pwm_frequency_hz == 2000


def test_load_config_legacy_led_bcm_pin_alias(tmp_path: Path) -> None:
    p = tmp_path / "old.toml"
    p.write_text("[hardware]\nled_bcm_pin = 7\n", encoding="utf-8")
    from pyspectrometer.config import load_config

    cfg, _ = load_config(p)
    assert cfg.hardware.led_pin == 7


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (0.0, 0.0),
        (0.5, 0.5),
        (1.0, 1.0),
        (50.0, 0.5),
        (100.0, 1.0),
    ],
)
def test_normalize_duty(raw: float, expected: float) -> None:
    assert _normalize_duty(raw) == expected


def test_normalize_duty_rejects_negative() -> None:
    with pytest.raises(ValueError, match=">="):
        _normalize_duty(-0.1)


def test_normalize_duty_rejects_over_100_percent() -> None:
    with pytest.raises(ValueError, match="between"):
        _normalize_duty(101.0)


def test_gpio_led_supported_false_on_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    assert gpio_led_supported() is False


def test_gpio_led_supported_pi_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(
        "pyspectrometer.hardware.led._read_device_tree_model",
        lambda: "Raspberry Pi Zero 2 W Rev 1.0",
    )
    assert gpio_led_supported() is True


def test_gpio_led_supported_pi4_not_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(
        "pyspectrometer.hardware.led._read_device_tree_model",
        lambda: "Raspberry Pi 4 Model B Rev 1.1",
    )
    assert gpio_led_supported() is False


def test_turn_on_non_linux_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    with pytest.raises(RuntimeError, match="Raspberry Pi Zero"):
        turn_on()


def test_turn_off_non_linux_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    with pytest.raises(RuntimeError, match="Raspberry Pi Zero"):
        turn_off()


def test_turn_on_pi4_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(
        "pyspectrometer.hardware.led._read_device_tree_model",
        lambda: "Raspberry Pi 4 Model B",
    )
    with pytest.raises(RuntimeError, match="Raspberry Pi Zero"):
        turn_on()
