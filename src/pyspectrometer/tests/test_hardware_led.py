"""Tests for lazy-import GPIO LED helpers (no gpiozero required on Windows)."""

import sys

import pytest

from pyspectrometer.hardware.led import (
    DEFAULT_PIN,
    _normalize_duty,
    gpio_led_supported,
    reset_gpio_led_support_cache,
    turn_off,
    turn_on,
)


@pytest.fixture(autouse=True)
def _reset_led_support_cache() -> None:
    reset_gpio_led_support_cache()
    yield
    reset_gpio_led_support_cache()


def test_default_pin() -> None:
    assert DEFAULT_PIN == 22


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
