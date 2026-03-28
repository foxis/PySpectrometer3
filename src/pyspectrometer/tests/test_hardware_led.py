"""Tests for lazy-import GPIO LED helpers (no gpiozero required on Windows)."""

import sys

import pytest

from pyspectrometer.hardware.led import DEFAULT_PIN, _normalize_duty, turn_off, turn_on


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


def test_turn_on_non_linux_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    with pytest.raises(RuntimeError, match="Linux"):
        turn_on()


def test_turn_off_non_linux_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    with pytest.raises(RuntimeError, match="Linux"):
        turn_off()
