"""Spectrometer lamp GPIO via gpiozero (Pi Zero only). Imports gpiozero only when used."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

# BCM pin for spectrometer LED (see README / config.txt gpio=22).
DEFAULT_PIN = 22

_supported_cache: bool | None = None
_board_pwm_led: Any = None


def _read_device_tree_model() -> str:
    try:
        raw = Path("/proc/device-tree/model").read_bytes()
    except OSError:
        return ""
    return raw.decode("utf-8", errors="replace").replace("\x00", "").strip()


def _is_pi_zero_model(model: str) -> bool:
    m = model.lower()
    return "raspberry pi" in m and "zero" in m


def gpio_led_supported() -> bool:
    """True on Raspberry Pi Zero / Zero W / Zero 2 W (Linux + device tree). Always False on Windows/macOS."""
    global _supported_cache
    if _supported_cache is not None:
        return _supported_cache
    if sys.platform != "linux":
        _supported_cache = False
        return False
    _supported_cache = _is_pi_zero_model(_read_device_tree_model())
    return _supported_cache


def reset_gpio_led_support_cache() -> None:
    """Clear cached board detection (for tests)."""
    global _supported_cache
    _supported_cache = None


def _require_gpio_led() -> None:
    if not gpio_led_supported():
        raise RuntimeError(
            "Spectrometer GPIO LED is only supported on Raspberry Pi Zero "
            "(Zero / Zero W / Zero 2 W) with gpiozero installed."
        )


def _pwm_led_type() -> Any:
    """Load PWMLED on demand so Windows and non-Pi hosts can import this module."""
    _require_gpio_led()
    try:
        from gpiozero import PWMLED
    except ImportError as exc:
        msg = "Install gpiozero on the Pi: poetry install (linux extras include gpiozero)."
        raise RuntimeError(msg) from exc
    return PWMLED


def _normalize_duty(raw: float) -> float:
    """0–1 fraction, or percent > 1 (e.g. 50 -> 0.5). Values in (1, 100] are percent."""
    if raw < 0:
        msg = "duty must be >= 0"
        raise ValueError(msg)
    if raw <= 1.0:
        return float(raw)
    if raw <= 100.0:
        return raw / 100.0
    msg = "duty must be between 0 and 1 (fraction) or 0 and 100 (percent)"
    raise ValueError(msg)


def _close_board_pwm() -> None:
    global _board_pwm_led
    if _board_pwm_led is not None:
        try:
            _board_pwm_led.close()
        except Exception:
            pass
        _board_pwm_led = None


def apply_board_lamp(on: bool, brightness_pct: float) -> None:
    """Drive the spectrometer lamp from the GUI slider (persistent software PWM). No-op if not Pi Zero."""
    global _board_pwm_led
    if not gpio_led_supported():
        return
    try:
        if not on:
            _close_board_pwm()
            return
        duty = max(0.0, min(100.0, brightness_pct)) / 100.0
        PWMLED = _pwm_led_type()
        if _board_pwm_led is None:
            _board_pwm_led = PWMLED(DEFAULT_PIN, frequency=100)
        _board_pwm_led.value = duty
    except Exception as exc:
        _close_board_pwm()
        print(f"[LED/HW] {exc}")


def turn_on(pin: int = DEFAULT_PIN) -> None:
    """Drive the LED fully on (software PWM at 100%)."""
    PWMLED = _pwm_led_type()
    with PWMLED(pin) as led:
        led.on()


def turn_off(pin: int = DEFAULT_PIN) -> None:
    """Drive the LED off."""
    PWMLED = _pwm_led_type()
    with PWMLED(pin) as led:
        led.off()


def set_brightness(
    duty: float,
    pin: int = DEFAULT_PIN,
    frequency: int = 100,
) -> None:
    """Set LED brightness once and release the pin (brief flash / sample)."""
    value = _normalize_duty(duty)
    PWMLED = _pwm_led_type()
    with PWMLED(pin, frequency=frequency) as led:
        led.value = value


def hold_pwm(
    duty: float,
    pin: int = DEFAULT_PIN,
    frequency: int = 100,
) -> None:
    """Hold software PWM until KeyboardInterrupt; use for `poetry run led-pwm`."""
    value = _normalize_duty(duty)
    PWMLED = _pwm_led_type()
    led = PWMLED(pin, frequency=frequency)
    try:
        led.value = value
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass
    finally:
        led.close()
