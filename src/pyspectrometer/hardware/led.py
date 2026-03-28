"""Status LED on Raspberry Pi via gpiozero (software PWM). Imports gpiozero only when used."""

from __future__ import annotations

import sys
import time
from typing import Any

# BCM pin for spectrometer LED (see README / config.txt gpio=22).
DEFAULT_PIN = 22


def _require_linux() -> None:
    if sys.platform != "linux":
        raise RuntimeError("GPIO LED control requires Linux (Raspberry Pi).")


def _pwm_led_type() -> Any:
    """Load PWMLED on demand so Windows and non-Pi Linux can import this module."""
    _require_linux()
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
