"""Spectrometer lamp GPIO via gpiozero (Pi Zero only). Imports gpiozero only when used.

Pin numbers follow the Broadcom (SoC) GPIO index, like gpiozero and `gpio readall` — not the 1–40 physical
header position. PWMLED uses software PWM on typical GPIOs (e.g. 22); GPIO 18 can be wired to hardware PWM
on the SoC when the board is configured for it, but brightness control here is still through gpiozero's PWMLED.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

# Fallback when no config has been applied (matches HardwareConfig defaults).
DEFAULT_PIN = 22
DEFAULT_PWM_FREQUENCY_HZ = 3000

_supported_cache: bool | None = None
_board_pwm_led: Any = None
_runtime_pin: int = DEFAULT_PIN
_runtime_freq_hz: int = DEFAULT_PWM_FREQUENCY_HZ


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


def effective_pin() -> int:
    """Broadcom GPIO number for the lamp after config / CLI apply."""
    return _runtime_pin


def effective_frequency_hz() -> int:
    """PWM frequency (Hz) after config / CLI apply (often software-timed on the chosen GPIO)."""
    return _runtime_freq_hz


def apply_led_config_from_values(pin: int, frequency_hz: int) -> None:
    """Set runtime pin and PWM frequency; drop persistent board PWM if they change."""
    global _runtime_pin, _runtime_freq_hz
    p = int(max(0, min(40, int(pin))))
    f = int(max(1, min(20_000, int(frequency_hz))))
    if p != _runtime_pin or f != _runtime_freq_hz:
        _close_board_pwm()
    _runtime_pin, _runtime_freq_hz = p, f


def reset_led_runtime() -> None:
    """Restore code defaults and release board PWM (for tests)."""
    global _runtime_pin, _runtime_freq_hz
    _close_board_pwm()
    _runtime_pin = DEFAULT_PIN
    _runtime_freq_hz = DEFAULT_PWM_FREQUENCY_HZ


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


def _digital_led_type() -> Any:
    """Digital on/off output for CLI full on/off (clean LOW; avoids PWM close leaving the line active)."""
    _require_gpio_led()
    try:
        from gpiozero import LED
    except ImportError as exc:
        msg = "Install gpiozero on the Pi: poetry install (linux extras include gpiozero)."
        raise RuntimeError(msg) from exc
    return LED


def _release_board_pwm_if_pin_matches(p: int) -> None:
    """Stop GUI persistent PWM on this pin so CLI can take over the same GPIO."""
    if p == effective_pin():
        _close_board_pwm()


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
        pin = effective_pin()
        freq = effective_frequency_hz()
        if _board_pwm_led is None:
            _board_pwm_led = PWMLED(pin, frequency=freq)
        _board_pwm_led.value = duty
    except Exception as exc:
        _close_board_pwm()
        print(f"[LED/HW] {exc}")


def turn_on(pin: int | None = None) -> None:
    """Drive the LED fully on (digital high; releases any GUI PWM on the same pin first)."""
    p = effective_pin() if pin is None else pin
    _release_board_pwm_if_pin_matches(p)
    LED = _digital_led_type()
    with LED(p) as led:
        led.on()


def turn_off(pin: int | None = None) -> None:
    """Drive the LED off (digital low; stops GUI persistent PWM if it uses this pin)."""
    p = effective_pin() if pin is None else pin
    _release_board_pwm_if_pin_matches(p)
    LED = _digital_led_type()
    with LED(p) as led:
        led.off()


def set_brightness(
    duty: float,
    pin: int | None = None,
    frequency: int | None = None,
) -> None:
    """Set LED brightness once and release the pin (brief flash / sample)."""
    value = _normalize_duty(duty)
    p = effective_pin() if pin is None else pin
    f = effective_frequency_hz() if frequency is None else frequency
    PWMLED = _pwm_led_type()
    with PWMLED(p, frequency=f) as led:
        led.value = value


def hold_pwm(
    duty: float,
    pin: int | None = None,
    frequency: int | None = None,
) -> None:
    """Hold software PWM until KeyboardInterrupt; use for `poetry run led-pwm`."""
    value = _normalize_duty(duty)
    p = effective_pin() if pin is None else pin
    f = effective_frequency_hz() if frequency is None else frequency
    _release_board_pwm_if_pin_matches(p)
    PWMLED = _pwm_led_type()
    led = PWMLED(p, frequency=f)
    try:
        led.value = value
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass
    finally:
        led.close()
