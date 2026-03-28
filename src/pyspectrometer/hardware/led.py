"""Spectrometer lamp GPIO via gpiozero (Pi Zero only). Imports gpiozero only when used.

Pin numbers follow the Broadcom (SoC) GPIO index, like gpiozero and `gpio readall` — not the 1–40 physical
header position. CLI full on/off uses **lgpio** (Python). ``led-pwm`` uses gpiozero ``PWMLED``; on exit we
``close()`` the device then **drive the pin LOW via lgpio**, because ``PWMLED.close()`` alone can leave the
pad averaging to the last duty (e.g. ~50 %) instead of a clean off.
"""

from __future__ import annotations

import atexit
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
# Open chip for CLI on/off so the line stays claimed as output until process exit (avoids gpiozero close → input).
_lgpio_cli_chip: Any = None
# Active PWMLED from hold_pwm — atexit closes if the interpreter shuts down without running hold_pwm's finally.
_hold_pwm_active: Any = None


def _atexit_close_hold_pwm() -> None:
    global _hold_pwm_active
    obj = _hold_pwm_active
    if obj is None:
        return
    try:
        obj.close()
    except Exception:
        pass
    _hold_pwm_active = None


atexit.register(_atexit_close_hold_pwm)


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
    global _runtime_pin, _runtime_freq_hz, _hold_pwm_active
    _close_board_pwm()
    _atexit_close_hold_pwm()
    _close_lgpio_cli_chip()
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


def _close_lgpio_cli_chip() -> None:
    global _lgpio_cli_chip
    if _lgpio_cli_chip is None:
        return
    try:
        import lgpio

        lgpio.gpiochip_close(_lgpio_cli_chip)
    except Exception:
        pass
    _lgpio_cli_chip = None


def _lgpio_latch_output(pin: int, high: bool) -> bool:
    """Claim BCM *pin* as push-pull output (same library gpiozero uses on modern Pi OS)."""
    global _lgpio_cli_chip
    try:
        import lgpio
    except ImportError:
        return False
    try:
        if _lgpio_cli_chip is None:
            _lgpio_cli_chip = lgpio.gpiochip_open(0)
        h = _lgpio_cli_chip
        try:
            lgpio.gpio_free(h, pin)
        except Exception:
            pass
        level = 1 if high else 0
        if lgpio.gpio_claim_output(h, pin, level) < 0:
            return False
        wr = lgpio.gpio_write(h, pin, level)
        if wr < 0:
            return False
        return True
    except Exception:
        return False


def _stop_pwm_led_and_drive_low(led: Any, pin: int) -> None:
    """Tear down gpiozero PWM then hold the pad LOW (PWM close alone can leave ~duty average)."""
    global _hold_pwm_active
    _hold_pwm_active = None
    try:
        led.close()
    except Exception:
        pass
    if gpio_led_supported():
        try:
            _lgpio_latch_output(pin, False)
        except Exception:
            pass


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
        _close_lgpio_cli_chip()
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
    if not gpio_led_supported():
        LED = _digital_led_type()
        with LED(p) as led:
            led.on()
        return
    if _lgpio_latch_output(p, True):
        return
    LED = _digital_led_type()
    with LED(p) as led:
        led.on()
    _lgpio_latch_output(p, True)


def turn_off(pin: int | None = None) -> None:
    """Drive the LED off (digital low; stops GUI persistent PWM if it uses this pin)."""
    p = effective_pin() if pin is None else pin
    _release_board_pwm_if_pin_matches(p)
    if not gpio_led_supported():
        LED = _digital_led_type()
        with LED(p) as led:
            led.off()
        return
    if _lgpio_latch_output(p, False):
        return
    LED = _digital_led_type()
    with LED(p) as led:
        led.off()
    if _lgpio_latch_output(p, False):
        return
    msg = (
        f"Could not drive lamp GPIO {p} LOW (led-off). Tried lgpio (Python) then gpiozero LED.off(); "
        "both failed — check pin number and that the lgpio package works on this OS build."
    )
    raise RuntimeError(msg)


def set_brightness(
    duty: float,
    pin: int | None = None,
    frequency: int | None = None,
) -> None:
    """Set LED brightness once and release the pin (brief flash / sample)."""
    value = _normalize_duty(duty)
    p = effective_pin() if pin is None else pin
    f = effective_frequency_hz() if frequency is None else frequency
    _close_lgpio_cli_chip()
    PWMLED = _pwm_led_type()
    with PWMLED(p, frequency=f) as led:
        led.value = value


def hold_pwm(
    duty: float,
    pin: int | None = None,
    frequency: int | None = None,
) -> None:
    """Hold software PWM until KeyboardInterrupt; use for `poetry run led-pwm`.

    On exit we ``close()`` the PWMLED and then drive the pin LOW via lgpio so the LED does not stay at the
    last PWM duty. ``InterruptedError`` during sleep is ignored so stray signals do not drop out of the
    hold loop without cleanup.
    """
    global _hold_pwm_active
    value = _normalize_duty(duty)
    p = effective_pin() if pin is None else pin
    f = effective_frequency_hz() if frequency is None else frequency
    _release_board_pwm_if_pin_matches(p)
    _close_lgpio_cli_chip()
    PWMLED = _pwm_led_type()
    led: Any = None
    try:
        led = PWMLED(p, frequency=f)
        _hold_pwm_active = led
        led.value = value
        while True:
            try:
                time.sleep(3600)
            except InterruptedError:
                continue
    except KeyboardInterrupt:
        pass
    finally:
        if led is not None:
            _stop_pwm_led_and_drive_low(led, p)
        else:
            _hold_pwm_active = None
