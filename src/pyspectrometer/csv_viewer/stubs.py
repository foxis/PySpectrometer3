"""Null stubs for camera-less CSV viewer operation."""

import numpy as np

from ..capture.base import CameraInterface


class NullCamera(CameraInterface):
    """Satisfies CameraInterface without hardware — always returns an empty frame."""

    def __init__(self, frame_width: int = 1280, frame_height: int = 480) -> None:
        self._width = frame_width
        self._height = frame_height
        self._gain: float = 0.0
        self.exposure: int = 10000
        self._running = False

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def gain(self) -> float:
        return self._gain

    @gain.setter
    def gain(self, value: float) -> None:
        self._gain = float(value)

    @property
    def is_running(self) -> bool:
        return self._running

    def start(self) -> None:
        self._running = True

    def stop(self) -> None:
        self._running = False

    def capture(self) -> np.ndarray:
        return np.zeros((self._height, self._width), dtype=np.uint16)
