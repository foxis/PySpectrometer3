"""Pytest configuration. Mocks cv2 so tests can run without opencv-python."""

import sys
from unittest.mock import MagicMock

# Must run before pyspectrometer imports spectrometer (which imports cv2)
sys.modules["cv2"] = MagicMock()
