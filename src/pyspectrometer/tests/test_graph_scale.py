"""Unit tests for graph scaling utilities."""

import numpy as np

from ..utils.graph_scale import scale_intensity_to_graph


def test_scale_intensity_to_graph_array_default_margin():
    """Array input with default margin=10: scale 0-1 to [0, graph_height-10]."""
    result = scale_intensity_to_graph(np.array([0.0, 0.5, 1.0]), graph_height=320)
    expected = np.array([0.0, 155.0, 310.0], dtype=np.float32)  # 320-10=310 scale
    np.testing.assert_array_almost_equal(result, expected)


def test_scale_intensity_to_graph_array_margin_one():
    """Array input with margin=1: use full height (height-1)."""
    result = scale_intensity_to_graph(np.array([0.0, 0.5, 1.0]), graph_height=320, margin=1)
    expected = np.array([0.0, 159.5, 319.0], dtype=np.float32)  # 319 scale
    np.testing.assert_array_almost_equal(result, expected)


def test_scale_intensity_to_graph_scalar():
    """Scalar input returns 0-d array; int() works for drawing."""
    result = scale_intensity_to_graph(0.5, graph_height=320, margin=1)
    assert result.shape == ()
    assert 159 <= float(result) <= 160


def test_scale_intensity_to_graph_clips_overflow():
    """Values > 1 are clipped to max."""
    result = scale_intensity_to_graph(np.array([0.0, 1.5, 2.0]), graph_height=100)
    expected = np.array([0.0, 90.0, 90.0], dtype=np.float32)  # max = 90
    np.testing.assert_array_almost_equal(result, expected)


def test_scale_intensity_to_graph_edge_graph_height_less_than_margin():
    """When graph_height <= margin, effective height is 0; scale falls back safely."""
    result = scale_intensity_to_graph(np.array([0.5]), graph_height=5, margin=10)
    # effective_height = max(0, -5) = 0; scale = 1.0; clip to [0, 0]
    np.testing.assert_array_almost_equal(result, np.array([0.0], dtype=np.float32))
