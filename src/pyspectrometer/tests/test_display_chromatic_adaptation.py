"""sRGB preview uses Bradford CAT to D65 when illuminant XYZ is known."""

import pytest

try:
    import colour  # noqa: F401

    from ..colorscience.swatches import xyz_to_display_bgr
except ImportError:
    pytest.skip("colour-science not installed", allow_module_level=True)


def test_transmittance_white_matches_illuminant_neutral_srgb():
    """When sample XYZ equals illuminant white XYZ, display is ~neutral."""
    xw, yw, zw = 107.54, 99.79, 32.60
    bgr = xyz_to_display_bgr(xw, yw, zw, (xw, yw, zw))
    r, g, b = bgr[2], bgr[1], bgr[0]
    assert abs(r - g) < 8 and abs(g - b) < 8
    assert min(r, g, b) > 200


def test_cyan_shift_vs_no_cat():
    """Under same illuminant, cup XYZ should look more cyan after CAT than raw sRGB."""
    xw, yw, zw = 107.54, 99.79, 32.60
    xs, ys, zs = 46.04, 51.73, 26.38
    raw = xyz_to_display_bgr(xs, ys, zs, None)
    cat = xyz_to_display_bgr(xs, ys, zs, (xw, yw, zw))
    # Raw path is yellow-biased (B low); adapted should raise B relative to R
    assert cat[0] > raw[0]  # B channel in BGR
