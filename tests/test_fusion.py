"""Phase 3: weights, pyramids and the fusion in YUV, RGB and Lab."""
import numpy as np
import pytest

from cdef.color_spaces import lab_unit_to_rgb, rgb_to_lab_unit, rgb_to_yuv, yuv_to_rgb
from cdef.fusion import (
    brightness_weights,
    collapse,
    combined_weight,
    contrast_weight,
    exposure_weight,
    laplacian_pyramid,
    multiscale_fusion,
    normalize_weights,
    pyramid_levels,
    saliency_weight,
    saturation_weight,
)


# ------------------------------------------------------- colour spaces
def test_yuv_round_trip_and_grey_has_no_chroma(greenish):
    assert np.abs(yuv_to_rgb(rgb_to_yuv(greenish)) - greenish).max() < 1e-5
    yuv = rgb_to_yuv(np.full((2, 2, 3), 0.5, np.float32))
    assert np.abs(yuv[..., 1:]).max() < 1e-4
    assert yuv[..., 0] == pytest.approx(0.5)


def test_lab_unit_round_trip(greenish):
    assert np.abs(lab_unit_to_rgb(rgb_to_lab_unit(greenish)) - greenish).max() < 1e-4


# ------------------------------------------------------------- weights
def test_contrast_weight_is_zero_on_flat_and_positive_at_a_peak():
    y = np.zeros((9, 9), np.float32)
    assert contrast_weight(y).max() == 0
    y[4, 4] = 1.0
    assert contrast_weight(y)[4, 4] == pytest.approx(4.0)


def test_saturation_weight_eq8():
    assert saturation_weight(np.float32(0.1), np.float32(-0.2)) == pytest.approx(1.3)


def test_saliency_weight_is_zero_on_flat_image():
    assert saliency_weight(np.full((4, 4, 3), 0.3, np.float32)).max() == pytest.approx(0)


def test_exposure_weight_peaks_at_phi():
    y = np.linspace(0, 1, 101, dtype=np.float32)
    assert y[np.argmax(exposure_weight(y, 0.5, 0.2))] == pytest.approx(0.5)


def test_normalised_weights_sum_to_one(greenish, bluish):
    inputs = [rgb_to_yuv(img).astype(np.float64) for img in (greenish, bluish, greenish * 0.5)]
    weights = normalize_weights([combined_weight(x) for x in inputs])
    assert np.abs(sum(weights) - 1.0).max() < 1e-6


def test_brightness_weights_sum_to_one_and_favour_brighter_image():
    w = [np.full((4, 4), 0.5), np.full((4, 4), 0.5)]
    out = brightness_weights(w, [np.full((4, 4, 3), 0.2), np.full((4, 4, 3), 0.8)], "yuv")
    assert np.allclose(out[0] + out[1], 1.0)
    assert (out[1] > out[0]).all()


# ------------------------------------------------------------ pyramids
def test_pyramid_levels_eq12():
    assert pyramid_levels(400, 300) == 8
    assert pyramid_levels(256, 512) == 8
    assert pyramid_levels(1, 1) == 1


@pytest.mark.parametrize("shape", [(64, 64), (65, 97), (101, 90), (37, 256)])
def test_pyramid_collapse_reconstructs_input(shape):
    img = np.random.default_rng(0).random(shape + (3,)).astype(np.float32)
    assert np.abs(collapse(laplacian_pyramid(img, pyramid_levels(*shape))) - img).max() < 1e-4


# -------------------------------------------------------------- fusion
@pytest.mark.parametrize("space", ["yuv", "rgb", "lab"])
def test_fusing_identical_images_returns_that_image(greenish, space):
    out, weights = multiscale_fusion([greenish] * 3, space=space)
    assert np.abs(out - greenish).max() < 1e-3
    assert np.abs(sum(weights) - 1.0).max() < 1e-5


def test_fusion_prefers_input_with_structure(greenish):
    _, weights = multiscale_fusion([greenish, np.full_like(greenish, 0.5)])
    assert weights[0].mean() > 0.8


def test_fusion_rejects_unknown_colour_space(greenish):
    with pytest.raises(ValueError):
        multiscale_fusion([greenish] * 2, space="hsv")
