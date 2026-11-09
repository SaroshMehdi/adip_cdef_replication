import numpy as np
import pytest

from cdef.fusion import (
    collapse,
    combined_weight,
    contrast_weight,
    exposure_weight,
    laplacian_pyramid,
    multiscale_fusion,
    normalize_weights,
    pyramid_levels,
    rgb_to_yuv,
    saliency_weight,
    saturation_weight,
    yuv_to_rgb,
)
from cdef.pipeline import enhance


def test_yuv_round_trip(greenish):
    assert np.abs(yuv_to_rgb(rgb_to_yuv(greenish)) - greenish).max() < 1e-5


def test_gray_has_no_chroma():
    yuv = rgb_to_yuv(np.full((2, 2, 3), 0.5, np.float32))
    assert np.abs(yuv[..., 1:]).max() < 1e-4
    assert yuv[..., 0] == pytest.approx(0.5)


def test_contrast_weight_is_zero_on_flat_and_positive_at_edges():
    y = np.zeros((9, 9), np.float32)
    assert contrast_weight(y).max() == 0
    y[4, 4] = 1.0
    assert contrast_weight(y)[4, 4] == pytest.approx(4.0)


def test_printed_kernel_differs_from_standard():
    y = np.tile(np.arange(9, dtype=np.float32)[:, None], (1, 9))  # a vertical ramp
    assert contrast_weight(y)[4, 4] == pytest.approx(0.0)
    assert contrast_weight(y, "printed")[4, 4] == pytest.approx(6.0)


def test_saturation_weight_eq8():
    assert saturation_weight(np.float32(0.1), np.float32(-0.2)) == pytest.approx(1.3)


def test_saliency_weight_is_zero_on_flat_image():
    assert saliency_weight(np.full((4, 4, 3), 0.3, np.float32)).max() == pytest.approx(0)


def test_exposure_weight_peaks_at_phi():
    y = np.linspace(0, 1, 101, dtype=np.float32)
    assert y[np.argmax(exposure_weight(y, 0.5, 0.2))] == pytest.approx(0.5)
    assert np.all(np.diff(exposure_weight(y, 2.0, 0.5)) > 0)


def test_weights_sum_to_one(greenish, bluish):
    weights = normalize_weights(
        [combined_weight(rgb_to_yuv(img), weight_range=255.0) for img in (greenish, bluish, greenish * 0.5)]
    )
    assert np.abs(sum(weights) - 1.0).max() < 1e-5


def test_printed_normalisation_does_not_sum_to_one():
    weights = normalize_weights([np.zeros((2, 2)), np.zeros((2, 2))], form="printed")
    assert sum(weights).max() == pytest.approx(2.0)


def test_pyramid_levels_eq12():
    assert pyramid_levels(400, 300) == 8
    assert pyramid_levels(256, 512) == 8
    assert pyramid_levels(1, 1) == 1


@pytest.mark.parametrize("shape", [(64, 64), (65, 97), (101, 90), (37, 256)])
def test_pyramid_collapse_reconstructs_input(shape):
    img = np.random.default_rng(0).random(shape + (3,)).astype(np.float32)
    levels = pyramid_levels(*shape)
    assert np.abs(collapse(laplacian_pyramid(img, levels)) - img).max() < 1e-4


def test_fusing_identical_images_returns_that_image(greenish):
    out, _ = multiscale_fusion([greenish] * 4)
    assert np.abs(out - greenish).max() < 1e-3


def test_fusion_prefers_input_with_structure(greenish):
    flat = np.full_like(greenish, 0.5)
    _, weights = multiscale_fusion([greenish, flat], phi=0.5, sigma=0.2, weight_range=255.0)
    assert weights[0].mean() > 0.8


def test_enhance_end_to_end(greenish, bluish):
    for img, tone, base in ((greenish, "greenish", "F1"), (bluish, "bluish", "F2")):
        out, debug = enhance(img, {"ide": {"patch": 15, "gf_radius": 7}})
        assert out.shape == img.shape
        assert np.isfinite(out).all()
        assert out.min() >= 0.0 and out.max() <= 1.0
        assert debug["tone"] == tone
        assert len(debug["weights"]) == 4
        assert np.abs(sum(debug["weights"]) - 1.0).max() < 1e-5
        assert out.std() > img.std()
