"""Tests for the options that follow the released code (method="reference",
Wang et al.'s fusion) and for the colour conversions."""
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse
from scipy.sparse.linalg import spsolve
from skimage.color import rgb2lab as skimage_rgb2lab

from cdef.color_comp import color_compensation, lab2rgb, rgb2lab
from cdef.detail.ide import channel_airlight, ide, mean_inside, rational_recovery
from cdef.fusion import brightness_weights, multiscale_fusion, rgb_to_ycbcr, ycbcr_to_rgb
from cdef.luminance import (
    luminance_compensation,
    modified_incomplete_cholesky,
    periodic_system,
    preconditioned_cg,
    resize_cubic,
    resize_mask,
    resize_matrix,
)
from cdef.pipeline import enhance, load_config

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "cdef_paper.yaml"


def random_weights(shape, seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(0.1, 5.0, shape), rng.uniform(0.1, 5.0, shape)


# ---------------------------------------------------------------- resizing
def test_resize_matrix_rows_sum_to_one():
    for n_in, n_out in ((37, 19), (19, 37), (50, 50)):
        m = resize_matrix(n_in, n_out).toarray()
        assert m.shape == (n_out, n_in)
        assert np.allclose(m.sum(axis=1), 1.0)


def test_resize_cubic_keeps_constant_and_rounds_size_up():
    img = np.full((37, 51, 3), 0.4)
    out = resize_cubic(img, scale=0.5)
    assert out.shape == (19, 26, 3)
    assert np.allclose(out, 0.4)
    assert np.allclose(resize_cubic(out, 37, 51), 0.4)


def test_resize_cubic_matches_dense_separable_product():
    rng = np.random.default_rng(1)
    img = rng.random((23, 31))
    expected = resize_matrix(23, 10).toarray() @ img @ resize_matrix(31, 14).toarray().T
    assert np.allclose(resize_cubic(img, 10, 14), expected)


def test_resize_mask_keeps_uniform_masks():
    assert resize_mask(np.ones((40, 60), bool), 10, 10).all()
    assert not resize_mask(np.zeros((40, 60), bool), 10, 10).any()


# ------------------------------------------------------------ BIMEF solve
def test_periodic_system_rows_sum_to_one():
    wx, wy = random_weights((6, 7))
    A = periodic_system(wx, wy, 0.5)
    assert np.allclose(A @ np.ones(42), 1.0)
    assert abs(A - A.T).max() < 1e-12


def test_modified_incomplete_cholesky_keeps_row_sums():
    wx, wy = random_weights((6, 7), seed=2)
    A = periodic_system(wx, wy, 0.5)
    L = modified_incomplete_cholesky(A)
    ones = np.ones(A.shape[0])
    assert np.allclose(L @ (L.T @ ones), A @ ones)
    assert sparse.triu(L, k=1).nnz == 0


def test_preconditioned_cg_reaches_exact_solution_with_tight_tolerance():
    wx, wy = random_weights((8, 9), seed=3)
    A = periodic_system(wx, wy, 0.5)
    b = np.random.default_rng(4).random(A.shape[0])
    x = preconditioned_cg(A, b, modified_incomplete_cholesky(A), tol=1e-10, max_iter=500)
    assert np.allclose(x, spsolve(A, b), atol=1e-6)


def test_reference_luminance_runs_and_brightens(greenish):
    dark = greenish * 0.5
    out, k = luminance_compensation(dark, method="reference", lam=0.5, offset=0.01, clip=False)
    assert out.shape == dark.shape and np.isfinite(out).all()
    assert k >= 1.0
    assert out.mean() > dark.mean()


def test_luminance_rejects_unknown_method(greenish):
    with pytest.raises(ValueError):
        luminance_compensation(greenish, method="nope")


# ------------------------------------------------------- colour conversion
def test_lab_matches_skimage_in_gamut(greenish):
    assert np.abs(rgb2lab(greenish) - skimage_rgb2lab(greenish)).max() < 0.05


def test_lab_round_trip_keeps_out_of_range_values():
    rgb = np.array([[[0.5, 1.1, 0.2], [-0.01, 0.3, 1.05]]])
    assert np.allclose(lab2rgb(rgb2lab(rgb)), rgb, atol=1e-6)


def test_compensation_without_mask_also_changes_highlights():
    img = np.full((60, 60, 3), (0.2, 0.6, 0.5), np.float32)
    img[:, 30:] = (0.95, 0.9, 0.9)
    out = color_compensation(img, 1.0, 1.0, smooth_mask=False, use_mask=False)
    assert np.abs(out[:, 45:] - img[:, 45:]).max() > 1e-3


# -------------------------------------------------------------------- IDE
def test_channel_airlight_takes_brightest_local_minimum():
    img = np.full((40, 40, 3), 0.2)
    img[5:25, 5:25] = (0.7, 0.8, 0.9)
    img[30, 30] = 1.0  # a single bright pixel is removed by the local minimum
    assert np.allclose(channel_airlight(img, 10), (0.7, 0.8, 0.9))


def test_mean_inside_keeps_constant_up_to_the_border():
    img = np.full((20, 30), 0.3)
    assert np.allclose(mean_inside(img, 7), 0.3)


def test_rational_recovery_maps_airlight_to_one_at_t_min():
    airlight = np.array([0.6, 0.7, 0.8])
    img = np.broadcast_to(airlight, (4, 4, 3)).copy()
    t = np.full((4, 4), 0.4)
    assert np.allclose(rational_recovery(img, airlight, t, 0.4), 1.0)


def test_reference_ide_runs(greenish):
    out, tmin = ide(greenish, method="reference")
    assert out.shape == greenish.shape and np.isfinite(out).all()
    assert 0.01 < tmin < 1.1


def test_reference_ide_full_size_search_and_guided_smoothing_run(greenish):
    out, _ = ide(greenish, method="reference", delta=1, smoothing="guided", gf_radius=7, clip=True)
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_ide_rejects_unknown_method(greenish):
    with pytest.raises(ValueError):
        ide(greenish, method="nope")


# ----------------------------------------------------------------- fusion
def test_ycbcr_round_trip_and_gray_is_centred(greenish):
    assert np.abs(ycbcr_to_rgb(rgb_to_ycbcr(greenish)) - greenish).max() < 1e-5
    gray = rgb_to_ycbcr(np.full((2, 2, 3), 0.5))
    assert np.allclose(gray[..., 1:], 128 / 255, atol=1e-5)


def test_brightness_weights_sum_to_one_and_favour_brighter_image():
    w = [np.full((4, 4), 0.5), np.full((4, 4), 0.5)]
    dark, bright = np.full((4, 4, 3), 0.2), np.full((4, 4, 3), 0.8)
    out = brightness_weights(w, [dark, bright])
    assert np.allclose(out[0] + out[1], 1.0)
    assert (out[1] > out[0]).all()


WANG_OPTIONS = dict(
    phi=0.5, sigma=0.2, eps=1e-12, exponents=(2.0, 1.0, 2.0, 1.0), saliency_blur=3.0, saliency_window=3,
    colour="ycbcr", brightness=True, drop_levels=2, coarse_rule=True,
)


def test_wang_options_return_identical_input(greenish):
    out, weights = multiscale_fusion([greenish] * 4, **WANG_OPTIONS)
    assert np.abs(out - greenish).max() < 1e-3
    assert np.abs(sum(weights) - 1.0).max() < 1e-5


def test_wang_options_fuse_different_inputs(greenish, bluish):
    darker = greenish * 0.6
    out, weights = multiscale_fusion([greenish, bluish, darker, bluish * 0.9], **WANG_OPTIONS)
    assert out.shape == greenish.shape and np.isfinite(out).all()
    assert out.min() >= 0.0 and out.max() <= 1.0
    assert len(weights) == 4


def test_fusion_rejects_unknown_colour_space(greenish):
    with pytest.raises(ValueError):
        multiscale_fusion([greenish] * 2, colour="hsv")


# --------------------------------------------------------------- pipeline
def test_enhance_with_main_config(greenish, bluish):
    cfg = load_config(CONFIG)
    for img, tone in ((greenish, "greenish"), (bluish, "bluish")):
        out, debug = enhance(img, cfg)
        assert out.shape == img.shape and np.isfinite(out).all()
        assert out.min() >= 0.0 and out.max() <= 1.0
        assert debug["tone"] == tone
        assert np.abs(sum(debug["weights"]) - 1.0).max() < 1e-5
