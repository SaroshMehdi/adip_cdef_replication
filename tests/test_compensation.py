"""Phase 1: tone test, luminance compensation (BIMEF) and colour compensation."""
import numpy as np
import pytest
from scipy.sparse.linalg import spsolve
from skimage.color import rgb2lab as skimage_rgb2lab

from cdef.color_compensation import color_compensation, highlight_mask
from cdef.color_spaces import lab_to_rgb, rgb_to_lab
from cdef.discrimination import classify_tone
from cdef.luminance import (
    brightness_transform,
    conjugate_gradients,
    entropy_8bit,
    illumination_map,
    incomplete_cholesky,
    luminance_compensation,
    smoothing_system,
)
from cdef.resize import resize_cubic, resize_mask, resize_matrix


# ------------------------------------------------------------ tone test
def test_classify_tone_uses_eq3_values(greenish, bluish):
    assert classify_tone(greenish) == ("greenish", 0.5, 0.15)
    assert classify_tone(bluish) == ("bluish", 1.0, 0.3)


def test_classify_tone_tie_is_bluish():
    assert classify_tone(np.full((4, 4, 3), 0.5)).tone == "bluish"


# --------------------------------------------------------------- resize
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


def test_resize_cubic_matches_separable_product():
    img = np.random.default_rng(1).random((23, 31))
    expected = resize_matrix(23, 10).toarray() @ img @ resize_matrix(31, 14).toarray().T
    assert np.allclose(resize_cubic(img, 10, 14), expected)


def test_resize_mask_keeps_uniform_masks():
    assert resize_mask(np.ones((40, 60), bool), 10, 10).all()
    assert not resize_mask(np.zeros((40, 60), bool), 10, 10).any()


# ---------------------------------------------------------------- BIMEF
def random_weights(shape, seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(0.1, 5.0, shape), rng.uniform(0.1, 5.0, shape)


def test_smoothing_system_is_symmetric_with_unit_row_sums():
    wx, wy = random_weights((6, 7))
    A = smoothing_system(wx, wy, 0.5)
    assert np.allclose(A @ np.ones(42), 1.0)
    assert abs(A - A.T).max() < 1e-12


def test_incomplete_cholesky_keeps_row_sums():
    wx, wy = random_weights((6, 7), seed=2)
    A = smoothing_system(wx, wy, 0.5)
    L = incomplete_cholesky(A)
    ones = np.ones(A.shape[0])
    assert np.allclose(L @ (L.T @ ones), A @ ones)


def test_conjugate_gradients_reaches_exact_solution():
    wx, wy = random_weights((8, 9), seed=3)
    A = smoothing_system(wx, wy, 0.5)
    b = np.random.default_rng(4).random(A.shape[0])
    x = conjugate_gradients(A, b, incomplete_cholesky(A), tol=1e-10, max_iter=500)
    assert np.allclose(x, spsolve(A, b), atol=1e-6)


def test_illumination_map_is_smooth_but_tracks_input(greenish):
    t = illumination_map(greenish.astype(np.float64))
    l = greenish.max(axis=2)
    assert t.shape == l.shape
    assert np.abs(np.diff(t, axis=1)).mean() < np.abs(np.diff(l, axis=1)).mean()
    assert abs(t.mean() - l.mean()) < 0.05


def test_brightness_transform_identity_at_k_one_and_brightens_above():
    x = np.linspace(0.05, 0.9, 10)
    assert np.allclose(brightness_transform(x, 1.0), x)
    assert np.all(brightness_transform(x, 3.0) > x)


def test_entropy_of_uniform_levels_is_eight_bits():
    assert entropy_8bit(np.arange(256) / 255.0) == pytest.approx(8.0)


def test_luminance_brightens_dark_image(greenish):
    dark = greenish * 0.25
    out, k = luminance_compensation(dark)
    assert out.min() >= 0.0 and out.max() <= 1.0
    assert k > 1.0
    assert out.mean() > dark.mean() * 1.2


def test_luminance_leaves_bright_image_alone():
    bright = np.full((40, 50, 3), 0.92, np.float32)
    bright[10:30, 10:30] = 0.8
    out, k = luminance_compensation(bright)
    assert k == 1.0
    assert np.abs(out - bright).max() < 1e-5


# ------------------------------------------------- colour compensation
def test_lab_matches_skimage_in_gamut(greenish):
    assert np.abs(rgb_to_lab(greenish) - skimage_rgb2lab(greenish)).max() < 0.05


def test_lab_round_trip_keeps_out_of_range_values():
    rgb = np.array([[[0.5, 1.1, 0.2], [-0.01, 0.3, 1.05]]])
    assert np.allclose(lab_to_rgb(rgb_to_lab(rgb)), rgb, atol=1e-6)


def test_highlight_mask_eq2():
    img = np.zeros((2, 2, 3), np.float32)
    img[0, 0] = 0.9
    img[0, 1] = 0.84
    assert highlight_mask(img).tolist() == [[0.0, 1.0], [1.0, 1.0]]


def test_compensation_moves_cast_towards_neutral(greenish):
    out = color_compensation(greenish, 1.0, 0.3, sigma=10.0)
    before = rgb_to_lab(greenish)[..., 1:].mean(axis=(0, 1))
    after = rgb_to_lab(out)[..., 1:].mean(axis=(0, 1))
    assert abs(after[0]) < 0.5 * abs(before[0])
    assert abs(after[1]) < abs(before[1])
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_full_strength_with_small_sigma_removes_chroma(greenish):
    """Paper Fig. 4, "Without Parameter": alpha = beta = 1 leaves a nearly grey image."""
    out = color_compensation(greenish, 1.0, 1.0, sigma=0.5)
    assert np.abs(rgb_to_lab(out)[..., 1:]).mean() < 0.2 * np.abs(rgb_to_lab(greenish)[..., 1:]).mean()


def test_compensation_is_noop_with_zero_strength(greenish):
    assert np.abs(color_compensation(greenish, 0.0, 0.0) - greenish).max() < 1e-3


def test_compensation_leaves_highlights_alone():
    img = np.full((60, 60, 3), (0.2, 0.6, 0.5), np.float32)
    img[:, 30:] = (0.95, 0.9, 0.9)
    out = color_compensation(img, 1.0, 1.0)
    assert np.abs(out[:, 45:] - img[:, 45:]).max() < 1e-3
    assert np.abs(out[:, :15] - img[:, :15]).max() > 0.05
