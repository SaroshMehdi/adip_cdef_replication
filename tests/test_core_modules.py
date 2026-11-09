import numpy as np
import pytest
from skimage.color import rgb2lab

from cdef.color_comp import color_compensation, highlight_mask
from cdef.color_discrimination import classify_tone
from cdef.luminance import brightness_transform, entropy, estimate_illumination, luminance_compensation
from cdef.tone_mapping import (
    gray_world_range,
    gray_world_tone_map,
    statistic_range,
    statistic_tone_map,
    stretch,
)


def test_classify_tone_uses_eq3_values(greenish, bluish):
    assert classify_tone(greenish) == ("greenish", 0.5, 0.15)
    assert classify_tone(bluish) == ("bluish", 1.0, 0.3)


def test_classify_tone_tie_is_bluish():
    assert classify_tone(np.full((4, 4, 3), 0.5)).tone == "bluish"


def test_illumination_is_smooth_but_tracks_input(greenish):
    t = estimate_illumination(greenish, solve_scale=1.0)
    l = greenish.max(axis=2)
    assert t.shape == l.shape
    assert 0.0 <= t.min() and t.max() <= 1.0
    assert np.abs(np.diff(t, axis=1)).mean() < np.abs(np.diff(l, axis=1)).mean()
    assert abs(t.mean() - l.mean()) < 0.05


def test_btf_identity_at_k_one():
    x = np.linspace(0, 1, 11)
    assert np.allclose(brightness_transform(x, 1.0), x)


def test_btf_brightens_for_k_above_one():
    x = np.linspace(0.05, 0.9, 10)
    assert np.all(brightness_transform(x, 3.0) > x)


def test_entropy_of_uniform_histogram():
    assert entropy(np.linspace(0, 1, 256 * 4, endpoint=False)) == pytest.approx(8.0, abs=1e-6)


def test_lc_range_and_brightens_dark_image(greenish):
    dark = greenish * 0.25
    out, k = luminance_compensation(dark, solve_scale=1.0)
    assert out.min() >= 0.0 and out.max() <= 1.0
    assert k > 1.0
    assert out.mean() > dark.mean() * 1.2


def test_lc_nearly_unchanged_on_bright_image():
    bright = np.full((40, 50, 3), 0.92, np.float32)
    bright[10:30, 10:30] = 0.8
    out, k = luminance_compensation(bright, solve_scale=1.0)
    assert k == 1.0
    assert np.abs(out - bright).max() < 1e-5


def test_highlight_mask_eq2():
    img = np.zeros((2, 2, 3), np.float32)
    img[0, 0] = 0.9
    img[0, 1] = 0.84
    assert highlight_mask(img).tolist() == [[0.0, 1.0], [1.0, 1.0]]


def test_compensation_moves_cast_towards_neutral(greenish):
    out = color_compensation(greenish, 1.0, 0.3, sigma_ratio=0.1)
    before = rgb2lab(greenish)[..., 1:].mean(axis=(0, 1))
    after = rgb2lab(out)[..., 1:].mean(axis=(0, 1))
    assert abs(after[0]) < 0.5 * abs(before[0])
    assert abs(after[1]) < abs(before[1])
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_full_weight_with_small_sigma_removes_chroma(greenish):
    """Paper Fig. 4, "Without Parameter": alpha = beta = 1 leaves a nearly gray image."""
    out = color_compensation(greenish, 1.0, 1.0, sigma=0.5)
    assert np.abs(rgb2lab(out)[..., 1:]).mean() < 0.2 * np.abs(rgb2lab(greenish)[..., 1:]).mean()


def test_compensation_is_noop_with_zero_weights(greenish):
    assert np.abs(color_compensation(greenish, 0.0, 0.0) - greenish).max() < 1e-3


def test_compensation_leaves_highlights_alone():
    img = np.full((60, 60, 3), (0.2, 0.6, 0.5), np.float32)
    img[:, 30:] = (0.95, 0.9, 0.9)
    out = color_compensation(img, 1.0, 1.0, smooth_mask=False)
    assert np.abs(out[:, 45:] - img[:, 45:]).max() < 1e-3
    assert np.abs(out[:, :15] - img[:, :15]).max() > 0.05


def test_statistic_range_eq4():
    channel = np.array([0.0, 1.0], np.float32)
    assert statistic_range(channel, a=2.0) == pytest.approx((-0.5, 1.5))


def test_gray_world_range_eq5_scales_with_channel_mean():
    channel = np.linspace(0, 1, 10001, dtype=np.float32)
    low, high = gray_world_range(channel, gray_mean=channel.mean() * 2, b=0.01)
    assert low == pytest.approx(0.02, abs=1e-3)
    assert high == pytest.approx(0.98, abs=1e-3)


def test_stretch_eq6_maps_limits_to_output_range():
    out = stretch(np.array([0.2, 0.4, 0.6, 0.8], np.float32), 0.4, 0.6)
    assert out.tolist() == pytest.approx([0.0, 0.0, 1.0, 1.0])


def test_stretch_keeps_flat_channel():
    flat = np.full((4, 4), 0.3, np.float32)
    assert np.array_equal(stretch(flat, 0.3, 0.3), flat)


def test_tone_maps_increase_contrast(greenish):
    for mapped in (statistic_tone_map(greenish), gray_world_tone_map(greenish)):
        assert mapped.shape == greenish.shape
        assert mapped.min() >= 0.0 and mapped.max() <= 1.0
        assert mapped.std() > greenish.std()
