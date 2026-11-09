import math

import numpy as np
import pytest

from cdef.detail.cbm import cbm, contour_element, opening, top_hat_series
from cdef.detail.ide import (
    atmospheric_light,
    golden_section,
    guided_filter,
    ide,
    patch_transmission,
    quadratic_coefficients,
    rational_log,
    scene_albedo,
)


def test_contour_element_is_hollow():
    element = contour_element(5)
    assert element.sum() == 16
    assert element[2, 2] == 0 and element[0, 0] == 1


def test_opening_with_hollow_element_is_anti_extensive(greenish):
    assert np.all(opening(greenish, contour_element(5)) <= greenish + 1e-6)


def test_cbm_is_identity_on_flat_image():
    flat = np.full((40, 40, 3), 0.4, np.float32)
    assert np.abs(cbm(flat) - flat).max() < 1e-6


def test_cbm_zero_omega_is_identity(greenish):
    assert np.array_equal(cbm(greenish, omega=0.0), greenish)


def test_cbm_matches_eq11_written_out(greenish):
    series = list(top_hat_series(greenish, iterations=3))
    wh, bh = zip(*series)
    expected = sum(w - b for w, b in zip(wh, bh))
    expected = expected + (wh[1] - wh[0]) - (bh[1] - bh[0])
    expected = expected + (wh[2] - wh[1]) - (bh[2] - bh[1])
    out = cbm(greenish, omega=0.1, iterations=3)
    assert np.abs(out - (greenish + 0.1 * expected)).max() < 1e-5


def test_cbm_boosts_local_contrast(greenish):
    out = cbm(greenish)
    assert np.abs(np.diff(out, axis=1)).mean() > np.abs(np.diff(greenish, axis=1)).mean()


def test_cbm_unclipped_keeps_overshoot_and_clip_removes_it():
    # Thin details near the ends of the range: the top-hats boost them past 0 and 1.
    img = np.full((40, 40, 3), 0.97, np.float32)
    img[10, :] = 1.0
    img[25:, :] = 0.03
    img[32, :] = 0.0
    unclipped = cbm(img)
    assert unclipped.min() < 0.0 and unclipped.max() > 1.0
    clipped = cbm(img, clip=True)
    assert clipped.min() >= 0.0 and clipped.max() <= 1.0
    assert np.array_equal(clipped, np.clip(unclipped, 0.0, 1.0))


def test_repeat_mode_filter_is_idempotent(greenish):
    """Why repeat mode adds the same detail n times: a second pass changes nothing."""
    from cdef.detail.cbm import close_open, open_close

    b1, b2 = contour_element(15), contour_element(5)
    once = open_close(greenish, b1, b2)
    assert np.array_equal(open_close(once, b1, b2), once)
    once = close_open(greenish, b1, b2)
    assert np.array_equal(close_open(once, b1, b2), once)


def test_cbm_scale_mode_runs(greenish):
    assert cbm(greenish, iterations=3, mode="scale").shape == greenish.shape


def test_cbm_rejects_unknown_mode(greenish):
    with pytest.raises(ValueError):
        cbm(greenish, mode="nope")


def test_rational_log_fits_ln_away_from_one():
    t = np.linspace(0.05, 0.7, 30)
    assert np.abs(rational_log(t) - np.log(t)).max() < 0.2


def test_atmospheric_light_finds_bright_region():
    img = np.full((64, 64, 3), 0.2, np.float32)
    img[:20, :20] = (0.8, 0.9, 0.7)
    assert atmospheric_light(img) == pytest.approx(np.array([0.8, 0.9, 0.7]), abs=1e-5)


def test_transmission_is_a_root_of_eq15():
    mean, a_mean, ln_tmin = np.linspace(0.1, 0.8, 8), 0.9, math.log(0.3)
    t = patch_transmission(mean, a_mean, ln_tmin)
    lam1, lam2, lam3 = quadratic_coefficients(mean, a_mean, ln_tmin)
    assert np.abs(lam1 * t**2 + lam2 * t + lam3).max() < 1e-9
    assert np.all(t > 0)


def test_transmission_solves_eq12_with_rational_log():
    mean, a_mean, ln_tmin = np.linspace(0.3, 0.8, 6), 0.9, math.log(0.3)
    t = patch_transmission(mean, a_mean, ln_tmin)
    rebuilt = a_mean * rational_log(t) / (2 * ln_tmin) * t + a_mean * (1 - t)
    assert np.abs(rebuilt - mean).max() < 1e-9


def test_albedo_inverts_easm_exact_log():
    albedo = np.array([[[0.2, 0.5, 0.8]]], np.float32)
    airlight = np.array([0.9, 0.95, 0.85], np.float32)
    t = np.array([[0.6]], np.float32)
    ln_tmin = math.log(0.2)
    hazy = airlight * (np.log(t[..., None]) / ln_tmin) * albedo * t[..., None]
    hazy = hazy + airlight * (1 - t[..., None])
    out = scene_albedo(hazy, airlight, t, ln_tmin, "exact")
    assert np.abs(out - albedo).max() < 1e-4


def test_rational_recovery_gives_half_albedo_on_gray_patch():
    airlight = np.array([0.9, 0.9, 0.9], np.float32)
    mean = np.array([[0.5]], np.float32)
    ln_tmin = math.log(0.3)
    t = patch_transmission(mean, 0.9, ln_tmin)
    out = scene_albedo(np.full((1, 1, 3), 0.5, np.float32), airlight, t, ln_tmin, "rational")
    assert out == pytest.approx(0.5, abs=1e-5)


def test_guided_filter_preserves_strong_edge():
    guide = np.zeros((40, 40), np.float32)
    guide[:, 20:] = 1.0
    out = guided_filter(guide, guide.copy(), 4, 1e-4)
    assert np.abs(out - guide).max() < 0.02


def test_golden_section_finds_minimum():
    assert golden_section(lambda x: (x - 0.37) ** 2, 0.0, 1.0) == pytest.approx(0.37, abs=0.01)


def test_ide_output_and_clipping_budget(greenish):
    out, tmin = ide(greenish, patch=15, gf_radius=7)
    assert out.shape == greenish.shape
    assert out.min() >= 0.0 and out.max() <= 1.0
    assert 0.0 < tmin < 1.0


def test_ide_exact_mode_runs(greenish):
    out, _ = ide(greenish, patch=15, gf_radius=7, log_mode="exact")
    assert np.isfinite(out).all()


def test_ide_rejects_unknown_log_mode(greenish):
    with pytest.raises(ValueError):
        ide(greenish, log_mode="nope")
