"""Phase 2: the four fusion inputs (tone maps F1, F2; CBM F3; IDE F4)."""
import math

import numpy as np
import pytest

from cdef.cbm import cbm, closing, contour_element, opening, top_hats
from cdef.ide import (
    ambient_light,
    golden_section,
    ide,
    mean_inside,
    patch_transmission,
    quadratic_coefficients,
    rational_log,
    recover_albedo,
)
from cdef.tone_mapping import gray_world_range, gray_world_tone_map, statistic_range, statistic_tone_map, stretch


# ----------------------------------------------------------- tone maps
def test_statistic_range_eq4():
    assert statistic_range(np.array([0.0, 1.0], np.float32), a=2.0) == pytest.approx((-0.5, 1.5))


def test_gray_world_range_eq5_scales_with_channel_mean():
    channel = np.linspace(0, 1, 10001, dtype=np.float32)
    low, high = gray_world_range(channel, gray_mean=channel.mean() * 2, b=0.01)
    assert low == pytest.approx(0.02, abs=1e-3)
    assert high == pytest.approx(0.98, abs=1e-3)


def test_stretch_eq6_maps_limits_to_unit_range():
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


# ----------------------------------------------------------------- CBM
def test_contour_element_is_hollow():
    element = contour_element(5)
    assert element.sum() == 16
    assert element[2, 2] == 0 and element[0, 0] == 1


def test_opening_is_anti_extensive_and_closing_extensive(greenish):
    assert np.all(opening(greenish, contour_element(5)) <= greenish + 1e-6)
    assert np.all(closing(greenish, contour_element(5)) >= greenish - 1e-6)


def test_cbm_is_identity_on_flat_image_and_with_zero_omega(greenish):
    flat = np.full((40, 40, 3), 0.4, np.float32)
    assert np.abs(cbm(flat) - flat).max() < 1e-6
    assert np.array_equal(cbm(greenish, omega=0.0), greenish)


def test_cbm_matches_eq11_written_out(greenish):
    wh, bh = zip(*top_hats(greenish, iterations=3))
    expected = sum(w - b for w, b in zip(wh, bh))
    expected = expected + (wh[1] - wh[0]) - (bh[1] - bh[0]) + (wh[2] - wh[1]) - (bh[2] - bh[1])
    assert np.abs(cbm(greenish, omega=0.1, iterations=3) - (greenish + 0.1 * expected)).max() < 1e-5


def test_cbm_boosts_local_contrast(greenish):
    out = cbm(greenish)
    assert np.abs(np.diff(out, axis=1)).mean() > np.abs(np.diff(greenish, axis=1)).mean()


def test_cbm_keeps_overshoot_for_the_fusion():
    img = np.full((40, 40, 3), 0.97, np.float32)
    img[10, :] = 1.0
    img[25:, :] = 0.03
    img[32, :] = 0.0
    out = cbm(img)
    assert out.min() < 0.0 and out.max() > 1.0


# ----------------------------------------------------------------- IDE
def test_rational_log_fits_ln_away_from_one():
    t = np.linspace(0.05, 0.7, 30)
    assert np.abs(rational_log(t) - np.log(t)).max() < 0.2


def test_ambient_light_takes_brightest_local_minimum():
    img = np.full((40, 40, 3), 0.2)
    img[5:25, 5:25] = (0.7, 0.8, 0.9)
    img[30, 30] = 1.0  # a single bright pixel is removed by the local minimum
    assert np.allclose(ambient_light(img, 10), (0.7, 0.8, 0.9))


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


def test_recovery_maps_airlight_to_one_at_tmin():
    airlight = np.array([0.6, 0.7, 0.8])
    img = np.broadcast_to(airlight, (4, 4, 3)).copy()
    assert np.allclose(recover_albedo(img, airlight, np.full((4, 4), 0.4), 0.4), 1.0)


def test_golden_section_finds_minimum():
    assert golden_section(lambda x: (x - 0.37) ** 2, 0.0, 1.0) == pytest.approx(0.37, abs=0.01)


def test_mean_inside_keeps_constant_up_to_the_border():
    assert np.allclose(mean_inside(np.full((20, 30), 0.3), 7), 0.3)


def test_ide_runs_and_returns_tmin(greenish):
    out, tmin = ide(greenish)
    assert out.shape == greenish.shape and np.isfinite(out).all()
    assert 0.01 < tmin < 1.1
