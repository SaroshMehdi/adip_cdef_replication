"""The paper version (cdef_paper): settings as printed, and a working pipeline."""
from pathlib import Path

import numpy as np

from cdef_paper import enhance, load_config
from cdef_paper.cbm import cbm
from cdef_paper.fusion import LAPLACIAN, multiscale_fusion, normalize_weights
from cdef_paper.ide import ide
from cdef_paper.luminance import illumination_map, luminance_compensation

PAPER_ONLY = Path(__file__).parent.parent / "cdef_paper" / "paper_only.yaml"


def test_paper_only_config_holds_the_printed_values():
    cfg = load_config(PAPER_ONLY)
    assert cfg["discrimination"] == {"greenish": [0.5, 0.15], "bluish": [1.0, 0.3]}  # Eq. 3
    assert cfg["tone_mapping"] == {"a": 2.0, "b": 0.01}  # Eqs. 4-5
    assert cfg["color_compensation"]["mask_threshold"] == 0.85  # Eq. 2
    assert cfg["fusion"]["eps"] == 0.1  # Eq. 11
    assert cfg["fusion"]["exponents"] == [1.0, 1.0, 1.0, 1.0]  # Eq. 11
    assert cfg["fusion"]["space"] == "yuv"
    assert cfg["luminance"]["lam"] == 1.0 and cfg["luminance"]["mu"] == 0.5  # BIMEF, Sec. V


def test_pipeline_runs_on_both_tones(greenish, bluish):
    for img, tone in ((greenish, "greenish"), (bluish, "bluish")):
        out, _ = enhance(img)  # the default, calibrated config
        assert out.min() >= 0.0 and out.max() <= 1.0
        out, debug = enhance(img, load_config(PAPER_ONLY))
        assert debug["tone"] == tone
        assert out.shape == img.shape
        assert out.min() >= 0.0 and out.max() <= 1.0
        for name in ("F1", "F2", "F3", "F4"):  # every fusion input is an image in [0, 1]
            assert debug[name].min() >= 0.0 and debug[name].max() <= 1.0


def test_weights_sum_to_one_with_printed_eps(greenish):
    rng = np.random.default_rng(0)
    weights = normalize_weights([rng.random((8, 8)) for _ in range(4)], eps=0.1)
    assert np.allclose(sum(weights), 1.0)
    _, maps = multiscale_fusion([greenish, greenish**0.8, greenish**1.2, greenish * 0.9])
    assert np.allclose(sum(maps), 1.0, atol=1e-5)


def test_laplacian_ignores_flat_regions():
    assert LAPLACIAN.sum() == 0


def test_fusing_identical_inputs_returns_the_input(greenish):
    out, _ = multiscale_fusion([greenish] * 4)
    assert np.abs(out - greenish).max() < 1e-4


def test_luminance_brightens_a_dark_image(greenish):
    dark = greenish * 0.3
    out, k = luminance_compensation(dark)
    assert k > 1.0
    assert out.mean() > dark.mean()
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_illumination_map_of_a_flat_image_is_flat():
    flat = np.full((20, 30, 3), 0.4, np.float32)
    assert np.allclose(illumination_map(flat), 0.4, atol=1e-6)


def test_cbm_and_ide_return_images(greenish):
    assert 0.0 <= cbm(greenish).min() and cbm(greenish).max() <= 1.0
    albedo, tmin = ide(greenish)
    assert 0.0 <= albedo.min() and albedo.max() <= 1.0
    assert 0.05 <= tmin <= 0.95


def test_default_config_keeps_the_paper_fusion_constants():
    cfg = load_config()
    assert cfg["methods"] == {"luminance": "code", "ide": "code"}
    assert cfg["fusion"]["eps"] == 0.1 and cfg["fusion"]["exponents"] == [1.0, 1.0, 1.0, 1.0]
