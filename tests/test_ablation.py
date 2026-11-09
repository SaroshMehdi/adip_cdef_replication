"""Tests for the ablation switches (Table IV): stages that can be turned off and
the RGB and Lab fusion spaces."""
import copy
from pathlib import Path

import numpy as np
import pytest

from cdef.fusion import lab_unit_to_rgb, multiscale_fusion, rgb_to_lab_unit
from cdef.pipeline import enhance, load_config

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "cdef_paper.yaml"


def config_with(**sections):
    cfg = copy.deepcopy(load_config(CONFIG))
    for name, values in sections.items():
        cfg.setdefault(name, {}).update(values)
    return cfg


@pytest.mark.parametrize(
    "stages, n_inputs",
    [
        ({"luminance": False}, 4),
        ({"color_compensation": False}, 4),
        ({"tone_mapping": False}, 4),
        ({"cbm": False}, 3),
        ({"ide": False}, 3),
    ],
)
def test_each_stage_can_be_switched_off(greenish, stages, n_inputs):
    out, debug = enhance(greenish, config_with(stages=stages))
    assert out.shape == greenish.shape and np.isfinite(out).all()
    assert out.min() >= 0.0 and out.max() <= 1.0
    assert len(debug["weights"]) == n_inputs


def test_without_luminance_the_compensation_sees_the_raw_image(greenish):
    _, debug = enhance(greenish, config_with(stages={"luminance": False}))
    assert np.array_equal(debug["luminance"], greenish)
    assert debug["k"] == 1.0


def test_without_tone_mapping_f1_and_f2_are_the_compensated_image(bluish):
    _, debug = enhance(bluish, config_with(stages={"tone_mapping": False}))
    assert np.array_equal(debug["F1"], debug["compensated"])
    assert np.array_equal(debug["F2"], debug["compensated"])


def test_unknown_stage_and_no_detail_input_are_rejected(greenish):
    with pytest.raises(ValueError):
        enhance(greenish, config_with(stages={"fusion": False}))
    with pytest.raises(ValueError):
        enhance(greenish, config_with(stages={"cbm": False, "ide": False}))


def test_lab_unit_round_trip(greenish):
    assert np.abs(lab_unit_to_rgb(rgb_to_lab_unit(greenish)) - greenish).max() < 1e-4


@pytest.mark.parametrize("colour", ["rgb", "lab"])
def test_rgb_and_lab_fusion_return_identical_input(greenish, colour):
    out, weights = multiscale_fusion([greenish] * 3, phi=0.5, sigma=0.2, colour=colour)
    assert np.abs(out - greenish).max() < 1e-3
    assert np.abs(sum(weights) - 1.0).max() < 1e-5


@pytest.mark.parametrize("colour", ["rgb", "lab"])
def test_rgb_and_lab_fusion_run_with_main_settings(greenish, bluish, colour):
    fusion = dict(load_config(CONFIG)["fusion"], colour=colour)
    out, _ = multiscale_fusion([greenish, bluish, greenish * 0.6], **fusion)
    assert out.shape == greenish.shape and np.isfinite(out).all()


def test_coarse_rule_is_rejected_in_rgb(greenish):
    with pytest.raises(ValueError):
        multiscale_fusion([greenish] * 2, colour="rgb", coarse_rule=True)
