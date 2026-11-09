"""The full pipeline, the configs and the ablation switches (Table IV)."""
from pathlib import Path

import numpy as np
import pytest

from cdef.cbm import cbm
from cdef.pipeline import enhance, load_config

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
MAIN = load_config(CONFIGS / "cdef.yaml")


def with_stages(**stages):
    return {**MAIN, "stages": stages}


def test_main_config_end_to_end(greenish, bluish):
    for img, tone in ((greenish, "greenish"), (bluish, "bluish")):
        out, debug = enhance(img, MAIN)
        assert out.shape == img.shape and np.isfinite(out).all()
        assert out.min() >= 0.0 and out.max() <= 1.0
        assert out.std() > img.std()
        assert debug["tone"] == tone
        assert len(debug["weights"]) == 4
        assert np.abs(sum(debug["weights"]) - 1.0).max() < 1e-5


def test_base_is_the_tone_matched_tone_map(greenish, bluish):
    # CBM is fed F1 for greenish images and F2 for bluish ones (Algorithm 1, lines 16-22).
    cfg = with_stages(ide=False)
    for img, base in ((greenish, "F1"), (bluish, "F2")):
        _, debug = enhance(img, cfg)
        assert np.array_equal(debug["F3"], cbm(debug[base], **MAIN["cbm"]))


@pytest.mark.parametrize("name", ["wo_lc", "wo_tm", "wo_cbm", "wo_ide", "mef_rgb", "mef_lab"])
def test_ablation_configs_change_one_thing(name):
    cfg = load_config(CONFIGS / "ablation" / f"{name}.yaml")
    changed = {s for s in set(cfg) | set(MAIN) if cfg.get(s) != MAIN.get(s)}
    assert changed in ({"stages"}, {"fusion"})
    if changed == {"fusion"}:
        assert {k for k in cfg["fusion"] if cfg["fusion"][k] != MAIN["fusion"][k]} == {"space"}


@pytest.mark.parametrize(
    "stage, n_inputs",
    [("luminance", 4), ("color_compensation", 4), ("tone_mapping", 4), ("cbm", 3), ("ide", 3)],
)
def test_each_stage_can_be_switched_off(greenish, stage, n_inputs):
    out, debug = enhance(greenish, with_stages(**{stage: False}))
    assert np.isfinite(out).all() and out.min() >= 0.0 and out.max() <= 1.0
    assert len(debug["weights"]) == n_inputs


def test_without_luminance_the_raw_image_is_compensated(greenish):
    _, debug = enhance(greenish, with_stages(luminance=False))
    assert np.array_equal(debug["luminance"], greenish)
    assert debug["k"] == 1.0


def test_without_tone_mapping_f1_and_f2_are_the_compensated_image(bluish):
    _, debug = enhance(bluish, with_stages(tone_mapping=False))
    assert np.array_equal(debug["F1"], debug["compensated"])
    assert np.array_equal(debug["F2"], debug["compensated"])


def test_unknown_stage_and_no_detail_input_are_rejected(greenish):
    with pytest.raises(ValueError):
        enhance(greenish, with_stages(fusion=False))
    with pytest.raises(ValueError):
        enhance(greenish, with_stages(cbm=False, ide=False))
