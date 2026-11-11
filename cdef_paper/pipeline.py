"""The full CDEF pipeline: Algorithm 1 followed by multiscale exposure fusion."""
from pathlib import Path

import numpy as np
import yaml

from . import ide as ide_paper
from . import ide_code
from . import luminance as luminance_paper
from . import luminance_code
from .cbm import cbm
from .color_compensation import color_compensation
from .discrimination import classify_tone
from .fusion import multiscale_fusion
from .tone_mapping import gray_world_tone_map, statistic_tone_map

DEFAULT_CONFIG = Path(__file__).with_name("config.yaml")

# Stages that can be switched off for the ablation study (Table IV).
STAGES = ("luminance", "color_compensation", "tone_mapping", "cbm", "ide")

# BIMEF and IDE each exist twice: "paper" follows the cited paper, "code"
# follows the code shipped with the authors' release. The `methods` section of
# a config picks one; its settings are in the section of the same name
# (luminance / luminance_code, ide / ide_code).
LUMINANCE = {"paper": (luminance_paper, "luminance"), "code": (luminance_code, "luminance_code")}
IDE = {"paper": (ide_paper, "ide"), "code": (ide_code, "ide_code")}


def load_config(path=DEFAULT_CONFIG):
    """Read a YAML config. Each section holds keyword arguments for one stage.

    A config may name a `base` config (relative to its own folder); its
    sections are then updated with this file's values, so a variant only
    lists what it changes.
    """
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    base = cfg.pop("base", None)
    if base is None:
        return cfg
    merged = load_config(path.parent / base)
    for section, values in cfg.items():
        merged.setdefault(section, {}).update(values)
    return merged


def enhance(img, cfg=None):
    """Enhance a float RGB image in [0, 1].

    Returns (output, debug). debug holds every intermediate image (F1-F4, the
    fusion weights, ...) for figures like the paper's Fig. 2.
    """
    cfg = load_config() if cfg is None else cfg
    stages = dict.fromkeys(STAGES, True)
    for name, on in cfg.get("stages", {}).items():
        if name not in stages:
            raise ValueError(f"unknown stage: {name}")
        stages[name] = on
    methods = cfg.get("methods", {})
    lum_module, lum_section = LUMINANCE[methods.get("luminance", "paper")]
    ide_module, ide_section = IDE[methods.get("ide", "paper")]

    # Degradation compensation (Algorithm 1, lines 2-11). The tone is decided
    # on the raw image (Eq. 3).
    decision = classify_tone(img, **cfg.get("discrimination", {}))
    if stages["luminance"]:
        exposed, k = lum_module.luminance_compensation(img, **cfg.get(lum_section, {}))
    else:
        exposed, k = img, 1.0
    if stages["color_compensation"]:
        compensated = color_compensation(exposed, decision.alpha, decision.beta, **cfg.get("color_compensation", {}))
    else:
        compensated = np.clip(exposed, 0.0, 1.0).astype(np.float32)

    # Fusion preparation (lines 12-22): two tone maps, then contours and
    # dehazing on the tone map that suits the image's tone.
    if stages["tone_mapping"]:
        tone_cfg = cfg.get("tone_mapping", {})
        f1 = statistic_tone_map(compensated, tone_cfg.get("a", 2.0))
        f2 = gray_world_tone_map(compensated, tone_cfg.get("b", 0.01))
    else:  # without TM, both tone-mapped inputs are the compensated image
        f1 = f2 = compensated
    base = f1 if decision.tone == "greenish" else f2
    # Ablation: without CBM its input is left out; without IDE the fourth
    # input is the tone-mapped image it would have dehazed. (The paper does not
    # say how it removed them; these are the readings closest to its Table IV.)
    f3 = cbm(base, **cfg.get("cbm", {})) if stages["cbm"] else None
    f4, tmin = ide_module.ide(base, **cfg.get(ide_section, {})) if stages["ide"] else (base, None)

    # Multiscale exposure fusion (Eqs. 7-14).
    inputs = [f for f in (f1, f2, f3, f4) if f is not None]
    output, weights = multiscale_fusion(inputs, **cfg.get("fusion", {}))

    debug = {
        "tone": decision.tone,
        "alpha": decision.alpha,
        "beta": decision.beta,
        "k": k,
        "tmin": tmin,
        "luminance": exposed,
        "compensated": compensated,
        "F1": f1,
        "F2": f2,
        "F3": f3,
        "F4": f4,
        "weights": weights,
    }
    return output, debug
