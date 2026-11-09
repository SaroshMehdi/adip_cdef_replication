"""The full CDEF pipeline: Algorithm 1 followed by multiscale exposure fusion."""
from pathlib import Path

import numpy as np
import yaml

from .cbm import cbm
from .color_compensation import color_compensation
from .discrimination import classify_tone
from .fusion import multiscale_fusion
from .ide import ide
from .luminance import luminance_compensation
from .tone_mapping import gray_world_tone_map, statistic_tone_map

# Stages that can be switched off for the ablation study (Table IV).
STAGES = ("luminance", "color_compensation", "tone_mapping", "cbm", "ide")


def load_config(path):
    """Read a YAML config. Each section holds keyword arguments for one stage.

    A config may name a `base` config (relative to its own folder); its
    sections are then updated with this file's values. The ablation configs
    use this to change one setting of the main config.
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
    cfg = cfg or {}
    stages = dict.fromkeys(STAGES, True)
    for name, on in cfg.get("stages", {}).items():
        if name not in stages:
            raise ValueError(f"unknown stage: {name}")
        stages[name] = on
    if not (stages["cbm"] or stages["ide"]):
        raise ValueError("at least one of cbm and ide must be on")

    # Phase 1: degradation compensation.
    decision = classify_tone(img, **cfg.get("discrimination", {}))
    if stages["luminance"]:
        exposed, k = luminance_compensation(img, **cfg.get("luminance", {}))
    else:
        exposed, k = img, 1.0
    if stages["color_compensation"]:
        compensated = color_compensation(exposed, decision.alpha, decision.beta, **cfg.get("color_compensation", {}))
    else:
        compensated = np.clip(exposed, 0.0, 1.0).astype(np.float32)

    # Phase 2: the four fusion inputs.
    if stages["tone_mapping"]:
        tone_cfg = cfg.get("tone_mapping", {})
        f1 = statistic_tone_map(compensated, tone_cfg.get("a", 2.0))
        f2 = gray_world_tone_map(compensated, tone_cfg.get("b", 0.01))
    else:  # without TM, both tone-mapped inputs are the compensated image
        f1 = f2 = compensated
    base = f1 if decision.tone == "greenish" else f2
    f3 = cbm(base, **cfg.get("cbm", {})) if stages["cbm"] else None
    f4, tmin = ide(base, **cfg.get("ide", {})) if stages["ide"] else (None, None)

    # Phase 3: fusion (a switched-off input is left out).
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
