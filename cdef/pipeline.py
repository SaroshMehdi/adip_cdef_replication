"""The full CDEF pipeline (Algorithm 1 and multiscale exposure fusion)."""
import numpy as np
import yaml

from .color_comp import color_compensation
from .color_discrimination import classify_tone
from .detail.cbm import cbm
from .detail.ide import ide
from .fusion import multiscale_fusion
from .luminance import luminance_compensation
from .tone_mapping import gray_world_tone_map, statistic_tone_map


def load_config(path):
    """Read a YAML config. Each section holds keyword arguments for one module."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def enhance(img, cfg=None):
    """Enhance a float RGB image in [0, 1].

    Returns (output, debug). debug holds every intermediate image, so figures
    like the paper's Fig. 2 can be built from it.
    """
    cfg = cfg or {}
    # Stage switches for the ablation study (Table IV); all on by default.
    stages = {"luminance": True, "color_compensation": True, "tone_mapping": True, "cbm": True, "ide": True}
    unknown = set(cfg.get("stages", {})) - set(stages)
    if unknown:
        raise ValueError(f"unknown stages: {sorted(unknown)}")
    stages.update(cfg.get("stages", {}))
    if not (stages["cbm"] or stages["ide"]):
        raise ValueError("at least one of cbm and ide must be on")

    decision = classify_tone(img, **cfg.get("discrimination", {}))  # lines 5-9
    if stages["luminance"]:
        exposed, k = luminance_compensation(img, **cfg.get("luminance", {}))  # line 2
    else:  # -w/o LC
        exposed, k = img, 1.0
    if stages["color_compensation"]:
        compensated = color_compensation(  # lines 3-11
            exposed, decision.alpha, decision.beta, **cfg.get("color_compensation", {})
        )
    else:  # -w/o CC
        compensated = np.clip(exposed, 0.0, 1.0).astype(np.float32)

    tone_cfg = cfg.get("tone_mapping", {})
    if stages["tone_mapping"]:
        f1 = statistic_tone_map(compensated, tone_cfg.get("a", 2.0))  # line 13
        f2 = gray_world_tone_map(compensated, tone_cfg.get("b", 0.01))  # line 14
    else:  # -w/o TM: both tone-mapped inputs become the compensated image
        f1, f2 = compensated, compensated

    base = f1 if decision.tone == "greenish" else f2  # lines 16-22
    f3 = cbm(base, **cfg.get("cbm", {})) if stages["cbm"] else None  # -w/o CBM drops F3
    f4, tmin = ide(base, **cfg.get("ide", {})) if stages["ide"] else (None, None)  # -w/o IDE drops F4

    inputs = [x for x in (f1, f2, f3, f4) if x is not None]
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
