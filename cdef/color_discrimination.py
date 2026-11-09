"""Greenish / bluish test (Algorithm 1, lines 5-9; Eq. 3)."""
from typing import NamedTuple


class ToneDecision(NamedTuple):
    tone: str  # "greenish" or "bluish"
    alpha: float
    beta: float


def classify_tone(raw, greenish=(0.5, 0.15), bluish=(1.0, 0.3)):
    """Eq. 3: pick the tone and (alpha, beta) from the raw G and B channel means.

    Eq. 3 leaves mean_G == mean_B undefined; Algorithm 1's else branch makes
    that case bluish.
    """
    mean_g = float(raw[..., 1].mean())
    mean_b = float(raw[..., 2].mean())
    if mean_g > mean_b:
        return ToneDecision("greenish", float(greenish[0]), float(greenish[1]))
    return ToneDecision("bluish", float(bluish[0]), float(bluish[1]))
