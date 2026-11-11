"""Colour tone discrimination (Algorithm 1, lines 5-9; Eq. 3)."""
from typing import NamedTuple


class ToneDecision(NamedTuple):
    tone: str  # "greenish" or "bluish"
    alpha: float  # colour compensation strength for a*
    beta: float  # colour compensation strength for b*


def classify_tone(raw, greenish=(0.5, 0.15), bluish=(1.0, 0.3)):
    """Eq. 3: greenish if mean(G) > mean(B), otherwise bluish (Algorithm 1's
    else branch, which also covers a tie). Returns the tone and its (alpha, beta)."""
    if float(raw[..., 1].mean()) > float(raw[..., 2].mean()):
        return ToneDecision("greenish", float(greenish[0]), float(greenish[1]))
    return ToneDecision("bluish", float(bluish[0]), float(bluish[1]))
