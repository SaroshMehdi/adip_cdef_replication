"""Contrast stretching for fusion inputs F1 and F2 (Algorithm 1, lines 12-14; Eqs. 4-6).

Both stretch each channel linearly so that a low limit maps to 0 and a high
limit to 1 (Eq. 6). F1 takes its limits from the channel statistics (Eq. 4),
F2 from gray-world-scaled quantiles (Eq. 5), which stretches channels darker
than the overall grey (usually red) more strongly.
"""
import numpy as np


def statistic_range(channel, a=2.0):
    """Eq. 4: mean -/+ a * std. (The paper says "variance"; only the standard
    deviation has the units of the mean.)"""
    mean, std = float(channel.mean()), float(channel.std())
    return mean - a * std, mean + a * std


def gray_world_range(channel, gray_mean, b=0.01):
    """Eq. 5: the quantiles at b * gray_mean / channel_mean and 1 minus that."""
    level = np.clip(b * gray_mean / max(float(channel.mean()), 1e-6), 0.0, 1.0)
    low, high = np.quantile(channel, [level, 1.0 - level])
    return float(low), float(high)


def stretch(channel, low, high):
    """Eq. 6 onto [0, 1], clipped. A channel with no usable range is kept as is."""
    if high <= low:
        return channel.copy()
    return np.clip((channel - low) / (high - low), 0.0, 1.0).astype(np.float32)


def statistic_tone_map(img, a=2.0):
    """F1: Eq. 6 with the Eq. 4 range, per channel."""
    return np.dstack([stretch(img[..., c], *statistic_range(img[..., c], a)) for c in range(3)])


def gray_world_tone_map(img, b=0.01):
    """F2: Eq. 6 with the Eq. 5 range, per channel."""
    gray_mean = float(img.mean())
    return np.dstack([stretch(img[..., c], *gray_world_range(img[..., c], gray_mean, b)) for c in range(3)])
