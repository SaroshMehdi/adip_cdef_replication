"""Contrast stretching for fusion inputs F1 and F2 (Eqs. 4-6)."""
import numpy as np


def statistic_range(channel, a=2.0):
    """Eq. 4: mu -/+ a * sigma. The paper says "variance"; sigma is the std."""
    mean, std = float(channel.mean()), float(channel.std())
    return mean - a * std, mean + a * std


def gray_world_range(channel, gray_mean, b=0.01):
    """Eq. 5: quantile limits at b * gray_mean / mean_c and 1 minus that."""
    level = np.clip(b * gray_mean / max(float(channel.mean()), 1e-6), 0.0, 1.0)
    low, high = np.quantile(channel, [level, 1.0 - level])
    return float(low), float(high)


def stretch(channel, low, high, out_min=0.0, out_max=1.0):
    """Eq. 6 with the result clipped. A channel with no usable range is kept."""
    if high <= low:
        return channel.copy()
    scaled = (channel - low) * (out_max - out_min) / (high - low) + out_min
    return np.clip(scaled, out_min, out_max).astype(np.float32)


def statistic_tone_map(img, a=2.0):
    """F1: Eq. 6 with the Eq. 4 range, per channel."""
    channels = [img[..., c] for c in range(3)]
    return np.dstack([stretch(ch, *statistic_range(ch, a)) for ch in channels])


def gray_world_tone_map(img, b=0.01):
    """F2: Eq. 6 with the Eq. 5 range, per channel."""
    gray_mean = float(img.mean())
    channels = [img[..., c] for c in range(3)]
    return np.dstack([stretch(ch, *gray_world_range(ch, gray_mean, b)) for ch in channels])
