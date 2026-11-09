"""Colour compensation in CIELab (Eqs. 1-2)."""
import numpy as np

from .utils import gaussian_blur

# sRGB primaries to CIE XYZ, and the D65 white point.
SRGB_TO_XYZ = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ]
)
XYZ_TO_SRGB = np.linalg.inv(SRGB_TO_XYZ)
WHITE_D65 = np.array([0.9504559, 1.0, 1.0890578])


def srgb_to_linear(c):
    a = np.abs(c)
    return np.sign(c) * np.where(a <= 0.04045, a / 12.92, ((a + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c):
    a = np.abs(c)
    return np.sign(c) * np.where(a <= 0.0031308, 12.92 * a, 1.055 * a ** (1 / 2.4) - 0.055)


def rgb2lab(rgb):
    """sRGB (D65) to CIELab. Values outside [0, 1] are converted, not clipped:
    the luminance stage can leave a few, and clipping them first shifts hue."""
    xyz = srgb_to_linear(np.asarray(rgb, np.float64)) @ SRGB_TO_XYZ.T / WHITE_D65
    f = np.where(xyz > 216 / 24389, np.cbrt(xyz), (24389 / 27 * xyz + 16) / 116)
    return np.stack(
        [116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], axis=-1
    )


def lab2rgb(lab):
    """CIELab to sRGB (D65), inverse of rgb2lab. Out-of-gamut colours are
    returned outside [0, 1] rather than clipped in XYZ, so the caller decides."""
    fy = (lab[..., 0] + 16) / 116
    f = np.stack([fy + lab[..., 1] / 500, fy, fy - lab[..., 2] / 200], axis=-1)
    xyz = np.where(f > 6 / 29, f**3, (116 * f - 16) * 27 / 24389) * WHITE_D65
    return linear_to_srgb(xyz @ XYZ_TO_SRGB.T)


def highlight_mask(img, threshold=0.85):
    """Eq. 2: 0 at light-source pixels (mean intensity > threshold), else 1."""
    return (img.mean(axis=2) <= threshold).astype(np.float32)


def compensate_channel(channel, weight, mask, sigma):
    """Eq. 1: I_c = I - weight * M * G(I)."""
    return channel - weight * mask * gaussian_blur(channel, sigma)


def color_compensation(
    img, alpha, beta, sigma=1.0, sigma_ratio=None, mask_threshold=0.85, smooth_mask=True, use_mask=True
):
    """Eqs. 1-2 applied to a* and b*. a* and b* keep the paper's -100..+100 range.

    The paper gives no Gaussian sigma. It must be small: Fig. 4 shows alpha =
    beta = 1 giving a nearly gray image, which needs G(a) close to a. sigma is
    in pixels; if sigma_ratio is set it replaces sigma with sigma_ratio *
    min(H, W). The text says G smooths the mask while Eq. 1 applies it to the
    channel; with smooth_mask both happen. use_mask=False sets M = 1 everywhere,
    which is what the authors' released code does in effect.
    """
    if sigma_ratio is not None:
        sigma = sigma_ratio * min(img.shape[:2])
    lab = rgb2lab(img).astype(np.float32)
    mask = highlight_mask(img, mask_threshold) if use_mask else np.ones(img.shape[:2], np.float32)
    if smooth_mask:
        mask = gaussian_blur(mask, sigma)
    lab[..., 1] = compensate_channel(lab[..., 1], alpha, mask, sigma)
    lab[..., 2] = compensate_channel(lab[..., 2], beta, mask, sigma)
    return np.clip(lab2rgb(lab), 0.0, 1.0).astype(np.float32)
