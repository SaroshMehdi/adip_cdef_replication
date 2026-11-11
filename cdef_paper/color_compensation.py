"""Colour compensation in CIELab (Algorithm 1, lines 3-11; Eqs. 1-2).

A colour cast shows up as a* and b* shifted away from zero. Following the 3C
prior (opponent channels average to zero), part of that shift is removed:

    I_a' = I_a - alpha * M * G(I_a),   I_b' = I_b - beta * M * G(I_b)   (Eq. 1)

G is a Gaussian blur and M masks out light sources (Eq. 2). alpha and beta
come from the tone test (Eq. 3): stronger for bluish images.
"""
import cv2
import numpy as np

from .color_spaces import lab_to_rgb, rgb_to_lab


def gaussian_blur(img, sigma):
    """Gaussian blur with replicated borders."""
    return cv2.GaussianBlur(img, (0, 0), float(sigma), borderType=cv2.BORDER_REPLICATE)


def highlight_mask(img, threshold=0.85):
    """Eq. 2: M = 0 where the mean of R, G, B is above the threshold, else 1."""
    return (img.mean(axis=2) <= threshold).astype(np.float32)


def color_compensation(img, alpha, beta, sigma=1.0, mask_threshold=0.85):
    """Eqs. 1-2 on a* and b* (in their -100..+100 range). Returns RGB in [0, 1].

    The paper gives no size for G; sigma = 1 pixel is the one value in this
    folder taken from the authors' code. Eq. 1 applies G to the channel and the
    text says G smooths the mask, so both are smoothed.
    """
    lab = rgb_to_lab(img).astype(np.float32)
    mask = gaussian_blur(highlight_mask(img, mask_threshold), sigma)
    lab[..., 1] = lab[..., 1] - alpha * mask * gaussian_blur(lab[..., 1], sigma)
    lab[..., 2] = lab[..., 2] - beta * mask * gaussian_blur(lab[..., 2], sigma)
    return np.clip(lab_to_rgb(lab), 0.0, 1.0).astype(np.float32)
