"""Contour Bougie Morphology.

Yuan, Cao, Cai, Su, "An Underwater Image Vision Enhancement Algorithm Based on
Contour Bougie Morphology", IEEE TGRS 59(10), 2021. Equation numbers in this
file refer to that paper.
"""
import cv2
import numpy as np

from ..tone_mapping import gray_world_tone_map


def contour_element(size):
    """Fig. 4(b): flat square structuring element keeping only its outer ring."""
    element = np.ones((size, size), np.uint8)
    element[1:-1, 1:-1] = 0
    return element


def opening(img, element):
    """Eq. 5: Mo(f, b) = (f erode b) dilate b."""
    return cv2.dilate(cv2.erode(img, element), element)


def closing(img, element):
    """Eq. 5: Mc(f, b) = (f dilate b) erode b."""
    return cv2.erode(cv2.dilate(img, element), element)


def open_close(img, b1, b2):
    """The filter inside Eq. 7: Mc(Mo(I, b1), b2)."""
    return closing(opening(img, b1), b2)


def close_open(img, b1, b2):
    """The filter inside Eq. 8: Mo(Mc(I, b1), b2)."""
    return opening(closing(img, b1), b2)


def top_hat_series(img, b1_size=15, b2_size=5, iterations=10, mode="repeat"):
    """Yield (Wh_i, Bh_i) for i = 1..n (Eqs. 7-8).

    The paper does not define Wh_i for i > 1:
    "repeat" feeds each filter its own output, so Wh_i = I - S_i;
    "scale" grows both elements by 2 per iteration, as in multi-scale top-hats.
    """
    if mode not in ("repeat", "scale"):
        raise ValueError(f"unknown CBM iteration mode: {mode}")
    b1, b2 = contour_element(b1_size), contour_element(b2_size)
    white, black = img, img
    for i in range(iterations):
        if mode == "repeat":
            white = open_close(white, b1, b2)
            black = close_open(black, b1, b2)
        else:
            b1 = contour_element(b1_size + 2 * i)
            b2 = contour_element(b2_size + 2 * i)
            white = open_close(img, b1, b2)
            black = close_open(img, b1, b2)
        yield img - white, black - img


def cbm(
    img, omega=0.1, b1_size=15, b2_size=5, iterations=10, mode="repeat", clip=False, acs=False, acs_lambda=0.008
):
    """Eq. 11: add the white and subtract the black top-hat details to I.

    clip=False leaves the edge overshoot below 0 and above 1 in place, so the
    fusion sees it; the output is clipped only after fusion. Neither paper says
    whether to clip here.

    With acs=True the result also goes through the paper's Adaptive Contrast
    Stretch (Eqs. 12-14). Eq. 14 uses the channel sum, three times the gray
    mean, so ACS is the Eq. 5 stretch of CDEF with b = 3 * lambda.
    """
    detail = np.zeros_like(img)
    previous = None
    for wh, bh in top_hat_series(img, b1_size, b2_size, iterations, mode):
        detail += wh - bh
        if previous is not None:
            detail += (wh - previous[0]) - (bh - previous[1])  # Eqs. 9-10
        previous = (wh, bh)
    enhanced = img + omega * detail
    if clip or acs:
        enhanced = np.clip(enhanced, 0.0, 1.0)
    if acs:
        enhanced = gray_world_tone_map(enhanced, 3.0 * acs_lambda)
    return enhanced.astype(np.float32)
