"""F3: Contour Bougie Morphology (CBM).

Yuan, Cao, Cai, Su, "An Underwater Image Vision Enhancement Algorithm Based on
Contour Bougie Morphology", IEEE TGRS 59(10), 2021. Equation numbers below
refer to that paper.

A top-hat is the difference between an image and its morphological opening
(white top-hat: small bright details) or closing (black top-hat: small dark
details). CBM collects these details over repeated filtering with hollow
square ("contour") structuring elements and adds them back to the image,
which sharpens object contours.
"""
import cv2
import numpy as np


def contour_element(size):
    """Fig. 4(b): a flat size x size square that keeps only its outer ring."""
    element = np.ones((size, size), np.uint8)
    element[1:-1, 1:-1] = 0
    return element


def opening(img, element):
    """Eq. 5: Mo(f, b) = (f eroded by b) dilated by b."""
    return cv2.dilate(cv2.erode(img, element), element)


def closing(img, element):
    """Eq. 5: Mc(f, b) = (f dilated by b) eroded by b."""
    return cv2.erode(cv2.dilate(img, element), element)


def top_hats(img, b1_size=15, b2_size=5, iterations=10):
    """Yield (Wh_i, Bh_i) for i = 1..iterations (Eqs. 7-8).

    The paper does not define Wh_i for i > 1 beyond "transformed n times";
    here each filter is applied to its own previous output, so
    Wh_i = I - Mc(Mo(S_(i-1), b1), b2), and likewise for Bh_i.
    """
    b1, b2 = contour_element(b1_size), contour_element(b2_size)
    white, black = img, img
    for _ in range(iterations):
        white = closing(opening(white, b1), b2)
        black = opening(closing(black, b1), b2)
        yield img - white, black - img


def cbm(img, omega=0.1, b1_size=15, b2_size=5, iterations=10, clip=True):
    """Eq. 11: I + omega * (sum of (Wh_i - Bh_i) + sum of the differences
    between consecutive iterations, Eqs. 9-10).

    The paper does not say what happens to values pushed outside the image
    range. clip=True keeps F3 a displayable image in [0, 1].
    """
    detail = np.zeros_like(img)
    previous = None
    for wh, bh in top_hats(img, b1_size, b2_size, iterations):
        detail += wh - bh
        if previous is not None:
            detail += (wh - previous[0]) - (bh - previous[1])  # Eqs. 9-10
        previous = (wh, bh)
    out = img + omega * detail
    if clip:
        out = np.clip(out, 0.0, 1.0)
    return out.astype(np.float32)
