"""Multiscale exposure fusion (Eqs. 7-14 of the CDEF paper).

Each input gets a per-pixel weight from four measures (contrast, saturation,
saliency, exposure). The weights are normalised to sum to 1, and the inputs are
blended level by level: Laplacian pyramids of the images, Gaussian pyramids of
the weights.

Everything is as printed (YUV, all exponents 1, eps = 0.1, Eq. 12's depth,
Eq. 14 at every level) except three equations whose printed form contradicts
the sentence next to them; those follow the sentence (see README.md):
Eq. 7's kernel, Eq. 10's constants and Eq. 11's normalisation.

RGB and Lab are the alternatives in the paper's ablation (Table IV). It gives
no weight definitions for them, so RGB uses Mertens et al.'s and Lab (scaled
by 1/100) uses the YUV ones.
"""
import math

import cv2
import numpy as np

from .color_spaces import FUSION_SPACES, LUMA

# Eq. 7 prints [0 1 0; -1 4 -1; 0 -1 0]. Its entries sum to 2, so it is not the
# Laplacian the text names; the sign of the top entry is taken as a misprint.
LAPLACIAN = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], np.float32)

# 5-tap binomial filter; along both axes it is the 5 x 5 Gaussian kernel of
# Eq. 13 (the paper gives no values; this is the standard pyramid kernel).
KERNEL_1D = np.array([1, 4, 6, 4, 1], np.float32) / 16.0


# ------------------------------------------------------------- weights
def contrast_weight(y):
    """Eq. 7: Wc = |Laplacian of Y|."""
    return np.abs(cv2.filter2D(y, -1, LAPLACIAN, borderType=cv2.BORDER_REPLICATE))


def saturation_weight(u, v):
    """Eq. 8: Ws = |U| + |V| + 1."""
    return np.abs(u) + np.abs(v) + 1.0


def saliency_weight(img, blur_sigma=None, blur_window=3):
    """Eq. 9: Wsal = (Y - mean Y)^2 + (U - mean U)^2 + (V - mean V)^2.

    blur_sigma (not in the paper) blurs the image first with a small Gaussian,
    as the authors' code and Achanta et al. do.
    """
    if blur_sigma:
        img = cv2.GaussianBlur(img, (blur_window, blur_window), blur_sigma, borderType=cv2.BORDER_REFLECT)
    return ((img - img.mean(axis=(0, 1))) ** 2).sum(axis=2)


def exposure_weight(y, phi=0.5, sigma=0.2):
    """Eq. 10: We = exp(-(Y - phi)^2 / (2 sigma^2)).

    The paper prints phi = 2, sigma = 0.5 but says "a well-exposed pixel tends
    to have a value of 0.5" and cites Mertens et al. [34], whose values are
    phi = 0.5, sigma = 0.2. Those are used.
    """
    return np.exp(-((y - phi) ** 2) / (2.0 * sigma * sigma))


def combined_weight(img, space="yuv", phi=0.5, sigma=0.2, exponents=(1.0, 1.0, 1.0, 1.0), value_range=1.0,
                    saliency_blur=None):
    """The product inside Eq. 11: Wc^wc * Ws^ws * Wsal^wsal * We^we.

    value_range is the scale the measures are taken on: 1 for images in
    [0, 1], 255 for 8-bit values (the range Eq. 6 maps to). It matters because
    Eq. 8 adds 1 and Eq. 11 adds eps = 0.1 to these measures. The exposure
    weight always uses Y in [0, 1], where "well exposed" is 0.5.
    """
    unit, img = img, img * value_range
    if space == "rgb":  # Mertens et al.'s measures
        w_c = contrast_weight(img @ LUMA)
        w_s = img.std(axis=2)
        w_e = np.prod([exposure_weight(unit[..., c], phi, sigma) for c in range(3)], axis=0)
    else:  # a luma channel and two chroma channels
        w_c = contrast_weight(img[..., 0])
        w_s = saturation_weight(img[..., 1], img[..., 2])
        w_e = exposure_weight(unit[..., 0], phi, sigma)
    w_sal = saliency_weight(img, saliency_blur)
    e_c, e_s, e_sal, e_e = exponents
    return (w_c**e_c) * (w_s**e_s) * (w_sal**e_sal) * (w_e**e_e)


def normalize_weights(weights, eps=0.1):
    """Eq. 11: (W_k + eps) / sum_j (W_j + eps).

    As printed, the divisor is (sum_j W_j + eps), which does not sum to 1 over
    the inputs; the text says the weights are normalised "to ensure the final
    weight in fusion is 1", so eps is added to every term of the sum.
    """
    stack = np.stack(weights, axis=0) + eps
    return list(stack / stack.sum(axis=0))


def brightness_weights(weights, inputs, space):
    """Not in the paper: scale each input's weight map by its brightness
    mean(Y)^2 and normalise again (Wang et al. [33], in the authors' code)."""
    means = [float((img @ LUMA if space == "rgb" else img[..., 0]).mean()) for img in inputs]
    scaled = [w * m * m for w, m in zip(weights, means)]
    total = sum(scaled)
    return [w / total for w in scaled]


# ------------------------------------------------------------ pyramids
def pyramid_levels(h, w):
    """Eq. 12: n = floor(log2(min(h, w))), at least 1."""
    return max(1, int(math.floor(math.log2(min(h, w)))))


def smooth(img):
    return cv2.sepFilter2D(img, -1, KERNEL_1D, KERNEL_1D, borderType=cv2.BORDER_REFLECT_101)


def downsample(img):
    """Down(G * kernel) in Eq. 13."""
    return smooth(img)[::2, ::2]


def upsample(img, shape):
    """Up(G) * kernel in Eq. 13: insert zeros, then smooth with gain 4."""
    h, w = shape
    up = np.zeros((h, w) + img.shape[2:], img.dtype)
    up[::2, ::2] = img[: (h + 1) // 2, : (w + 1) // 2]
    return smooth(up) * 4.0


def gaussian_pyramid(img, levels):
    """Eq. 13, first line."""
    pyramid = [img]
    for _ in range(levels - 1):
        pyramid.append(downsample(pyramid[-1]))
    return pyramid


def laplacian_pyramid(img, levels):
    """Eq. 13, second line. The coarsest level keeps the Gaussian residual."""
    gaussian = gaussian_pyramid(img, levels)
    details = [gaussian[i] - upsample(gaussian[i + 1], gaussian[i].shape[:2]) for i in range(levels - 1)]
    return details + [gaussian[-1]]


def collapse(pyramid):
    """Rebuild an image from a Laplacian pyramid."""
    image = pyramid[-1]
    for level in reversed(pyramid[:-1]):
        image = level + upsample(image, level.shape[:2])
    return image


def blend_pyramids(inputs, weights, levels):
    """Eq. 14: R_n = sum_k G_n{W_k} * L_n{I_k}."""
    blended = [0.0] * levels
    for image, weight in zip(inputs, weights):
        image_pyramid = laplacian_pyramid(image, levels)
        weight_pyramid = gaussian_pyramid(weight, levels)
        for n in range(levels):
            blended[n] = blended[n] + weight_pyramid[n][..., None] * image_pyramid[n]
    return blended


# -------------------------------------------------------------- fusion
def multiscale_fusion(images, space="yuv", phi=0.5, sigma=0.2, eps=0.1, exponents=(1.0, 1.0, 1.0, 1.0),
                      value_range=1.0, saliency_blur=None, brightness=False):
    """Fuse RGB images in [0, 1]. Returns (fused RGB in [0, 1], weight maps)."""
    if space not in FUSION_SPACES:
        raise ValueError(f"unknown fusion colour space: {space}")
    forward, backward = FUSION_SPACES[space]
    inputs = [forward(img).astype(np.float32) for img in images]
    weights = [combined_weight(img.astype(np.float64), space, phi, sigma, exponents, value_range, saliency_blur)
               for img in inputs]
    weights = normalize_weights(weights, eps)
    if brightness:
        weights = brightness_weights(weights, inputs, space)
    weights = [w.astype(np.float32) for w in weights]
    fused = collapse(blend_pyramids(inputs, weights, pyramid_levels(*images[0].shape[:2])))
    return np.clip(backward(fused), 0.0, 1.0), weights
