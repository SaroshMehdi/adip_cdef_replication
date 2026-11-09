"""Multiscale exposure fusion (Eqs. 7-14).

Each input gets a per-pixel weight from four measures (contrast, saturation,
saliency, exposure). The weights are normalised to sum to 1, and the inputs are
blended level by level: Laplacian pyramids of the images, Gaussian pyramids of
the weights. Blending across scales avoids the seams and halos a direct
weighted sum would give.

The fusion runs in YUV (the paper). RGB and Lab are the alternatives in the
paper's ablation (Table IV); it gives no weight definitions for them, so RGB
uses Mertens et al.'s and Lab (scaled by 1/100) uses the YUV ones.
"""
import math

import cv2
import numpy as np

from .color_spaces import FUSION_SPACES, LUMA

# Eq. 7 prints [0 1 0; -1 4 -1; 0 -1 0], which is not a Laplacian (typo).
LAPLACIAN = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], np.float32)

# 5-tap binomial filter; along both axes it is the 5 x 5 kernel of Eq. 13.
KERNEL_1D = np.array([1, 4, 6, 4, 1], np.float32) / 16.0


# ------------------------------------------------------------- weights
def contrast_weight(y):
    """Eq. 7: Wc = |Laplacian of Y|."""
    return np.abs(cv2.filter2D(y, -1, LAPLACIAN, borderType=cv2.BORDER_REPLICATE))


def saturation_weight(u, v):
    """Eq. 8: Ws = |U| + |V| + 1."""
    return np.abs(u) + np.abs(v) + 1.0


def saliency_weight(img, blur_sigma=3.0, blur_window=3):
    """Eq. 9: squared distance of each pixel from the channel means, after a
    small Gaussian blur (Achanta et al.; a 3 x 3 window of sigma 3 as in the
    authors' code)."""
    smoothed = cv2.GaussianBlur(img, (blur_window, blur_window), blur_sigma, borderType=cv2.BORDER_REFLECT)
    return ((smoothed - smoothed.mean(axis=(0, 1))) ** 2).sum(axis=2)


def exposure_weight(y, phi=0.5, sigma=0.2):
    """Eq. 10: We = exp(-(Y - phi)^2 / (2 sigma^2)).

    Eq. 10 prints phi = 2 and sigma = 0.5, but the text says a well-exposed
    pixel is near 0.5; phi = 0.5, sigma = 0.2 are Mertens et al.'s values.
    """
    return np.exp(-((y - phi) ** 2) / (2.0 * sigma * sigma))


def combined_weight(img, space="yuv", phi=0.5, sigma=0.2, exponents=(2.0, 1.0, 2.0, 1.0),
                    saliency_blur=3.0, saliency_window=3):
    """The product inside Eq. 11: Wc^e1 * Ws^e2 * Wsal^e3 * We^e4.

    The paper sets every exponent to 1; the authors' code squares contrast and
    saliency. For RGB the measures are Mertens et al.'s: contrast on the grey
    image, saturation as the spread of R, G, B, and exposure as the product
    over the three channels.
    """
    if space == "rgb":
        w_c = contrast_weight(img @ LUMA)
        w_s = img.std(axis=2)
        w_e = np.prod([exposure_weight(img[..., c], phi, sigma) for c in range(3)], axis=0)
    else:  # a luma channel and two chroma channels
        w_c = contrast_weight(img[..., 0])
        w_s = saturation_weight(img[..., 1], img[..., 2])
        w_e = exposure_weight(img[..., 0], phi, sigma)
    w_sal = saliency_weight(img, saliency_blur, saliency_window)
    e_c, e_s, e_sal, e_e = exponents
    return (w_c**e_c) * (w_s**e_s) * (w_sal**e_sal) * (w_e**e_e)


def normalize_weights(weights, eps=1e-12):
    """Eq. 11: (W_k + eps) / sum_j (W_j + eps), so the weights sum to 1 at every
    pixel. (The printed form, (W_k + eps) / (sum_j W_j + eps), does not.)"""
    stack = np.stack(weights, axis=0) + eps
    return list(stack / stack.sum(axis=0))


def luma(img, space):
    """The brightness channel: Y or L, or the BT.601 grey of RGB."""
    return img @ LUMA if space == "rgb" else img[..., 0]


def brightness_weights(weights, inputs, space):
    """Scale each input's weight map by its brightness B = mean(Y)^2, then
    normalise again (Wang et al. [33], whose fusion the authors' code uses)."""
    scaled = [w * float(luma(img, space).mean()) ** 2 for w, img in zip(weights, inputs)]
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
def multiscale_fusion(images, space="yuv", phi=0.5, sigma=0.2, eps=1e-12, exponents=(2.0, 1.0, 2.0, 1.0),
                      saliency_blur=3.0, saliency_window=3):
    """Fuse RGB images in [0, 1]. Returns (fused RGB in [0, 1], weight maps)."""
    if space not in FUSION_SPACES:
        raise ValueError(f"unknown fusion colour space: {space}")
    forward, backward = FUSION_SPACES[space]
    inputs = [forward(img).astype(np.float32) for img in images]
    weights = [
        combined_weight(img.astype(np.float64), space, phi, sigma, exponents, saliency_blur, saliency_window)
        for img in inputs
    ]
    weights = brightness_weights(normalize_weights(weights, eps), inputs, space)
    weights = [w.astype(np.float32) for w in weights]
    fused = collapse(blend_pyramids(inputs, weights, pyramid_levels(*images[0].shape[:2])))
    return np.clip(backward(fused), 0.0, 1.0), weights
