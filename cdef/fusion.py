"""Multiscale exposure fusion in YUV (Eqs. 7-14).

Options also cover the fusion of Wang et al. [33] that the authors' code
uses: a per-image brightness measure, fewer pyramid levels and a separate rule
at the coarsest level (see multiscale_fusion).
"""
import math

import cv2
import numpy as np

from .color_comp import lab2rgb, rgb2lab
from .utils import gaussian_blur

# BT.601 RGB to YUV.
RGB_TO_YUV = np.array(
    [
        [0.299, 0.587, 0.114],
        [-0.14713, -0.28886, 0.436],
        [0.615, -0.51499, -0.10001],
    ],
    dtype=np.float32,
)
YUV_TO_RGB = np.linalg.inv(RGB_TO_YUV).astype(np.float32)

# BT.601 RGB to YCbCr for [0, 1] data, with the 16..235 / 16..240 offsets.
RGB_TO_YCBCR = (
    np.array(
        [[65.481, 128.553, 24.966], [-37.797, -74.203, 112.0], [112.0, -93.786, -18.214]], np.float64
    )
    / 255.0
)
YCBCR_TO_RGB = np.linalg.inv(RGB_TO_YCBCR)
YCBCR_OFFSET = np.array([16.0, 128.0, 128.0]) / 255.0

LAPLACIAN = {
    "standard": np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], np.float32),
    "printed": np.array([[0, 1, 0], [-1, 4, -1], [0, -1, 0]], np.float32),  # Eq. 7 as printed
}

# 5-tap binomial filter; applied along both axes it is the 5 x 5 kernel of Eq. 13.
KERNEL_1D = np.array([1, 4, 6, 4, 1], np.float32) / 16.0


def rgb_to_yuv(img):
    return img @ RGB_TO_YUV.T


def yuv_to_rgb(img):
    return img @ YUV_TO_RGB.T


def rgb_to_ycbcr(img):
    return (img @ RGB_TO_YCBCR.T + YCBCR_OFFSET).astype(np.float32)


def ycbcr_to_rgb(img):
    return (img - YCBCR_OFFSET) @ YCBCR_TO_RGB.T


def rgb_to_lab_unit(img):
    """CIELab scaled by 1/100: L in [0, 1], a and b roughly in [-1, 1]."""
    return (rgb2lab(img) / 100.0).astype(np.float32)


def lab_unit_to_rgb(img):
    return lab2rgb(np.asarray(img, np.float64) * 100.0)


def identity(img):
    return np.array(img, np.float32, copy=True)


# Colour spaces for the fusion (Table IV's ablation compares YUV, RGB and Lab).
COLOUR_SPACES = {
    "yuv": (rgb_to_yuv, yuv_to_rgb),
    "ycbcr": (rgb_to_ycbcr, ycbcr_to_rgb),
    "lab": (rgb_to_lab_unit, lab_unit_to_rgb),
    "rgb": (identity, identity),
}
LUMA = np.array([0.299, 0.587, 0.114])


def luma(img, colour="yuv"):
    """The brightness channel the weights use: Y or L, or the BT.601 gray of RGB."""
    return img @ LUMA if colour == "rgb" else img[..., 0]


def contrast_weight(y, kernel="standard"):
    """Eq. 7: Wc = |Laplacian of Y|."""
    return np.abs(cv2.filter2D(y, -1, LAPLACIAN[kernel], borderType=cv2.BORDER_REPLICATE))


def saturation_weight(u, v):
    """Eq. 8: Ws = |U| + |V| + 1."""
    return np.abs(u) + np.abs(v) + 1.0


def saliency_weight(yuv, blur_sigma=0.0, blur_window=0):
    """Eq. 9: squared distance from the channel means.

    Achanta et al. blur the image first; blur_sigma > 0 does the same. Eq. 9
    prints no blur. blur_window > 0 truncates the Gaussian to that window, with
    mirrored borders, as the authors' code does (3 x 3, sigma 3).
    """
    if blur_sigma > 0 and blur_window > 0:
        smoothed = cv2.GaussianBlur(
            yuv, (blur_window, blur_window), blur_sigma, borderType=cv2.BORDER_REFLECT
        )
    else:
        smoothed = gaussian_blur(yuv, blur_sigma)
    return ((smoothed - smoothed.mean(axis=(0, 1))) ** 2).sum(axis=2)


def exposure_weight(y, phi=2.0, sigma=0.5):
    """Eq. 10: We = exp(-(Y - phi)^2 / (2 sigma^2)), with Y in [0, 1]."""
    return np.exp(-((y - phi) ** 2) / (2.0 * sigma * sigma))


def combined_weight(
    yuv,
    phi=2.0,
    sigma=0.5,
    exponents=(1.0, 1.0, 1.0, 1.0),
    laplacian="standard",
    saliency_blur=0.0,
    weight_range=1.0,
    saliency_window=0,
    colour="yuv",
):
    """The product inside Eq. 11: Wc^wc * Ws^ws * Wsal^wsal * We^we.

    weight_range is the scale of the values the first three maps are measured
    on: 1 for [0, 1] data, 255 for 8-bit data. The paper does not say, and its
    regularisation term of 0.1 only matters relative to this scale. The
    exposure weight always uses Y in [0, 1].
    """
    scaled = yuv * weight_range
    if colour == "rgb":
        # The paper defines the weights for YUV only. In RGB we use Mertens et
        # al.'s definitions, the exposure fusion CDEF builds on: contrast on the
        # gray image, saturation as the spread across R, G and B, and
        # well-exposedness as the product over the three channels.
        w_c = contrast_weight(scaled @ LUMA, laplacian)
        w_s = scaled.std(axis=2)
        w_e = np.prod([exposure_weight(yuv[..., c], phi, sigma) for c in range(3)], axis=0)
    else:
        # YUV, YCbCr and Lab (scaled to about [0, 1] for L): luma and two chroma channels.
        w_c = contrast_weight(scaled[..., 0], laplacian)
        w_s = saturation_weight(scaled[..., 1], scaled[..., 2])
        w_e = exposure_weight(yuv[..., 0], phi, sigma)
    w_sal = saliency_weight(scaled, saliency_blur, saliency_window)
    wc, ws, wsal, we = exponents
    return (w_c**wc) * (w_s**ws) * (w_sal**wsal) * (w_e**we)


def normalize_weights(weights, eps=0.1, form="sum_to_one"):
    """Eq. 11.

    "sum_to_one": (W_k + eps) / sum_j (W_j + eps), so the weights sum to 1 as
    the text requires. "printed": (W_k + eps) / (sum_j W_j + eps), which does not.
    """
    stack = np.stack(weights, axis=0)
    if form == "sum_to_one":
        denominator = (stack + eps).sum(axis=0)
    elif form == "printed":
        denominator = stack.sum(axis=0) + eps
    else:
        raise ValueError(f"unknown weight normalisation: {form}")
    return list((stack + eps) / denominator)


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
    details = [
        gaussian[i] - upsample(gaussian[i + 1], gaussian[i].shape[:2])
        for i in range(levels - 1)
    ]
    return details + [gaussian[-1]]


def collapse(pyramid):
    """Rebuild an image from a Laplacian pyramid."""
    image = pyramid[-1]
    for level in reversed(pyramid[:-1]):
        image = level + upsample(image, level.shape[:2])
    return image


def blend_pyramids(inputs, weights, levels):
    """Eq. 14: R_n = sum_k G_n{W_k} * L_n{I_k}, one Laplacian pyramid out."""
    blended = [0.0] * levels
    for image, weight in zip(inputs, weights):
        image_pyramid = laplacian_pyramid(image, levels)
        weight_pyramid = gaussian_pyramid(weight, levels)
        for n in range(levels):
            blended[n] = blended[n] + weight_pyramid[n][..., None] * image_pyramid[n]
    return blended


def brightness_weights(weights, yuv_inputs, colour="yuv"):
    """Wang et al. [33] (from Kou et al. [31]): scale each normalised map by the
    image's brightness measure B = mean(Y)^2, then normalise again."""
    scaled = [w * float(luma(yuv, colour).mean()) ** 2 for w, yuv in zip(weights, yuv_inputs)]
    total = sum(scaled)
    return [w / total for w in scaled]


def gaussian_3x3(img):
    """Gaussian of sigma 1 on a 3 x 3 window, replicated borders."""
    kernel = cv2.getGaussianKernel(3, 1.0)
    return cv2.sepFilter2D(img, -1, kernel, kernel, borderType=cv2.BORDER_REPLICATE)


def coarse_level_luma(yuv_inputs, weights, levels, darker, alpha=1.5, eps=1e-12):
    """Wang et al. [33] Eq. 10 for the luma channel at the coarsest level.

    Each weight map's coarsest level is smoothed by a 3 x 3 Gaussian. For the
    images in `darker` (treated as under-exposed) the smoothed magnitude of the
    next finer detail of the image's coarsest level is added, times alpha.
    The weights are then normalised at this level.
    """
    top = levels - 1
    fused, total = 0.0, 0.0
    for i, (yuv, weight) in enumerate(zip(yuv_inputs, weights)):
        base = laplacian_pyramid(yuv[..., 0], levels)[top]
        w_top = gaussian_3x3(gaussian_pyramid(weight, levels)[top])
        if i in darker:
            detail = np.abs(base - upsample(downsample(base), base.shape))
            w_top = w_top + alpha * (detail > 0) * gaussian_3x3(detail)
        fused = fused + w_top * base
        total = total + w_top + eps
    return fused / total


def multiscale_fusion(
    images,
    phi=2.0,
    sigma=0.5,
    eps=0.1,
    exponents=(1.0, 1.0, 1.0, 1.0),
    laplacian="standard",
    saliency_blur=0.0,
    weight_range=1.0,
    normalization="sum_to_one",
    colour="yuv",
    saliency_window=0,
    brightness=False,
    drop_levels=0,
    coarse_rule=False,
    coarse_alpha=1.5,
):
    """Fuse RGB inputs (Eqs. 7-14). Returns (RGB in [0, 1], weight maps).

    The remaining options follow Wang et al. [33], whose fusion the authors'
    code uses: brightness=True adds the per-image measure B = mean(Y)^2;
    drop_levels=2 gives n = floor(log2 min(h, w)) - 2 levels (Wang Eq. 6);
    coarse_rule=True applies Wang Eq. 10 to luma at the coarsest level, with
    the darker half of the inputs (ceil(N / 2) - 1 of them, by mean value)
    treated as under-exposed. colour="ycbcr" fuses in offset YCbCr, as the
    authors' code does, instead of the paper's YUV. "rgb" and "lab" are the
    colour spaces of Table IV's ablation (see combined_weight for their weights).
    """
    if colour not in COLOUR_SPACES:
        raise ValueError(f"unknown colour space: {colour}")
    if coarse_rule and colour == "rgb":
        raise ValueError("coarse_rule needs a luma channel; RGB has none")
    forward, backward = COLOUR_SPACES[colour]
    yuv_inputs = [forward(img).astype(np.float32) for img in images]
    raw_weights = [
        combined_weight(
            yuv.astype(np.float64), phi, sigma, exponents, laplacian, saliency_blur, weight_range,
            saliency_window, colour,
        )
        for yuv in yuv_inputs
    ]
    weights = normalize_weights(raw_weights, eps, normalization)
    if brightness:
        weights = brightness_weights(weights, yuv_inputs, colour)
    weights = [w.astype(np.float32) for w in weights]
    levels = pyramid_levels(*images[0].shape[:2])
    if drop_levels:
        levels = max(2, levels - drop_levels)
    blended = blend_pyramids(yuv_inputs, weights, levels)
    if coarse_rule:
        order = np.argsort([float(np.mean(img)) for img in images])
        darker = set(order[: int(math.ceil(len(images) / 2)) - 1].tolist())
        top = np.array(blended[-1], copy=True)
        top[..., 0] = coarse_level_luma(yuv_inputs, weights, levels, darker, coarse_alpha, eps)
        blended[-1] = top
    fused = collapse(blended)
    return np.clip(backward(fused), 0.0, 1.0), weights
