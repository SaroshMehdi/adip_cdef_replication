"""Antialiased bicubic resizing, used by the luminance stage (BIMEF).

BIMEF's released implementation resizes with Keys' cubic kernel, widened when
shrinking so that the result is antialiased, and with mirrored borders. OpenCV's
bicubic resize is neither, so the resampling is done here with sparse matrices:
one matrix per axis, applied as a matrix product.
"""
import math

import numpy as np
from scipy import sparse


def keys_cubic(x):
    """Keys' cubic convolution kernel with a = -0.5."""
    ax = np.abs(x)
    ax2, ax3 = ax * ax, ax * ax * ax
    return (1.5 * ax3 - 2.5 * ax2 + 1.0) * (ax <= 1) + (-0.5 * ax3 + 2.5 * ax2 - 4.0 * ax + 2.0) * (
        (ax > 1) & (ax <= 2)
    )


def resize_matrix(n_in, n_out, scale=None):
    """Sparse (n_out, n_in) matrix that resamples one axis.

    scale defaults to n_out / n_in. When shrinking, the kernel is widened by
    1 / scale (antialiasing). Borders are mirrored.
    """
    scale = n_out / n_in if scale is None else scale
    width = 4.0 / scale if scale < 1 else 4.0
    centres = np.arange(1, n_out + 1) / scale + 0.5 * (1.0 - 1.0 / scale)
    left = np.floor(centres - width / 2.0)
    taps = int(math.ceil(width)) + 2
    idx = left[:, None] + np.arange(taps)[None, :]
    distance = centres[:, None] - idx
    weights = scale * keys_cubic(scale * distance) if scale < 1 else keys_cubic(distance)
    weights = weights / weights.sum(axis=1, keepdims=True)
    mirror = np.concatenate([np.arange(n_in), np.arange(n_in)[::-1]])
    cols = mirror[np.mod(idx.astype(np.int64) - 1, 2 * n_in)]
    rows = np.repeat(np.arange(n_out), taps)
    # Duplicate (row, col) pairs created by the mirroring are summed.
    return sparse.csr_matrix((weights.ravel(), (rows, cols.ravel())), shape=(n_out, n_in))


def resample_axis(img, matrix, axis):
    """Apply a resize matrix along one axis of an array."""
    moved = np.moveaxis(img, axis, 0)
    flat = moved.reshape(moved.shape[0], -1)
    out = (matrix @ flat).reshape((matrix.shape[0],) + moved.shape[1:])
    return np.moveaxis(out, 0, axis)


def axis_passes(h, w, out_h, out_w, scale_h, scale_w):
    """The two 1-D passes, the axis that shrinks most first."""
    passes = [(0, h, out_h, scale_h), (1, w, out_w, scale_w)]
    if scale_w < scale_h:
        passes.reverse()
    return passes


def resize_cubic(img, out_h=None, out_w=None, scale=None):
    """Resize to (out_h, out_w), or by `scale` with the size rounded up."""
    img = np.asarray(img, np.float64)
    h, w = img.shape[:2]
    if scale is not None:
        out_h, out_w = int(math.ceil(h * scale)), int(math.ceil(w * scale))
        scale_h = scale_w = scale
    else:
        scale_h, scale_w = out_h / h, out_w / w
    for axis, n_in, n_out, s in axis_passes(h, w, out_h, out_w, scale_h, scale_w):
        img = resample_axis(img, resize_matrix(n_in, n_out, s), axis)
    return img


def resize_mask(mask, out_h, out_w):
    """Resize a binary mask the way an 8-bit image is resized: values 0 or
    255, rounded and saturated after each pass, then thresholded at the middle."""
    out = mask.astype(np.float64) * 255.0
    h, w = out.shape
    for axis, n_in, n_out, s in axis_passes(h, w, out_h, out_w, out_h / h, out_w / w):
        out = np.clip(np.round(resample_axis(out, resize_matrix(n_in, n_out, s), axis)), 0.0, 255.0)
    return out > 127
