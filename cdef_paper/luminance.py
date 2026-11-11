"""Luminance compensation, LC (Algorithm 1, line 2).

CDEF's "simple light enhancement method" [36] is BIMEF: Ying, Li, Gao, "A
Bio-Inspired Multi-Exposure Fusion Framework for Low-light Image Enhancement",
arXiv:1711.00591. Equation numbers below refer to that paper, and every value
is the one it prints (Sec. V: lambda = 1, eps = 0.001, mu = 1/2, window 5).

The image P is fused with a brighter synthetic exposure g(P, k) of itself:

    R = W * P + (1 - W) * g(P, k),   W = T^mu                    (Eqs. 4-5)

T is a smoothed illumination map, so well-lit pixels keep their value and dark
ones take the brighter exposure.
"""
import cv2
import numpy as np
from scipy import sparse
from scipy.optimize import minimize_scalar
from scipy.sparse.linalg import splu


def forward_gradients(l):
    """Forward differences along x and y, zero on the last column and row."""
    gx = np.zeros_like(l)
    gy = np.zeros_like(l)
    gx[:, :-1] = l[:, 1:] - l[:, :-1]
    gy[:-1, :] = l[1:, :] - l[:-1, :]
    return gx, gy


def smoothness_weights(l, eps=0.001, window=5):
    """Eqs. 8 and 10: m_d / (|grad_d L| + eps), with
    m_d = 1 / (|sum of grad_d L over the window| + eps)."""
    weights = []
    for g in forward_gradients(l):
        window_sum = cv2.boxFilter(g, -1, (window, window), normalize=False, borderType=cv2.BORDER_REPLICATE)
        weights.append(1.0 / ((np.abs(window_sum) + eps) * (np.abs(g) + eps)))
    return weights


def illumination_system(l, lam=1.0, eps=0.001, window=5):
    """The matrix of Eq. 10: I + lam * sum_d D_d^T Diag(w_d) D_d."""
    h, w = l.shape
    wx, wy = smoothness_weights(l, eps, window)
    wx[:, -1] = 0.0  # no x edge leaves the last column
    wy[-1, :] = 0.0  # no y edge leaves the last row

    # Each edge joins two pixels, so it adds to both of their diagonal entries.
    diagonal = wx + wy
    diagonal[:, 1:] += wx[:, :-1]
    diagonal[1:, :] += wy[:-1, :]

    off_x = -lam * wx.ravel()[:-1]
    off_y = -lam * wy.ravel()[:-w]
    return sparse.diags(
        [1.0 + lam * diagonal.ravel(), off_x, off_x, off_y, off_y],
        [0, 1, -1, w, -w],
        shape=(h * w, h * w),
        format="csc",
    )


def solve_system(system, rhs):
    """Solve Eq. 10 exactly, by sparse factorisation. (The paper uses a
    preconditioned conjugate-gradient solver for speed; the solution of the
    system is the same.) The ordering suits a symmetric matrix."""
    return splu(system, permc_spec="MMD_AT_PLUS_A", options={"SymmetricMode": True}).solve(rhs)


def illumination_map(img, lam=1.0, eps=0.001, window=5):
    """T: the largest of R, G, B at each pixel (Eq. 6), smoothed where the
    image has no structure (Eqs. 7-10). Solved at full resolution."""
    l = img.max(axis=2).astype(np.float64)
    t = solve_system(illumination_system(l, lam, eps, window), l.ravel())
    return np.clip(t, 0.0, 1.0).reshape(l.shape)


def brightness_transform(img, k, a=-0.3293, b=1.1258):
    """Eq. 16: g(P, k) = exp(b (1 - k^a)) * P^(k^a)."""
    gamma = k**a
    return np.exp(b * (1.0 - gamma)) * np.power(np.maximum(img, 0.0), gamma)


def entropy(values, bins=256):
    """Eq. 20: Shannon entropy in bits of values in [0, 1], 256 bins."""
    hist, _ = np.histogram(np.clip(values, 0.0, 1.0), bins=bins, range=(0.0, 1.0))
    p = hist[hist > 0] / hist.sum()
    return float(-(p * np.log2(p)).sum())


def exposure_ratio(img, t, threshold=0.5, a=-0.3293, b=1.1258, k_range=(1.0, 7.0), search_size=50):
    """Eqs. 17-21: the k that gives the under-exposed pixels (T < threshold)
    the most entropy, searched on a 50 x 50 copy (Sec. IV-C).

    The paper gives no range for k and does not cover an image with no
    under-exposed pixel; here k is searched in k_range, and such an image
    keeps k = 1 (no change).
    """
    size = (search_size, search_size)
    small = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    small_t = cv2.resize(t, size, interpolation=cv2.INTER_AREA)
    pixels = small[small_t < threshold]
    if pixels.size == 0:
        return 1.0
    brightness = np.cbrt(np.prod(pixels, axis=1))  # Eq. 18

    def negative_entropy(k):
        return -entropy(brightness_transform(brightness, k, a, b))

    return float(minimize_scalar(negative_entropy, bounds=tuple(k_range), method="bounded").x)


def luminance_compensation(img, mu=0.5, lam=1.0, eps=0.001, window=5, threshold=0.5,
                           a=-0.3293, b=1.1258, k_range=(1.0, 7.0), search_size=50):
    """Eqs. 4-5. Returns (image in [0, 1] as float32, k)."""
    img = img.astype(np.float64)
    t = illumination_map(img, lam, eps, window)
    k = exposure_ratio(img, t, threshold, a, b, k_range, search_size)
    if k == 1.0:
        return img.astype(np.float32), k
    weight = np.power(t, mu)[..., None]
    fused = weight * img + (1.0 - weight) * brightness_transform(img, k, a, b)
    return np.clip(fused, 0.0, 1.0).astype(np.float32), k
