"""Luminance compensation, LC (Algorithm 1, line 2).

CDEF's "simple light enhancement method" [36] is BIMEF: Ying, Li, Gao, "A
Bio-Inspired Multi-Exposure Fusion Framework for Low-light Image Enhancement",
arXiv:1711.00591. Equation numbers below refer to that paper.

BIMEF fuses the image with a brighter synthetic exposure of itself:

    R = W * P + (1 - W) * g(P, k),   W = T^mu            (Eqs. 4-5)

T is a smoothed illumination map (how well lit each pixel is), g is a camera
response model that simulates exposure ratio k, and k is chosen to maximise
the entropy of the under-exposed pixels. Implementation details (line-shaped
smoothness weights, wrap-around borders, approximate solve, antialiased
bicubic resizing) follow BIMEF's released code, which the CDEF authors use.
"""
import math

import cv2
import numpy as np
from scipy import sparse
from scipy.optimize import minimize_scalar
from scipy.sparse.linalg import spsolve_triangular

from .resize import resize_cubic, resize_mask


# ------------------------------------------------------- illumination map T
def smoothness_weights(l, length=5, eps=0.001):
    """Eqs. 8-10: edge weights 1 / (|gradient summed along a line of `length`
    pixels| * |gradient| + eps), with forward differences that wrap around."""
    ones = np.ones(length, np.float64)
    one = np.ones(1, np.float64)
    gx = np.roll(l, -1, axis=1) - l
    gy = np.roll(l, -1, axis=0) - l
    sx = cv2.sepFilter2D(gx, -1, ones, one, borderType=cv2.BORDER_CONSTANT)
    sy = cv2.sepFilter2D(gy, -1, one, ones, borderType=cv2.BORDER_CONSTANT)
    return 1.0 / (np.abs(sx) * np.abs(gx) + eps), 1.0 / (np.abs(sy) * np.abs(gy) + eps)


def smoothing_system(wx, wy, lam):
    """The sparse matrix I + lam * sum over edges of w (e_p - e_q)(e_p - e_q)^T.

    wx[i, j] joins pixel (i, j) to (i, j + 1) and wy[i, j] joins it to
    (i + 1, j); indices wrap around the border.
    """
    h, w = wx.shape
    n = h * w
    idx = np.arange(n).reshape(h, w)
    p = np.concatenate([idx.ravel(), idx.ravel()])
    q = np.concatenate([np.roll(idx, -1, axis=1).ravel(), np.roll(idx, -1, axis=0).ravel()])
    wt = lam * np.concatenate([wx.ravel(), wy.ravel()])
    off = sparse.coo_matrix((-wt, (p, q)), shape=(n, n))
    diag = np.bincount(p, wt, n) + np.bincount(q, wt, n)
    return (off + off.T + sparse.diags(1.0 + diag)).tocsc()


def incomplete_cholesky(A):
    """Modified incomplete Cholesky: A ~ L L^T with L on the pattern of
    tril(A). Each dropped fill value is moved onto the two affected diagonal
    entries, so L L^T keeps the row sums of A. Returns L as CSR."""
    low = sparse.tril(A, format="csc")
    n = low.shape[0]
    columns = []
    for k in range(n):
        start, end = low.indptr[k], low.indptr[k + 1]
        columns.append(dict(zip(low.indices[start:end].tolist(), low.data[start:end].tolist())))
    diag = [columns[k][k] for k in range(n)]
    for k in range(n):
        column = columns[k]
        lkk = math.sqrt(diag[k])
        below = sorted((i, v / lkk) for i, v in column.items() if i > k)
        column.clear()
        column[k] = lkk
        column.update(below)
        for a_pos, (j, ljk) in enumerate(below):
            target = columns[j]
            diag[j] -= ljk * ljk
            for i, lik in below[a_pos + 1:]:
                v = lik * ljk
                if i in target:
                    target[i] -= v
                else:
                    diag[i] -= v
                    diag[j] -= v
    rows, cols, vals = [], [], []
    for k, column in enumerate(columns):
        for i, v in column.items():
            rows.append(i)
            cols.append(k)
            vals.append(v)
    return sparse.csr_matrix((vals, (rows, cols)), shape=(n, n))


def conjugate_gradients(A, b, L, tol=0.1, max_iter=50):
    """Preconditioned conjugate gradients from x = 0, preconditioner (L L^T)^-1.

    Stops at relative residual `tol`; if that is never reached, returns the
    iterate with the smallest residual.
    """
    upper = L.T.tocsr()

    def precondition(r):
        return spsolve_triangular(upper, spsolve_triangular(L, r, lower=True), lower=False)

    x = np.zeros_like(b)
    r = b.copy()
    norm_b = np.linalg.norm(b)
    best, best_res = x.copy(), 1.0
    z = precondition(r)
    p = z.copy()
    rz = r @ z
    for _ in range(max_iter):
        q = A @ p
        alpha = rz / (p @ q)
        x = x + alpha * p
        r = r - alpha * q
        res = np.linalg.norm(r) / norm_b
        if res < best_res:
            best, best_res = x.copy(), res
        if res <= tol:
            return x
        z = precondition(r)
        rz_new = r @ z
        p = z + (rz_new / rz) * p
        rz = rz_new
    return best


def illumination_map(img, lam=0.5, eps=0.001, window=5, solve_scale=0.5, pcg_tol=0.1):
    """Eqs. 6-10: refined illumination map T.

    Start from L = max over channels (Eq. 6) and solve the edge-preserving
    smoothing problem (Eq. 10) on an image scaled by solve_scale, then resize
    the result back.
    """
    h, w = img.shape[:2]
    small = resize_cubic(img.max(axis=2), scale=solve_scale)
    wx, wy = smoothness_weights(small, window, eps)
    # Unknowns are ordered column by column, so the system is built on the
    # transposed image: the x-edges then run along axis 0.
    A = smoothing_system(wy.T.copy(), wx.T.copy(), lam)
    x = conjugate_gradients(A, small.T.ravel(), incomplete_cholesky(A), pcg_tol)
    return resize_cubic(x.reshape(small.T.shape).T, h, w)


# ----------------------------------------------------- exposure ratio k
def brightness_transform(img, k, a=-0.3293, b=1.1258):
    """Eq. 16, the camera response model: g(P, k) = exp(b(1 - k^a)) * P^(k^a)."""
    gamma = k**a
    beta = np.exp(b * (1.0 - gamma))
    return beta * np.power(np.maximum(img, 0.0), gamma)


def entropy_8bit(values):
    """Eq. 20: Shannon entropy in bits of values in [0, 1], quantised to 8 bits."""
    levels = np.round(np.clip(values, 0.0, 1.0) * 255.0).astype(np.int64)
    hist = np.bincount(levels.ravel(), minlength=256).astype(np.float64)
    p = hist[hist > 0] / hist.sum()
    return float(-(p * np.log2(p)).sum())


def exposure_ratio(img, t, threshold=0.5, a=-0.3293, b=1.1258, k_range=(1.0, 7.0), search_size=50):
    """Eqs. 17-21: the k in k_range that maximises the entropy of the
    under-exposed pixels (T < threshold), searched on a search_size^2 copy.
    Returns 1 (no change) when no pixel is under-exposed."""
    small = np.maximum(resize_cubic(img, search_size, search_size), 0.0)
    under = resize_mask(t < threshold, search_size, search_size)
    brightness = np.cbrt(small.prod(axis=2))[under]  # Eq. 18
    if brightness.size == 0:
        return 1.0

    def negative_entropy(k):
        return -entropy_8bit(brightness_transform(brightness, k, a, b))

    result = minimize_scalar(negative_entropy, bounds=tuple(k_range), method="bounded", options={"xatol": 1e-4})
    return float(result.x)


# ------------------------------------------------------------- the stage
def luminance_compensation(
    img,
    mu=0.5,
    lam=0.5,
    eps=0.001,
    window=5,
    solve_scale=0.5,
    pcg_tol=0.1,
    threshold=0.5,
    a=-0.3293,
    b=1.1258,
    k_range=(1.0, 7.0),
    search_size=50,
):
    """Eqs. 4-5: R = W * P + (1 - W) * g(P, k) with W = T^mu, clipped to [0, 1].

    Returns (R as float32, k).
    """
    img = img.astype(np.float64)
    t = illumination_map(img, lam, eps, window, solve_scale, pcg_tol)
    k = exposure_ratio(img, t, threshold, a, b, k_range, search_size)
    w = np.power(np.maximum(t, 0.0), mu)[..., None]
    exposed = brightness_transform(img, k, a, b) if k != 1.0 else img
    out = np.clip(w * img + (1.0 - w) * exposed, 0.0, 1.0)
    return out.astype(np.float32), k
