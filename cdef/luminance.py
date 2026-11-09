"""Luminance compensation, LC (Algorithm 1, line 2).

CDEF's "simple light enhancement method" [36] is BIMEF: Ying, Li, Gao, "A
Bio-Inspired Multi-Exposure Fusion Framework for Low-light Image Enhancement",
arXiv:1711.00591. Equation numbers in this file refer to that paper.
"""
import math

import cv2
import numpy as np
from scipy import sparse
from scipy.optimize import minimize_scalar
from scipy.sparse.linalg import spsolve, spsolve_triangular


def initial_illumination(img):
    """BIMEF Eq. 6: L(x) = max over c of P_c(x)."""
    return img.max(axis=2)


def forward_gradients(l):
    """Forward differences along x and y, zero on the last column / row."""
    gx = np.zeros_like(l)
    gy = np.zeros_like(l)
    gx[:, :-1] = l[:, 1:] - l[:, :-1]
    gy[:-1, :] = l[1:, :] - l[:-1, :]
    return gx, gy


def smoothness_weights(l, eps=0.001, window=5):
    """BIMEF Eqs. 8 and 10: m_d / (|grad_d l| + eps) for d in {x, y}."""
    gx, gy = forward_gradients(l)
    weights = []
    for g in (gx, gy):
        window_sum = cv2.boxFilter(
            g, -1, (window, window), normalize=False, borderType=cv2.BORDER_REPLICATE
        )
        m = 1.0 / (np.abs(window_sum) + eps)
        weights.append(m / (np.abs(g) + eps))
    return weights[0], weights[1]


def solve_illumination(l, lam=1.0, eps=0.001, window=5):
    """BIMEF Eq. 10: solve (I + lam * sum_d D_d^T Diag(w_d) D_d) t = l."""
    h, w = l.shape
    wx, wy = smoothness_weights(l.astype(np.float64), eps, window)
    wx[:, -1] = 0.0  # no x edge leaves the last column
    wy[-1, :] = 0.0  # no y edge leaves the last row

    # Each edge couples its two pixels, so it adds to both diagonal entries.
    main = wx + wy
    main[:, 1:] += wx[:, :-1]
    main[1:, :] += wy[:-1, :]

    n = h * w
    off_x = -lam * wx.ravel()[:-1]
    off_y = -lam * wy.ravel()[:-w] if h > 1 else np.zeros(0)
    system = sparse.diags(
        [1.0 + lam * main.ravel(), off_x, off_x, off_y, off_y],
        [0, 1, -1, w, -w],
        shape=(n, n),
        format="csc",
    )
    return np.clip(spsolve(system, l.ravel().astype(np.float64)), 0.0, 1.0).reshape(h, w)


def estimate_illumination(img, lam=1.0, eps=0.001, window=5, solve_scale=0.5):
    """Refined illumination map T (BIMEF Eqs. 6-10).

    The paper solves at full resolution. solve_scale < 1 solves on a smaller
    image and upsamples, which keeps the direct solver fast.
    """
    l = initial_illumination(img)
    h, w = l.shape
    if solve_scale >= 1.0:
        return solve_illumination(l, lam, eps, window).astype(np.float32)
    small = cv2.resize(l, None, fx=solve_scale, fy=solve_scale, interpolation=cv2.INTER_AREA)
    t = solve_illumination(small, lam, eps, window).astype(np.float32)
    return np.clip(cv2.resize(t, (w, h), interpolation=cv2.INTER_LINEAR), 0.0, 1.0)


def brightness_transform(img, k, a=-0.3293, b=1.1258):
    """BIMEF Eq. 16: g(P, k) = exp(b(1 - k^a)) * P^(k^a)."""
    gamma = k**a
    beta = np.exp(b * (1.0 - gamma))
    return beta * np.power(np.maximum(img, 0.0), gamma)


def entropy(values, bins=256):
    """BIMEF Eq. 20: Shannon entropy in bits of values in [0, 1]."""
    hist, _ = np.histogram(np.clip(values, 0.0, 1.0), bins=bins, range=(0.0, 1.0))
    p = hist[hist > 0] / hist.sum()
    return float(-(p * np.log2(p)).sum())


def find_exposure_ratio(
    img, t, threshold=0.5, a=-0.3293, b=1.1258, k_range=(1.0, 7.0), search_size=50
):
    """BIMEF Eqs. 17-21: the k that maximises the entropy of the under-exposed pixels.

    Sec. IV-C resizes to 50 x 50 for this search. Returns 1 (no change) when no
    pixel is under-exposed, a case the paper does not cover.
    """
    size = (search_size, search_size)
    small = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    small_t = cv2.resize(t, size, interpolation=cv2.INTER_AREA)
    pixels = small[small_t < threshold]
    if pixels.size == 0:
        return 1.0
    brightness = np.cbrt(np.prod(pixels, axis=1))  # Eq. 18

    def negative_entropy(k):
        return -entropy(np.clip(brightness_transform(brightness, k, a, b), 0.0, 1.0))

    return float(
        minimize_scalar(negative_entropy, bounds=tuple(k_range), method="bounded").x
    )


def luminance_compensation(
    img,
    mu=0.5,
    lam=1.0,
    eps=0.001,
    window=5,
    solve_scale=0.5,
    threshold=0.5,
    a=-0.3293,
    b=1.1258,
    k_range=(1.0, 7.0),
    search_size=50,
    method="paper",
    pcg_tol=0.1,
    offset=0.0,
    clip=True,
):
    """BIMEF Eqs. 4-5: R = W * P + (1 - W) * g(P, k), with W = T^mu.

    method="paper" follows the BIMEF paper as written. method="reference"
    follows the choices of BIMEF's released implementation, which the CDEF
    authors use (see reference_illumination). offset is subtracted from
    g(P, k) and clip=False leaves values outside [0, 1] for the next stage,
    both as in that implementation.
    """
    if method not in ("paper", "reference"):
        raise ValueError(f"unknown luminance method: {method}")
    img = img.astype(np.float64)
    if method == "paper":
        t = estimate_illumination(img, lam, eps, window, solve_scale).astype(np.float64)
        k = find_exposure_ratio(img, t, threshold, a, b, k_range, search_size)
    else:
        t = reference_illumination(img, lam, eps, window, solve_scale, pcg_tol)
        k = reference_exposure_ratio(img, t, threshold, a, b, k_range, search_size)
    w = np.power(np.maximum(t, 0.0), mu)[..., None]
    enhanced = brightness_transform(img, k, a, b) - offset if k != 1.0 else img
    out = w * img + (1.0 - w) * enhanced
    if clip:
        out = np.clip(out, 0.0, 1.0)
    return out.astype(np.float32), k


# --------------------------------------------------------------------------
# BIMEF as in its released implementation (method="reference").
#
# That implementation differs from the paper in four ways that matter here:
# the smoothness weights sum the gradient along a line of `window` pixels in
# its own direction, not over a square; the image wraps around at its border
# when the illumination map is solved; the solve is approximate (conjugate
# gradients preconditioned by a modified incomplete Cholesky factor, stopped
# at a loose relative residual); and resizing is antialiased bicubic.


def keys_cubic(x):
    """Keys' cubic convolution kernel with a = -0.5."""
    ax = np.abs(x)
    ax2, ax3 = ax * ax, ax * ax * ax
    return (1.5 * ax3 - 2.5 * ax2 + 1.0) * (ax <= 1) + (-0.5 * ax3 + 2.5 * ax2 - 4.0 * ax + 2.0) * (
        (ax > 1) & (ax <= 2)
    )


def resize_matrix(n_in, n_out, scale=None):
    """Sparse (n_out, n_in) resampling matrix: Keys cubic, widened by 1/scale
    when shrinking (antialiasing), with mirrored borders. scale defaults to
    n_out / n_in."""
    scale = n_out / n_in if scale is None else scale
    width = 4.0 / scale if scale < 1 else 4.0
    u = np.arange(1, n_out + 1) / scale + 0.5 * (1.0 - 1.0 / scale)
    left = np.floor(u - width / 2.0)
    taps = int(math.ceil(width)) + 2
    idx = left[:, None] + np.arange(taps)[None, :]
    d = u[:, None] - idx
    weights = scale * keys_cubic(scale * d) if scale < 1 else keys_cubic(d)
    weights = weights / weights.sum(axis=1, keepdims=True)
    mirror = np.concatenate([np.arange(n_in), np.arange(n_in)[::-1]])
    cols = mirror[np.mod(idx.astype(np.int64) - 1, 2 * n_in)]
    rows = np.repeat(np.arange(n_out), taps)
    # Duplicate (row, col) pairs from the mirroring are summed.
    return sparse.csr_matrix((weights.ravel(), (rows, cols.ravel())), shape=(n_out, n_in))


def _resample_axis(img, matrix, axis):
    moved = np.moveaxis(img, axis, 0)
    flat = moved.reshape(moved.shape[0], -1)
    out = (matrix @ flat).reshape((matrix.shape[0],) + moved.shape[1:])
    return np.moveaxis(out, 0, axis)


def resize_cubic(img, out_h=None, out_w=None, scale=None):
    """Antialiased bicubic resize to (out_h, out_w), or by `scale` with the
    size rounded up. The axis with the smaller scale is resized first."""
    img = np.asarray(img, np.float64)
    h, w = img.shape[:2]
    if scale is not None:
        out_h, out_w = int(math.ceil(h * scale)), int(math.ceil(w * scale))
        sh = sw = scale
    else:
        sh, sw = out_h / h, out_w / w
    passes = [(0, h, out_h, sh), (1, w, out_w, sw)]
    if sw < sh:
        passes.reverse()
    for axis, n_in, n_out, s in passes:
        img = _resample_axis(img, resize_matrix(n_in, n_out, s), axis)
    return img


def resize_mask(mask, out_h, out_w):
    """Resize a binary mask as 8-bit data (0 or 255), rounding and saturating
    after each pass, then keep values above the midpoint."""
    out = mask.astype(np.float64) * 255.0
    h, w = out.shape
    passes = [(0, h, out_h, out_h / h), (1, w, out_w, out_w / w)]
    if out_w / w < out_h / h:
        passes.reverse()
    for axis, n_in, n_out, s in passes:
        out = np.clip(np.round(_resample_axis(out, resize_matrix(n_in, n_out, s), axis)), 0.0, 255.0)
    return out > 127


def line_weights(l, length=5, eps=0.001):
    """Edge weights 1 / (|gradient summed along a line of `length` pixels| *
    |gradient| + eps), for forward differences that wrap around the border."""
    ones = np.ones(length, np.float64)
    one = np.ones(1, np.float64)
    gx = np.roll(l, -1, axis=1) - l
    gy = np.roll(l, -1, axis=0) - l
    sx = cv2.sepFilter2D(gx, -1, ones, one, borderType=cv2.BORDER_CONSTANT)
    sy = cv2.sepFilter2D(gy, -1, one, ones, borderType=cv2.BORDER_CONSTANT)
    return 1.0 / (np.abs(sx) * np.abs(gx) + eps), 1.0 / (np.abs(sy) * np.abs(gy) + eps)


def periodic_system(wx, wy, lam):
    """I + lam * sum over edges of w (e_p - e_q)(e_p - e_q)^T, where wx[i, j]
    joins (i, j) to (i, j + 1) and wy[i, j] joins (i, j) to (i + 1, j), with
    indices wrapping around."""
    h, w = wx.shape
    n = h * w
    idx = np.arange(n).reshape(h, w)
    p = np.concatenate([idx.ravel(), idx.ravel()])
    q = np.concatenate([np.roll(idx, -1, axis=1).ravel(), np.roll(idx, -1, axis=0).ravel()])
    wt = lam * np.concatenate([wx.ravel(), wy.ravel()])
    off = sparse.coo_matrix((-wt, (p, q)), shape=(n, n))
    diag = np.bincount(p, wt, n) + np.bincount(q, wt, n)
    return (off + off.T + sparse.diags(1.0 + diag)).tocsc()


def modified_incomplete_cholesky(A):
    """A ~ L L^T with L on the pattern of tril(A) (no fill). Each fill value
    that is dropped is taken off both affected diagonals, so L L^T keeps the
    row sums of A. Returns L as CSR."""
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


def preconditioned_cg(A, b, L, tol=0.1, max_iter=50):
    """Conjugate gradients from x = 0 with preconditioner (L L^T)^-1. Stops at
    relative residual `tol`; otherwise returns the iterate with the smallest
    residual."""
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


def reference_illumination(img, lam=0.5, eps=0.001, window=5, solve_scale=0.5, pcg_tol=0.1):
    """Refined illumination map T as in BIMEF's released implementation.

    pcg_tol=None solves the same system exactly instead (slower, and within
    about 0.2 grey levels of the approximate result in the final output).
    """
    h, w = img.shape[:2]
    small = resize_cubic(img.max(axis=2), scale=solve_scale)
    wx, wy = line_weights(small, window, eps)
    if pcg_tol is None:
        t_small = spsolve(periodic_system(wx, wy, lam), small.ravel()).reshape(small.shape)
    else:
        # Unknowns ordered column by column: transpose, so the x-edges run along
        # axis 0 and go in as the second weight map.
        A = periodic_system(wy.T.copy(), wx.T.copy(), lam)
        x = preconditioned_cg(A, small.T.ravel(), modified_incomplete_cholesky(A), pcg_tol)
        t_small = x.reshape(small.T.shape).T
    return resize_cubic(t_small, h, w)


def entropy_8bit(values):
    """Shannon entropy in bits of values in [0, 1] after rounding to 8 bits."""
    levels = np.round(np.clip(values, 0.0, 1.0) * 255.0).astype(np.int64)
    hist = np.bincount(levels.ravel(), minlength=256).astype(np.float64)
    p = hist[hist > 0] / hist.sum()
    return float(-(p * np.log2(p)).sum())


def reference_exposure_ratio(
    img, t, threshold=0.5, a=-0.3293, b=1.1258, k_range=(1.0, 7.0), search_size=50
):
    """Eqs. 17-21 with antialiased resizing and 8-bit entropy. Returns 1 when
    no pixel is under-exposed."""
    small = np.maximum(resize_cubic(img, search_size, search_size), 0.0)
    under = resize_mask(t < threshold, search_size, search_size)
    brightness = np.cbrt(small.prod(axis=2))[under]
    if brightness.size == 0:
        return 1.0

    def negative_entropy(k):
        return -entropy_8bit(brightness_transform(brightness, k, a, b))

    result = minimize_scalar(negative_entropy, bounds=tuple(k_range), method="bounded", options={"xatol": 1e-4})
    return float(result.x)
