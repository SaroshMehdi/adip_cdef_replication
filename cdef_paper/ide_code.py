"""F4: Image Dehazing and Exposure (IDE).

Ju, Ding, Ren, Yang, Zhang, Guo, "IDE: Image Dehazing and Exposure Using an
Enhanced Atmospheric Scattering Model", IEEE TIP 30, 2021. Equation numbers
below refer to that paper.

The enhanced scattering model (Eq. 8) is

    I = A * (ln t / ln t_min) * rho * t + A * (1 - t)

with ambient light A, transmission t and scene albedo rho. Assuming each
75 x 75 patch has mean albedo 1/2 gives t in closed form (Eqs. 12-17); the only
free parameter, t_min, is set per image so that only a small share of the
recovered pixels would fall outside [0, 1] (Eq. 22).

Details follow the IDE code shipped with CDEF: per-channel ambient light, the
two-step t_min search, f(t) in place of ln(t) in the recovery, and box
smoothing of t (which is what the shipped guided filter reduces to).
"""
import math

import cv2
import numpy as np


def rational_log(t, p1=-0.397, p2=0.07747):
    """Eq. 13: f(t) = p1 / (p2 + t), the fit used in place of ln(t)."""
    return p1 / (p2 + t)


def ambient_light(img, window=10):
    """A per channel: the largest value of the channel's local minimum over a
    window x window neighbourhood (Meng et al.'s estimate)."""
    kernel = np.ones((window, window), np.uint8)
    return np.array(
        [
            cv2.erode(np.ascontiguousarray(img[..., c]), kernel, borderType=cv2.BORDER_REFLECT).max()
            for c in range(img.shape[2])
        ]
    )


def quadratic_coefficients(local_mean, a_mean, ln_tmin, p1=-0.397, p2=0.07747):
    """Eq. 16: lambda_1, lambda_2, lambda_3 of the quadratic in t (Eq. 15)."""
    lam1 = a_mean * ln_tmin
    lam2 = ln_tmin * (local_mean + a_mean * p2 - a_mean) - 0.5 * a_mean * p1
    lam3 = p2 * ln_tmin * (local_mean - a_mean)
    return lam1, lam2, lam3


def patch_transmission(local_mean, a_mean, ln_tmin, p1=-0.397, p2=0.07747):
    """Eq. 17: the root of Eq. 15, before smoothing."""
    lam1, lam2, lam3 = quadratic_coefficients(local_mean, a_mean, ln_tmin, p1, p2)
    root = np.sqrt(np.maximum(lam2 * lam2 - 4.0 * lam1 * lam3, 0.0))
    return (-lam2 - root) / (2.0 * lam1)


def recover_albedo(img, airlight, t, tmin, p1=-0.397, p2=0.07747):
    """Eq. 20, inverting Eq. 8 for rho, with f(t) from Eq. 13 in place of ln(t)
    (the function t was solved with), so a pixel with t = t_min keeps a gain of 1."""
    t3 = t[..., None]
    gain = rational_log(tmin, p1, p2) / rational_log(t3, p1, p2)
    return gain * (img - airlight * (1.0 - t3)) / (t3 * airlight)


def clipped_share(albedo):
    """Psi in Eq. 22: the share of values at or beyond 0 and 1."""
    return float(np.mean((albedo <= 0.0) | (albedo >= 1.0)))


def golden_section(objective, low, high, tolerance=0.001):
    """Minimise a unimodal function on [low, high] by golden-section search."""
    ratio = (math.sqrt(5.0) - 1.0) / 2.0
    x1 = high - ratio * (high - low)
    x2 = low + ratio * (high - low)
    f1, f2 = objective(x1), objective(x2)
    while high - low > tolerance:
        if f1 <= f2:
            high, x2, f2 = x2, x1, f1
            x1 = high - ratio * (high - low)
            f1 = objective(x1)
        else:
            low, x1, f1 = x1, x2, f2
            x2 = low + ratio * (high - low)
            f2 = objective(x2)
    return (low + high) / 2.0


def find_tmin(clipped, epsilon=0.02, tolerance=0.001):
    """Eq. 22 as in the shipped code. First find the t_min with the fewest
    clipped values; then, below it, bisect for the t_min whose clipped share
    is epsilon above that minimum (lower t_min means stronger dehazing)."""
    best = golden_section(clipped, 0.001, 1.0, tolerance)
    target = clipped(best) + epsilon
    low, high = 0.01, best
    f_low = clipped(low) - target
    while high - low > tolerance:
        mid = 0.5 * (low + high)
        f_mid = clipped(mid) - target
        if f_low * f_mid <= 0:
            high = mid
        else:
            low, f_low = mid, f_mid
    return 0.5 * (low + high)


def mean_inside(img, radius):
    """Mean over a (2 radius + 1)^2 window, counting only pixels inside the image."""
    size = (2 * radius + 1, 2 * radius + 1)
    total = cv2.boxFilter(img, -1, size, normalize=False, borderType=cv2.BORDER_CONSTANT)
    count = cv2.boxFilter(np.ones_like(img), -1, size, normalize=False, borderType=cv2.BORDER_CONSTANT)
    return total / count


def ide(
    img,
    p1=-0.397,
    p2=0.07747,
    epsilon=0.02,
    delta=4,
    patch=75,
    air_window=10,
    tmin_offset=0.02,
    smooth_radius=50,
    tolerance=0.001,
):
    """Dehaze an RGB image. Returns (albedo as float32, t_min).

    The t_min search runs on a copy downsampled by delta (Ju's paper; the
    shipped code searches at full size, about ten times slower), and
    tmin_offset is added to its result as in the shipped code. The albedo is
    not clipped, so values just outside [0, 1] reach the fusion.
    """
    img = img.astype(np.float64)
    airlight = ambient_light(img, air_window)
    a_mean = float(airlight.mean())  # Eq. 10
    local_mean = cv2.blur(img.mean(axis=2), (patch, patch), borderType=cv2.BORDER_REFLECT)

    if delta > 1:
        s_img = cv2.resize(img, None, fx=1.0 / delta, fy=1.0 / delta, interpolation=cv2.INTER_AREA)
        s_mean = cv2.resize(local_mean, (s_img.shape[1], s_img.shape[0]), interpolation=cv2.INTER_AREA)
    else:
        s_img, s_mean = img, local_mean

    def clipped(tmin):
        t = np.abs(patch_transmission(s_mean, a_mean, math.log(tmin), p1, p2))
        return clipped_share(recover_albedo(s_img, airlight, t, tmin, p1, p2))

    tmin = find_tmin(clipped, epsilon, tolerance) + tmin_offset

    t = np.abs(patch_transmission(local_mean, a_mean, math.log(tmin), p1, p2))
    t = mean_inside(mean_inside(t, smooth_radius), smooth_radius)  # Eq. 18's smoothing
    t = np.maximum(t, tmin)
    return recover_albedo(img, airlight, t, tmin, p1, p2).astype(np.float32), tmin
