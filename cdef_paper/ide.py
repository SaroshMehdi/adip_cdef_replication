"""F4: Image Dehazing and Exposure (IDE).

Ju, Ding, Ren, Yang, Zhang, Guo, "IDE: Image Dehazing and Exposure Using an
Enhanced Atmospheric Scattering Model", IEEE TIP 30, 2021. Equation numbers
below refer to that paper, and the steps are its Algorithm 1.

The enhanced scattering model (Eq. 8) is

    I = A * (ln t / ln t_min) * rho * t + A * (1 - t)

with ambient light A, transmission t and scene albedo rho. Assuming each
75 x 75 patch has mean albedo 1/2 gives t in closed form (Eqs. 12-17); the only
free parameter, t_min, is set per image so that 2% of the recovered values
fall outside [0, 1] (Eq. 22).
"""
import math

import cv2
import numpy as np


def box_mean(img, size):
    """Mean over a size x size window with replicated borders."""
    return cv2.blur(img, (size, size), borderType=cv2.BORDER_REPLICATE)


def rational_log(t, p1=-0.397, p2=0.07747):
    """Eq. 13: f(t) = p1 / (p2 + t), the fit used in place of ln(t)."""
    return p1 / (p2 + t)


def atmospheric_light(img, patch=15, top_fraction=0.001):
    """Step 1 of Algorithm 1, "via [17]" (He et al.): among the 0.1% of pixels
    with the largest dark channel, the brightest one. Returns one value per
    channel."""
    dark = cv2.erode(img.min(axis=2), np.ones((patch, patch), np.uint8))
    count = max(1, int(dark.size * top_fraction))
    candidates = np.argpartition(dark.ravel(), -count)[-count:]
    pixels = img.reshape(-1, 3)[candidates]
    return np.maximum(pixels[np.argmax(pixels.sum(axis=1))], 1e-3)


def guided_filter(guide, src, radius, eps):
    """He et al.'s guided filter with a grey guide (F in Eq. 18)."""
    size = 2 * radius + 1
    mean_guide = box_mean(guide, size)
    mean_src = box_mean(src, size)
    covariance = box_mean(guide * src, size) - mean_guide * mean_src
    variance = box_mean(guide * guide, size) - mean_guide * mean_guide
    a = covariance / (variance + eps)
    b = mean_src - a * mean_guide
    return box_mean(a, size) * guide + box_mean(b, size)


def patch_transmission(local_mean, a_mean, ln_tmin, p1=-0.397, p2=0.07747):
    """Eqs. 15-17: the root of lambda_1 t^2 + lambda_2 t + lambda_3 = 0."""
    lam1 = a_mean * ln_tmin
    lam2 = ln_tmin * (local_mean + a_mean * p2 - a_mean) - 0.5 * a_mean * p1
    lam3 = p2 * ln_tmin * (local_mean - a_mean)
    root = np.sqrt(np.maximum(lam2 * lam2 - 4.0 * lam1 * lam3, 0.0))
    return (-lam2 - root) / (2.0 * lam1)


def transmission_map(gray, local_mean, a_mean, tmin, radius, gf_eps, t_max, p1, p2):
    """Eq. 18: the patch transmission refined by the guided filter, kept in
    [t_min, t_max]."""
    raw = patch_transmission(local_mean, a_mean, math.log(tmin), p1, p2)
    return np.clip(guided_filter(gray, raw, radius, gf_eps), tmin, t_max)


def scene_albedo(img, airlight, t, tmin, recovery="rational", p1=-0.397, p2=0.07747):
    """Eq. 20: Eq. 8 solved for rho.

    Eq. 20 prints ln(t). t itself was solved with f(t) from Eq. 13 in place of
    ln(t) (Eqs. 14-17), and f fits ln badly near t = 1 (f(1) = -0.368, ln 1 = 0),
    so the printed form divides by almost zero in dark patches. "rational" uses
    f(t) here too, which returns each patch to its assumed mean albedo of 1/2;
    "log" is the printed form.
    """
    t = t[..., None]
    log_t = np.log(t) if recovery == "log" else rational_log(t, p1, p2)
    return (math.log(tmin) / log_t) * (img - airlight * (1.0 - t)) / (airlight * t)


def clipped_share(albedo):
    """Psi in Eq. 22: the share of values outside [0, 1]."""
    return float(np.mean((albedo < 0.0) | (albedo > 1.0)))


def golden_section(objective, low, high, tolerance=0.005):
    """Minimise a function on [low, high] by golden-section search (the
    search Algorithm 1 names)."""
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


def ide(img, p1=-0.397, p2=0.07747, epsilon=0.02, delta=4, patch=75, dark_patch=15,
        top_fraction=0.001, gf_radius=37, gf_eps=0.001, tmin_range=(0.05, 0.95),
        tolerance=0.005, recovery="rational", clip=True):
    """Algorithm 1 of the IDE paper. Returns (albedo as float32, t_min).

    Printed in the paper: p1, p2, epsilon = 2%, delta = 4, the 75 x 75 patch.
    Not given there, and chosen here: the dark-channel settings (He et al.'s),
    the guided filter (grey guide, radius of half a patch, eps 0.001), the
    search range of t_min, and t limited to [t_min, 1].
    """
    img = img.astype(np.float64)
    t_max = 0.99 if recovery == "log" else 1.0  # ln(1) = 0 in the printed Eq. 20
    airlight = atmospheric_light(img, dark_patch, top_fraction)
    a_mean = float(airlight.mean())  # Eq. 10
    gray = img.mean(axis=2)
    local_mean = box_mean(gray, patch)  # Eq. 10

    # Eq. 22: t_min is searched on a copy downsampled by delta.
    small = cv2.resize(img, None, fx=1.0 / delta, fy=1.0 / delta, interpolation=cv2.INTER_AREA)
    small_gray = small.mean(axis=2)
    small_mean = box_mean(small_gray, max(3, (patch // delta) | 1))
    small_radius = max(1, gf_radius // delta)

    def distance_from_target(tmin):
        t = transmission_map(small_gray, small_mean, a_mean, tmin, small_radius, gf_eps, t_max, p1, p2)
        return abs(clipped_share(scene_albedo(small, airlight, t, tmin, recovery, p1, p2)) - epsilon)

    tmin = golden_section(distance_from_target, *tmin_range, tolerance=tolerance)
    t = transmission_map(gray, local_mean, a_mean, tmin, gf_radius, gf_eps, t_max, p1, p2)
    albedo = scene_albedo(img, airlight, t, tmin, recovery, p1, p2)
    if clip:
        albedo = np.clip(albedo, 0.0, 1.0)
    return albedo.astype(np.float32), tmin
