"""Image Dehazing and Exposure.

Ju, Ding, Ren, Yang, Zhang, Guo, "IDE: Image Dehazing and Exposure Using an
Enhanced Atmospheric Scattering Model", IEEE TIP 30, 2021. Equation numbers in
this file refer to that paper.
"""
import math

import cv2
import numpy as np

from ..utils import box_mean


def rational_log(t, p1=-0.397, p2=0.07747):
    """Eq. 13: f(t) = p1 / (p2 + t), the fit used in place of ln(t)."""
    return p1 / (p2 + t)


def atmospheric_light(img, patch=15, top_fraction=0.001):
    """Step 1 of Algorithm 1 (from He et al.): brightest input pixel among the
    top dark-channel pixels. Returns a 3-vector."""
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (patch, patch))
    dark = cv2.erode(img.min(axis=2), element)
    count = max(1, int(dark.size * top_fraction))
    candidates = np.argpartition(dark.ravel(), -count)[-count:]
    pixels = img.reshape(-1, 3)[candidates]
    return np.maximum(pixels[np.argmax(pixels.sum(axis=1))], 1e-3)


def guided_filter(guide, src, radius, eps):
    """Guided filter of He et al. with a gray guide (F in Eq. 18)."""
    size = 2 * radius + 1
    mean_guide = box_mean(guide, size)
    mean_src = box_mean(src, size)
    covariance = box_mean(guide * src, size) - mean_guide * mean_src
    variance = box_mean(guide * guide, size) - mean_guide * mean_guide
    a = covariance / (variance + eps)
    b = mean_src - a * mean_guide
    return box_mean(a, size) * guide + box_mean(b, size)


def quadratic_coefficients(local_mean, a_mean, ln_tmin, p1=-0.397, p2=0.07747):
    """Eq. 16: lambda_1, lambda_2, lambda_3."""
    lam1 = a_mean * ln_tmin
    lam2 = ln_tmin * (local_mean + a_mean * p2 - a_mean) - 0.5 * a_mean * p1
    lam3 = p2 * ln_tmin * (local_mean - a_mean)
    return lam1, lam2, lam3


def patch_transmission(local_mean, a_mean, ln_tmin, p1=-0.397, p2=0.07747):
    """Eq. 17: the positive root of Eq. 15, before the guided filter."""
    lam1, lam2, lam3 = quadratic_coefficients(local_mean, a_mean, ln_tmin, p1, p2)
    root = np.sqrt(np.maximum(lam2 * lam2 - 4.0 * lam1 * lam3, 0.0))
    return (-lam2 - root) / (2.0 * lam1)


def scene_albedo(img, airlight, t, ln_tmin, log_mode="rational", p1=-0.397, p2=0.07747):
    """Eq. 20: invert Eq. 8 for the albedo.

    Eq. 20 prints ln(t). "rational" uses f(t) from Eq. 13 instead, which is the
    function t was solved with, so a patch recovers to the mean albedo of 1/2.
    Eq. 13 is wrong near t = 1, where ln(t) goes to 0 but f(t) does not, so
    "exact" blows up in dark regions and its clipped fraction is U-shaped in
    t_min, which the golden-section search of Eq. 22 cannot handle.
    """
    t = t[..., None]
    denominator = rational_log(t, p1, p2) if log_mode == "rational" else np.log(t)
    return (ln_tmin / denominator) * (1.0 + (img - airlight) / (airlight * t))


def clipped_fraction(albedo):
    """Psi in Eq. 22: the share of values outside [0, 1]."""
    return float(np.mean((albedo < 0.0) | (albedo > 1.0)))


def golden_section(objective, low, high, tolerance=0.005):
    """Golden-section minimiser on [low, high]."""
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


def transmission_map(gray, local_mean, a_mean, tmin, radius, gf_eps, t_max, p1, p2):
    """Eq. 18, limited to [t_min, t_max]."""
    raw = patch_transmission(local_mean, a_mean, math.log(tmin), p1, p2)
    return np.clip(guided_filter(gray, raw, radius, gf_eps), tmin, t_max)


def ide(
    img,
    p1=-0.397,
    p2=0.07747,
    epsilon=0.02,
    delta=4,
    patch=75,
    dark_patch=15,
    top_fraction=0.001,
    gf_radius=37,
    gf_eps=0.001,
    log_mode="rational",
    tmin_range=(0.05, 0.95),
    t_max=1.0,
    method="paper",
    air_window=10,
    tmin_offset=0.02,
    smoothing="box",
    smooth_radius=50,
    tolerance=0.001,
    clip=False,
    airlight="channel",
    search="relative",
    recovery="rational",
):
    """Algorithm 1 of the paper. Returns (albedo, t_min).

    method="paper" follows the IDE paper (albedo clipped to [0, 1]).
    method="reference" follows the IDE code shipped with CDEF; see
    reference_ide for what changes and which arguments it uses.
    """
    if method == "reference":
        return reference_ide(
            img, p1, p2, epsilon, delta, patch, air_window, tmin_offset,
            smoothing, smooth_radius, gf_radius, gf_eps, tolerance, clip,
            airlight, search, recovery, dark_patch, top_fraction, tmin_range,
        )
    if method != "paper":
        raise ValueError(f"unknown IDE method: {method}")
    if log_mode not in ("rational", "exact"):
        raise ValueError(f"unknown IDE log mode: {log_mode}")
    if log_mode == "exact":
        t_max = min(t_max, 0.99)  # ln(1) = 0 in Eq. 20

    airlight = atmospheric_light(img, dark_patch, top_fraction).astype(np.float32)
    a_mean = float(airlight.mean())  # Eq. 10
    gray = img.mean(axis=2)
    local_mean = box_mean(gray, patch)  # Eq. 10

    # Eq. 22: search t_min on images downsampled by delta.
    scale = 1.0 / delta
    small = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    small_gray = small.mean(axis=2)
    small_mean = box_mean(small_gray, max(3, (patch // delta) | 1))
    small_radius = max(1, gf_radius // delta)

    def saturation_error(tmin):
        t = transmission_map(small_gray, small_mean, a_mean, tmin, small_radius, gf_eps, t_max, p1, p2)
        albedo = scene_albedo(small, airlight, t, math.log(tmin), log_mode, p1, p2)
        return abs(clipped_fraction(albedo) - epsilon)

    tmin = golden_section(saturation_error, *tmin_range)

    t = transmission_map(gray, local_mean, a_mean, tmin, gf_radius, gf_eps, t_max, p1, p2)
    albedo = scene_albedo(img, airlight, t, math.log(tmin), log_mode, p1, p2)
    return np.clip(albedo, 0.0, 1.0).astype(np.float32), tmin


# --------------------------------------------------------------------------
# IDE as in the code shipped with CDEF (method="reference").


def channel_airlight(img, window=10):
    """Per channel, the largest value of the local-minimum image over a
    window x window neighbourhood (Meng et al.'s estimate), not He et al.'s
    dark-channel pixel."""
    kernel = np.ones((window, window), np.uint8)
    return np.array(
        [
            cv2.erode(np.ascontiguousarray(img[..., c]), kernel, borderType=cv2.BORDER_REFLECT).max()
            for c in range(img.shape[2])
        ]
    )


def mean_inside(img, radius):
    """Mean over a (2 radius + 1)^2 window, counting only pixels inside the image."""
    size = (2 * radius + 1, 2 * radius + 1)
    total = cv2.boxFilter(img, -1, size, normalize=False, borderType=cv2.BORDER_CONSTANT)
    count = cv2.boxFilter(np.ones_like(img), -1, size, normalize=False, borderType=cv2.BORDER_CONSTANT)
    return total / count


def rational_recovery(img, airlight, t, tmin, p1=-0.397, p2=0.07747, recovery="rational"):
    """Eq. 20. recovery="rational" uses f from Eq. 13 in both places,
    f(t_min) / f(t), so a pixel with t = t_min keeps a factor of 1.
    recovery="log" is Eq. 20 as printed, ln(t_min) / ln(t); t must be below 1."""
    t3 = t[..., None]
    if recovery == "log":
        gain = math.log(tmin) / np.log(t3)
    else:
        gain = rational_log(tmin, p1, p2) / rational_log(t3, p1, p2)
    return gain * (img - airlight * (1.0 - t3)) / (t3 * airlight)


def clipped_share(albedo):
    """Share of values at or beyond 0 and 1."""
    return float(np.mean((albedo <= 0.0) | (albedo >= 1.0)))


def reference_ide(
    img,
    p1=-0.397,
    p2=0.07747,
    epsilon=0.02,
    delta=4,
    patch=75,
    air_window=10,
    tmin_offset=0.02,
    smoothing="box",
    smooth_radius=50,
    gf_radius=37,
    gf_eps=0.001,
    tolerance=0.001,
    clip=False,
    airlight="channel",
    search="relative",
    recovery="rational",
    dark_patch=15,
    top_fraction=0.001,
    tmin_range=(0.05, 0.95),
):
    """IDE as in the code shipped with CDEF. Differences from the paper:

    - airlight per channel (channel_airlight), not from the dark channel;
    - t_min search: first the t_min with the fewest clipped values, then the
      lower t_min where the clipped share is epsilon above that minimum, plus
      tmin_offset. The search uses the unsmoothed t, on an image downsampled
      by delta (their code searches at full size, delta = 1, which costs about
      ten times as long and scores slightly lower);
    - f(t_min) / f(t) in Eq. 20 (rational_recovery);
    - t smoothed by "box": two passes of a (2 smooth_radius + 1)^2 mean. Their
      guided filter runs on an image scaled down twice, so its guide has almost
      no variance and it reduces to exactly this. "guided" uses a real guided
      filter on the gray image instead;
    - t is kept at least t_min but has no upper limit, and with clip=False the
      albedo is not clipped, so the fusion sees the overshoot.

    Each of the first three can be set back to the paper's rule:
    airlight="dark_channel" (He et al., dark_patch, top_fraction);
    search="absolute", Eq. 22's argmin |clipped share - epsilon| over tmin_range
    (tmin_offset still added); recovery="log", Eq. 20's ln(t_min) / ln(t),
    with t limited to 0.99 because ln(1) = 0.
    """
    if smoothing not in ("box", "guided"):
        raise ValueError(f"unknown IDE smoothing: {smoothing}")
    if airlight not in ("channel", "dark_channel"):
        raise ValueError(f"unknown IDE airlight: {airlight}")
    if search not in ("relative", "absolute"):
        raise ValueError(f"unknown IDE search: {search}")
    if recovery not in ("rational", "log"):
        raise ValueError(f"unknown IDE recovery: {recovery}")
    t_cap = 0.99 if recovery == "log" else np.inf
    img = img.astype(np.float64)
    if airlight == "channel":
        light = channel_airlight(img, air_window)
    else:
        light = atmospheric_light(img, dark_patch, top_fraction).astype(np.float64)
    a_mean = float(light.mean())
    gray = img.mean(axis=2)
    local_mean = cv2.blur(gray, (patch, patch), borderType=cv2.BORDER_REFLECT)

    if delta > 1:
        scale = 1.0 / delta
        s_img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        s_mean = cv2.resize(local_mean, (s_img.shape[1], s_img.shape[0]), interpolation=cv2.INTER_AREA)
    else:
        s_img, s_mean = img, local_mean

    def clipped(tmin):
        t = np.minimum(np.abs(patch_transmission(s_mean, a_mean, math.log(tmin), p1, p2)), t_cap)
        return clipped_share(rational_recovery(s_img, light, t, tmin, p1, p2, recovery))

    if search == "absolute":
        tmin = golden_section(lambda x: abs(clipped(x) - epsilon), *tmin_range, tolerance=tolerance)
    else:
        best = golden_section(clipped, 0.001, 1.0, tolerance=tolerance)
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
        tmin = 0.5 * (low + high)
    tmin = tmin + tmin_offset
    if recovery == "log":
        tmin = min(tmin, 0.98)

    t = np.abs(patch_transmission(local_mean, a_mean, math.log(tmin), p1, p2))
    if smoothing == "box":
        t = mean_inside(mean_inside(t, smooth_radius), smooth_radius)
    else:
        t = guided_filter(gray, t, gf_radius, gf_eps)
    t = np.minimum(np.maximum(t, tmin), t_cap)
    albedo = rational_recovery(img, light, t, tmin, p1, p2, recovery)
    if clip:
        albedo = np.clip(albedo, 0.0, 1.0)
    return albedo.astype(np.float32), tmin
