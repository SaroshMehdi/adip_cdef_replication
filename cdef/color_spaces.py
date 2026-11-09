"""Colour space conversions: CIELab for colour compensation, YUV (and Lab, RGB
for the ablation) for the fusion."""
import numpy as np

# ---------------------------------------------------------------- CIELab
# sRGB primaries to CIE XYZ, and the D65 white point.
SRGB_TO_XYZ = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ]
)
XYZ_TO_SRGB = np.linalg.inv(SRGB_TO_XYZ)
WHITE_D65 = np.array([0.9504559, 1.0, 1.0890578])


def srgb_to_linear(c):
    a = np.abs(c)
    return np.sign(c) * np.where(a <= 0.04045, a / 12.92, ((a + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c):
    a = np.abs(c)
    return np.sign(c) * np.where(a <= 0.0031308, 12.92 * a, 1.055 * a ** (1 / 2.4) - 0.055)


def rgb_to_lab(rgb):
    """sRGB (D65) to CIELab: L in [0, 100], a and b roughly in [-100, 100].

    Values outside [0, 1] are converted rather than clipped: the luminance
    stage can leave a few, and clipping them first would shift their hue.
    """
    xyz = srgb_to_linear(np.asarray(rgb, np.float64)) @ SRGB_TO_XYZ.T / WHITE_D65
    f = np.where(xyz > 216 / 24389, np.cbrt(xyz), (24389 / 27 * xyz + 16) / 116)
    return np.stack(
        [116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], axis=-1
    )


def lab_to_rgb(lab):
    """CIELab to sRGB (D65), the inverse of rgb_to_lab. Out-of-gamut colours
    come back outside [0, 1]; the caller decides whether to clip."""
    fy = (lab[..., 0] + 16) / 116
    f = np.stack([fy + lab[..., 1] / 500, fy, fy - lab[..., 2] / 200], axis=-1)
    xyz = np.where(f > 6 / 29, f**3, (116 * f - 16) * 27 / 24389) * WHITE_D65
    return linear_to_srgb(xyz @ XYZ_TO_SRGB.T)


# ---------------------------------------------------------------- YUV
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
LUMA = np.array([0.299, 0.587, 0.114])


def rgb_to_yuv(img):
    return img @ RGB_TO_YUV.T


def yuv_to_rgb(img):
    return img @ YUV_TO_RGB.T


# ------------------------------------------------- fusion colour spaces
def rgb_to_lab_unit(img):
    """CIELab divided by 100, so L is in [0, 1] like Y."""
    return (rgb_to_lab(img) / 100.0).astype(np.float32)


def lab_unit_to_rgb(img):
    return lab_to_rgb(np.asarray(img, np.float64) * 100.0)


def rgb_copy(img):
    return np.array(img, np.float32, copy=True)


# Name -> (forward, backward) conversion for the fusion. The paper fuses in
# YUV; RGB and Lab are the alternatives compared in its ablation (Table IV).
FUSION_SPACES = {
    "yuv": (rgb_to_yuv, yuv_to_rgb),
    "lab": (rgb_to_lab_unit, lab_unit_to_rgb),
    "rgb": (rgb_copy, rgb_copy),
}
