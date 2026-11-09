"""Small helpers shared by the CDEF modules."""
import cv2
import numpy as np


def read_image(path):
    """Read an image file as float32 RGB in [0, 1]."""
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"could not read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def write_image(path, img):
    """Write a float RGB image in [0, 1] as 8-bit."""
    out = np.round(np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
    if not cv2.imwrite(str(path), cv2.cvtColor(out, cv2.COLOR_RGB2BGR)):
        raise OSError(f"could not write image: {path}")


def gaussian_blur(img, sigma):
    """Gaussian blur with replicated borders. sigma <= 0 returns a copy."""
    if sigma <= 0:
        return img.copy()
    return cv2.GaussianBlur(img, (0, 0), float(sigma), borderType=cv2.BORDER_REPLICATE)


def box_mean(img, size):
    """Mean over a size x size window with replicated borders."""
    return cv2.blur(img, (size, size), borderType=cv2.BORDER_REPLICATE)
