"""Reading and writing images as float RGB in [0, 1]."""
import cv2
import numpy as np


def read_image(path):
    """Read an image file as float32 RGB in [0, 1]."""
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"could not read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def write_image(path, img):
    """Write a float RGB image in [0, 1] as an 8-bit file."""
    out = np.round(np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
    if not cv2.imwrite(str(path), cv2.cvtColor(out, cv2.COLOR_RGB2BGR)):
        raise OSError(f"could not write image: {path}")
