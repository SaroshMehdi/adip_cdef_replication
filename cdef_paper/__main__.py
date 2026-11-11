"""Enhance one image or a folder of images with the paper version:

python -m cdef_paper --input data/UIEB/raw --output outputs/UIEB_paper
"""
import argparse
from pathlib import Path

from tqdm import tqdm

from .image_io import read_image, write_image
from .pipeline import DEFAULT_CONFIG, enhance, load_config

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--input", required=True, help="image file or folder")
    parser.add_argument("--output", required=True, help="output folder")
    args = parser.parse_args()

    cfg = load_config(args.config)
    source = Path(args.input)
    if source.is_dir():
        paths = sorted(p for p in source.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    else:
        paths = [source]
    if not paths:
        raise SystemExit(f"no images found in {source}")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    for path in tqdm(paths):
        result, _ = enhance(read_image(path), cfg)
        write_image(output / f"{path.stem}.png", result)


if __name__ == "__main__":
    main()
