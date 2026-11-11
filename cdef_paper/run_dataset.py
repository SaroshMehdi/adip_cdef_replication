"""Enhance a dataset with the paper version and save every output.

Writes outputs/<method>/<dataset>/<image name>.png, the layout the scoring
scripts in metrics/ read. Images already saved are skipped, so a stopped run
resumes where it left off.

python -m cdef_paper.run_dataset --dataset UIEB C60 EUVP UCCS

--sample uses the fixed subset of every 15th UIEB image and every 3rd C60
image (60 and 20 images).

The luminance stage solves its linear system exactly at full resolution, which
reserves about 4.5 GB of memory per megapixel. Images are therefore run in
groups by size, each with as many workers as fit in --memory-gb.
"""
import argparse
import os
from functools import partial
from multiprocessing import Pool
from pathlib import Path

os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")

import cv2
from PIL import Image

from .image_io import read_image, write_image
from .pipeline import DEFAULT_CONFIG, enhance, load_config

cv2.setNumThreads(1)
DATASETS = {
    "UIEB": Path("data/UIEB/raw"),
    "C60": Path("data/UIEB/challenging"),
    "EUVP": Path("data/EUVP/test_samples/Inp"),  # the 515-image test set
    "UCCS": Path("data/UCCS"),  # blue/, green/ and blue-green/, 100 each
}
SAMPLE_STEP = {"UIEB": 15, "C60": 3}
IMAGE_TYPES = {".png", ".jpg", ".jpeg"}
GB_PER_MEGAPIXEL = 4.5  # memory reserved by the luminance solve
SIZE_GROUPS = (0.25, 0.5, 1.0, float("inf"))  # upper limits in megapixels


def dataset_images(dataset, sample=False):
    images = sorted(p for p in DATASETS[dataset].rglob("*") if p.suffix.lower() in IMAGE_TYPES)
    if not sample:
        return images
    if dataset not in SAMPLE_STEP:
        raise ValueError(f"--sample is defined for {sorted(SAMPLE_STEP)} only")
    return images[:: SAMPLE_STEP[dataset]]


def megapixels(path):
    with Image.open(path) as image:
        return image.width * image.height / 1e6


def size_groups(jobs, workers, memory_gb):
    """Split jobs into (jobs, workers) groups by image size, so that the
    workers of a group together stay within memory_gb."""
    sizes = [megapixels(source) for source, _ in jobs]
    lower = 0.0
    for upper in SIZE_GROUPS:
        group = [job for job, size in zip(jobs, sizes) if lower < size <= upper]
        if group:
            yield group, max(1, min(workers, int(memory_gb // (GB_PER_MEGAPIXEL * upper))))
        lower = upper


def enhance_and_save(cfg, job):
    source, target = job
    result, _ = enhance(read_image(source), cfg)
    # Written under a temporary name, so a stop can never leave a partial PNG.
    partial_file = target.with_suffix(".tmp.png")
    write_image(partial_file, result)
    os.replace(partial_file, target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--dataset", nargs="+", choices=sorted(DATASETS), default=["UIEB", "C60"])
    parser.add_argument("--sample", action="store_true", help="fixed 60 UIEB + 20 C60 subset")
    parser.add_argument("--method", default="cdef_paper", help="output folder name")
    parser.add_argument("--output", default="outputs")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--memory-gb", type=float, default=9.0, help="memory the workers may use together")
    args = parser.parse_args()

    jobs = []
    for dataset in args.dataset:
        folder = Path(args.output) / args.method / dataset
        folder.mkdir(parents=True, exist_ok=True)
        for leftover in folder.glob("*.tmp.png"):
            leftover.unlink()
        for source in dataset_images(dataset, args.sample):
            target = folder / (source.stem + ".png")
            if not target.exists():
                jobs.append((source, target))
    print(f"{len(jobs)} images to enhance", flush=True)

    worker = partial(enhance_and_save, load_config(args.config))
    done = 0
    for group, workers in size_groups(jobs, args.workers, args.memory_gb):
        with Pool(workers) as pool:
            for _ in pool.imap_unordered(worker, group):
                done += 1
                if done % 50 == 0 or done == len(jobs):
                    print(f"{done}/{len(jobs)}", flush=True)


if __name__ == "__main__":
    main()
