"""Enhance a dataset and save every output.

Writes outputs/<method>/<dataset>/<image name>.png. Images already saved are skipped,
so a stopped run resumes where it left off and finished work is never redone.

python -m eval.run_dataset --config configs/cdef_paper.yaml --dataset UIEB C60

--sample uses a fixed subset of every 15th UIEB image and every 3rd C60 image
(60 and 20 images), the subset used for the comparisons in docs/deviations.md.
"""
import argparse
import os
from functools import partial
from multiprocessing import Pool
from pathlib import Path

os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")

import cv2

from cdef.pipeline import enhance, load_config
from cdef.utils import read_image, write_image

cv2.setNumThreads(1)
DATASETS = {
    "UIEB": Path("data/UIEB/raw"),
    "C60": Path("data/UIEB/challenging"),
    "EUVP": Path("data/EUVP/test_samples/Inp"),  # the 515-image test set
    "UCCS": Path("data/UCCS"),  # blue/, green/ and blue-green/, 100 each
}
SAMPLE_STEP = {"UIEB": 15, "C60": 3}
IMAGE_TYPES = {".png", ".jpg", ".jpeg"}


def dataset_images(dataset, sample=False):
    images = sorted(p for p in DATASETS[dataset].rglob("*") if p.suffix.lower() in IMAGE_TYPES)
    if not sample:
        return images
    if dataset not in SAMPLE_STEP:
        raise ValueError(f"--sample is defined for {sorted(SAMPLE_STEP)} only")
    return images[:: SAMPLE_STEP[dataset]]


def enhance_and_save(cfg, job):
    source, target = job
    result, _ = enhance(read_image(source), cfg)
    # Written under a temporary name, so a stop can never leave a partial PNG.
    partial_file = target.with_suffix(".tmp.png")
    write_image(partial_file, result)
    os.replace(partial_file, target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/cdef_paper.yaml")
    parser.add_argument("--dataset", nargs="+", choices=sorted(DATASETS), default=["UIEB", "C60"])
    parser.add_argument("--sample", action="store_true", help="fixed 60 UIEB + 20 C60 subset")
    parser.add_argument("--method", default="cdef", help="output folder name")
    parser.add_argument("--output", default="outputs")
    parser.add_argument("--workers", type=int, default=11)
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
    with Pool(args.workers) as pool:
        for i, _ in enumerate(pool.imap_unordered(worker, jobs, chunksize=2), 1):
            if i % 50 == 0 or i == len(jobs):
                print(f"{i}/{len(jobs)}", flush=True)


if __name__ == "__main__":
    main()
