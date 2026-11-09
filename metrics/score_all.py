"""Score image folders with the authors' MATLAB metrics, using several processes.

Every score is saved as soon as it is computed, under
results/scores/<method>_<dataset>/, so a stopped run resumes. When all images are
done, the scores are joined into results/scores/<method>_<dataset>.csv, and a
method with a finished table is never scored again.

python -m metrics.score_all --methods cdef_sample --processes 6
"""
import argparse
import csv
import subprocess
from pathlib import Path

MATLAB = r"C:\Program Files\MATLAB\R2026b\bin\matlab.exe"
METRICS = ("UCIQE", "CEIQ", "FDUM", "CCF")
RAW_FOLDERS = {
    "UIEB": "data/UIEB/raw",
    "C60": "data/UIEB/challenging",
    "EUVP": "data/EUVP/test_samples/Inp",
    "UCCS": "data/UCCS",
}
IMAGE_TYPES = {".png", ".jpg", ".jpeg"}


def list_images(folder):
    """Image files in a folder and its subfolders (UCCS keeps one per colour cast)."""
    return sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in IMAGE_TYPES)


def image_folder(method, dataset):
    return RAW_FOLDERS[dataset] if method == "raw" else f"outputs/{method}/{dataset}"


def start_shard(folder, work_dir, shard, shards):
    command = (
        f"addpath('metrics'); "
        f"score_folder('{folder}', '{work_dir.as_posix()}', '{','.join(METRICS)}', {shard}, {shards})"
    )
    return subprocess.Popen([MATLAB, "-singleCompThread", "-batch", command])


def read_scores(work_dir, metric):
    scores = {}
    for part in sorted(work_dir.glob(f"{metric}_shard*.csv")):
        with open(part, newline="") as f:
            scores.update({image: value for image, value in csv.reader(f)})
    return scores


def join_scores(work_dir, images, out_csv):
    """Write the final table, or return False if any score is still missing."""
    scores = {metric: read_scores(work_dir, metric) for metric in METRICS}
    if any(image not in scores[m] for m in METRICS for image in images):
        return False
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(("image",) + METRICS)
        for image in sorted(images):
            writer.writerow([image] + [scores[m][image] for m in METRICS])
    return True


def score(method, dataset, processes, out_dir):
    out_csv = out_dir / f"{method}_{dataset}.csv"
    if out_csv.exists():
        return
    folder = image_folder(method, dataset)
    images = [p.name for p in list_images(folder)]
    if not images:
        raise FileNotFoundError(f"no images in {folder}")
    work_dir = out_dir / f"{method}_{dataset}"
    work_dir.mkdir(parents=True, exist_ok=True)

    jobs = [start_shard(folder, work_dir, i, processes) for i in range(1, processes + 1)]
    if any(job.wait() != 0 for job in jobs):
        raise RuntimeError(f"a MATLAB process failed on {folder}")
    if not join_scores(work_dir, images, out_csv):
        raise RuntimeError(f"scores are missing for {folder}")
    print(f"done {method} {dataset}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", nargs="+", required=True)
    parser.add_argument("--datasets", nargs="+", default=["UIEB", "C60"])
    parser.add_argument("--processes", type=int, default=6)
    parser.add_argument("--output", default="results/scores")
    args = parser.parse_args()
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    for method in args.methods:
        for dataset in args.datasets:
            score(method, dataset, args.processes, out_dir)


if __name__ == "__main__":
    main()
