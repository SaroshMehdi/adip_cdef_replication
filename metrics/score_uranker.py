"""Score image folders with the official URanker and its pretrained weights.

Each score is the same computation as ranker_demo.py in metrics/third_party, and
is saved to results/scores/<method>_<dataset>/URanker.csv as soon as it is
computed, so a stopped run resumes. When all images are done, a URanker column
is added to results/scores/<method>_<dataset>.csv.

python -m metrics.score_uranker --methods cdef_sample
"""
import argparse
import csv
import os
import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from metrics.score_all import image_folder, list_images

REPO = Path(__file__).parent / "third_party" / "UnderwaterRanker"
sys.path.insert(0, str(REPO))

import utils  # noqa: E402  (the URanker repo's utils.py)


def load_model():
    options = utils.get_option(str(REPO / "options" / "URanker.yaml"))
    options["model"]["resume_ckpt_path"] = str(REPO / "checkpoints" / "URanker_ckpt.pth")
    options["model"]["cuda"] = torch.cuda.is_available()
    # build_model imports all_model from the repo root.
    cwd = os.getcwd()
    os.chdir(REPO)
    model = utils.build_model(options["model"])
    os.chdir(cwd)
    return model


def score_image(model, path):
    """The body of the loop in ranker_demo.py, with gradients off to save memory."""
    img = Image.open(path)
    img_w, img_h = img.size[0], img.size[1]
    img = transforms.Resize((img_h // 2, img_w // 2))(img)
    img = transforms.ToTensor()(img).unsqueeze(0)
    if torch.cuda.is_available():
        img = img.cuda()
    with torch.no_grad():
        inputs = utils.preprocessing(img)
        return float(model(**inputs)["final_result"][0][0][0])


def read_scores(path):
    if not path.exists():
        return {}
    with open(path, newline="") as f:
        return {image: value for image, value in csv.reader(f)}


def add_column(table_path, scores):
    with open(table_path, newline="") as f:
        rows = list(csv.reader(f))
    header, body = rows[0], rows[1:]
    if "URanker" in header:
        return
    with open(table_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header + ["URanker"])
        writer.writerows(row + [scores[row[0]]] for row in body)


def score(model, method, dataset, out_dir):
    folder = Path(image_folder(method, dataset))
    work_dir = out_dir / f"{method}_{dataset}"
    work_dir.mkdir(parents=True, exist_ok=True)
    out_csv = work_dir / "URanker.csv"
    scores = read_scores(out_csv)
    images = [p for p in list_images(folder) if p.name not in scores]
    for i, path in enumerate(images, 1):
        value = f"{score_image(model, path):.10g}"
        with open(out_csv, "a", newline="") as f:
            f.write(f"{path.name},{value}\n")
        if i % 100 == 0:
            print(f"{method} {dataset}: {i}/{len(images)}", flush=True)
    add_column(out_dir / f"{method}_{dataset}.csv", read_scores(out_csv))
    print(f"done {method} {dataset}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", nargs="+", required=True)
    parser.add_argument("--datasets", nargs="+", default=["UIEB", "C60"])
    parser.add_argument("--output", default="results/scores")
    args = parser.parse_args()
    model = load_model()
    for method in args.methods:
        for dataset in args.datasets:
            score(model, method, dataset, Path(args.output))


if __name__ == "__main__":
    main()
