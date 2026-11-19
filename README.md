# CDEF from scratch

A Python implementation of:

> J. Yuan, Y. Zhang, Z. Cai, "Underwater Scene Enhancement via Adaptive Color Analysis and Multispace Fusion," *IEEE Journal of Oceanic Engineering*, vol. 50, no. 4, pp. 2516–2528, Oct. 2025.

CDEF enhances underwater images without training. It decides whether an image is greenish or bluish, compensates its brightness and colour, makes four differently enhanced versions of it (two contrast stretches, a sharpened one and a dehazed one) and fuses them pixel by pixel with image pyramids in YUV.

The code is in [`cdef_paper/`](cdef_paper). Every stage follows the CDEF paper, and the three methods it uses without settings (BIMEF, CBM and IDE) are implemented from their own papers. The default config switches a few settings to the authors' released code, which brings the scores within about 2% of the paper; [cdef_paper/README.md](cdef_paper/README.md) lists every choice, the ablation and where the paper's equations contradict its text.

## Results

All scores use the metric authors' own code, on every image of each dataset. Raw-image scores match the paper's Table I on all four datasets, so the images and metrics are the same as the paper's. The gap to the paper is in brackets.

| Dataset | UCIQE | FDUM | CEIQ | CCF | URanker |
|---|---|---|---|---|---|
| UIEB (890) | 0.6502 (+0.5%) | 0.8688 (+4.6%) | 3.5697 (+0.4%) | 44.3099 (−1.8%) | 2.3612 (−1.6%) |
| C60 (60) | 0.6331 (+0.3%) | 0.6120 (+4.9%) | 3.4760 (−0.3%) | 36.2678 (+0.6%) | 1.8193 (−0.1%) |
| EUVP (515) | 0.6442 (+1.2%) | 0.7385 (+4.6%) | 3.6156 (+0.2%) | 58.9950 (+2.0%) | 3.2586 (−1.9%) |
| UCCS (300) | 0.6158 (+1.3%) | 0.6673 (+12.6%) | 3.6493 (+0.5%) | 44.4585 (+1.7%) | 2.6008 (+1.0%) |

UCIQE, CEIQ, CCF and URanker are within 2% of the paper on all four datasets. FDUM is about 5% high on three datasets and 12.6% high on UCCS; the authors' released code, run unchanged, is also about 6% high on FDUM.

## Setup

Python 3.10 or newer.

```bash
python -m pip install -r requirements.txt
```

Datasets go in `data/`; see [data/README.md](data/README.md). Scoring needs MATLAB and the metric code in `metrics/third_party/`; see [metrics/README.md](metrics/README.md).

## Usage

Enhance one image or a folder:

```bash
python -m cdef_paper --input data/UIEB/raw --output outputs/UIEB
```

From Python:

```python
from cdef_paper import enhance, load_config
from cdef_paper.image_io import read_image, write_image

output, debug = enhance(read_image("input.png"), load_config())
write_image("output.png", output)
```

`debug` holds every intermediate image (`luminance`, `compensated`, `F1` to `F4`, `weights`) and the values `tone`, `alpha`, `beta`, `k` and `tmin`.

Enhance whole datasets in parallel, then score them. Each step saves as it goes and skips finished work, so it can be stopped and rerun. `--sample` uses a fixed subset of 60 UIEB and 20 C60 images.

```bash
python -m cdef_paper.run_dataset --dataset UIEB C60 EUVP UCCS --workers 8
```

```bash
python -m metrics.score_all --methods cdef_paper --datasets UIEB C60 EUVP UCCS
```

```bash
python -m metrics.score_uranker --methods cdef_paper --datasets UIEB C60 EUVP UCCS
```

An ablation row is the same with its config, for example `--config cdef_paper/ablation/wo_ide.yaml --method paper_abl_wo_ide`.

Run the tests:

```bash
python -m pytest
```

## Layout

| Path | Contents |
|---|---|
| `cdef_paper/` | The implementation, its configs and ablation configs; see its README |
| `metrics/` | Runners for the metric authors' code |
| `tests/` | Unit tests |
| `data/` | Datasets (not in git) |
| `docs/` | The CDEF paper and the papers it cites (not in git) |
| `figures/` | Ablation on single images, ours vs the authors' code |
| `outputs/`, `results/scores/` | Enhanced images and per-image scores (not in git) |
