# CDEF from scratch

A Python implementation of:

> J. Yuan, Y. Zhang, Z. Cai, "Underwater Scene Enhancement via Adaptive Color Analysis and Multispace Fusion," *IEEE Journal of Oceanic Engineering*, vol. 50, no. 4, pp. 2516–2528, Oct. 2025.

CDEF enhances underwater images without training. It decides whether an image is greenish or bluish, compensates its brightness and colour, makes four differently enhanced versions of it (two contrast stretches, a sharpened one and a dehazed one) and fuses them pixel by pixel with image pyramids in YUV.

The implementation was first written from the papers alone, then the authors' released code was read to explain the remaining gaps. Their choices were re-implemented where they bring the replication closer, and the paper's own rule was kept wherever it changed no score by more than 1%. [docs/implementation_notes.md](docs/implementation_notes.md) lists every departure from the paper with its reason and measured effect. [docs/CDEF_method_explained.pdf](docs/CDEF_method_explained.pdf) explains how the method works and why each step is there.

## Results

All scores use the metric authors' own code. Raw-image scores match the paper's Table I on all four datasets (to every printed digit on UCIQE, FDUM and CEIQ), so the images and metrics are the same as the paper's.

| Full set | UCIQE | FDUM | CEIQ | CCF | URanker |
|---|---|---|---|---|---|
| UIEB (890), ours | 0.6525 | 0.9122 | 3.5760 | 48.2208 | 2.4658 |
| UIEB, authors' code | 0.6455 | 0.8859 | 3.5618 | 46.5553 | 2.4443 |
| UIEB, paper | 0.6468 | 0.8307 | 3.5544 | 45.1072 | 2.3990 |
| C60 (60), ours | 0.6367 | 0.6397 | 3.5119 | 38.3852 | 1.9241 |
| C60, authors' code | 0.6293 | 0.6182 | 3.4881 | 36.3923 | 1.8696 |
| C60, paper | 0.6315 | 0.5832 | 3.4867 | 36.0389 | 1.8214 |
| EUVP (515), ours | 0.6447 | 0.7762 | 3.5689 | 63.1464 | 3.3582 |
| EUVP, paper | 0.6367 | 0.7062 | 3.6076 | 57.8109 | 3.3213 |
| UCCS (300), ours | 0.6160 | 0.7033 | 3.6317 | 47.4777 | 2.6746 |
| UCCS, paper | 0.6077 | 0.5925 | 3.6328 | 43.7368 | 2.5748 |

Ours is within 1.5% of the paper on UCIQE and CEIQ and above it on FDUM, CCF and URanker. The authors' released code, run unchanged on the same images, is also above the paper. Higher scores do not mean better images: these metrics reward contrast and colourfulness.

### Ablation (Table IV, UIEB)

Each row turns off one stage, or fuses in another colour space. Change from the full model, ours / paper:

| | FDUM | CCF | URanker |
|---|---|---|---|
| w/o LC | −8.7% / −9.7% | −8.7% / −10.8% | −1.8% / −6.6% |
| w/o TM | −19.9% / −15.9% | −25.3% / −17.3% | −21.5% / −19.0% |
| w/o CBM | −6.1% / −2.3% | −2.0% / −1.8% | −3.6% / −3.5% |
| w/o IDE | −3.9% / −15.2% | −3.2% / −23.9% | −1.8% / −17.8% |
| MEF in RGB | −6.8% / −13.6% | −13.1% / −26.4% | −4.3% / −16.1% |
| MEF in Lab | +0.2% / −15.4% | +0.4% / −17.0% | −2.4% / −11.1% |

LC, TM and CBM reproduce. The IDE, RGB and Lab rows do not, with the authors' own code either; see [docs/implementation_notes.md](docs/implementation_notes.md).

## Setup

Python 3.10 or newer.

```bash
python -m pip install -r requirements.txt
```

Datasets go in `data/`; see [data/README.md](data/README.md). Scoring needs MATLAB and the metric code in `metrics/third_party/`; see [metrics/README.md](metrics/README.md).

## Usage

Enhance one image or a folder:

```bash
python -m cdef --input data/UIEB/raw --output outputs/UIEB
```

From Python:

```python
from cdef import enhance, load_config
from cdef.image_io import read_image, write_image

output, debug = enhance(read_image("input.png"), load_config("configs/cdef.yaml"))
write_image("output.png", output)
```

`debug` holds every intermediate image (`luminance`, `compensated`, `F1` to `F4`, `weights`) and the values `tone`, `alpha`, `beta`, `k` and `tmin`.

Enhance a dataset in parallel, then score it. Each step saves as it goes and skips finished work, so it can be stopped and rerun. `--sample` uses a fixed subset of 60 UIEB and 20 C60 images.

```bash
python -m eval.run_dataset --config configs/cdef.yaml --dataset UIEB C60 --method cdef
```

```bash
python -m metrics.score_all --methods cdef --datasets UIEB C60
```

```bash
python -m metrics.score_uranker --methods cdef --datasets UIEB C60
```

An ablation row is the same with its config, for example `--config configs/ablation/wo_ide.yaml --method ablation_wo_ide`.

Run the tests:

```bash
python -m pytest
```

## Layout

| Path | Contents |
|---|---|
| `cdef/pipeline.py` | The full method (Algorithm 1 and the fusion) and config loading |
| `cdef/discrimination.py` | Greenish / bluish test (Eq. 3) |
| `cdef/luminance.py` | Luminance compensation (BIMEF) |
| `cdef/color_compensation.py` | Colour compensation in Lab (Eqs. 1–2) |
| `cdef/tone_mapping.py` | F1 and F2 (Eqs. 4–6) |
| `cdef/cbm.py` | F3, Contour Bougie Morphology |
| `cdef/ide.py` | F4, Image Dehazing and Exposure |
| `cdef/fusion.py` | Fusion weights and pyramids (Eqs. 7–14) |
| `cdef/color_spaces.py`, `cdef/resize.py`, `cdef/image_io.py` | Lab and YUV conversions, the bicubic resize BIMEF uses, image files |
| `configs/cdef.yaml` | Every parameter, marked by where its value comes from |
| `configs/ablation/` | One config per row of Table IV, each changing one setting of `cdef.yaml` |
| `eval/run_dataset.py` | Enhance a dataset and save the outputs |
| `metrics/` | Runners for the metric authors' code |
| `tests/` | Unit tests |
| `docs/implementation_notes.md` | Departures from the paper, results, ablation, comparison with the authors' code |
| `docs/CDEF_method_explained.pdf` | How the method works, step by step |
| `docs/CDEF_PLAN.md` | The original project plan |
| `figures/` | Ablation outputs, ours vs the authors' code, ours vs a UIEB reference, a LOL-v2 example |
| `outputs/`, `results/scores/` | Enhanced images and per-image scores (not in git) |
