# CDEF from scratch

A Python implementation of:

> J. Yuan, Y. Zhang, Z. Cai, "Underwater Scene Enhancement via Adaptive Color Analysis and Multispace Fusion," *IEEE Journal of Oceanic Engineering*, vol. 50, no. 4, pp. 2516–2528, Oct. 2025.

First written from the papers alone. The authors' released code was then read to explain the remaining gaps, and the main configuration now follows it where that makes the replication closer, as our own implementation of each choice. Every such choice is an option, with the paper's reading kept as the alternative; [docs/deviations.md](docs/deviations.md) lists them. This covers the core method on all four of the paper's datasets (UIEB, C60, EUVP and UCCS) and the ablation study on UIEB. The extensions in [docs/CDEF_PLAN.md](docs/CDEF_PLAN.md) are not done yet.

## Results

The main configuration on all of UIEB (890 images) and C60 (60 images), against the paper's Tables I–II and the authors' released code run on the same images. All scores use the metric authors' code. Details, the reasons behind each setting and the earlier paper-only results are in [docs/deviations.md](docs/deviations.md).

| Full set | UCIQE | FDUM | CEIQ | CCF | URanker |
|---|---|---|---|---|---|
| UIEB, main configuration | 0.6525 | 0.9122 | 3.5760 | 48.2208 | 2.4658 |
| UIEB, authors' code | 0.6455 | 0.8859 | 3.5618 | 46.5553 | 2.4443 |
| UIEB, paper | 0.6468 | 0.8307 | 3.5544 | 45.1072 | 2.3990 |
| UIEB, ours vs paper | +0.9% | +9.8% | +0.6% | +6.9% | +2.8% |
| UIEB, ours vs authors' code | +1.1% | +3.0% | +0.4% | +3.6% | +0.9% |
| C60, main configuration | 0.6367 | 0.6397 | 3.5119 | 38.3852 | 1.9241 |
| C60, authors' code | 0.6293 | 0.6182 | 3.4881 | 36.3923 | 1.8696 |
| C60, paper | 0.6315 | 0.5832 | 3.4867 | 36.0389 | 1.8214 |
| C60, ours vs paper | +0.8% | +9.7% | +0.7% | +6.5% | +5.6% |
| C60, ours vs authors' code | +1.2% | +3.5% | +0.7% | +5.5% | +2.9% |

EUVP (its 515-image test set) and UCCS (300 images), against the paper; the authors' code was not run on these. Their raw-image scores match the paper's to every printed digit, so they are the same images.

| Set | UCIQE | FDUM | CEIQ | CCF | URanker |
|---|---|---|---|---|---|
| EUVP (515), main configuration | 0.6447 | 0.7762 | 3.5689 | 63.1464 | 3.3582 |
| EUVP, paper | 0.6367 | 0.7062 | 3.6076 | 57.8109 | 3.3213 |
| EUVP, ours vs paper | +1.3% | +9.9% | −1.1% | +9.2% | +1.1% |
| UCCS (300), main configuration | 0.6160 | 0.7033 | 3.6317 | 47.4777 | 2.6746 |
| UCCS, paper | 0.6077 | 0.5925 | 3.6328 | 43.7368 | 2.5748 |
| UCCS, ours vs paper | +1.4% | +18.7% | 0.0% | +8.6% | +3.9% |

Ours scores above the paper on every metric except CEIQ on EUVP (−1.1%; UCCS CEIQ matches). Most of that comes from following the paper where the authors' code departs from it, and their code is itself above the paper. Higher scores do not mean better images; see the comparison in [figures/main_version.jpg](figures/main_version.jpg).

### Ablation (Table IV, UIEB)

Each row turns off one stage, or fuses in another colour space, starting from the main configuration. The table gives the change from our full model next to the paper's change from its full model.

| Change vs full model | FDUM ours / paper | CCF ours / paper | URanker ours / paper |
|---|---|---|---|
| w/o LC | −8.7% / −9.7% | −8.7% / −10.8% | −1.8% / −6.6% |
| w/o TM | −19.9% / −15.9% | −25.3% / −17.3% | −21.5% / −19.0% |
| w/o CBM | −6.1% / −2.3% | −2.0% / −1.8% | −3.6% / −3.5% |
| w/o IDE | −3.9% / −15.2% | −3.2% / −23.9% | −1.8% / −17.8% |
| MEF in RGB | −6.8% / −13.6% | −13.1% / −26.4% | −4.3% / −16.1% |
| MEF in Lab | +0.2% / −15.4% | +0.4% / −17.0% | −2.4% / −11.1% |

LC, TM and CBM reproduce. IDE and RGB fusion matter less here than in the paper, and Lab fusion does not hurt at all. The paper does not say how each stage was removed, or what fusion weights it used in RGB and Lab. Full numbers and the possible reasons are in [docs/deviations.md](docs/deviations.md). Outputs are in `outputs/ablation_*/` and scores in `results/scores/ablation_*_UIEB.csv`.

Outputs of the current configuration are in `outputs/cdef_main_v3/` and its scores in `results/scores/cdef_main_v3_*.csv`. Earlier runs were deleted; their scores are recorded in [docs/deviations.md](docs/deviations.md). The run commands skip finished images, so give a run with changed settings a new `--method` name.

## Setup

Python 3.10 or newer.

```bash
python -m pip install -r requirements.txt
```

Datasets go in `data/`. See [data/README.md](data/README.md).

## Usage

Enhance one image or a whole folder:

```bash
python -m cdef --config configs/cdef_paper.yaml --input data/UIEB/raw --output outputs/UIEB
```

From Python:

```python
from cdef import enhance, load_config
from cdef.utils import read_image, write_image

output, debug = enhance(read_image("input.png"), load_config("configs/cdef_paper.yaml"))
write_image("output.png", output)
```

`debug` holds every intermediate image (`luminance`, `compensated`, `F1` to `F4`, `weights`) and the values `tone`, `alpha`, `beta`, `k` and `tmin`.

Enhance UIEB and C60, then score them. Both steps save as they go and skip finished work, so they can be stopped and rerun. Scoring needs MATLAB; see [metrics/README.md](metrics/README.md). Drop `--sample` and use a new `--method` name, such as `cdef_main_v2`, for the full sets; add `--dataset EUVP UCCS` (and `--datasets EUVP UCCS` to the scorers) for the other two sets.

```bash
python -m eval.run_dataset --config configs/cdef_paper.yaml --sample --method cdef_main_sample
```

```bash
python -m metrics.score_all --methods cdef_main_sample
```

```bash
python -m metrics.score_uranker --methods cdef_main_sample
```

Run the tests:

```bash
python -m pytest
```

## Layout

| Path | Contents |
|---|---|
| `cdef/color_discrimination.py` | Greenish / bluish test (Eq. 3) |
| `cdef/luminance.py` | Luminance compensation (BIMEF) |
| `cdef/color_comp.py` | Lab compensation (Eqs. 1–2) |
| `cdef/tone_mapping.py` | F1 and F2 (Eqs. 4–6) |
| `cdef/detail/cbm.py` | F3, Contour Bougie Morphology |
| `cdef/detail/ide.py` | F4, Image Dehazing and Exposure |
| `cdef/fusion.py` | Fusion weights and pyramids (Eqs. 7–14), in YUV, YCbCr, RGB or Lab |
| `cdef/pipeline.py` | The full method |
| `eval/run_dataset.py` | Enhance a dataset and save the outputs |
| `metrics/` | Runners for the metric authors' code in `metrics/third_party/` |
| `configs/cdef_paper.yaml` | Every parameter, marked "ours" where the paper gives no value and "code" where it follows the authors' code |
| `configs/ablation/` | One config per row of the paper's ablation study (Table IV) |
| `tests/` | Unit tests (`python -m pytest`) |
| `docs/CDEF_PLAN.md` | The project plan |
| `docs/*.pdf` | The CDEF paper and the papers it builds on (Ju, Yuan, Kou, Mertens, Wang) |
| `docs/cbm_ide_notes.md` | Reading notes on the CBM and IDE papers |
| `docs/deviations.md` | Every gap between the papers and this code, with evidence |
| `outputs/`, `results/scores/` | Enhanced images and per-image scores |
