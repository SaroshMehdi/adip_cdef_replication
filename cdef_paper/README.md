# CDEF, paper version

A second implementation of CDEF, built to reproduce the scores printed in the paper:

> J. Yuan, Y. Zhang, Z. Cai, "Underwater Scene Enhancement via Adaptive Color Analysis and Multispace Fusion," *IEEE Journal of Oceanic Engineering*, vol. 50, no. 4, pp. 2516–2528, Oct. 2025.

The main version in [`cdef/`](../cdef) takes several choices from the authors' released code and scores above the paper. This folder holds two settings files:

- [`paper_only.yaml`](paper_only.yaml) takes every value from the CDEF paper, and implements the three methods it uses without settings (BIMEF, CBM and IDE) from their own papers. On a fixed 80-image sample it scores well below the paper: about 25% on CCF and 9% on FDUM and URanker.
- [`config.yaml`](config.yaml), the default, starts from `paper_only.yaml` and changes a few settings to the authors' released code. It is the mix whose scores are closest to the paper's Tables I, II and IV.

## Results

All scores use the metric authors' own code, on every image of each dataset. The gap to the paper is in brackets.

| Dataset | UCIQE | FDUM | CEIQ | CCF | URanker |
|---|---|---|---|---|---|
| UIEB (890) | 0.6502 (+0.5%) | 0.8688 (+4.6%) | 3.5697 (+0.4%) | 44.3099 (−1.8%) | 2.3612 (−1.6%) |
| C60 (60) | 0.6331 (+0.3%) | 0.6120 (+4.9%) | 3.4760 (−0.3%) | 36.2678 (+0.6%) | 1.8193 (−0.1%) |
| EUVP (515) | 0.6442 (+1.2%) | 0.7385 (+4.6%) | 3.6156 (+0.2%) | 58.9950 (+2.0%) | 3.2586 (−1.9%) |
| UCCS (300) | 0.6158 (+1.3%) | 0.6673 (+12.6%) | 3.6493 (+0.5%) | 44.4585 (+1.7%) | 2.6008 (+1.0%) |

UCIQE, CEIQ, CCF and URanker are within 2% of the paper on all four datasets. FDUM is about 5% high on three datasets and 12.6% high on UCCS. The authors' released code, run unchanged, is also about 6% high on FDUM (UIEB 0.8859, C60 0.6182), and no setting tried here lowered FDUM without lowering CCF by a similar amount.

### Ablation (Table IV, UIEB, 890 images)

Each cell is the change from the full model, ours / paper.

| Row | UCIQE | FDUM | CEIQ | CCF | URanker |
|---|---|---|---|---|---|
| w/o LC | −1.7% / −1.3% | −8.8% / −9.7% | −0.3% / −0.2% | −8.9% / −10.8% | −2.0% / −6.6% |
| w/o TM | −11.8% / −6.0% | −22.9% / −15.9% | −6.6% / −4.6% | −30.8% / −17.3% | −26.8% / −19.0% |
| w/o CBM | −0.0% / +0.3% | −6.1% / −2.3% | −0.9% / −0.4% | −3.4% / −1.8% | −3.1% / −3.5% |
| w/o IDE | −2.0% / −4.0% | −8.7% / −15.2% | +0.6% / −0.7% | −8.9% / −23.9% | −3.7% / −17.8% |
| MEF in RGB | −1.7% / −3.7% | −9.0% / −13.6% | −0.9% / −3.2% | −15.6% / −26.4% | −5.7% / −16.1% |
| MEF in Lab | −1.9% / −3.4% | −8.9% / −15.4% | −0.5% / −1.4% | −7.9% / −17.0% | −3.0% / −11.1% |

Every row moves in the paper's direction on FDUM, CCF and URanker. LC and CBM are close to the paper. TM drops more than the paper's, and IDE, RGB and Lab drop about half as much. The paper does not say how it removed a stage; the readings used are:

| Row | Reading (configs in [`ablation/`](ablation)) |
|---|---|
| w/o LC | BIMEF skipped |
| w/o TM | F1 = F2 = the compensated image |
| w/o CBM | F3 left out; three inputs are fused |
| w/o IDE | F4 replaced by the tone-mapped image it would have dehazed |
| MEF in RGB | Mertens et al.'s weights on RGB |
| MEF in Lab | The YUV weights on CIELab as it is (L in 0–100) |

## What the default config takes from the authors' code

| Setting | Papers only | Default |
|---|---|---|
| BIMEF | λ = 1, exact solve at full size | Its released code: λ = 0.5, illumination solved at half size with an approximate conjugate-gradient solve |
| IDE | The IDE paper's Algorithm 1 | The code shipped with CDEF: ambient light from a 10 × 10 local minimum, t_min searched at full size, box-smoothed t |
| CBM output | Clipped to [0, 1] | Not clipped |
| Fusion measures (Eqs. 7–9) | On values in [0, 1] | On 8-bit values, the range Eq. 6 maps to |
| Saliency (Eq. 9) | No blur | 3 × 3 Gaussian first |
| Brightness weight | None | Each input's weights scaled by mean(Y)² (Wang et al. [33]) |

The fusion keeps the paper's ε = 0.1, exponents of 1, YUV and pyramid depth. The authors' code squares the contrast and saliency weights and uses ε = 1e-12; those are not used here.

## Usage

Run from the project root. Enhance one image or a folder:

```bash
python -m cdef_paper --input data/UIEB/raw --output outputs/UIEB_paper
```

Enhance whole datasets into `outputs/cdef_paper/<dataset>/`, then score them with the same metric scripts as the main version:

```bash
python -m cdef_paper.run_dataset --dataset UIEB C60 EUVP UCCS --workers 8
```

Add `--config cdef_paper/paper_only.yaml` for the papers-only settings, or `--config cdef_paper/ablation/wo_ide.yaml --method paper_abl_wo_ide` for an ablation row.

```bash
python -m metrics.score_all --methods cdef_paper --datasets UIEB C60 EUVP UCCS
```

```bash
python -m metrics.score_uranker --methods cdef_paper --datasets UIEB C60 EUVP UCCS
```

With `paper_only.yaml` the luminance stage reserves about 4.5 GB per megapixel; `--memory-gb` limits how many workers run together.

## Papers-only settings: what is as printed

This section and the next two describe `paper_only.yaml`.

| Stage | Source | Values |
|---|---|---|
| Tone test | CDEF Eq. 3, Algorithm 1 | Greenish if mean G > mean B: α = 0.5, β = 0.15. Otherwise bluish: α = 1.0, β = 0.3 |
| Luminance compensation | BIMEF paper, Eqs. 4–21 and Sec. V | λ = 1, ε = 0.001, μ = 0.5, window 5, a = −0.3293, b = 1.1258, T < 0.5 is under-exposed, k searched on a 50 × 50 copy, T solved at full resolution |
| Colour compensation | CDEF Eqs. 1–2 | Mask threshold 0.85, on a\* and b\* of CIELab |
| Tone maps F1, F2 | CDEF Eqs. 4–6 | a = 2, b = 0.01, output range 0 to 255 |
| CBM (F3) | CBM paper, Eqs. 5–11 | Hollow squares 15 × 15 and 5 × 5, ω = 0.1, n = 10 |
| IDE (F4) | IDE paper, Algorithm 1 | p1 = −0.397, p2 = 0.07747, patch 75 × 75, ε = 2%, δ = 4, ambient light from the dark channel, guided filter, golden-section search |
| Fusion | CDEF Eqs. 7–14 | YUV, Ws = \|U\| + \|V\| + 1, saliency without blur, all exponents 1, ε = 0.1, n = ⌊log2 min(h, w)⌋ levels, Eq. 14 at every level |

## Where the printed equation contradicts the text

In four places the paper's equation and the sentence beside it disagree. The sentence is followed.

| Equation | Printed | Text | Used |
|---|---|---|---|
| Eq. 4 | σ is called the "variance" | Limits are μ ± aσ | Standard deviation, which has the units of the mean |
| Eq. 7 | Kernel [0 1 0; −1 4 −1; 0 −1 0] | "A Laplacian filter" | [0 −1 0; −1 4 −1; 0 −1 0]. The printed entries sum to 2, so it would respond to flat regions |
| Eq. 10 | ϕ = 2, σ = 0.5 | "A well-exposed pixel tends to have a value of 0.5 [34]" | ϕ = 0.5, σ = 0.2, the values of Mertens et al. [34] |
| Eq. 11 | (W_k + ε) / (Σ W_j + ε) | "To ensure the final weight in fusion is 1" | (W_k + ε) / Σ (W_j + ε), which sums to 1 |

## Gaps: what no paper gives

| Gap | Choice | Reason |
|---|---|---|
| Gaussian G of Eq. 1 (CDEF gives no size) | σ = 1 pixel | The authors' code, the only source. It fits Fig. 4, where α = β = 1 gives a nearly grey image |
| Eq. 1 applies G to the channel; the text says G smooths the mask | Both are smoothed | Satisfies both; the difference is negligible |
| Eq. 3 when mean G = mean B | Bluish | The else branch of Algorithm 1 |
| Lab and YUV definitions | sRGB with D65 white; BT.601 YUV | The standard definitions |
| Eq. 5 "gray average value" | Mean over all three channels | Gray-world assumption |
| Eq. 5 quantile level outside [0, 1]; Eq. 6 with an empty range | Level clamped; channel left unchanged | Avoids division by zero |
| Eq. 13 "5 × 5 Gaussian kernel" | Binomial [1 4 6 4 1] / 16 along both axes | The standard pyramid kernel |
| BIMEF: range of the exposure ratio k | [1, 7] | The BIMEF paper gives none and a bounded search needs one. This is the limit in BIMEF's released code |
| BIMEF: image with no under-exposed pixel | Left unchanged (k = 1) | Nothing to compensate |
| BIMEF: output range | Clipped to [0, 1] | It is an image |
| BIMEF: solver | Exact sparse solve of Eq. 10 | The paper's conjugate-gradient solver approximates the same solution |
| CBM: meaning of "transformed n times" | Each filter applied to its own previous output | The element sizes are fixed |
| CBM: adaptive contrast stretch (second half of the CBM paper) | Not used | CDEF cites CBM for contours only; F2 is already that stretch |
| CBM: values pushed outside the image range | Clipped to [0, 1] (`cbm: clip`) | F3 is an image |
| IDE: dark-channel settings | 15 × 15 patch, brightest of the top 0.1% | He et al., which the IDE paper cites for this step |
| IDE: guided filter | Grey guide, radius 37 (half a patch), ε = 0.001 | Not given |
| IDE: search range of t_min | [0.05, 0.95] | Not given |
| IDE: Eq. 22 | argmin of \|clipped share − 2%\| | The printed signed form has no minimum |
| IDE: range of t | [t_min, 1] | Eq. 17 gives t above 1 in dark patches |
| IDE: Eq. 20 divides by ln t | Divides by f(t) of Eq. 13 (`ide: recovery`) | t was solved with f(t) in place of ln t. f(1) = −0.368 but ln 1 = 0, so the printed form divides by almost zero in dark patches |
| IDE: second Φ of Eq. 20 | Â and Î swapped back | Inverting Eq. 8 gives this order |
| How Table IV removes a stage | See the ablation readings above | Not described in the paper |

Every choice is a value in [`paper_only.yaml`](paper_only.yaml), marked with its source.

## Layout

| File | Contents |
|---|---|
| `pipeline.py` | Algorithm 1 and the fusion; config loading |
| `discrimination.py` | Greenish / bluish test (Eq. 3) |
| `luminance.py` | BIMEF from its paper |
| `luminance_code.py`, `resize.py` | BIMEF as in its released code (copies of the main version's files) |
| `color_compensation.py` | Eqs. 1–2 |
| `tone_mapping.py` | F1 and F2 (Eqs. 4–6) |
| `cbm.py` | F3 from the CBM paper |
| `ide.py` | F4 from the IDE paper |
| `ide_code.py` | IDE as in the code shipped with CDEF (copy of the main version's file) |
| `fusion.py` | Eqs. 7–14 |
| `color_spaces.py`, `image_io.py` | Lab and YUV conversions, image files |
| `paper_only.yaml` | Every papers-only parameter and where it comes from |
| `config.yaml` | The default: `paper_only.yaml` with the changes above |
| `ablation/` | One config per row of Table IV |
| `run_dataset.py` | Enhance a dataset in parallel |
| `../tests/test_paper_version.py` | Tests |
