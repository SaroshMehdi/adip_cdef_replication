# CDEF Replication + Extensions — Implementation Plan

Plan for implementing, from scratch in Python, the paper:

> J. Yuan, Y. Zhang, Z. Cai, "Underwater Scene Enhancement via Adaptive Color Analysis and Multispace Fusion," *IEEE Journal of Oceanic Engineering*, vol. 50, no. 4, pp. 2516–2528, Oct. 2025. DOI: 10.1109/JOE.2025.3591405

Course: Applied Digital Image Processing (CS/CE 446/452), Fall 2026. Solo project, 40% of the grade.

---

## 0. Ground rules

- **Implement everything from scratch.** The authors released code at https://bilityniu.github.io/CDEF. Do **not** copy or port it. You may only use it to *compare outputs and numbers* after our version works.
- Baseline methods (Fusion, MLLE, WWPF) and learned metrics (URanker, LPIPS) may use official code or weights. Only CDEF itself must be written from scratch.
- Every place where the paper is unclear or looks wrong must be:
  1. made a config option,
  2. given a sensible default, and
  3. listed in `docs/deviations.md` with the reason.
- Keep code simple and readable. One function per paper equation where possible, with the equation number in the docstring.

---

## 1. Environment and project layout

**Stack:** Python 3.10+, `numpy`, `opencv-python`, `scikit-image`, `scipy`, `pyyaml`, `pandas`, `matplotlib`, `tqdm`, `pytest`. Use `torch` only for URanker and LPIPS. Use `bm3d` (pip) for extension 2.

```
cdef/
├── configs/
│   ├── cdef_paper.yaml        # paper settings (replication)
│   └── cdef_ext.yaml          # extension settings
├── cdef/
│   ├── __init__.py
│   ├── color_discrimination.py  # greenish / bluish test, alpha/beta
│   ├── luminance.py             # LC: light compensation (Ying et al. 2017)
│   ├── color_comp.py            # Lab compensation, Eqs. 1–3
│   ├── tone_mapping.py          # Eqs. 4–6
│   ├── detail/
│   │   ├── cbm.py               # Contour Bougie Morphology
│   │   └── ide.py               # Image Dehazing and Exposure
│   ├── fusion.py                # weights Eqs. 7–11, pyramids Eqs. 12–14
│   ├── pipeline.py              # full CDEF (Algorithm 1 + MEF)
│   └── ext/
│       ├── adaptive_params.py   # Extension 1
│       └── noise_aware.py       # Extension 2
├── metrics/
│   ├── uciqe.py  ceiq.py  fdum.py  ccf.py  uranker.py
│   └── fullref.py               # PSNR, SSIM, LPIPS (Extension 3)
├── eval/
│   ├── run_dataset.py           # enhance a whole dataset → outputs/
│   ├── score.py                 # outputs/ → results/*.csv
│   ├── ablation.py              # Table IV
│   ├── downstream.py            # Extension 3 keypoint test
│   └── make_tables.py           # CSV → markdown/LaTeX tables
├── tests/                       # pytest unit tests per module
├── docs/deviations.md
├── data/                        # not committed
├── outputs/  results/  figures/
└── README.md
```

CLI entry points:
```
python -m eval.run_dataset --config configs/cdef_paper.yaml --dataset UIEB
python -m eval.score --method cdef_paper --dataset UIEB
```

---

## 2. Data

| Dataset | Use | Notes |
|---|---|---|
| UIEB (890 raw + references) | Main replication, full-reference tests | Paper calls it "UIEB" |
| UIEB C60 (60 challenging, no refs) | Replication (hard cases) | Paper calls it "C60" |
| EUVP | Replication | Paper does not say which subset → use the test set, record choice in `deviations.md` |
| UCCS | Replication | Split into blue / green / blue-green subsets |
| LOL-v2 (real) | Extension 2 & 3 | Low-light, has references |

Write a small `data/README.md` with download links and the expected folder structure. Images are processed at original resolution (paper does this).

---

## 3. CDEF replication — module by module

All images are float32 in [0, 1] inside the pipeline unless stated. Convert to 8-bit only at the end.

### 3.1 Color discrimination (Algorithm 1, lines 5–9; Eq. 3)
- Compute mean of G and B channels of the **raw** image.
- If `mean_G > mean_B` → greenish: `alpha = 0.5, beta = 0.15`.
- Else → bluish: `alpha = 1.0, beta = 0.3`.
- Return `(tone, alpha, beta)`.

### 3.2 Luminance compensation, LC (Algorithm 1, line 2)
- Paper uses "a simple light enhancement method" [36] = Ying, Li, Gao, *A Bio-Inspired Multi-Exposure Fusion Framework for Low-light Image Enhancement* (BIMEF), arXiv:1711.00591.
- Implement BIMEF from that paper:
  1. Illumination map T = max over RGB, refined (smoothing as in the paper).
  2. Camera response model (Beta–Gamma, `a = -0.3293`, `b = 1.1258`).
  3. Choose exposure ratio k by maximizing entropy of under-exposed pixels.
  4. Fuse original and synthetic exposure with weight `W = T^mu`, `mu = 0.5`.
- Unit test: output range [0, 1], brighter mean than input on a dark image, unchanged-ish on a bright image.

### 3.3 Color compensation in Lab (Eqs. 1–2)
- Convert LC output to CIELab with `skimage.color.rgb2lab` (float Lab, a*/b* roughly −100…+100). Do not use OpenCV's 8-bit Lab (offset 128).
- Mask: `M = 0` where per-pixel mean intensity `> 0.85`, else `M = 1` (Eq. 2). Smooth with the same Gaussian as below.
- `a_c = a − alpha · M · G(a)`, `b_c = b − beta · M · G(b)` (Eq. 1). `G` = Gaussian blur.
- **Unclear:** Gaussian sigma is not given. Default: sigma = 0.1 × min(H, W) (large local mean, close to 3C). Make it a config option. Record in `deviations.md`.
- Convert back to RGB, clip to [0, 1].

### 3.4 Tone mapping (Eqs. 4–6)
Two contrast-stretched images, both always computed (they are fusion inputs F1, F2):
- **F1 – statistic-based (Eq. 4):** per channel, `Imin = mu − a·sigma`, `Imax = mu + a·sigma`, `a = 2`. Paper says "variance" but means standard deviation — use std, note in `deviations.md`.
- **F2 – gray-world quantile (Eq. 5):** per channel c, `Imin = Q(b · gray_mean / mean_c)`, `Imax = Q(1 − b · gray_mean / mean_c)`, `b = 0.01`, Q = quantile of that channel. Clamp the quantile level to [0, 1].
- **Mapping (Eq. 6):** `(I − Imin) · (Omax − Omin) / (Imax − Imin)`, `Omin = 0`, `Omax = 1` (255 in 8-bit). Clip.

### 3.5 Detail inputs F3, F4 (Algorithm 1, lines 15–22)
- Greenish → apply both to F1. Bluish → apply both to F2.
- **F3 = CBM(·)** — Contour Bougie Morphology: Yuan, Cao, Cai, Su, *IEEE TGRS* 59(10), 2021, DOI 10.1109/TGRS.2020.3033407.
- **F4 = IDE(·)** — Ju et al., *IDE: Image Dehazing and Exposure Using an Enhanced Atmospheric Scattering Model*, *IEEE TIP* 30, 2021.
- **Research task first:** read both papers and write a short summary in `docs/cbm_ide_notes.md` before coding.
- If either paper cannot be fully reproduced in reasonable time, implement a documented simplified version (e.g. CBM → multi-scale morphological top-hat/black-hat contour boost; IDE → dark-channel-style dehazing with exposure correction) and record it in `deviations.md`. Keep the interface identical so the real version can be dropped in later.

### 3.6 Multiscale exposure fusion in YUV (Eqs. 7–14)
Convert F1–F4 to YUV (BT.601). For each input k compute:
- **Contrast (Eq. 7):** `Wc = |Laplacian(Y)|`. Paper prints kernel `[0,1,0; −1,4,−1; 0,−1,0]` — this is a typo; use standard `[0,−1,0; −1,4,−1; 0,−1,0]`.
- **Saturation (Eq. 8):** `Ws = |U| + |V| + 1`.
- **Saliency (Eq. 9):** `Wsal = (Y−mean Y)² + (U−mean U)² + (V−mean V)²`. Option: blur the image first (as in Achanta et al.); default = no blur (as printed).
- **Exposure (Eq. 10):** `We = exp(−(Y − phi)² / (2·sigma²))`.
  - **Known issue:** paper text says well-exposed ≈ 0.5, but sets `phi = 2, sigma = 0.5`. With Y in [0, 1] this rewards the brightest pixels most.
  - Config: `exposure_phi`, `exposure_sigma`. Run **both** `(2, 0.5)` (as printed) and `(0.5, 0.2)` (Mertens). Check the authors' released code to see which they really used, and use that for the replication tables.
- **Combine and normalize (Eq. 11):** `Wk = Wc^wc · Ws^ws · Wsal^wsal · We^we` (all exponents 1), `eps = 0.1`.
  - As printed, `(Wk + eps) / (sum Wk + eps)` does not sum to 1. Default: `(Wk + eps) / sum_j (Wj + eps)` so weights sum to 1. Keep the printed form as an option.
- **Pyramids (Eqs. 12–14):**
  - Levels `n = floor(log2(min(H, W)))`, 5×5 Gaussian kernel.
  - Gaussian pyramid of each weight map, Laplacian pyramid of each input (per Y, U, V channel).
  - `R_level = sum_k G_level(Wk) · L_level(Ik)`, then collapse.
  - Convert YUV → RGB, clip, save.
- Unit tests: weights sum to 1 per pixel; fusing 4 identical images returns that image; pyramid collapse of an unmodified Laplacian pyramid reconstructs the input.

### 3.7 Pipeline
`pipeline.enhance(img, cfg) -> img_out, debug_dict`. The debug dict holds every intermediate image (LC, compensated, F1–F4, weight maps) so figures like the paper's Fig. 2 can be generated.

---

## 4. Metrics

| Metric | Direction | Implementation note |
|---|---|---|
| UCIQE | ↑ | Implement (Yang & Sowmya 2015). Implementations vary — fix one and document it |
| CEIQ | ↑ | Needs the authors' trained regressor (Yan et al. 2019) — use released model/code |
| FDUM | ↑ | Yang et al. 2021, frequency-domain. Port from the original paper/code |
| CCF | ↑ | Wang et al. 2018. Port from the original paper/code |
| URanker | ↑ | Official pretrained weights (Guo et al. AAAI 2023). Do not retrain |
| PSNR / SSIM | ↑ | `skimage.metrics` (Extension 3) |
| LPIPS | ↓ | `lpips` package, AlexNet (Extension 3) |

Sanity check each no-reference metric on a few raw images against values in the paper's "Raw img" row before trusting it.

---

## 5. Replication targets

Paper results for CDEF ("Ours"). Goal: get close (same ranking vs baselines; small absolute gaps are fine and should be discussed).

**Table I (no-reference metrics)**

| Metric | EUVP | UIEB | C60 | UCCS |
|---|---|---|---|---|
| UCIQE | 0.6367 | 0.6468 | 0.6315 | 0.6077 |
| FDUM | 0.7062 | 0.8307 | 0.5832 | 0.5925 |
| CEIQ | 3.6076 | 3.5544 | 3.4867 | 3.6328 |
| CCF | 57.8109 | 45.1072 | 36.0389 | 43.7368 |

**Table II (URanker):** EUVP 3.3213, UIEB 2.3990, C60 1.8214, UCCS 2.5748.

**Raw-image reference row (for metric sanity checks), UIEB:** UCIQE 0.5196, FDUM 0.4470, CEIQ 3.0644, CCF 20.5141, URanker 0.3600.

**Table IV ablation (UIEB):** reproduce rows −w/o LC, −w/o TM, −w/o CBM, −w/o IDE, MEF in RGB, MEF in Lab, Full. Full model: UCIQE 0.6468, FDUM 0.8307, CEIQ 3.5544, CCF 45.1072, URanker 2.3990.

**Runtime:** paper reports 0.723 h for all 890 UIEB images (≈ 2.9 s/image, MATLAB, i7-12700KF). Report ours with hardware details.

**Baselines to compare against:** Fusion (Ancuti 2018), MLLE (TIP 2022), WWPF (TCSVT 2024), using official code/outputs.

---

## 6. Extension 1 — Continuous adaptive compensation

**Problem:** CDEF uses a hard switch (`mean_G > mean_B`) and fixed alpha/beta. The paper itself lists fixed parameters as its main limitation. Images near the boundary can flip between very different outputs.

**Method:**
1. Cast direction: `d = (mean_G − mean_B) / (mean_G + mean_B + 1e-6)` in [−1, 1].
2. Soft label: `t = sigmoid(−d / tau)` (t → 1 bluish, t → 0 greenish), `tau` default 0.05.
3. Interpolate: `alpha = 0.5 + 0.5·t`, `beta = 0.15 + 0.15·t` (matches paper values at the extremes).
4. Optional severity scaling: multiply by `s = clip(k / k_ref, 0.5, 1.5)`, where k = colour-bias factor D/M (equivalent-circle method, Xu et al. 2008; also used in Zhou et al. IJCV 2025, Eqs. 15–19).
5. Tone-map / detail selection: instead of choosing F1 or F2 for CBM/IDE, use the blended base `t·F2 + (1−t)·F1`.

**Experiments:**
- Full datasets: all Table I/II metrics, CDEF vs CDEF+Ext1.
- **Borderline subset:** images with `|d| < 0.05` — report metrics separately here (where the benefit should show).
- **Stability test:** apply small hue/white-balance shifts (±2–5%) to borderline images; measure output change (mean ΔE in Lab) for original vs Ext1. Plot output ΔE vs input shift.
- Plot: metric vs d for both versions (shows the jump in the original).

---

## 7. Extension 2 — Noise-aware fusion

**Problem:** LC and contrast stretching amplify sensor noise in dark regions (C60, LOL-v2). CDEF has no noise handling.

**Method:**
1. Estimate noise sigma on the Y channel of the LC output with Immerkær's fast method.
2. **Option A – pre-denoise:** denoise Y of the compensated image before tone mapping. Strength scales with estimated sigma and a darkness mask (dark pixels denoised more). Denoisers: guided filter (fast default), BM3D (quality).
3. **Option B – noise weight:** add a 5th weight term `Wn` that lowers the weight of inputs with high local high-frequency energy in dark regions, e.g. `Wn = exp(−local_HF_energy / (c · sigma²))`, applied mainly at the finest pyramid levels.
4. Implement both, compare, keep the better as the final extension (report both).

**Experiments:**
- **Synthetic:** add Gaussian and Poisson–Gaussian noise (sigma ∈ {5, 10, 15, 25}/255) to UIEB raw images; score against UIEB references with PSNR/SSIM/LPIPS.
- **Real low-light:** LOL-v2 real (has references) — PSNR/SSIM/LPIPS.
- **C60:** no-reference metrics + estimated output noise sigma.
- Check there is no loss on clean images (UIEB without added noise).
- Runtime overhead per image.

---

## 8. Extension 3 — Stronger evaluation

**Problem:** the paper uses only no-reference metrics (and shows they can reward failed outputs). No full-reference or task-based results.

**Work:**
1. **Full-reference:** PSNR, SSIM, LPIPS vs UIEB references for CDEF, CDEF+Ext1, CDEF+Ext2, baselines. Also LOL-v2. Note UIEB references are themselves chosen from enhanced outputs — mention as a limitation.
2. **Downstream keypoint test:**
   - Detect SIFT (and ORB) keypoints on raw vs enhanced images; report keypoint count.
   - Repeatability: warp each enhanced image with a random homography + mild blur, match back to the unwarped image, report RANSAC inlier ratio and matching accuracy.
   - Optional (only if time): pretrained detector on a labelled underwater set (e.g. RUOD) — mAP on raw vs enhanced.
3. **Metric agreement:** Spearman correlation between metrics across methods (shows where UCIQE/FDUM disagree with URanker and full-reference scores).
4. **Exposure-weight check:** report results for `phi/sigma = (2, 0.5)` vs `(0.5, 0.2)` (from §3.6).

---

## 9. Milestones (do in this order)

1. Setup: repo, configs, data loaders, metric implementations + sanity checks on raw images.
2. Core modules: color discrimination, LC, Lab compensation, tone mapping, YUV fusion (with unit tests).
3. CBM + IDE: read papers → notes → implement (or documented simplified versions).
4. Full pipeline, run all 4 datasets, Tables I/II, compare with authors' released outputs, fill `deviations.md`.
5. Ablation (Table IV) + runtime.
6. Extension 1.
7. Extension 2.
8. Extension 3 (full-reference, keypoints, metric agreement).
9. Figures (pipeline intermediates, visual comparisons, failure cases) and final tables.

Checkpoint after step 4: replication must be working before starting extensions.

---

## 10. Deliverables

- Clean repo with README (setup, data, how to reproduce every table).
- `results/*.csv` + generated markdown/LaTeX tables.
- `figures/`: pipeline diagram with real intermediates, visual comparisons, borderline/stability plots (Ext 1), noise plots (Ext 2), keypoint results (Ext 3).
- `docs/deviations.md`: every gap between the paper and our implementation, with reasons.
- Short summary of replication results: what matched, what did not, and likely causes.
