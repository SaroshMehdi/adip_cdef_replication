# Implementation notes

What this implementation does where the CDEF paper, or a paper it cites, is unclear or looks wrong, and how the final settings were chosen. It also holds the replication results, the ablation study and the comparison with the authors' released code.

## How the implementation was built

1. **From the papers alone.** Every stage was first written from the CDEF paper and the papers it cites (BIMEF, CBM, IDE, Mertens et al., Wang et al.). That version scored close to the paper on UCIQE, FDUM and CEIQ but about 17% low on CCF.
2. **Guided by the released code.** The authors' MATLAB code (`demo_img.m`, `lib/`, `IDE/`, `EF/`) was then read to explain the gap, and its choices were re-implemented here, not translated line by line. Our implementation of all of the authors' stages, chained end to end, reproduced their final images to under 1 grey level on average and their scores to within 0.1% (UIEB) and 0.8% (C60) on a fixed 80-image sample.
3. **Back to the paper where it costs nothing.** Every place where that version departed from a rule the papers state was set back to the paper's rule, one at a time, on the 80-image sample (60 UIEB, every 15th; 20 C60, every 3rd). A paper rule was kept when, together with those already kept, no score moved by more than 1%. Fusion in YUV was then adopted as well (full-set scores moved by 1% or less).

The code now holds only the final configuration (`configs/cdef.yaml`) and the ablation switches. The alternatives tried along the way are recorded below with their measured effect; they are in the git history before the cleanup commit.

## Settings that depart from the paper

Marks match `configs/cdef.yaml`: **paper**, **cited** (the paper the stage cites), **code** (the authors' released code), **ours**.

### Tone test and colour compensation (Eqs. 1-3)

| Item | Paper | This code | Why |
|---|---|---|---|
| Eq. 3 tie, mean G = mean B | Undefined | Bluish | Algorithm 1's else branch |
| Gaussian G in Eq. 1 | No sigma given | sigma = 1 pixel (code) | Fig. 4 shows alpha = beta = 1 giving a nearly grey image, which needs G(a) close to a, so sigma must be small. A large blur (0.1 x min(H, W) was tried) gives red and magenta halos around dark objects. Results barely change between 0.5 and 2 pixels |
| Mask smoothing | Text says G smooths the mask; Eq. 1 applies it to the channel | Both | Hardly matters (colour difference 11.33 vs 11.34) |
| Eq. 2 mask | Light sources excluded | Kept (paper) | The authors' code builds the mask and then overwrites it with ones. The mask only matters above 0.85, where it protects light sources as the paper intends |
| Lab conversion | Not stated | sRGB/D65, no clipping of values outside [0, 1] | The luminance stage can leave values just above 1; clipping them first shifts hue. MATLAB's conversion, which the authors use, does not clip either |

### Luminance compensation (BIMEF)

The paper uses BIMEF [36] without settings. This code follows BIMEF's released implementation, which the authors' code calls:

| Item | BIMEF paper | This code |
|---|---|---|
| Smoothness weights | Gradient summed over a square window | Summed along a 5-pixel line in the gradient's own direction |
| Border | Not stated | Wraps around when T is solved |
| Resolution of T | Full | Solved at half size, then resized back |
| Smoothing strength lambda | Not fixed | 0.5 |
| Solver | Preconditioned CG | CG preconditioned by a modified incomplete Cholesky factor, stopped at 10% relative residual (an exact solve changes the output by about 0.2 grey levels and is slower) |
| Resizing | Not stated | Antialiased bicubic (Keys, a = -0.5); the under-exposure mask is resized as 8-bit data |
| Exposure ratio range | Not given | k in [1, 7]; k reaches 7 on some images |
| No under-exposed pixels | Not covered | k = 1, image unchanged |
| Output | g(P, k), in [0, 1] | As in the paper (paper). The released code subtracts 0.01 and does not clip; both changed no score by more than 0.4% |

Following the released implementation moved each score by about 1% either way but brought the images much closer to the authors' (from about 4 to under 1 grey level).

### Tone maps F1, F2 (Eqs. 4-6)

| Item | Paper | This code | Why |
|---|---|---|---|
| Eq. 4 | "variance" | Standard deviation | mean +/- 2 sigma needs matching units |
| Eq. 5 grey mean | "Gray average value" | Mean over all three channels | Gray-world assumption |
| Eq. 5 quantile level | Can leave [0, 1] | Clamped; a channel with no usable range is left unchanged | Avoids division by zero |
| F1, F2 | As printed | As printed (paper) | The authors' F1 (a = 2.3, unclipped) and 8-bit F2 lowered FDUM, CCF and URanker by 1-3% |

### CBM (F3)

| Item | CBM paper | This code | Why |
|---|---|---|---|
| Wh_i for i > 1 | "Transformed n times", not defined | Each filter applied to its own previous output | The element sizes are stated as fixed, and "iterations" fits this. (Growing the elements with i was the other reading.) The open-close filter does not change on a second pass, so the same detail is added n times |
| Output clipping | Not stated | Not clipped (code) | Edge overshoot reaches the fusion, as in the authors' pipeline. On the sample this alone cut the CCF gap to the authors' outputs from 19.8% to 11.6% |
| Adaptive contrast stretch (second stage of the CBM paper) | Part of CBM | Not used | CDEF cites CBM only for contours, and F2 is already that stretch with a different constant |

### IDE (F4)

The IDE paper describes Algorithm 1; the IDE code shipped with CDEF differs from it in several places. This code follows the shipped code except for the search size and p2:

| Item | IDE paper | This code | Why |
|---|---|---|---|
| Ambient light A | He et al.'s dark channel | Per channel, the largest value of a 10 x 10 local minimum (code; Meng et al.'s estimate) | The dark-channel airlight lowered C60 URanker by 0.9% alone and 2.1% with the kept paper rules |
| Eq. 20 | ln(t_min) / ln(t) | f(t_min) / f(t), with f from Eq. 13 (code) | f is the function t was solved with, so a patch recovers to its assumed mean albedo of 1/2. Eq. 13 fits ln badly near t = 1 (f(1) = -0.368, ln 1 = 0), so the printed form blows up in dark patches: CCF +70% and broken images |
| Eq. 22 search | argmin of (clipped share - 2%) | First the t_min with the fewest clipped values, then the lower t_min whose share is 2% above that minimum, plus 0.02 (code) | The paper's rule lowered C60 CCF by 14%; without the +0.02, CCF rose 3.5% |
| Search size | Downsampled by delta = 4 | delta = 4 (cited) | The shipped code searches at full size: 10x slower and 0.7-1.7% lower on FDUM and CCF |
| p2 | 0.07747 | 0.07747 (cited) | The shipped code has 0.07774, two digits swapped |
| Smoothing of t | Guided filter, settings not given | Two passes of a 101 x 101 mean (code) | The shipped guided filter runs on an image scaled down twice, so its guide is almost constant and it reduces to this. A real guided filter lowered C60 CCF by 9% |
| t range | Not stated | At least t_min, no upper limit (code) | |
| Output | In [0, 1] | Not clipped (code) | The overshoot reaches the fusion; clipping it cost CCF 4.9% (UIEB) and 11.4% (C60) |
| Second Phi in Eq. 20 | Â and Î swapped | Treated as a typo | Inverting Eq. 8 gives the correct form |

### Fusion (Eqs. 7-14)

| Item | Paper | This code | Why |
|---|---|---|---|
| Colour space | YUV | YUV (paper) | The authors' code uses MATLAB's offset YCbCr; YUV moved every full-set score by 1% or less |
| Eq. 7 kernel | [0 1 0; -1 4 -1; 0 -1 0] | Standard Laplacian [0 -1 0; -1 4 -1; 0 -1 0] | The printed kernel's entries sum to 2, so it responds to flat regions (a typo). The printed kernel lowered C60 CCF by 12% |
| Eq. 9 | No blur | 3 x 3 Gaussian of sigma 3 first (code) | Achanta et al. also blur first; without it FDUM rose 3% |
| Eq. 10 | Text: well exposed is 0.5; printed phi = 2, sigma = 0.5 | phi = 0.5, sigma = 0.2 (cited, Mertens) | Matches the text. The printed values reward the brightest pixels |
| Weight exponents | All 1 | Contrast and saliency squared (code) | Documented in neither CDEF nor Wang et al. All ones lowered UIEB CCF by 8.4% |
| Eq. 11 normalisation | (W + eps) / (sum W + eps) | (W + eps) / sum (W + eps) | The printed form does not sum to 1, which the text requires: in flat areas its weights sum to about 4 |
| Eq. 11 eps | 0.1 | 1e-12 (code) | At 0.1 the weights flatten towards a plain average of F1-F4; 0.1 lowered UIEB URanker by 8.5% |
| Brightness measure | Not described | Weights scaled by each input's mean(Y)^2, then renormalised (code; Wang et al. [33]) | CDEF's fusion is Wang et al.'s in the authors' code |
| Pyramid depth | Eq. 12 | Eq. 12 (paper) | The code uses Wang's two fewer levels; the paper's depth changed no score by more than 0.7% |
| Coarsest level | Eq. 14 at every level | Eq. 14 at every level (paper) | The code adds Wang's extra detail weight for under-exposed inputs; dropping it changed no score by more than 0.4% |
| Eq. 13 kernel | "5 x 5 Gaussian" | Binomial [1 4 6 4 1] / 16 along both axes | The standard pyramid kernel; the authors' `EF/pyramid_filter.m` has the same values |

### Ablation switches (Table IV)

The paper does not say how it removed each stage. The `stages:` section of a config turns one off:
- **w/o LC**: colour compensation sees the raw image.
- **w/o TM**: F1 = F2 = the colour-compensated image.
- **w/o CBM**, **w/o IDE**: that input is left out and three inputs are fused.
- **MEF in RGB / Lab** (`fusion: space`): the paper defines no weights for these spaces. RGB uses Mertens et al.'s (contrast on grey, saturation as the spread of R, G, B, exposure as the product over channels); Lab is divided by 100 so L is in [0, 1], and uses the YUV weights.

## The paper rules checked one at a time

Each rule below was set back to the paper's, alone, on the 80-image sample. The table gives the largest change on any of the five metrics on either set. Within +/-1%, a rule was a candidate; candidates were added smallest first and kept while the combined change stayed within 1%. Four were kept (largest combined change 0.78%).

| Paper rule tried | Largest change alone | Result |
|---|---|---|
| BIMEF output clipped to [0, 1] | +0.16% (CCF, UIEB) | Kept |
| BIMEF g(P, k) without the -0.01 | +0.37% (FDUM, C60) | Kept |
| Fusion without Wang's coarse-level rule | -0.38% (CCF, UIEB) | Kept |
| Fusion with Eq. 12's full pyramid depth | -0.68% (CCF, C60) | Kept |
| Fusion without the brightness measure | +0.68% (URanker, C60) | Not kept: 1.09% combined |
| Fusion in YUV | -0.86% (FDUM, C60) | 1.44% combined; adopted later on the full sets (see above) |
| IDE airlight from the dark channel | -0.92% (URanker, C60) | Not kept: 2.06% combined |
| Eq. 10 as printed (phi = 2, sigma = 0.5) | -1.97% (CCF, C60) | Not kept |
| BIMEF solved at full resolution | +2.75% (FDUM, UIEB) | Not kept |
| BIMEF as in its paper | +2.75% (CCF, UIEB) | Not kept |
| Eq. 9 without blur | +3.00% (FDUM, UIEB) | Not kept |
| IDE t_min without the +0.02 | +3.53% (CCF, C60) | Not kept |
| Fusion exponents all 1 | -8.44% (CCF, UIEB) | Not kept |
| Eq. 11 eps = 0.1 | -8.52% (URanker, UIEB) | Not kept |
| IDE t smoothed by a guided filter | -9.29% (CCF, C60) | Not kept |
| Eq. 7 kernel as printed | -12.17% (CCF, C60) | Not kept |
| IDE Eq. 22 search as printed | -14.11% (CCF, C60) | Not kept |
| IDE Eq. 20 as printed | +70.38% (CCF, UIEB) | Not kept: breaks images |

## Results

All scores use the metric authors' code ([metrics/README.md](../metrics/README.md)). Per-image scores are in `results/scores/cdef_main_v3_*.csv`.

### Raw images

The raw-image scores match the paper's Table I to every printed digit on UCIQE, FDUM and CEIQ, and within 0.1% on CCF, on all four datasets, so the images and metric code are the same as the paper's. This also identifies the paper's EUVP set as `test_samples/Inp` (515 images).

| Raw | UCIQE | FDUM | CEIQ | CCF | URanker |
|---|---|---|---|---|---|
| UIEB, ours / paper | 0.5196 / 0.5196 | 0.4470 / 0.4470 | 3.0644 / 3.0644 | 20.5112 / 20.5141 | 0.3600 / 0.3600 |
| C60, ours / paper | 0.4800 / 0.4800 | 0.2697 / 0.2697 | 2.9000 / 2.9000 | 17.3015 / 17.3162 | -0.3182 / -0.3184 |
| EUVP, ours / paper | 0.5498 / 0.5498 | 0.4451 / 0.4451 | 3.1454 / 3.1454 | 30.2663 / 30.2674 | 1.2686 / 1.2629 |
| UCCS, ours / paper | 0.4111 / 0.4111 | 0.2065 / 0.2065 | 2.6583 / 2.6583 | 12.7528 / 12.7529 | -0.9435 / -0.9443 |

### Enhanced, all four datasets

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

Against the paper, ours is within 1.5% on UCIQE and CEIQ and above it on FDUM (10-19%), CCF (6.5-9%) and URanker (1-6%). The authors' released code, run unchanged on the same images, is also above the paper (UIEB FDUM +6.6%, CCF +3.2%), so it is probably not exactly the code behind the paper's table. Ours is 0.4-5.5% above the authors' code; most of the excess comes from following the paper's F1/F2 and the 4x smaller t_min search. Higher scores do not mean better images: these metrics reward contrast and colourfulness. No image failed under any metric, and URanker improved on the raw image for 889/890 UIEB, 60/60 C60, 514/515 EUVP and 300/300 UCCS images.

### Ablation study (Table IV), UIEB

All 890 UIEB images, one config per row in `configs/ablation/`. Per-image scores are in `results/scores/ablation_*_UIEB.csv`.

| UIEB | UCIQE | FDUM | CEIQ | CCF | URanker |
|---|---|---|---|---|---|
| w/o LC, ours / paper | 0.6431 / 0.6383 | 0.8327 / 0.7498 | 3.5734 / 3.5490 | 44.0475 / 40.2333 | 2.4206 / 2.2407 |
| w/o TM, ours / paper | 0.6023 / 0.6078 | 0.7309 / 0.6988 | 3.4012 / 3.3910 | 36.0210 / 37.3245 | 1.9346 / 1.9421 |
| w/o CBM, ours / paper | 0.6572 / 0.6486 | 0.8566 / 0.8112 | 3.5664 / 3.5408 | 47.2488 / 44.2872 | 2.3781 / 2.3153 |
| w/o IDE, ours / paper | 0.6384 / 0.6207 | 0.8762 / 0.7042 | 3.5863 / 3.5294 | 46.6709 / 34.3077 | 2.4223 / 1.9708 |
| MEF in RGB, ours / paper | 0.6473 / 0.6228 | 0.8506 / 0.7178 | 3.5791 / 3.4403 | 41.9208 / 33.1875 | 2.3592 / 2.0132 |
| MEF in Lab, ours / paper | 0.6522 / 0.6248 | 0.9142 / 0.7030 | 3.5589 / 3.5053 | 48.4276 / 37.4296 | 2.4074 / 2.1327 |
| Full model, ours / paper | 0.6525 / 0.6468 | 0.9122 / 0.8307 | 3.5760 / 3.5544 | 48.2208 / 45.1072 | 2.4658 / 2.3990 |

Since our full model is itself above the paper, the change from the full model is the fairer comparison:

| Change vs full model | FDUM ours / paper | CCF ours / paper | URanker ours / paper |
|---|---|---|---|
| w/o LC | -8.7% / -9.7% | -8.7% / -10.8% | -1.8% / -6.6% |
| w/o TM | -19.9% / -15.9% | -25.3% / -17.3% | -21.5% / -19.0% |
| w/o CBM | -6.1% / -2.3% | -2.0% / -1.8% | -3.6% / -3.5% |
| w/o IDE | -3.9% / -15.2% | -3.2% / -23.9% | -1.8% / -17.8% |
| MEF in RGB | -6.8% / -13.6% | -13.1% / -26.4% | -4.3% / -16.1% |
| MEF in Lab | +0.2% / -15.4% | +0.4% / -17.0% | -2.4% / -11.1% |

**Reproduced.** Every variant scores below the full model on URanker, as in the paper. LC, TM and CBM match the paper in direction and roughly in size; w/o TM is the worst row and within 4% of the paper's own numbers on every metric.

**Not reproduced: IDE, RGB and Lab.** The authors' released code was run with the same six changes on the 60-image UIEB sample (its full model matched their full run pixel for pixel). CCF change from the full model:

| CCF change | Authors' code | Ours, same sample | Paper |
|---|---|---|---|
| w/o LC | -7.7% | -7.6% | -10.8% |
| w/o TM | -25.3% | -24.7% | -17.3% |
| w/o CBM | -4.2% | -2.4% | -1.8% |
| w/o IDE | -4.6% | -4.3% | -23.9% |
| MEF in RGB | -3.5% | -12.1% | -26.4% |
| MEF in Lab (their code: unscaled MATLAB Lab) | -8.1% | +0.9% | -17.0% |

Ours tracks the authors' code within 2 points on LC, TM, CBM and IDE, so the small IDE drop is a property of the released method, not of this implementation; the paper's IDE and RGB rows were probably produced with other code. For Lab, fusing unscaled Lab (L in [0, 100]) makes the exposure weight near zero everywhere; that costs the authors' code 8-13%, about half the paper's drop. [figures/ablation_side_by_side.jpg](../figures/ablation_side_by_side.jpg) shows the variants from both implementations.

## Reading notes: CBM and IDE

**CBM** (Yuan, Cao, Cai, Su, IEEE TGRS 59(10), 2021). Hollow square structuring elements b1 = 15 x 15 and b2 = 5 x 5 (Fig. 4b). Opening Mo(f, b) = (f eroded by b) dilated by b; closing Mc the reverse (Eq. 5). White top-hat Wh = I - Mc(Mo(I, b1), b2), black top-hat Bh = Mo(Mc(I, b1), b2) - I (Eqs. 7-8). Differences between consecutive iterations Whs, Bhs (Eqs. 9-10). Result I + omega (sum (Wh_i - Bh_i) + sum (Whs - Bhs)) with omega = 0.1, n = 10 (Eq. 11).

**IDE** (Ju, Ding, Ren, Yang, Zhang, Guo, IEEE TIP 30, 2021). The enhanced scattering model adds a light absorption coefficient so dehazed scenes are not dim (Eqs. 4-8): I = A (ln t / ln t_min) rho t + A (1 - t). Assuming each patch has mean albedo 1/2 and constant t (Eqs. 11-12) and replacing ln t by f(t) = p1 / (p2 + t) (Eq. 13, p1 = -0.397, p2 = 0.07747) gives a quadratic in t with a closed-form root (Eqs. 14-17; re-derived and correct as printed). t_min is the only free parameter, set from the share of clipped pixels on a copy downsampled by delta = 4 (Eq. 22, epsilon = 2%). Patch size 75 x 75.

## Comparison with the authors' code

Their `demo_img.m` was run unchanged on all of UIEB and C60 (scores in the Results table). Differences between their fusion (`EF/cdef_fusion.m`) and Eqs. 7-14 that the paper does not describe: squared contrast and saliency weights, eps = 1e-12, a 3 x 3 blur before saliency, Wang et al.'s brightness measure, two fewer pyramid levels, an extra detail weight at the coarsest level for the darker inputs, and fusion in offset YCbCr. Of these, the squared weights and the near-zero eps account for nearly all of the effect on scores. Their IDE and their fusion together also cause the dark halo around bright objects in their outputs (the ROV in C60 `12753.png`): the box-smoothed t gives a glow and the squared weights sharpen its edge. The same halo appears in ours, which uses both. [figures/authors_vs_ours.jpg](../figures/authors_vs_ours.jpg) compares the two implementations.

## Runtime

The paper reports 0.723 h for the 890 UIEB images (about 2.9 s per image; MATLAB, i7-12700KF). Here, in Python with single-threaded image operations on a 12-thread desktop CPU, the 950 UIEB and C60 images took 12 min 17 s of wall time with 11 workers. Single-process time per image: about 0.2 s at 256 x 256, 1.2 s at 800 x 450, 3.2 s at 1280 x 720 and 11-12 s at 2180 x 1447. BIMEF takes most of it: its preconditioner is built in a Python loop and grows with image size.

## Metrics

The metrics are the authors' code, unmodified ([metrics/README.md](../metrics/README.md)). UCIQE uses `UCIQE.m`; other implementations disagree badly (`UCIQE.py` from the same repository gives 0.54 on raw UIEB because it uses OpenCV Lab and a per-image range). FDUM's `demo_FDUM.m` is a script with a fixed folder, so `metrics/score_folder.m` repeats the lines of its loop with the same coefficients.

## Open problems

1. **Scores above the paper**, by about 10% on FDUM and 8% on CCF. Most of it comes from following the paper where the authors' code departs from it, and their code is itself above the paper.
2. **The IDE, RGB and Lab ablation rows** do not reproduce the paper's drops, with the authors' code either.
3. **IDE's t_min on Ju's Fig. 4** (0.58 and 0.48 with the IDE paper's algorithm, against 0.43 and 0.28 reported) has not been checked with the shipped search.
4. **Blue and green images end up less tinted than the paper's figures**: on crops from its figures, b* averaged about -12 where the paper's results have -20 on blue images.
5. **Slower than the paper** on large images (see Runtime).
