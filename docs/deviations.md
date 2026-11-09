# Deviations from the papers

Every place the papers are unclear or look wrong, what this code does, and why. Each choice is an option in [configs/cdef_paper.yaml](../configs/cdef_paper.yaml).

The first version was written from the papers alone, without the authors' released CDEF code. That version left CCF about 12% below the authors' own outputs. The released code (`demo_img.m`, `lib/`, `IDE/`, `EF/`) was then read to find out why, and used as a guide to which choices the authors made. The code here is our own implementation of those choices, not a translation: each stage was checked against the output of the authors' matching stage on the same input (see Main configuration). Every choice taken from their code is an option, and the paper's reading is kept as the other value.

## Main configuration

`configs/cdef_paper.yaml` mixes the paper and the released code, choosing per stage for faithfulness, scores, output quality and run time:

| Stage | Setting | Source | Why |
|---|---|---|---|
| Luminance (BIMEF) | `method: reference` | BIMEF's released implementation, which CDEF's code uses | Faithful to the cited method's own code. Moves each metric by about 1% either way, but brings the images much closer to the authors' (4 → under 1 grey level). Its approximate solve is faster than an exact one |
| Luminance (BIMEF) | `offset: 0`, `clip: true` | Paper | The released code's −0.01 and missing clip changed no score by more than 0.4% (see Paper rules checked one at a time) |
| Colour compensation | Eq. 2 mask kept (`use_mask: true`) | Paper | The authors' code builds the mask and then overwrites it with ones, so in effect it compensates every pixel. The mask only matters on pixels brighter than 0.85, where it protects light sources as the paper intends |
| Colour compensation | Lab conversion without clipping | Code | The luminance stage leaves a few values above 1; clipping them before Lab shifts hue |
| F1, F2 | Eqs. 4-6 as printed (`a: 2`, clipped) | Paper | The authors' F1 (`a = 2.3`, unclipped) and 8-bit F2 lowered FDUM, CCF and URanker by 1-3% |
| CBM | Unclipped | Code | Already matched the authors' stage to 0.24 grey levels |
| IDE | `method: reference` | Code | The largest effect on scores of any stage, with fusion |
| IDE | `delta: 4` | Ju's paper | Their code searches `t_min` at full size, 10× slower (1.6 s vs 0.16 s per image) and slightly lower scoring |
| IDE | `p2: 0.07747` | Ju's paper | Their code has 0.07774, the digits swapped |
| Fusion | Squared contrast and saliency, `eps: 1e-12`, Wang's brightness measure | Code, and Wang's paper | The largest effect on scores of any stage, with IDE |
| Fusion | YUV (`colour: yuv`) | Paper | The authors' code fuses in offset YCbCr. YUV moved each full-set score by 1% or less (see Replication results) |
| Fusion | Full pyramid depth (`drop_levels: 0`), no coarse-level rule | Paper (Eqs. 12, 14) | Wang's two fewer levels and coarse-level rule changed no score by more than 0.8%, together with the BIMEF rules above |

### Paper rules checked one at a time

Each place where the main configuration departs from a rule the papers state was set back to the paper's rule alone, with everything else unchanged, and the 60 + 20 sample was rerun and scored. The table gives the largest change on any of the five metrics, on either set, against the main configuration as it was before this check. Where it stayed within ±1%, the paper's rule was a candidate.

Candidates were then added one at a time, smallest first, keeping each only if all those kept so far still changed no score by more than 1% together. Four were kept, with a largest combined change of 0.78% (C60 CCF). Three more were within 1% alone but pushed the combined change over: their effects add up.

| Paper rule tried | Largest change alone | Result | Note |
|---|---|---|---|
| Fusion Eq. 11 normalisation as printed | 0.00% | Not used | Identical here only because the brightness measure renormalises the weights. With that off, its weights sum to about 4 in flat areas and images break |
| BIMEF output clipped to [0, 1] | +0.16% (CCF, UIEB) | **Paper rule** | |
| BIMEF `g(P, k)` without the −0.01 | +0.37% (FDUM, C60) | **Paper rule** | |
| Fusion without Wang's coarse-level rule | −0.38% (CCF, UIEB) | **Paper rule** | |
| Fusion with Eq. 12's full pyramid depth | −0.68% (CCF, C60) | **Paper rule** | |
| Fusion without Wang's brightness measure | +0.68% (URanker, C60) | Not used | 1.09% (C60 CCF) added to the four kept |
| Fusion in YUV | −0.86% (FDUM, C60) | Not used | 1.44% (C60 URanker) added to the four kept |
| IDE airlight from He et al.'s dark channel | −0.92% (URanker, C60) | Not used | 2.06% (C60 URanker) added to the four kept |
| Fusion Eq. 10 as printed: `φ = 2, σ = 0.5` | −1.97% (CCF, C60) | Not used | |
| BIMEF illumination solved at full resolution | +2.75% (FDUM, UIEB) | Not used | |
| BIMEF as in its paper (`method: paper`) | +2.75% (CCF, UIEB) | Not used | |
| Fusion Eq. 9 without blur | +3.00% (FDUM, UIEB) | Not used | |
| IDE `t_min` without the +0.02 | +3.53% (CCF, C60) | Not used | |
| Fusion exponents all 1 | −8.44% (CCF, UIEB) | Not used | |
| Fusion Eq. 11 `ε = 0.1` | −8.52% (URanker, UIEB) | Not used | |
| IDE `t` smoothed by a guided filter | −9.29% (CCF, C60) | Not used | |
| Fusion Eq. 7 kernel as printed | −12.17% (CCF, C60) | Not used | |
| IDE Eq. 22: `argmin \|clipped share − 2%\|` | −14.11% (CCF, C60) | Not used | |
| IDE Eq. 20 as printed: `ln(t_min) / ln(t)` | +70.38% (CCF, UIEB) | Not used | Divides by `ln(t)`, which nears 0 as `t` nears 1 |

The IDE rows use the `airlight`, `search` and `recovery` options of `method: reference`. Rules the papers do not state (for example whether CBM's output is clipped) were not part of this check.

**Checks.** Our implementation of every stage of the authors' code, chained end to end, reproduced their final images to 0.94 (UIEB) and 0.67 (C60) grey levels on average over the 60 + 20 sample, and their scores to within 0.1% on UIEB and 0.8% on C60. Each stage on its own, given the authors' input to that stage, matched their output to within 0.2 grey levels on average over 8 images.

**The main configuration on the sample** (from the paper-rule check, with YCbCr fusion; YUV moves full-set scores by 1% or less), with gaps to the authors' code on the same images:

| Sample (60 UIEB, 20 C60) | UCIQE | FDUM | CEIQ | CCF | URanker |
|---|---|---|---|---|---|
| UIEB, main configuration | 0.6464 | 0.8677 | 3.5221 | 44.8475 | 2.3508 |
| UIEB, authors' code | 0.6409 | 0.8412 | 3.5160 | 43.4993 | 2.2908 |
| UIEB gap | +0.9% | +3.1% | +0.2% | +3.1% | +2.6% |
| C60, main configuration | 0.6434 | 0.6941 | 3.4846 | 40.7839 | 1.8295 |
| C60, authors' code | 0.6339 | 0.6603 | 3.4621 | 37.8578 | 1.7661 |
| C60 gap | +1.5% | +5.1% | +0.7% | +7.7% | +3.6% |

It scores above the authors' code on every metric, most on C60 CCF. Its images differ from the authors' by 5.7 grey levels on average (UIEB 5.73, C60 5.67), against under 1 for the fully checked chain. The places where it departs from that chain account for this: the colour mask, F1/F2, `delta` and `p2`, and the four paper rules from the check above. From the earlier runs, the paper's F1/F2 is the largest part. Before the four paper rules, the gaps were up to 0.9% larger (C60 CCF +8.6%). They look alike: see [figures/main_version.jpg](../figures/main_version.jpg), which shows the main version slightly more contrasty than the authors', with the same halo around the ROV. The 80 images took 61 s of wall time with 11 workers.

## Replication results

### Main configuration, full sets

All 890 UIEB and 60 C60 images (`python -m eval.run_dataset --method cdef_main_v3`), scored with the metric authors' code. Per-image scores are in `results/scores/cdef_main_v3_*.csv`. Two earlier full runs are recorded only by their scores here; their outputs were deleted. The run in YCbCr (`cdef_main_v2`) scored within 1.03% of these on every metric and set (UIEB CCF −1.03% with YUV). The run before the four paper rules (`cdef_main`) scored within 0.9% of `cdef_main_v2`. The authors' code was run on the same images; its outputs and scores were kept outside the project.

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

The main configuration scores above both the paper and the authors' released code on every metric, by 0.4-1.2% on UCIQE and CEIQ, 0.9-6.9% on URanker and CCF, and about 10% on FDUM against the paper (3-3.5% against the authors' code, whose own FDUM is already 6% above the paper). No image failed under any metric. URanker improved on the raw image for 889 of 890 UIEB images and all 60 C60 images.

The excess comes from the paper-faithful choices described under Main configuration, chiefly F1/F2 as printed and the smaller `t_min` search. Higher scores here do not mean better images: these metrics reward contrast and colourfulness, and the images look much like the authors' with slightly harder contrast (see [figures/main_version.jpg](../figures/main_version.jpg)).

### Main configuration, EUVP and UCCS

EUVP is its 515-image test set (`test_samples/Inp`), UCCS all 300 images (blue, green and blue-green). The paper does not say which EUVP images it used, but the raw-image scores below match its Table I to every printed digit (CCF to 0.004%), as they do for UIEB, so this is the same test set. The authors' code was not run on these two sets.

| Raw images | UCIQE | FDUM | CEIQ | CCF | URanker |
|---|---|---|---|---|---|
| EUVP, ours | 0.5498 | 0.4451 | 3.1454 | 30.2663 | 1.2686 |
| EUVP, paper | 0.5498 | 0.4451 | 3.1454 | 30.2674 | 1.2629 |
| UCCS, ours | 0.4111 | 0.2065 | 2.6583 | 12.7528 | −0.9435 |
| UCCS, paper | 0.4111 | 0.2065 | 2.6583 | 12.7529 | −0.9443 |

| Enhanced | UCIQE | FDUM | CEIQ | CCF | URanker |
|---|---|---|---|---|---|
| EUVP (515), main configuration | 0.6447 | 0.7762 | 3.5689 | 63.1464 | 3.3582 |
| EUVP, paper | 0.6367 | 0.7062 | 3.6076 | 57.8109 | 3.3213 |
| EUVP, ours vs paper | +1.3% | +9.9% | −1.1% | +9.2% | +1.1% |
| UCCS (300), main configuration | 0.6160 | 0.7033 | 3.6317 | 47.4777 | 2.6746 |
| UCCS, paper | 0.6077 | 0.5925 | 3.6328 | 43.7368 | 2.5748 |
| UCCS, ours vs paper | +1.4% | +18.7% | 0.0% | +8.6% | +3.9% |

The pattern is the same as on UIEB and C60: above the paper on UCIQE, URanker and especially FDUM and CCF, and on CEIQ slightly below on EUVP (−1.1%) and level on UCCS. UCCS FDUM is the largest gap, at 18.7%. No image failed under any metric. URanker improved on the raw image for 514 of 515 EUVP images and all 300 UCCS images.

### Ablation study (Table IV), UIEB

All 890 UIEB images, one config per row in `configs/ablation/`. Each config is the main configuration with one change:
- **w/o LC** skips BIMEF, so colour compensation sees the raw image.
- **w/o TM** sets F1 = F2 = the compensated image.
- **w/o CBM** and **w/o IDE** drop that input from the fusion, leaving three inputs.
- **MEF in RGB** and **MEF in Lab** run the fusion in those spaces.

The full model is `cdef_main_v3`. The paper does not say how it removed each stage, so these are our reading.

| UIEB | UCIQE | FDUM | CEIQ | CCF | URanker |
|---|---|---|---|---|---|
| w/o LC, ours | 0.6431 | 0.8327 | 3.5734 | 44.0475 | 2.4206 |
| w/o LC, paper | 0.6383 | 0.7498 | 3.5490 | 40.2333 | 2.2407 |
| w/o TM, ours | 0.6023 | 0.7309 | 3.4012 | 36.0210 | 1.9346 |
| w/o TM, paper | 0.6078 | 0.6988 | 3.3910 | 37.3245 | 1.9421 |
| w/o CBM, ours | 0.6572 | 0.8566 | 3.5664 | 47.2488 | 2.3781 |
| w/o CBM, paper | 0.6486 | 0.8112 | 3.5408 | 44.2872 | 2.3153 |
| w/o IDE, ours | 0.6384 | 0.8762 | 3.5863 | 46.6709 | 2.4223 |
| w/o IDE, paper | 0.6207 | 0.7042 | 3.5294 | 34.3077 | 1.9708 |
| MEF in RGB, ours | 0.6473 | 0.8506 | 3.5791 | 41.9208 | 2.3592 |
| MEF in RGB, paper | 0.6228 | 0.7178 | 3.4403 | 33.1875 | 2.0132 |
| MEF in Lab, ours | 0.6522 | 0.9142 | 3.5589 | 48.4276 | 2.4074 |
| MEF in Lab, paper | 0.6248 | 0.7030 | 3.5053 | 37.4296 | 2.1327 |
| Full model, ours | 0.6525 | 0.9122 | 3.5760 | 48.2208 | 2.4658 |
| Full model, paper | 0.6468 | 0.8307 | 3.5544 | 45.1072 | 2.3990 |

The change from the full model is the fairer comparison, since our full model is itself above the paper. FDUM, CCF and URanker:

| Change vs full model | FDUM ours | FDUM paper | CCF ours | CCF paper | URanker ours | URanker paper |
|---|---|---|---|---|---|---|
| w/o LC | −8.7% | −9.7% | −8.7% | −10.8% | −1.8% | −6.6% |
| w/o TM | −19.9% | −15.9% | −25.3% | −17.3% | −21.5% | −19.0% |
| w/o CBM | −6.1% | −2.3% | −2.0% | −1.8% | −3.6% | −3.5% |
| w/o IDE | −3.9% | −15.2% | −3.2% | −23.9% | −1.8% | −17.8% |
| MEF in RGB | −6.8% | −13.6% | −13.1% | −26.4% | −4.3% | −16.1% |
| MEF in Lab | +0.2% | −15.4% | +0.4% | −17.0% | −2.4% | −11.1% |

**Reproduced.** Every variant is below the full model on URanker, as in the paper. LC, TM and CBM match the paper in direction and roughly in size: w/o TM is the worst row in ours, and is within 4% of the paper's own numbers on every metric.

**Not reproduced.**

- **IDE matters much less here.** Without it, CCF drops 3% where the paper drops 24%. One possible reason is that our CBM output is unclipped (`cbm: clip: false`) and already adds much of the contrast that CCF and FDUM reward, so the fusion loses little without F4. The authors' code also leaves CBM unclipped, so this would only explain the gap if their ablation was run with different code. This has not been tested.
- **RGB fusion is worse, but half as much as in the paper.** Our RGB weights are Mertens' (contrast on grey, saturation as the channel standard deviation, exposure per channel).
- **Lab fusion is no worse than YUV.** We fuse Lab scaled by 1/100 with the same weights as YUV. L is then a luma in [0, 1] and a, b are opponent chroma, so it behaves like YUV. The paper does not define weights for RGB or Lab.

**The authors' code does not reproduce these rows either.** We ran their released pipeline with one change per row on the 60-image UIEB sample:
- w/o LC skips BIMEF.
- w/o TM sets F1 = F2 = the colour-corrected image.
- w/o CBM and w/o IDE drop that input.
- MEF in RGB and MEF in Lab replace `rgb2ycbcr` with no conversion or with MATLAB's unscaled `rgb2lab`, and the inverse after `cdef_fusion`.

Its full-model outputs matched the earlier full run of their code pixel for pixel. Change from the full model, on the sample:

| CCF change | Authors' code | Ours, same sample | Paper |
|---|---|---|---|
| w/o LC | −7.7% | −7.6% | −10.8% |
| w/o TM | −25.3% | −24.7% | −17.3% |
| w/o CBM | −4.2% | −2.4% | −1.8% |
| w/o IDE | −4.6% | −4.3% | −23.9% |
| MEF in RGB | −3.5% | −12.1% | −26.4% |
| MEF in Lab | −8.1% | +0.9% | −17.0% |

| FDUM change | Authors' code | Ours, same sample | Paper |
|---|---|---|---|
| w/o IDE | −3.4% | −3.8% | −15.2% |
| MEF in RGB | −2.7% | −5.8% | −13.6% |
| MEF in Lab | −12.8% | +0.1% | −15.4% |

- **LC, TM, CBM and IDE:** ours tracks their code within 2 points on every row. So the small w/o IDE drop is a property of the method as released, not of our implementation. The paper's IDE and RGB rows were probably produced with other code or another way of removing the stage.
- **Lab:** fusing unscaled Lab (L in [0, 100]) costs their code 8–13%, which accounts for about half of the paper's Lab drop. With L far outside [0, 1], the exposure weight is near zero everywhere and the fusion falls back on the brightness term.

Scripts and outputs are in the session scratchpad (`abl/run_authors_ablation.m`, `abl/score_and_rank.py`); they are not part of the project.

### Previous configuration, fixed sample

These rows are from the earlier configuration: the paper's IDE and fusion, no Wang options. They have not been rerun with the main configuration.

`configs/cdef_paper.yaml` (as it was then) on every 15th UIEB image and every 3rd C60 image (`python -m eval.run_dataset --sample`), against the authors' `demo_img.m` run on the same images. The authors' per-image scores are in `results/scores/authors_sample_*.csv`; ours from this run were later deleted.

| Sample (60 UIEB, 20 C60) | UCIQE | FDUM | CEIQ | CCF | URanker |
|---|---|---|---|---|---|
| UIEB, ours | 0.6428 | 0.8111 | 3.5227 | 38.4473 | 2.1188 |
| UIEB, authors' code | 0.6409 | 0.8412 | 3.5160 | 43.4993 | 2.2908 |
| UIEB gap | +0.3% | −3.6% | +0.2% | −11.6% | −7.5% |
| C60, ours | 0.6309 | 0.6068 | 3.4930 | 33.4219 | 1.6809 |
| C60, authors' code | 0.6339 | 0.6603 | 3.4621 | 37.8578 | 1.7661 |
| C60 gap | −0.5% | −8.1% | +0.9% | −11.7% | −4.8% |

### Earlier full run, CBM clipped

All of UIEB and C60 with an earlier version of the code, before CBM's output was unclipped, scored with the metric authors' code. See [metrics/README.md](../metrics/README.md). The output images and per-image scores from this run have been removed; the raw-image scores are still in `results/scores/raw_*.csv`.

| | UCIQE | FDUM | CEIQ | CCF |
|---|---|---|---|---|
| **UIEB raw**, ours | 0.5196 | 0.4470 | 3.0644 | 20.5112 |
| UIEB raw, paper | 0.5196 | 0.4470 | 3.0644 | 20.5141 |
| **UIEB enhanced**, ours | 0.6424 | 0.8244 | 3.5678 | 37.3082 |
| UIEB enhanced, paper | 0.6468 | 0.8307 | 3.5544 | 45.1072 |
| Gap | −0.7% | −0.8% | +0.4% | **−17.3%** |
| **C60 raw**, ours | 0.4800 | 0.2697 | 2.9000 | 17.3015 |
| C60 raw, paper | 0.4800 | 0.2697 | 2.9000 | 17.3162 |
| **C60 enhanced**, ours | 0.6253 | 0.5550 | 3.5039 | 30.1949 |
| C60 enhanced, paper | 0.6315 | 0.5832 | 3.4867 | 36.0389 |
| Gap | −1.0% | −4.8% | +0.5% | **−16.2%** |

The raw rows match the paper to all printed digits for UCIQE, FDUM and CEIQ, so those metrics are measuring what the paper measured. CCF is within 0.1% on raw images, for a reason we have not found. No image failed under any metric.

URanker, the paper's preferred metric (Table II), using the official code and pretrained weights:

| | UIEB | C60 |
|---|---|---|
| Raw, ours | 0.3600 | −0.3182 |
| Raw, paper | 0.3600 | −0.3184 |
| Enhanced, ours | 2.2427 | 1.7469 |
| Enhanced, paper | 2.3990 | 1.8214 |
| Gap | −6.5% | −4.1% |

URanker improved on the raw image for 887 of 890 UIEB images and all 60 C60 images. These scores are above every traditional method listed in Table II except the paper's own CDEF.


These full-set numbers predate unclipping CBM's output (`cbm: clip: false`). On the sample, that change moved UIEB CCF from 19.8% to 11.6% below the authors' own outputs. The full set has not been rerun with the current code.

## Comparison with the authors' code

Their `demo_img.m` was run unchanged in MATLAB on all of UIEB and C60, and its outputs scored with the same metrics:

| Full set | UCIQE | FDUM | CEIQ | CCF | URanker |
|---|---|---|---|---|---|
| UIEB, paper | 0.6468 | 0.8307 | 3.5544 | 45.1072 | 2.3990 |
| UIEB, authors' code | 0.6455 (−0.2%) | 0.8859 (+6.6%) | 3.5618 (+0.2%) | 46.5553 (+3.2%) | 2.4443 (+1.9%) |
| C60, paper | 0.6315 | 0.5832 | 3.4867 | 36.0389 | 1.8214 |
| C60, authors' code | 0.6293 (−0.3%) | 0.6182 (+6.0%) | 3.4881 (0.0%) | 36.3923 (+1.0%) | 1.8696 (+2.6%) |

So the released code lands close to the paper but not on it, mostly above it, and FDUM about 6% high. It is probably not exactly the code behind the paper's table. Matching the released code is therefore the closest a replication can get, and the main configuration aims for that.

On the 60 + 20 subset, with gaps to the authors' own outputs, before the main configuration:

| UIEB, 60 images | UCIQE | FDUM | CEIQ | CCF | URanker |
|---|---|---|---|---|---|
| CBM clipped, our fusion | −0.4% | −6.7% | 0.0% | −19.8% | −7.9% |
| CBM unclipped, our fusion (current code) | +0.3% | −3.6% | +0.2% | −11.6% | −7.5% |
| CBM unclipped, their fusion (scratch reconstruction) | −0.1% | +0.6% | +0.6% | −2.8% | −1.2% |

| C60, 20 images | UCIQE | FDUM | CEIQ | CCF | URanker |
|---|---|---|---|---|---|
| CBM clipped, our fusion | −0.9% | −10.7% | +0.7% | −18.5% | −5.6% |
| CBM unclipped, our fusion (current code) | −0.5% | −8.1% | +0.9% | −11.7% | −4.8% |
| CBM unclipped, their fusion (scratch reconstruction) | −0.7% | −2.9% | +1.9% | −6.1% | +2.6% |

Their fusion (`EF/cdef_fusion.m`) differs from Eqs. 7–14 in ways the paper does not describe:

| | Paper | Their code |
|---|---|---|
| Weight exponents | All 1 | Contrast² and saliency² |
| Regularisation | 0.1 | 1e-12 |
| Brightness bonus | None | Weights multiplied by each input's mean Y², then renormalised |
| Coarsest level | Same rule as all levels | The darkest input gets an extra 1.5 × detail weight |
| Pyramid depth | `floor(log2(min(h, w)))` | Two fewer levels |
| Colour space | YUV | MATLAB YCbCr |
| Saliency | No blur | 3×3 Gaussian blur first |

Removing one rule at a time showed that the squared weights and the near-zero regularisation account for nearly all of its effect; the brightness bonus and coarse-level rule made almost no difference. Wang et al.'s paper documents the brightness measure (from Kou et al. [31]), the two fewer levels (its Eq. 6) and the coarse-level rule (its Eq. 10, `α = 1.5`, a 3 × 3 Gaussian of sigma 1, under-exposed inputs only). It gives all exponents as 1, so the squaring is not documented anywhere. All of these are now options of `multiscale_fusion`, on in the main configuration.

Other differences in their code, now options here: F1 uses mean ± 2.3 std with the limits clamped to `[0, 1]` and the output left unclipped (not used); IDE divides an already normalised image by 256, which turns its guided filter into a box blur (`smoothing: box`). Swapping in one of their stages at a time, with the rest ours: their IDE raised FDUM and CCF most; their fusion closed most of the CCF and URanker gap; their F1 alone made every metric worse. The dark halo around bright objects in their outputs (the ROV in C60 `12753.png`) appears only with their IDE and their fusion together: the box-smoothed `t` gives a glow, and the fusion's squared weights sharpen its edge. `smoothing: guided` is available to avoid it, at an unmeasured cost in scores.

## Open problems

1. **The main configuration scores above the paper**, by about 10% on FDUM and 8% on CCF on the full sets (see Replication results). Most of the excess comes from following the paper where the authors' code departs from it, and the authors' code is itself above the paper. Our reproduction of all of the authors' stages matched their code to within 1% on the sample; switching the four differing choices to theirs would do the same here, but their F1/F2 are not yet options in this code.
2. **The CCF gap of the paper-only version is explained.** It came from IDE and fusion as the authors implemented them, not as the papers describe them: with our implementation of their code, CCF matches their outputs to within 1%.
3. **IDE does not reproduce Ju's `t_min`** in its paper form: on Ju Fig. 4 we get 0.58 and 0.48 where the paper reports 0.43 and 0.28. The IDE code shipped with CDEF searches differently (see the IDE table), which likely explains this but has not been checked on those two images.
4. **Blue and green images end up less tinted than the paper's.** On pairs cropped from the paper's figures, measured before the settings were chosen, b* averaged about −12 where the paper has −20 on blue images.
5. **Slower than the paper.** See Runtime below.

## How the settings were chosen

| Check | Data | Limits |
|---|---|---|
| Against the paper's own outputs | 10 raw / "Ours" pairs cropped from CDEF Figs. 4 and 6 | Small JPEG figure crops, so a guide to direction only |
| Against the IDE paper | 2 hazy images and their `t_min` values from Ju Fig. 4 | Small figure crops. The 75-pixel patch covers far more of these than of a full-size image |
| Candidate comparison | All of UIEB, five settings | Used a rough UCIQE before the authors' code was added, so these gains are not comparable with the table above |
| Visual comparison | [figures/candidate_settings.jpg](../figures/candidate_settings.jpg) | Four of the settings on three UIEB and four C60 images |

The candidates were:

| Setting | UCIQE gain UIEB (rough) | PSNR | SSIM |
|---|---|---|---|
| `range255_mertens` (chosen) | 1.279 | 17.61 | 0.764 |
| `range255_printed`: `phi: 2, sigma: 0.5` | 1.248 | 16.88 | 0.748 |
| Literal `weight_range: 1`, `phi: 2, sigma: 0.5` | 1.194 | 18.54 | 0.818 |
| `plan_sigma0.1`: literal, with `sigma_ratio: 0.1` | 1.244 | 18.85 | 0.820 |
| `ide_exact`: literal, with `log_mode: exact` | 1.221 | 16.15 | 0.763 |

PSNR and SSIM are against the UIEB references. Only `range255_mertens` was scored with the authors' metrics, so this table does not show it beats the others on those metrics.

**Why `weight_range: 255`.** On `[0, 1]` values the weight product is around 1e-4, far below `eps = 0.1`, so Eq. 11 gives about 0.25 everywhere and fusion is close to a plain average of F1–F4. That is unlikely to be what the authors ran: with equal weights, RGB and YUV fusion would be nearly identical because the two spaces are linearly related, yet their Table IV shows them differing.

## CDEF

| Item | Paper | This code | Evidence or reason |
|---|---|---|---|
| Gaussian in Eq. 1 | No sigma given | `sigma: 1.0` pixel. **Replaces the plan's `0.1 × min(H, W)`** | See below |
| Mask smoothing | Text says `G` smooths the mask; Eq. 1 applies it to the channel | Both (`smooth_mask: true`) | Follows the plan. ΔE 11.33 vs 11.34 without, so it hardly matters |
| Eq. 2 mask | Light sources excluded | Kept (`use_mask: true`) | The authors' code sets the mask to 0 and then every value up to 0.85, including those zeros, to 1, so its mask is all ones. `use_mask: false` reproduces that |
| Lab conversion | Not stated | sRGB / D65, no clipping of out-of-range values | The luminance stage can return values above 1; MATLAB's conversion, which the authors use, does not clip them |
| Eq. 3 tie | `mean_G = mean_B` undefined | Bluish | Algorithm 1's else branch |
| Eq. 4 | Says "variance" | Standard deviation | `μ ± 2σ` needs matching units |
| Eq. 5 gray mean | "Gray average value" | Mean over all three channels | Gray-world assumption, and CBM Eq. 14 |
| Eq. 5 quantile level | Can leave `[0, 1]` | Clamped; a channel with no usable range is left unchanged | Avoids division by zero |
| Eq. 7 kernel | `[0,1,0; −1,4,−1; 0,−1,0]` | Standard Laplacian (`laplacian: standard`) | Printed kernel is a typo: its entries sum to 2, so it responds to flat regions |
| Eqs. 7 to 9 scale | Not stated | `[0, 1]` values (`weight_range: 1`) | Authors' code. With `eps: 1e-12` the scale no longer decides the weights. The earlier choice of 255 is explained under How the settings were chosen |
| Eq. 9 blur | None printed | 3 × 3 Gaussian of sigma 3 (`saliency_blur: 3, saliency_window: 3`) | Authors' code; Achanta et al. also blur first |
| Weight exponents | All 1 (text) | Contrast and saliency squared (`exponents: [2, 1, 2, 1]`) | Authors' code. Documented in neither CDEF nor Wang et al. |
| Colour space | YUV | Offset YCbCr (`colour: ycbcr`) | Authors' code. `yuv` follows the paper |
| Pyramid depth, coarsest level, brightness | Not described | Wang et al. [33] (`drop_levels: 2`, `coarse_rule: true`, `brightness: true`) | CDEF's fusion is Wang et al.'s in the authors' code, and these three rules are in Wang's paper |
| Eq. 10 | Text says well-exposed is 0.5, but `φ = 2, σ = 0.5` | `phi: 0.5, sigma: 0.2` (Mertens) | Matches the paper's text. The printed values reward the brightest pixels |
| Eq. 11 | `(W + ε) / (ΣW + ε)` | `(W + ε) / Σ(W + ε)` (`normalization: sum_to_one`) | Printed form does not sum to 1, which the text requires |
| Eq. 11 `ε` | 0.1 | `1e-12` | Authors' code. At 0.1 the weights flatten towards a plain average |
| Eq. 13 kernel | "5 × 5 Gaussian" | Binomial `[1 4 6 4 1] / 16` along both axes | Chosen as the standard pyramid kernel, then confirmed: the authors' `EF/pyramid_filter.m` (Mertens' filter) is `[.0625 .25 .375 .25 .0625]`, the same values. Fixed in code, not a config option |
| Ablation switches | Table IV, how each stage is removed is not described | `stages:` section (w/o LC skips BIMEF, w/o TM sets F1 = F2 = compensated, w/o CBM/IDE drops that input) and `fusion: colour: rgb \| lab` | Ours. The paper defines no weights for RGB or Lab: RGB uses Mertens' weights, Lab is scaled by 1/100 and uses the YUV weights |

### Why the Gaussian sigma changed

The plan's default was a large blur. Three things point the other way:

1. Paper Fig. 4 shows `α = β = 1` giving a nearly gray image. That only happens if `G(a) ≈ a`, which needs a small sigma.
2. In the paper's bluish results, red-green colour nearly disappears (pink coral turns olive). ΔE to those results falls steadily as sigma shrinks, from 13.04 at 32 px to 11.06 at 0.5 px.
3. At `0.1 × min(H, W)`, real images get red and magenta halos around dark objects, the "red artifact noise" the paper warns about. See the whale, the ROV and the diver's legs in the last column of [the figure](../figures/candidate_settings.jpg).

`plan_sigma0.1` does score well on UCIQE and PSNR. We read that as UCIQE rewarding the false colour, as the paper itself warns of in Fig. 8, and as PSNR rewarding images that stay closer to the raw tint. The exact value is not pinned down: results barely change between 0.5 and 2 px. To go back, set `sigma_ratio: 0.1`.

## Luminance compensation (BIMEF)

| Item | Paper | This code | Reason |
|---|---|---|---|
| Implementation | Paper | BIMEF's released implementation (`method: reference`) | Used by the authors' code. `method: paper` keeps the reading below |
| Resolution of `T` | Full | Half (`solve_scale: 0.5`) | As in the released implementation |
| Smoothing strength | Not fixed | `lam: 0.5` | Released implementation |
| Smoothness weights | Window sum over a square | Gradient summed along a 5-pixel line in its own direction (`window: 5`) | Released implementation |
| Border | Not stated | The image wraps around when `T` is solved | Released implementation |
| Solver | Multi-resolution preconditioned CG | CG preconditioned by a modified incomplete Cholesky factor, stopped at 10% relative residual (`pcg_tol: 0.1`) | Released implementation. `pcg_tol: null` solves exactly; the final images differ by about 0.2 grey levels and it is slower |
| Resizing | Not stated | Antialiased bicubic (Keys, `a = −0.5`), the mask for the `k` search resized as 8-bit data | Released implementation |
| Brightened image | `g(P, k)` | `g(P, k) − 0.01`, output not clipped (`offset: 0.01, clip: false`) | Released implementation |
| Eq. 8 vs Eq. 9 (paper path) | Window sum in one, plain gradient in the other | Eq. 10, which uses both | Eq. 10 is the system actually solved |
| `k` search range | Not given | `[1, 7]` | Our choice. `k` reaches 7 on some images, so the limit is active |
| No under-exposed pixels | Not covered | `k = 1`, output equals input | Nothing to compensate |

## CBM

| Item | Paper | This code | Reason |
|---|---|---|---|
| Meaning of `Wh_i` for `i > 1` | "Transformed n times" | `mode: repeat`, filter applied to its own output | Sizes are stated as fixed, and "iterations" fits this. The open-close filter does not change on a second pass, so repeat mode adds the same detail n times. `scale` is available |
| Final clip | Not stated | Not clipped (`clip: false`), so edge overshoot reaches the fusion | The authors' pipeline passes it on unclipped. On the subset this alone cut the CCF gap to the authors from 19.8% to 11.6% |
| ACS stage | Part of the CBM paper | Off for F3 (`acs: false`) | CDEF cites CBM for contours only, and F2 is already ACS with a different constant |
| Eq. 12 limits | Defined by pixel counts | Quantiles | Same idea; only used when `acs: true` |

## IDE

| Item | Paper | This code | Evidence or reason |
|---|---|---|---|
| Eq. 20 | Uses `ln(t)` | `f(t)` from Eq. 13 (`log_mode: rational`) | See below |
| Second `Φ` in Eq. 20 | `Â` and `Î` swapped | Treated as a typo | Inverting Eq. 8 gives the correct form |
| Eq. 22 | `argmin{Ψ − ε}` | `argmin \|Ψ − ε\|` | A signed version would just push to the extreme |
| `t_min` search range | Not given | `[0.05, 0.95]` | Our choice |
| `Ψ` | "Clip 2% in the shadows and in the highlights" | 2% in total | Our reading |
| Guided filter | No guide, radius or regularisation | Gray guide, `gf_radius: 37`, `gf_eps: 0.001` | Our choice: half the 75-pixel patch |
| `t` range | Not stated | Limited to `[t_min, 1]` | Eq. 17 gives `t > 1` in dark patches |
| Dark channel | "Via [17]" | 15-pixel patch, top 0.1% | He et al.'s values |

The rows above are the paper path (`method: paper`). The main configuration uses `method: reference`, the IDE code shipped with CDEF:

| Item | Paper | Reference path | Note |
|---|---|---|---|
| Airlight | He et al.'s dark channel | Per channel, the largest value of a 10 × 10 local minimum (`air_window: 10`) | Meng et al.'s estimate |
| Eq. 22 | Clipped share equal to `ε` | First the `t_min` with the fewest clipped values, then the lower `t_min` where the share is `ε` above that minimum, plus 0.02 (`tmin_offset`) | The search uses the unsmoothed `t` |
| Search size | Downsampled by `δ = 4` | `delta: 4` | Their code searches at full size: 10× slower, and on the sample it scored 0.7-1.7% lower on FDUM and CCF and matched their images about 0.5 grey levels more closely |
| Eq. 20 | `ln(t_min) / ln(t)` | `f(t_min) / f(t)` | A pixel with `t = t_min` keeps a factor of 1 |
| Smoothing of `t` | Guided filter | Two passes of a 101 × 101 mean (`smoothing: box`) | Their guided filter runs on an image scaled down twice, so its guide has almost no variance and it reduces to this. It causes the halos described above; `smoothing: guided` is a real guided filter |
| `t` range | Not stated | At least `t_min`, no upper limit | |
| Output | In `[0, 1]` | Not clipped (`clip: false`) | The overshoot reaches the fusion, as in their code |

### Why Eq. 20 uses f(t)

Eq. 13 fits `ln(t)` badly near `t = 1`: `f(1) = −0.368` while `ln(1) = 0`. Dark patches therefore get `t` at or above 1, and `ln t_min / ln t` blows up.

With `ln(t)` as printed, the clipped fraction `Ψ` is U-shaped in `t_min` and never reaches 2%, so the golden-section search just stops at the bottom of the U. With `f(t)`, `Ψ` falls steadily and the search works. Because `t` was solved with `f`, a patch then recovers exactly to the assumed mean albedo of 1/2.

It also matches published images better: ΔE 11.9 and 12.5 vs 14.9 and 19.6 on Ju Fig. 4, and 11.3 vs 12.6 on the CDEF figures. `ln(t)` does land nearer Ju's `t_min` values (0.49 and 0.23), but for the wrong reason given the above, so that may be a clue to how the authors bound `t`.

## Runtime

The paper reports 0.723 h for the 890 UIEB images (about 2.9 s per image; MATLAB, i7-12700KF).

These timings were taken with the literal weights; the chosen setting differs only in fusion. Here the same images take 1.50 h of summed per-image time (about 6.1 s per image, slowest about 43 s), in Python with single-threaded image operations on a 12-thread AMD desktop CPU. That was with 11 workers running at once, which slows each one, so it is not a clean single-image figure; the 950 images took about 10 minutes of wall time. Most of the time goes to the illumination solve on large images. Alone, a 1280×720 image takes 2.2 s and a 1920×1080 image takes 5.8 s, of which 3.3 s is luminance compensation.

### Main configuration

On the full sets, the 950 images took 12 min 17 s of wall time with 11 workers on the same 12-thread CPU. Single-process timings of each stage, averaged over four sample images of 400 × 225 to 640 × 360, in seconds per image: luminance (reference) 1.05, colour 0.28, CBM 0.33, IDE (reference, `delta: 4`) 0.16, fusion 0.38, tone maps 0.06. That adds up to about 2.3 s per image. The paper-only version took 1.66 s on the same images. The approximate BIMEF solve builds its preconditioner in a Python loop, so it grows with image size: for the largest UIEB images (about 3 megapixels) expect several seconds for that step alone.

## Metrics

The metrics are the authors' code, unmodified. See [metrics/README.md](../metrics/README.md) for the sources and versions.

| Item | Note |
|---|---|
| UCIQE variant | `UCIQE.m`. Other implementations disagree badly: the rough one used earlier gave 0.38 on raw UIEB, and `UCIQE.py` from the same repo gives 0.54, because it uses OpenCV Lab and a per-image histogram range |
| FDUM | `demo_FDUM.m` is a script with a fixed folder, so `score_folder.m` repeats the lines of its loop, using the same coefficients |

## Not done yet

- The IDE and Lab-fusion rows of the ablation study do not reproduce the paper's drops (see Ablation study). The reasons are untested.
