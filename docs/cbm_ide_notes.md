# CBM and IDE: reading notes

Summaries of the two detail methods CDEF uses for its inputs F3 and F4, written before coding (plan §3.5). Where the paper is unclear, the choice we made is in [deviations.md](deviations.md).

## CBM: Contour Bougie Morphology

Yuan, Cao, Cai, Su, *IEEE TGRS* 59(10), 2021. DOI 10.1109/TGRS.2020.3033407.

**Goal.** Recover object contours lost to forward scattering. The CBM paper has two stages: CB morphology (Sec. III-A) and Adaptive Contrast Stretch (ACS, Sec. III-B). CDEF cites CBM only "to highlight the object contour", and its Eq. 5 is already ACS with a different constant, so F3 uses the morphology stage. ACS is available as an option.

**Structuring elements (Fig. 4b).** Hollow squares, meaning only the outer ring of the square. `b1` is 15×15 and `b2` is 5×5. They are flat, so `b(s, t) = 0` in Eq. 4.

**Operations.**

| Eq. | Definition |
|---|---|
| 4 | Grayscale erosion and dilation |
| 5 | `Mo(f, b) = (f ⊖ b) ⊕ b`, `Mc(f, b) = (f ⊕ b) ⊖ b` |
| 7 | `Wh(I, b1, b2) = I − Mc(Mo(I, b1), b2)` |
| 8 | `Bh(I, b1, b2) = Mo(Mc(I, b1), b2) − I` |
| 9 | `Whs_(n−1) = Wh_n − Wh_(n−1)` |
| 10 | `Bhs_(n−1) = Bh_n − Bh_(n−1)` |
| 11 | `I_en = I + ω ( Σ_(i=1..n) (Wh_i − Bh_i) + Σ_(i=2..n) (Whs_(i−1) − Bhs_(i−1)) )` |

**Parameters.** `ω = 0.1`, `n = 10`.

**Unclear.** The paper never defines `Wh_i` for `i > 1`, only "transformed n times". Two readings:

- Repeat: apply the filter to its own output, so `Wh_i = I − S_i` with `S_i = Mc(Mo(S_(i−1), b1), b2)`.
- Scale: grow the elements with `i`, as in earlier multi-scale top-hat work, where the `Wh` / `Whs` structure comes from.

The paper states fixed sizes for `b1` and `b2` and calls the process iterations, which fits the repeat reading better.

## IDE: Image Dehazing and Exposure

Ju, Ding, Ren, Yang, Zhang, Guo, *IEEE TIP* 30, 2021. DOI 10.1109/TIP.2021.3050643.

**Model (EASM).** Plain haze models leave dehazed scenes dim, so IDE adds a light absorption coefficient `α`:

- Eq. 4: `I = A (1 − α) ρ t + A (1 − t)`
- Eq. 7: `α = 1 − ln(t) / ln(t_min)`
- Eq. 8: `I = A · (ln t / ln t_min) · ρ · t + A (1 − t)`

**Gray-world step.** Treat each patch as a small scene with mean albedo 1/2 (Eq. 11) and constant `t` inside it. Taking the patch mean of Eq. 8 gives Eq. 12:

`Î = Â · ln(t) / (2 ln t_min) · t + Â (1 − t)`

Here `Î` is the local mean over the patch and all three channels, and `Â` is the mean of the atmospheric light over the channels (Eq. 10).

**Closed-form transmission.** Replacing `ln(t)` by `f(t) = p1 / (p2 + t)` (Eq. 13, `p1 = −0.397`, `p2 = 0.07747`) turns Eq. 12 into a quadratic `λ1 t² + λ2 t + λ3 = 0` (Eqs. 14–16):

- `λ1 = Â ln t_min`
- `λ2 = ln t_min (Î + Â p2 − Â) − ½ Â p1`
- `λ3 = p2 ln t_min (Î − Â)`

Eq. 17 takes `t = (−λ2 − √(λ2² − 4 λ1 λ3)) / (2 λ1)`, and Eq. 18 smooths it with a guided filter. We re-derived Eqs. 14–16 from Eq. 12 and they are correct as printed.

**Recovery (Eq. 20).** Inverting Eq. 8 gives

`ρ = (ln t_min / ln t) · (1 + (I − A) / (A t))`

The second `Φ` in the printed Eq. 20 swaps `Â` and `Î` in its arguments. This is a typo.

**Finding t_min (Eq. 22).** The only free unknown. Search with golden section so that the fraction of clipped pixels in the recovered image, computed on a version downsampled by `δ = 4`, equals `ε = 0.02`. Eq. 21 gives a simpler estimate from the atmospheric light pixel, which the paper shows is worse.

**Algorithm.** Locate `A` with the dark channel method; compute `Î`, `Â`; search `t_min`; compute `t`; recover `ρ`.

**Parameters.** `p1 = −0.397`, `p2 = 0.07747`, `ε = 0.02`, `δ = 4`, patch `s = 75×75`.

**Unclear.**

- Guided filter guide image, radius and regularisation are not given.
- The dark channel patch size and top-pixel fraction come from He et al., not from this paper.
- Eq. 22 prints `argmin{Ψ − ε}`, with no absolute value and no search range.
- `Ψ` is "clip 2% in the shadows and in the highlights", which could mean 2% in total or 2% each.
- **Eq. 13 is a poor fit near t = 1.** `f(1) = −0.368` while `ln(1) = 0`. So dark patches get `t > 1` from Eq. 17 (for example `Î/Â = 0.11` with `ln t_min = −1.2` gives `t ≈ 1.04`, where the true root is about 0.92). `ln t` in Eq. 20 is then near zero or positive and `ρ` blows up, so an implementation must either clamp `t` or use `f(t)` in Eq. 20.
