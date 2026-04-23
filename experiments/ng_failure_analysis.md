# DSGE + Non-Gaussian Training Failure: Systematic Analysis

**Status:** Phase A2 COMPLETE (all 5 probes: H1, H2, H3, H4, H5, H6).
**Model:** HybridDSGE-UNet (Variant A, robust basis S=3, width=16, ~10k params).
**Noise:** non_gaussian (polygaussian).
**Epochs:** 8. **Seeds:** 42, 43, 44.
**Data fractions:** FPV 25% (~25k samples), deep_space 10% (~40k samples).
**Run:** `experiments/results/a2_h1_20260420_131943.json` (36 configs).

---

## H1 — Loss incompatibility

**Hypothesis:** The default `SmoothL1(β=0.02)` is effectively L1 under heavy-tailed NG
noise; gradient magnitudes too small to escape a degenerate mask minimum. Swapping
for MSE, larger-β SmoothL1, Huber, or Charbonnier should recover training signal.

### Results — FPV (25% data, n=3 seeds)

| loss | μ val_SNR (dB) | σ | per-seed {42, 43, 44} |
|---|---|---|---|
| mse                 | **+6.41** | 1.05 | 5.56, 7.58, 6.10 |
| smoothl1_b0.02      | +2.75 | 4.75 | 0.00, 8.24, 0.00 |
| smoothl1_b0.1       | +2.75 | 4.76 | 0.00, 8.24, 0.00 |
| smoothl1_b1.0       | +2.84 | 4.91 | 0.00, 8.50, 0.00 |
| huber_d1.0          | +2.84 | 4.91 | 0.00, 8.50, 0.00 |
| charbonnier_e1e-3   | +2.75 | 4.75 | 0.00, 8.24, 0.00 |

### Results — deep_space (10% data, n=3 seeds)

| loss | μ val_SNR (dB) | σ | per-seed {42, 43, 44} |
|---|---|---|---|
| mse                 | −0.011 | 0.003 | −0.014, −0.012, −0.008 |
| smoothl1_b0.02      | −0.001 | 0.000 | −0.000, −0.001, −0.000 |
| smoothl1_b0.1       | −0.001 | 0.000 | −0.001, −0.001, −0.000 |
| smoothl1_b1.0       | −0.001 | 0.000 | −0.001, −0.002, −0.001 |
| huber_d1.0          | −0.0014 | 0.0002 | −0.0013, −0.0017, −0.0012 |
| charbonnier_e1e-3   | −0.0005 | 0.0002 | −0.0005, −0.0008, −0.0004 |

**All 36 runs complete.** Every deep_space configuration collapses to |μ| < 0.015 dB
regardless of loss; σ across seeds ≤ 0.0003 dB, confirming the collapse is not seed-driven
but structural.

### Findings

1. **MSE rescues FPV.** MSE is the *only* loss that trains reliably across all seeds on
   FPV (μ=+6.41 dB, all three seeds >+5.5 dB). Every robust loss exhibits seed-dependent
   bimodal behavior.

2. **Bimodal convergence under robust losses.** On FPV, seeds {42, 44} collapse to ≈0.00 dB
   (degenerate mask μ→0) while seed 43 converges to +8.2…+8.5 dB. The cross-loss agreement
   for collapsed seeds is extreme (0.0017–0.0034 dB) and equally extreme for seed 43
   (+8.236, +8.240, +8.500, +8.500, +8.236). This indicates a narrow basin of attraction
   — the network either finds it in the first epoch or never escapes the sigmoid-mask dead
   zone. The robust losses dampen gradient on large residuals, removing the kick needed to
   escape. MSE's unbounded gradient provides it.

3. **deep_space collapses universally.** No loss recovers training on deep_space; all are
   indistinguishable from zero (|μ| < 0.015 dB). Notably MSE also fails here despite
   rescuing FPV. This **falsifies H1 as a sufficient explanation** — the loss is necessary
   but not sufficient. A second mechanism drives the deep_space failure.

4. **Non-equivalence of Charbonnier vs SmoothL1 vs Huber.** Given the cross-loss numeric
   agreement, the three are behaving identically up to ~3-4 decimal places in both collapse
   and recovery modes. This is consistent with them all being quadratic near origin and
   linear-ish far from it for the β/δ/ε values used.

### Mask statistics (observational evidence)

Per-epoch `sigmoid(mask_logits)` statistics logged by `training_hybrid.py`:

| config | min | max | μ | σ | val_SNR |
|---|---|---|---|---|---|
| FPV, MSE, seed 42, ep 8       | 0.00 | 1.00 | **0.58** | 0.24 | +5.56 dB |
| FPV, SmoothL1 β=0.02, seed 43 | 0.00 | 0.98 | **0.55** | 0.23 | +8.24 dB |
| FPV, SmoothL1 β=0.02, seed 42 | 0.00 | 0.21 | **0.003** | 0.005 | 0.00 dB |
| deep_space, any loss, any seed, ep 8 | 0.00 | ~0.75 | **≈0.05** | ~0.09 | ≈0 dB |

Healthy training → mask distributes around μ≈0.5 with σ≈0.2 (selective attenuation).
Collapsed training → mask μ<0.1 (mostly zeros, killing signal and noise alike).

### Implications for next experiments

- **H1 is real but partial.** Losses with larger effective gradient (MSE, larger β) help
  on simpler scenarios (FPV) but do not address the deep_space failure.
- **Loss is scenario-gated.** On FPV, the NG-training question reduces to: how do we make
  the **robust** losses as reliable as MSE? Warm-up? Annealing β? Gradient clipping
  from below? These are plausible A2 side-experiments *if* robust loss is a prerequisite
  downstream (which the plan implies for NG generalization).
- **deep_space needs H2.** Since every loss collapses on deep_space and the mask statistics
  are symptomatic of the same dead-zone the collapsed FPV seeds show, **the ratio mask
  mechanism is the next suspect**. This motivates H2 directly.

---

## H2 — Ratio mask collapse

**Hypothesis:** `sigmoid(·) * |STFT|` is a poor parameterization under heavy-tailed
magnitudes. The mask saturates toward 0 because the residual-magnitude targets have
extreme dynamic range, and the model cannot produce a multiplicative mask ∈ [0,1] that
attenuates outliers without also zeroing the signal.

**Test:** swap for additive residual — `output = noisy_mag + raw_residual` — which is
unconstrained and can produce negative corrections for outliers.

**Script:** `experiments/a2_h2_mask_type.py` (12 runs: 2 scenarios × 2 mask_types ×
3 seeds × 8 epochs). Uses SmoothL1(β=0.02) (H1 baseline condition) to isolate the mask
effect from the loss effect.

**Predictions:**
- If H2 correct → `additive` rescues both scenarios regardless of loss.
- If H2 wrong → additive collapses similarly; failure is DSGE-input or normalization
  driven (→ H3).

### Results — FPV (25% data, n=3 seeds)

| mask_type | μ val_SNR (dB) | σ | per-seed {42, 43, 44} |
|---|---|---|---|
| ratio (baseline) | +2.75 | 4.75 | 0.002, 8.236, 0.002 |
| additive         | **+1.96** | **0.29** | 2.219, 2.020, 1.654 |

### Results — deep_space (10% data, n=3 seeds)

| mask_type | μ val_SNR (dB) | σ | per-seed {42, 43, 44} |
|---|---|---|---|
| ratio (baseline) | −0.0005 | 0.00023 | −0.00046, −0.00083, −0.00037 |
| additive         | **−0.989** | **0.746** | −1.849, −0.525, −0.593 |

### Mask statistics under additive

| config | min | max | μ | σ |
|---|---|---|---|---|
| FPV additive seed 42       |  −38.05 | −0.009 |   −3.51 |   3.59 |
| FPV additive seed 43       |  −57.12 | −0.076 |   −5.40 |   4.89 |
| FPV additive seed 44       |  −45.46 | −0.012 |   −4.57 |   4.47 |
| deep_space additive seed 42| **−332.44** | −0.223 | **−23.59** | **19.30** |
| deep_space additive seed 43| **−501.53** | −0.574 | **−41.84** | **33.05** |
| deep_space additive seed 44| **−365.21** | ≈0     | **−30.70** | **24.2**  |

FPV additive residuals: moderate (order-of-unity, symmetric across seeds).
deep_space additive residuals: **an order of magnitude larger** and all negative —
the network is producing aggressive negative-offset predictions that push the
reconstructed magnitude toward zero and overshoot.

### Findings

1. **Additive rescues FPV bimodality.** σ drops from 4.75 dB (ratio) to 0.29 dB
   (additive). All three seeds train consistently with additive; with ratio only
   seed 43 escapes the dead zone. The peak value is *lower* with additive
   (+1.96 vs the lucky +8.24), but the median case is dramatically better
   (+2.02 vs 0.002).

2. **Additive does NOT rescue deep_space.** All three additive seeds gave
   *negative* SNR (−1.85, −0.53, −0.59 dB; μ=−0.99 σ=0.75), i.e. **worse
   than the ratio-mask collapse** at ~0. The residual magnitudes grow by an
   order of magnitude (mean ≈ −30, min ≈ −400) compared to FPV, indicating
   a qualitatively different failure: instead of saturating near zero, the
   network over-shoots into large negative corrections. σ=0.75 dB across
   seeds is roughly 3000× larger than the ratio-mask σ=0.00023 dB, but
   centered around a worse value — the variance comes from *how badly* each
   seed diverges, not whether any escape.

3. **Partial confirmation of H2.** The ratio-mask collapse hypothesis correctly
   predicts FPV bimodality (mechanism: `sigmoid→0` dead zone; unconstrained
   residual escapes it). It fails on deep_space where at SNR ≈ −20 dB the
   clean signal magnitude is comparable to or smaller than the noise magnitude,
   so both mask parameterizations have problems — ratio collapses toward 0,
   additive overshoots toward −∞. **Mask parameterization is necessary but not
   sufficient.**

4. **Deep_space is a qualitatively harder regime.** The universal ratio-mask
   collapse on deep_space reproduces across H1 (all 6 losses) and H2 (ratio
   arm), with mask statistics essentially identical (μ ≈ 0.013, σ ≈ 0.035
   regardless of loss or seed). The additive arm's divergence further
   implicates the **target magnitude scale** rather than just the mask
   parameterization.

### Implications for subsequent experiments

- **H3 (robust normalization) is now more load-bearing.** Additive residuals on
  deep_space hit min values of −500 with mean −40; this is a symptom of
  per-channel input statistics being dominated by NG outliers, which H3 is
  designed to attenuate.
- **Consider a third mask family** beyond ratio/additive: e.g. bounded-residual
  (tanh times a data-scale factor) or log-domain prediction. A future H2'
  experiment could test whether the deep_space divergence is driven by
  unbounded outputs or by unregularized training on outlier-dominated data.
- **Mask-type × loss interaction** is likely (not tested here): additive
  with MSE might either stabilize or diverge even faster. Flagged for follow-up.

---

## H6 — DSGE fit target: signal vs noise vs n2n

**Hypothesis:** The DSGE fit (`target=clean_signal`, `input=noisy_signal`) embeds
signal information into the basis channels the NN sees. Feeding alternative fits
(noise-subspace, or N2N-style noise-vs-noise) changes the information content of
those channels and, if the NN is learning to extract signal from DSGE artefacts,
should change outcomes.

**Test:** three fit modes, identical everything else. `signal` (baseline),
`noise` (target=noise, input=noise), `n2n` (target=noise_B shuffled, input=noise_A;
theoretically K→0 by the N2N identity for iid noise, effectively a zero-information
DSGE channel).

**Script:** `experiments/a2_h6_noise_fit_nn.py` (18 runs: 2 scenarios × 3 fit_targets
× 3 seeds × 8 epochs). SmoothL1(β=0.02), ratio mask (H1 baseline conditions).

**Predictions:**
- If DSGE features are load-bearing → SNR differs across fit_targets.
- If DSGE features are passengers → SNR is invariant to fit_target; whatever
  drives the NN is independent of DSGE.

### Results — FPV (25% data, n=3 seeds)

| fit_target | μ val_SNR (dB) | σ | per-seed {42, 43, 44} |
|---|---|---|---|
| signal (baseline) | +2.75 | 4.75 | 0.002, 8.236, 0.002 |
| noise             | +2.76 | 4.76 | 0.007, 8.260, 0.002 |
| n2n               | +2.69 | 4.65 | −0.00003, 8.058, 0.001 |

### Results — deep_space (10% data, n=3 seeds)

| fit_target | μ val_SNR (dB) | σ | per-seed {42, 43, 44} |
|---|---|---|---|
| signal | −0.00055 | 0.00024 | −0.00046, −0.00083, −0.00037 |
| noise  | −0.00074 | 0.00017 | −0.00063, −0.00093, −0.00065 |
| n2n    | −0.00126 | 0.00096 | −0.00121, −0.00224, −0.00033 |

### Fitted DSGE coefficients (μ K across seeds, robust S=3)

| scenario | fit_target | K₁ | K₂ | K₃ | k₀ |
|---|---|---|---|---|---|
| FPV        | signal | +0.533 | +0.006 | +0.366 | −0.003 |
| FPV        | noise  | +0.449 | +0.363 | +0.726 | −0.181 |
| FPV        | n2n    | −0.002 | +0.0001 | −0.001 | −0.0002 |
| deep_space | signal | +0.190 | +0.008 | +0.030 | −0.005 |
| deep_space | noise  | **−4.353** | **+2.479** | **+5.501** | **−1.244** |
| deep_space | n2n    | −0.001 | +0.002 | −0.002 | +0.003 |

**n2n K-values are numerically ≈ 0 on both scenarios**, confirming the N2N
theoretical prediction (orthogonality of independent noise realizations under
the normal-equations fit). The DSGE channels for the n2n arm are therefore
effectively noise-with-some-basis-expansion-of-itself — near-zero useful
information content for signal extraction.

### Findings

1. **Bimodal pattern replicates DSGE-independently on FPV.** For every fit_target
   seed 43 converges to ≈+8 dB and seeds {42, 44} collapse. The numeric agreement
   across fit_targets is extreme:
   - seed 43 recovered values: **+8.236 / +8.260 / +8.058 dB** (signal / noise / n2n).
   - seeds {42, 44} collapsed values: all within 0.007 dB of zero.

   Since the three DSGE fits differ structurally — two-subspace K with k₀≈−0.18
   (noise), signal-tracking K (signal), or essentially null K (n2n) — the invariance
   of the outcome to the DSGE content **demonstrates the bimodality is driven by
   the NN training dynamics** (ratio mask + SmoothL1(β=0.02) + seed), not by DSGE.

2. **n2n K≈0 still gives +8 dB on the lucky seed.** The n2n arm gives the NN a
   near-zero DSGE signal (channels 2 and 3 carry essentially no signal information
   by construction), yet seed 43 still reaches +8.06 dB on FPV. This directly
   **falsifies the claim that DSGE features are required for the ≈+8 dB ceiling
   on FPV**. The NN is recovering the signal from the *raw* STFT channel; the
   DSGE channels, when informative, are apparently neither necessary nor sufficient
   in this regime.

3. **deep_space universal collapse across all fit_targets.** Every one of 9
   deep_space runs collapses to |SNR| < 0.003 dB. Neither switching DSGE fit
   target nor changing the K-magnitude regime (|K|≈0.03 for signal vs |K|≈5 for
   noise) perturbs the collapse. This is **consistent with H1** (all 6 losses
   collapsed deep_space) and **H2** (ratio arm): deep_space collapse is
   insensitive to loss, mask-type-within-ratio, and DSGE input content. Something
   else — data-scale, normalization, or GE-adequacy — drives it.

4. **Dramatic K-magnitude swing on deep_space/noise fit.** The deep_space noise-fit
   K ≈ [−4.35, +2.48, +5.50] with k₀ ≈ −1.24 is 20–100× larger than every other
   fit configuration. This reflects heavier-tailed noise-only statistics on
   deep_space (SNR −20 dB noise distribution vs FPV SNR >−5 dB) and the robust
   basis elements compensating. Despite injecting channels with this very
   different scale, the NN outcome is identical to the K≈0.03 case — further
   confirming the DSGE channels are not driving the NN's behavior on deep_space
   in the ratio-mask regime.

### Implications for subsequent experiments

- **Bimodality is a ratio-mask + robust-loss property, not a DSGE property.**
  Combined with H1 (MSE rescues FPV) and H2 (additive rescues FPV), three
  independent perturbations (loss, mask-parameterization, DSGE input) all
  confirm the FPV bimodal basin is a property of the sigmoid-mask + SmoothL1
  combination, not of DSGE.
- **deep_space failure is NOT a DSGE-input problem.** Three independent tests
  (H1 losses, H2 mask, H6 fit_target) all collapse deep_space at ≈0 dB with
  tiny σ. The failure is upstream of the NN head's training dynamics — in the
  statistical properties of the input data itself, or in how it is normalized,
  or in the adequacy of a single GE across a 23-dB-wide SNR range.
- **Ranks H3 vs H4 priority.** Since H6 rules out DSGE-input on deep_space,
  the remaining candidates are (H3) input normalization against NG outliers,
  and (H4) single-GE inadequacy across the full SNR range. H4 is more
  mechanically plausible for deep_space because the SNR range is 23 dB
  (−20..+3) vs FPV's ~20 dB (−5..+15) — the adequate-GE assumption is
  qualitatively shakier when the signal-to-noise ratio spans both sign-changes
  of the log-amplitude. (See priority decision after summary.)

---

## H3 — Per-channel normalization robustness (MAD vs p99)

**Hypothesis.** Current `training_hybrid.py` uses `_p99` (99th percentile) as
per-channel scale reference. Under heavy-tailed NG noise the top-1% tail still
dominates p99, so NN-input scales may vary strongly across signals. H3 tests
MAD-based scaling (median(|x|) × 1.4826, std-equivalent under Gaussian and
**insensitive** to extreme tails) as an alternative.

**Protocol.** Same baseline as H1/H4 (Variant A, robust S=3, width=16,
SmoothL1(β=0.02), ratio mask, non_gaussian, 8 epochs, 3 seeds).
Implementation: new `_robust_scale(x, method)` helper in `training_hybrid.py:47–60`;
`dsge_norm_method ∈ {p99, mad}` constructor param.
Script: `experiments/a2_h3_robust_norm.py` · FPV 25% / deep_space 10%.
Raw results: `experiments/results/a2_h3_20260421_102833.json`.

### FPV results (6/6 runs)

| method | n_runs | μ val_SNR (dB) | σ (dB) | per-seed val_SNR |
|---|---|---|---|---|
| p99 (baseline) | 3 | **+2.747** | **4.754** | 0.0018 / 8.2357 / 0.0020 |
| mad (robust) | 3 | **+2.699** | **4.671** | 0.0015 / 8.0927 / 0.0021 |

Same bimodal pattern, identical seed fate: seed 43 → +8 dB basin in both; seeds
{42, 44} → collapse in both. MAD-norm shifts the basin value by 0.14 dB (8.24 → 8.09)
but doesn't change which seed enters it.

### deep_space results (6/6 runs)

| method | n_runs | μ val_SNR (dB) | σ (dB) | per-seed val_SNR |
|---|---|---|---|---|
| p99 (baseline) | 3 | **−0.0006** | **0.0002** | −0.0005 / −0.0008 / −0.0004 |
| mad (robust) | 3 | **−0.0006** | **0.0002** | −0.0005 / −0.0008 / −0.0004 |

Bitwise-identical to 4 decimal places across all 3 seeds. MAD normalization has
zero effect on deep_space universal collapse.

### Verdict: H3 refuted on both scenarios

FPV bimodality and deep_space universal collapse are **invariant to the
pre-NN robust-scale choice** (p99 vs MAD). The NG failures are thus **not**
caused by heavy-tail influence on input normalization.

### A2 closure

With H3 complete, **all five DSGE + normalization intervention dimensions have
been tested** on the baseline failure configuration:

| probe | intervention | FPV bimodality fixed? | deep_space collapse fixed? |
|---|---|---|---|
| H1 | loss swap (MSE) | ✅ | ❌ |
| H2 | mask type (additive) | ✅ | ❌❌ worse |
| H3 | normalization (MAD) | ❌ | ❌ |
| H4 | per-SNR class-specific K | ❌ | ❌ |
| H6 | DSGE fit_target | ❌ | ❌ |

**FPV bimodality** has two demonstrated fixes (MSE loss, additive mask) — a
parameterization artifact in the `ratio-mask + SmoothL1(β=0.02)` interaction.

**deep_space universal collapse** survives every A2 probe. It is a property of
the NN + mask-parameterization under extreme SNR at small data volumes; no
DSGE-side or normalization-side change rescues it. The mechanism lives in loss
landscape / training dynamics at very low SNR. Deeper investigation (learning
rate schedule, mask initialization, larger data volumes) is beyond A2 scope —
documented as paper finding §9 "NG training limitations at extreme low SNR".

---

## H4 — Single global generating element (class-specific K per SNR bucket)

**Hypothesis.** DSGE assumes a homogeneous source. A clean-signal GE fitted across
the entire SNR range (−5..+15 dB on FPV, −20..0 dB on deep_space) may be sub-adequate:
per-sample SNR changes the informative frequency range, so one averaged K could wash
out bucket-specific structure.

**Protocol.** Same baseline as H1 (Variant A, robust S=3, width=16, SmoothL1(β=0.02),
ratio mask, non_gaussian, 8 epochs). Two conditions: `n_bins=0` (global, = H1 baseline)
vs `n_bins=3` (quantile-binned on per-sample oracle SNR from `snr_values.npy`;
separate K, k0 per bucket; oracle routing at inference via known SNR). 3 seeds.

**Script:** `experiments/a2_h4_snr_bucketed.py` · FPV 25% / deep_space 10%.

### FPV results (6/6 runs complete)

| n_bins | n_runs | μ val_SNR (dB) | σ (dB) | per-seed val_SNR |
|---|---|---|---|---|
| 0 (global) | 3 | **2.747** | **4.754** | 0.002 / 8.236 / 0.002 |
| 3 (bucketed) | 3 | **2.726** | **4.718** | 0.002 / 8.174 / 0.002 |

Per-seed outcomes are **identical** between conditions. The same seed (43) reaches
the +8 dB basin in both settings; the same seeds (42, 44) collapse in both.

**K-coefficient analysis (mean across seeds).**

Global fit (bins=0): K = [0.533, **0.006**, 0.366], k₀ = −0.003

Per-bucket fit (bins=3):

| Bucket | SNR mean | K₁ (tanh) | K₂ (sigmoid) | K₃ (arctan) | k₀ |
|---|---|---|---|---|---|
| 0 low | −1.6 dB | 0.21 | **0.18** | 0.35 | −0.093 |
| 1 mid | +5.0 dB | 0.53 | **0.08** | 0.46 | −0.033 |
| 2 high | +11.7 dB | 0.56 | **0.15** | 0.55 | −0.074 |

The global K₂ ≈ 0.006 hides bucket-level coefficients 0.08–0.18 — a classic mixture
cancellation: averaging over SNR produces a near-zero sigmoid term even though every
bucket needs a non-trivial one. Buckets also differ sharply in K₁ (0.21 at low SNR vs
0.56 at high) and k₀ magnitude (~30× higher at low SNR). **So the bucketed fit
genuinely produces different, richer features** than the global fit.

**Verdict for FPV: H4 refuted.** Despite substantially richer per-bucket DSGE features
(oracle-routed at inference), the NN outcome is unchanged — same two seeds collapse
to 0.002 dB, same one seed hits the +8.17 dB basin. The bimodal floor on FPV is
therefore **not** driven by single-GE inadequacy; it is a training-dynamics property
of the NN, consistent with H6 (fit_target irrelevant) and H2 (mask structure drives
collapse). Class-specific GE does not lift the ceiling nor stabilize the basin.

### deep_space results (6/6 runs complete)

| n_bins | n_runs | μ val_SNR (dB) | σ (dB) | per-seed val_SNR |
|---|---|---|---|---|
| 0 (global) | 3 | **−0.0006** | **0.0002** | −0.0005 / −0.0008 / −0.0004 |
| 3 (bucketed) | 3 | **−0.0006** | **0.0002** | −0.0005 / −0.0007 / −0.0004 |

Per-seed outcomes are **identical to 4 decimal places** between conditions — a much
stronger result than on FPV, since no seed-basin variance exists here to obscure the
signal. Every run is in the universal collapse attractor regardless of whether the
DSGE is fit globally or per bucket.

**K-coefficient analysis (mean across seeds).**

Global fit (bins=0): K = [+0.190, **+0.008**, +0.030], k₀ = −0.005

Per-bucket fit (bins=3):

| Bucket | SNR mean | K₁ (tanh) | K₂ (sigmoid) | K₃ (arctan) | k₀ |
|---|---|---|---|---|---|
| 0 low | −16.6 dB | **−0.220** | 0.063 | 0.210 | −0.032 |
| 1 mid | −10.0 dB | **−0.306** | 0.286 | 0.337 | −0.146 |
| 2 high | −3.3 dB | **+0.059** | 0.259 | 0.355 | −0.129 |

Mixture cancellation on deep_space is even more extreme than on FPV:

- K₁ (tanh): global +0.190 hides a **sign flip** — the mid bucket wants −0.306, the
  high bucket wants +0.059. Averaging across a wide SNR range washes out the bucket
  structure entirely.
- K₂ (sigmoid): global ≈+0.008 hides per-bucket 0.063–0.286 (up to **35×** larger).
- K₃ (arctan): global 0.030 hides per-bucket 0.210–0.355 (up to **12×** larger).
- k₀: global −0.005 vs per-bucket up to −0.146 (**30×** deeper offset).

The global DSGE on deep_space is essentially a meaningless average. Per-bucket fit
recovers dramatically different per-SNR-regime solutions — much richer features than
the global fit provides. Yet the NN outcome is unchanged to 10⁻⁴ dB.

**Verdict for deep_space: H4 refuted.** Class-specific K per SNR bucket yields
qualitatively different (and theoretically more adequate) DSGE features across every
SNR regime in −20..0 dB, but the NN collapse is bitwise-identical. Single-GE
inadequacy is **not** the mechanism for deep_space collapse either.

### Combined H4 verdict

Across both scenarios, class-specific GE fit passes its own internal sanity check
(buckets produce genuinely different K values, confirming the global fit was lossy)
but **does not change NN outcomes on either dataset**:

- FPV: same bimodal pattern, same seeds in the same attractors.
- deep_space: universal 0 dB collapse, identical to 4 decimal places.

This refutes the "mixed-SNR averaging washes out the useful GE" hypothesis as a
cause of the NG-training failures. Combined with H6 (fit_target irrelevant) and H5
(standalone noise-subspace DSGE is near-trivial), the DSGE layer is **not the
bottleneck** for either failure mode. Both the FPV bimodality and the deep_space
collapse live in the NN training dynamics / loss landscape.

---

## Summary so far (A2 diagnostic phase)

- ✅ **H1 tested, 36/36 runs.** Loss matters on FPV, not on deep_space. MSE is the only
  seed-robust choice on FPV among those tested.
- ✅ **H2 tested, 12/12 runs.** Additive rescues FPV bimodality (σ 4.75 → 0.29 dB) but
  makes deep_space *worse* (μ=−0.99 σ=0.75 dB vs −0.0005 dB ratio collapse). Ratio-mask
  collapse is one mechanism but not a universal fix.
- ✅ **H5 noise-subspace DSGE diagnostic (standalone denoiser).** Largely falsified:
  near-identity trivial collapse on robust/fractional/trig bases; only polynomial on FPV
  shows honest +0.5–1.3 dB with positive Δcorr. See
  `experiments/results/a2_h5_noise_dsge_20260420_225909.md`.
- ✅ **H6 tested, 18/18 runs.** DSGE fit_target is irrelevant to NN outcome:
  FPV bimodality replicates identically across {signal, noise, n2n} (seed 43 gives
  +8.06…+8.26 dB regardless; seeds {42, 44} collapse regardless). n2n with fitted K≈0
  still reaches +8.06 dB on lucky seed → DSGE features are neither necessary nor
  sufficient for the FPV ceiling. deep_space universally collapses across all 3
  fit_targets even when K magnitudes differ by ~100×.
- ✅ **H4 tested, 12/12 runs.** Class-specific K per SNR bucket (3 buckets, oracle
  routing) produces genuinely richer features — classic mixture cancellation is
  confirmed (global K₁ on deep_space hides a sign flip; K₂ ≈ 0.008 hides per-bucket
  0.06–0.29 with 35× range). Yet NN outcomes are unchanged: FPV same bimodal
  per-seed pattern, deep_space identical to 4 decimal places. Single-GE inadequacy
  is **not** the mechanism for either failure. Combined with H5/H6, DSGE-side
  interventions are ruled out as rescues for NG training.
- ✅ **H3 tested, 12/12 runs.** p99 (baseline) vs MAD normalization — FPV per-seed
  bimodal pattern identical (0.002/+8.24/0.002 vs 0.002/+8.09/0.002); deep_space
  identical to 4 decimal places (both ≈−0.0006 dB). Normalization irrelevant on
  both scenarios. **A2 diagnostic phase COMPLETE.**

### Consolidated mechanism picture

Cross-referencing H1, H2, H6:

| probe | FPV bimodal? | deep_space collapse? |
|---|---|---|
| H1: swap loss MSE | Rescued (μ=+6.41 dB, σ=1.05) | Still collapses |
| H1: robust losses | Bimodal (σ=4.75) | Still collapses |
| H2: additive mask | Rescued (σ=0.29) | Worse (μ=−0.99, σ=0.75) |
| H6: fit_target=n2n (K≈0) | Bimodal (σ=4.65) | Still collapses |
| H6: fit_target=noise (|K|×5) | Bimodal (σ=4.76) | Still collapses |
| H4: SNR-bucketed K (oracle routing) | Bimodal (σ=4.72) | Still collapses (ident. to 10⁻⁴ dB) |
| H3: MAD normalization | Bimodal (σ=4.67) | Still collapses (ident. to 10⁻⁴ dB) |

**FPV bimodal behavior** is a property of `ratio-mask + SmoothL1(β=0.02)`,
independent of DSGE content, DSGE fit_target, DSGE per-SNR granularity, and input
normalization scale. Fixed by MSE or additive mask.

**deep_space collapse** is independent of loss (H1), mask-parameterization-within-ratio
(H2 ratio arm), DSGE input content (H6), per-SNR class-specific fit (H4), and input
normalization scale (H3). Worsened by unbounded additive mask. Every A2 probe is
exhausted — the mechanism lives beyond A2 scope (training dynamics / data volume).

### Novel finding worth flagging for the paper

**Bimodal convergence under robust losses on FPV** — at fixed loss and hyperparameters,
identical training differs by >8 dB depending solely on seed. Cross-seed reproducibility
of the *collapsed* solution (4 decimal places) and of the *recovered* solution (also
4 decimal places) suggests two isolated attractors rather than continuous noise, pointing
to a sharp loss-landscape feature that the robust losses fail to cross but MSE crosses.
H6 confirms this is a **training-dynamics property**, not a DSGE property — it replicates
with fit_target ∈ {signal, noise, n2n} spanning K∈{0.01, 0.4, 5} and even with K≈0
(n2n). This is directly relevant to reliability claims for DSGE and is plausibly
generalizable beyond DSGE to other sigmoid-masked STFT denoisers with robust losses.

---

## Phase A2 complete — moving to Phase B

All 5 probes executed (90 hybrid runs + 24 standalone-DSGE = 114 experimental
runs total over A2). Mechanistic picture:

**FPV bimodality** — a **parameterization artifact** of `ratio-mask + SmoothL1(β=0.02)`.
Two rescues: MSE loss (H1) or additive mask (H2). Not DSGE-related. Paper finding.

**deep_space collapse** — a **training-dynamics / low-SNR limitation** of the
STFT-masked denoiser at 10% data / 8 epochs. Invariant to every A2 lever
(loss, mask type, DSGE content, DSGE fit_target, DSGE per-SNR granularity,
normalization). Pre-B1 diagnostic note: baseline U-Net achieves −1.27 dB on full
deep_space (compared to Transformer's +1.65 dB and the 10%-hybrid's collapse) —
so collapse is not a deep-space-wide limitation, just the A2 config's.

**Next step — Phase B1 FPV main experiment.** Full FPV, 3 seeds, MSE loss
(H1 winner, seed-robust), all models. See RESEARCH_STATUS §10.11 for the
full post-A2 Phase-B plan.
