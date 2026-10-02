Capped = control's first EOS at the last position (255) or absent; see is_capped().

Questions: 1319; uncapped in control: 805; capped: 514 (39.0%). Control strict answer missing: 178 (uncapped: 88).

### Cross-container reproducibility (control's first 64 questions, re-run in containers B and C)

- control_sentinel@p2B: 64/64 identical in every committed token and commit step
- control_sentinel@p2C: 64/64 identical in every committed token and commit step

### Churn per arm vs control

Primary = strict extraction, uncapped questions. Correctness and McNemar use the repo's grader.

| arm | perturbation | strict churn, uncapped (primary) | strict churn, all | lenient churn, all | acc ctrl → arm | McNemar p | trajectories diverged | median first-divergence step [IQR] |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| fp32_all | fp32, every pass | 24.6 [21.7, 27.7] | 35.4 [32.9, 38.0] | 22.3 | 75.7 → 76.6 | 0.375 | 100.0% | 4 [1–10] |
| fp32_prime | fp32, prime pass only | 23.5 [20.7, 26.5] | 33.3 [30.8, 35.9] | 22.0 | 75.7 → 75.9 | 0.88 | 100.0% | 4 [1–10] |
| fp32_finalize | fp32, finalize passes only | 19.8 [17.1, 22.6] | 29.4 [27.0, 31.9] | 18.4 | 75.7 → 76.6 | 0.309 | 100.0% | 38 [34–45] |
| fp32_steps_within_early | fp32, steps 0–7 of each block | 13.3 [11.1, 15.8] | 20.5 [18.5, 22.8] | 13.3 | 75.7 → 75.5 | 0.923 | 100.0% | 5 [2–36] |
| fp32_steps_within_late | fp32, steps 8–31 of each block | 19.4 [16.8, 22.3] | 29.6 [27.2, 32.2] | 19.3 | 75.7 → 75.4 | 0.869 | 100.0% | 14 [11–20] |
| fp32_steps_within_last25 | fp32, steps 24–31 of each block | 8.9 [7.2, 11.1] | 16.2 [14.3, 18.3] | 10.8 | 75.7 → 76.7 | 0.12 | 99.9% | 31 [26–63] |
| fp32_steps_global_early | fp32, steps of blocks 0–1 | 18.5 [16.0, 21.3] | 29.0 [26.7, 31.5] | 19.3 | 75.7 → 75.8 | 0.936 | 100.0% | 5 [2–12] |
| fp32_steps_global_last25 | fp32, steps of blocks 6–7 | 1.6 [0.9, 2.7] | 6.1 [5.0, 7.6] | 2.7 | 75.7 → 76.3 | 0.0574 | 99.5% | 196 [193–202] |
| fp32_layer0 | fp32, layer 0 only, every pass | 24.6 [21.7, 27.7] | 34.6 [32.1, 37.2] | 23.1 | 75.7 → 76.6 | 0.38 | 100.0% | 4 [1–10] |
| fp32_layers11_14 | fp32, layers 11–14 only, every pass | 12.5 [10.4, 15.0] | 20.7 [18.6, 23.0] | 13.5 | 75.7 → 76.0 | 0.694 | 100.0% | 14 [5–30] |
| fp32_layer15 | fp32, layer 15 only, every pass | 6.8 [5.3, 8.8] | 12.5 [10.8, 14.4] | 8.5 | 75.7 → 76.0 | 0.716 | 99.3% | 31 [13–61] |
| noise_s0 | bf16 + U(±1 ULP) noise, seed 0 | 25.1 [22.2, 28.2] | 35.0 [32.4, 37.6] | 23.3 | 75.7 → 76.6 | 0.434 | 100.0% | 3 [1–10] |
| noise_s1 | bf16 + U(±1 ULP) noise, seed 1 | 24.5 [21.6, 27.6] | 33.7 [31.2, 36.3] | 21.8 | 75.7 → 75.9 | 0.878 | 100.0% | 3 [1–9] |
| noise_s2 | bf16 + U(±1 ULP) noise, seed 2 | 25.1 [22.2, 28.2] | 34.0 [31.5, 36.6] | 22.1 | 75.7 → 75.0 | 0.55 | 100.0% | 4 [1–10] |

First-divergence step is global: block = step // 32.

### Windows: churn normalized by the share of the run perturbed

| arm | share of denoising steps | share of forward passes | share of routed token-rows | flip rate inside window (Phase 1 telemetry, control, n=200) | strict churn, uncapped | churn / step share |
|---|---:|---:|---:|---:|---:|---:|
| fp32_all | 100.0% | 100.0% | 100.0% | 26.8% | 24.6 | 0.246 |
| fp32_prime | 0.0% | 0.4% | 4.5% | 24.1% | 23.5 | — |
| fp32_finalize | 0.0% | 3.0% | 2.9% | 26.4% | 19.8 | — |
| fp32_steps_within_early | 25.0% | 24.2% | 23.2% | 27.4% | 13.3 | 0.532 |
| fp32_steps_within_late | 75.0% | 72.5% | 69.5% | 26.8% | 19.4 | 0.258 |
| fp32_steps_within_last25 | 25.0% | 24.2% | 23.2% | 26.6% | 8.9 | 0.358 |
| fp32_steps_global_early | 25.0% | 24.2% | 38.6% | 27.3% | 18.5 | 0.740 |
| fp32_steps_global_last25 | 25.0% | 24.2% | 7.7% | 25.5% | 1.6 | 0.065 |
| fp32_layer0 (all layers' passes counted; fp32 only at the named layers) | 100.0% | 100.0% | 100.0% | 94.8% | 24.6 | — |
| fp32_layers11_14 (all layers' passes counted; fp32 only at the named layers) | 100.0% | 100.0% | 100.0% | 10.0% | 12.5 | — |
| fp32_layer15 (all layers' passes counted; fp32 only at the named layers) | 100.0% | 100.0% | 100.0% | 14.4% | 6.8 | — |

### Pairwise window contrasts (paired over the same uncapped questions)

Each cell: questions churned only under the first arm vs only under the second; exact McNemar p.

| contrast | only first | only second | churn first | churn second | McNemar p |
|---|---:|---:|---:|---:|---:|
| fp32_steps_within_early vs fp32_steps_within_last25 | 77 | 42 | 13.3 | 8.9 | 0.00171 |
| fp32_steps_global_early vs fp32_steps_global_last25 | 146 | 10 | 18.5 | 1.6 | 4.11e-32 |
| fp32_steps_within_early vs fp32_steps_within_late | 56 | 105 | 13.3 | 19.4 | 0.000139 |
| fp32_prime vs fp32_finalize | 94 | 64 | 23.5 | 19.8 | 0.0208 |
| fp32_layer0 vs fp32_layer15 | 163 | 20 | 24.6 | 6.8 | 4.61e-29 |
| fp32_layer0 vs fp32_layers11_14 | 136 | 39 | 24.6 | 12.5 | 8.75e-14 |
| fp32_layers11_14 vs fp32_layer15 | 78 | 32 | 12.5 | 6.8 | 1.36e-05 |

### Additivity: does the partition reproduce fp32_all? (uncapped, strict)

- sum of part churn counts: 611; |union|: 319; fp32_all churn: 198
- fp32_all churned questions also churned by at least one part: 161/198; Jaccard(union, fp32_all) = 0.45

### Perturbation family: noise vs fp32 (uncapped, strict)

Caveat: U(−1, +1) ULP noise has mean |δ| = 0.5 ULP per logit and reaches 1 ULP; bf16 rounding moves a logit by at most 0.5 ULP (mean 0.25). The noise arms are the larger perturbation, so equal churn would mean saturation, not equivalence.

- fp32_all: 24.6%; noise seeds: 25.1%, 24.5%, 25.1% (mean 24.9%, range 0.6 pts)
- pairwise overlap of churned sets (Jaccard):
  - noise_s0 / noise_s1: 0.43
  - noise_s0 / noise_s2: 0.42
  - noise_s0 / fp32_all: 0.42
  - noise_s1 / noise_s2: 0.37
  - noise_s1 / fp32_all: 0.39
  - noise_s2 / fp32_all: 0.43

### Saturation: how many perturbed arms change each question's strict answer (uncapped)

14 perturbed arms. Null = each arm churns independently of the question, at its own observed rate (Poisson-binomial).

| arms changing the answer | questions observed | expected under independence |
|---:|---:|---:|
| 0 | 340 | 49.5 |
| 1 | 118 | 156.1 |
| 2 | 71 | 224.5 |
| 3 | 48 | 195.1 |
| 4 | 46 | 114.1 |
| 5 | 39 | 47.4 |
| 6 | 27 | 14.4 |
| 7 | 22 | 3.2 |
| 8 | 25 | 0.5 |
| 9 | 24 | 0.1 |
| 10 | 24 | 0.0 |
| 11 | 10 | 0.0 |
| 12 | 10 | 0.0 |
| 13 | 1 | 0.0 |
| 14 | 0 | 0.0 |

Churn events (question × arm): 1998. Questions churned by ≥ half the arms: 116 (14.4% of uncapped) carry 1053 events (52.7%). Questions churned by any arm: 465.

### Do control-arm signals predict fragility? (uncapped questions)

Target A: churn count k over the perturbed arms (Spearman ρ). Target B: churned under fp32_all (AUC = P(signal of a churned question > signal of a stable one)). 95% bootstrap CIs, 2000 resamples.

| signal | n | Spearman ρ with k [95% CI] | AUC for fp32_all churn [95% CI] |
|---|---:|---:|---:|
| answer-token confidence at commit (min over the number's tokens) | 717 | -0.058 [-0.132, +0.018] | 0.462 [0.406, 0.516] |
| mean commit confidence, whole generation | 805 | -0.411 [-0.472, -0.349] | 0.314 [0.274, 0.357] |
| generated length (tokens before EOS) | 805 | +0.306 [+0.241, +0.368] | 0.630 [0.588, 0.672] |
| first-divergence step vs fp32_all (256 if none) | 805 | -0.020 [-0.091, +0.049] | 0.467 [0.422, 0.514] |

Capped status (all 1319 questions, strict): mean k = 5.61 capped vs 2.48 uncapped; AUC(capped → k ≥ 1) = 0.676.


---

# Follow-up analyses (no GPU)

Generated by `python -m research.sq2_extra` on the same Phase 2 data and the
Phase 1 telemetry. Setup unchanged: n=1319 (805 uncapped in control), seed-42
question order, A100-40GB, `max_tokens=256 block_length=32 steps=256`, fixed
schedule, B=32.

## Summary

- **Flip count alone does not explain churn.** A saturating curve in expected
  flips fits the 11 fp32 arms with χ² = 543 on 9 dof. The deviations are
  systematic:
  - The cache-writing arms sit far above the curve: prime z = +13.6,
    finalize z = +12.5, with only 5–7k flips per question each.
  - The late arms sit below it: blocks 6–7 z = −8.5, steps 24–31 z = −4.1.
  - Weighting a flip in a cache-writing pass as ~27 step flips cuts χ² to
    61 on 8 dof. Even then, blocks 6–7 stay below the curve (z = −4.6), and
    layer 0 (z = +3.7) and the early step windows (z = +2.7, +2.9) stay above.
  - So where a flip lands matters, not just how many: a flip in a pass that
    writes the KV cache counts for more than one in a step, and an early step
    for more than a late one.
- **The layer ranking is consistent with flip count but not explained by it.**
  - Order: layer 15 has 5.7k flips per question and 6.8% churn, layers 11–14
    have 15.9k and 12.5%, layer 0 has 37.7k and 24.6%.
  - All three sit above the flip-only curve, because each includes its share
    of cache-writing flips.
  - Layer 0 reaches 24.6% churn with 37.7k flips. The step-only window with
    steps 0–7 has 40.4k flips and reaches 13.3%.
  - With cache writes weighted, layer 0 is still above the curve (+3.7) and
    layer 15 below it (−2.3).
- **Fragility is a property of the question.** Using only the three
  independent noise seeds:
  - 3-of-3 churn: 83 questions (10.3%) observed against 12.4 (1.5%) expected
    if churn were independent of the question.
  - Pairwise Jaccard: 0.41 observed against 0.14 expected under independence.
  - The beta-binomial fit puts **20.7% [18.0, 23.8] of questions at a flip
    probability above 0.5**, and 47% below 0.1. The distribution is
    U-shaped: most questions are stable, and a fifth are coin-flips or worse.
- **Blocks 6–7.** On the 251 questions whose answer line is written in blocks
  6–7, fp32 everywhere churns 27.9% and the steps of blocks 0–1 churn 19.9%.
  The steps of blocks 6–7 churn only 2.8%. So these questions are not
  inherently stable; perturbing them late, after the reasoning is in the KV
  cache, rarely changes their answer.
- **Accuracy noise floor.**
  - fp32 everywhere disagrees with control on 13.9% of questions' correctness.
  - At that rate the SD of a paired accuracy difference is 2.6 points at n=200
    and 1.0 point at n=1319 (95% half-width ±5.2 and ±2.0 points).
  - The largest observed |Δacc| across the 14 arms is 1.1 points.
  - No arm survives Holm correction (smallest adjusted p = 0.80), so every
    accuracy difference here is "not resolved".
  - The repo's "±3 pt at n=200" is the 1-SD value at an 18% discordance rate,
    consistent with §9b's 40/200 = 20%.

**Caveats.**
- Expected flips come from the control trajectory at n=200. An arm's own
  flips after it diverges are not measured.
- The noise arms' flip counts rest on a boundary model that underpredicts
  measured rounding flips by 1.6–2.5×. It counts only the 8th/9th boundary,
  while near-uniform logits tie several experts at once, so the noise arms
  are plotted but kept out of both fits.
- Three seeds per question identify the spread of per-question flip
  probability only coarsely; read the bootstrap CI, not the point estimate.

## Dose-response: churn vs expected flipped top-8 decisions

Expected flips per question = flipped top-8 decisions (bf16 vs fp32, same hidden states) summed over the arm's window — every pass type, layer and position type it covers — from the Phase 1 telemetry (control trajectory, n=200, same config), divided by 200. fp32 arms: measured. Noise arms: no direct measurement exists, so the measured rounding flips are rescaled per cell by a boundary model (flip iff the error difference at the 8th/9th logits exceeds their gap; rounding errors U(±0.5 ULP), noise U(±1 ULP)), using the measured gap histogram.

Model check on rounding, where flips were measured (all passes):

| layer | measured flip rate | model, rounding | model, ±1 ULP noise |
|---:|---:|---:|---:|
| 0 | 94.8% | 38.3% | 43.9% |
| 1 | 66.3% | 30.3% | 38.0% |
| 2 | 44.2% | 22.3% | 31.4% |
| 3 | 38.8% | 20.4% | 29.6% |
| 4 | 32.2% | 17.6% | 26.6% |
| 5 | 32.7% | 17.9% | 27.1% |
| 6 | 19.4% | 11.4% | 18.8% |
| 7 | 14.8% | 9.0% | 15.4% |
| 8 | 12.1% | 7.3% | 12.8% |
| 9 | 9.8% | 6.1% | 11.0% |
| 10 | 9.5% | 5.9% | 10.5% |
| 11 | 8.3% | 5.1% | 9.3% |
| 12 | 8.5% | 5.3% | 9.5% |
| 13 | 9.1% | 5.6% | 10.0% |
| 14 | 14.1% | 8.5% | 14.6% |
| 15 | 14.4% | 8.0% | 13.0% |

The model counts only the 8th/9th boundary from a coarse 10-bin histogram, so it is a rough estimate: noise-arm x positions are model-based, not measured.

**Model A, flip count alone** (11 fp32 arms, weighted by binomial SE): churn = c_max·(1 − exp(−k·flips)); c_max = 23.5% ± 1.4, k = 2.08e-05 ± 2.2e-06; χ² = 543 on 9 dof.

**Model B, cache-write flips weighted separately**: x = step flips + w·(prime + finalize flips); c_max = 26.6%, k = 1.19e-05, w = 26.9; χ² = 61 on 8 dof.

| arm | expected flips / question | of which in cache-writing passes | source | churn (primary) | model A | z (A) | model B | z (B) |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| fp32_finalize | 4,875 | 4,875 | measured | 19.8 | 2.3 | **+12.5** | 20.9 | -0.9 |
| fp32_layer15 | 5,719 | 1,104 | measured | 6.8 | 2.6 | **+4.7** | 8.9 | **-2.3** |
| fp32_prime | 6,830 | 6,830 | measured | 23.5 | 3.1 | **+13.6** | 23.5 | -0.0 |
| fp32_steps_global_last25 | 12,549 | 0 | measured | 1.6 | 5.4 | **-8.5** | 3.7 | **-4.6** |
| fp32_layers11_14 | 15,929 | 1,110 | measured | 12.5 | 6.6 | **+5.1** | 10.9 | +1.4 |
| fp32_layer0 | 37,727 | 2,658 | measured | 24.6 | 12.8 | **+7.8** | 19.1 | **+3.7** |
| fp32_steps_within_last25 | 39,150 | 0 | measured | 8.9 | 13.1 | **-4.1** | 9.9 | -0.9 |
| fp32_steps_within_early | 40,369 | 0 | measured | 13.3 | 13.4 | -0.1 | 10.1 | **+2.7** |
| fp32_steps_global_early | 67,170 | 0 | measured | 18.5 | 17.7 | +0.6 | 14.6 | **+2.9** |
| fp32_steps_within_late | 118,661 | 0 | measured | 19.4 | 21.5 | -1.5 | 20.1 | -0.5 |
| fp32_all | 170,735 | 11,705 | measured | 24.6 | 22.8 | +1.2 | 26.5 | -1.2 |
| noise_s0 | 241,651 | — | model | 25.1 | 23.3 | +1.1 | — | — |
| noise_s1 | 241,651 | — | model | 24.5 | 23.3 | +0.7 | — | — |
| noise_s2 | 241,651 | — | model | 25.1 | 23.3 | +1.1 | — | — |

Bold z: the arm is off that curve (|z| > 2). Noise arms are not in either fit.


Plot: `research/plots/sq2_dose_response.png`

## Question-level fragility from the three noise seeds only (uncapped, strict)

Per-seed churn: 25.1%, 24.5%, 25.1%.

| seeds changing the answer | observed | expected if independent of the question |
|---:|---:|---:|
| 0 | 468 | 341.2 |
| 1 | 156 | 339.1 |
| 2 | 98 | 112.3 |
| 3 | 83 | 12.4 |

Pairwise Jaccard: observed 0.43, 0.42, 0.37 (mean 0.41); expected under independence at these rates ≈ 0.14. 3-of-3: observed 83 (10.3%), expected 12.4 (1.5%).

Beta-binomial fit over the 3 seeds: per-question flip probability p ~ Beta(α = 0.322, β = 0.969) [95% bootstrap: α 0.265–0.394, β 0.789–1.199]; mean p = 24.9% [22.7, 27.4]. Implied fraction of questions with p > 0.5: **20.7%** [18.0, 23.8].
 Implied fraction with p < 0.1: 47.0%, 0.25: 63.3%, 0.5: 79.3%.
Three trials per question identify the spread of p only coarsely; the bootstrap CI is the honest statement of that.

## Baseline for the blocks 6–7 result

251 uncapped questions have their control answer line in blocks 6–7.

| arm | churn on those questions [95% CI] |
|---|---:|
| fp32_all | 27.9% (70/251) [22.7, 33.7] |
| fp32_steps_global_early | 19.9% (50/251) [15.4, 25.3] |
| fp32_steps_global_last25 | 2.8% (7/251) [1.4, 5.6] |

## Accuracy noise floor and Holm correction (repo grader, all 1319 questions)

Discordant (correctness-churn) rate d: fp32_all 13.9%; across the 14 arms 1.1–15.0%. Under symmetric churn (no true effect) the paired accuracy difference has SD = √(d/n).

| n | SD of paired Δacc at d = fp32_all | 95% half-width | SD at the lowest-churn arm's d |
|---:|---:|---:|---:|
| 200 | 2.63 pt | ±5.16 pt | 0.73 pt |
| 500 | 1.67 pt | ±3.26 pt | 0.46 pt |
| 1000 | 1.18 pt | ±2.31 pt | 0.33 pt |
| 1319 | 1.03 pt | ±2.01 pt | 0.28 pt |

Check against the observed arms: z = (only-arm − only-control)/√(discordant) over the 14 arms has mean +0.58, SD 0.76 (≈ 1 expected under no effect; the arms share one control and overlapping fragile questions, so they are not independent draws). Largest |Δacc| = 1.1 pt, against a 95% half-width of ±2.0 pt at n = 1319. §9b's "±3 pt at n=200" corresponds to the 1-SD value at d ≈ 18.0%.

| arm | only control ✓ | only arm ✓ | Δacc | McNemar p | Holm-adjusted p | survives (α = 0.05) |
|---|---:|---:|---:|---:|---:|---|
| fp32_steps_global_last25 | 3 | 11 | +0.6 pt | 0.0574 | 0.803 | no |
| fp32_steps_within_last25 | 28 | 42 | +1.1 pt | 0.12 | 1 | no |
| fp32_finalize | 63 | 76 | +1.0 pt | 0.309 | 1 | no |
| fp32_all | 85 | 98 | +1.0 pt | 0.375 | 1 | no |
| fp32_layer0 | 87 | 100 | +1.0 pt | 0.38 | 1 | no |
| noise_s0 | 93 | 105 | +0.9 pt | 0.434 | 1 | no |
| noise_s2 | 94 | 85 | -0.7 pt | 0.55 | 1 | no |
| fp32_layers11_14 | 49 | 54 | +0.4 pt | 0.694 | 1 | no |
| fp32_layer15 | 32 | 36 | +0.3 pt | 0.716 | 1 | no |
| fp32_steps_within_late | 75 | 72 | -0.2 pt | 0.869 | 1 | no |
| noise_s1 | 83 | 86 | +0.2 pt | 0.878 | 1 | no |
| fp32_prime | 86 | 89 | +0.2 pt | 0.88 | 1 | no |
| fp32_steps_within_early | 54 | 52 | -0.2 pt | 0.923 | 1 | no |
| fp32_steps_global_early | 77 | 79 | +0.2 pt | 0.936 | 1 | no |
