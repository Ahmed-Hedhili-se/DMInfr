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


Plot: `research/data/by_answer_definition/strict/sq2_dose_response.png`

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
