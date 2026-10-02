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

**Model A, flip count alone** (11 fp32 arms, weighted by binomial SE): churn = c_max·(1 − exp(−k·flips)); c_max = 18.0% ± 1.5, k = 1.61e-05 ± 2.3e-06; χ² = 374 on 9 dof.

**Model B, cache-write flips weighted separately**: x = step flips + w·(prime + finalize flips); c_max = 19.2%, k = 1.07e-05, w = 25.8; χ² = 60 on 8 dof.

| arm | expected flips / question | of which in cache-writing passes | source | churn (primary) | model A | z (A) | model B | z (B) |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| fp32_finalize | 4,875 | 4,875 | measured | 12.5 | 1.4 | **+9.6** | 14.2 | -1.4 |
| fp32_layer15 | 5,719 | 1,104 | measured | 4.3 | 1.6 | **+3.8** | 5.7 | -1.9 |
| fp32_prime | 6,830 | 6,830 | measured | 17.1 | 1.9 | **+11.5** | 16.3 | +0.6 |
| fp32_steps_global_last25 | 12,549 | 0 | measured | 0.9 | 3.3 | **-7.4** | 2.4 | **-4.7** |
| fp32_layers11_14 | 15,929 | 1,110 | measured | 8.3 | 4.1 | **+4.4** | 7.1 | +1.2 |
| fp32_layer0 | 37,727 | 2,658 | measured | 17.6 | 8.2 | **+7.0** | 12.9 | **+3.6** |
| fp32_steps_within_last25 | 39,150 | 0 | measured | 6.0 | 8.4 | **-2.9** | 6.6 | -0.7 |
| fp32_steps_within_early | 40,369 | 0 | measured | 9.9 | 8.6 | +1.3 | 6.7 | **+3.0** |
| fp32_steps_global_early | 67,170 | 0 | measured | 12.7 | 11.9 | +0.7 | 9.8 | **+2.4** |
| fp32_steps_within_late | 118,661 | 0 | measured | 14.0 | 15.4 | -1.1 | 13.8 | +0.2 |
| fp32_all | 170,735 | 11,705 | measured | 17.3 | 16.9 | +0.3 | 19.1 | -1.4 |
| noise_s0 | 241,651 | — | model | 17.8 | 17.7 | +0.1 | — | — |
| noise_s1 | 241,651 | — | model | 17.4 | 17.7 | -0.2 | — | — |
| noise_s2 | 241,651 | — | model | 17.4 | 17.7 | -0.2 | — | — |

Bold z: the arm is off that curve (|z| > 2). Noise arms are not in either fit.


Plot: `research/data/by_answer_definition/strict_v2/sq2_dose_response.png`

## Question-level fragility from the three noise seeds only (uncapped, strict)

Per-seed churn: 17.8%, 17.4%, 17.4%.

| seeds changing the answer | observed | expected if independent of the question |
|---:|---:|---:|
| 0 | 571 | 451.8 |
| 1 | 107 | 287.8 |
| 2 | 65 | 61.1 |
| 3 | 62 | 4.3 |

Pairwise Jaccard: observed 0.41, 0.45, 0.40 (mean 0.42); expected under independence at these rates ≈ 0.10. 3-of-3: observed 62 (7.7%), expected 4.3 (0.5%).

Beta-binomial fit over the 3 seeds: per-question flip probability p ~ Beta(α = 0.171, β = 0.804) [95% bootstrap: α 0.132–0.217, β 0.609–1.023]; mean p = 17.6% [15.4, 19.6]. Implied fraction of questions with p > 0.5: **14.6%** [11.9, 16.9].
 Implied fraction with p < 0.1: 63.9%, 0.25: 75.1%, 0.5: 85.4%.
Three trials per question identify the spread of p only coarsely; the bootstrap CI is the honest statement of that.

## Baseline for the blocks 6–7 result

251 uncapped questions have their control answer line in blocks 6–7.

| arm | churn on those questions [95% CI] |
|---|---:|
| fp32_all | 23.5% (59/251) [18.7, 29.1] |
| fp32_steps_global_early | 17.1% (43/251) [13.0, 22.3] |
| fp32_steps_global_last25 | 2.4% (6/251) [1.1, 5.1] |

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
