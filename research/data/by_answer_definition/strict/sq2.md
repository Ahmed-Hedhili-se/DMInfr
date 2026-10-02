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
