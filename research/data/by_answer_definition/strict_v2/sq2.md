Capped = control's first EOS at the last position (255) or absent; see is_capped().

Questions: 1319; uncapped in control: 805; capped: 514 (39.0%). Control strict answer missing: 104 (uncapped: 32).

### Cross-container reproducibility (control's first 64 questions, re-run in containers B and C)

- control_sentinel@p2B: 64/64 identical in every committed token and commit step
- control_sentinel@p2C: 64/64 identical in every committed token and commit step

### Churn per arm vs control

Primary = strict extraction, uncapped questions. Correctness and McNemar use the repo's grader.

| arm | perturbation | strict churn, uncapped (primary) | strict churn, all | lenient churn, all | acc ctrl → arm | McNemar p | trajectories diverged | median first-divergence step [IQR] |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| fp32_all | fp32, every pass | 17.3 [14.8, 20.0] | 29.3 [26.9, 31.8] | 22.3 | 75.7 → 76.6 | 0.375 | 100.0% | 4 [1–10] |
| fp32_prime | fp32, prime pass only | 17.1 [14.7, 19.9] | 28.5 [26.1, 31.0] | 22.0 | 75.7 → 75.9 | 0.88 | 100.0% | 4 [1–10] |
| fp32_finalize | fp32, finalize passes only | 12.5 [10.4, 15.0] | 23.4 [21.1, 25.7] | 18.4 | 75.7 → 76.6 | 0.309 | 100.0% | 38 [34–45] |
| fp32_steps_within_early | fp32, steps 0–7 of each block | 9.9 [8.1, 12.2] | 17.7 [15.7, 19.8] | 13.3 | 75.7 → 75.5 | 0.923 | 100.0% | 5 [2–36] |
| fp32_steps_within_late | fp32, steps 8–31 of each block | 14.0 [11.8, 16.6] | 24.9 [22.7, 27.3] | 19.3 | 75.7 → 75.4 | 0.869 | 100.0% | 14 [11–20] |
| fp32_steps_within_last25 | fp32, steps 24–31 of each block | 6.0 [4.5, 7.8] | 13.9 [12.1, 15.8] | 10.8 | 75.7 → 76.7 | 0.12 | 99.9% | 31 [26–63] |
| fp32_steps_global_early | fp32, steps of blocks 0–1 | 12.7 [10.5, 15.1] | 24.2 [22.0, 26.6] | 19.3 | 75.7 → 75.8 | 0.936 | 100.0% | 5 [2–12] |
| fp32_steps_global_last25 | fp32, steps of blocks 6–7 | 0.9 [0.4, 1.8] | 5.6 [4.5, 7.0] | 2.7 | 75.7 → 76.3 | 0.0574 | 99.5% | 196 [193–202] |
| fp32_layer0 | fp32, layer 0 only, every pass | 17.6 [15.2, 20.4] | 29.5 [27.1, 32.0] | 23.1 | 75.7 → 76.6 | 0.38 | 100.0% | 4 [1–10] |
| fp32_layers11_14 | fp32, layers 11–14 only, every pass | 8.3 [6.6, 10.4] | 17.1 [15.1, 19.2] | 13.5 | 75.7 → 76.0 | 0.694 | 100.0% | 14 [5–30] |
| fp32_layer15 | fp32, layer 15 only, every pass | 4.3 [3.1, 6.0] | 10.6 [9.1, 12.4] | 8.5 | 75.7 → 76.0 | 0.716 | 99.3% | 31 [13–61] |
| noise_s0 | bf16 + U(±1 ULP) noise, seed 0 | 17.8 [15.3, 20.6] | 29.7 [27.3, 32.2] | 23.3 | 75.7 → 76.6 | 0.434 | 100.0% | 3 [1–10] |
| noise_s1 | bf16 + U(±1 ULP) noise, seed 1 | 17.4 [14.9, 20.2] | 28.4 [26.1, 30.9] | 21.8 | 75.7 → 75.9 | 0.878 | 100.0% | 3 [1–9] |
| noise_s2 | bf16 + U(±1 ULP) noise, seed 2 | 17.4 [14.9, 20.2] | 28.0 [25.6, 30.5] | 22.1 | 75.7 → 75.0 | 0.55 | 100.0% | 4 [1–10] |

First-divergence step is global: block = step // 32.

### Windows: churn normalized by the share of the run perturbed

| arm | share of denoising steps | share of forward passes | share of routed token-rows | flip rate inside window (Phase 1 telemetry, control, n=200) | strict churn, uncapped | churn / step share |
|---|---:|---:|---:|---:|---:|---:|
| fp32_all | 100.0% | 100.0% | 100.0% | 26.8% | 17.3 | 0.173 |
| fp32_prime | 0.0% | 0.4% | 4.5% | 24.1% | 17.1 | — |
| fp32_finalize | 0.0% | 3.0% | 2.9% | 26.4% | 12.5 | — |
| fp32_steps_within_early | 25.0% | 24.2% | 23.2% | 27.4% | 9.9 | 0.398 |
| fp32_steps_within_late | 75.0% | 72.5% | 69.5% | 26.8% | 14.0 | 0.187 |
| fp32_steps_within_last25 | 25.0% | 24.2% | 23.2% | 26.6% | 6.0 | 0.239 |
| fp32_steps_global_early | 25.0% | 24.2% | 38.6% | 27.3% | 12.7 | 0.507 |
| fp32_steps_global_last25 | 25.0% | 24.2% | 7.7% | 25.5% | 0.9 | 0.035 |
| fp32_layer0 (all layers' passes counted; fp32 only at the named layers) | 100.0% | 100.0% | 100.0% | 94.8% | 17.6 | — |
| fp32_layers11_14 (all layers' passes counted; fp32 only at the named layers) | 100.0% | 100.0% | 100.0% | 10.0% | 8.3 | — |
| fp32_layer15 (all layers' passes counted; fp32 only at the named layers) | 100.0% | 100.0% | 100.0% | 14.4% | 4.3 | — |

### Pairwise window contrasts (paired over the same uncapped questions)

Each cell: questions churned only under the first arm vs only under the second; exact McNemar p.

| contrast | only first | only second | churn first | churn second | McNemar p |
|---|---:|---:|---:|---:|---:|
| fp32_steps_within_early vs fp32_steps_within_last25 | 59 | 27 | 9.9 | 6.0 | 0.000732 |
| fp32_steps_global_early vs fp32_steps_global_last25 | 102 | 7 | 12.7 | 0.9 | 9.84e-23 |
| fp32_steps_within_early vs fp32_steps_within_late | 46 | 79 | 9.9 | 14.0 | 0.00402 |
| fp32_prime vs fp32_finalize | 79 | 42 | 17.1 | 12.5 | 0.000983 |
| fp32_layer0 vs fp32_layer15 | 119 | 12 | 17.6 | 4.3 | 2.59e-23 |
| fp32_layer0 vs fp32_layers11_14 | 104 | 29 | 17.6 | 8.3 | 4.12e-11 |
| fp32_layers11_14 vs fp32_layer15 | 50 | 18 | 8.3 | 4.3 | 0.000131 |

### Additivity: does the partition reproduce fp32_all? (uncapped, strict)

- sum of part churn counts: 432; |union|: 244; fp32_all churn: 139
- fp32_all churned questions also churned by at least one part: 114/139; Jaccard(union, fp32_all) = 0.42

### Perturbation family: noise vs fp32 (uncapped, strict)

Caveat: U(−1, +1) ULP noise has mean |δ| = 0.5 ULP per logit and reaches 1 ULP; bf16 rounding moves a logit by at most 0.5 ULP (mean 0.25). The noise arms are the larger perturbation, so equal churn would mean saturation, not equivalence.

- fp32_all: 17.3%; noise seeds: 17.8%, 17.4%, 17.4% (mean 17.5%, range 0.4 pts)
- pairwise overlap of churned sets (Jaccard):
  - noise_s0 / noise_s1: 0.41
  - noise_s0 / noise_s2: 0.45
  - noise_s0 / fp32_all: 0.40
  - noise_s1 / noise_s2: 0.40
  - noise_s1 / fp32_all: 0.40
  - noise_s2 / fp32_all: 0.42

### Saturation: how many perturbed arms change each question's strict answer (uncapped)

14 perturbed arms. Null = each arm churns independently of the question, at its own observed rate (Poisson-binomial).

| arms changing the answer | questions observed | expected under independence |
|---:|---:|---:|
| 0 | 453 | 123.3 |
| 1 | 112 | 251.2 |
| 2 | 50 | 233.5 |
| 3 | 41 | 131.1 |
| 4 | 30 | 49.6 |
| 5 | 22 | 13.3 |
| 6 | 19 | 2.6 |
| 7 | 18 | 0.4 |
| 8 | 14 | 0.0 |
| 9 | 14 | 0.0 |
| 10 | 11 | 0.0 |
| 11 | 11 | 0.0 |
| 12 | 9 | 0.0 |
| 13 | 1 | 0.0 |
| 14 | 0 | 0.0 |

Churn events (question × arm): 1395. Questions churned by ≥ half the arms: 78 (9.7% of uncapped) carry 716 events (51.3%). Questions churned by any arm: 352.

### Do control-arm signals predict fragility? (uncapped questions)

Target A: churn count k over the perturbed arms (Spearman ρ). Target B: churned under fp32_all (AUC = P(signal of a churned question > signal of a stable one)). 95% bootstrap CIs, 2000 resamples.

| signal | n | Spearman ρ with k [95% CI] | AUC for fp32_all churn [95% CI] |
|---|---:|---:|---:|
| answer-token confidence at commit (min over the number's tokens) | 717 | +0.011 [-0.063, +0.084] | 0.496 [0.436, 0.557] |
| mean commit confidence, whole generation | 805 | -0.436 [-0.493, -0.376] | 0.288 [0.242, 0.335] |
| generated length (tokens before EOS) | 805 | +0.254 [+0.189, +0.318] | 0.610 [0.559, 0.658] |
| first-divergence step vs fp32_all (256 if none) | 805 | -0.020 [-0.089, +0.049] | 0.462 [0.412, 0.513] |

Capped status (all 1319 questions, strict): mean k = 5.26 capped vs 1.73 uncapped; AUC(capped → k ≥ 1) = 0.691.
