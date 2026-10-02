Capped = control's first EOS at the last position (255) or absent; see is_capped().

Questions: 1319; uncapped in control: 805; capped: 514 (39.0%). Control strict answer missing: 0 (uncapped: 0).

### Cross-container reproducibility (control's first 64 questions, re-run in containers B and C)

- control_sentinel@p2B: 64/64 identical in every committed token and commit step
- control_sentinel@p2C: 64/64 identical in every committed token and commit step

### Churn per arm vs control

Primary = strict extraction, uncapped questions. Correctness and McNemar use the repo's grader.

| arm | perturbation | strict churn, uncapped (primary) | strict churn, all | lenient churn, all | acc ctrl → arm | McNemar p | trajectories diverged | median first-divergence step [IQR] |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| fp32_all | fp32, every pass | 11.9 [9.9, 14.3] | 22.3 [20.1, 24.6] | 22.3 | 75.7 → 76.6 | 0.375 | 100.0% | 4 [1–10] |
| fp32_prime | fp32, prime pass only | 12.3 [10.2, 14.7] | 22.0 [19.8, 24.3] | 22.0 | 75.7 → 75.9 | 0.88 | 100.0% | 4 [1–10] |
| fp32_finalize | fp32, finalize passes only | 9.2 [7.4, 11.4] | 18.4 [16.4, 20.6] | 18.4 | 75.7 → 76.6 | 0.309 | 100.0% | 38 [34–45] |
| fp32_steps_within_early | fp32, steps 0–7 of each block | 6.8 [5.3, 8.8] | 13.3 [11.6, 15.3] | 13.3 | 75.7 → 75.5 | 0.923 | 100.0% | 5 [2–36] |
| fp32_steps_within_late | fp32, steps 8–31 of each block | 9.7 [7.8, 11.9] | 19.3 [17.3, 21.6] | 19.3 | 75.7 → 75.4 | 0.869 | 100.0% | 14 [11–20] |
| fp32_steps_within_last25 | fp32, steps 24–31 of each block | 4.2 [3.0, 5.8] | 10.8 [9.3, 12.6] | 10.8 | 75.7 → 76.7 | 0.12 | 99.9% | 31 [26–63] |
| fp32_steps_global_early | fp32, steps of blocks 0–1 | 10.3 [8.4, 12.6] | 19.3 [17.3, 21.6] | 19.3 | 75.7 → 75.8 | 0.936 | 100.0% | 5 [2–12] |
| fp32_steps_global_last25 | fp32, steps of blocks 6–7 | 0.2 [0.1, 0.9] | 2.7 [1.9, 3.7] | 2.7 | 75.7 → 76.3 | 0.0574 | 99.5% | 196 [193–202] |
| fp32_layer0 | fp32, layer 0 only, every pass | 12.9 [10.8, 15.4] | 23.1 [20.9, 25.5] | 23.1 | 75.7 → 76.6 | 0.38 | 100.0% | 4 [1–10] |
| fp32_layers11_14 | fp32, layers 11–14 only, every pass | 6.2 [4.7, 8.1] | 13.5 [11.8, 15.4] | 13.5 | 75.7 → 76.0 | 0.694 | 100.0% | 14 [5–30] |
| fp32_layer15 | fp32, layer 15 only, every pass | 3.0 [2.0, 4.4] | 8.5 [7.1, 10.1] | 8.5 | 75.7 → 76.0 | 0.716 | 99.3% | 31 [13–61] |
| noise_s0 | bf16 + U(±1 ULP) noise, seed 0 | 12.7 [10.5, 15.1] | 23.3 [21.1, 25.6] | 23.3 | 75.7 → 76.6 | 0.434 | 100.0% | 3 [1–10] |
| noise_s1 | bf16 + U(±1 ULP) noise, seed 1 | 12.3 [10.2, 14.7] | 21.8 [19.7, 24.1] | 21.8 | 75.7 → 75.9 | 0.878 | 100.0% | 3 [1–9] |
| noise_s2 | bf16 + U(±1 ULP) noise, seed 2 | 13.4 [11.2, 15.9] | 22.1 [19.9, 24.4] | 22.1 | 75.7 → 75.0 | 0.55 | 100.0% | 4 [1–10] |

First-divergence step is global: block = step // 32.

### Windows: churn normalized by the share of the run perturbed

| arm | share of denoising steps | share of forward passes | share of routed token-rows | flip rate inside window (Phase 1 telemetry, control, n=200) | strict churn, uncapped | churn / step share |
|---|---:|---:|---:|---:|---:|---:|
| fp32_all | 100.0% | 100.0% | 100.0% | 26.8% | 11.9 | 0.119 |
| fp32_prime | 0.0% | 0.4% | 4.5% | 24.1% | 12.3 | — |
| fp32_finalize | 0.0% | 3.0% | 2.9% | 26.4% | 9.2 | — |
| fp32_steps_within_early | 25.0% | 24.2% | 23.2% | 27.4% | 6.8 | 0.273 |
| fp32_steps_within_late | 75.0% | 72.5% | 69.5% | 26.8% | 9.7 | 0.129 |
| fp32_steps_within_last25 | 25.0% | 24.2% | 23.2% | 26.6% | 4.2 | 0.169 |
| fp32_steps_global_early | 25.0% | 24.2% | 38.6% | 27.3% | 10.3 | 0.412 |
| fp32_steps_global_last25 | 25.0% | 24.2% | 7.7% | 25.5% | 0.2 | 0.010 |
| fp32_layer0 (all layers' passes counted; fp32 only at the named layers) | 100.0% | 100.0% | 100.0% | 94.8% | 12.9 | — |
| fp32_layers11_14 (all layers' passes counted; fp32 only at the named layers) | 100.0% | 100.0% | 100.0% | 10.0% | 6.2 | — |
| fp32_layer15 (all layers' passes counted; fp32 only at the named layers) | 100.0% | 100.0% | 100.0% | 14.4% | 3.0 | — |

### Pairwise window contrasts (paired over the same uncapped questions)

Each cell: questions churned only under the first arm vs only under the second; exact McNemar p.

| contrast | only first | only second | churn first | churn second | McNemar p |
|---|---:|---:|---:|---:|---:|
| fp32_steps_within_early vs fp32_steps_within_last25 | 38 | 17 | 6.8 | 4.2 | 0.00646 |
| fp32_steps_global_early vs fp32_steps_global_last25 | 83 | 2 | 10.3 | 0.2 | 1.89e-22 |
| fp32_steps_within_early vs fp32_steps_within_late | 31 | 54 | 6.8 | 9.7 | 0.0165 |
| fp32_prime vs fp32_finalize | 54 | 29 | 12.3 | 9.2 | 0.00804 |
| fp32_layer0 vs fp32_layer15 | 88 | 8 | 12.9 | 3.0 | 3.67e-18 |
| fp32_layer0 vs fp32_layers11_14 | 74 | 20 | 12.9 | 6.2 | 1.85e-08 |
| fp32_layers11_14 vs fp32_layer15 | 41 | 15 | 6.2 | 3.0 | 0.000686 |

### Additivity: does the partition reproduce fp32_all? (uncapped, strict)

- sum of part churn counts: 306; |union|: 162; fp32_all churn: 96
- fp32_all churned questions also churned by at least one part: 80/96; Jaccard(union, fp32_all) = 0.45

### Perturbation family: noise vs fp32 (uncapped, strict)

Caveat: U(−1, +1) ULP noise has mean |δ| = 0.5 ULP per logit and reaches 1 ULP; bf16 rounding moves a logit by at most 0.5 ULP (mean 0.25). The noise arms are the larger perturbation, so equal churn would mean saturation, not equivalence.

- fp32_all: 11.9%; noise seeds: 12.7%, 12.3%, 13.4% (mean 12.8%, range 1.1 pts)
- pairwise overlap of churned sets (Jaccard):
  - noise_s0 / noise_s1: 0.40
  - noise_s0 / noise_s2: 0.45
  - noise_s0 / fp32_all: 0.43
  - noise_s1 / noise_s2: 0.41
  - noise_s1 / fp32_all: 0.46
  - noise_s2 / fp32_all: 0.44

### Saturation: how many perturbed arms change each question's strict answer (uncapped)

14 perturbed arms. Null = each arm churns independently of the question, at its own observed rate (Poisson-binomial).

| arms changing the answer | questions observed | expected under independence |
|---:|---:|---:|
| 0 | 575 | 213.9 |
| 1 | 53 | 300.5 |
| 2 | 42 | 192.6 |
| 3 | 24 | 74.5 |
| 4 | 19 | 19.4 |
| 5 | 17 | 3.6 |
| 6 | 13 | 0.5 |
| 7 | 17 | 0.0 |
| 8 | 8 | 0.0 |
| 9 | 15 | 0.0 |
| 10 | 7 | 0.0 |
| 11 | 8 | 0.0 |
| 12 | 7 | 0.0 |
| 13 | 0 | 0.0 |
| 14 | 0 | 0.0 |

Churn events (question × arm): 1008. Questions churned by ≥ half the arms: 62 (7.7% of uncapped) carry 560 events (55.6%). Questions churned by any arm: 230.

### Do control-arm signals predict fragility? (uncapped questions)

Target A: churn count k over the perturbed arms (Spearman ρ). Target B: churned under fp32_all (AUC = P(signal of a churned question > signal of a stable one)). 95% bootstrap CIs, 2000 resamples.

| signal | n | Spearman ρ with k [95% CI] | AUC for fp32_all churn [95% CI] |
|---|---:|---:|---:|
| answer-token confidence at commit (min over the number's tokens) | 717 | +0.027 [-0.045, +0.104] | 0.494 [0.423, 0.566] |
| mean commit confidence, whole generation | 805 | -0.440 [-0.495, -0.379] | 0.237 [0.189, 0.290] |
| generated length (tokens before EOS) | 805 | +0.210 [+0.143, +0.272] | 0.615 [0.556, 0.674] |
| first-divergence step vs fp32_all (256 if none) | 805 | -0.014 [-0.085, +0.057] | 0.474 [0.412, 0.534] |

Capped status (all 1319 questions, strict): mean k = 4.21 capped vs 1.25 uncapped; AUC(capped → k ≥ 1) = 0.684.
