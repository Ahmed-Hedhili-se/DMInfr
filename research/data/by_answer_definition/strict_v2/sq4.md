## SQ4: threshold decoding (0.9 / 0.4) vs the fixed schedule

Setup: n=1319 (full GSM8K test set, seed-42 order), A100-40GB, `max_tokens=256 block_length=32 steps=256` (32 steps per block = block length, ratio 1.0), B=32, identical batches in all four arms. Fixed pair from Phase 2 (container A); threshold pair from Phase 4. Threshold = `select_transfer_indices_hierarchy`, confidence 0.9, low 0.4.

Denoising steps actually run (control, summed over 1319 questions in batches): fixed 10,752 step passes, threshold 6,672 (62.1% of fixed).

Capped in control: fixed 514/1319 (39.0%), threshold 545/1319 (41.3%). Uncapped under both: 684.

### Churn of each pair (control vs fp32_all)

| population | fixed: strict churn [95% CI] | threshold: strict churn [95% CI] | churn only fixed | churn only threshold | McNemar p (fixed vs threshold churn) |
|---|---:|---:|---:|---:|---:|
| uncapped under both (primary) (n=684) | 15.4 [12.8, 18.2] | 20.8 [17.9, 24.0] | 49 | 86 | 0.00184 |
| all 1319 (n=1319) | 29.3 [26.9, 31.8] | 35.5 [32.9, 38.1] | 147 | 229 | 2.76e-05 |

- fixed, own uncapped (n=805): 17.3% [14.8, 20.0]

- threshold, own uncapped (n=774): 23.4% [20.5, 26.5]

### Accuracy (repo grader, all 1319; paired McNemar)

| comparison | acc A | acc B | only A ✓ | only B ✓ | McNemar p |
|---|---:|---:|---:|---:|---:|
| fixed: control vs fp32_all | 75.7 | 76.6 | 85 | 98 | 0.375 |
| threshold: control vs fp32_all | 74.2 | 74.7 | 104 | 110 | 0.733 |
| control: fixed vs threshold | 75.7 | 74.2 | 111 | 92 | 0.206 |
| fp32_all: fixed vs threshold | 76.6 | 74.7 | 110 | 84 | 0.0724 |

The schedule switch itself, as a perturbation (control fixed vs control threshold, strict, uncapped under both): 14.9% [12.4, 17.8] of answers change.

### Trajectory divergence within each pair

| pair | diverged | median first-divergence step [IQR] |
|---|---:|---:|
| fixed | 684/684 | 4 [2–10] |
| threshold | 684/684 | 3 [1–5] |

Steps are global (block × 32 + step); under threshold decoding a block can finish early, so a step index there is an upper bound on elapsed forward passes.

### Decoding collapse (h100x2_bench.md §4d/§8c): one token repeated ≥ 12 times in a row

| arm | collapses / 1319 | longest repeat run, max | questions with run ≥ 8 |
|---|---:|---:|---:|
| fixed control | 0 | 4 | 0 |
| fixed fp32_all | 0 | 7 | 0 |
| threshold control | 0 | 4 | 0 |
| threshold fp32_all | 0 | 5 | 0 |

This config runs 32 steps per block (ratio 1.0), the setting §8c found removes the collapse; the §4d failures were at ratio 0.5, a 1024-token budget and block length 64, which this study does not test.

