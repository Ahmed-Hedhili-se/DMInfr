## Churn decomposed into transitions

C = graded correct by the repo grader, W = wrong; W→W′ = the strict answer changed and both are wrong. Populations: LLaDA's own uncapped set (805); the 802 uncapped in both models' controls for the cross-model rows; the 684 uncapped under both schedules for the threshold pair.

| model / arm | population | churned | C→W | W→C | W→W′ | C→C′ | net accuracy change |
|---|---:|---:|---:|---:|---:|---:|---:|
| LLaDA fp32_all | 805 | 96 (11.9%) | 34 | 35 | 27 (28.1% of churn) | 0 | +1 (+0.1 pt) |
| LLaDA fp32_prime | 805 | 99 (12.3%) | 39 | 32 | 28 (28.3% of churn) | 0 | -7 (-0.9 pt) |
| LLaDA fp32_finalize | 805 | 74 (9.2%) | 28 | 23 | 23 (31.1% of churn) | 0 | -5 (-0.6 pt) |
| LLaDA fp32_steps_within_early | 805 | 55 (6.8%) | 19 | 21 | 15 (27.3% of churn) | 0 | +2 (+0.2 pt) |
| LLaDA fp32_steps_within_late | 805 | 78 (9.7%) | 30 | 26 | 22 (28.2% of churn) | 0 | -4 (-0.5 pt) |
| LLaDA fp32_steps_within_last25 | 805 | 34 (4.2%) | 11 | 10 | 13 (38.2% of churn) | 0 | -1 (-0.1 pt) |
| LLaDA fp32_steps_global_early | 805 | 83 (10.3%) | 35 | 29 | 19 (22.9% of churn) | 0 | -6 (-0.7 pt) |
| LLaDA fp32_steps_global_last25 | 805 | 2 (0.2%) | 0 | 1 | 1 (50.0% of churn) | 0 | +1 (+0.1 pt) |
| LLaDA fp32_layer0 | 805 | 104 (12.9%) | 43 | 34 | 27 (26.0% of churn) | 0 | -9 (-1.1 pt) |
| LLaDA fp32_layers11_14 | 805 | 50 (6.2%) | 20 | 15 | 15 (30.0% of churn) | 0 | -5 (-0.6 pt) |
| LLaDA fp32_layer15 | 805 | 24 (3.0%) | 10 | 8 | 6 (25.0% of churn) | 0 | -2 (-0.2 pt) |
| LLaDA noise_s0 | 805 | 102 (12.7%) | 44 | 32 | 26 (25.5% of churn) | 0 | -12 (-1.5 pt) |
| LLaDA noise_s1 | 805 | 99 (12.3%) | 41 | 31 | 27 (27.3% of churn) | 0 | -10 (-1.2 pt) |
| LLaDA noise_s2 | 805 | 108 (13.4%) | 45 | 38 | 25 (23.1% of churn) | 0 | -7 (-0.9 pt) |
| LLaDA threshold: control vs fp32_all | 684 | 93 (13.6%) | 35 | 37 | 21 (22.6% of churn) | 0 | +2 (+0.3 pt) |
| LLaDA fixed: control vs fp32_all (same 684) | 684 | 63 (9.2%) | 21 | 26 | 16 (25.4% of churn) | 0 | +5 (+0.7 pt) |

### Same 802 questions, both models

| model / arm | population | churned | C→W | W→C | W→W′ | C→C′ | net accuracy change |
|---|---:|---:|---:|---:|---:|---:|---:|
| LLaDA fp32_all | 802 | 96 (12.0%) | 34 | 35 | 27 (28.1% of churn) | 0 | +1 (+0.1 pt) |
| LLaDA fp32_prime | 802 | 99 (12.3%) | 39 | 32 | 28 (28.3% of churn) | 0 | -7 (-0.9 pt) |
| LLaDA noise_s0 | 802 | 101 (12.6%) | 43 | 32 | 26 (25.7% of churn) | 0 | -11 (-1.4 pt) |
| OLMoE fp32_all | 802 | 155 (19.3%) | 28 | 25 | 102 (65.8% of churn) | 0 | -3 (-0.4 pt) |
| OLMoE fp32_prefill | 802 | 162 (20.2%) | 27 | 39 | 96 (59.3% of churn) | 0 | +12 (+1.5 pt) |
| OLMoE noise_s0 | 802 | 163 (20.3%) | 28 | 37 | 98 (60.1% of churn) | 0 | +9 (+1.1 pt) |

C→W and W→C are near-balanced in every arm (hence no accuracy effect); the excess of churn over |C→W − W→C| is answers moving between wrong values.
