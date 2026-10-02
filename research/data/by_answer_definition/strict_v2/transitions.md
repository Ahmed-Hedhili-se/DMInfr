## Churn decomposed into transitions

C = graded correct by the repo grader, W = wrong; W→W′ = the strict answer changed and both are wrong. Populations: LLaDA's own uncapped set (805); the 802 uncapped in both models' controls for the cross-model rows; the 684 uncapped under both schedules for the threshold pair.

| model / arm | population | churned | C→W | W→C | W→W′ | C→C′ | net accuracy change |
|---|---:|---:|---:|---:|---:|---:|---:|
| LLaDA fp32_all | 805 | 139 (17.3%) | 33 | 35 | 27 (19.4% of churn) | 44 | +2 (+0.2 pt) |
| LLaDA fp32_prime | 805 | 138 (17.1%) | 37 | 32 | 29 (21.0% of churn) | 40 | -5 (-0.6 pt) |
| LLaDA fp32_finalize | 805 | 101 (12.5%) | 27 | 23 | 23 (22.8% of churn) | 28 | -4 (-0.5 pt) |
| LLaDA fp32_steps_within_early | 805 | 80 (9.9%) | 19 | 21 | 16 (20.0% of churn) | 24 | +2 (+0.2 pt) |
| LLaDA fp32_steps_within_late | 805 | 113 (14.0%) | 29 | 25 | 24 (21.2% of churn) | 35 | -4 (-0.5 pt) |
| LLaDA fp32_steps_within_last25 | 805 | 48 (6.0%) | 11 | 9 | 16 (33.3% of churn) | 12 | -2 (-0.2 pt) |
| LLaDA fp32_steps_global_early | 805 | 102 (12.7%) | 35 | 28 | 21 (20.6% of churn) | 18 | -7 (-0.9 pt) |
| LLaDA fp32_steps_global_last25 | 805 | 7 (0.9%) | 0 | 1 | 2 (28.6% of churn) | 4 | +1 (+0.1 pt) |
| LLaDA fp32_layer0 | 805 | 142 (17.6%) | 43 | 33 | 28 (19.7% of churn) | 38 | -10 (-1.2 pt) |
| LLaDA fp32_layers11_14 | 805 | 67 (8.3%) | 17 | 14 | 14 (20.9% of churn) | 22 | -3 (-0.4 pt) |
| LLaDA fp32_layer15 | 805 | 35 (4.3%) | 10 | 8 | 6 (17.1% of churn) | 11 | -2 (-0.2 pt) |
| LLaDA noise_s0 | 805 | 143 (17.8%) | 43 | 32 | 28 (19.6% of churn) | 40 | -11 (-1.4 pt) |
| LLaDA noise_s1 | 805 | 140 (17.4%) | 41 | 31 | 32 (22.9% of churn) | 36 | -10 (-1.2 pt) |
| LLaDA noise_s2 | 805 | 140 (17.4%) | 44 | 38 | 27 (19.3% of churn) | 31 | -6 (-0.7 pt) |
| LLaDA threshold: control vs fp32_all | 684 | 142 (20.8%) | 35 | 37 | 22 (15.5% of churn) | 48 | +2 (+0.3 pt) |
| LLaDA fixed: control vs fp32_all (same 684) | 684 | 105 (15.4%) | 21 | 26 | 16 (15.2% of churn) | 42 | +5 (+0.7 pt) |

### Same 802 questions, both models

| model / arm | population | churned | C→W | W→C | W→W′ | C→C′ | net accuracy change |
|---|---:|---:|---:|---:|---:|---:|---:|
| LLaDA fp32_all | 802 | 139 (17.3%) | 33 | 35 | 27 (19.4% of churn) | 44 | +2 (+0.2 pt) |
| LLaDA fp32_prime | 802 | 138 (17.2%) | 37 | 32 | 29 (21.0% of churn) | 40 | -5 (-0.6 pt) |
| LLaDA noise_s0 | 802 | 142 (17.7%) | 42 | 32 | 28 (19.7% of churn) | 40 | -10 (-1.2 pt) |
| OLMoE fp32_all | 802 | 155 (19.3%) | 28 | 25 | 102 (65.8% of churn) | 0 | -3 (-0.4 pt) |
| OLMoE fp32_prefill | 802 | 163 (20.3%) | 27 | 39 | 97 (59.5% of churn) | 0 | +12 (+1.5 pt) |
| OLMoE noise_s0 | 802 | 163 (20.3%) | 28 | 37 | 98 (60.1% of churn) | 0 | +9 (+1.1 pt) |

C→W and W→C are near-balanced in every arm (hence no accuracy effect); the excess of churn over |C→W − W→C| is answers moving between wrong values.
