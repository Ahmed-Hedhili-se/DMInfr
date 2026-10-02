## Churn decomposed into transitions

C = graded correct by the repo grader, W = wrong; W→W′ = the strict answer changed and both are wrong. Populations: LLaDA's own uncapped set (805); the 802 uncapped in both models' controls for the cross-model rows; the 684 uncapped under both schedules for the threshold pair.

| model / arm | population | churned | C→W | W→C | W→W′ | C→C′ | net accuracy change |
|---|---:|---:|---:|---:|---:|---:|---:|
| LLaDA fp32_all | 805 | 198 (24.6%) | 30 | 34 | 33 (16.7% of churn) | 101 | +4 (+0.5 pt) |
| LLaDA fp32_prime | 805 | 189 (23.5%) | 36 | 30 | 30 (15.9% of churn) | 93 | -6 (-0.7 pt) |
| LLaDA fp32_finalize | 805 | 159 (19.8%) | 28 | 22 | 28 (17.6% of churn) | 81 | -6 (-0.7 pt) |
| LLaDA fp32_steps_within_early | 805 | 107 (13.3%) | 18 | 20 | 18 (16.8% of churn) | 51 | +2 (+0.2 pt) |
| LLaDA fp32_steps_within_late | 805 | 156 (19.4%) | 29 | 26 | 27 (17.3% of churn) | 74 | -3 (-0.4 pt) |
| LLaDA fp32_steps_within_last25 | 805 | 72 (8.9%) | 11 | 10 | 18 (25.0% of churn) | 33 | -1 (-0.1 pt) |
| LLaDA fp32_steps_global_early | 805 | 149 (18.5%) | 33 | 29 | 21 (14.1% of churn) | 66 | -4 (-0.5 pt) |
| LLaDA fp32_steps_global_last25 | 805 | 13 (1.6%) | 0 | 1 | 3 (23.1% of churn) | 9 | +1 (+0.1 pt) |
| LLaDA fp32_layer0 | 805 | 198 (24.6%) | 42 | 33 | 29 (14.6% of churn) | 94 | -9 (-1.1 pt) |
| LLaDA fp32_layers11_14 | 805 | 101 (12.5%) | 19 | 15 | 17 (16.8% of churn) | 50 | -4 (-0.5 pt) |
| LLaDA fp32_layer15 | 805 | 55 (6.8%) | 10 | 7 | 10 (18.2% of churn) | 28 | -3 (-0.4 pt) |
| LLaDA noise_s0 | 805 | 202 (25.1%) | 43 | 28 | 33 (16.3% of churn) | 98 | -15 (-1.9 pt) |
| LLaDA noise_s1 | 805 | 197 (24.5%) | 41 | 30 | 34 (17.3% of churn) | 92 | -11 (-1.4 pt) |
| LLaDA noise_s2 | 805 | 202 (25.1%) | 44 | 37 | 31 (15.3% of churn) | 90 | -7 (-0.9 pt) |
| LLaDA threshold: control vs fp32_all | 684 | 179 (26.2%) | 34 | 37 | 23 (12.8% of churn) | 85 | +3 (+0.4 pt) |
| LLaDA fixed: control vs fp32_all (same 684) | 684 | 149 (21.8%) | 19 | 25 | 19 (12.8% of churn) | 86 | +6 (+0.9 pt) |

### Same 802 questions, both models

| model / arm | population | churned | C→W | W→C | W→W′ | C→C′ | net accuracy change |
|---|---:|---:|---:|---:|---:|---:|---:|
| LLaDA fp32_all | 802 | 198 (24.7%) | 30 | 34 | 33 (16.7% of churn) | 101 | +4 (+0.5 pt) |
| LLaDA fp32_prime | 802 | 189 (23.6%) | 36 | 30 | 30 (15.9% of churn) | 93 | -6 (-0.7 pt) |
| LLaDA noise_s0 | 802 | 201 (25.1%) | 42 | 28 | 33 (16.4% of churn) | 98 | -14 (-1.7 pt) |
| OLMoE fp32_all | 802 | 157 (19.6%) | 28 | 25 | 100 (63.7% of churn) | 4 | -3 (-0.4 pt) |
| OLMoE fp32_prefill | 802 | 161 (20.1%) | 27 | 38 | 95 (59.0% of churn) | 1 | +11 (+1.4 pt) |
| OLMoE noise_s0 | 802 | 166 (20.7%) | 28 | 37 | 98 (59.0% of churn) | 3 | +9 (+1.1 pt) |

C→W and W→C are near-balanced in every arm (hence no accuracy effect); the excess of churn over |C→W − W→C| is answers moving between wrong values.
