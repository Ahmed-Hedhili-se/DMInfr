## Divergence × hazard decomposition

LLaDA divergence definition: **content**: first position where the final generated tokens differ, commit order ignored (comparable with OLMoE's first differing token); bins are output positions (block = position // 32).

A trajectory that never diverges reproduces control token for token, so it cannot churn; churn = Σ over diverged questions of h(when it diverged). One hazard curve h, estimated from the other arms (leave-one-arm-out), predicts each arm's churn from that arm's own first-divergence distribution. z = (observed − predicted) / √Σh(1−h). Arms share questions, so the z-scores are not independent; χ² is reported as a summary, not a calibrated test.

### LLaDA-MoE, 14 arms, hazard binned by block of first divergence (n=805 uncapped)

Pooled hazard h(block) = P(strict answer changes | first divergence in that block):

| block | diverging there (pooled over arms) | h [95% CI] |
|---:|---:|---:|
| 0 | 4158 | 14.6 [13.6, 15.8] |
| 1 | 1921 | 10.1 [8.8, 11.5] |
| 2 | 838 | 12.5 [10.5, 14.9] |
| 3 | 513 | 9.7 [7.5, 12.6] |
| 4 | 336 | 9.8 [7.1, 13.5] |
| 5 | 181 | 4.4 [2.3, 8.5] |
| 6 | 122 | 7.4 [3.9, 13.4] |
| 7 | 16 | 0.0 [0.0, 19.4] |

| LLaDA arm | n | diverged | churn \| diverged | observed churn | predicted (hazard from the other arms) | z |
|---|---:|---:|---:|---:|---:|---:|
| fp32_all | 805 | 94.5% | 12.6% | 11.9% | 12.5% | -0.5 |
| fp32_prime | 805 | 93.7% | 13.1% | 12.3% | 12.2% | +0.1 |
| fp32_finalize | 805 | 83.2% | 11.0% | 9.2% | 8.5% | +0.7 |
| fp32_steps_within_early | 805 | 51.3% | 13.3% | 6.8% | 6.2% | +0.8 |
| fp32_steps_within_late | 805 | 83.6% | 11.6% | 9.7% | 10.3% | -0.6 |
| fp32_steps_within_last25 | 805 | 48.7% | 8.7% | 4.2% | 5.6% | -1.7 |
| fp32_steps_global_early | 805 | 70.8% | 14.6% | 10.3% | 9.5% | +0.8 |
| fp32_steps_global_last25 | 805 | 8.3% | 3.0% | 0.2% | 0.9% | **-2.1** |
| fp32_layer0 | 805 | 94.5% | 13.7% | 12.9% | 12.2% | +0.6 |
| fp32_layers11_14 | 805 | 58.8% | 10.6% | 6.2% | 6.9% | -0.8 |
| fp32_layer15 | 805 | 32.2% | 9.3% | 3.0% | 3.6% | -1.0 |
| noise_s0 | 805 | 95.8% | 13.2% | 12.7% | 12.7% | -0.0 |
| noise_s1 | 805 | 94.3% | 13.0% | 12.3% | 12.4% | -0.1 |
| noise_s2 | 805 | 94.7% | 14.2% | 13.4% | 12.3% | +1.0 |

χ² = 12.7 over 14 arms; compare the flip-count fits in sq2_extra run under the same RESEARCH_ANSWER definition.

Sensitivity: the same leave-one-arm-out test with finer bins (8-step bins over blocks 0-1, whole blocks after), since block 0 holds most first divergences:

| LLaDA (fine bins) arm | n | diverged | churn \| diverged | observed churn | predicted (hazard from the other arms) | z |
|---|---:|---:|---:|---:|---:|---:|
| fp32_all | 805 | 94.5% | 12.6% | 11.9% | 12.5% | -0.5 |
| fp32_prime | 805 | 93.7% | 13.1% | 12.3% | 12.2% | +0.1 |
| fp32_finalize | 805 | 83.2% | 11.0% | 9.2% | 8.5% | +0.7 |
| fp32_steps_within_early | 805 | 51.3% | 13.3% | 6.8% | 5.9% | +1.2 |
| fp32_steps_within_late | 805 | 83.6% | 11.6% | 9.7% | 10.6% | -0.9 |
| fp32_steps_within_last25 | 805 | 48.7% | 8.7% | 4.2% | 5.7% | -1.9 |
| fp32_steps_global_early | 805 | 70.8% | 14.6% | 10.3% | 9.5% | +0.8 |
| fp32_steps_global_last25 | 805 | 8.3% | 3.0% | 0.2% | 0.9% | **-2.1** |
| fp32_layer0 | 805 | 94.5% | 13.7% | 12.9% | 12.2% | +0.6 |
| fp32_layers11_14 | 805 | 58.8% | 10.6% | 6.2% | 6.9% | -0.8 |
| fp32_layer15 | 805 | 32.2% | 9.3% | 3.0% | 3.6% | -1.0 |
| noise_s0 | 805 | 95.8% | 13.2% | 12.7% | 12.8% | -0.1 |
| noise_s1 | 805 | 94.3% | 13.0% | 12.3% | 12.4% | -0.1 |
| noise_s2 | 805 | 94.7% | 14.2% | 13.4% | 12.4% | +0.9 |

χ² = 14.2 over 14 arms with fine bins.

### Out of sample: the Phase 4 threshold pair, predicted from the 14 fixed-schedule arms

| pair | n (uncapped in both schedules) | diverged | observed churn | predicted | z |
|---|---:|---:|---:|---:|---:|
| fixed control vs fp32_all | 684 | 94.2% | 9.2% | 12.3% | -2.5 |
| threshold control vs fp32_all | 684 | 97.8% | 13.6% | 13.1% | +0.4 |
| schedule switch: fixed control vs threshold control | 684 | 92.4% | 10.4% | 12.1% | -1.4 |

The fixed pair is in-sample for h (its arm contributes to the pooled hazard); the threshold pair and the schedule switch are not. For the schedule switch the 'first divergence' compares commit steps across two schedules, which is only meaningful as the first committed-token difference.

### OLMoE, 3 arms, hazard binned by first divergent token (32-token bins; n=802)

| tokens | diverging there | h [95% CI] |
|---|---:|---:|
| 0–31 | 719 | 45.2 [41.6, 48.9] |
| 32–63 | 320 | 31.6 [26.7, 36.8] |
| 64–95 | 116 | 27.6 [20.3, 36.3] |
| 96–127 | 65 | 29.2 [19.6, 41.2] |
| 128–159 | 14 | 21.4 [7.6, 47.6] |
| 160–191 | 9 | 0.0 [0.0, 29.9] |
| 192–223 | 0 | — [nan, nan] |
| 224–255 | 2 | 0.0 [0.0, 65.8] |

| OLMoE arm | n | diverged | churn \| diverged | observed churn | predicted (hazard from the other arms) | z |
|---|---:|---:|---:|---:|---:|---:|
| fp32_all | 802 | 50.2% | 38.5% | 19.3% | 19.1% | +0.2 |
| fp32_prefill | 802 | 50.4% | 40.1% | 20.2% | 19.2% | +0.8 |
| noise_s0 | 802 | 54.6% | 37.2% | 20.3% | 22.0% | -1.3 |

χ² = 2.4 over 3 arms.

### Hazard by first-divergence position as a fraction of control's generated length

| fraction of control's length | LLaDA diverging | LLaDA h | OLMoE diverging | OLMoE h |
|---|---:|---:|---:|---:|
| 0.0–0.1 | 3076 | 15.3% | 287 | 51.6% |
| 0.1–0.2 | 1282 | 13.9% | 182 | 54.9% |
| 0.2–0.3 | 1077 | 10.2% | 145 | 46.9% |
| 0.3–0.4 | 777 | 12.2% | 117 | 29.9% |
| 0.4–0.5 | 432 | 12.3% | 105 | 33.3% |
| 0.5–0.6 | 373 | 7.5% | 78 | 41.0% |
| 0.6–0.7 | 382 | 11.0% | 93 | 35.5% |
| 0.7–0.8 | 257 | 4.3% | 66 | 19.7% |
| 0.8–0.9 | 192 | 7.8% | 56 | 21.4% |
| 0.9–1.0 | 237 | 2.1% | 78 | 5.1% |
| ≥ 1.0 (after control's EOS) | 0 | —% | 38 | 0.0% |

LLaDA's position is the generated-sequence position of its first differing commit; OLMoE's is the first differing token. LLaDA can diverge first at a position *beyond* control's EOS (in the EOS-padding region) because it commits positions out of order.

### Cross-model: P(output diverges) and P(answer changes | diverged), same questions

Population: the 802 questions uncapped in both controls. LLaDA divergence: content.

| LLaDA arm | P(div) | P(change \| div) | OLMoE arm | P(div) | P(change \| div) |
|---|---:|---:|---|---:|---:|
| fp32_all | 94.5% | 12.7% | fp32_all | 50.2% | 38.5% |
| fp32_prime | 93.6% | 13.2% | fp32_prefill | 50.4% | 40.1% |
| noise_s0 | 95.8% | 13.2% | noise_s0 | 54.6% | 37.2% |
| fp32_single_b0s0 | 10.6% | 12.9% | fp32_dec0 | 39.9% | 42.5% |
| fp32_single_b0s16 | 2.7% | 18.2% | fp32_dec16 | 32.0% | 33.9% |

Plot: `research/data/by_answer_definition/lenient/hazard_content.png`

