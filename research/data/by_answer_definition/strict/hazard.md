## Divergence × hazard decomposition

A trajectory that never diverges reproduces control token for token, so it cannot churn; churn = Σ over diverged questions of h(when it diverged). One hazard curve h, estimated from the other arms (leave-one-arm-out), predicts each arm's churn from that arm's own first-divergence distribution. z = (observed − predicted) / √Σh(1−h). Arms share questions, so the z-scores are not independent; χ² is reported as a summary, not a calibrated test.

### LLaDA-MoE, 14 arms, hazard binned by block of first divergence (n=805 uncapped)

Pooled hazard h(block) = P(strict answer changes | first divergence in that block):

| block | diverging there (pooled over arms) | h [95% CI] |
|---:|---:|---:|
| 0 | 8097 | 20.7 [19.9, 21.6] |
| 1 | 1697 | 15.6 [13.9, 17.4] |
| 2 | 366 | 6.6 [4.4, 9.6] |
| 3 | 164 | 7.9 [4.7, 13.1] |
| 4 | 79 | 6.3 [2.7, 14.0] |
| 5 | 30 | 3.3 [0.6, 16.7] |
| 6 | 798 | 1.6 [1.0, 2.8] |
| 7 | 27 | 0.0 [0.0, 12.5] |

| LLaDA arm | n | diverged | churn \| diverged | observed churn | predicted (hazard from the other arms) | z |
|---|---:|---:|---:|---:|---:|---:|
| fp32_all | 805 | 100.0% | 24.6% | 24.6% | 20.2% | **+3.1** |
| fp32_prime | 805 | 100.0% | 23.5% | 23.5% | 20.4% | **+2.2** |
| fp32_finalize | 805 | 100.0% | 19.8% | 19.8% | 11.7% | **+7.1** |
| fp32_steps_within_early | 805 | 100.0% | 13.3% | 13.3% | 17.2% | **-3.0** |
| fp32_steps_within_late | 805 | 100.0% | 19.4% | 19.4% | 20.3% | -0.7 |
| fp32_steps_within_last25 | 805 | 100.0% | 8.9% | 8.9% | 16.2% | **-5.7** |
| fp32_steps_global_early | 805 | 100.0% | 18.5% | 18.5% | 20.7% | -1.5 |
| fp32_steps_global_last25 | 805 | 99.4% | 1.6% | 1.6% | 2.7% | -1.9 |
| fp32_layer0 | 805 | 100.0% | 24.6% | 24.6% | 20.2% | **+3.1** |
| fp32_layers11_14 | 805 | 100.0% | 12.5% | 12.5% | 19.4% | **-4.9** |
| fp32_layer15 | 805 | 99.1% | 6.9% | 6.8% | 16.5% | **-7.5** |
| noise_s0 | 805 | 100.0% | 25.1% | 25.1% | 20.2% | **+3.5** |
| noise_s1 | 805 | 100.0% | 24.5% | 24.5% | 20.2% | **+3.0** |
| noise_s2 | 805 | 100.0% | 25.1% | 25.1% | 20.1% | **+3.5** |

χ² = 235.0 over 14 arms; compare the flip-count fits in sq2_extra run under the same RESEARCH_ANSWER definition.

Sensitivity: the same leave-one-arm-out test with finer bins (8-step bins over blocks 0-1, whole blocks after), since block 0 holds most first divergences:

| LLaDA (fine bins) arm | n | diverged | churn \| diverged | observed churn | predicted (hazard from the other arms) | z |
|---|---:|---:|---:|---:|---:|---:|
| fp32_all | 805 | 100.0% | 24.6% | 24.6% | 21.3% | **+2.3** |
| fp32_prime | 805 | 100.0% | 23.5% | 23.5% | 21.4% | +1.4 |
| fp32_finalize | 805 | 100.0% | 19.8% | 19.8% | 12.0% | **+6.8** |
| fp32_steps_within_early | 805 | 100.0% | 13.3% | 13.3% | 19.2% | **-4.3** |
| fp32_steps_within_late | 805 | 100.0% | 19.4% | 19.4% | 18.5% | +0.6 |
| fp32_steps_within_last25 | 805 | 100.0% | 8.9% | 8.9% | 12.3% | **-3.0** |
| fp32_steps_global_early | 805 | 100.0% | 18.5% | 18.5% | 21.4% | -2.0 |
| fp32_steps_global_last25 | 805 | 99.4% | 1.6% | 1.6% | 2.7% | -1.9 |
| fp32_layer0 | 805 | 100.0% | 24.6% | 24.6% | 21.4% | **+2.3** |
| fp32_layers11_14 | 805 | 100.0% | 12.5% | 12.5% | 18.9% | **-4.6** |
| fp32_layer15 | 805 | 99.1% | 6.9% | 6.8% | 15.5% | **-6.9** |
| noise_s0 | 805 | 100.0% | 25.1% | 25.1% | 21.4% | **+2.6** |
| noise_s1 | 805 | 100.0% | 24.5% | 24.5% | 21.4% | **+2.1** |
| noise_s2 | 805 | 100.0% | 25.1% | 25.1% | 21.3% | **+2.6** |

χ² = 180.1 over 14 arms with fine bins.

### Out of sample: the Phase 4 threshold pair, predicted from the 14 fixed-schedule arms

| pair | n (uncapped in both schedules) | diverged | observed churn | predicted | z |
|---|---:|---:|---:|---:|---:|
| fixed control vs fp32_all | 684 | 100.0% | 21.8% | 20.6% | +0.8 |
| threshold control vs fp32_all | 684 | 100.0% | 26.2% | 20.5% | +3.7 |
| schedule switch: fixed control vs threshold control | 684 | 100.0% | 23.2% | 20.7% | +1.6 |

The fixed pair is in-sample for h (its arm contributes to the pooled hazard); the threshold pair and the schedule switch are not. For the schedule switch the 'first divergence' compares commit steps across two schedules, which is only meaningful as the first committed-token difference.

### OLMoE, 3 arms, hazard binned by first divergent token (32-token bins; n=802)

| tokens | diverging there | h [95% CI] |
|---|---:|---:|
| 0–31 | 719 | 45.1 [41.5, 48.7] |
| 32–63 | 320 | 31.9 [27.0, 37.2] |
| 64–95 | 116 | 27.6 [20.3, 36.3] |
| 96–127 | 65 | 32.3 [22.2, 44.4] |
| 128–159 | 14 | 35.7 [16.3, 61.2] |
| 160–191 | 9 | 0.0 [0.0, 29.9] |
| 192–223 | 0 | — [nan, nan] |
| 224–255 | 2 | 0.0 [0.0, 65.8] |

| OLMoE arm | n | diverged | churn \| diverged | observed churn | predicted (hazard from the other arms) | z |
|---|---:|---:|---:|---:|---:|---:|
| fp32_all | 802 | 50.2% | 39.0% | 19.6% | 19.2% | +0.3 |
| fp32_prefill | 802 | 50.4% | 39.9% | 20.1% | 19.5% | +0.5 |
| noise_s0 | 802 | 54.6% | 37.9% | 20.7% | 22.0% | -1.0 |

χ² = 1.4 over 3 arms.

### Hazard by first-divergence position as a fraction of control's generated length

| fraction of control's length | LLaDA diverging | LLaDA h | OLMoE diverging | OLMoE h |
|---|---:|---:|---:|---:|
| 0.0–0.1 | 6634 | 22.7% | 287 | 52.3% |
| 0.1–0.2 | 1823 | 14.4% | 182 | 53.3% |
| 0.2–0.3 | 891 | 16.6% | 145 | 46.2% |
| 0.3–0.4 | 404 | 6.9% | 117 | 30.8% |
| 0.4–0.5 | 245 | 7.8% | 105 | 33.3% |
| 0.5–0.6 | 155 | 3.9% | 78 | 38.5% |
| 0.6–0.7 | 102 | 6.9% | 93 | 35.5% |
| 0.7–0.8 | 54 | 7.4% | 66 | 21.2% |
| 0.8–0.9 | 140 | 4.3% | 56 | 23.2% |
| 0.9–1.0 | 126 | 5.6% | 78 | 11.5% |
| ≥ 1.0 (after control's EOS) | 684 | 0.3% | 38 | 0.0% |

LLaDA's position is the generated-sequence position of its first differing commit; OLMoE's is the first differing token. LLaDA can diverge first at a position *beyond* control's EOS (in the EOS-padding region) because it commits positions out of order.

Plot: `research/data/by_answer_definition/strict/hazard.png`

