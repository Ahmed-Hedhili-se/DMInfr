## Divergence × hazard decomposition

A trajectory that never diverges reproduces control token for token, so it cannot churn; churn = Σ over diverged questions of h(when it diverged). One hazard curve h, estimated from the other arms (leave-one-arm-out), predicts each arm's churn from that arm's own first-divergence distribution. z = (observed − predicted) / √Σh(1−h). Arms share questions, so the z-scores are not independent; χ² is reported as a summary, not a calibrated test.

### LLaDA-MoE, 14 arms, hazard binned by block of first divergence (n=805 uncapped)

Pooled hazard h(block) = P(strict answer changes | first divergence in that block):

| block | diverging there (pooled over arms) | h [95% CI] |
|---:|---:|---:|
| 0 | 8097 | 14.6 [13.9, 15.4] |
| 1 | 1697 | 10.0 [8.6, 11.5] |
| 2 | 366 | 5.7 [3.8, 8.6] |
| 3 | 164 | 6.7 [3.8, 11.6] |
| 4 | 79 | 2.5 [0.7, 8.8] |
| 5 | 30 | 3.3 [0.6, 16.7] |
| 6 | 798 | 0.9 [0.4, 1.8] |
| 7 | 27 | 0.0 [0.0, 12.5] |

| LLaDA arm | n | diverged | churn \| diverged | observed churn | predicted (hazard from the other arms) | z |
|---|---:|---:|---:|---:|---:|---:|
| fp32_all | 805 | 100.0% | 17.3% | 17.3% | 14.2% | **+2.4** |
| fp32_prime | 805 | 100.0% | 17.1% | 17.1% | 14.3% | **+2.3** |
| fp32_finalize | 805 | 100.0% | 12.5% | 12.5% | 7.7% | **+5.2** |
| fp32_steps_within_early | 805 | 100.0% | 9.9% | 9.9% | 12.0% | -1.8 |
| fp32_steps_within_late | 805 | 100.0% | 14.0% | 14.0% | 14.2% | -0.1 |
| fp32_steps_within_last25 | 805 | 100.0% | 6.0% | 6.0% | 11.3% | **-4.8** |
| fp32_steps_global_early | 805 | 100.0% | 12.7% | 12.7% | 14.6% | -1.6 |
| fp32_steps_global_last25 | 805 | 99.4% | 0.9% | 0.9% | 2.7% | **-3.2** |
| fp32_layer0 | 805 | 100.0% | 17.6% | 17.6% | 14.2% | **+2.8** |
| fp32_layers11_14 | 805 | 100.0% | 8.3% | 8.3% | 13.6% | **-4.4** |
| fp32_layer15 | 805 | 99.1% | 4.4% | 4.3% | 11.6% | **-6.5** |
| noise_s0 | 805 | 100.0% | 17.8% | 17.8% | 14.2% | **+2.9** |
| noise_s1 | 805 | 100.0% | 17.4% | 17.4% | 14.2% | **+2.6** |
| noise_s2 | 805 | 100.0% | 17.4% | 17.4% | 14.2% | **+2.6** |

χ² = 167.8 over 14 arms; compare the flip-count fits in sq2_extra run under the same RESEARCH_ANSWER definition.

Sensitivity: the same leave-one-arm-out test with finer bins (8-step bins over blocks 0-1, whole blocks after), since block 0 holds most first divergences:

| LLaDA (fine bins) arm | n | diverged | churn \| diverged | observed churn | predicted (hazard from the other arms) | z |
|---|---:|---:|---:|---:|---:|---:|
| fp32_all | 805 | 100.0% | 17.3% | 17.3% | 15.2% | +1.7 |
| fp32_prime | 805 | 100.0% | 17.1% | 17.1% | 15.2% | +1.6 |
| fp32_finalize | 805 | 100.0% | 12.5% | 12.5% | 8.0% | **+4.7** |
| fp32_steps_within_early | 805 | 100.0% | 9.9% | 9.9% | 14.0% | **-3.3** |
| fp32_steps_within_late | 805 | 100.0% | 14.0% | 14.0% | 12.0% | +1.7 |
| fp32_steps_within_last25 | 805 | 100.0% | 6.0% | 6.0% | 8.4% | **-2.6** |
| fp32_steps_global_early | 805 | 100.0% | 12.7% | 12.7% | 15.1% | -1.9 |
| fp32_steps_global_last25 | 805 | 99.4% | 0.9% | 0.9% | 2.7% | **-3.2** |
| fp32_layer0 | 805 | 100.0% | 17.6% | 17.6% | 15.2% | +2.0 |
| fp32_layers11_14 | 805 | 100.0% | 8.3% | 8.3% | 13.0% | **-3.9** |
| fp32_layer15 | 805 | 99.1% | 4.4% | 4.3% | 10.6% | **-5.8** |
| noise_s0 | 805 | 100.0% | 17.8% | 17.8% | 15.2% | **+2.0** |
| noise_s1 | 805 | 100.0% | 17.4% | 17.4% | 15.2% | +1.7 |
| noise_s2 | 805 | 100.0% | 17.4% | 17.4% | 15.2% | +1.7 |

χ² = 125.5 over 14 arms with fine bins.

### Out of sample: the Phase 4 threshold pair, predicted from the 14 fixed-schedule arms

| pair | n (uncapped in both schedules) | diverged | observed churn | predicted | z |
|---|---:|---:|---:|---:|---:|
| fixed control vs fp32_all | 684 | 100.0% | 15.4% | 14.5% | +0.6 |
| threshold control vs fp32_all | 684 | 100.0% | 20.8% | 14.5% | +4.7 |
| schedule switch: fixed control vs threshold control | 684 | 100.0% | 14.9% | 14.6% | +0.2 |

The fixed pair is in-sample for h (its arm contributes to the pooled hazard); the threshold pair and the schedule switch are not. For the schedule switch the 'first divergence' compares commit steps across two schedules, which is only meaningful as the first committed-token difference.

### OLMoE, 3 arms, hazard binned by first divergent token (32-token bins; n=802)

| tokens | diverging there | h [95% CI] |
|---|---:|---:|
| 0–31 | 719 | 45.2 [41.6, 48.9] |
| 32–63 | 320 | 31.6 [26.7, 36.8] |
| 64–95 | 116 | 27.6 [20.3, 36.3] |
| 96–127 | 65 | 30.8 [20.9, 42.8] |
| 128–159 | 14 | 21.4 [7.6, 47.6] |
| 160–191 | 9 | 0.0 [0.0, 29.9] |
| 192–223 | 0 | — [nan, nan] |
| 224–255 | 2 | 0.0 [0.0, 65.8] |

| OLMoE arm | n | diverged | churn \| diverged | observed churn | predicted (hazard from the other arms) | z |
|---|---:|---:|---:|---:|---:|---:|
| fp32_all | 802 | 50.2% | 38.5% | 19.3% | 19.1% | +0.2 |
| fp32_prefill | 802 | 50.4% | 40.3% | 20.3% | 19.2% | +0.9 |
| noise_s0 | 802 | 54.6% | 37.2% | 20.3% | 22.0% | -1.3 |

χ² = 2.6 over 3 arms.

### Hazard by first-divergence position as a fraction of control's generated length

| fraction of control's length | LLaDA diverging | LLaDA h | OLMoE diverging | OLMoE h |
|---|---:|---:|---:|---:|
| 0.0–0.1 | 6634 | 16.2% | 287 | 51.6% |
| 0.1–0.2 | 1823 | 9.8% | 182 | 54.9% |
| 0.2–0.3 | 891 | 8.8% | 145 | 46.9% |
| 0.3–0.4 | 404 | 5.4% | 117 | 29.9% |
| 0.4–0.5 | 245 | 7.8% | 105 | 33.3% |
| 0.5–0.6 | 155 | 2.6% | 78 | 41.0% |
| 0.6–0.7 | 102 | 6.9% | 93 | 35.5% |
| 0.7–0.8 | 54 | 3.7% | 66 | 19.7% |
| 0.8–0.9 | 140 | 5.0% | 56 | 23.2% |
| 0.9–1.0 | 126 | 0.8% | 78 | 5.1% |
| ≥ 1.0 (after control's EOS) | 684 | 0.0% | 38 | 0.0% |

LLaDA's position is the generated-sequence position of its first differing commit; OLMoE's is the first differing token. LLaDA can diverge first at a position *beyond* control's EOS (in the EOS-padding region) because it commits positions out of order.

Plot: `research/data/by_answer_definition/strict_v2/hazard.png`

