## Divergence × hazard decomposition

A trajectory that never diverges reproduces control token for token, so it cannot churn; churn = Σ over diverged questions of h(when it diverged). One hazard curve h, estimated from the other arms (leave-one-arm-out), predicts each arm's churn from that arm's own first-divergence distribution. z = (observed − predicted) / √Σh(1−h). Arms share questions, so the z-scores are not independent; χ² is reported as a summary, not a calibrated test.

### LLaDA-MoE, 14 arms, hazard binned by block of first divergence (n=805 uncapped)

Pooled hazard h(block) = P(strict answer changes | first divergence in that block):

| block | diverging there (pooled over arms) | h [95% CI] |
|---:|---:|---:|
| 0 | 8097 | 10.6 [9.9, 11.3] |
| 1 | 1697 | 7.2 [6.1, 8.6] |
| 2 | 366 | 4.4 [2.7, 7.0] |
| 3 | 164 | 4.9 [2.5, 9.3] |
| 4 | 79 | 2.5 [0.7, 8.8] |
| 5 | 30 | 3.3 [0.6, 16.7] |
| 6 | 798 | 0.3 [0.1, 0.9] |
| 7 | 27 | 0.0 [0.0, 12.5] |

| LLaDA arm | n | diverged | churn \| diverged | observed churn | predicted (hazard from the other arms) | z |
|---|---:|---:|---:|---:|---:|---:|
| fp32_all | 805 | 100.0% | 11.9% | 11.9% | 10.4% | +1.5 |
| fp32_prime | 805 | 100.0% | 12.3% | 12.3% | 10.4% | +1.8 |
| fp32_finalize | 805 | 100.0% | 9.2% | 9.2% | 5.7% | **+4.4** |
| fp32_steps_within_early | 805 | 100.0% | 6.8% | 6.8% | 8.8% | -2.0 |
| fp32_steps_within_late | 805 | 100.0% | 9.7% | 9.7% | 10.3% | -0.6 |
| fp32_steps_within_last25 | 805 | 100.0% | 4.2% | 4.2% | 8.4% | **-4.3** |
| fp32_steps_global_early | 805 | 100.0% | 10.3% | 10.3% | 10.4% | -0.1 |
| fp32_steps_global_last25 | 805 | 99.4% | 0.2% | 0.2% | 2.7% | **-4.3** |
| fp32_layer0 | 805 | 100.0% | 12.9% | 12.9% | 10.3% | **+2.5** |
| fp32_layers11_14 | 805 | 100.0% | 6.2% | 6.2% | 9.8% | **-3.5** |
| fp32_layer15 | 805 | 99.1% | 3.0% | 3.0% | 8.6% | **-5.7** |
| noise_s0 | 805 | 100.0% | 12.7% | 12.7% | 10.3% | **+2.2** |
| noise_s1 | 805 | 100.0% | 12.3% | 12.3% | 10.3% | +1.8 |
| noise_s2 | 805 | 100.0% | 13.4% | 13.4% | 10.2% | **+3.0** |

χ² = 133.1 over 14 arms; compare the flip-count fits in sq2_extra run under the same RESEARCH_ANSWER definition.

Sensitivity: the same leave-one-arm-out test with finer bins (8-step bins over blocks 0-1, whole blocks after), since block 0 holds most first divergences:

| LLaDA (fine bins) arm | n | diverged | churn \| diverged | observed churn | predicted (hazard from the other arms) | z |
|---|---:|---:|---:|---:|---:|---:|
| fp32_all | 805 | 100.0% | 11.9% | 11.9% | 11.0% | +0.8 |
| fp32_prime | 805 | 100.0% | 12.3% | 12.3% | 10.9% | +1.2 |
| fp32_finalize | 805 | 100.0% | 9.2% | 9.2% | 5.8% | **+4.1** |
| fp32_steps_within_early | 805 | 100.0% | 6.8% | 6.8% | 10.0% | **-3.0** |
| fp32_steps_within_late | 805 | 100.0% | 9.7% | 9.7% | 9.4% | +0.3 |
| fp32_steps_within_last25 | 805 | 100.0% | 4.2% | 4.2% | 6.0% | **-2.2** |
| fp32_steps_global_early | 805 | 100.0% | 10.3% | 10.3% | 10.8% | -0.5 |
| fp32_steps_global_last25 | 805 | 99.4% | 0.2% | 0.2% | 2.7% | **-4.3** |
| fp32_layer0 | 805 | 100.0% | 12.9% | 12.9% | 10.9% | +1.9 |
| fp32_layers11_14 | 805 | 100.0% | 6.2% | 6.2% | 9.5% | **-3.2** |
| fp32_layer15 | 805 | 99.1% | 3.0% | 3.0% | 8.0% | **-5.3** |
| noise_s0 | 805 | 100.0% | 12.7% | 12.7% | 11.0% | +1.5 |
| noise_s1 | 805 | 100.0% | 12.3% | 12.3% | 11.0% | +1.2 |
| noise_s2 | 805 | 100.0% | 13.4% | 13.4% | 10.9% | **+2.3** |

χ² = 102.6 over 14 arms with fine bins.

### Out of sample: the Phase 4 threshold pair, predicted from the 14 fixed-schedule arms

| pair | n (uncapped in both schedules) | diverged | observed churn | predicted | z |
|---|---:|---:|---:|---:|---:|
| fixed control vs fp32_all | 684 | 100.0% | 9.2% | 10.5% | -1.1 |
| threshold control vs fp32_all | 684 | 100.0% | 13.6% | 10.5% | +2.7 |
| schedule switch: fixed control vs threshold control | 684 | 100.0% | 10.4% | 10.6% | -0.2 |

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
| 0.0–0.1 | 6634 | 11.8% | 287 | 51.6% |
| 0.1–0.2 | 1823 | 6.7% | 182 | 54.9% |
| 0.2–0.3 | 891 | 7.3% | 145 | 46.9% |
| 0.3–0.4 | 404 | 4.0% | 117 | 29.9% |
| 0.4–0.5 | 245 | 4.9% | 105 | 33.3% |
| 0.5–0.6 | 155 | 1.9% | 78 | 41.0% |
| 0.6–0.7 | 102 | 4.9% | 93 | 35.5% |
| 0.7–0.8 | 54 | 3.7% | 66 | 19.7% |
| 0.8–0.9 | 140 | 1.4% | 56 | 21.4% |
| 0.9–1.0 | 126 | 0.8% | 78 | 5.1% |
| ≥ 1.0 (after control's EOS) | 684 | 0.0% | 38 | 0.0% |

LLaDA's position is the generated-sequence position of its first differing commit; OLMoE's is the first differing token. LLaDA can diverge first at a position *beyond* control's EOS (in the EOS-padding region) because it commits positions out of order.

Plot: `research/data/by_answer_definition/lenient/hazard.png`

