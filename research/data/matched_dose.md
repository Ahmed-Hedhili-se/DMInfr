## Matched dose: fp32 router on exactly one forward pass

Questions: the 802 uncapped in both models' controls. Controls reused (LLaDA: Phase 2 container A, same 1319-question batches; OLMoE: Phase 3, same 805-question batches). 'Injected flips' = top-8 sets that differ between bf16 and fp32 gates on the same hidden states, summed over all layers and routed positions of that one pass, counted in the arm itself.

OLMoE cross-container check: control re-run in the matched-dose container, first batch: 64/64 token-identical to Phase 3's control.

| model | perturbed pass | injected flips / question: mean (median) | questions with ≥ 1 injected flip | P(diverged) [95% CI] | P(diverged \| ≥ 1 flip) | P(diverged \| 0 flips) | median first divergence |
|---|---|---:|---:|---:|---:|---:|---:|
| LLaDA-MoE | block 0, step 0 | 1,140.3 (1,141) | 802/802 | 17.2% [14.8, 20.0] | 17.2% [14.8, 20.0] | — | step 0 |
| LLaDA-MoE | block 0, step 16 | 1,118.5 (1,119) | 802/802 | 9.0% [7.2, 11.2] | 9.0% [7.2, 11.2] | — | step 16 |
| OLMoE | decode forward routing token 0 | 0.2 (0) | 154/802 | 39.9% [36.6, 43.3] | 34.4% [27.4, 42.2] | 41.2% [37.5, 45.0] | token 34 |
| OLMoE | decode forward routing token 16 | 0.2 (0) | 136/802 | 32.0% [28.9, 35.4] | 36.0% [28.4, 44.4] | 31.2% [27.8, 34.9] | token 50 |

churn, per answer definition (all three shown; see the note on strict v1 in the reports):

| model | perturbed pass | churn, strict v1 | churn, strict v2 | churn, lenient (grader) | churn \| diverged (lenient) |
|---|---|---:|---:|---:|---:|
| LLaDA-MoE | block 0, step 0 | 3.1% [2.1, 4.6] | 2.1% [1.3, 3.4] | 1.4% [0.8, 2.4] | 8.0% [4.5, 13.7] |
| LLaDA-MoE | block 0, step 16 | 1.2% [0.7, 2.3] | 0.5% [0.2, 1.3] | 0.5% [0.2, 1.3] | 5.6% [2.2, 13.4] |
| OLMoE | decode forward routing token 0 | 17.2% [14.8, 20.0] | 17.0% [14.5, 19.7] | 17.0% [14.5, 19.7] | 42.5% [37.2, 48.0] |
| OLMoE | decode forward routing token 16 | 11.5% [9.4, 13.9] | 10.8% [8.9, 13.2] | 10.8% [8.9, 13.2] | 33.9% [28.3, 39.8] |
