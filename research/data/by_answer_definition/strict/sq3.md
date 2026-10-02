## SQ3: autoregressive MoE (OLMoE-1B-7B-0924-Instruct) vs LLaDA-MoE

Setup: HF transformers 4.53.2, bf16, SDPA attention, greedy, max_new_tokens=256, batch 64 left-padded, A100-40GB. Same system prompt and 4-shot GSM8K user prompt as LLaDA, rendered by OLMoE's chat template: `<|endoftext|><|system|>\n{system}\n<|user|>\n{prompt}\n<|assistant|>\n` (tokenized without extra special tokens; the template supplies BOS). Perturbation: a forward hook on each layer's `mlp.gate` replacing its output (HF code untouched). Questions: the 805 questions uncapped in LLaDA's control (budget cap for Phase 3, as instructed). OLMoE capped = control hit 256 new tokens without EOS.

### Determinism

Two control runs, same batches (n=200, B=64): 200/200 token-identical. Telemetry hook inert: control_tel is the first of the two.

### Router flatness per layer (bf16 vs fp32 on the same hidden states)

OLMoE: control telemetry on the first 200 questions (the Phase 1 question set), prefill = prompt tokens, decode = generated tokens. LLaDA: Phase 1 telemetry, same 200 questions, prompt tokens of the prime pass and all routed positions of all passes.

| layer | entropy LLaDA (prompt) | entropy OLMoE (prefill) | top-1 LLaDA | top-1 OLMoE | median gap ULP LLaDA | median gap ULP OLMoE | flip LLaDA prompt / all | flip OLMoE prefill / decode |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.996 | 0.930 | 2.7% | 9.5% | 0.18 | 36.97 | 87.3% / 94.8% | 0.8% / 0.9% |
| 1 | 0.994 | 0.961 | 3.2% | 6.8% | 0.45 | 14.32 | 33.1% / 66.3% | 1.5% / 1.7% |
| 2 | 0.993 | 0.959 | 3.3% | 7.2% | 0.68 | 14.18 | 22.6% / 44.2% | 1.7% / 1.8% |
| 3 | 0.993 | 0.943 | 3.4% | 7.8% | 0.73 | 27.30 | 21.7% / 38.8% | 1.0% / 1.2% |
| 4 | 0.994 | 0.955 | 3.2% | 6.8% | 0.68 | 23.11 | 22.7% / 32.2% | 1.0% / 1.2% |
| 5 | 0.997 | 0.951 | 2.6% | 6.7% | 0.49 | 22.42 | 29.9% / 32.7% | 0.8% / 1.3% |
| 6 | 0.991 | 0.943 | 3.4% | 7.2% | 0.95 | 26.27 | 17.0% / 19.4% | 0.9% / 1.0% |
| 7 | 0.984 | 0.940 | 4.0% | 6.8% | 1.19 | 24.28 | 13.8% / 14.8% | 0.9% / 1.1% |
| 8 | 0.975 | 0.916 | 4.6% | 7.8% | 1.51 | 31.70 | 11.4% / 12.1% | 0.7% / 0.8% |
| 9 | 0.967 | 0.896 | 5.3% | 8.7% | 1.57 | 37.36 | 10.8% / 9.8% | 0.7% / 0.6% |
| 10 | 0.963 | 0.883 | 5.0% | 10.0% | 1.93 | 33.43 | 9.4% / 9.5% | 0.5% / 1.0% |
| 11 | 0.946 | 0.880 | 6.4% | 9.8% | 2.17 | 32.91 | 8.3% / 8.3% | 0.7% / 0.9% |
| 12 | 0.945 | 0.861 | 6.5% | 11.4% | 1.97 | 30.06 | 9.3% / 8.5% | 0.9% / 1.2% |
| 13 | 0.907 | 0.859 | 9.5% | 11.6% | 2.48 | 20.95 | 7.4% / 9.1% | 1.2% / 1.4% |
| 14 | 0.948 | 0.862 | 6.1% | 12.0% | 1.52 | 12.71 | 11.2% / 14.1% | 1.7% / 2.2% |
| 15 | 0.995 | 0.879 | 2.2% | 9.7% | 0.22 | 9.16 | 59.4% / 14.4% | 2.1% / 3.0% |
| **all** | 0.974 | 0.913 | 4.5% | 8.7% | 0.89 | 22.33 | 23.5% / 26.8% | 1.1% / 1.3% |

### Churn on the same questions

OLMoE primary population: the 805, minus the 3 capped in OLMoE's control → 802. LLaDA columns use the same 802 questions (all uncapped in LLaDA's control by construction).

| OLMoE arm | strict churn, primary [95% CI] | lenient churn, all 805 | acc ctrl → arm (805) | McNemar p | diverged | median first-divergent token [IQR] | LLaDA counterpart | LLaDA strict churn, same questions |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| fp32_all | 19.6 [17.0, 22.5] | 19.4 | 55.8 → 55.4 | 0.784 | 50.3% | 26 [11–53] | fp32_all | 24.7 [21.8, 27.8] |
| fp32_prefill | 20.1 [17.4, 23.0] | 20.4 | 55.8 → 57.3 | 0.175 | 50.6% | 23 [10–50] | fp32_prime | 23.6 [20.8, 26.6] |
| noise_s0 | 20.7 [18.0, 23.6] | 20.5 | 55.8 → 56.9 | 0.321 | 54.8% | 24 [9–46] | noise_s0 | 25.1 [22.2, 28.2] |

### Churn given divergence, and whether the same questions are fragile in both models

| model / arm | trajectories diverged | strict churn among diverged [95% CI] |
|---|---:|---:|
| OLMoE fp32_all | 403/802 | 39.0 [34.3, 43.8] |
| OLMoE fp32_prefill | 404/802 | 39.9 [35.2, 44.7] |
| OLMoE noise_s0 | 438/802 | 37.9 [33.5, 42.5] |
| LLaDA fp32_all | 802/802 | 24.7 [21.8, 27.8] |
| LLaDA fp32_prime | 802/802 | 23.6 [20.8, 26.6] |
| LLaDA noise_s0 | 802/802 | 25.1 [22.2, 28.2] |

Cross-model: Spearman ρ between a question's churn count in LLaDA (fp32_all + 3 noise seeds) and in OLMoE (its 3 arms) = 0.116 [0.049, 0.188] (bootstrap, n=802). Churned under fp32_all in both models: 45 questions vs 38.8 expected if independent (Jaccard 0.15).

| model | fp32_all churn, control wrong | fp32_all churn, control right |
|---|---:|---:|
| OLMoE | 35.4% (n=353) | 7.1% (n=449) |
| LLaDA | 63.8% (n=105) | 18.8% (n=697) |

### OLMoE on LLaDA's dose-response curves

Expected flips per OLMoE question from its control telemetry (200 questions). Every OLMoE forward writes the KV cache (prefill and each decode step alike), so model B is evaluated with all OLMoE flips counted as cache-writing flips, and, as a bound, as step flips. Note the unit differs in kind: LLaDA re-routes every masked position at every step, so its per-question flip count includes many re-decisions of the same position; OLMoE routes each token once.

| OLMoE arm | expected flips / question | source | observed churn [95% CI] | LLaDA model A predicts | model B, as cache flips | model B, as step flips |
|---|---:|---|---:|---:|---:|---:|
| fp32_all | 249 | measured | 19.6 [17.0, 22.5] | 0.1 | 2.0 | 0.1 |
| fp32_prefill | 227 | measured | 20.1 [17.4, 23.0] | 0.1 | 1.9 | 0.1 |
| noise_s0 | 488 | model | 20.7 [18.0, 23.6] | 0.2 | 3.8 | 0.2 |

Plots: `research/data/by_answer_definition/strict\sq3_router_flatness.png`, `research/data/by_answer_definition/strict\sq3_dose_response.png`

