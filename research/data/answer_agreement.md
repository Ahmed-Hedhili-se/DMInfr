## Grader extraction vs strict v3 (control-uncapped questions)

agree = fraction of the arm's rows where strict v3 returns the grader's answer. Flag: |churn(grader) − churn(v3)| > 1 point.

| model | arm | n | agree (arm rows) | v3 = None | churn, grader | churn, strict v3 | Δ | flag |
|---|---|---:|---:|---:|---:|---:|---:|---|
| LLaDA | control | 805 | 99.3% | 3 | — | — | — | |
| OLMoE | control | 802 | 100.0% | 0 | — | — | — | |
| LLaDA | fp32_all | 805 | 99.0% | 4 | 11.9% | 12.9% | +1.0 |  |
| LLaDA | fp32_finalize | 805 | 99.1% | 6 | 9.2% | 9.8% | +0.6 |  |
| LLaDA | fp32_layer0 | 805 | 99.4% | 4 | 12.9% | 13.7% | +0.7 |  |
| LLaDA | fp32_prime | 805 | 98.5% | 9 | 12.3% | 12.9% | +0.6 |  |
| LLaDA | fp32_single_b0s0 | 805 | 99.0% | 5 | 1.4% | 1.5% | +0.1 |  |
| LLaDA | fp32_single_b0s16 | 805 | 99.3% | 3 | 0.5% | 0.5% | +0.0 |  |
| LLaDA | fp32_steps_global_early | 805 | 99.3% | 3 | 10.3% | 10.7% | +0.4 |  |
| LLaDA | fp32_steps_global_last25 | 805 | 99.5% | 2 | 0.2% | 0.5% | +0.2 |  |
| LLaDA | fp32_steps_within_early | 805 | 98.8% | 8 | 6.8% | 7.8% | +1.0 |  |
| LLaDA | fp32_steps_within_last25 | 805 | 99.4% | 3 | 4.2% | 4.7% | +0.5 |  |
| LLaDA | fp32_steps_within_late | 805 | 98.6% | 7 | 9.7% | 10.7% | +1.0 |  |
| LLaDA | fp32_layer15 | 805 | 99.1% | 5 | 3.0% | 3.1% | +0.1 |  |
| LLaDA | fp32_layers11_14 | 805 | 99.3% | 3 | 6.2% | 6.5% | +0.2 |  |
| LLaDA | noise_s0 | 805 | 98.8% | 7 | 12.7% | 13.4% | +0.7 |  |
| LLaDA | noise_s1 | 805 | 98.4% | 7 | 12.3% | 13.5% | +1.2 | **> 1 pt** |
| LLaDA | noise_s2 | 805 | 99.1% | 4 | 13.4% | 13.8% | +0.4 |  |
| LLaDA threshold | fp32_all | 774 | 99.1% | 4 | 16.3% | 16.9% | +0.6 |  |
| OLMoE | fp32_all | 802 | 99.3% | 6 | 19.3% | 19.3% | +0.0 |  |
| OLMoE | fp32_dec0 | 802 | 99.4% | 5 | 17.0% | 17.0% | +0.0 |  |
| OLMoE | fp32_dec16 | 802 | 99.5% | 4 | 10.8% | 10.8% | +0.0 |  |
| OLMoE | fp32_prefill | 802 | 99.8% | 2 | 20.2% | 20.3% | +0.1 |  |
| OLMoE | noise_s0 | 802 | 99.8% | 2 | 20.3% | 20.3% | +0.0 |  |

1 arm(s) flagged.
