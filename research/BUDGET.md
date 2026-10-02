# Routing-fragility study: Modal budget

Hard cap **$22.00** of a $27 balance. The local entrypoint in
`research/modal_app.py` refuses any run whose own cap exceeds what is left. Each
GPU run also stops itself between batches once its own cost passes its cap.

## Prices (modal.com/pricing, read 2026-10-01)

| Resource | $/s | $/h |
|---|---:|---:|
| A100 40 GB | 0.000583 | 2.10 |
| L40S 48 GB | 0.000542 | 1.95 |
| H100 SXM5 | 0.001097 | 3.95 |
| CPU, per physical core | 0.0000131 | 0.047 |
| Memory, per GiB | 0.00000222 | 0.008 |

## GPU choice: A100-40GB

- The model is ~14 GB resident in BF16. A100-40GB leaves ~25 GB for a KV
  cache, and the README's 24 GB OOM warning does not apply.
- This MoE kernel is bandwidth-bound on GDDR-class cards (A6000, 66–68% DRAM
  utilization, README §Kernel profile). The A100 has 1.55 TB/s of HBM against
  0.86 TB/s for the L40S, at 7.6% more per hour. The expected speed per dollar
  favours the A100. *This is reasoned from bandwidth, not measured on these
  cards.*
- Container: 2 cores + 16 GiB host RAM. Weights stream from safetensors straight
  to the GPU, so the host never holds the model.
  Rate: 0.000583 + 2×0.0000131 + 16×0.00000222 = **$0.000645/s ($2.32/h)**.
- Every run is billed with a 60 s boot allowance on top of its own clock.

## Estimates per phase (to be replaced with measured throughput after the smoke test)

Workload unit: one GSM8K question, 4-shot chat prompt (~1–1.5k tokens), cheap
config `max_tokens=256 block_length=32 steps=256` (fixed schedule), batch 32.

**Estimate, not measured:** 1.0–2.5 s per question per arm on an A100. This is
derived from H100 PCIe 657 tok/s at 128/128/32 (README). It allows ~1.8× more
MoE work per token at L=256, the A100 being ~0.6× an H100, and the longer
prompts.

| Phase | Work | GPU-min (est.) | Cost (est.) |
|---|---|---:|---:|
| 0 | fetch weights + GSM8K (CPU) | — | $0.03 |
| 0 | autotune A100 + GPU test suite | 15–25 | $0.60–1.00 |
| 0 | smoke n=5 | 3–5 | $0.15 |
| 0 | control ×2 at n=200 (self-churn + accuracy) | 10–20 | $0.40–0.80 |
| 1 | telemetry control, n=200 | 6–12 | $0.25–0.50 |
| 2 | 6 arms × n=1000 | 100–250 | $4–10 |
| 3 | OLMoE, 2 arms × n=1000, HF eager MoE | 30–80 | $1.2–3.1 |
| 4 | threshold control + fp32, n=1000 (the fixed pair is reused from Phase 2) | 25–60 | $1.0–2.3 |
| **Total** | | | **$7.6–17.9** |

Phase 0 is under $3 in total, so it runs without a separate go-ahead.
Phase 2 is the one to confirm with measured numbers before launch.

## Measured throughput (Phase 0) and revised estimates

Measured on A100-SXM4-40GB with the A100-tuned MoE config, cheap config, B=32,
4-shot prompts 1497–1645 tokens wide: **26.0–26.6 s per full batch of 32 =
0.82 s/question/arm** (~313 tok/s). The first batch of a process is 32 s
(warm-up). Model load takes 32–36 s.

Modal served one "A100-40GB" request on an A100-SXM4-80GB (the smoke run).
The tuned config is now installed under whatever A100 name the container
gets, and every run asserts that it loaded (`research/modal_app.py`).

| Phase | Work | GPU-min | Cost (revised) |
|---|---|---:|---:|
| 1 | telemetry control, n=200 (+ overhead of the second gate, unmeasured) | 4–6 | $0.15–0.25 |
| 2 | 6 arms × n=1000 | ~85 | ~$3.4 |
| 3 | OLMoE via HF, 2 arms × n=1000 (HF eager MoE loop, not measured yet) | 30–80 | $1.2–3.1 |
| 4 | threshold control + fp32, n=1000 | 20–30 | $0.8–1.2 |

## Phase 2 plan, second revision: 16 arms (supersedes the 11-arm plan below)

Added on review: three causal layer arms (fp32 at all passes but only at layer
0 / layers 11–14 / layer 15) and two equal-size (25%) late step windows
(`steps_within:0.75:1`, `steps_global:0.75:1` = blocks 6–7, to compare with
blocks 0–1). The 25/75 partitions stay for the additivity check.

| # | arm | fp32 / perturbation |
|---:|---|---|
| 1 | control | none |
| 2 | fp32_all | every pass, every layer |
| 3 | fp32_prime | prime pass only |
| 4 | fp32_finalize | finalize passes only |
| 5 | fp32_steps_within_early | steps 0–7 of each block |
| 6 | fp32_steps_within_late | steps 8–31 of each block |
| 7 | fp32_steps_within_last25 | steps 24–31 of each block |
| 8 | fp32_steps_global_early | steps of blocks 0–1 |
| 9 | fp32_steps_global_late | steps of blocks 2–7 |
| 10 | fp32_steps_global_last25 | steps of blocks 6–7 |
| 11 | fp32_layer0 | every pass, layer 0 only |
| 12 | fp32_layers11_14 | every pass, layers 11–14 only |
| 13 | fp32_layer15 | every pass, layer 15 only |
| 14–16 | noise_s0 / s1 / s2 | every pass: bf16 + U(−1, +1) ULP. Larger than bf16 rounding (≤ 0.5 ULP); compared with fp32_all only with that caveat |
| + | control_sentinel ×2 | control's first 64 q, in containers B and C |

**Cost.** One arm at n=1319 = 41 full batches × 26.3 s + one 7-row batch
≈ 1,091 s (from the Phase 0 measurement; prompt widths at n=1319 are not yet
measured, ±3%). 16 × 1,091 + sentinels 105 + 3 × ~100 s load/boot ≈ 17,860 s
× $0.000645 = **$11.5**, above the $11 threshold, so this waits for a decision.

| container | arms | est. | cap |
|---|---|---:|---:|
| A | control, fp32_all, fp32_prime, fp32_finalize, fp32_layer0 | $3.58 | $4.20 |
| B | sentinel + the 6 step-window arms (5–10) | $4.32 | $5.00 |
| C | sentinel + noise_s0–s2 + fp32_layers11_14 + fp32_layer15 | $3.62 | $4.20 |

Cheapest cut that loses no question asked: drop arm 9 (fp32_steps_global_late,
75% of blocks). Its only role is the additivity check, which the within-block
partition (3 + 4 + 5 + 6) already provides; the equal-size global comparison
uses arms 8 vs 10. 15 arms ≈ **$10.8**.

Projected study total with 16 arms: $1.39 + $11.5 + Phase 3 ($1.5–4) +
Phase 4 (~$1.5) ≈ **$16–18.5** of $22.

**Decision (user, 2026-10-01): 15 arms, arm 9 dropped, ≈ $10.8.** Launched as

| container | results dir | arms | est. | cap |
|---|---|---|---:|---:|
| A | `p2A/` | control, fp32_all, fp32_prime, fp32_finalize, fp32_layer0 | $3.58 | $4.20 |
| B | `p2B/` | control_sentinel, steps_within early / late / last25, steps_global early / last25 | $3.62 | $4.20 |
| C | `p2C/` | control_sentinel, noise_s0–s2, fp32_layers11_14, fp32_layer15 | $3.62 | $4.20 |

One results directory per container, so no two containers ever write the same
volume path.

**Actual: $11.66 against the $10.8 estimate, $0.66 over the $11 threshold.**
Per container 5,886 / 6,059 / 6,141 s against ~5,555 s estimated (+6–11%):
per-batch bookkeeping (volume commit, JSONL write, commit-step / confidence
recording) and slightly wider n=1319 prompts (up to 1645 tokens) were not in
the n=200 timing. The per-container caps ($4.20 each, $12.60 total) could not
have enforced the $11 line; future phases set caps that sum to the approved
total. Spent after Phase 2: **$13.05**, remaining **$8.95**.

Data note: downloading results with `modal volume get` straight into this
OneDrive-synced folder left null-byte holes in two of 17 files. The volume
copies were intact. Results are now downloaded to a local temp dir, validated
(row count, unique ids, no null bytes) and only then copied in.

## Phase 2 plan, first revision (11 arms; superseded)

All Phase 2–4 arms run on the **full GSM8K test set, n=1319** (seed-42
`_stable_subset` order; the first 1000 are the n=1000 selection, verified in
the volume). Cheap config, fixed schedule, B=32: 42 batches per arm.

| # | arm | what is fp32 / perturbed |
|---:|---|---|
| 1 | control | nothing (bf16 router, production) |
| 2 | fp32_all | every pass |
| 3 | fp32_prime | the cache-prime pass only (commits prompt K/V) |
| 4 | fp32_finalize | the 8 finalize passes only (commit each block's K/V) |
| 5 | fp32_steps_within_early | denoising steps with s/32 < 0.25 (steps 0–7 of every block) |
| 6 | fp32_steps_within_late | denoising steps 8–31 of every block |
| 7 | fp32_steps_global_early | denoising steps of blocks 0–1 |
| 8 | fp32_steps_global_late | denoising steps of blocks 2–7 |
| 9–11 | noise_s0 / s1 / s2 | every pass: bf16 logits + U(−1, +1) bf16 ULP, seeded per (seed, batch) |
| + | control_sentinel ×2 | control's first 2 batches (64 q), re-run in each extra container |

Arms 3 + 4 + 5 + 6 (or 3 + 4 + 7 + 8) partition every forward pass of arm 2
exactly (tested). Step-only windows never touch the cache-writing passes.

**Cost, from measured 0.82 s/question/arm:** 11 arms × 1319 q × 0.82 s =
11,900 s, plus sentinels (2 × 64 q = 105 s) and 3 model loads + warm-up
(~200 s) ≈ 12,200 GPU-s × $0.000645 = **$7.9**. Split across 3 containers run
in parallel (~70 min each; every function's timeout is 3 h, and each has its
own cost cap):

| container | arms | est. | cap |
|---|---|---:|---:|
| A | control, fp32_all, fp32_prime, fp32_finalize | $2.85 | $3.50 |
| B | control_sentinel, 4 step-window arms | $2.90 | $3.50 |
| C | control_sentinel, noise_s0–s2 | $2.20 | $2.80 |

Projected totals: $1.39 spent + $7.9 (Phase 2) + Phase 3 OLMoE at n=1319
($1.5–4, unmeasured) + Phase 4 threshold pair at n=1319 (~$1.5) ≈
**$12–15** of the $22 cap.

## Phase 3 (OLMoE): actual

Cap $4.50 including the smoke test; **spent $3.45**: fetch $0.007, smoke
$0.29, self-churn and telemetry at n=200 $0.47, main $2.68. Measured
throughput with HF eager MoE at B=64 was ~85 s per batch (1.33 s per
question per arm). Four arms at n=1319 would have cost ~$4.75, so as
instructed the main run used the **805 questions uncapped in LLaDA's
control**, with all four arms kept. The run-log row says "n=1319" because
that is the selection file; `--ids-file research/data/llada_uncapped_ids.json`
restricted it to 805.

Spent after Phase 3: **$16.49**; remaining **$5.51**, of which $1.60 is
reserved for Phase 4.

## Phase 4 (threshold decoding): actual

Estimate ≤ $1.40 (an upper bound from the fixed-schedule speed); cap $1.60;
**spent $1.07**. Threshold decoding ran at ~15.5 s per batch of 32, against
26 s for the fixed schedule (62% of the denoising steps).

**Study total after Phase 4: $17.57 of the $22 cap** (remaining $4.43). Phase
5 (write-up) needs no GPU.

## Matched-dose experiment: plan

fp32 on exactly one forward pass. LLaDA: step 0 and step 16 of block 0, run
over all 1319 questions so the batches match Phase 2's control, which is
reused. OLMoE: the decode forward that routes generated token 0, and the one
that routes token 16, over the same 805 questions in the same batches as
Phase 3's control (reused), plus a one-batch control sentinel for
cross-container reproducibility. Analysed on the 802 questions uncapped in
both models' controls.

| run | arithmetic | estimate | cap |
|---|---|---:|---:|
| LLaDA | 2 arms × 42 batches × 26.5 s + 100 s load/boot | $1.50 | $1.70 |
| OLMoE | 2 arms × 13 batches × 85 s + 85 s sentinel + 100 s | $1.56 | $1.75 |
| **total** | | **$3.06** | $3.45 |

Remaining before: $4.43. Expected left after: $1.37; worst case at both
caps: $0.98. Both clear the required $0.50.

## Run log

Columns: seconds include the 60 s boot allowance; cost = seconds × rate; the
cumulative column is the running total against the $22 cap.

| run | what | GPU | seconds | $/s | cost | cumulative | status |
|---|---|---|---:|---:|---:|---:|---|
<!-- ledger -->
| run-20261001-111858-fetch | weights + GSM8K selection | CPU | 127 | 0.000044 | $0.0056 | $0.0056 | done |
| run-20261001-112030-setup | autotune + GPU tests | A100-40GB | 1216 | 0.000645 | $0.7839 | $0.7895 | done |
| run-20261001-114019-p0_smoke | p0_smoke n=5 arms=control,control_rep,control_tel,fp32_all | A100-40GB | 167 | 0.000645 | $0.1078 | $0.8973 | done |
| run-20261001-114610-p0 | p0 n=200 arms=control,control_rep | A100-40GB | 463 | 0.000645 | $0.2985 | $1.1958 | done |
| run-20261001-115924-p1 | p1 n=200 arms=control_tel | A100-40GB | 291 | 0.000645 | $0.1875 | $1.3833 | done |
| run-20261001-120634-fetch | weights + GSM8K selection | CPU | 67 | 0.000044 | $0.0029 | $1.3862 | done |
| run-20261001-122501-p2C | p2C n=1319 arms=control_sentinel,noise_s0,noise_s1,noise_s2,fp32_layers11_14,fp32_layer15 | A100-40GB | 5886 | 0.000645 | $3.7946 | $5.1808 | done |
| run-20261001-122441-p2B | p2B n=1319 arms=control_sentinel,fp32_steps_within_early,fp32_steps_within_late,fp32_steps_within_last25,fp32_steps_global_early,fp32_steps_global_last25 | A100-40GB | 6059 | 0.000645 | $3.9065 | $9.0873 | done |
| run-20261001-122423-p2A | p2A n=1319 arms=control,fp32_all,fp32_prime,fp32_finalize,fp32_layer0 | A100-40GB | 6141 | 0.000645 | $3.9595 | $13.0468 | done |
| run-20261001-164532-fetch | weights + GSM8K selection | CPU | 150 | 0.000044 | $0.0066 | $13.0534 | done |
| run-20261001-164720-p3_smoke | p3_smoke OLMoE n=5 arms=control,control_rep,control_tel,fp32_all,fp32_prefill,noise_s0 | A100-40GB | 446 | 0.000645 | $0.2878 | $13.3412 | done |
| run-20261001-165550-p3_selfchurn | p3_selfchurn OLMoE n=200 arms=control_tel,control_rep | A100-40GB | 734 | 0.000645 | $0.4731 | $13.8143 | done |
| run-20261001-170731-p3 | p3 OLMoE n=1319 arms=control,fp32_all,fp32_prefill,noise_s0 | A100-40GB | 4157 | 0.000645 | $2.6803 | $16.4946 | done |
| run-20261001-203122-p4 | p4 n=1319 arms=control,fp32_all | A100-40GB | 1667 | 0.000645 | $1.0748 | $17.5694 | done |
| run-20261001-221116-p2A | p2A n=1319 arms=fp32_single_b0s0,fp32_single_b0s16 | A100-40GB | 1939 | 0.000645 | $1.2501 | $18.8195 | interrupted: local client lost network, app stopped; billed to last heartbeat (up to 1 batch more) |
| run-20261001-221145-p3 | p3 OLMoE n=1319 arms=control_sentinel,fp32_dec0,fp32_dec16 | A100-40GB | 1876 | 0.000645 | $1.2098 | $20.0293 | interrupted: local client lost network, app stopped; billed to last heartbeat (up to 1 batch more) |
| run-20261002-095932-p2A | p2A n=1319 arms=fp32_single_b0s0,fp32_single_b0s16 | A100-40GB | 262 | 0.000645 | $0.1692 | $20.1985 | interrupted: local client lost network a second time; app stopped despite --detach (blocking .remote) |
| run-20261002-095951-p3 | p3 OLMoE n=1319 arms=control_sentinel,fp32_dec0,fp32_dec16 | A100-40GB | 252 | 0.000645 | $0.1626 | $20.3611 | interrupted: local client lost network a second time; app stopped despite --detach (blocking .remote) |
| run-20261002-120905-p2A | p2A n=1319 arms=fp32_single_b0s0,fp32_single_b0s16 | A100-40GB | 410 | 0.000645 | $0.2644 | $20.6255 | done |
| run-20261002-120918-p3 | p3 OLMoE n=1319 arms=control_sentinel,fp32_dec0,fp32_dec16 | A100-40GB | 240 | 0.000645 | $0.1545 | $20.7800 | done |
