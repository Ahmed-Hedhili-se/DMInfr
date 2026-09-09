<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/logo/dminfr-lockup.svg">
    <img src="assets/logo/dminfr-lockup-light.svg" alt="DMInfr" width="340">
  </picture>
</p>

Self-contained PyTorch reimplementation of [inclusionAI/LLaDA-MoE-7B-A1B-Instruct](https://huggingface.co/inclusionAI/LLaDA-MoE-7B-A1B-Instruct), a masked-diffusion LLM. Triton fused MoE, block-wise KV caching, fused RMSNorm/decode kernels, variable-length batching, an OpenAI-compatible server, a data-parallel router, and an end-to-end-aware autotuner.

**~955 tok/s on 2× H100 PCIe** · **10.2× single-request and ~179× single-GPU total-pipeline throughput** vs the unoptimized baseline.

Measured on 2× H100 PCIe (full log: [`h100x2_bench.md`](docs/h100x2_bench.md)) and on a single RTX A6000. Ratios are **not hardware-portable** — the baseline is CPU-dispatch-bound, so the same engine scores 8.70× on A6000 and 10.2× on H100 with no code difference.

> The mark is an 8×10 token lattice shaped as a **D**: masked in the upper left, resolved in the lower right, with four tokens stepping violet→blue through the counter — this engine's own decoding process, drawn. Full identity in [`assets/logo/`](assets/logo/README.md).

---

---

## Contents

- **[Benchmarks](#benchmarks)** · **[Correctness](#correctness--tests)** · **[Quantization](#quantization)** · [Quick start](#quick-start)
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — model internals, the 7 optimizations, autotuner, project layout
- [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) — install, launch, multi-GPU topology, decoding configuration
- [`docs/h100x2_bench.md`](docs/h100x2_bench.md) — the full 2× H100 validation log: every measurement, the bugs found, and the hypotheses tested and rejected
- [`docs/INVESTIGATION_LOG.md`](docs/INVESTIGATION_LOG.md) — the correctness investigation, including what was retracted

---

## How to read these numbers

Two things dominate every figure below, and both were measured rather than assumed:

| | Measured noise floor | Consequence |
|---|---|---|
| **Throughput** | **~±5%** run-to-run (±15% at high batch) — an *unchanged* arm moved 4.7% between runs | Anything under ~10% is not a result. H100 figures are 3 reps/point with `sd`; A6000 figures are single samples. |
| **Accuracy** | **~17% of GSM8K questions flip** under *any* numerical perturbation on this checkpoint | n=200 carries ±3pt of intrinsic churn. Marginal comparison at that scale is invalid — **paired McNemar required**, and resolving a 3-point effect needs n≈1000. |

Several previously-published numbers in this repo did not survive that
standard. They are corrected below and the retractions are recorded in
[`docs/h100x2_bench.md`](docs/h100x2_bench.md) rather than quietly dropped.

---

## Benchmarks


> Two hardware sets below. **H100 numbers are 3 repetitions per point**; A6000
> numbers are single samples and should be read as approximate — the measured
> run-to-run noise floor on this harness is **~±5%** (up to ±15% at high batch),
> so anything under ~10% is not a result. See [`docs/h100x2_bench.md`](docs/h100x2_bench.md) §11.

### 2× H100 PCIe — current

**Throughput, single GPU** (fused RoPE on, 3 reps/point)

| `BATCH_MAX_SIZE` | Tok/s | sd | p50 |
|---:|---:|---:|---:|
| **32** ← optimum | **656.9** | 3.5 | **5.33 s** |
| 64 | 585.0 | 34.7 | 12.07 s |
| 128 | 607.2 | 54.4 | 23.04 s |

**Throughput, DP=2 across both GPUs**

| Concurrency | Tok/s | sd | p50 |
|---:|---:|---:|---:|
| 32 | 871.6 | 23.9 | **4.18 s** |
| **64** ← peak | **954.5** | 52.8 | 5.48 s |
| 128 | 599.3 | 129.3 | 16.17 s |

Past concurrency 64 throughput *drops* while p50 triples — 64 total (32/replica)
is the operating point, not a floor to push past.

**Speedup** (5 runs, 0/128 token divergence vs baseline) — **all four rows
measured on the 2× H100 box**, including the baseline

| | Tok/s | vs baseline |
|---|---:|---:|
| Baseline (`dminfr/reference/`), single request | 3.67 | 1.00× |
| Optimized, single request | 37.29 | **10.2×** |
| 1 GPU batched (`BATCH_MAX_SIZE=32`) | 656.9 | **~179×** |
| 2 GPUs (DP=2, concurrency 64) | 954.5 | **~260×** |

Decomposition cross-checks (10.16 × 17.62 = 179.0), but the `dminfr/reference/` baseline
drifted 7.8% between two runs of identical code — **quote a range: ~165–180×
on one GPU, ~225–260× on two**, not a single digit.

> **These ratios have not been updated with the 8× H100 numbers, deliberately.**
> That run measured throughput only; the baseline and the single-request
> optimized figure were never re-taken on that machine. Dividing the new
> throughput by the old baseline would give 183× / 372× / 508× for 1 / 2 / 4
> GPUs, but it crosses hardware — and the baseline drifts almost 8% between
> runs of identical code on *one* machine, so crossing two is worse. Restoring
> a sound ladder needs one command on whatever GPU you have next:
> `python -m benchmarks.check_time_inference --weight-dir weights --mode both`.
>
> What can be said without a new baseline: single-GPU throughput is unchanged
> (656.9 → 670.2, inside the ±5% noise floor), while **two-GPU throughput
> measured 43% higher** (954.5 → 1366.3), which is well outside it.

### 8× H100 PCIe — parallelism matrix

Measured 2026-09-06 on an 8× H100 PCIe node (252 cores, 1.4 TB RAM). The GPUs
are **NVLink-paired** — `0-1`, `2-3`, `4-5`, `6-7` are `NV12`, everything else
is PCIe — which finally answers a caveat this README carried for a month:
whether tensor parallelism looks better with NVLink. It does not.

**Data-parallel scaling** (3 reps/point, one replica per GPU)

| Replicas | Concurrency 32 | Concurrency 64 | Concurrency 128 |
|---:|---:|---:|---:|
| 1 GPU | 670.2 ± 29.7 | 644.2 ± 31.2 | — |
| 2 GPUs | 1032.1 ± 34.0 | **1366.3 ± 2.6** | 752.3 ± 106.4 |
| 4 GPUs | 1132.3 ± 45.1 | **1863.2 ± 161.5** | 921.4 ± 109.0 |

Scaling at the per-configuration optimum is **2.04× at 2 GPUs and 2.78× at 4**,
against 1.45× measured previously on a 2-GPU box. Two things changed: this node
has far more host headroom (252 cores vs. a much smaller box, and host
contention was the measured cause of the old shortfall), and the tuned MoE
config now actually loads — see the note below.

Every configuration peaks at **concurrency 64 and collapses at 128**. That is
consistent across replica counts, so treat 64 as the operating point rather
than a floor to push past.

**All three topologies at once, on disjoint GPUs**

`--gpu-list` and `--master-port-base` allow independent deployments to share a
node, so the full matrix can run simultaneously: TP/EP on GPUs 0-1, DP on 2-3,
hybrid DP×TP on 4-7.

| Topology | GPUs | Alone | Concurrent with the other two |
|---|---:|---:|---:|
| **DP = 2** | 2 | **1348.8 ± 3.3** (conc 64) | 961.5 ± 62.9 (conc 32) |
| **Hybrid DP=2 × TP=2** | 4 | 66.1 ± 0.9 (conc 16) | 71.3 ± 1.0 (conc 8) |
| **TP/EP = 2** | 2 | **4.7** (conc 8, n=1) | timed out |

Read that table carefully, because the ordering is not intuitive:

- **DP on 2 GPUs beats hybrid on 4 GPUs by roughly 20×**, and beats TP/EP on the
  same 2 GPUs by nearly 300×.
- **NVLink does not rescue tensor parallelism.** Each hybrid replica sits on an
  NVLink pair and each TP/EP rank pair likewise; the result is still an order of
  magnitude short of plain replication. The cause is not interconnect bandwidth,
  it is that `server.py` disables request batching whenever `tp_size > 1`, so
  every request serialises through one lock. Interconnect was never the
  bottleneck; batching was.
- TP/EP at 4.7 tok/s is slow enough that a 24-request run exceeds the client's
  600 s timeout, which is why most of its cells failed rather than returning a
  number. The single surviving measurement is reported with `n=1` and should be
  read as an order of magnitude, not a figure.
- DP loses about **29%** when the other two topologies are hammering the same
  host (1348.8 → 961.5). Splitting a node between deployments is not free.

**Use data parallelism.** The hybrid and TP paths exist so the trade-off can be
measured, and are wired up (`--tp-size`, `--gpu-list`); they are not the fast
path, and this measurement is the reason.

**DP=8 is not in the table because it never loaded.** All eight replicas
launched and stayed alive, but the GPUs never went past 4 MiB and the router
never came up. The likely cause is disk: eight replicas each reading a 14 GB
checkpoint at once is ~112 GB of concurrent reads from one volume, and DP=1
through DP=4 loaded without trouble. That is a plausible explanation, not a
verified one — the rented node expired before it could be confirmed. Extrapolating
2.78× at four GPUs suggests roughly 2400–2700 tok/s at eight, which is why it
is worth measuring, and why it is stated here as an extrapolation rather than
entered in the table.

> **A TP deployment that receives no traffic for 10 minutes kills itself.** The
> worker ranks block in `broadcast_object_list()` inside `worker_loop()` with a
> 600 s timeout, and rank 0 only joins that collective when a request arrives.
> Benchmark a TP arm immediately after it comes up, or it will be dead by the
> time you reach it.

> **Tuned MoE configs were silently not loading between `1fca44f` and
> `1fb0370`.** The restructure moved `fused_moe_triton.py` one directory deeper,
> and its two-level walk to the repo root started resolving to `dminfr/`, so
> `TUNED_CONFIGS` loaded empty and the kernel fell back to hardcoded tile
> shapes. The autotuner had the mirror bug and wrote its output where the
> loader did not look. Any throughput number taken in that window on a machine
> without a repo-root config was measuring untuned kernels.

### Single request vs baseline — RTX A6000

```bash
python -m benchmarks.check_time_inference --weight-dir weights \
    --gen-length 128 --steps 128 --block-length 32 --mode both --num-runs 3
```

| | Time | Tok/s | Speedup |
|---|---:|---:|---:|
| Baseline (`dminfr/reference/`, unfused, no cache) | 46.67 s | 2.74 | 1.00× |
| **Optimized** | **5.37 s** | **23.85** | **8.70×** |

> The ratio is **not hardware-portable**. `dminfr/reference/`'s MoE loops 64 experts in Python, so the baseline is CPU-dispatch-bound while the optimized path is GPU-bound. An earlier measurement on a different box (A40-24Q) gave 6.46× with a 29.69 s baseline — the difference is mostly the host CPU, not the engine.

### Total pipeline: 103×

Both arms through the same HTTP client and load harness, so this is a deployment-to-deployment number:

| Pipeline | Tok/s |
|---|---:|
| `dminfr/reference/` over HTTP, serialized | **2.7** |
| `dminfr/engine/`, `BATCH_MAX_SIZE=32`, concurrency 32 | **278.3** |
| | **103×** |

It decomposes as **8.70× engine × 11.67× batching = 101.5**, within 1.5% of the direct ratio — the check that makes the figure trustworthy rather than a headline. The baseline is single-request by construction: `dminfr/reference/generate.py` hardcodes the batch dimension to 1.

### Batched throughput — A6000, `BATCH_MAX_SIZE=32`, concurrency 32

| | Tok/s |
|---|---:|
| Fixed prompts, fusions off | 232.1 |
| **Fixed prompts, fusions on** | **278.3 (+19.7%)** |
| **Varied prompts** (8 prompts, 6 lengths) | **240.0** |

The fusion gain is a **batched** gain; the same fusions on single-request GSM8K measure ~2%, because MoE weight streaming dominates at batch 1.

> Figures marked *fixed prompts* send an identical prompt for every request. Before variable-length batching that was the only way to fill a batch — with varied lengths the queue fragmented into batches of ~4. Treat fixed-prompt numbers as a ceiling, not as expected traffic.

### Throughput vs batch size

| `BATCH_MAX_SIZE` | Tok/s | Δ throughput | Δ batch | p50 |
|---:|---:|---:|---:|---:|
| 8 | 150.3 | — | — | 6.77 s |
| 16 | 204.8 | +36.3% | +100% | 9.93 s |
| 24 | 224.8 | +9.8% | +50% | 13.56 s |
| **32** | **243.2** | +8.2% | +33% | 16.71 s |
| 48 | **0/96 succeeded** | — | — | — |

Throughput is past its knee by 32: 4× the batch buys 1.62× the throughput. The default of 64 was tuned on a 48 GB A6000 and **kills a 24 GB card** — set it per deployment. Rough sizing per sequence at length `L`, block `b`: KV `128 KB × L`, MoE transients `~48 KB × L`, logits `307 KB × b`.

### Kernel profile

`fused_moe_kernel` is the dominant kernel — **58.75% of GPU time on A6000, 55.3% on H100**. Where the bottleneck sits is **hardware-dependent**, so the levers differ per card.

On A6000 it is close to a weight-streaming wall: the expert weights (805 MB/layer) stream every forward regardless of token count.

On H100 it is **not**. Nsight Compute at the production shape measures **DRAM throughput 14.5%, L2 79.7%, compute 49.6%** — ncu names the L2 as the bottleneck outright, and DRAM is nowhere near saturated. An earlier revision of this section claimed "81% of theoretical weight-streaming bandwidth" and "no kernel headroom left"; that is an A6000 figure and does not transfer. See [`docs/h100x2_bench.md`](docs/h100x2_bench.md) §10.

> Nsight's "Memory Throughput ~96%" is **L2**, not DRAM (DRAM sits at 66–68%). Read as a DRAM ceiling it says "nothing to gain"; read as an L2 ceiling it says "remove intermediate traffic" — which is what produced optimization 3.

---

## Correctness & Tests


**GSM8K on 2× H100: 75.2% at n=1000** (`steps=1024 block_length=64 threshold 0.9/0.4`).
Regression suite **9/9**, `LLaDA_Quant` **276/276**, autotuner validates
`cos_sim = 1.000000` at every M, optimized output is **character-identical** to
the baseline (0/128 divergence).

> **Sample size matters more than anything else here.** The same engine and config
> scores 74.0% at n=50, 71.0% at n=200 and **75.2% at n=1000** — the small
> subsets are simply unrepresentative. Worse, **~17% of GSM8K questions flip
> outcome under *any* numerical perturbation** on this checkpoint (measured: a
> router-precision change with no accuracy effect still churned 167/1000
> questions). So n=200 carries ±3pt of intrinsic noise, marginal comparison at
> that scale is invalid, and **paired McNemar is required** — resolving a 3-point
> effect needs n≈1000. Every accuracy figure below n=1000 in this repo's history
> inherits this. See [`docs/h100x2_bench.md`](docs/h100x2_bench.md) §9.

Against the HuggingFace reference at the logit level: **3,219/3,219 weights mapped**, logit cosine **0.9781**, top-1 token match **91.0%**.

A residual ~6pt fixed-schedule gap vs HF was investigated and closed as **inherent, not a bug**: this checkpoint's router is near-uniform (top-1 weight ~1.7–5%), so bf16-level noise flips top-8 expert membership for 43–90% of positions per layer. A 2×2 kernel-isolation matrix exonerated the Triton MoE kernel entirely. See [`docs/INVESTIGATION_LOG.md`](docs/INVESTIGATION_LOG.md) §2.9–2.11.

Regression tests (`python -m tests.<name>`):

| | |
|---|---|
| `test_fusions` | RMSNorm / decode / QKV numerics + speed, and token identity |
| `test_variable_length_batch` | a padded row reproduces its solo run **exactly** |
| `test_num_logits_slice` | narrowed `lm_head` is bit-exact |
| `test_moe_align_block_size` | vectorized alignment vs a frozen reference |
| `test_select_transfer_indices` | vectorized per-row top-k vs a frozen reference |
| `test_router` | dispatch, health, failover, recovery (no GPU needed) |

> **Accuracy figures predating `_stable_subset` are not comparable across machines.** `random.shuffle` over a HuggingFace dataset whose row order varies by version meant the same `--seed` selected *different questions* on different boxes — two runs shared only 6 of their first 10. Fixed by sorting on a content hash before shuffling. Re-measure before comparing any historical accuracy number.

---

### Decoding: anchored accuracy pair

`ANCHORED` — A6000, GSM8K n=50 seed=42, `max_tokens=1024 steps=512 block_length=64`, threshold 0.9/low 0.4, under the fixed question selection:

| Config | Accuracy | s/question |
|---|---:|---:|
| **BF16** | **74.0%** (37/50) | **7.2** |
| **INT8 + fused W8A16** | **76.0%** (38/50) | **5.8** |

Same 50 questions, same box — the first genuinely comparable accuracy pair in this project. INT8 is 19% faster *and* one question ahead, which at n=50 is noise on the accuracy axis and a real win on the time axis.

> Configuration guidance — including why `steps_per_block >= block_length` is
> the recommended setting rather than the documented floor — is in
> [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md).

---

## Quantization


**This engine runs BF16.** Nothing in `dminfr/engine/` quantizes anything. (The
`use_fp8_w8a8` / `use_int8_w8a16` parameters that used to sit in
`fused_moe_kernel` were inherited from vLLM, always `False`, never referenced —
they have since been removed.)

INT8 is available through the optional [`LLaDA_Quant`](https://github.com/Ahmed-Hedhili-se/LLaDA_Quant) toolkit, imported lazily and absent from `requirements.txt`:

```bash
bash scripts/start.sh --backend fast_dense --weight-dir weights --quantize int8
bash scripts/start.sh --backend fast_dense --weight-dir weights --quantize fp8   # E4M3
```

### Measured on 2× H100

Resident memory, tensor-level accounting (`LLaDA_Quant.memory.resident_memory` —
**not** `nvidia-smi`, which reports the allocator's peak reserve and made
quantization look like it *grew* the model):

| | Resident | vs BF16 |
|---|---:|---:|
| BF16 | 14032 MiB | — |
| INT8 packed | 8080 MiB | **−42.4%** |
| FP8-E4M3 packed | 8080 MiB | **−42.4%** |

Identical, as they must be — both are 1 byte/weight over the same shapes.

Accuracy, each arm **paired against BF16 on its own identical questions**,
McNemar:

| Arm | n | BF16 on *those* n | quantized | **Δ vs own BF16** | McNemar p |
|---|---:|---:|---:|---:|---:|
| INT8 (packed + fused W8A16) | 200 | 71.0% | 68.5% | **−2.5 pt** | 0.50 |
| FP8-E4M3 (packed, no fused kernel) | 100 | 73.0% | 70.0% | **−3.0 pt** | 0.61 |

> ⚠️ **Do not compare the two `quantized` percentages to each other.** The rows
> use *different question sets* — INT8 was scored on the first 200 of the seeded
> subset, FP8 on the first 100, and the first 100 are easier (visible in the
> BF16 column: 73.0% vs 71.0%). FP8's 70.0% is **not** "better than" INT8's
> 68.5%. Only the Δ column, each row against its own BF16 baseline, is
> meaningful — and by that measure FP8 is marginally *worse*, not better.
>
> FP8's arm stops at n=100 because without a fused kernel it runs ~3× slower per
> question; n=1000 would be ~3.5 h. A direct INT8-vs-FP8 pairing on one shared
> question set has not been run.

**Neither Δ is significant** on its own. But pooling every accuracy
measurement this project has taken of the two formats — including the A6000
historical figure below — separates them:

| | measurements | pattern | read |
|---|---|---|---|
| **INT8** | A6000 n=50 **+2.0**, H100 n=50 **0.0**, H100 n=200 **−2.5** | scatters around zero, across two GPUs | **no measurable accuracy cost** |
| **FP8** | H100 n=50 **−4.0**, H100 n=100 **−3.0** | both negative, stable estimate | small real cost **not excluded** |

Three INT8 samples landing positive, zero and negative is what "no effect"
looks like — the A6000 run that put INT8 *ahead* was noise in the same way our
n=200 run that puts it behind is. FP8's two both point the same way, which is
weak evidence rather than proof, but it is a different pattern.

So the defensible statement is **"INT8 is measurably free; FP8 is probably
slightly lossy but unconfirmed"** — not that the two are interchangeable.
Settling FP8 needs n≈1000, which it cannot reach without a fused kernel.

**On speed the two are not close:** INT8 has a fused W8A16 kernel and runs at
parity with BF16; FP8 has none and runs ~3× slower per question. **INT8 is the
deployable format**; FP8 is a memory play only until it gets a kernel.

### Historical — one A6000, 128-token generation

| Load | BF16 | INT8 PACKED | INT8 PACKED + fused W8A16 |
|---|---:|---:|---:|
| GSM8K, per question (anchored) | 7.2 s | — | **5.8 s (1.24×)** |
| Single request | 11.88 s | 60.57 s (0.20×) | **10.71 s (1.11×)** |
| Concurrency 32 | 230.5 tok/s | — | 227.6 tok/s (0.99×) |
| Concurrency 64 | 230.8 tok/s | 98.3 (0.43×) | 114.4 tok/s (0.50×) |
| Resident | 13.70 GiB | 7.89 GiB | **7.89 GiB (0.58×)** |

Memory drops 42% at every load; speed **inverts with batch size** — faster than BF16 at batch 1, parity at 32, half the speed at 64. The concurrency-64 collapse is not explained by the roofline (BF16 is flat from 32→64 while the quantized arm halves) and remains open.

> ### The A6000 speed win does not transfer to H100 — and the profiling says why
>
> On A6000, `INT8 + fused W8A16` was **19% faster** than BF16 on GSM8K
> (5.8 vs 7.2 s/question). On H100 the same configuration is **24% slower**
> (4.2 vs 3.4 s/question). Same code, opposite sign.
>
> | | DRAM throughput | What INT8 does |
> |---|---:|---|
> | A6000 | 66–68% | close to the bandwidth wall, so halving the weight bytes relieves the **binding** constraint → faster |
> | H100 | **14.5%** | DRAM was never the constraint (L2 is, at 79.7%) → fewer bytes buys nothing, while the W8A16 dequantize work still costs → slower |
>
> **INT8 buys speed only when you are bandwidth-starved.** The A6000 is; the
> H100 is not. This is the same measurement that overturned the "no kernel
> headroom" claim in [Kernel profile](#kernel-profile) — once you know DRAM sits
> at 14.5% on H100, INT8 losing there is the predictable outcome rather than a
> surprise. Memory savings (−42.4%) are hardware-independent and hold on both.

**Recommendation:** on a card that fits the model, stay BF16 for throughput.
Quantization earns its place when memory is the binding constraint, or for
single-request latency. On H100 specifically, using the freed memory to run
*more replicas per GPU* was tested and is a **net loss** — the MoE kernel is
bandwidth-bound, so co-located replicas compete for bytes rather than adding
throughput ([`docs/h100x2_bench.md`](docs/h100x2_bench.md) §8d).

---

## Quick start

```bash
bash scripts/setup.sh                              # venv + torch + ~15 GB weights
python -m dminfr.tuning.autotune_moe --model FULL_CFG   # do not skip
bash scripts/start.sh --backend fast_dense --weight-dir weights
```

```bash
# regression suite
python -m tests.test_fusions

# latency vs the unoptimized baseline
python -m benchmarks.check_time_inference --weight-dir weights --mode both

# throughput
python -m benchmarks.throughput.run_throughput --base-url http://localhost:8000     --concurrency 32 --n-requests 96 --max-tokens 128 --steps 128 --block-length 32

# accuracy
python -m benchmarks.correctness.run_math_reasoning_code --task gsm8k     --limit 200 --seed 42 --max-tokens 1024 --steps 1024 --block-length 64     --confidence-threshold 0.9 --low-confidence-threshold 0.4
```

Full installation, multi-GPU launch and configuration:
**[`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md)**.
