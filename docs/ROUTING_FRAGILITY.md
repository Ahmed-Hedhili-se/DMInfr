# Routing fragility in LLaDA-MoE: what a tiny numerical perturbation does to answers

> **Question.** LLaDA-MoE's router is close to uniform. Computing it in bf16
> instead of fp32 already changes the top-8 expert set on about a quarter of
> routed positions. How does that fragility propagate through the denoising
> trajectory, and how large is the perturbation floor that every inference
> optimization has to be measured against?
>
> **Answer.**
> - A minimal perturbation (fp32 instead of bf16 router logits, or ±1 bf16 ULP
>   of noise on them) **leaves accuracy unchanged**, but **changes the final
>   answer on ~12% of questions for LLaDA-MoE and ~20% for an autoregressive
>   MoE with a far sharper router (OLMoE)**.
> - In both models, **7–9% of questions flip between correct and wrong**, in
>   both directions equally.
> - Router flatness decides how much routing noise there is. It does not
>   decide whether the answer changes. What fits the results best is whether
>   the perturbation reaches **state that later computation reads back**:
>   cache-writing passes and the tokens a pass commits.

Measured on Modal A100-40GB, October 2026, for $20.78 of a $22 budget. Every
claim below names its n, question set and configuration. All headline numbers
are re-derived independently from the raw data by `research/verify_headlines.py`
(35/35 match). Code, reports and plots are under [`research/`](../research/).

---

## 1. Headline, and what it means for evaluating optimizations

| | LLaDA-MoE | OLMoE-1B-7B-Instruct |
|---|---:|---:|
| answer changes under fp32 routing (802 questions uncapped in both) | **12.0%** [9.9, 14.4] | **19.3%** [16.7, 22.2] |
| correct ↔ wrong flips (C→W + W→C) | 69 (8.6%): 34 + 35 | 53 (6.6%): 28 + 25 |
| accuracy, control → fp32 (same 802) | 86.9 → 87.0% | 56.0 → 55.6% |
| exact McNemar p, every perturbation arm | ≥ 0.057, none survives Holm | ≥ 0.18 |

"Answer" is the repo grader's own extraction, on questions whose control
generation finished before the 256-token budget; see §7 for why. Correct and
wrong flips balance, so accuracy doesn't move. The perturbation reshuffles
which questions are right.

### The practical consequence: the accuracy noise floor

Any change to an inference stack that perturbs routing numerics (a kernel,
a quantization scheme, a different GPU, a batching change) will move this
model's paired accuracy by noise of this size even if it has no real effect.
Under symmetric churn, the paired accuracy difference has SD = √(d/n), where
d is the fraction of questions whose correctness flips. The figures below use
measured d values from n=1319, LLaDA-MoE, at
`max_tokens=256 block_length=32 steps=256`:

| n | SD, fixed schedule (d = 13.9%) | 95% half-width | SD, threshold 0.9/0.4 (d = 16.2%) | effect needed for 80% power (≈ 2.8 SD, fixed) |
|---:|---:|---:|---:|---:|
| 200 | 2.63 pt | ±5.2 pt | 2.85 pt | ~7.4 pt |
| 500 | 1.67 pt | ±3.3 pt | 1.80 pt | ~4.7 pt |
| 1000 | 1.18 pt | ±2.3 pt | 1.27 pt | ~3.3 pt |
| 1319 (full test set) | 1.03 pt | ±2.0 pt | 1.11 pt | ~2.9 pt |

- d comes from the grader's correct/wrong verdicts, so it doesn't depend on
  how answers are extracted.
- The 80%-power column is a normal approximation.
- The repo's earlier rule of thumb, "±3 pt at n=200" (README;
  `h100x2_bench.md` §9b), is the 1-SD value at d ≈ 18%. It is consistent with
  this table.
- The threshold-decoding d (16.2%, 214/1319) reproduces §9e's 167/1000 =
  16.7%, measured on a different GPU at `steps=1024 block_length=64`.

**Rule:** an accuracy change below ~3 points on the full GSM8K test set, or
~7 points at n=200, is not resolvable for this checkpoint without more data.
Always use paired McNemar, never marginal accuracies. Across the 14
perturbation arms of SQ2, none survives Holm correction (smallest adjusted
p = 0.80), as it should not.

---

## 2. Mechanism: the hypothesis that best fits

> **Hypothesis.** A routing perturbation changes the answer when it reaches
> state that later computation reads back: a KV-cache write, or a committed
> token. Router flatness controls how many routing decisions flip. It does
> not control whether those flips matter.

Evidence for it:

- **Cache-writing passes dominate in LLaDA.** fp32 routing on only the one
  cache-priming pass (prompt K/V) changes 12.3% of answers, against 11.9%
  for every pass. The finalize passes alone (they cache each finished block)
  change 9.2%. Both are far above what their share of routing flips predicts
  (§5).
- **One denoising step barely matters.** fp32 on only block 0, step 0 of
  LLaDA flips ~1,140 top-8 sets, yet the output diverges on just 10.6% of
  questions and the answer changes on 1.4%. That step writes no cache. It
  matters only through the one token per row it commits.
- **One AR decode step matters a lot.** fp32 on only the decode step that
  routes OLMoE's first generated token flips ~0.2 top-8 sets on average, yet
  the output diverges on 39.9% of questions and the answer changes on 17.0%.
  The perturbed hidden state goes into the KV cache that every later token
  reads. It acts even when no expert set flips: 41% divergence among
  questions with zero flipped sets. The fp32 gate also shifts the routing
  *weights*.
- **One hazard curve fits every LLaDA arm.** Define divergence as the first
  differing output token. Then a single function h(position) =
  P(answer changes | output first diverges there), estimated from the other
  arms, predicts all 14 arms' churn with χ² = 12.7. The only borderline arm
  is blocks 6–7, at z = −2.1. It also predicts the threshold pair, held out,
  at z = +0.4. So the windows, layers and pass types differ in how often they
  change the output, not in how much an output change costs.

What it does not explain, and why this is a hypothesis rather than a result:

- **Not tested directly.** I never perturbed a cache write while holding its
  token commits fixed, or the reverse. The cache-write reading is inferred
  from which arms matter.
- **OLMoE's higher cost per divergence isn't covered.** Given that its output
  diverged, OLMoE's answer changes 38.5% of the time, against 12.7% for
  LLaDA. Part of that is model quality: OLMoE is wrong on 44% of these
  questions and most of its churn is wrong → different wrong. The cache
  story doesn't account for it.
- **The late layers.** Layers 11–15 have a moderate flip rate but rarely
  change the output: layer 15 alone diverges the output on 32% of questions,
  against 94.5% for layer 0. *Why* deep-layer flips seldom reach the output
  is not established.
- **Which divergence definition fits matters.** Commit order versus output
  content: the hazard model fits only when divergence is defined by output
  content (χ² 12.7). With the first differing commit event it fits poorly
  (χ² 133), and finalize and the late-layer arms look unexplained (§5,
  "what didn't hold").

---

## 3. Common setup

| | |
|---|---|
| LLaDA-MoE | `inclusionAI/LLaDA-MoE-7B-A1B-Instruct`, this repo's engine (`dminfr.engine`), bf16, fused Triton MoE with an A100-tuned config (installed under every A100 device name; asserted loaded) |
| OLMoE | `allenai/OLMoE-1B-7B-0924-Instruct`, HF transformers 4.53.2, bf16, SDPA, greedy. Its `config.json` checked: 64 experts, top-8, `norm_topk_prob: false`, the same routing algorithm as LLaDA-MoE |
| Hardware | Modal A100-SXM4-40GB (one smoke run landed on an 80GB part; same SM80 chip, same tuned config) |
| Questions | GSM8K test, seed-42 `_stable_subset` order. n=200 (Phases 0–1) is the first 200 of n=1319; n=1319 is the full test set, ordered by the same sort-then-shuffle, whose first 1000 are verified equal to the n=1000 selection |
| Prompt | the repo harness's 4-shot GSM8K prompt and system prompt, through each model's own chat template. OLMoE's renders as `<|endoftext|><|system|>\n…\n<|user|>\n…\n<|assistant|>\n` |
| Generation (LLaDA) | `max_tokens=256 block_length=32 steps=256` (32 steps per 32-token block), fixed schedule unless noted, temperature 0, batch 32 |
| Generation (OLMoE) | `max_new_tokens=256`, batch 64 |
| Batches | fixed per question set (sorted by prompt length, then chunked) and identical across every arm paired against one control |
| Determinism | LLaDA control vs repeat: 200/200 token-identical. Across containers: 200/200 (Phase 0 vs Phase 1), 64/64 twice (Phase 2 sentinels). OLMoE: 200/200 same container, 64/64 across containers |
| Capped | LLaDA pads unused positions with EOS, so "emitted an EOS" is almost always true. The first-EOS position is bimodal: continuous up to ~244, empty at 245–254, a spike at exactly 255. **Capped = control's first EOS at position 255 or none.** 514/1319 (39%) for LLaDA's fixed schedule; OLMoE: 3/805 hit 256 tokens |
| Primary metric | **answer churn**: the grader's extracted answer differs between paired arms, on questions uncapped in that comparison's control(s). 95% Wilson CIs; exact McNemar on correctness discordants |
| Perturbations | fp32 router logits (`F.linear` in fp32 on the same bf16 hidden states) in a chosen window; or bf16 logits + U(−1, +1) bf16 ULP noise, seeded per (seed, batch). All behind `LLADA_RESEARCH_HOOKS=1` plus the research env flags, default off; production is bit-identical with the hooks unset (tested) |

---

## 4. SQ1: does masking cause the flat routing?

**Setup.** Telemetry-on control, n=200, measuring at every MoE call:
- the bf16 and fp32 gate on the same hidden states;
- entropy / log 64, top-1 weight, the 8th–9th logit gap in bf16 ULP, and the
  top-8 flip rate;
- per pass kind (prime, step, finalize), layer and position type.

Telemetry doesn't change outputs: 200/200 identical to Phase 0's control.

| pass | position type | entropy / log 64 | top-1 weight | median 8th–9th gap (ULP) | top-8 flip rate |
|---|---|---:|---:|---:|---:|
| prime | prompt | 0.974 | 4.46% | 0.89 | 23.5% |
| step | active block, unmasked | 0.958 | 6.08% | 0.99 | 22.9% |
| step | active block, masked | 0.981 | 3.77% | 0.85 | 24.8% |
| step | future blocks, masked | 0.985 | 3.40% | 0.72 | 27.8% |
| finalize | active block, unmasked | 0.960 | 5.75% | 0.98 | 22.8% |

**What held.**
- **Masked positions are flatter.** At the same block, step and layer they
  have higher entropy (+0.023) and lower top-1 weight (−2.4 points) in 92–93%
  of 3,968 comparisons. They sharpen as their block fills (a per-block
  sawtooth, [`research/plots/sq1_*.png`](../research/plots/)).
- **But not more fragile.** Their flip rate is only +0.9 points higher, and
  higher in just 50% of comparisons.
- **Layer, not masking, sets fragility.** The flip rate runs from 94–96% at
  layer 0 to 7–9% at layers 11–13. The layer spread is 17.7× the
  position-type spread.
- **Layer 0 settles it.** It is near-uniform for every position, prompt
  tokens included (entropy ≥ 0.9995).
- **Agreement with the repo:** the per-layer flip rates agree with
  `h100x2_bench.md` §9a (94.2% at layer 0; 24.1% overall, on a different
  prompt and GPU).

**Answer.** Masking makes routing measurably flatter but is not what makes it
fragile. Layer is. Layer 15 is the exception: unmasked positions there flip
38% of the time, against 9–12% for masked.

**What this doesn't settle.** Flatness by entropy and fragility by ULP margin
are different properties; see SQ3. Gaps below 0.25 ULP aren't resolved by the
10-bin histogram, and layer 0 sits at that floor.

---

## 5. SQ2: do early perturbations matter more than late ones?

**Setup.**
- n=1319, 14 perturbed arms plus one control, all paired. Primary population:
  805 uncapped.
- The windows separate the cache-writing passes (prime, finalize) from
  denoising steps, since a cache write persists into every later pass. The
  first design mixed the two.
- There are equal-size 25% windows (early vs last within each block; blocks
  0–1 vs 6–7), because churn saturates.
- Three noise seeds act as independent replicates.

| arm | answer churn [95% CI] | output content diverged | acc ctrl → arm | McNemar p |
|---|---:|---:|---:|---:|
| fp32, every pass | 11.9 [9.9, 14.3] | 94.5% | 75.7 → 76.6 | 0.38 |
| fp32, prime pass only | 12.3 [10.2, 14.7] | 93.7% | 75.7 → 75.9 | 0.88 |
| fp32, finalize passes only | 9.2 [7.4, 11.4] | 83.2% | 75.7 → 76.6 | 0.31 |
| fp32, steps 0–7 of each block | 6.8 [5.3, 8.8] | 51.3% | 75.7 → 75.5 | 0.92 |
| fp32, steps 8–31 of each block | 9.7 [7.8, 11.9] | 83.6% | 75.7 → 75.4 | 0.87 |
| fp32, steps 24–31 of each block | 4.2 [3.0, 5.8] | 48.7% | 75.7 → 76.7 | 0.12 |
| fp32, steps of blocks 0–1 | 10.3 [8.4, 12.6] | 70.8% | 75.7 → 75.8 | 0.94 |
| fp32, steps of blocks 6–7 | 0.2 [0.1, 0.9] | 8.3% | 75.7 → 76.3 | 0.057 |
| fp32, layer 0 only (every pass) | 12.9 [10.8, 15.4] | 94.5% | 75.7 → 76.6 | 0.38 |
| fp32, layers 11–14 only | 6.2 [4.7, 8.1] | 58.8% | 75.7 → 76.0 | 0.69 |
| fp32, layer 15 only | 3.0 [2.0, 4.4] | 32.2% | 75.7 → 76.0 | 0.72 |
| noise ±1 ULP, seeds 0 / 1 / 2 | 12.7 / 12.3 / 13.4 | 94–96% | — | 0.43–0.88 |

| equal-size contrast | only first | only second | McNemar p |
|---|---:|---:|---:|
| steps 0–7 vs steps 24–31 (each block) | 38 | 17 | 0.0065 |
| blocks 0–1 vs blocks 6–7 | 83 | 2 | 2e-22 |
| prime vs finalize | 54 | 29 | 0.008 |
| layer 0 vs layer 15 | 88 | 8 | 4e-18 |
| layers 11–14 vs layer 15 | 41 | 15 | 7e-4 |

**What held.**
- **Early matters more, for both definitions of "early".** Within a block,
  steps 0–7 beat steps 24–31. Across the generation, blocks 0–1 beat blocks
  6–7.
- **Late blocks rarely matter, even when the answer is written there.** On the
  251 questions whose answer line falls in blocks 6–7, perturbing those
  blocks' steps changes 2.8% of answers. fp32 everywhere changes 27.9% and
  blocks 0–1 change 19.9%, in strict-v1 terms (`followups_report.md`).
- **Causal layer test.** fp32 at layer 0 alone, the near-uniform layer with a
  95% flip rate, reproduces the full effect. Layer 15's high flip rate on
  unmasked positions is not decisive.
- **Fragility is a property of the question.**
  - Over the three independent noise seeds, 45 questions churn under all 3,
    against 1.7 expected if churn were independent of the question. Pairwise
    Jaccard is 0.42, against 0.07 expected.
  - A beta-binomial fit puts **10.5% [8.3, 12.6]** of uncapped questions at a
    flip probability above 0.5. Most questions are stable.
- **Noise and fp32 are the same family.** ±1 ULP noise is the larger
  perturbation (bf16 rounding moves a logit by ≤ 0.5 ULP) and churns about
  the same (12.3–13.4% vs 11.9%). Equal churn from unequal perturbations
  points to saturation, not equivalence.
- **Predicting fragility from the control alone:** questions with low mean
  commit confidence and long generations are more fragile (AUC 0.69 and 0.63
  under strict v1; `p2_report.md`). Confidence on the answer number's own
  tokens predicts nothing.

**What didn't hold, or changed during the study.**
- **Flip count doesn't predict churn.** A saturating curve in expected flips
  per question fits badly (χ² = 334 on 9 dof). Weighting cache-pass flips
  ~37× still doesn't fit (χ² = 90 on 8).
  - The divergence × hazard model with commit-event divergence also failed
    (χ² = 133 on 14 arms), leaving finalize (z = +4.4), layer 15 (z = −5.7)
    and layers 11–14 (z = −3.5) unexplained.
  - With output-content divergence, one hazard curve fits all arms
    (χ² = 12.7). Most "divergences" under the commit definition were the
    same tokens committed at a different step.
- **The strict-v1 magnitudes (~25%) were roughly double the real ones**
  (§7). Every qualitative contrast above survives under the grader's
  extraction, with p-values in the same range or smaller.

---

## 6. SQ3: is this a diffusion problem or an MoE problem?

**Setup.**
- OLMoE through HF, with the same perturbations applied by a forward hook on
  each layer's `mlp.gate`.
- Arms: control, fp32 everywhere, fp32 on the prefill only (the counterpart
  of LLaDA's prime pass), noise seed 0.
- The 805 questions uncapped in LLaDA's control, a cut forced by the Phase 3
  budget cap; 802 are uncapped in both models.
- Telemetry on the first 200 questions.

| on the same 200 prompts | LLaDA-MoE | OLMoE |
|---|---:|---:|
| entropy / log 64 | 0.974 | 0.913 |
| top-1 weight | 4.5% | 8.7% |
| median 8th–9th gap | 0.9 bf16 ULP | **22 bf16 ULP** |
| top-8 flip rate, bf16 vs fp32 | 23.5% (prompt) | **1.1%** (prefill) / 1.3% (decode) |

| same 802 questions | LLaDA-MoE | OLMoE |
|---|---:|---:|
| answer churn, fp32 everywhere / prompt pass / noise | 12.0 / 12.3 / 12.6% | 19.3 / 20.2 / 20.3% |
| P(output content diverges), fp32 everywhere | 94.5% | 50.2% |
| P(answer changes \| diverged) | 12.7% | 38.5% |
| one pass only: P(diverged), answer churn | step 0: 10.6%, 1.4% | first decode step: 39.9%, 17.0% |
| correctness flips (C→W / W→C / wrong → different wrong) | 34 / 35 / 27 | 28 / 25 / 102 |

**Answer.**
- **It's an MoE-and-decoding problem, not a diffusion one, and router
  flatness doesn't decide the outcome.** OLMoE's router is far sharper in the
  sense that matters (25× wider margin in ULP, 1% flips), yet its answers
  change *more* often than LLaDA's.
- **Opposite routes to churn.**
  - LLaDA's flat router makes output divergence near-certain, but a
    divergence rarely changes the answer.
  - OLMoE's sharp router leaves half its generations untouched, but an AR
    divergence is 3× more likely to change the answer, and mostly between
    wrong answers.
- **The raw churn difference reflects the decoding regime and each model's
  accuracy on these questions,** not AR vs diffusion routing.
- **Fragility belongs to each (model, question) pair.** The cross-model rank
  correlation of per-question churn is ρ = 0.18 [0.10, 0.25]. In both models,
  churn concentrates on questions the control gets wrong.

**Retracted during the study.**
- *"OLMoE diverged on 88–92% of questions, and churn given divergence was
  the same in both models (22–23% vs 24–25%)."* That was a bug. Padded batch
  outputs were compared beyond EOS, and HF `generate` pads a row differently
  depending on when its batch finished. Corrected: 50–55% divergence; the
  retraction is recorded in `research/data/p3_report.md`.
- *"~20–25% churn in both models."* That was strict-v1 inflation (§7).

---

## 7. SQ4: does threshold decoding make it better or worse?

**Setup.**
- Control and fp32 everywhere under hierarchical threshold decoding (0.9 /
  0.4), on n=1319 with the same batches.
- Compared with Phase 2's fixed-schedule pair, on the 684 questions uncapped
  under both schedules.

| | fixed schedule | threshold 0.9/0.4 |
|---|---:|---:|
| answer churn, fp32 pair (684) | 9.2% [7.3, 11.6] | **13.6%** [11.2, 16.4] |
| questions churned only under this schedule | 31 | 61 |
| paired McNemar on churn | — | **p = 0.0023** |
| accuracy, control → fp32 (1319) | 75.7 → 76.6 (p = 0.38) | 74.2 → 74.7 (p = 0.73) |
| denoising steps actually run | 100% | 62% |
| decoding collapses (a token repeated ≥ 12 times) | 0 / 1319 | 0 / 1319 |

**Answer.**
- **Slightly worse:** +4.4 points of answer churn, well past the noise
  (p = 0.002), while it runs 38% fewer steps.
- **Explained by divergence, not by step count.** The hazard model, fitted
  only on fixed-schedule arms, predicts the threshold pair's churn from its
  divergence pattern (z = +0.4). Threshold decoding's outputs diverge more
  often (97.8% vs 94.2%) and earlier.
- **Switching schedule is itself a perturbation of the same size.** Fixed
  control vs threshold control changes 10.4% of answers.
- **No decoding collapse in any arm.** This config runs 32 steps per block
  (ratio 1.0), the setting `h100x2_bench.md` §8c found removes the collapse.
  The §4d failure conditions (ratio 0.5, 1024 tokens, block 64) were not
  tested.

---

## 8. Corrections made during the study

Recorded the way `INVESTIGATION_LOG.md` §2.10 records its retraction of §2.9:
the earlier text stays in the phase reports, marked.

1. **Answer extraction (affects every churn magnitude).**
   - The strict extraction first used (v1) recognized only the repo's
     `Final Answer:` and `The answer is` markers. It read `The final answer
     is **375**` or `**Answer: 120**` as no answer, so rephrasing an answer
     line counted as changing it.
   - Of LLaDA's 198 v1 "changes" under fp32 everywhere, 101 were the same
     correct number written differently.
   - The primary metric is now the grader's extraction on uncapped questions.
     Its fallback only misbehaves on truncated outputs, which the primary
     population excludes.
   - A strict v3, built from a survey of every answer format in the ~31k
     outputs, agrees with the grader on 98.4–100% of rows in every arm. Churn
     differs by ≤ 1.0 point, except one noise seed at +1.2
     (`research/data/answer_agreement.md`).
   - v1 results are kept as a sensitivity table in
     `research/data/by_answer_definition/strict/`.
2. **OLMoE divergence: the padding bug.** See SQ3.
3. **Divergence definition for LLaDA.** "First differing commit event" counts
   the same token committed at a different step as divergence. The
   cross-model comparison and the hazard model now use output content.
   Results under both definitions are in
   `research/data/by_answer_definition/lenient/hazard_{commit,content}.md`.
4. **A noise-flip estimate that failed its own check.** A boundary model for
   noise-arm flip counts underpredicted measured rounding flips 1.6–2.5×,
   because near-uniform logits tie several experts at once. Noise arms were
   kept out of every fit.
5. **Process.**
   - Downloading results into the OneDrive-synced repo folder left null-byte
     holes in 2 of 17 files, while the volume copies were intact. Results are
     now downloaded to temp, validated, then copied.
   - A local network outage twice killed in-flight Modal runs. Runs now
     launch with `fn.spawn` under `--detach` and are billed from per-batch
     heartbeat ledgers (`research/ledger.py`).

**Relation to earlier repo conclusions.**
- `INVESTIGATION_LOG.md` §2.11 attributed the engine-vs-HF divergence to
  inherent numerical noise amplified by the near-uniform router. This study
  agrees that router flips are pervasive (24% from gate precision alone,
  §9a). It adds that such flips change answers on ~12% of questions without
  changing accuracy, and that whether they matter depends on where they land.
- §9e's conclusion that the fp32 router has no accuracy effect is reproduced:
  no arm survives Holm correction.

---

## 9. What this does not show

- **One task, one budget.** GSM8K only, with a 256-token budget (39% of LLaDA
  generations hit it, and they are excluded from the primary metric). The
  production config (`max_tokens=1024 block_length=64`) and other tasks were
  not run. Magnitudes may differ there; the §9e correctness discordance
  suggests similar.
- **One perturbation family.** fp32 vs bf16 gate and ±1 ULP noise. Kernel
  changes, quantization or cross-GPU differences perturb more than the router.
  The noise floor applies to them only as a lower bound.
- **Limited model comparison.** The cross-model claim rests on one AR model
  with its own chat template, on 802 questions selected by LLaDA's capping
  (easy for LLaDA, at 86.9%). A second AR MoE, or OLMoE on all 1319 questions,
  would test whether "AR divergences cost 3× more" generalizes.
- **The mechanism is inferred, not isolated** (§2). Why late-layer flips
  rarely reach the output is open.
- **Measurements on the control's trajectory.** Expected flips, flatness and
  margins describe the control's path, at n=200. An arm's own flips after it
  diverges are not measured.
- **One GPU type.** A100 only; ratios measured elsewhere in this repo have not
  always transferred across hardware.

**A pre-existing issue, not fixed here:** `tests/test_oom_retry.py` opens
`src/server.py`, a path that no longer exists since the `dminfr/` restructure.
It fails independently of this work. Every other test passes on an A100 with
the research hooks merged.

---

## 10. Reproducing

```bash
python research/fetch_data.py                    # raw JSONL from the GitHub Release, SHA256-verified
python research/verify_headlines.py              # 35 headline numbers from raw data, no shared code
RESEARCH_ANSWER=lenient python -m research.sq2 --dirs research/data/p2{A,B,C}/L256_b32_s256_fixed_B32 \
    --telemetry research/data/p1/L256_b32_s256_fixed_B32
python -m research.hazard ... --llada-div content
LLADA_RESEARCH_HOOKS=1 python -m research.tests.test_probe   # 17 CPU tests
python -m research.tests.test_olmoe                          # 7 CPU tests
```

| file | what it is |
|---|---|
| `research/metrics.py` | every metric: flatness, margins, flips, answer extractions, Wilson, McNemar |
| `research/probe.py` | telemetry, fp32 windows, noise, commit tracking (LLaDA) |
| `research/olmoe.py` | the same for OLMoE via HF hooks |
| `research/runner.py`, `research/modal_app.py` | paired-arm runner and Modal entry points |
| `research/sq1.py` … `sq4.py`, `sq2_extra.py`, `hazard.py`, `transitions.py`, `matched_dose.py`, `answer_agreement.py` | analyses |
| `research/data/*_report.md`, `followups_report.md`, `by_answer_definition/` | generated reports |
| `research/BUDGET.md` | every run's GPU-seconds × rate, and the running total |

## 11. Spend

**$20.78 of the $22 cap** (A100-40GB at $0.000645/s including CPU and memory;
each run billed with a 60 s boot allowance):

| phase | cost |
|---|---:|
| 0: setup, autotune, determinism, cheap-config accuracy | $1.20 |
| 1: telemetry (SQ1) | $0.19 |
| 2: 15 arms × 1319 (SQ2) | $11.66 (estimate $10.8) |
| 3: OLMoE (SQ3) | $3.45 |
| 4: threshold pair (SQ4) | $1.07 |
| matched dose (incl. two network-interrupted attempts) | $3.21 |
| extra CPU fetch (full-set selection) | $0.003 |

$1.22 left unspent by decision.
