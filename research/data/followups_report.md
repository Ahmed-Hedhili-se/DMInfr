# Follow-ups before the write-up: answer definition, hazard, transitions, matched dose

Setup unless stated otherwise:
- LLaDA-MoE: n=1319 (seed-42 `_stable_subset` order), A100-40GB,
  `max_tokens=256 block_length=32 steps=256`, fixed schedule, B=32.
- OLMoE: the 805 questions uncapped in LLaDA's control, greedy,
  `max_new_tokens=256`, B=64.
- Full outputs under each answer definition are in
  `by_answer_definition/{strict,strict_v2,lenient}/`. The definition is
  selected with `RESEARCH_ANSWER`; see `research/runner._read_jsonl`.

## 1. A flaw in the primary metric (must be decided before the write-up)

The strict extraction (v1, `metrics.gsm8k_strict_answer`) only recognizes
the repo's two answer markers, `Final Answer:` and `The answer is`. LLaDA-MoE
often writes `The final answer is **375**` or `**Answer: 120**`. v1 reads
those as no answer, so a run that rephrases its answer line counts as a
changed answer.

Of the 198 v1 "answer changes" under fp32_all, **101 have both outputs graded
correct with the same number**: pure rephrasing. OLMoE almost always writes
`The answer is`, so it is barely affected (1–4 such cases).

| definition | what counts as the answer | LLaDA fp32_all, 805 uncapped | LLaDA noise_s0 | OLMoE fp32_all, 802 |
|---|---|---:|---:|---:|
| strict v1 (as run) | number on a `Final Answer:` / `The answer is` line, else None | 24.6% | 25.1% | 19.6% |
| strict v2 | same, also `final answer is`, `Answer:`; else None | 17.3% | 17.8% | 19.3% |
| lenient (repo grader) | grader's own extraction (answer line, else `\boxed{}`, else last number) | **11.9%** | **12.7%** | **19.3%** |

- On uncapped questions, v2 and the grader agree on every control row that
  has an answer line (0 disagreements). The residual v2-vs-lenient gap is
  answer line present in one output and absent in the other, with the same
  correct number.
- The grader's fallback was only a problem for capped (truncated)
  generations, which the primary population already excludes.
- **Recommendation: the grader's extraction on control-uncapped questions as
  the primary metric.**

What survives the change of definition, and what doesn't:

| claim | v1 | grader's answer |
|---|---|---|
| layer 0 alone ≈ fp32 everywhere; prime ≈ all | yes | yes (12.9 / 12.3 / 11.9%) |
| early steps > late steps, within block | 13.3 vs 8.9%, p = 0.0017 | 6.8 vs 4.2%, p = 0.0065 |
| blocks 0–1 ≫ blocks 6–7 | 18.5 vs 1.6% | 10.3 vs 0.2%, p = 2e-22 |
| layers: 0 > 11–14 > 15 | yes | yes (p ≤ 7e-4 for each step) |
| threshold decoding worse (684 uncapped in both) | 21.8 → 26.2%, p = 0.034 | **9.2 → 13.6%, p = 0.0023** |
| fragility concentrated in a fixed set of questions | 3-of-3 seeds: 83 vs 12 expected | 45 vs 1.7 expected; p > 0.5 for 10.5% [8.3, 12.6] of questions |
| no accuracy effect in any arm (Holm) | yes | yes (correctness does not depend on the definition) |
| **LLaDA churns more than OLMoE** | 24.7 vs 19.6% | **reversed: 12.0 vs 19.3%** |
| "~20–25% churn in both models" | roughly | **no: LLaDA ~12%, OLMoE ~20%** |

## 2. OLMoE capped and subset bias

- 3 of OLMoE's 805 control generations hit 256 tokens without EOS. The
  population uncapped in both controls is 802 questions.
- All OLMoE vs LLaDA comparisons are on those 802 questions:
  - grader's answer: OLMoE fp32_all / prefill / noise churn 19.3 / 20.2 /
    20.3%, against LLaDA fp32_all / prime / noise 12.0 / 12.3 / 12.6%;
  - v1 numbers are in `by_answer_definition/strict/sq3.md`.
- The 802 were selected by being uncapped in LLaDA's control. That is an
  easy subset for LLaDA (86.9% accuracy, against 75.7% on all 1319 questions)
  but not for OLMoE (55.8%).

## 3. Divergence × hazard decomposition

Churn = Σ over diverged questions of h(when the first divergence happened).
A non-diverged output is identical to control and cannot churn. h is
estimated leaving out the arm being predicted.

**LLaDA (grader's answer).** χ² = 133 over 14 arms with block bins, 103 with
finer bins. That is better than flip count alone (334 on 9 dof) and worse
than the cache-weighted flip model (90 on 8).
- Explained by timing: fp32_all, prime, global_early and within_late all
  have |z| < 2.
- Not explained:
  - finalize sits above the prediction (z = +4.4). It diverges at step
    32–45 but changes answers like an early perturbation, because its effect
    persists through the KV cache.
  - layer 15 (z = −5.7) and layers 11–14 (z = −3.5) sit below. They diverge
    as early as layer 0 but change answers far less, so how large the
    divergence is matters, not only when it starts.
  - blocks 6–7 (z = −4.3; the hazard is estimated from very few
    block-6 divergences in the other arms).
- Out of sample: threshold decoding churns more than its divergence timing
  predicts (z = +2.7). The fixed-schedule pair (z = −1.1) and the schedule
  switch (z = −0.2) are predicted well.

**OLMoE.** The 3 arms are consistent with one hazard (χ² = 2.4).

| | LLaDA-MoE | OLMoE |
|---|---:|---:|
| P(output diverges) under fp32 everywhere | 100% | **50.3%** |
| P(answer changes \| diverged), grader's answer | 12.0% | **38.5%** |
| hazard, first divergence in the first 10% of control's length | 22.7% (v1) | 52.3% (v1) |
| hazard, first divergence after control's EOS | 0.3% | 0% |

The flat router makes LLaDA's divergence certain, but most LLaDA divergences
are harmless. OLMoE's sharp router leaves half its generations untouched,
but an AR divergence is three times as likely to change the answer.
(Plots: `by_answer_definition/*/hazard.png`.)

*Correction to the Phase 3 report:* OLMoE "diverged 88–92%" was a bug.
Padded batch outputs were compared beyond EOS. It is now fixed and retracted
in `p3_report.md`.

## 4. Churn as transitions (grader's answer, same 802 questions)

| arm | changed | C→W | W→C | wrong → different wrong | correctness flips (C→W + W→C) |
|---|---:|---:|---:|---:|---:|
| LLaDA fp32_all | 96 (12.0%) | 34 | 35 | 27 | **69 (8.6%)** |
| LLaDA noise_s0 | 101 (12.6%) | 43 | 32 | 26 | 75 (9.4%) |
| OLMoE fp32_all | 155 (19.3%) | 28 | 25 | **102** | **53 (6.6%)** |
| OLMoE noise_s0 | 163 (20.3%) | 28 | 37 | 98 | 65 (8.1%) |

- In both models C→W and W→C balance, which is why accuracy never moves.
- Two-thirds of OLMoE's churn is one wrong answer turning into another; it
  is wrong on 44% of these questions.
- **Counting only correctness flips, LLaDA flips as much as or more than
  OLMoE.** Which model "churns more" depends on the metric.

## 5. Matched dose: fp32 router on exactly one forward pass (802 questions)

| perturbed pass | flips injected / question | P(diverged) | P(div \| ≥ 1 flip) | P(div \| 0 flips) | churn, grader's answer | churn \| diverged |
|---|---:|---:|---:|---:|---:|---:|
| LLaDA, block 0 step 0 | 1,140 | 17.2% [14.8, 20.0] | 17.2% | — | 1.4% [0.8, 2.4] | 8.0% |
| LLaDA, block 0 step 16 | 1,119 | 9.0% [7.2, 11.2] | 9.0% | — | 0.5% [0.2, 1.3] | 5.6% |
| OLMoE, decode step for token 0 | 0.2 | **39.9%** [36.6, 43.3] | 34.4% | 41.2% | **17.0%** [14.5, 19.7] | 42.5% |
| OLMoE, decode step for token 16 | 0.2 | **32.0%** [28.9, 35.4] | 36.0% | 31.2% | **10.8%** [8.9, 13.2] | 33.9% |

**Answer:** No. With one perturbed pass at the matched position, OLMoE
diverges 2.3–3.6× more often than LLaDA (39.9 vs 17.2%, 32.0 vs 9.0%), and
its answers change 12–22× more often (17.0 vs 1.4%, 10.8 vs 0.5%), while
~5,000× fewer expert flips are injected.

- **Flips are not the effective dose.** OLMoE diverges as often when the
  pass flipped *no* top-8 set (41%) as when it flipped one (34%). An fp32
  gate also shifts the routing *weights* when the same 8 experts win, and in
  an AR decode step that change is written into the KV cache that every
  later token reads.
- **A LLaDA denoising step writes no cache.** It matters only through the
  token(s) it commits, here one per row in block 0. That is the same
  mechanism as the Phase 2 cache-write result: prime and finalize, which do
  write the cache, matter far more than their flip counts suggest.
- Reuse checks: OLMoE's control, re-run in the matched-dose container,
  matched Phase 3's control on 64/64 questions. The LLaDA control is Phase
  2's, with identical batches; the cross-container check was already done in
  Phase 2 (64/64 twice).

**Spend.** $3.20, against a $3.06 estimate. The local client lost its network
twice. The first time, both ephemeral apps stopped mid-run. The second
time, `--detach` didn't save them, because the entrypoint was blocked in a
synchronous `.remote()` call. The runs were finished with `fn.spawn` under
`--detach` and billed from their heartbeat ledgers (`research/ledger.py`).
Study total: **$20.78 of $22**.
