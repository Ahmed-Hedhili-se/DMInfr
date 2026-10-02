"""SQ4: does hierarchical threshold decoding (0.9 / 0.4) change routing fragility?

    python -m research.sq4 --fixed <p2A dir> --threshold <p4 dir>

Two paired arms per schedule on the same 1319 questions and batches:
control (bf16 router) and fp32_all. The fixed-schedule pair is Phase 2's.
Capped is decided per schedule from that schedule's control (first EOS at
the last position); results are given on each schedule's own uncapped set
and on the questions uncapped under both.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import Counter

import numpy as np

from research.metrics import (COLLAPSE_RUN, first_divergence, mcnemar_exact, paired_compare,
                              wilson_ci)
from research.runner import _read_jsonl
from research.sq2 import is_capped, pct


def load(d):
    arms = {os.path.basename(p)[:-6]: _read_jsonl(p) for p in sorted(glob.glob(os.path.join(d, "*.jsonl")))}
    counts = {}
    for p in glob.glob(os.path.join(d, "*.counts.seg*.json")):
        name = os.path.basename(p).split(".counts")[0]
        c = Counter(json.load(open(p))["counts"])
        counts[name] = counts.get(name, Counter()) + c
    return arms, counts


def churn_set(ctrl, arm, ids, key="answer_strict"):
    return {i for i in ids if ctrl[i][key] != arm[i][key]}


def report(fx, fx_counts, th, th_counts):
    out = []
    P = out.append
    fc, ff = fx["control"], fx["fp32_all"]
    tc, tf = th["control"], th["fp32_all"]
    ids = sorted(set(fc) & set(tc))
    unc_f = [i for i in ids if not is_capped(fc[i])]
    unc_t = [i for i in ids if not is_capped(tc[i])]
    both = sorted(set(unc_f) & set(unc_t))

    P("## SQ4: threshold decoding (0.9 / 0.4) vs the fixed schedule\n")
    P("Setup: n=1319 (full GSM8K test set, seed-42 order), A100-40GB, `max_tokens=256 block_length=32 "
      "steps=256` (32 steps per block = block length, ratio 1.0), B=32, identical batches in all four arms. "
      "Fixed pair from Phase 2 (container A); threshold pair from Phase 4. Threshold = "
      "`select_transfer_indices_hierarchy`, confidence 0.9, low 0.4.\n")
    fp = fx_counts.get("control", {})
    tp = th_counts.get("control", {})
    if fp and tp:
        P(f"Denoising steps actually run (control, summed over 1319 questions in batches): fixed "
          f"{fp['step_passes']:,} step passes, threshold {tp['step_passes']:,} "
          f"({pct(tp['step_passes'] / fp['step_passes'])}% of fixed).\n")
    P(f"Capped in control: fixed {len(ids) - len(unc_f)}/{len(ids)} ({pct(1 - len(unc_f) / len(ids))}%), "
      f"threshold {len(ids) - len(unc_t)}/{len(ids)} ({pct(1 - len(unc_t) / len(ids))}%). Uncapped under both: "
      f"{len(both)}.\n")

    P("### Churn of each pair (control vs fp32_all)\n")
    P("| population | fixed: strict churn [95% CI] | threshold: strict churn [95% CI] | churn only fixed "
      "| churn only threshold | McNemar p (fixed vs threshold churn) |")
    P("|---|---:|---:|---:|---:|---:|")
    for name, pop in (("uncapped under both (primary)", both), ("all 1319", ids)):
        a = churn_set(fc, ff, pop)
        b = churn_set(tc, tf, pop)
        ca, cb = wilson_ci(len(a), len(pop)), wilson_ci(len(b), len(pop))
        P(f"| {name} (n={len(pop)}) | {pct(len(a) / len(pop))} [{pct(ca[0])}, {pct(ca[1])}] "
          f"| {pct(len(b) / len(pop))} [{pct(cb[0])}, {pct(cb[1])}] | {len(a - b)} | {len(b - a)} "
          f"| {mcnemar_exact(len(a - b), len(b - a)):.3g} |")
    for name, ctrl, arm, pop in (("fixed, own uncapped", fc, ff, unc_f), ("threshold, own uncapped", tc, tf, unc_t)):
        s = churn_set(ctrl, arm, pop)
        c = wilson_ci(len(s), len(pop))
        P(f"\n- {name} (n={len(pop)}): {pct(len(s) / len(pop))}% [{pct(c[0])}, {pct(c[1])}]")
    P("")

    P("### Accuracy (repo grader, all 1319; paired McNemar)\n")
    P("| comparison | acc A | acc B | only A ✓ | only B ✓ | McNemar p |")
    P("|---|---:|---:|---:|---:|---:|")
    for name, a, b in (("fixed: control vs fp32_all", fc, ff), ("threshold: control vs fp32_all", tc, tf),
                       ("control: fixed vs threshold", fc, tc), ("fp32_all: fixed vs threshold", ff, tf)):
        c = paired_compare(a, b, ids, "answer")
        P(f"| {name} | {pct(c['acc_a'])} | {pct(c['acc_b'])} | {c['only_a']} | {c['only_b']} | {c['mcnemar_p']:.3g} |")
    sc = paired_compare(fc, tc, both, "answer_strict")
    P(f"\nThe schedule switch itself, as a perturbation (control fixed vs control threshold, strict, "
      f"uncapped under both): {pct(sc['outcome_churn'])}% [{pct(sc['outcome_churn_ci'][0])}, "
      f"{pct(sc['outcome_churn_ci'][1])}] of answers change.\n")

    P("### Trajectory divergence within each pair\n")
    P("| pair | diverged | median first-divergence step [IQR] |")
    P("|---|---:|---:|")
    for name, a, b in (("fixed", fc, ff), ("threshold", tc, tf)):
        d = [first_divergence(a[i]["out"], a[i]["commit"], b[i]["out"], b[i]["commit"]) for i in both]
        dv = [x for x in d if x is not None]
        q = np.percentile(dv, [25, 50, 75])
        P(f"| {name} | {len(dv)}/{len(both)} | {q[1]:.0f} [{q[0]:.0f}–{q[2]:.0f}] |")
    P("\nSteps are global (block × 32 + step); under threshold decoding a block can finish early, so a step "
      "index there is an upper bound on elapsed forward passes.\n")

    P(f"### Decoding collapse (h100x2_bench.md §4d/§8c): one token repeated ≥ {COLLAPSE_RUN} times in a row\n")
    P("| arm | collapses / 1319 | longest repeat run, max | questions with run ≥ 8 |")
    P("|---|---:|---:|---:|")
    for name, a in (("fixed control", fc), ("fixed fp32_all", ff), ("threshold control", tc), ("threshold fp32_all", tf)):
        runs = [a[i]["max_run"] for i in ids]
        P(f"| {name} | {sum(a[i]['collapse'] for i in ids)} | {max(runs)} | {sum(r >= 8 for r in runs)} |")
    P("\nThis config runs 32 steps per block (ratio 1.0), the setting §8c found removes the collapse; the "
      "§4d failures were at ratio 0.5, a 1024-token budget and block length 64, which this study does not "
      "test.\n")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixed", required=True)
    ap.add_argument("--threshold", required=True)
    a = ap.parse_args()
    fx, fxc = load(a.fixed)
    th, thc = load(a.threshold)
    print(report(fx, fxc, th, thc))


if __name__ == "__main__":
    main()
