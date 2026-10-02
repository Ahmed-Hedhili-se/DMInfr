"""Matched-dose experiment: fp32 router on exactly one forward pass.

LLaDA: one denoising step (block 0 step 0, block 0 step 16).
OLMoE: one decode forward (the one routing generated token 0, token 16).
Each arm is paired against its model's existing control on the questions
uncapped in both controls.

    python -m research.matched_dose --llada <p2A dir> --olmoe <p3 dir>
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np

from research.metrics import first_divergence, gsm8k_strict_answer_v2, wilson_ci
from research.runner import _read_jsonl
from research.sq2 import is_capped, pct
from research.sq3 import ar_divergence

DEFS = {
    "strict v1": lambda r: r["answer_strict_v1"],
    "strict v2": lambda r: gsm8k_strict_answer_v2(r["text"]),
    "lenient (grader)": lambda r: r["answer"],
}


def ci(k, n):
    lo, hi = wilson_ci(k, n)
    return f"{pct(k / n)}% [{pct(lo)}, {pct(hi)}]"


def report(l, o):
    out = []
    P = out.append
    lc, oc = l["control"], o["control"]
    ids = [i for i in sorted(oc) if not oc[i]["capped"] and not is_capped(lc[i])]
    P("## Matched dose: fp32 router on exactly one forward pass\n")
    P(f"Questions: the {len(ids)} uncapped in both models' controls. Controls reused (LLaDA: Phase 2 "
      "container A, same 1319-question batches; OLMoE: Phase 3, same 805-question batches). "
      "'Injected flips' = top-8 sets that differ between bf16 and fp32 gates on the same hidden states, "
      "summed over all layers and routed positions of that one pass, counted in the arm itself.\n")
    if "control_sentinel" in o:
        s = o["control_sentinel"]
        same = sum(s[i]["out"] == oc[i]["out"] for i in s)
        P(f"OLMoE cross-container check: control re-run in the matched-dose container, first batch: "
          f"{same}/{len(s)} token-identical to Phase 3's control.\n")
    P("| model | perturbed pass | injected flips / question: mean (median) | questions with ≥ 1 injected flip "
      "| P(diverged) [95% CI] | P(diverged \\| ≥ 1 flip) | P(diverged \\| 0 flips) | median first divergence |")
    P("|---|---|---:|---:|---:|---:|---:|---:|")
    rows = [("LLaDA-MoE", "block 0, step 0", l, lc, "fp32_single_b0s0", "llada"),
            ("LLaDA-MoE", "block 0, step 16", l, lc, "fp32_single_b0s16", "llada"),
            ("OLMoE", "decode forward routing token 0", o, oc, "fp32_dec0", "olmoe"),
            ("OLMoE", "decode forward routing token 16", o, oc, "fp32_dec16", "olmoe")]
    churn_rows = []
    for model, desc, arms, ctrl, arm, kind in rows:
        if arm not in arms:
            continue
        a = arms[arm]
        inj = np.array([a[i]["injected_flips"] or 0.0 for i in ids])
        if kind == "llada":
            d = [first_divergence(ctrl[i]["out"], ctrl[i]["commit"], a[i]["out"], a[i]["commit"]) for i in ids]
        else:
            d = [ar_divergence(ctrl[i]["out"], a[i]["out"]) for i in ids]
        dv = np.array([x is not None for x in d])
        pos = inj > 0
        med = np.median([x for x in d if x is not None]) if dv.any() else float("nan")
        unit = "step" if kind == "llada" else "token"
        P(f"| {model} | {desc} | {inj.mean():,.1f} ({np.median(inj):,.0f}) | {pos.sum()}/{len(ids)} "
          f"| {ci(int(dv.sum()), len(ids))} | {ci(int((dv & pos).sum()), int(pos.sum())) if pos.any() else '—'} "
          f"| {ci(int((dv & ~pos).sum()), int((~pos).sum())) if (~pos).any() else '—'} | {unit} {med:.0f} |")
        churn_rows.append((model, desc, ctrl, a, dv))
    P("\nchurn, per answer definition (all three shown; see the note on strict v1 in the reports):\n")
    P("| model | perturbed pass | " + " | ".join(f"churn, {k}" for k in DEFS) + " | churn \\| diverged (lenient) |")
    P("|---|---|" + "---:|" * (len(DEFS) + 1))
    for model, desc, ctrl, a, dv in churn_rows:
        cells = []
        for k, f in DEFS.items():
            c = sum(f(ctrl[i]) != f(a[i]) for i in ids)
            cells.append(ci(c, len(ids)))
        cd = sum(DEFS["lenient (grader)"](ctrl[i]) != DEFS["lenient (grader)"](a[i]) for i, d_ in zip(ids, dv) if d_)
        P(f"| {model} | {desc} | " + " | ".join(cells) + f" | {ci(cd, int(dv.sum())) if dv.any() else '—'} |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llada", required=True)
    ap.add_argument("--olmoe", required=True)
    a = ap.parse_args()
    load = lambda d: {os.path.basename(p)[:-6]: _read_jsonl(p) for p in glob.glob(os.path.join(d, "*.jsonl"))}
    print(report(load(a.llada), load(a.olmoe)))


if __name__ == "__main__":
    main()
