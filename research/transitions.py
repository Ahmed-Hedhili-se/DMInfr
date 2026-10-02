"""Decompose answer churn into transitions, for both models.

    python -m research.transitions --llada <p2A> <p2B> <p2C> --threshold <p4 dir> --olmoe <p3 dir>

Per arm vs its control, over a stated population: correct -> wrong,
wrong -> correct, wrong -> a different wrong answer (strict answers differ,
both graded wrong), and answer changed while both graded correct (possible
only through the grader's lenient fallback). Correctness is the repo grader;
'changed' is the strict extraction, as everywhere else in this study.
"""

from __future__ import annotations

import argparse
import glob
import os

from research.runner import _read_jsonl
from research.sq2 import ARMS, is_capped, load as load_llada, pct


def transitions(ctrl, arm, ids):
    t = {"C→W": 0, "W→C": 0, "W→W′": 0, "C→C′": 0}
    for i in ids:
        if ctrl[i]["answer_strict"] == arm[i]["answer_strict"]:
            continue
        a, b = ctrl[i]["correct"], arm[i]["correct"]
        t["C→W" if a and not b else "W→C" if b and not a else "W→W′" if not a else "C→C′"] += 1
    return t


def table(rows, P):
    P("| model / arm | population | churned | C→W | W→C | W→W′ | C→C′ | net accuracy change |")
    P("|---|---:|---:|---:|---:|---:|---:|---:|")
    for name, ctrl, arm, ids in rows:
        t = transitions(ctrl, arm, ids)
        n = sum(t.values())
        net = t["W→C"] - t["C→W"]
        P(f"| {name} | {len(ids)} | {n} ({pct(n / len(ids))}%) | {t['C→W']} | {t['W→C']} | {t['W→W′']} "
          f"({pct(t['W→W′'] / n) if n else '—'}% of churn) | {t['C→C′']} | {net:+d} ({100 * net / len(ids):+.1f} pt) |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llada", nargs="+", required=True)
    ap.add_argument("--threshold", required=True)
    ap.add_argument("--olmoe", required=True)
    a = ap.parse_args()
    l, _ = load_llada(a.llada)
    th = {os.path.basename(p)[:-6]: _read_jsonl(p) for p in glob.glob(os.path.join(a.threshold, "*.jsonl"))}
    o = {os.path.basename(p)[:-6]: _read_jsonl(p) for p in glob.glob(os.path.join(a.olmoe, "*.jsonl"))}
    lc, oc, tc = l["control"], o["control"], th["control"]
    l_unc = [i for i in sorted(lc) if not is_capped(lc[i])]
    both = [i for i in sorted(oc) if not oc[i]["capped"] and i in set(l_unc)]
    both_sched = [i for i in l_unc if not is_capped(tc[i])]
    out = []
    P = out.append
    P("## Churn decomposed into transitions\n")
    P("C = graded correct by the repo grader, W = wrong; W→W′ = the strict answer changed and both are "
      "wrong. Populations: LLaDA's own uncapped set (805); the 802 uncapped in both models' controls for "
      "the cross-model rows; the 684 uncapped under both schedules for the threshold pair.\n")
    rows = [(f"LLaDA {arm}", lc, l[arm], l_unc) for arm in ARMS if arm in l]
    rows += [("LLaDA threshold: control vs fp32_all", tc, th["fp32_all"], both_sched),
             ("LLaDA fixed: control vs fp32_all (same 684)", lc, l["fp32_all"], both_sched)]
    table(rows, P)
    P("\n### Same 802 questions, both models\n")
    table([(f"LLaDA {x}", lc, l[x], both) for x in ("fp32_all", "fp32_prime", "noise_s0")] +
          [(f"OLMoE {x}", oc, o[x], both) for x in ("fp32_all", "fp32_prefill", "noise_s0")], P)
    P("\nC→W and W→C are near-balanced in every arm (hence no accuracy effect); the excess of churn over "
      "|C→W − W→C| is answers moving between wrong values.")
    print("\n".join(out))


if __name__ == "__main__":
    main()
