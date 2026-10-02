"""Grader extraction vs strict v3, per arm, on control-uncapped questions.

    python -m research.answer_agreement

For every arm: how often the two extractions return the same answer (on the
arm's own rows and on its control's), and the churn each gives against that
control. Flags any arm whose churn differs by more than 1 point.
"""

from __future__ import annotations

import glob
import json
import os

from research.metrics import gsm8k_strict_answer_v3

EOS = 156892
D = "research/data"
LLADA = {"p2A": "p2A/L256_b32_s256_fixed_B32", "p2B": "p2B/L256_b32_s256_fixed_B32",
         "p2C": "p2C/L256_b32_s256_fixed_B32"}


def rows(path):
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            out[r["idx"]] = r
    return out


def llada_capped(r):
    o = r["out"]
    return EOS not in o or o.index(EOS) >= len(o) - 1


def compare(ctrl, arm, ids):
    v3 = lambda r: gsm8k_strict_answer_v3(r["text"])
    agree_arm = sum(v3(arm[i]) == arm[i]["answer"] for i in ids) / len(ids)
    churn_g = sum(ctrl[i]["answer"] != arm[i]["answer"] for i in ids) / len(ids)
    churn_v3 = sum(v3(ctrl[i]) != v3(arm[i]) for i in ids) / len(ids)
    none_v3 = sum(v3(arm[i]) is None for i in ids)
    return agree_arm, churn_g, churn_v3, none_v3


def main():
    out = ["## Grader extraction vs strict v3 (control-uncapped questions)\n",
           "agree = fraction of the arm's rows where strict v3 returns the grader's answer. "
           "Flag: |churn(grader) − churn(v3)| > 1 point.\n",
           "| model | arm | n | agree (arm rows) | v3 = None | churn, grader | churn, strict v3 | Δ | flag |",
           "|---|---|---:|---:|---:|---:|---:|---:|---|"]
    lc = rows(os.path.join(D, LLADA["p2A"], "control.jsonl"))
    unc = [i for i in sorted(lc) if not llada_capped(lc[i])]
    v3 = lambda r: gsm8k_strict_answer_v3(r["text"])
    a0 = sum(v3(lc[i]) == lc[i]["answer"] for i in unc) / len(unc)
    out.append(f"| LLaDA | control | {len(unc)} | {100 * a0:.1f}% | {sum(v3(lc[i]) is None for i in unc)} | — | — | — | |")
    flags = 0
    arms = []
    for tag, sub in LLADA.items():
        for p in sorted(glob.glob(os.path.join(D, sub, "*.jsonl"))):
            name = os.path.basename(p)[:-6]
            if name.startswith("control"):
                continue
            arms.append(("LLaDA", name, lc, rows(p), unc))
    tc = rows(os.path.join(D, "p4/L256_b32_s256_thr0.9-0.4_B32/control.jsonl"))
    tf = rows(os.path.join(D, "p4/L256_b32_s256_thr0.9-0.4_B32/fp32_all.jsonl"))
    t_unc = [i for i in sorted(tc) if not llada_capped(tc[i])]
    arms.append(("LLaDA threshold", "fp32_all", tc, tf, t_unc))
    oc = rows(os.path.join(D, "p3/olmoe_L256_greedy_B64/control.jsonl"))
    o_unc = [i for i in sorted(oc) if not oc[i]["capped"]]
    a0 = sum(v3(oc[i]) == oc[i]["answer"] for i in o_unc) / len(o_unc)
    for p in sorted(glob.glob(os.path.join(D, "p3/olmoe_L256_greedy_B64/*.jsonl"))):
        name = os.path.basename(p)[:-6]
        if not name.startswith("control"):
            arms.append(("OLMoE", name, oc, rows(p), o_unc))
    for model, name, ctrl, arm, ids in arms:
        ag, cg, cv, nn = compare(ctrl, arm, ids)
        d = cv - cg
        flag = abs(d) > 0.01
        flags += flag
        out.append(f"| {model} | {name} | {len(ids)} | {100 * ag:.1f}% | {nn} | {100 * cg:.1f}% | {100 * cv:.1f}% "
                   f"| {100 * d:+.1f} | {'**> 1 pt**' if flag else ''} |")
    out.insert(4 + 1, f"| OLMoE | control | {len(o_unc)} | {100 * a0:.1f}% | {sum(v3(oc[i]) is None for i in o_unc)} | — | — | — | |")
    out.append(f"\n{flags} arm(s) flagged.")
    print("\n".join(out))


if __name__ == "__main__":
    main()
