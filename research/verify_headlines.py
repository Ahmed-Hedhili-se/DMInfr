"""Independent re-derivation of the routing-fragility headline numbers.

    python research/verify_headlines.py [--data research/data]

Deliberately self-contained: standard library only, and nothing imported from
research/ (no sq2, metrics, runner, ...). It reads the raw per-question JSONL
and recomputes every headline number from the stored fields:
  correct   the repo grader's verdict at run time
  answer    the repo grader's extracted answer (the primary answer definition)
  out       the full generated token row (LLaDA: 256 committed tokens; OLMoE:
            the padded generate() row)
  commit    LLaDA only: the global step that committed each position
  capped    OLMoE only: hit max_new_tokens without EOS
It prints verified vs reported and exits non-zero on any mismatch.
"""

import argparse
import json
import math
import os
import sys

LLADA_EOS, OLMOE_EOS, OLMOE_PAD = 156892, 50279, 1

# Values as reported in research/data/*_report.md and followups_report.md.
# Percentages to 0.1 point, p-values to 2 significant figures.
REPORTED = {
    "LLaDA control accuracy, fixed, n=1319": 75.7,
    "LLaDA control accuracy, threshold, n=1319": 74.2,
    "OLMoE control accuracy, n=805": 55.8,
    "LLaDA capped in control (of 1319)": 514,
    "OLMoE capped in control (of 805)": 3,
    "LLaDA fp32_all churn, grader, uncapped (805)": 11.9,
    "LLaDA fp32_all churn, grader, both-uncapped (802)": 12.0,
    "OLMoE fp32_all churn, grader, both-uncapped (802)": 19.3,
    "LLaDA fp32_all correctness flips (802)": 69,
    "OLMoE fp32_all correctness flips (802)": 53,
    "LLaDA noise_s0 correctness flips (802)": 75,
    "OLMoE noise_s0 correctness flips (802)": 65,
    "LLaDA blocks 0-1 churn (805)": 10.3,
    "LLaDA blocks 6-7 churn (805)": 0.2,
    "blocks 0-1 vs 6-7 McNemar p": 1.9e-22,
    "LLaDA layer0-only churn (805)": 12.9,
    "LLaDA prime-only churn (805)": 12.3,
    "fixed pair churn (684 both-schedule uncapped)": 9.2,
    "threshold pair churn (684)": 13.6,
    "threshold vs fixed McNemar p": 0.0023,
    "matched dose LLaDA b0s0 P(diverged), commit def (802)": 17.2,
    "matched dose LLaDA b0s16 P(diverged), commit def (802)": 9.0,
    "matched dose LLaDA b0s0 P(diverged), content def (802)": 10.6,
    "matched dose LLaDA b0s16 P(diverged), content def (802)": 2.7,
    "matched dose OLMoE dec0 P(diverged) (802)": 39.9,
    "matched dose OLMoE dec16 P(diverged) (802)": 32.0,
    "matched dose LLaDA b0s0 churn (802)": 1.4,
    "matched dose LLaDA b0s16 churn (802)": 0.5,
    "matched dose OLMoE dec0 churn (802)": 17.0,
    "matched dose OLMoE dec16 churn (802)": 10.8,
    "noise floor: fp32_all correctness discordance d, n=1319": 13.9,
    "noise floor SD at n=200 (pt)": 2.63,
    "noise floor SD at n=500 (pt)": 1.67,
    "noise floor SD at n=1000 (pt)": 1.18,
    "noise floor SD at n=1319 (pt)": 1.03,
}


def load(path):
    rows = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            rows[r["idx"]] = r
    return rows


def llada_capped(r):
    o = r["out"]
    return LLADA_EOS not in o or o.index(LLADA_EOS) >= len(o) - 1


def llada_div_commit(a, b):
    """Any commit event differs (token, or the step it was committed at)."""
    return any(x != y or c != d for x, c, y, d in zip(a["out"], a["commit"], b["out"], b["commit"]))


def llada_div_content(a, b):
    def cut(o):
        return o[: o.index(LLADA_EOS) + 1] if LLADA_EOS in o else o
    return cut(a["out"]) != cut(b["out"])


def olmoe_div(a, b):
    def cut(o):
        k = o.index(OLMOE_EOS) + 1 if OLMOE_EOS in o else len(o)
        return [t for t in o[:k] if t != OLMOE_PAD]
    return cut(a["out"]) != cut(b["out"])


def mcnemar(b, c):
    n, k = b + c, min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0


def pct(k, n):
    return round(100 * k / n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="research/data")
    D = ap.parse_args().data
    L = lambda arm, d="p2A": load(os.path.join(D, d, "L256_b32_s256_fixed_B32", f"{arm}.jsonl"))
    O = lambda arm: load(os.path.join(D, "p3", "olmoe_L256_greedy_B64", f"{arm}.jsonl"))
    T = lambda arm: load(os.path.join(D, "p4", "L256_b32_s256_thr0.9-0.4_B32", f"{arm}.jsonl"))

    lc, lf = L("control"), L("fp32_all")
    oc, of = O("control"), O("fp32_all")
    tc, tf = T("control"), T("fp32_all")
    all_ids = sorted(lc)
    l_unc = [i for i in all_ids if not llada_capped(lc[i])]
    both = [i for i in sorted(oc) if not oc[i]["capped"] and not llada_capped(lc[i])]
    sched = [i for i in l_unc if not llada_capped(tc[i])]

    churn = lambda c, a, ids: sum(c[i]["answer"] != a[i]["answer"] for i in ids)
    flips = lambda c, a, ids: sum(c[i]["correct"] != a[i]["correct"] for i in ids)
    V = {}
    V["LLaDA control accuracy, fixed, n=1319"] = pct(sum(lc[i]["correct"] for i in all_ids), len(all_ids))
    V["LLaDA control accuracy, threshold, n=1319"] = pct(sum(tc[i]["correct"] for i in tc), len(tc))
    V["OLMoE control accuracy, n=805"] = pct(sum(oc[i]["correct"] for i in oc), len(oc))
    V["LLaDA capped in control (of 1319)"] = len(all_ids) - len(l_unc)
    V["OLMoE capped in control (of 805)"] = sum(oc[i]["capped"] for i in oc)
    V["LLaDA fp32_all churn, grader, uncapped (805)"] = pct(churn(lc, lf, l_unc), len(l_unc))
    V["LLaDA fp32_all churn, grader, both-uncapped (802)"] = pct(churn(lc, lf, both), len(both))
    V["OLMoE fp32_all churn, grader, both-uncapped (802)"] = pct(churn(oc, of, both), len(both))
    V["LLaDA fp32_all correctness flips (802)"] = flips(lc, lf, both)
    V["OLMoE fp32_all correctness flips (802)"] = flips(oc, of, both)
    ln, on = L("noise_s0", "p2C"), O("noise_s0")
    V["LLaDA noise_s0 correctness flips (802)"] = flips(lc, ln, both)
    V["OLMoE noise_s0 correctness flips (802)"] = flips(oc, on, both)

    g01, g67 = L("fp32_steps_global_early", "p2B"), L("fp32_steps_global_last25", "p2B")
    s01 = {i for i in l_unc if lc[i]["answer"] != g01[i]["answer"]}
    s67 = {i for i in l_unc if lc[i]["answer"] != g67[i]["answer"]}
    V["LLaDA blocks 0-1 churn (805)"] = pct(len(s01), len(l_unc))
    V["LLaDA blocks 6-7 churn (805)"] = pct(len(s67), len(l_unc))
    V["blocks 0-1 vs 6-7 McNemar p"] = float(f"{mcnemar(len(s01 - s67), len(s67 - s01)):.2g}")
    V["LLaDA layer0-only churn (805)"] = pct(churn(lc, L("fp32_layer0"), l_unc), len(l_unc))
    V["LLaDA prime-only churn (805)"] = pct(churn(lc, L("fp32_prime"), l_unc), len(l_unc))

    sf = {i for i in sched if lc[i]["answer"] != lf[i]["answer"]}
    st = {i for i in sched if tc[i]["answer"] != tf[i]["answer"]}
    V["fixed pair churn (684 both-schedule uncapped)"] = pct(len(sf), len(sched))
    V["threshold pair churn (684)"] = pct(len(st), len(sched))
    V["threshold vs fixed McNemar p"] = float(f"{mcnemar(len(sf - st), len(st - sf)):.2g}")

    for tag, arm in (("b0s0", "fp32_single_b0s0"), ("b0s16", "fp32_single_b0s16")):
        a = L(arm)
        V[f"matched dose LLaDA {tag} P(diverged), commit def (802)"] = pct(sum(llada_div_commit(lc[i], a[i]) for i in both), len(both))
        V[f"matched dose LLaDA {tag} P(diverged), content def (802)"] = pct(sum(llada_div_content(lc[i], a[i]) for i in both), len(both))
        V[f"matched dose LLaDA {tag} churn (802)"] = pct(churn(lc, a, both), len(both))
    for tag in ("dec0", "dec16"):
        a = O(f"fp32_{tag}")
        V[f"matched dose OLMoE {tag} P(diverged) (802)"] = pct(sum(olmoe_div(oc[i], a[i]) for i in both), len(both))
        V[f"matched dose OLMoE {tag} churn (802)"] = pct(churn(oc, a, both), len(both))

    d = flips(lc, lf, all_ids) / len(all_ids)
    V["noise floor: fp32_all correctness discordance d, n=1319"] = round(100 * d, 1)
    for n in (200, 500, 1000, 1319):
        V[f"noise floor SD at n={n} (pt)"] = round(100 * math.sqrt(d / n), 2)

    bad = 0
    print(f"{'quantity':<62} {'reported':>10} {'verified':>10}  ok")
    for k, rep in REPORTED.items():
        v = V[k]
        ok = (abs(v - rep) <= max(0.05 * abs(rep), 1e-30) if isinstance(rep, float) and rep < 1e-2
              else abs(v - rep) < 0.051 if isinstance(rep, float) else v == rep)
        bad += not ok
        print(f"{k:<62} {rep:>10} {v:>10}  {'ok' if ok else 'MISMATCH'}")
    print(f"\n{len(REPORTED) - bad}/{len(REPORTED)} match.")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
