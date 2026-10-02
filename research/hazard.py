"""Divergence x hazard decomposition of churn.

A non-diverged trajectory reproduces control exactly, so it cannot churn.
Churn therefore factors as
    churn(arm) = sum over diverged questions of P(answer changes | first divergence at s)
              ~= (1/N) * sum_i 1[diverged_i] * h(bin(s_i)),
with one hazard curve h shared by every arm. The test: estimate h from all
arms but one, predict the held-out arm's churn from its own first-divergence
distribution, and score with z = (observed - predicted) / sqrt(sum h(1-h)).

LLaDA bins s by block (global step // 32). OLMoE bins the first divergent
token position by 32 tokens. Both are also binned by first-divergence
position as a fraction of control's generated length, for an overlay.

    python -m research.hazard --llada <p2A> <p2B> <p2C> --threshold <p4 dir> \
        --olmoe <p3 dir> --plot research/plots/hazard.png
"""

from __future__ import annotations

import argparse
import glob
import math
import os

import numpy as np

from research.metrics import wilson_ci
from research.runner import _read_jsonl
from research.sq2 import ARMS, is_capped, load as load_llada, pct

#: LLaDA divergence definition: "commit" = first differing commit event
#: (token or commit step), binned by global step; "content" = first position
#: where the final committed tokens differ (up to EOS), binned by position --
#: the definition comparable with OLMoE's first differing token.
LLADA_DIV = "commit"
LLADA_EOS = 156892

FRAC_EDGES = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0, math.inf]


def llada_div(ctrl_row, arm_row):
    """(first divergence step, position of that divergence) or (None, None)."""
    best, pos = None, None
    for p, (ta, ca, tb, cb) in enumerate(zip(ctrl_row["out"], ctrl_row["commit"], arm_row["out"], arm_row["commit"])):
        if ta != tb or ca != cb:
            s = min(ca, cb)
            if best is None or s < best:
                best, pos = s, p
    return best, pos


def llada_content_div(ctrl_row, arm_row):
    """First position where the final generated tokens differ (content up to
    the first EOS; commit order ignored), or (None, None)."""
    def content(o):
        return o[: o.index(LLADA_EOS) + 1] if LLADA_EOS in o else o
    a, b = content(ctrl_row["out"]), content(arm_row["out"])
    for k, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return k, k
    return (None, None) if len(a) == len(b) else (min(len(a), len(b)),) * 2


def olmoe_div(ctrl_row, arm_row):
    from research.sq3 import ar_divergence
    k = ar_divergence(ctrl_row["out"], arm_row["out"])
    return k, k


def step_bin(s):
    """Finer LLaDA bins: 8-step bins over blocks 0-1, then whole blocks."""
    return s // 8 if s < 64 else 8 + (s // 32 - 2)


STEP_LABELS = [f"steps {8 * j}-{8 * j + 7}" for j in range(8)] + [f"block {b}" for b in range(2, 8)]


def records(ctrl, arm, ids, kind):
    """Per question: (diverged, bin, frac_bin, churned)."""
    out = []
    for i in ids:
        if kind == "llada":
            s, pos = (llada_div if LLADA_DIV == "commit" else llada_content_div)(ctrl[i], arm[i])
            b = None if s is None else s // 32
        else:
            s, pos = olmoe_div(ctrl[i], arm[i])
            b = None if s is None else s // 32
        n = max(1, ctrl[i]["n_tokens"])
        fb = None if pos is None else int(np.searchsorted(FRAC_EDGES, pos / n, side="right") - 1)
        fine = None if s is None else (step_bin(s) if kind == "llada" else b)
        out.append((s is not None, b, fb, ctrl[i]["answer_strict"] != arm[i]["answer_strict"], fine))
    return out


def hazard(recs, key, nbins):
    k = np.zeros(nbins)
    n = np.zeros(nbins)
    for r in recs:
        if r[0]:
            b = {"block": r[1], "frac": r[2], "fine": r[4]}[key]
            n[b] += 1
            k[b] += r[3]
    return k, n


def predict(recs, h, key):
    mu = var = 0.0
    for r in recs:
        if r[0]:
            p = h[{"block": r[1], "frac": r[2], "fine": r[4]}[key]]
            mu += p
            var += p * (1 - p)
    return mu, var


def loo_table(per_arm, nbins, key, P, label):
    """Leave-one-arm-out prediction of every arm's churn count."""
    P(f"| {label} arm | n | diverged | churn \\| diverged | observed churn | predicted (hazard from the other arms) | z |")
    P("|---|---:|---:|---:|---:|---:|---:|")
    chi = 0.0
    zs = {}
    for a, recs in per_arm.items():
        k = np.zeros(nbins)
        n = np.zeros(nbins)
        for b, rr in per_arm.items():
            if b != a:
                kk, nn = hazard(rr, key, nbins)
                k += kk
                n += nn
        h = (k + 0.5) / (n + 1.0)            # Jeffreys prior: an empty bin cannot predict exactly 0
        mu, var = predict(recs, h, key)
        obs = sum(r[3] for r in recs)
        dv = sum(r[0] for r in recs)
        z = (obs - mu) / math.sqrt(var) if var > 0 else float("nan")
        zs[a] = z
        chi += z * z if z == z else 0
        N = len(recs)
        cd = obs / dv if dv else float("nan")
        P(f"| {a} | {N} | {pct(dv / N)}% | {pct(cd)}% | {pct(obs / N)}% | {pct(mu / N)}% | "
          f"{'**' if abs(z) > 2 else ''}{z:+.1f}{'**' if abs(z) > 2 else ''} |")
    return chi, zs


def report(l_arms, th, o_arms, plot):
    out = []
    P = out.append
    lc = l_arms["control"]
    l_unc = [i for i in sorted(lc) if not is_capped(lc[i])]
    present = [a for a in ARMS if a in l_arms]
    per = {a: records(lc, l_arms[a], l_unc, "llada") for a in present}

    P("## Divergence × hazard decomposition\n")
    P(f"LLaDA divergence definition: **{LLADA_DIV}**: " + (
        "first differing commit event (a token, or the step it was committed at); bins are global "
        "denoising steps (block = step // 32)." if LLADA_DIV == "commit" else
        "first position where the final generated tokens differ, commit order ignored (comparable with "
        "OLMoE's first differing token); bins are output positions (block = position // 32).") + "\n")
    P("A trajectory that never diverges reproduces control token for token, so it cannot churn; churn "
      "= Σ over diverged questions of h(when it diverged). One hazard curve h, estimated from the other "
      "arms (leave-one-arm-out), predicts each arm's churn from that arm's own first-divergence "
      "distribution. z = (observed − predicted) / √Σh(1−h). Arms share questions, so the z-scores are "
      "not independent; χ² is reported as a summary, not a calibrated test.\n")

    P(f"### LLaDA-MoE, 14 arms, hazard binned by block of first divergence (n={len(l_unc)} uncapped)\n")
    nb = 8
    kk = np.zeros(nb)
    nn = np.zeros(nb)
    for r in per.values():
        k, n = hazard(r, "block", nb)
        kk += k
        nn += n
    P("Pooled hazard h(block) = P(strict answer changes | first divergence in that block):\n")
    P("| block | diverging there (pooled over arms) | h [95% CI] |")
    P("|---:|---:|---:|")
    for b in range(nb):
        lo, hi = wilson_ci(int(kk[b]), int(nn[b])) if nn[b] else (float("nan"),) * 2
        P(f"| {b} | {int(nn[b])} | {pct(kk[b] / nn[b]) if nn[b] else '—'} [{pct(lo)}, {pct(hi)}] |")
    P("")
    chi, zs = loo_table(per, nb, "block", P, "LLaDA")
    P(f"\nχ² = {chi:.1f} over {len(per)} arms; compare the flip-count fits in sq2_extra run under the same RESEARCH_ANSWER definition.\n")
    P("Sensitivity: the same leave-one-arm-out test with finer bins (8-step bins over blocks 0-1, whole "
      "blocks after), since block 0 holds most first divergences:\n")
    chi_f, _ = loo_table(per, len(STEP_LABELS), "fine", P, "LLaDA (fine bins)")
    P(f"\nχ² = {chi_f:.1f} over {len(per)} arms with fine bins.\n")

    # threshold out of sample
    P("### Out of sample: the Phase 4 threshold pair, predicted from the 14 fixed-schedule arms\n")
    tc, tf = th["control"], th["fp32_all"]
    both = [i for i in l_unc if not is_capped(tc[i])]
    h_all = (kk + 0.5) / (nn + 1.0)
    P("| pair | n (uncapped in both schedules) | diverged | observed churn | predicted | z |")
    P("|---|---:|---:|---:|---:|---:|")
    for name, c_, a_ in (("fixed control vs fp32_all", lc, l_arms["fp32_all"]),
                         ("threshold control vs fp32_all", tc, tf),
                         ("schedule switch: fixed control vs threshold control", lc, tc)):
        recs = records(c_, a_, both, "llada")
        mu, var = predict(recs, h_all, "block")
        obs = sum(r[3] for r in recs)
        z = (obs - mu) / math.sqrt(var)
        P(f"| {name} | {len(recs)} | {pct(sum(r[0] for r in recs) / len(recs))}% | {pct(obs / len(recs))}% "
          f"| {pct(mu / len(recs))}% | {z:+.1f} |")
    P("\nThe fixed pair is in-sample for h (its arm contributes to the pooled hazard); the threshold pair "
      "and the schedule switch are not. For the schedule switch the 'first divergence' compares commit "
      "steps across two schedules, which is only meaningful as the first committed-token difference.\n")

    # OLMoE
    oc = o_arms["control"]
    o_ids = [i for i in sorted(oc) if not oc[i]["capped"] and i in set(l_unc)]
    o_present = [a for a in ("fp32_all", "fp32_prefill", "noise_s0") if a in o_arms]
    o_per = {a: records(oc, o_arms[a], o_ids, "olmoe") for a in o_present}
    P(f"### OLMoE, 3 arms, hazard binned by first divergent token (32-token bins; n={len(o_ids)})\n")
    okk = np.zeros(nb)
    onn = np.zeros(nb)
    for r in o_per.values():
        k, n = hazard(r, "block", nb)
        okk += k
        onn += n
    P("| tokens | diverging there | h [95% CI] |")
    P("|---|---:|---:|")
    for b in range(nb):
        lo, hi = wilson_ci(int(okk[b]), int(onn[b])) if onn[b] else (float("nan"),) * 2
        P(f"| {32 * b}–{32 * b + 31} | {int(onn[b])} | {pct(okk[b] / onn[b]) if onn[b] else '—'} [{pct(lo)}, {pct(hi)}] |")
    P("")
    ochi, _ = loo_table(o_per, nb, "block", P, "OLMoE")
    P(f"\nχ² = {ochi:.1f} over {len(o_per)} arms.\n")

    # normalized overlay
    P("### Hazard by first-divergence position as a fraction of control's generated length\n")
    P("| fraction of control's length | LLaDA diverging | LLaDA h | OLMoE diverging | OLMoE h |")
    P("|---|---:|---:|---:|---:|")
    lf_k, lf_n = np.zeros(len(FRAC_EDGES) - 1), np.zeros(len(FRAC_EDGES) - 1)
    of_k, of_n = np.zeros_like(lf_k), np.zeros_like(lf_k)
    for r in per.values():
        k, n = hazard(r, "frac", len(lf_k))
        lf_k += k
        lf_n += n
    for r in o_per.values():
        k, n = hazard(r, "frac", len(of_k))
        of_k += k
        of_n += n
    labels = [f"{FRAC_EDGES[j]:.1f}–{FRAC_EDGES[j + 1]:.1f}" if FRAC_EDGES[j + 1] < math.inf else "≥ 1.0 (after control's EOS)"
              for j in range(len(lf_k))]
    for j, lab in enumerate(labels):
        P(f"| {lab} | {int(lf_n[j])} | {pct(lf_k[j] / lf_n[j]) if lf_n[j] else '—'}% | {int(of_n[j])} "
          f"| {pct(of_k[j] / of_n[j]) if of_n[j] else '—'}% |")
    P("\nLLaDA's position is the generated-sequence position of its first differing commit; OLMoE's is the "
      "first differing token. LLaDA can diverge first at a position *beyond* control's EOS (in the "
      "EOS-padding region) because it commits positions out of order.\n")

    P("### Cross-model: P(output diverges) and P(answer changes | diverged), same questions\n")
    P(f"Population: the {len(o_ids)} questions uncapped in both controls. LLaDA divergence: {LLADA_DIV}.\n")
    P("| LLaDA arm | P(div) | P(change \\| div) | OLMoE arm | P(div) | P(change \\| div) |")
    P("|---|---:|---:|---|---:|---:|")
    for la_, oa_ in (("fp32_all", "fp32_all"), ("fp32_prime", "fp32_prefill"), ("noise_s0", "noise_s0"),
                     ("fp32_single_b0s0", "fp32_dec0"), ("fp32_single_b0s16", "fp32_dec16")):
        cells = []
        for arms_, ctrl_, arm_, kind_ in ((l_arms, lc, la_, "llada"), (o_arms, oc, oa_, "olmoe")):
            if arm_ not in arms_:
                cells += [arm_, "—", "—"]
                continue
            rr = records(ctrl_, arms_[arm_], o_ids, kind_)
            dv = sum(r[0] for r in rr)
            ch = sum(r[3] for r in rr if r[0])
            cells += [arm_, f"{pct(dv / len(rr))}%", f"{pct(ch / dv) if dv else '—'}%"]
        P("| " + " | ".join(cells) + " |")
    P("")

    if plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        os.makedirs(os.path.dirname(plot), exist_ok=True)
        SURF, INK, MUTED = "#fcfcfb", "#1a1a19", "#6f6e69"
        fig, ax = plt.subplots(figsize=(8.5, 4.8), facecolor=SURF)
        ax.set_facecolor(SURF)
        xs = np.arange(len(lf_k))
        for k_, n_, col, mk, lab, dx in ((lf_k, lf_n, "#2a78d6", "o", "LLaDA-MoE (14 arms pooled)", -0.08),
                                          (of_k, of_n, "#eb6834", "s", "OLMoE (3 arms pooled)", 0.08)):
            ok = n_ >= 10
            h = np.where(ok, k_ / np.maximum(n_, 1), np.nan)
            ci = np.array([wilson_ci(int(a), int(b)) if b >= 10 else (np.nan, np.nan) for a, b in zip(k_, n_)])
            ax.errorbar(xs[ok] + dx, 100 * h[ok], yerr=[100 * (h[ok] - ci[ok, 0]), 100 * (ci[ok, 1] - h[ok])],
                        fmt=mk + "-", color=col, ms=6, lw=1.6, elinewidth=1, capsize=0, label=lab)
        ax.set_xticks(xs)
        ax.set_xticklabels([l.replace(" (after control's EOS)", "") for l in labels], rotation=35, fontsize=8, color=MUTED)
        ax.set_xlabel("first divergence, as a fraction of control's generated length (bins with ≥ 10 questions)", color=MUTED)
        ax.set_ylabel("P(strict answer changes | diverged there) (%)", color=MUTED)
        ax.set_title("Churn hazard by where the trajectory first diverges", color=INK, fontsize=11)
        ax.grid(color="#e4e3dc", lw=0.6)
        ax.legend(frameon=False, fontsize=8)
        for s in ax.spines.values():
            s.set_color("#d0cfc7")
        fig.tight_layout()
        fig.savefig(plot, dpi=140, facecolor=SURF)
        P(f"Plot: `{plot}`\n")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llada", nargs="+", required=True)
    ap.add_argument("--threshold", required=True)
    ap.add_argument("--olmoe", required=True)
    ap.add_argument("--plot", default=None)
    ap.add_argument("--llada-div", choices=("commit", "content"), default="commit")
    a = ap.parse_args()
    global LLADA_DIV
    LLADA_DIV = a.llada_div
    l_arms, _ = load_llada(a.llada)
    th = {os.path.basename(p)[:-6]: _read_jsonl(p) for p in glob.glob(os.path.join(a.threshold, "*.jsonl"))}
    o_arms = {os.path.basename(p)[:-6]: _read_jsonl(p) for p in glob.glob(os.path.join(a.olmoe, "*.jsonl"))}
    print(report(l_arms, th, o_arms, a.plot))


if __name__ == "__main__":
    main()
