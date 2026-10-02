"""Phase 2 follow-up analyses (no GPU): dose-response, noise-replicate fragility,
the blocks 6-7 baseline, and the accuracy noise floor with Holm correction.

    python -m research.sq2_extra --dirs <p2A> <p2B> <p2C> --telemetry <p1 dir> \
        --tokenizer <dir> --plot research/plots/sq2_dose_response.png

Appends nothing by itself; prints markdown to paste into p2_report.md.
"""

from __future__ import annotations

import argparse
import math
import re
from collections import Counter

import numpy as np

from research.metrics import MARGIN_EDGES_ULP, mcnemar_exact, paired_compare, wilson_ci
from research.sq2 import ARMS, churned, is_capped, load, pct

N_TEL = 200          # questions behind the Phase 1 telemetry


# ── expected flipped top-8 decisions per question ───────────────────────────

def _cells(tel, arm):
    """Boolean selection over [blocks+1, steps+1, L] for an arm's window."""
    import torch
    acc = tel["acc"]
    S = tel["steps_per_block"]
    sel = torch.zeros(acc.shape[:3], dtype=torch.bool)
    if arm in ("fp32_all",) or arm.startswith("noise"):
        sel[:] = True
    elif arm == "fp32_prime":
        sel[0, 0] = True
    elif arm == "fp32_finalize":
        sel[1:, S] = True
    elif arm.startswith("fp32_steps_within"):
        lo, hi = {"early": (0, 8), "late": (8, 32), "last25": (24, 32)}[arm.rsplit("_", 1)[1]]
        sel[1:, lo:hi] = True
    elif arm.startswith("fp32_steps_global"):
        lo, hi = {"early": (0, 2), "last25": (6, 8)}[arm.rsplit("_", 1)[1]]
        sel[1 + lo:1 + hi, :S] = True
    elif arm.startswith("fp32_layer"):
        layers = {"fp32_layer0": [0], "fp32_layers11_14": [11, 12, 13, 14], "fp32_layer15": [15]}[arm]
        sel[:, :, layers] = True
    return sel


def _p_flip(m_lo, m_hi, half_width):
    """Mean over margin m ~ U[m_lo, m_hi) (ULP) of P(e9 - e8 > m), with e8, e9
    i.i.d. U(-w, w): the difference is triangular on [-2w, 2w], so
    P(d > m) = (2w - m)^2 / (8 w^2) for 0 <= m <= 2w. Closed-form average."""
    a = 2 * half_width
    lo, hi = min(m_lo, a), min(m_hi, a)
    if hi <= lo:
        return 0.0
    integral = ((a - lo) ** 3 - (a - hi) ** 3) / (3 * 8 * half_width ** 2)
    return integral / (m_hi - m_lo)


def flip_split(tel, arm):
    """Expected flips per question in the arm's window, split into the
    cache-writing passes (prime + finalize, whose K/V persist) and steps."""
    acc, cols = tel["acc"], tel["stat_cols"]
    S = tel["steps_per_block"]
    sel = _cells(tel, arm)
    flips = acc[..., cols.index("flip_top8")].sum(-1)            # [blocks+1, steps+1, L]
    cache = sel.clone()
    cache[1:, :S] = False
    step = sel & ~cache
    return (flips * cache).sum().item() / N_TEL, (flips * step).sum().item() / N_TEL


def expected_flips(tel, arm):
    """(expected flips per question, how obtained). fp32 arms: measured
    bf16-vs-fp32 flips in the window. Noise arms: measured rounding flips
    rescaled per (cell, layer, type) by the boundary model's
    P_noise / P_round, computed from the margin histogram."""
    import torch
    acc, cols = tel["acc"], tel["stat_cols"]
    sel = _cells(tel, arm).unsqueeze(-1)                       # [..., L, 1] over types
    flips = acc[..., cols.index("flip_top8")]
    if not arm.startswith("noise"):
        return (flips * sel).sum().item() / N_TEL, "measured"
    h0 = cols.index("hist_0")
    edges = list(MARGIN_EDGES_ULP[:-1]) + [64.0]
    hist = acc[..., h0:h0 + len(edges) - 1]
    pr = torch.tensor([_p_flip(edges[i], edges[i + 1], 0.5) for i in range(len(edges) - 1)], dtype=torch.float64)
    pn = torch.tensor([_p_flip(edges[i], edges[i + 1], 1.0) for i in range(len(edges) - 1)], dtype=torch.float64)
    pred_round = (hist * pr).sum(-1)
    pred_noise = (hist * pn).sum(-1)
    ratio = torch.where(pred_round > 0, pred_noise / pred_round.clamp(min=1e-12), torch.zeros_like(pred_round))
    est = torch.minimum(flips * ratio, acc[..., cols.index("count")])   # a flip rate cannot exceed 1
    return (est * sel).sum().item() / N_TEL, "model"


def model_calibration(tel):
    """Boundary model vs measured bf16-vs-fp32 flip rate, per layer."""
    import torch
    acc, cols = tel["acc"], tel["stat_cols"]
    h0 = cols.index("hist_0")
    edges = list(MARGIN_EDGES_ULP[:-1]) + [64.0]
    hist = acc[..., h0:h0 + len(edges) - 1].sum((0, 1, 3))           # [L, bins]
    cnt = acc[..., cols.index("count")].sum((0, 1, 3))
    flips = acc[..., cols.index("flip_top8")].sum((0, 1, 3))
    pr = torch.tensor([_p_flip(edges[i], edges[i + 1], 0.5) for i in range(len(edges) - 1)], dtype=torch.float64)
    pn = torch.tensor([_p_flip(edges[i], edges[i + 1], 1.0) for i in range(len(edges) - 1)], dtype=torch.float64)
    return flips / cnt, (hist * pr).sum(-1) / cnt, (hist * pn).sum(-1) / cnt


def sat_curve(x, cmax, k):
    return cmax * (1 - np.exp(-k * x))


def model_b(X, cmax, k, logw):
    """Saturating curve in step flips + w * cache-writing-pass flips."""
    c, st = X
    return sat_curve(st + np.exp(logw) * c, cmax, k)


def fit_dose(xs, ys, ses, cx, sx):
    """Model A (flips alone) and model B (cache-write flips weighted), both
    bounded with several starts: the unbounded model-A fit degenerates to a
    flat line because churn is not monotone in flips."""
    from scipy.optimize import curve_fit
    best = None
    for k0 in (1e-6, 1e-5, 1e-4, 1e-3):
        try:
            po, pc = curve_fit(sat_curve, xs, ys, p0=[0.25, k0], sigma=ses, absolute_sigma=True,
                               bounds=([0, 1e-9], [1, 1]), maxfev=50000)
        except RuntimeError:
            continue
        chi = np.sum(((ys - sat_curve(xs, *po)) / ses) ** 2)
        if best is None or chi < best[2]:
            best = (po, pc, chi)
    bestb = None
    for k0 in (1e-6, 1e-5, 1e-4):
        for w0 in (0.0, 2.0, 4.0):
            try:
                pb, pcb = curve_fit(model_b, (cx, sx), ys, p0=[0.25, k0, w0], sigma=ses, absolute_sigma=True,
                                    bounds=([0, 1e-9, -5], [1, 1, 10]), maxfev=50000)
            except RuntimeError:
                continue
            chi = np.sum(((ys - model_b((cx, sx), *pb)) / ses) ** 2)
            if bestb is None or chi < bestb[2]:
                bestb = (pb, pcb, chi)
    return (*best, *bestb)


def dose_points(arms, tel):
    """Per Phase 2 arm: expected flips, churn (primary), SE, source, and the
    cache/step split. Returns the arrays the fits and plots use."""
    ctrl = arms["control"]
    unc = [i for i in sorted(ctrl) if not is_capped(ctrl[i])]
    names, xs, ys, ses, how = [], [], [], [], []
    for a in [a for a in ARMS if a in arms]:
        x, h = expected_flips(tel, a)
        c = len(churned(ctrl, arms[a], unc)) / len(unc)
        names.append(a); xs.append(x); ys.append(c); how.append(h)
        ses.append(math.sqrt(max(c * (1 - c), 1e-4) / len(unc)))
    xs, ys, ses = map(np.array, (xs, ys, ses))
    mask = np.array([h == "measured" for h in how])
    split = {n: flip_split(tel, n) for n, m in zip(names, mask) if m}
    cx = np.array([split[n][0] for n, m in zip(names, mask) if m])
    sx = np.array([split[n][1] for n, m in zip(names, mask) if m])
    return names, xs, ys, ses, how, mask, cx, sx


# ── beta-binomial over the noise seeds ──────────────────────────────────────

def betabinom_fit(counts, n):
    from scipy.optimize import minimize
    from scipy.special import betaln, gammaln
    c = np.asarray(counts)

    def nll(theta):
        a, b = np.exp(theta)
        return -np.sum(gammaln(n + 1) - gammaln(c + 1) - gammaln(n - c + 1)
                       + betaln(c + a, n - c + b) - betaln(a, b))
    r = minimize(nll, x0=np.log([0.5, 1.5]), method="Nelder-Mead", options={"xatol": 1e-8, "fatol": 1e-10, "maxiter": 5000})
    a, b = np.exp(r.x)
    return a, b, r.fun


def report(arms, tel, tok, plot_path):
    from scipy.optimize import curve_fit
    from scipy.special import betainc
    ctrl = arms["control"]
    ids = sorted(ctrl)
    unc = [i for i in ids if not is_capped(ctrl[i])]
    present = [a for a in ARMS if a in arms]
    out = []
    P = out.append

    # 1. dose-response
    P("## Dose-response: churn vs expected flipped top-8 decisions\n")
    meas, rnd, noi = model_calibration(tel)
    P("Expected flips per question = flipped top-8 decisions (bf16 vs fp32, same hidden states) summed "
      "over the arm's window — every pass type, layer and position type it covers — from the Phase 1 "
      "telemetry (control trajectory, n=200, same config), divided by 200. fp32 arms: measured. Noise "
      "arms: no direct measurement exists, so the measured rounding flips are rescaled per cell by a "
      "boundary model (flip iff the error difference at the 8th/9th logits exceeds their gap; rounding "
      "errors U(±0.5 ULP), noise U(±1 ULP)), using the measured gap histogram.\n")
    P("Model check on rounding, where flips were measured (all passes):\n")
    P("| layer | measured flip rate | model, rounding | model, ±1 ULP noise |")
    P("|---:|---:|---:|---:|")
    for l in range(len(meas)):
        P(f"| {l} | {pct(meas[l])}% | {pct(rnd[l])}% | {pct(noi[l])}% |")
    P("\nThe model counts only the 8th/9th boundary from a coarse 10-bin histogram, so it is a rough "
      "estimate: noise-arm x positions are model-based, not measured.\n")

    xs, ys, ses, names, how = [], [], [], [], []
    for a in present:
        x, h = expected_flips(tel, a)
        c = len(churned(ctrl, arms[a], unc)) / len(unc)
        xs.append(x); ys.append(c); names.append(a); how.append(h)
        ses.append(math.sqrt(max(c * (1 - c), 1e-4) / len(unc)))
    xs, ys, ses = map(np.array, (xs, ys, ses))
    fit_mask = np.array([h == "measured" for h in how])
    split = {names[j]: flip_split(tel, names[j]) for j in range(len(names)) if how[j] == "measured"}
    fit_names = [n for n, m in zip(names, fit_mask) if m]
    cx = np.array([split[n][0] for n in fit_names])
    sx = np.array([split[n][1] for n in fit_names])
    popt, pcov, chiA, pb, pcb, chiB = fit_dose(xs[fit_mask], ys[fit_mask], ses[fit_mask], cx, sx)
    perr = np.sqrt(np.diag(pcov))
    dof_a, dof_b = int(fit_mask.sum()) - 2, int(fit_mask.sum()) - 3
    P(f"**Model A, flip count alone** (11 fp32 arms, weighted by binomial SE): churn = "
      f"c_max·(1 − exp(−k·flips)); c_max = {pct(popt[0])}% ± {pct(perr[0])}, k = {popt[1]:.3g} ± "
      f"{perr[1]:.2g}; χ² = {chiA:.0f} on {dof_a} dof.\n")
    P(f"**Model B, cache-write flips weighted separately**: x = step flips + w·(prime + finalize "
      f"flips); c_max = {pct(pb[0])}%, k = {pb[1]:.3g}, w = {np.exp(pb[2]):.1f}; χ² = {chiB:.0f} on "
      f"{dof_b} dof.\n")
    P("| arm | expected flips / question | of which in cache-writing passes | source | churn (primary) "
      "| model A | z (A) | model B | z (B) |")
    P("|---|---:|---:|---|---:|---:|---:|---:|---:|")
    order = np.argsort(xs)
    for j in order:
        fa = sat_curve(xs[j], *popt)
        za = (ys[j] - fa) / ses[j]
        za_s = f"**{za:+.1f}**" if abs(za) > 2 else f"{za:+.1f}"
        if names[j] in split:
            c, st = split[names[j]]
            fb = model_b((np.array([c]), np.array([st])), *pb)[0]
            zb = (ys[j] - fb) / ses[j]
            fb_s, zb_s, c_s = pct(fb), (f"**{zb:+.1f}**" if abs(zb) > 2 else f"{zb:+.1f}"), f"{c:,.0f}"
        else:
            fb_s, zb_s, c_s = "—", "—", "—"
        P(f"| {names[j]} | {xs[j]:,.0f} | {c_s} | {how[j]} | {pct(ys[j])} | {pct(fa)} | {za_s} | {fb_s} | {zb_s} |")
    P("\nBold z: the arm is off that curve (|z| > 2). Noise arms are not in either fit.\n")

    if plot_path:
        import os
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        fig, ax = plt.subplots(figsize=(8.5, 5.2), facecolor="#fcfcfb")
        ax.set_facecolor("#fcfcfb")
        grid = np.logspace(np.log10(max(1, xs.min() / 2)), np.log10(xs.max() * 2), 200)
        ax.plot(grid, 100 * sat_curve(grid, *popt), color="#6f6e69", lw=1.5, label="saturating fit (fp32 arms)")
        for j in range(len(xs)):
            is_noise = how[j] != "measured"
            ax.errorbar(xs[j], 100 * ys[j], yerr=196 * ses[j], fmt="s" if is_noise else "o", ms=7,
                        color="#eb6834" if is_noise else "#2a78d6", ecolor="#b9b8b0", elinewidth=1, capsize=0)
            if names[j] in ("noise_s0", "noise_s2"):
                continue                              # one label for the three overlapping seeds
            label = "noise s0–s2" if names[j] == "noise_s1" else names[j].replace("fp32_", "").replace("steps_", "")
            ax.annotate(label, (xs[j], 100 * ys[j]),
                        textcoords="offset points", xytext=(6, -3), fontsize=7.5, color="#1a1a19")
        ax.plot([], [], "o", color="#2a78d6", label="fp32 arm (flips measured)")
        ax.plot([], [], "s", color="#eb6834", label="noise arm (flips model-based)")
        ax.set_xscale("log")
        ax.set_xlabel("expected flipped top-8 decisions per question (log scale)", color="#6f6e69")
        ax.set_ylabel("strict outcome churn, uncapped (%)", color="#6f6e69")
        ax.set_title("LLaDA-MoE: churn vs routing flips, 14 Phase 2 arms (n=805, 95% CI)", color="#1a1a19", fontsize=11)
        ax.grid(color="#e4e3dc", lw=0.6)
        ax.legend(frameon=False, fontsize=8, loc="upper left")
        for s in ax.spines.values():
            s.set_color("#d0cfc7")
        fig.tight_layout()
        fig.savefig(plot_path, dpi=140, facecolor="#fcfcfb")
        P(f"\nPlot: `{plot_path}`\n")

    # 2. noise replicates
    P("## Question-level fragility from the three noise seeds only (uncapped, strict)\n")
    seeds = [a for a in ("noise_s0", "noise_s1", "noise_s2") if a in arms]
    sets = [churned(ctrl, arms[a], unc) for a in seeds]
    rates = [len(s) / len(unc) for s in sets]
    cnt = Counter(sum(i in s for s in sets) for i in unc)
    n = len(unc)
    # independence: question-agnostic, each seed churns at its own rate
    from itertools import product
    exp = Counter()
    for pattern in product([0, 1], repeat=len(seeds)):
        p = np.prod([r if b else 1 - r for r, b in zip(rates, pattern)])
        exp[sum(pattern)] += p * n
    P(f"Per-seed churn: {', '.join(pct(r) + '%' for r in rates)}.\n")
    P("| seeds changing the answer | observed | expected if independent of the question |")
    P("|---:|---:|---:|")
    for v in range(len(seeds) + 1):
        P(f"| {v} | {cnt.get(v, 0)} | {exp[v]:.1f} |")
    jac = []
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            jac.append(len(sets[i] & sets[j]) / len(sets[i] | sets[j]))
    ri = np.mean(rates)
    jac_ind = ri * ri / (2 * ri - ri * ri)
    P(f"\nPairwise Jaccard: observed {', '.join(f'{v:.2f}' for v in jac)} (mean {np.mean(jac):.2f}); "
      f"expected under independence at these rates ≈ {jac_ind:.2f}. 3-of-3: observed {cnt.get(3, 0)} "
      f"({pct(cnt.get(3, 0) / n)}%), expected {exp[3]:.1f} ({pct(exp[3] / n)}%).\n")
    counts = [sum(i in s for s in sets) for i in unc]
    a, b, nll = betabinom_fit(counts, len(seeds))
    frac_half = 1 - betainc(a, b, 0.5)
    rng = np.random.default_rng(0)
    boot = []
    arr = np.array(counts)
    for _ in range(500):
        aa, bb, _ = betabinom_fit(arr[rng.integers(0, n, n)], len(seeds))
        boot.append((aa, bb, 1 - betainc(aa, bb, 0.5), aa / (aa + bb)))
    boot = np.array(boot)
    lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
    P(f"Beta-binomial fit over the 3 seeds: per-question flip probability p ~ Beta(α = {a:.3f}, β = {b:.3f}) "
      f"[95% bootstrap: α {lo[0]:.3f}–{hi[0]:.3f}, β {lo[1]:.3f}–{hi[1]:.3f}]; mean p = {pct(a / (a + b))}% "
      f"[{pct(lo[3])}, {pct(hi[3])}]. Implied fraction of questions with p > 0.5: **{pct(frac_half)}%** "
      f"[{pct(lo[2])}, {pct(hi[2])}].")
    qs = [0.1, 0.25, 0.5]
    from scipy.stats import beta as beta_dist
    P(" Implied fraction with p < " + ", ".join(f"{q}: {pct(beta_dist.cdf(q, a, b))}%" for q in qs) + ".")
    P("Three trials per question identify the spread of p only coarsely; the bootstrap CI is the honest "
      "statement of that.\n")

    # 3. blocks 6-7 baseline
    if tok is not None:
        from research.sq2 import EOS_ID
        P("## Baseline for the blocks 6–7 result\n")
        late = set()
        for i in unc:
            r = ctrl[i]
            t = r["text"]
            m = None
            for pat in (r"final\s*answer\s*[:\-]\s*(.+)", r"the\s+answer\s+is\s*[:\-]?\s*(.+)"):
                m = re.search(pat, t.strip(), re.IGNORECASE)
                if m:
                    break
            if not m:
                continue
            pos_c = len(t) - len(t.lstrip()) + m.start(1)
            gen = r["out"][: r["out"].index(EOS_ID)] if EOS_ID in r["out"] else r["out"]
            lo_, hi_ = 0, len(gen)
            while lo_ < hi_:
                mid = (lo_ + hi_) // 2
                if len(tok.decode(gen[:mid + 1], skip_special_tokens=True)) > pos_c:
                    hi_ = mid
                else:
                    lo_ = mid + 1
            if lo_ // 32 >= 6:
                late.add(i)
        P(f"{len(late)} uncapped questions have their control answer line in blocks 6–7.\n")
        P("| arm | churn on those questions [95% CI] |")
        P("|---|---:|")
        for a in ("fp32_all", "fp32_steps_global_early", "fp32_steps_global_last25"):
            k = len(churned(ctrl, arms[a], late))
            P(f"| {a} | {pct(k / len(late))}% ({k}/{len(late)}) [{pct(wilson_ci(k, len(late))[0])}, {pct(wilson_ci(k, len(late))[1])}] |")
        P("")

    # 4. accuracy noise floor
    P("## Accuracy noise floor and Holm correction (repo grader, all 1319 questions)\n")
    rows = []
    for a in present:
        c = paired_compare(ctrl, arms[a], ids, "answer")
        rows.append((a, c["only_a"], c["only_b"], c["mcnemar_p"], c["acc_b"] - c["acc_a"]))
    d_rates = [(b + c) / len(ids) for _, b, c, _, _ in rows]
    d_fp32 = next((b + c) / len(ids) for a, b, c, _, _ in rows if a == "fp32_all")
    P(f"Discordant (correctness-churn) rate d: fp32_all {pct(d_fp32)}%; across the 14 arms "
      f"{pct(min(d_rates))}–{pct(max(d_rates))}%. Under symmetric churn (no true effect) the paired "
      f"accuracy difference has SD = √(d/n).\n")
    P("| n | SD of paired Δacc at d = fp32_all | 95% half-width | SD at the lowest-churn arm's d |")
    P("|---:|---:|---:|---:|")
    for nn in (200, 500, 1000, 1319):
        sd = math.sqrt(d_fp32 / nn)
        P(f"| {nn} | {100 * sd:.2f} pt | ±{196 * sd:.2f} pt | {100 * math.sqrt(min(d_rates) / nn):.2f} pt |")
    zs = [(c - b) / math.sqrt(b + c) for _, b, c, _, _ in rows if b + c > 0]
    P(f"\nCheck against the observed arms: z = (only-arm − only-control)/√(discordant) over the 14 arms has "
      f"mean {np.mean(zs):+.2f}, SD {np.std(zs, ddof=1):.2f} (≈ 1 expected under no effect; the arms share "
      f"one control and overlapping fragile questions, so they are not independent draws). Largest "
      f"|Δacc| = {100 * max(abs(r[4]) for r in rows):.1f} pt, against a 95% half-width of "
      f"±{196 * math.sqrt(d_fp32 / len(ids)):.1f} pt at n = 1319. §9b's \"±3 pt at n=200\" corresponds to "
      f"the 1-SD value at d ≈ {pct(0.03 ** 2 * 200)}%.\n")
    m = len(rows)
    order = sorted(range(m), key=lambda j: rows[j][3])
    holm = [None] * m
    running = 0.0
    for rank, j in enumerate(order):
        adj = min(1.0, (m - rank) * rows[j][3])
        running = max(running, adj)
        holm[j] = running
    P("| arm | only control ✓ | only arm ✓ | Δacc | McNemar p | Holm-adjusted p | survives (α = 0.05) |")
    P("|---|---:|---:|---:|---:|---:|---|")
    for j in order:
        a, b, c, p, d = rows[j]
        P(f"| {a} | {b} | {c} | {100 * d:+.1f} pt | {p:.3g} | {holm[j]:.3g} | {'yes' if holm[j] < 0.05 else 'no'} |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True)
    ap.add_argument("--telemetry", required=True)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--plot", default=None)
    a = ap.parse_args()
    from research.sq1 import load as load_tel
    arms, _ = load(a.dirs)
    tok = None
    if a.tokenizer:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(a.tokenizer, trust_remote_code=True)
    print(report(arms, load_tel(a.telemetry), tok, a.plot))


if __name__ == "__main__":
    main()
