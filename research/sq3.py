"""SQ3: OLMoE (autoregressive) vs LLaDA-MoE (diffusion) under the same router
perturbations: router flatness per layer, churn, and placement on LLaDA's
dose-response curves.

    python -m research.sq3 --olmoe <p3 dir> --olmoe-selfchurn <p3_selfchurn dir> \
        --llada <p2A> <p2B> <p2C> --llada-telemetry <p1 dir> --plots research/plots
"""

from __future__ import annotations

import argparse
import glob
import math
import os

import numpy as np
import torch

from research.metrics import MARGIN_EDGES_ULP, first_divergence, paired_compare, wilson_ci
from research.runner import _read_jsonl
from research.sq1 import load as load_llada_tel, reduce
from research.sq2 import churned, is_capped, load as load_llada, pct
from research.sq2_extra import _p_flip, dose_points, fit_dose, model_b, sat_curve

N_TEL = 200
LLADA_COUNTERPART = {"fp32_all": "fp32_all", "fp32_prefill": "fp32_prime", "noise_s0": "noise_s0"}


def load_dir(d):
    return {os.path.basename(p)[:-6]: _read_jsonl(p) for p in sorted(glob.glob(os.path.join(d, "*.jsonl")))}


def olmoe_tel(d):
    tel = None
    for p in sorted(glob.glob(os.path.join(d, "*.telemetry.seg*.pt"))):
        t = torch.load(p)
        if tel is None:
            tel = t
        else:
            tel["acc"] += t["acc"]
    return tel


OLMOE_EOS, OLMOE_PAD = 50279, 1


def ar_content(out):
    """Generated tokens up to (and including) the first EOS, pads dropped.

    `out` is the row of the batch's padded output tensor. HF generate() stops
    a batch when every row has finished, so the same row can come back with
    a different amount of trailing padding in two arms even when its text is
    identical; only the content up to EOS is the generation.
    """
    k = out.index(OLMOE_EOS) + 1 if OLMOE_EOS in out else len(out)
    return [t for t in out[:k] if t != OLMOE_PAD]


def ar_divergence(a, b):
    """First generated-token index where two greedy continuations differ
    (comparing content up to EOS), or None if they are identical."""
    a, b = ar_content(a), ar_content(b)
    for k, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return k
    return None if len(a) == len(b) else min(len(a), len(b))


def olmoe_flips(tel, arm, calib=None):
    acc, cols = tel["acc"], tel["stat_cols"]
    f = acc[..., cols.index("flip_top8")]                    # [phase, L]
    if arm == "fp32_all":
        return f.sum().item() / N_TEL, "measured"
    if arm == "fp32_prefill":
        return f[0].sum().item() / N_TEL, "measured"
    if arm.startswith("noise"):
        edges = list(MARGIN_EDGES_ULP[:-1]) + [64.0]
        h0 = cols.index("hist_0")
        hist = acc[..., h0:h0 + len(edges) - 1]
        pr = torch.tensor([_p_flip(edges[i], edges[i + 1], 0.5) for i in range(len(edges) - 1)], dtype=torch.float64)
        pn = torch.tensor([_p_flip(edges[i], edges[i + 1], 1.0) for i in range(len(edges) - 1)], dtype=torch.float64)
        ratio = (hist * pn).sum(-1) / (hist * pr).sum(-1).clamp(min=1e-12)
        est = torch.minimum(f * ratio, acc[..., cols.index("count")])
        return est.sum().item() / N_TEL, "model"
    raise ValueError(arm)


def report(o_arms, o_self, o_tel, l_arms, l_tel, plots):
    out = []
    P = out.append
    ctrl = o_arms["control"]
    ids805 = sorted(ctrl)
    unc = [i for i in ids805 if not ctrl[i]["capped"]]
    l_ctrl = l_arms["control"]

    P("## SQ3: autoregressive MoE (OLMoE-1B-7B-0924-Instruct) vs LLaDA-MoE\n")
    P("Setup: HF transformers 4.53.2, bf16, SDPA attention, greedy, max_new_tokens=256, batch 64 "
      "left-padded, A100-40GB. Same system prompt and 4-shot GSM8K user prompt as LLaDA, rendered by "
      "OLMoE's chat template: `<|endoftext|><|system|>\\n{system}\\n<|user|>\\n{prompt}\\n<|assistant|>\\n` "
      "(tokenized without extra special tokens; the template supplies BOS). Perturbation: a forward "
      "hook on each layer's `mlp.gate` replacing its output (HF code untouched). Questions: the 805 "
      "questions uncapped in LLaDA's control (budget cap for Phase 3, as instructed). OLMoE capped = "
      "control hit 256 new tokens without EOS.\n")

    P("### Determinism\n")
    a, b = o_self["control_tel"], o_self["control_rep"]
    same = sum(a[i]["out"] == b[i]["out"] for i in a)
    P(f"Two control runs, same batches (n={len(a)}, B=64): {same}/{len(a)} token-identical. "
      f"Telemetry hook inert: control_tel is the first of the two.\n")

    # router flatness
    P("### Router flatness per layer (bf16 vs fp32 on the same hidden states)\n")
    P("OLMoE: control telemetry on the first 200 questions (the Phase 1 question set), prefill = prompt "
      "tokens, decode = generated tokens. LLaDA: Phase 1 telemetry, same 200 questions, prompt tokens of "
      "the prime pass and all routed positions of all passes.\n")
    oc = o_tel["stat_cols"]
    o_r = {ph: reduce(o_tel["acc"][k], oc) for k, ph in enumerate(o_tel["phases"])}
    la = l_tel["acc"]
    lc = l_tel["stat_cols"]
    l_prompt = reduce(la[0, 0][:, 0], lc)                              # prime pass, prompt type, [L]
    l_all = reduce(la.sum((0, 1, 3)), lc)                              # every pass and type, [L]
    P("| layer | entropy LLaDA (prompt) | entropy OLMoE (prefill) | top-1 LLaDA | top-1 OLMoE "
      "| median gap ULP LLaDA | median gap ULP OLMoE | flip LLaDA prompt / all | flip OLMoE prefill / decode |")
    P("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    L = la.shape[2]
    for l in range(L):
        P(f"| {l} | {l_prompt['entropy_norm'][l]:.3f} | {o_r['prefill']['entropy_norm'][l]:.3f} "
          f"| {pct(l_prompt['top1_weight'][l])}% | {pct(o_r['prefill']['top1_weight'][l])}% "
          f"| {l_prompt['margin_median_ulp'][l]:.2f} | {o_r['prefill']['margin_median_ulp'][l]:.2f} "
          f"| {pct(l_prompt['flip_top8'][l])}% / {pct(l_all['flip_top8'][l])}% "
          f"| {pct(o_r['prefill']['flip_top8'][l])}% / {pct(o_r['decode']['flip_top8'][l])}% |")
    o_all = reduce(o_tel["acc"].sum((0, 1)), oc)
    l_tot = reduce(la.sum((0, 1, 2, 3)), lc)
    l_pr = reduce(la[0, 0][:, 0].sum(0), lc)
    P(f"| **all** | {l_pr['entropy_norm']:.3f} | {o_all['entropy_norm']:.3f} | {pct(l_pr['top1_weight'])}% "
      f"| {pct(o_all['top1_weight'])}% | {l_pr['margin_median_ulp']:.2f} | {o_all['margin_median_ulp']:.2f} "
      f"| {pct(l_pr['flip_top8'])}% / {pct(l_tot['flip_top8'])}% | {pct(reduce(o_tel['acc'][0].sum(0), oc)['flip_top8'])}% "
      f"/ {pct(reduce(o_tel['acc'][1].sum(0), oc)['flip_top8'])}% |")
    P("")

    # churn
    P("### Churn on the same questions\n")
    P(f"OLMoE primary population: the 805, minus the {len(ids805) - len(unc)} capped in OLMoE's control "
      f"→ {len(unc)}. LLaDA columns use the same {len(unc)} questions (all uncapped in LLaDA's control "
      f"by construction).\n")
    P("| OLMoE arm | strict churn, primary [95% CI] | lenient churn, all 805 | acc ctrl → arm (805) | McNemar p "
      "| diverged | median first-divergent token [IQR] | LLaDA counterpart | LLaDA strict churn, same questions |")
    P("|---|---:|---:|---:|---:|---:|---:|---|---:|")
    o_churn = {}
    for arm in ("fp32_all", "fp32_prefill", "noise_s0"):
        if arm not in o_arms:
            continue
        r = o_arms[arm]
        pu = paired_compare(ctrl, r, unc, "answer_strict")
        pl = paired_compare(ctrl, r, ids805, "answer")
        d = [ar_divergence(ctrl[i]["out"], r[i]["out"]) for i in ids805]
        dv = [x for x in d if x is not None]
        q = np.percentile(dv, [25, 50, 75]) if dv else [float("nan")] * 3
        la_arm = LLADA_COUNTERPART[arm]
        lu = paired_compare(l_ctrl, l_arms[la_arm], unc, "answer_strict")
        o_churn[arm] = (pu["outcome_churn"], pu["outcome_diff"], pu["n"])
        P(f"| {arm} | {pct(pu['outcome_churn'])} [{pct(pu['outcome_churn_ci'][0])}, {pct(pu['outcome_churn_ci'][1])}] "
          f"| {pct(pl['outcome_churn'])} | {pct(pl['acc_a'])} → {pct(pl['acc_b'])} | {pl['mcnemar_p']:.3g} "
          f"| {pct(len(dv) / len(ids805))}% | {q[1]:.0f} [{q[0]:.0f}–{q[2]:.0f}] | {la_arm} "
          f"| {pct(lu['outcome_churn'])} [{pct(lu['outcome_churn_ci'][0])}, {pct(lu['outcome_churn_ci'][1])}] |")
    P("")

    # divergence-conditioned churn and cross-model fragility
    P("### Churn given divergence, and whether the same questions are fragile in both models\n")
    P("| model / arm | trajectories diverged | strict churn among diverged [95% CI] |")
    P("|---|---:|---:|")
    for arm in ("fp32_all", "fp32_prefill", "noise_s0"):
        dv = [i for i in unc if ar_divergence(ctrl[i]["out"], o_arms[arm][i]["out"]) is not None]
        ch = [i for i in dv if ctrl[i]["answer_strict"] != o_arms[arm][i]["answer_strict"]]
        lo, hi = wilson_ci(len(ch), len(dv))
        P(f"| OLMoE {arm} | {len(dv)}/{len(unc)} | {pct(len(ch) / len(dv))} [{pct(lo)}, {pct(hi)}] |")
    for arm in ("fp32_all", "fp32_prime", "noise_s0"):
        dv = [i for i in unc if first_divergence(l_ctrl[i]["out"], l_ctrl[i]["commit"],
                                                 l_arms[arm][i]["out"], l_arms[arm][i]["commit"]) is not None]
        ch = [i for i in dv if l_ctrl[i]["answer_strict"] != l_arms[arm][i]["answer_strict"]]
        lo, hi = wilson_ci(len(ch), len(dv))
        P(f"| LLaDA {arm} | {len(dv)}/{len(unc)} | {pct(len(ch) / len(dv))} [{pct(lo)}, {pct(hi)}] |")
    from scipy.stats import spearmanr
    oset = {a: {i for i in unc if ctrl[i]["answer_strict"] != o_arms[a][i]["answer_strict"]}
            for a in ("fp32_all", "fp32_prefill", "noise_s0")}
    lset = {a: {i for i in unc if l_ctrl[i]["answer_strict"] != l_arms[a][i]["answer_strict"]}
            for a in ("fp32_all", "noise_s0", "noise_s1", "noise_s2")}
    ko = np.array([sum(i in s for s in oset.values()) for i in unc])
    kl = np.array([sum(i in s for s in lset.values()) for i in unc])
    rho = spearmanr(kl, ko).statistic
    rng = np.random.default_rng(0)
    bs = [spearmanr(kl[j], ko[j]).statistic for j in (rng.integers(0, len(unc), len(unc)) for _ in range(1000))]
    A, B = oset["fp32_all"], lset["fp32_all"]
    exp_both = len(A) * len(B) / len(unc)
    P(f"\nCross-model: Spearman ρ between a question's churn count in LLaDA (fp32_all + 3 noise seeds) and "
      f"in OLMoE (its 3 arms) = {rho:.3f} [{np.percentile(bs, 2.5):.3f}, {np.percentile(bs, 97.5):.3f}] "
      f"(bootstrap, n={len(unc)}). Churned under fp32_all in both models: {len(A & B)} questions vs "
      f"{exp_both:.1f} expected if independent (Jaccard {len(A & B) / len(A | B):.2f}).\n")
    P("| model | fp32_all churn, control wrong | fp32_all churn, control right |")
    P("|---|---:|---:|")
    for name, c_, S_ in (("OLMoE", ctrl, A), ("LLaDA", l_ctrl, B)):
        w = [i for i in unc if not c_[i]["correct"]]
        r_ = [i for i in unc if c_[i]["correct"]]
        P(f"| {name} | {pct(len(S_ & set(w)) / len(w))}% (n={len(w)}) | {pct(len(S_ & set(r_)) / len(r_))}% (n={len(r_)}) |")
    P("")

    # dose-response placement
    P("### OLMoE on LLaDA's dose-response curves\n")
    names, xs, ys, ses, how, mask, cx, sx = dose_points(l_arms, l_tel)
    popt, _, chiA, pb, _, chiB = fit_dose(xs[mask], ys[mask], ses[mask], cx, sx)
    P("Expected flips per OLMoE question from its control telemetry (200 questions). Every OLMoE "
      "forward writes the KV cache (prefill and each decode step alike), so model B is evaluated with "
      "all OLMoE flips counted as cache-writing flips, and, as a bound, as step flips. Note the unit "
      "differs in kind: LLaDA re-routes every masked position at every step, so its per-question flip "
      "count includes many re-decisions of the same position; OLMoE routes each token once.\n")
    P("| OLMoE arm | expected flips / question | source | observed churn [95% CI] | LLaDA model A predicts "
      "| model B, as cache flips | model B, as step flips |")
    P("|---|---:|---|---:|---:|---:|---:|")
    o_pts = []
    for arm in ("fp32_all", "fp32_prefill", "noise_s0"):
        if arm not in o_churn:
            continue
        x, h = olmoe_flips(o_tel, arm)
        c, k, n = o_churn[arm]
        lo, hi = wilson_ci(k, n)
        pa = sat_curve(x, *popt)
        pbc = model_b((np.array([x]), np.array([0.0])), *pb)[0]
        pbs = model_b((np.array([0.0]), np.array([x])), *pb)[0]
        o_pts.append((arm, x, c, lo, hi, h))
        P(f"| {arm} | {x:,.0f} | {h} | {pct(c)} [{pct(lo)}, {pct(hi)}] | {pct(pa)} | {pct(pbc)} | {pct(pbs)} |")
    P("")

    if plots:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        os.makedirs(plots, exist_ok=True)
        INK, MUTED, SURF = "#1a1a19", "#6f6e69", "#fcfcfb"
        BLUE, ORANGE = "#2a78d6", "#eb6834"
        # per-layer flatness
        fig, axes = plt.subplots(1, 4, figsize=(15, 3.8), facecolor=SURF)
        specs = [("entropy_norm", "router entropy / log 64", 1), ("top1_weight", "top-1 weight (%)", 100),
                 ("margin_median_ulp", "median 8th–9th gap (bf16 ULP)", 1), ("flip_top8", "top-8 flip rate (%)", 100)]
        for ax, (m, title, sc) in zip(axes, specs):
            ax.set_facecolor(SURF)
            ax.plot(range(L), sc * l_prompt[m].numpy(), "-o", ms=4, lw=1.6, color=BLUE, label="LLaDA-MoE (prompt, prime pass)")
            ax.plot(range(L), sc * o_r["prefill"][m].numpy(), "-s", ms=4, lw=1.6, color=ORANGE, label="OLMoE (prompt, prefill)")
            ax.set_title(title, fontsize=10, color=INK)
            ax.set_xlabel("layer", fontsize=9, color=MUTED)
            ax.grid(color="#e4e3dc", lw=0.6)
            ax.tick_params(labelsize=8, colors=MUTED)
            for s in ax.spines.values():
                s.set_color("#d0cfc7")
        axes[0].legend(frameon=False, fontsize=8)
        fig.suptitle("Router flatness on the same 200 GSM8K prompts (bf16 vs fp32 gate)", color=INK, fontsize=11)
        fig.tight_layout()
        p1 = os.path.join(plots, "sq3_router_flatness.png")
        fig.savefig(p1, dpi=140, facecolor=SURF)
        plt.close(fig)
        # dose-response with OLMoE points
        fig, ax = plt.subplots(figsize=(8.5, 5.2), facecolor=SURF)
        ax.set_facecolor(SURF)
        allx = np.concatenate([xs, [p[1] for p in o_pts]])
        grid = np.logspace(np.log10(max(1, allx.min() / 2)), np.log10(allx.max() * 2), 200)
        ax.plot(grid, 100 * sat_curve(grid, *popt), color=MUTED, lw=1.5, label="LLaDA fit A (flips alone)")
        ax.plot(grid, 100 * model_b((grid, np.zeros_like(grid)), *pb), color=MUTED, lw=1.2, ls="--",
                label="LLaDA fit B, all flips as cache-writing")
        for j in range(len(xs)):
            ax.errorbar(xs[j], 100 * ys[j], yerr=196 * ses[j], fmt="o", ms=6, color=BLUE, alpha=0.55,
                        ecolor="#cfcec6", elinewidth=1)
        ax.plot([], [], "o", color=BLUE, alpha=0.55, label="LLaDA arms (Phase 2)")
        for arm, x, c, lo, hi, h in o_pts:
            ax.errorbar(x, 100 * c, yerr=[[100 * (c - lo)], [100 * (hi - c)]], fmt="D", ms=8, color=ORANGE,
                        ecolor=ORANGE, elinewidth=1.2)
            ax.annotate(f"OLMoE {arm}" + (" (flips model-based)" if h != "measured" else ""), (x, 100 * c),
                        textcoords="offset points", fontsize=8, color=INK,
                        xytext={"fp32_prefill": (7, 7), "fp32_all": (7, -13)}.get(arm, (7, -3)))
        ax.set_xscale("log")
        ax.set_xlabel("expected flipped top-8 decisions per question (log scale)", color=MUTED)
        ax.set_ylabel("strict outcome churn (%)", color=MUTED)
        ax.set_title("Churn vs routing flips: OLMoE on LLaDA-MoE's dose-response curves", color=INK, fontsize=11)
        ax.grid(color="#e4e3dc", lw=0.6)
        ax.legend(frameon=False, fontsize=8, loc="upper left")
        for s in ax.spines.values():
            s.set_color("#d0cfc7")
        fig.tight_layout()
        p2 = os.path.join(plots, "sq3_dose_response.png")
        fig.savefig(p2, dpi=140, facecolor=SURF)
        plt.close(fig)
        P(f"Plots: `{p1}`, `{p2}`\n")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--olmoe", required=True)
    ap.add_argument("--olmoe-selfchurn", required=True)
    ap.add_argument("--llada", nargs="+", required=True)
    ap.add_argument("--llada-telemetry", required=True)
    ap.add_argument("--plots", default=None)
    a = ap.parse_args()
    o_arms = load_dir(a.olmoe)
    o_self = load_dir(a.olmoe_selfchurn)
    o_tel = olmoe_tel(a.olmoe_selfchurn)
    l_arms, _ = load_llada(a.llada)
    print(report(o_arms, o_self, o_tel, l_arms, load_llada_tel(a.llada_telemetry), a.plots))


if __name__ == "__main__":
    main()
