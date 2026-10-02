"""SQ1: router flatness vs mask fraction, by pass kind, layer and position type.

    python -m research.sq1 research/data/p1/L256_b32_s256_fixed_B32 [--plots research/plots]

Reads <arm>.telemetry.seg*.pt (summed over segments) written by the probe.
Slot layout: [0, 0] = cache-prime pass; [b+1, s < S] = denoising step s of
block b; [b+1, S] = block b's finalize pass.
"""

from __future__ import annotations

import argparse
import glob
import math
import os

import torch

from research.metrics import MARGIN_EDGES_ULP

SERIES = {"active_masked": "#2a78d6", "active_unmasked": "#eb6834", "future_masked": "#1baf7a"}
INK, MUTED, SURFACE = "#1a1a19", "#6f6e69", "#fcfcfb"


def load(d: str) -> dict:
    tel = None
    for p in sorted(glob.glob(os.path.join(d, "*.telemetry.seg*.pt"))):
        t = torch.load(p)
        if tel is None:
            tel = t
        else:
            tel["acc"] += t["acc"]
            tel["mask_acc"] += t["mask_acc"]
    if tel is None:
        raise SystemExit(f"no telemetry in {d}")
    return tel


def reduce(acc: torch.Tensor, cols: list) -> dict:
    """acc[..., K] summed over leading dims -> per-position means."""
    c = acc[..., cols.index("count")]
    safe = c.clamp(min=1)
    out = {"count": c}
    for k in ("entropy_norm", "top1_weight", "flip_top8", "flip_top1", "tie_bf16"):
        out[k] = acc[..., cols.index(k)] / safe
    h0 = cols.index("hist_0")
    hist = acc[..., h0:h0 + len(MARGIN_EDGES_ULP) - 1]
    out["margin_lt_0.5ulp"] = hist[..., :2].sum(-1) / safe          # [0,.25) + [.25,.5)
    out["margin_lt_1ulp"] = hist[..., :3].sum(-1) / safe
    out["margin_median_ulp"] = _hist_median(hist)
    return out


def _hist_median(hist: torch.Tensor) -> torch.Tensor:
    """Median from the margin histogram, log-interpolated inside its bin."""
    edges = torch.tensor(MARGIN_EDGES_ULP[:-1] + (128.0,), dtype=torch.float64)
    cum = hist.cumsum(-1)
    tot = cum[..., -1:].clamp(min=1)
    frac = cum / tot
    idx = (frac < 0.5).sum(-1).clamp(max=hist.shape[-1] - 1)
    prev = torch.where(idx > 0, frac.gather(-1, (idx - 1).clamp(min=0).unsqueeze(-1)).squeeze(-1),
                       torch.zeros_like(idx, dtype=frac.dtype))
    cur = frac.gather(-1, idx.unsqueeze(-1)).squeeze(-1)
    t = ((0.5 - prev) / (cur - prev).clamp(min=1e-12)).clamp(0, 1)
    lo, hi = edges[idx], edges[idx + 1]
    lo_l = torch.where(lo > 0, lo, torch.full_like(lo, 0.125))
    return torch.exp(torch.log(lo_l) + t * (torch.log(hi) - torch.log(lo_l)))


METRICS = ("entropy_norm", "top1_weight", "margin_median_ulp", "margin_lt_0.5ulp", "flip_top8")


def tables(tel: dict) -> str:
    acc, cols, types = tel["acc"], tel["stat_cols"], tel["types"]
    S = tel["steps_per_block"]
    prime = acc[0, 0]                 # [L, 4, K]
    steps = acc[1:, :S]               # [NB, S, L, 4, K]
    fin = acc[1:, S]                  # [NB, L, 4, K]
    lines = []

    lines.append("### By pass kind and position type (all layers, all blocks)\n")
    lines.append("| pass | position type | positions | entropy/log64 | top-1 weight | median margin (bf16 ULP) "
                 "| margin < 0.5 ULP | top-8 flip bf16 vs fp32 |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for pname, a in (("prime", prime.sum(0)), ("step", steps.sum((0, 1, 2))), ("finalize", fin.sum((0, 1)))):
        r = reduce(a, cols)
        for ti, t in enumerate(types):
            if r["count"][ti] == 0:
                continue
            lines.append(f"| {pname} | {t} | {int(r['count'][ti]):,} | {r['entropy_norm'][ti]:.4f} "
                         f"| {100 * r['top1_weight'][ti]:.2f}% | {r['margin_median_ulp'][ti]:.2f} "
                         f"| {100 * r['margin_lt_0.5ulp'][ti]:.1f}% | {100 * r['flip_top8'][ti]:.1f}% |")

    lines.append("\n### Per layer, denoising steps only\n")
    r = reduce(steps.sum((0, 1)), cols)    # [L, 4]
    hdr = "| layer | " + " | ".join(f"flip {t}" for t in SERIES) + " | " + \
          " | ".join(f"entropy {t}" for t in SERIES) + " | top-1 (all) | median margin (all) |"
    lines.append(hdr)
    lines.append("|---:|" + "---:|" * (2 * len(SERIES) + 2))
    r_all = reduce(steps.sum((0, 1, 3)), cols)   # [L]
    for l in range(acc.shape[2]):
        f = " | ".join(f"{100 * r['flip_top8'][l, types.index(t)]:.1f}%" for t in SERIES)
        e = " | ".join(f"{r['entropy_norm'][l, types.index(t)]:.4f}" for t in SERIES)
        lines.append(f"| {l} | {f} | {e} | {100 * r_all['top1_weight'][l]:.2f}% | {r_all['margin_median_ulp'][l]:.2f} |")

    lines.append("\n### Which factor is bigger: layer or position type? (denoising steps)\n")
    lines.append("Spread = max minus min of the marginal means. Marginal means are "
                 "count-weighted over everything else.\n")
    lines.append("| metric | spread across 16 layers | spread across position types | ratio layer/type |")
    lines.append("|---|---:|---:|---:|")
    by_type = reduce(steps.sum((0, 1, 2)), cols)
    for m in ("flip_top8", "entropy_norm", "top1_weight", "margin_median_ulp"):
        lay = r_all[m]
        typ = torch.stack([by_type[m][types.index(t)] for t in SERIES])
        sl, st = (lay.max() - lay.min()).item(), (typ.max() - typ.min()).item()
        lines.append(f"| {m} | {sl:.4g} | {st:.4g} | {sl / st if st else math.inf:.1f} |")

    lines.append("\n### Within-block paired contrast: masked vs unmasked at the same (block, step, layer)\n")
    am, au = types.index("active_masked"), types.index("active_unmasked")
    both = (steps[..., am, 0] > 0) & (steps[..., au, 0] > 0)
    rm = reduce(steps[..., am, :], cols)
    ru = reduce(steps[..., au, :], cols)
    w = torch.minimum(steps[..., am, 0], steps[..., au, 0]) * both
    lines.append("| metric | masked − unmasked (weighted mean over cells) | cells where masked is flatter/flips more |")
    lines.append("|---|---:|---:|")
    for m, sign in (("entropy_norm", 1), ("top1_weight", -1), ("margin_median_ulp", -1), ("flip_top8", 1)):
        d = (rm[m] - ru[m])
        mean = (d * w).sum() / w.sum()
        frac = ((sign * d > 0) & both).sum().item() / both.sum().item()
        lines.append(f"| {m} | {mean:+.4g} | {100 * frac:.0f}% of {int(both.sum())} |")
    return "\n".join(lines)


def mask_fraction(tel: dict) -> torch.Tensor:
    """[NB, S]: fraction of the whole generation region still [MASK] at each step."""
    S, NB = tel["steps_per_block"], tel["num_blocks"]
    m = tel["mask_acc"][1:, :S]
    gen_len = None
    # rows * gen_length = masked-in-suffix at block 0 step 0 (everything masked)
    gen_len = (m[0, 0, 2] / m[0, 0, 0]).item()
    return m[..., 2] / (m[..., 0] * gen_len)


def plots(tel: dict, out: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out, exist_ok=True)
    acc, cols, types = tel["acc"], tel["stat_cols"], tel["types"]
    S = tel["steps_per_block"]
    steps = acc[1:, :S]                                   # [NB, S, L, 4, K]
    mf = mask_fraction(tel).flatten()                     # [NB*S]
    flat = steps.reshape(-1, *steps.shape[2:])            # [NB*S, L, 4, K]
    titles = {"entropy_norm": "Router entropy / log 64", "top1_weight": "Top-1 routing weight",
              "margin_median_ulp": "Median 8th–9th logit gap (bf16 ULP)", "flip_top8": "Top-8 set flip rate, bf16 vs fp32"}
    for m, title in titles.items():
        fig, axes = plt.subplots(4, 4, figsize=(13, 10), sharex=True, sharey=True, facecolor=SURFACE)
        for l, ax in enumerate(axes.flat):
            ax.set_facecolor(SURFACE)
            for t, color in SERIES.items():
                r = reduce(flat[:, l, types.index(t)], cols)
                ok = r["count"] > 0
                ax.plot(mf[ok].numpy(), r[m][ok].numpy(), "-", lw=1.4, color=color, label=t.replace("_", " "))
            ax.set_title(f"layer {l}", fontsize=9, color=INK)
            ax.grid(color="#e4e3dc", lw=0.6)
            ax.tick_params(labelsize=7, colors=MUTED)
            for s in ax.spines.values():
                s.set_color("#d0cfc7")
        axes.flat[0].invert_xaxis()   # shared x: invert once, not once per panel
        for ax in axes[-1]:
            ax.set_xlabel("fraction of generation still [MASK]", fontsize=8, color=MUTED)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.suptitle(f"{title}, by layer, across denoising steps", y=0.995, fontsize=12, color=INK)
        fig.text(0.5, 0.962, "x runs left to right from all-masked (1.0) to all-revealed (0.0); "
                 "each sawtooth tooth is one 32-token block", ha="center", fontsize=9, color=MUTED)
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.95), ncol=3,
                   frameon=False, fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        p = os.path.join(out, f"sq1_{m}.png")
        fig.savefig(p, dpi=130, facecolor=SURFACE)
        plt.close(fig)
        print("wrote", p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--plots", default=None)
    a = ap.parse_args()
    tel = load(a.dir)
    print(tables(tel))
    if a.plots:
        plots(tel, a.plots)


if __name__ == "__main__":
    main()
