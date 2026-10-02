"""SQ2 (+ saturation and perturbation-family) analysis of the Phase 2 arms.

    python -m research.sq2 --dirs research/data/p2A/... research/data/p2B/... research/data/p2C/... \
        --telemetry research/data/p1/L256_b32_s256_fixed_B32 --tokenizer <dir with LLaDA tokenizer>

Every comparison is paired against the single control arm (container A) over
the same 1319 questions. Primary metric: outcome churn under the STRICT answer
extraction on questions that are uncapped in control (control emitted EOS
within the 256-token budget). Capped/uncapped is decided from control only.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from collections import Counter

import numpy as np

from research.metrics import first_divergence, mcnemar_exact, paired_compare, wilson_ci
from research.runner import _read_jsonl

# arm -> (description, window kind for the telemetry lookup)
ARMS = {
    "fp32_all": "fp32, every pass",
    "fp32_prime": "fp32, prime pass only",
    "fp32_finalize": "fp32, finalize passes only",
    "fp32_steps_within_early": "fp32, steps 0–7 of each block",
    "fp32_steps_within_late": "fp32, steps 8–31 of each block",
    "fp32_steps_within_last25": "fp32, steps 24–31 of each block",
    "fp32_steps_global_early": "fp32, steps of blocks 0–1",
    "fp32_steps_global_last25": "fp32, steps of blocks 6–7",
    "fp32_layer0": "fp32, layer 0 only, every pass",
    "fp32_layers11_14": "fp32, layers 11–14 only, every pass",
    "fp32_layer15": "fp32, layer 15 only, every pass",
    "noise_s0": "bf16 + U(±1 ULP) noise, seed 0",
    "noise_s1": "bf16 + U(±1 ULP) noise, seed 1",
    "noise_s2": "bf16 + U(±1 ULP) noise, seed 2",
}
NB, S, BL = 8, 32, 32          # blocks, steps per block, block length (cheap config)
EOS_ID = 156892


def is_capped(row) -> bool:
    """The control generation used the whole budget.

    The model pads unused positions with EOS, so 'emitted an EOS' is true of
    almost every row and cannot define truncation. The first-EOS position is
    bimodal instead: continuous up to ~244, empty from 245 to 254, and a spike
    at exactly 255 (the last slot), i.e. the model stopped only because the
    budget did. Capped = first EOS at the last position, or none at all.
    """
    out = row["out"]
    return EOS_ID not in out or out.index(EOS_ID) >= len(out) - 1


def pct(x):
    return f"{100 * x:.1f}"


def ci(c):
    return f"[{pct(c[0])}, {pct(c[1])}]"


def load(dirs):
    arms, counts = {}, {}
    for d in dirs:
        tag = os.path.basename(os.path.dirname(os.path.normpath(d)))
        for p in sorted(glob.glob(os.path.join(d, "*.jsonl"))):
            name = os.path.basename(p)[:-6]
            key = f"{name}@{tag}" if name.startswith("control_sentinel") else name
            arms[key] = _read_jsonl(p)
            c = Counter()
            for cp in glob.glob(os.path.join(d, f"{name}.counts.seg*.json")):
                c.update(json.load(open(cp))["counts"])
            counts[key] = dict(c)
    return arms, counts


def churned(ctrl, arm, ids, key="answer_strict"):
    return {i for i in ids if ctrl[i][key] != arm[i][key]}


# ── telemetry: flip rate inside each window (Phase 1 control trajectory) ──

def window_flip(tel, arm):
    import torch
    acc, cols = tel["acc"], tel["stat_cols"]
    cnt = acc[..., cols.index("count")]
    flp = acc[..., cols.index("flip_top8")]
    Sx = tel["steps_per_block"]
    sel = torch.zeros(cnt.shape[:3], dtype=torch.bool)       # [blocks+1, steps+1, L]
    if arm in ("fp32_all", "noise_s0", "noise_s1", "noise_s2"):
        sel[:] = True
    elif arm == "fp32_prime":
        sel[0, 0] = True
    elif arm == "fp32_finalize":
        sel[1:, Sx] = True
    elif arm.startswith("fp32_steps_within"):
        lo, hi = {"early": (0, 8), "late": (8, 32), "last25": (24, 32)}[arm.rsplit("_", 1)[1]]
        sel[1:, lo:hi] = True
    elif arm.startswith("fp32_steps_global"):
        lo, hi = {"early": (0, 2), "last25": (6, 8)}[arm.rsplit("_", 1)[1]]
        sel[1 + lo:1 + hi, :Sx] = True
    elif arm.startswith("fp32_layer"):
        layers = {"fp32_layer0": [0], "fp32_layers11_14": [11, 12, 13, 14], "fp32_layer15": [15]}[arm]
        sel[0, 0, layers] = True
        sel[1:, :, layers] = True
    else:
        return float("nan")
    m = sel.unsqueeze(-1)
    return (flp * m).sum().item() / (cnt * m).sum().item()


# ── answer-token confidence ──

def answer_token_conf(row, tok):
    """Min commit-confidence over the tokens of the strict answer's number, or None."""
    text = row["text"]
    for pat in (r"final\s*answer\s*[:\-]\s*(.+)", r"the\s+answer\s+is\s*[:\-]?\s*(.+)"):
        m = re.search(pat, text.strip(), re.IGNORECASE)
        if m:
            break
    else:
        return None
    base = len(text) - len(text.lstrip()) + m.start(1)
    line = m.group(1).split("\n")[0]
    nums = list(re.finditer(r"-?\d[\d,]*\.?\d*", line))
    if not nums:
        return None
    a, b = base + nums[-1].start(), base + nums[-1].end()
    gen = row["out"]
    eos = tok.eos_token_id
    gen = gen[: gen.index(eos)] if eos in gen else gen

    def chars(k):                       # decoded length of the first k tokens
        return len(tok.decode(gen[:k], skip_special_tokens=True))

    lo, hi = 0, len(gen)                # first token whose prefix reaches past a
    while lo < hi:
        mid = (lo + hi) // 2
        if chars(mid + 1) > a:
            hi = mid
        else:
            lo = mid + 1
    first = lo
    lo, hi = first, len(gen)
    while lo < hi:
        mid = (lo + hi) // 2
        if chars(mid + 1) >= b:
            hi = mid
        else:
            lo = mid + 1
    conf = [c for c in row["conf"][first:lo + 1] if c == c]
    return min(conf) if conf else None


# ── statistics ──

def auc(scores, labels):
    """P(score of a positive > score of a negative), ties counted half."""
    from scipy.stats import rankdata
    s, y = np.asarray(scores, float), np.asarray(labels, bool)
    n1, n0 = y.sum(), (~y).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(s)
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def boot_ci(fn, *cols, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    cols = [np.asarray(c) for c in cols]
    m = len(cols[0])
    vals = []
    for _ in range(n):
        ix = rng.integers(0, m, m)
        v = fn(*[c[ix] for c in cols])
        if v == v:
            vals.append(v)
    return np.percentile(vals, [2.5, 97.5]) if vals else (float("nan"),) * 2


def spearman(x, y):
    from scipy.stats import spearmanr
    return spearmanr(x, y).statistic


def poisson_binomial(ps):
    dist = np.zeros(len(ps) + 1)
    dist[0] = 1.0
    for p in ps:
        dist[1:] = dist[1:] * (1 - p) + dist[:-1] * p
        dist[0] *= (1 - p)
    return dist


# ── report ──

def report(arms, counts, tel, tok):
    ctrl = arms["control"]
    common = set(ctrl)
    for a in ARMS:
        if a in arms:
            common &= set(arms[a])
    if len(common) < len(ctrl):
        print(f"[note] only {len(common)}/{len(ctrl)} questions present in every arm; analysing those")
    ids = sorted(common)
    uncapped = [i for i in ids if not is_capped(ctrl[i])]
    capped = [i for i in ids if is_capped(ctrl[i])]
    out = []
    P = out.append

    P("Capped = control's first EOS at the last position (255) or absent; see is_capped().\n")
    P(f"Questions: {len(ids)}; uncapped in control: {len(uncapped)}; capped: {len(capped)} "
      f"({pct(len(capped) / len(ids))}%). Control strict answer missing: "
      f"{sum(ctrl[i]['answer_strict'] is None for i in ids)} (uncapped: "
      f"{sum(ctrl[i]['answer_strict'] is None for i in uncapped)}).\n")

    P("### Cross-container reproducibility (control's first 64 questions, re-run in containers B and C)\n")
    for k in [k for k in arms if k.startswith("control_sentinel")]:
        s = arms[k]
        same = sum(s[i]["out"] == ctrl[i]["out"] and s[i]["commit"] == ctrl[i]["commit"] for i in s)
        P(f"- {k}: {same}/{len(s)} identical in every committed token and commit step")
    P("")

    P("### Churn per arm vs control\n")
    P("Primary = strict extraction, uncapped questions. Correctness and McNemar use the repo's grader.\n")
    P("| arm | perturbation | strict churn, uncapped (primary) | strict churn, all | lenient churn, all "
      "| acc ctrl → arm | McNemar p | trajectories diverged | median first-divergence step [IQR] |")
    P("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    present = [a for a in ARMS if a in arms]
    for a in present:
        r = arms[a]
        pu = paired_compare(ctrl, r, uncapped, "answer_strict")
        pa = paired_compare(ctrl, r, ids, "answer_strict")
        pl = paired_compare(ctrl, r, ids, "answer")
        divs = [first_divergence(ctrl[i]["out"], ctrl[i]["commit"], r[i]["out"], r[i]["commit"]) for i in ids]
        dv = [d for d in divs if d is not None]
        q = np.percentile(dv, [25, 50, 75]) if dv else [float("nan")] * 3
        P(f"| {a} | {ARMS[a]} | {pct(pu['outcome_churn'])} {ci(pu['outcome_churn_ci'])} "
          f"| {pct(pa['outcome_churn'])} {ci(pa['outcome_churn_ci'])} | {pct(pl['outcome_churn'])} "
          f"| {pct(pl['acc_a'])} → {pct(pl['acc_b'])} | {pl['mcnemar_p']:.3g} "
          f"| {pct(len(dv) / len(ids))}% | {q[1]:.0f} [{q[0]:.0f}–{q[2]:.0f}] |")
    P("\nFirst-divergence step is global: block = step // 32.\n")

    P("### Windows: churn normalized by the share of the run perturbed\n")
    P("| arm | share of denoising steps | share of forward passes | share of routed token-rows "
      "| flip rate inside window (Phase 1 telemetry, control, n=200) | strict churn, uncapped "
      "| churn / step share |")
    P("|---|---:|---:|---:|---:|---:|---:|")
    for a in present:
        if not a.startswith("fp32"):
            continue
        c = counts.get(a, {})
        st = c.get("step_passes_fp32", 0) / max(1, c.get("step_passes", 1))
        ps = c.get("passes_fp32", 0) / max(1, c.get("passes", 1))
        rw = c.get("rows_fp32", 0) / max(1, c.get("rows", 1))
        ch = paired_compare(ctrl, arms[a], uncapped, "answer_strict")["outcome_churn"]
        norm = f"{ch / st:.3f}" if st > 0 and not a.startswith("fp32_layer") else "—"
        note = " (all layers' passes counted; fp32 only at the named layers)" if a.startswith("fp32_layer") else ""
        P(f"| {a}{note} | {pct(st)}% | {pct(ps)}% | {pct(rw)}% | {pct(window_flip(tel, a))}% | {pct(ch)} | {norm} |")

    P("\n### Pairwise window contrasts (paired over the same uncapped questions)\n")
    P("Each cell: questions churned only under the first arm vs only under the second; exact McNemar p.\n")
    P("| contrast | only first | only second | churn first | churn second | McNemar p |")
    P("|---|---:|---:|---:|---:|---:|")
    for x, y in [("fp32_steps_within_early", "fp32_steps_within_last25"),
                 ("fp32_steps_global_early", "fp32_steps_global_last25"),
                 ("fp32_steps_within_early", "fp32_steps_within_late"),
                 ("fp32_prime", "fp32_finalize"),
                 ("fp32_layer0", "fp32_layer15"),
                 ("fp32_layer0", "fp32_layers11_14"),
                 ("fp32_layers11_14", "fp32_layer15")]:
        if x in arms and y in arms:
            cx, cy = churned(ctrl, arms[x], uncapped), churned(ctrl, arms[y], uncapped)
            b, c = len(cx - cy), len(cy - cx)
            P(f"| {x} vs {y} | {b} | {c} | {pct(len(cx) / len(uncapped))} | {pct(len(cy) / len(uncapped))} "
              f"| {mcnemar_exact(b, c):.3g} |")

    parts = ["fp32_prime", "fp32_finalize", "fp32_steps_within_early", "fp32_steps_within_late"]
    if all(p in arms for p in parts) and "fp32_all" in arms:
        P("\n### Additivity: does the partition reproduce fp32_all? (uncapped, strict)\n")
        sets = {p: churned(ctrl, arms[p], uncapped) for p in parts}
        full = churned(ctrl, arms["fp32_all"], uncapped)
        union = set().union(*sets.values())
        P(f"- sum of part churn counts: {sum(len(s) for s in sets.values())}; |union|: {len(union)}; "
          f"fp32_all churn: {len(full)}")
        P(f"- fp32_all churned questions also churned by at least one part: {len(full & union)}/{len(full)}; "
          f"Jaccard(union, fp32_all) = {len(full & union) / max(1, len(full | union)):.2f}")

    noise = [a for a in ("noise_s0", "noise_s1", "noise_s2") if a in arms]
    if noise and "fp32_all" in arms:
        P("\n### Perturbation family: noise vs fp32 (uncapped, strict)\n")
        P("Caveat: U(−1, +1) ULP noise has mean |δ| = 0.5 ULP per logit and reaches 1 ULP; "
          "bf16 rounding moves a logit by at most 0.5 ULP (mean 0.25). The noise arms are the "
          "larger perturbation, so equal churn would mean saturation, not equivalence.\n")
        nch = [paired_compare(ctrl, arms[a], uncapped, "answer_strict")["outcome_churn"] for a in noise]
        fch = paired_compare(ctrl, arms["fp32_all"], uncapped, "answer_strict")["outcome_churn"]
        P(f"- fp32_all: {pct(fch)}%; noise seeds: {', '.join(pct(v) + '%' for v in nch)} "
          f"(mean {pct(np.mean(nch))}%, range {pct(max(nch) - min(nch))} pts)")
        sets = [churned(ctrl, arms[a], uncapped) for a in noise] + [churned(ctrl, arms["fp32_all"], uncapped)]
        names = noise + ["fp32_all"]
        P("- pairwise overlap of churned sets (Jaccard):")
        for i in range(len(sets)):
            for j in range(i + 1, len(sets)):
                P(f"  - {names[i]} / {names[j]}: {len(sets[i] & sets[j]) / max(1, len(sets[i] | sets[j])):.2f}")

    # ── saturation ──
    P("\n### Saturation: how many perturbed arms change each question's strict answer (uncapped)\n")
    pert = present
    k = {i: sum(ctrl[i]["answer_strict"] != arms[a][i]["answer_strict"] for a in pert) for i in uncapped}
    rates = [len(churned(ctrl, arms[a], uncapped)) / len(uncapped) for a in pert]
    null = poisson_binomial(rates) * len(uncapped)
    obs = Counter(k.values())
    P(f"{len(pert)} perturbed arms. Null = each arm churns independently of the question, "
      "at its own observed rate (Poisson-binomial).\n")
    P("| arms changing the answer | questions observed | expected under independence |")
    P("|---:|---:|---:|")
    for v in range(len(pert) + 1):
        P(f"| {v} | {obs.get(v, 0)} | {null[v]:.1f} |")
    events = sum(k.values())
    heavy = [i for i in uncapped if k[i] >= len(pert) / 2]
    P(f"\nChurn events (question × arm): {events}. Questions churned by ≥ half the arms: {len(heavy)} "
      f"({pct(len(heavy) / len(uncapped))}% of uncapped) carry {sum(k[i] for i in heavy)} events "
      f"({pct(sum(k[i] for i in heavy) / max(1, events))}%). Questions churned by any arm: "
      f"{sum(v > 0 for v in k.values())}.")

    # ── predictors ──
    P("\n### Do control-arm signals predict fragility? (uncapped questions)\n")
    P("Target A: churn count k over the perturbed arms (Spearman ρ). Target B: churned under fp32_all "
      "(AUC = P(signal of a churned question > signal of a stable one)). 95% bootstrap CIs, 2000 resamples.\n")
    full = churned(ctrl, arms["fp32_all"], uncapped) if "fp32_all" in arms else set()
    sig = {}
    if tok is not None:
        sig["answer-token confidence at commit (min over the number's tokens)"] = {
            i: answer_token_conf(ctrl[i], tok) for i in uncapped}
    sig["mean commit confidence, whole generation"] = {
        i: float(np.nanmean([c for c, t in zip(ctrl[i]["conf"], ctrl[i]["out"]) if c == c])) for i in uncapped}
    sig["generated length (tokens before EOS)"] = {i: ctrl[i]["n_tokens"] for i in uncapped}
    if "fp32_all" in arms:
        fd = {}
        for i in uncapped:
            d = first_divergence(ctrl[i]["out"], ctrl[i]["commit"], arms["fp32_all"][i]["out"], arms["fp32_all"][i]["commit"])
            fd[i] = NB * S if d is None else d
        sig["first-divergence step vs fp32_all (256 if none)"] = fd
    P("| signal | n | Spearman ρ with k [95% CI] | AUC for fp32_all churn [95% CI] |")
    P("|---|---:|---:|---:|")
    for name, vals in sig.items():
        qs = [i for i in uncapped if vals.get(i) is not None]
        x = np.array([vals[i] for i in qs], float)
        kk = np.array([k[i] for i in qs], float)
        y = np.array([i in full for i in qs])
        rho = spearman(x, kk)
        rci = boot_ci(spearman, x, kk)
        au = auc(x, y)
        aci = boot_ci(auc, x, y)
        P(f"| {name} | {len(qs)} | {rho:+.3f} [{rci[0]:+.3f}, {rci[1]:+.3f}] | {au:.3f} [{aci[0]:.3f}, {aci[1]:.3f}] |")
    # capped status needs all questions: it is the population split itself
    kall = {i: sum(ctrl[i]["answer_strict"] != arms[a][i]["answer_strict"] for a in pert) for i in ids}
    yc = np.array([is_capped(ctrl[i]) for i in ids])
    kc = np.array([kall[i] for i in ids], float)
    P(f"\nCapped status (all {len(ids)} questions, strict): mean k = {kc[yc].mean():.2f} capped vs "
      f"{kc[~yc].mean():.2f} uncapped; AUC(capped → k ≥ 1) = {auc(yc.astype(float), kc >= 1):.3f}.")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True)
    ap.add_argument("--telemetry", required=True)
    ap.add_argument("--tokenizer", default=None)
    a = ap.parse_args()
    import torch
    from research.sq1 import load as load_tel
    arms, counts = load(a.dirs)
    tel = load_tel(a.telemetry)
    tok = None
    if a.tokenizer:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(a.tokenizer, trust_remote_code=True)
    print(report(arms, counts, tel, tok))


if __name__ == "__main__":
    main()
