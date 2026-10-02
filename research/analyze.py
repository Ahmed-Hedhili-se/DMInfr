"""Paired tables from arm JSONL files.

    python -m research.analyze research/data/<phase>/<config_dir> --control control [--vs a,b,...]

Every comparison is paired over the shared question ids. It reports outcome
churn (extracted final answer differs), correctness churn, accuracies, all with
95% Wilson CIs, and the exact McNemar p on the correctness discordants.
"""

from __future__ import annotations

import argparse
import glob
import os

from research.metrics import paired_compare, wilson_ci
from research.runner import _read_jsonl


def load_dir(d: str) -> dict:
    return {os.path.basename(p)[:-len(".jsonl")]: _read_jsonl(p) for p in sorted(glob.glob(os.path.join(d, "*.jsonl")))}


def pct(x: float) -> str:
    return f"{100 * x:.1f}"


def ci(c: tuple) -> str:
    return f"[{pct(c[0])}, {pct(c[1])}]"


def arm_line(name: str, rows: dict) -> str:
    n = len(rows)
    k = sum(r["correct"] for r in rows.values())
    col = sum(r["collapse"] for r in rows.values())
    return (f"| {name} | {n} | {pct(k / n)}% {ci(wilson_ci(k, n))} | {col} "
            f"| {sum(r['n_tokens'] for r in rows.values()) / n:.0f} |")


def compare_table(arms: dict, control: str, others: list[str]) -> str:
    out = ["| arm vs " + control + " | n | outcome churn % [95% CI] | correct churn % [95% CI] "
           "| only ctrl ✓ | only arm ✓ | acc ctrl | acc arm | McNemar p | token-identical |",
           "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for o in others:
        c = paired_compare(arms[control], arms[o])
        out.append(
            f"| {o} | {c['n']} | {pct(c['outcome_churn'])} {ci(c['outcome_churn_ci'])} "
            f"| {pct(c['correct_churn'])} {ci(c['correct_churn_ci'])} | {c['only_a']} | {c['only_b']} "
            f"| {pct(c['acc_a'])} | {pct(c['acc_b'])} | {c['mcnemar_p']:.3g} | {c['n'] - c['token_diff']} |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--control", default="control")
    ap.add_argument("--vs", default=None, help="comma-separated arms; default: all others")
    a = ap.parse_args()
    arms = load_dir(a.dir)
    print("| arm | n | accuracy % [95% CI] | collapses | mean gen tokens |\n|---|---:|---:|---:|---:|")
    for name, rows in arms.items():
        print(arm_line(name, rows))
    others = a.vs.split(",") if a.vs else [k for k in arms if k != a.control]
    if others:
        print()
        print(compare_table(arms, a.control, others))


if __name__ == "__main__":
    main()
