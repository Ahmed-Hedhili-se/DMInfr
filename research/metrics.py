"""Every metric the routing-fragility study reports, defined once.

Two families:

  * router flatness / fragility, computed on GPU tensors per routed position
    (router_stats) -- used by the telemetry probe for both LLaDA-MoE and OLMoE;
  * paired outcome statistics over per-question JSONL rows (paired_compare) --
    churn, McNemar, Wilson CIs.

Nothing here touches the engine.
"""

from __future__ import annotations

import math
import re
from typing import Iterable, Optional

import torch
import torch.nn.functional as F

#: Boundary-margin histogram edges, in bf16 ULPs at the 8th logit's magnitude.
#: Below ~0.5 ULP, rounding the logits to bf16 can tie or swap experts 8 and 9.
MARGIN_EDGES_ULP = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, math.inf)

#: Columns of the per-position stat vector router_stats() returns, in order.
#: hist_<i> counts positions whose margin falls in [EDGES[i], EDGES[i+1]).
STAT_COLS = (
    ["count", "entropy_norm", "top1_weight", "margin_ulp_clip64", "flip_top8", "flip_top1", "tie_bf16"]
    + [f"hist_{i}" for i in range(len(MARGIN_EDGES_ULP) - 1)]
)


def bf16_ulp(v: torch.Tensor) -> torch.Tensor:
    """Spacing of the bf16 grid at |v| (8 significant bits: 2^(floor(log2|v|) - 7)).

    frexp gives |v| = m * 2^e with m in [0.5, 1), so floor(log2|v|) = e - 1.
    Zero maps to the smallest normal's spacing rather than 0, so a 0/0 margin
    cannot appear.
    """
    _, e = torch.frexp(v.abs().float())
    ulp = torch.ldexp(torch.ones_like(v, dtype=torch.float32), (e - 8).to(torch.int32))
    tiny = torch.finfo(torch.bfloat16).tiny * 2.0 ** -7
    return torch.where(v == 0, torch.full_like(ulp, tiny), ulp)


def router_stats(logits32: torch.Tensor, logits16: torch.Tensor, top_k: int = 8) -> torch.Tensor:
    """Per-position router statistics, [M, len(STAT_COLS)] float32.

    logits32: the gate projection computed in fp32 (the reference).
    logits16: the same projection in bf16, as production computes it.
    Both come from the same hidden states, so any top-k difference is due to
    the gate's rounding alone.

    Flatness (entropy, top-1 weight, margin) is measured on the fp32 logits.
    The flip is computed exactly as the engine selects experts: topk over an
    fp32 softmax of the given logits.
    """
    l32 = logits32.float()
    n_exp = l32.shape[-1]
    p32 = F.softmax(l32, dim=-1)
    entropy = -(p32 * torch.log(p32.clamp_min(1e-30))).sum(-1) / math.log(n_exp)
    top1 = p32.max(dim=-1).values

    top_vals = torch.topk(l32, top_k + 1, dim=-1).values
    kth, k1th = top_vals[:, top_k - 1], top_vals[:, top_k]
    margin = (kth - k1th) / bf16_ulp(kth)

    ids32 = torch.topk(p32, top_k, dim=-1).indices.sort(dim=-1).values
    p16 = F.softmax(logits16, dim=-1, dtype=torch.float32)
    ids16 = torch.topk(p16, top_k, dim=-1).indices.sort(dim=-1).values
    flip = (ids16 != ids32).any(dim=-1)
    flip1 = p16.argmax(-1) != p32.argmax(-1)

    v16 = torch.topk(logits16.float(), top_k + 1, dim=-1).values
    tie = v16[:, top_k - 1] == v16[:, top_k]

    edges = torch.tensor(MARGIN_EDGES_ULP[1:-1], device=l32.device, dtype=torch.float32)
    bins = torch.bucketize(margin, edges, right=True)
    hist = F.one_hot(bins, len(MARGIN_EDGES_ULP) - 1).float()

    cols = [torch.ones_like(top1), entropy, top1, margin.clamp(max=64.0),
            flip.float(), flip1.float(), tie.float()]
    return torch.cat([torch.stack(cols, dim=-1), hist], dim=-1)


def topk_flips(logits16: torch.Tensor, logits32: torch.Tensor, top_k: int) -> torch.Tensor:
    """[M] bool: top-k expert set differs between bf16 and fp32 logits,
    selected exactly as the engine selects (topk over an fp32 softmax)."""
    a = torch.topk(F.softmax(logits16, dim=-1, dtype=torch.float32), top_k, dim=-1).indices.sort(-1).values
    b = torch.topk(F.softmax(logits32.float(), dim=-1), top_k, dim=-1).indices.sort(-1).values
    return (a != b).any(-1)


# ── Answers ─────────────────────────────────────────────────────────────────

def canonical_number(s: Optional[str]) -> Optional[str]:
    """'18', '18.0', '18.00' -> '18'; non-numbers -> None."""
    if s is None:
        return None
    try:
        v = float(s.replace(",", ""))
    except ValueError:
        return None
    if math.isfinite(v) and v == int(v):
        return str(int(v))
    return repr(round(v, 6))


def gsm8k_final_answer(raw: str) -> Optional[str]:
    """The number the repo's GSM8K grader compares against the reference.

    Mirrors _grade_gsm8k's precedence exactly (extracted marker line, then
    last \\boxed{}, then last number anywhere), so outcome churn and
    correctness churn are measured on the same extracted quantity.
    """
    from benchmarks.correctness.run_math_reasoning_code import extract_final_answer

    predicted = extract_final_answer(raw)
    if predicted is not None:
        nums = re.findall(r"-?\d+\.?\d*", predicted.replace(",", ""))
        if nums:
            return canonical_number(nums[-1])
    boxed = re.findall(r"\\boxed\{?\s*(-?\d[\d,]*\.?\d*)", raw)
    if boxed:
        return canonical_number(boxed[-1])
    nums = re.findall(r"-?\d+\.?\d*", raw.replace(",", ""))
    if nums:
        return canonical_number(nums[-1])
    return None


_MARKERS = (re.compile(r"final\s*answer\s*[:\-]\s*(.+)", re.IGNORECASE),
            re.compile(r"the\s+answer\s+is\s*[:\-]?\s*(.+)", re.IGNORECASE))


def gsm8k_strict_answer(raw: str) -> Optional[str]:
    """The number on the model's explicit answer line, or None.

    Same markers and precedence as extract_final_answer ('Final Answer:'
    first, then 'The answer is'), first occurrence, but with no fallbacks: a
    response without an answer line, or whose answer line has no number,
    yields None. A truncated generation then counts as 'no answer' instead
    of donating whatever number it was halfway through writing.
    """
    text = raw.strip()
    for pat in _MARKERS:
        m = pat.search(text)
        if m:
            line = m.group(1).strip().splitlines()[0]
            nums = re.findall(r"-?\d+\.?\d*", line.replace(",", ""))
            return canonical_number(nums[-1]) if nums else None
    return None


_MARKERS_V2 = (re.compile(r"final\s*answer\s*(?:is)?\s*[:\-]?\s*(.+)", re.IGNORECASE),
               re.compile(r"\banswer\s*(?:is\s*[:\-]?|[:\-])\s*(.+)", re.IGNORECASE))


def gsm8k_strict_answer_v2(raw: str) -> Optional[str]:
    """Strict extraction with the answer-line phrasings v1 missed.

    v1 (gsm8k_strict_answer) inherited the repo's two markers, 'Final
    Answer:' and 'The answer is', and so read 'The final answer is **375**'
    or '**Answer: 120**' as no answer at all -- in LLaDA-MoE's outputs that
    made about half of all v1 'answer changes' pure rephrasings of the same
    correct number. v2 takes the earliest line matching either a 'final
    answer (is)' or an 'answer is / answer:' marker; still no fallback to a
    number outside the answer line.
    """
    text = raw.strip()
    best = None
    for pat in _MARKERS_V2:
        m = pat.search(text)
        if m and (best is None or m.start() < best.start()):
            best = m
    if best is None:
        return None
    line = best.group(1).strip().splitlines()[0]
    nums = re.findall(r"-?\d+\.?\d*", line.replace(",", ""))
    return canonical_number(nums[-1]) if nums else None


_MARKER_V3 = re.compile(
    r"final\s*answer\b(?:\*\*)?\s*(?:is\b)?\s*[:\-]?"          # Final Answer: / The final answer is / **Final answer:**
    r"|\banswer\b(?:\*\*)?\s*(?:is\b\s*[:\-]?|[:\-])"            # The answer is / Answer: / **Answer**: / answer is:
    r"|\\boxed\{",                                               # \boxed{N}
    re.IGNORECASE)
_NUM = re.compile(r"-?\d+\.?\d*")


def gsm8k_strict_answer_v3(raw: str) -> Optional[str]:
    """Strict extraction covering every answer format found in this study's
    outputs (survey of all ~27k generations, see the routing-fragility doc):
    'Final Answer:', 'The final answer is', 'The answer is', 'Answer:' /
    '**Answer**:' / '**Answer: N**' (bold anywhere on the line), and
    '\\boxed{N}'.

    Markers are scanned in text order; the first one that yields a number
    wins. The number is the last one on the marker's own line (the grader's
    rule), or, for a header-style marker with nothing after it ('**Answer:**'
    then the value on the next line), the last number on the next non-empty
    line. v1 and v2 stopped at the first marker even when it carried no
    number, so an early 'the answer is not obvious' or a bare '**Answer:**'
    header hid the real answer line. Still no fallback to a number outside an
    answer line: a response without one yields None.
    """
    text = raw.strip()
    for m in _MARKER_V3.finditer(text):
        rest = text[m.end():]
        lines = rest.split("\n")
        if m.group(0).lower().startswith("\\boxed"):
            body = re.match(r"\s*\$?\s*(-?[\d,]*\.?\d+)", rest)
            if body:
                return canonical_number(body.group(1).replace(",", ""))
            continue
        nums = _NUM.findall(lines[0].replace(",", ""))
        if not nums:
            nxt = next((l for l in lines[1:] if l.strip() and l.strip() not in ("$$", "\\[", "\\]")), "")
            if len(lines[0].strip(" *:$")) == 0:          # header-style marker, value on the next line
                nums = _NUM.findall(nxt.replace(",", ""))
        if nums:
            return canonical_number(nums[-1])
    return None


def first_divergence(out_a: list, commit_a: list, out_b: list, commit_b: list) -> Optional[int]:
    """Earliest global denoising step at which two trajectories committed
    something different, or None if every commit event matches.

    A trajectory is the set of commit events (step, position, token). An event
    of A is missing from B exactly when B committed that position at another
    step or with another token, so the first differing event is the minimum,
    over mismatching positions, of the earlier of the two commit steps.
    Divergence that changes hidden states but never a commit is invisible
    here by design.
    """
    best = None
    for ta, ca, tb, cb in zip(out_a, commit_a, out_b, commit_b):
        if ta != tb or ca != cb:
            s = min(ca, cb)
            if best is None or s < best:
                best = s
    return best


def max_repeat_run(token_ids: Iterable[int]) -> int:
    """Longest run of one token id repeated back to back."""
    best = run = 0
    prev = None
    for t in token_ids:
        run = run + 1 if t == prev else 1
        prev = t
        best = max(best, run)
    return best


#: A generation counts as decoding collapse (h100x2_bench.md §4d) when one
#: token repeats this many times in a row. The §4d failures repeated a token
#: hundreds of times; normal arithmetic text rarely repeats a token more than
#: 3-4 times ("0000"), so 12 sits well clear of both.
COLLAPSE_RUN = 12


# ── Paired statistics ───────────────────────────────────────────────────────

def wilson_ci(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """95% Wilson score interval for a proportion k/n."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p: binomial test of b vs c discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def paired_compare(rows_a: dict, rows_b: dict, ids: Optional[Iterable[int]] = None,
                   answer_key: str = "answer") -> dict:
    """Paired comparison of two arms over their shared question ids.

    rows_*: {idx: row} with row[answer_key] (canonical str or None) and
    row["correct"] (bool). ids restricts the comparison to a subset (e.g. the
    questions uncapped in control). Returns counts, churn with Wilson CIs,
    accuracies and exact McNemar p on the correctness discordants (b = only A
    correct, c = only B correct).
    """
    shared = set(rows_a) & set(rows_b)
    if ids is not None:
        shared &= set(ids)
    shared = sorted(shared)
    n = len(shared)
    out_diff = sum(rows_a[i][answer_key] != rows_b[i][answer_key] for i in shared)
    b = sum(rows_a[i]["correct"] and not rows_b[i]["correct"] for i in shared)
    c = sum(rows_b[i]["correct"] and not rows_a[i]["correct"] for i in shared)
    acc_a = sum(rows_a[i]["correct"] for i in shared)
    acc_b = sum(rows_b[i]["correct"] for i in shared)
    tok = "out" if all("out" in rows_a[i] and "out" in rows_b[i] for i in shared) else "tokens"
    text_diff = sum(rows_a[i].get(tok) != rows_b[i].get(tok) for i in shared)
    return {
        "n": n,
        "outcome_churn": out_diff / n if n else float("nan"),
        "outcome_churn_ci": wilson_ci(out_diff, n),
        "outcome_diff": out_diff,
        "correct_churn": (b + c) / n if n else float("nan"),
        "correct_churn_ci": wilson_ci(b + c, n),
        "only_a": b,
        "only_b": c,
        "acc_a": acc_a / n if n else float("nan"),
        "acc_b": acc_b / n if n else float("nan"),
        "acc_a_ci": wilson_ci(acc_a, n),
        "acc_b_ci": wilson_ci(acc_b, n),
        "mcnemar_p": mcnemar_exact(b, c),
        "token_diff": text_diff,
    }
