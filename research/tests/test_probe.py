"""CPU checks for the routing-fragility instrumentation (no GPU, no weights).

    LLADA_RESEARCH_HOOKS=1 python -m research.tests.test_probe

A tiny random-init LLaDA-MoE (eager MoE, bf16 on CPU) stands in for the 7B
model. What is pinned:
  * the hooks cannot be installed without the env gate, and the engine with
    hooks removed is bit-identical to the engine never hooked;
  * telemetry is inert: a control arm with telemetry on produces the same
    tokens as with no probe at all;
  * the probe's "all" window reproduces LLADA_FP32_ROUTER=1 bit-exactly;
  * early/late windows partition every forward pass, for both window kinds;
  * position-type counts match the decode schedule exactly;
  * metric definitions: bf16 ULP against the real bf16 grid, McNemar against
    the two p-values already published in docs/h100x2_bench.md §9;
  * the runner writes one row per question, resumes without redoing work,
    and a control run twice has zero self-churn.
"""

import json
import os
import sys
import tempfile

os.environ.setdefault("LLADA_RESEARCH_HOOKS", "1")

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from research.tests import _triton_stub
_triton_stub.install()

import dminfr.engine.model as engine_model
from dminfr.engine.generate import MASK_ID, generate_cached, set_research_hooks
from dminfr.engine.model import Cfg, LLaDAMoEKV
from research import metrics
from research.probe import RoutingProbe, TYPES, Window
from research.runner import Arm, GenConfig, _read_jsonl, run_arms

TINY = Cfg(H=64, NH=4, KVH=4, NL=3, NE=16, TOPK=4, EI=32, VS=157184)
GEN = dict(gen_length=16, steps=16, block_length=4, temperature=0.0)


def _model(dtype=torch.bfloat16):
    torch.manual_seed(0)
    m = LLaDAMoEKV(TINY, use_fused_moe=False).to(dtype).eval()
    with torch.no_grad():
        for layer in m.layers:   # sharpen nothing: keep the router near-uniform, like the real one
            layer.mlp.gate.weight.mul_(0.5)
    return m


def _prompts():
    g = torch.Generator().manual_seed(1)
    short = torch.randint(0, 1000, (1, 5), generator=g)
    long_ = torch.randint(0, 1000, (1, 9), generator=g)
    ids = torch.cat([torch.cat([torch.zeros(1, 4, dtype=torch.long), short], 1), long_], 0)
    return ids, torch.tensor([5, 9])


def _gen(model, probe=None, **kw):
    ids, lens = _prompts()
    if probe is not None:
        probe.install(model)
    try:
        return generate_cached(model, ids.clone(), prompt_lens=lens, **{**GEN, **kw})
    finally:
        RoutingProbe.uninstall()


def test_gate_refuses_without_env():
    old = os.environ.pop("LLADA_RESEARCH_HOOKS")
    try:
        set_research_hooks(router=lambda g, x: g(x))
        raise AssertionError("hooks installed without LLADA_RESEARCH_HOOKS")
    except RuntimeError:
        pass
    finally:
        os.environ["LLADA_RESEARCH_HOOKS"] = old
    set_research_hooks()   # removing is always allowed


def test_unhooked_and_telemetry_are_inert():
    m = _model()
    base = _gen(m)
    assert engine_model._ROUTER_HOOK is None
    assert torch.equal(_gen(m, RoutingProbe(TINY.NL)), base), "probe with no window/telemetry changed output"
    assert torch.equal(_gen(m, RoutingProbe(TINY.NL, telemetry=True)), base), "telemetry changed output"
    assert torch.equal(_gen(m), base), "engine not restored after uninstall"


def test_all_window_matches_fp32_router_flag():
    m = _model()
    via_probe = _gen(m, RoutingProbe(TINY.NL, window=Window.parse("all")))
    engine_model.FP32_ROUTER = True
    try:
        via_flag = _gen(m)
    finally:
        engine_model.FP32_ROUTER = False
    assert torch.equal(via_probe, via_flag), "window 'all' != LLADA_FP32_ROUTER=1"
    assert not torch.equal(via_probe, _gen(m)), "fp32 router changed nothing -- tiny model too peaked to test"


def test_windows_partition_passes():
    m = _model()
    for kind in ("within", "global"):
        full = RoutingProbe(TINY.NL, window=Window.parse("all"))
        early = RoutingProbe(TINY.NL, window=Window.parse(f"{kind}:0:0.25"))
        late = RoutingProbe(TINY.NL, window=Window.parse(f"{kind}:0.25:1"))
        for p in (full, early, late):
            _gen(m, p)
        for key in ("passes_fp32", "rows_fp32", "step_passes_fp32"):
            assert early.counts[key] + late.counts[key] == full.counts[key], (kind, key)
        assert full.counts["passes_fp32"] == full.counts["passes"]
        assert early.counts["passes_fp32"] > 0 and late.counts["passes_fp32"] > 0, kind
    # 4 blocks x 4 steps: within-block [0,.25) = step 0 of each block + prime
    p = RoutingProbe(TINY.NL, window=Window.parse("within:0:0.25"))
    _gen(m, p)
    assert p.counts["step_passes_fp32"] == 4 and p.counts["passes_fp32"] == 5, p.counts
    # global [0,.25) = block 0 (4 steps + finalize) + prime
    p = RoutingProbe(TINY.NL, window=Window.parse("global:0:0.25"))
    _gen(m, p)
    assert p.counts["step_passes_fp32"] == 4 and p.counts["passes_fp32"] == 6, p.counts


def test_cache_and_step_windows_partition_passes():
    m = _model()
    full = RoutingProbe(TINY.NL, window=Window.parse("all"))
    _gen(m, full)
    for kind in ("steps_within", "steps_global"):
        parts = [RoutingProbe(TINY.NL, window=Window.parse(w))
                 for w in ("prime", "finalize", f"{kind}:0:0.25", f"{kind}:0.25:1")]
        for p in parts:
            _gen(m, p)
        for key in ("passes_fp32", "rows_fp32", "step_passes_fp32"):
            assert sum(p.counts[key] for p in parts) == full.counts[key], (kind, key)
        prime, fin, early, late = parts
        assert prime.counts["passes_fp32"] == 1 and prime.counts["step_passes_fp32"] == 0
        assert fin.counts["passes_fp32"] == 4 and fin.counts["step_passes_fp32"] == 0
        assert early.counts["passes_fp32"] == early.counts["step_passes_fp32"] == 4, early.counts
        assert late.counts["passes_fp32"] == late.counts["step_passes_fp32"] == 12, late.counts


def test_layer_restricted_fp32():
    m = _model()
    full = _gen(m, RoutingProbe(TINY.NL, window=Window.parse("all")))
    assert torch.equal(_gen(m, RoutingProbe(TINY.NL, window=Window("all"), layers=(0, TINY.NL - 1))), full)
    base = _gen(m)
    outs = [_gen(m, RoutingProbe(TINY.NL, window=Window("all"), layers=(l, l))) for l in range(TINY.NL)]
    assert any(not torch.equal(o, base) for o in outs), "no single-layer fp32 arm changed anything"
    os.environ["LLADA_FP32_ROUTER_LAYERS"] = "1-2"
    try:
        p = RoutingProbe.from_env(TINY.NL)
        assert p.layers == frozenset({1, 2}) and p.window.kind == "all"
    finally:
        del os.environ["LLADA_FP32_ROUTER_LAYERS"]


def test_single_pass_window():
    m = _model()
    base = _gen(m)
    p = RoutingProbe(TINY.NL, window=Window.parse("single:1:2"))
    out = _gen(m, p)
    assert p.counts["passes_fp32"] == 1 and p.counts["step_passes_fp32"] == 1, p.counts
    inj = p.injected_flips()
    assert inj is not None and inj.shape == (2,) and (inj >= 0).all()
    # equal to fp32 via an all-layers window restricted to that pass's coordinates? check against a
    # manual reference: the same pass in a steps_within window that only contains step 2, block 1
    q = RoutingProbe(TINY.NL, window=Window.parse("single:0:16"))   # step 16 does not exist here (4 steps/block)
    assert torch.equal(_gen(m, q), base) and q.counts["passes_fp32"] == 0 and q.injected_flips() is None
    assert str(Window.parse("single:0:16")) == "single:0:16"


def test_noise_arm_seeded_and_bounded():
    m = _model()
    base = _gen(m)
    a = _gen(m, RoutingProbe(TINY.NL, noise_seed=0))
    assert torch.equal(a, _gen(m, RoutingProbe(TINY.NL, noise_seed=0))), "same seed not reproducible"
    b = _gen(m, RoutingProbe(TINY.NL, noise_seed=1))
    assert not (torch.equal(a, base) and torch.equal(b, base)), "noise changed nothing"
    p = RoutingProbe(TINY.NL, noise_seed=3)
    l16 = (torch.randn(512, 16) * 0.05 + 2.0).to(torch.bfloat16)
    d = (p._noisy(l16) - l16.float()).abs()
    assert (d <= metrics.bf16_ulp(l16)).all() and d.max() > 0.5 * metrics.bf16_ulp(l16).min()
    try:
        RoutingProbe(TINY.NL, window=Window.parse("all"), noise_seed=0)
        raise AssertionError("window + noise accepted")
    except ValueError:
        pass


def test_commit_steps_fixed_and_threshold():
    m = _model()
    p = RoutingProbe(TINY.NL)
    _gen(m, p)
    c = p.commit_steps()                         # [2, 16]; 4 blocks x 4 steps, 1 reveal/step/row
    assert (c >= 0).all(), c
    for blk in range(4):
        blkc = c[:, blk * 4:(blk + 1) * 4]
        for row in blkc:
            assert sorted(row.tolist()) == list(range(blk * 4, blk * 4 + 4)), (blk, row)
    p = RoutingProbe(TINY.NL)
    _gen(m, p, confidence_threshold=0.9, low_confidence_threshold=0.4)
    c = p.commit_steps()
    assert (c >= 0).all()
    conf = p.commit_confidence()
    assert not torch.isnan(conf).any() and (conf > 0).all() and (conf <= 1).all(), conf
    for blk in range(4):
        blkc = c[:, blk * 4:(blk + 1) * 4]
        assert ((blkc >= blk * 4) & (blkc < blk * 4 + 4)).all(), blkc


def test_first_divergence():
    assert metrics.first_divergence([1, 2, 3], [0, 1, 2], [1, 2, 3], [0, 1, 2]) is None
    assert metrics.first_divergence([1, 2, 3], [0, 1, 2], [1, 9, 3], [0, 1, 2]) == 1
    assert metrics.first_divergence([1, 2, 3], [0, 3, 2], [1, 2, 3], [0, 1, 2]) == 1   # same token, other step
    assert metrics.first_divergence([5, 2, 3], [4, 1, 2], [1, 2, 3], [0, 1, 2]) == 0


def test_strict_answer():
    S = metrics.gsm8k_strict_answer
    assert S("so 7. The answer is **7**.") == "7"
    assert S("Final Answer: $1,250.00") == "1250"
    assert S("Final Answer:\n\\[\n\\boxed{16\n\\]") is None          # no number on the answer line
    assert S("blah 3 + 4 = 7") is None                                       # no answer line, no fallback
    assert S("The answer is 12.\nQuestion: x\nThe answer is 5") == "12"   # first answer line wins
    assert metrics.gsm8k_final_answer("blah 3 + 4 = 7") == "7"               # lenient one still falls back
    # v3: every format observed in this study's outputs; first marker that yields a number
    V3 = metrics.gsm8k_strict_answer_v3
    for raw, want in [("The final answer is **375**.", "375"), ("**Answer: 120**", "120"),
                      ("**Answer:**\n80 dollars", "80"), ("**Answer**: $45", "45"),
                      ("the answer is not obvious. ... The answer is 12.", "12"),
                      ("$ \\boxed{16} $", "16"), ("Final Answer: $1,250.00", "1250"),
                      ("There are **9 blue golf balls**.", None), ("blah 3 + 4 = 7", None),
                      ("The answer is 12.\nQuestion: x\nThe answer is 5", "12")]:
        assert V3(raw) == want, (raw, V3(raw), want)


def test_telemetry_position_types():
    m = _model()
    p = RoutingProbe(TINY.NL, telemetry=True)
    _gen(m, p)
    t = p.telemetry_state()
    acc, cols = t["acc"], t["stat_cols"]
    cnt = acc[..., cols.index("count")]          # [blocks+1, steps+1, L, 4]
    B, P, gl, bl = 2, 9, 16, 4
    prime = cnt[0, 0, 0]
    assert prime[TYPES.index("prompt")] == 5 + 9, prime          # pads excluded
    assert prime[TYPES.index("future_masked")] == B * gl
    for blk in range(4):
        for s in range(4):   # fixed schedule, 1 reveal per step per row
            c = cnt[blk + 1, s, 0]
            assert c[TYPES.index("active_masked")] == B * (bl - s), (blk, s, c)
            assert c[TYPES.index("active_unmasked")] == B * s
            assert c[TYPES.index("future_masked")] == B * (gl - (blk + 1) * bl)
        fin = cnt[blk + 1, 4, 0]
        assert fin[TYPES.index("active_unmasked")] == B * bl and fin[TYPES.index("active_masked")] == 0
    # identical across layers
    assert torch.equal(cnt[..., 0, :], cnt[..., 2, :])
    # stat sanity: entropy in (0,1], top1 >= 1/E, flips present in bf16
    n = cnt.sum()
    ent = acc[..., cols.index("entropy_norm")].sum() / n
    top1 = acc[..., cols.index("top1_weight")].sum() / n
    assert 0.5 < ent <= 1.0 and top1 >= 1 / TINY.NE, (ent, top1)
    hist = acc[..., cols.index("hist_0"):].sum(-1)
    assert torch.equal(hist, cnt), "margin histogram does not cover every position"
    mask = t["mask_acc"]
    assert mask[1, 0, 1] == B * bl and mask[1, 3, 1] == B * 1 and mask[1, 0, 2] == B * gl


def test_bf16_ulp_matches_grid():
    v = torch.tensor([1.0, 1.5, 3.0, -3.0, 0.3, 17.0, 1e-3])
    ulp = metrics.bf16_ulp(v)
    vb = v.abs().to(torch.bfloat16)
    nxt = torch.nextafter(vb, torch.tensor(float("inf"), dtype=torch.bfloat16))
    assert torch.equal(ulp, (nxt.float() - vb.float())), (ulp, nxt.float() - vb.float())


def test_router_stats_margin_and_flip():
    torch.manual_seed(0)
    l32 = torch.randn(4096, 64) * 0.05 + 3.0       # near-uniform, magnitude ~3
    l16 = l32.to(torch.bfloat16)
    s = metrics.router_stats(l32, l16)
    c = metrics.STAT_COLS
    flip = s[:, c.index("flip_top8")].bool()
    margin = s[:, c.index("margin_ulp_clip64")]
    # rounding moves each logit by <= 0.5 ULP, so a gap >= 1 ULP cannot invert
    assert not flip[margin >= 1.0].any(), "flip with margin >= 1 ULP"
    assert flip.any() and (margin < 1.0).any()


def test_mcnemar_reproduces_published_p():
    assert abs(metrics.mcnemar_exact(23, 17) - 0.43) < 0.005, metrics.mcnemar_exact(23, 17)   # §9b
    assert abs(metrics.mcnemar_exact(86, 81) - 0.757) < 0.0005, metrics.mcnemar_exact(86, 81)  # §9e
    lo, hi = metrics.wilson_ci(40, 200)
    assert abs(lo - 0.1508) < 1e-3 and abs(hi - 0.2600) < 1e-3, (lo, hi)


def test_final_answer_agrees_with_grader():
    from benchmarks.correctness.run_math_reasoning_code import _grade_gsm8k, extract_final_answer
    cases = [("so 3+4 = 7. The answer is 7.", "7"), ("The answer is $1,250.00", "1250"),
             ("Final Answer:\n\\[\n\\boxed{16\n\\]", "16"), ("no digits here", "3"),
             ("x is 5 and then 12", "12"), ("The answer is 0.5", "0.5")]
    for raw, exp in cases:
        graded = bool(_grade_gsm8k(extract_final_answer(raw), exp, raw))
        assert graded == (metrics.gsm8k_final_answer(raw) == metrics.canonical_number(exp)), raw


class _Tok:
    eos_token_id = 7
    pad_token_id = 0

    def apply_chat_template(self, msgs, add_generation_prompt, tokenize):
        return msgs[-1]["content"]

    def __call__(self, text):
        return {"input_ids": [100 + (ord(ch) % 50) for ch in text]}

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(i % 10) for i in ids)


def test_runner_resume_and_self_churn():
    m = _model()
    qs = [{"idx": i, "prompt": "q" * (3 + i % 4) + str(i), "expected": "1", "qhash": str(i)} for i in range(7)]
    gen = GenConfig(max_tokens=16, block_length=4, steps=16)
    grade = lambda raw, exp: raw.split()[-1:] == [exp] if raw else False
    with tempfile.TemporaryDirectory() as d:
        calls = []
        stop_after = lambda: len(calls) >= 2
        run_arms(m, _Tok(), qs, [Arm("control")], gen, d, 3, "sys",
                 on_batch=lambda: calls.append(1), should_stop=stop_after, grade=grade)
        part = _read_jsonl(os.path.join(d, "control.jsonl"))
        assert len(part) == 6, len(part)               # 2 of 3 batches before the stop
        run_arms(m, _Tok(), qs, [Arm("control"), Arm("control_rep"),
                                 Arm("fp32", {"LLADA_FP32_ROUTER_WINDOW": "all",
                                              "LLADA_ROUTER_TELEMETRY": "1"})],
                 gen, d, 3, "sys", grade=grade)
        a = _read_jsonl(os.path.join(d, "control.jsonl"))
        b = _read_jsonl(os.path.join(d, "control_rep.jsonl"))
        with open(os.path.join(d, "control.jsonl")) as f:
            assert sum(1 for _ in f) == 7, "resume re-ran finished batches"
        cmp = metrics.paired_compare(a, b)
        assert cmp["n"] == 7 and cmp["outcome_diff"] == 0 and cmp["token_diff"] == 0, cmp
        f32 = _read_jsonl(os.path.join(d, "fp32.jsonl"))
        assert metrics.paired_compare(a, f32)["token_diff"] > 0
        assert os.path.exists(os.path.join(d, "fp32.telemetry.seg0.pt"))
        assert "LLADA_FP32_ROUTER_WINDOW" not in os.environ, "arm env leaked"
        row = next(iter(a.values()))
        assert {"answer", "answer_strict", "correct", "tokens", "out", "commit", "hit_eos",
                "collapse", "batch"} <= set(row)
        assert len(row["out"]) == len(row["commit"]) == 16 and min(row["commit"]) >= 0
        for i in a:   # identical arms: no divergence; fp32 arm: divergence wherever outputs differ
            assert metrics.first_divergence(a[i]["out"], a[i]["commit"], b[i]["out"], b[i]["commit"]) is None
            div = metrics.first_divergence(a[i]["out"], a[i]["commit"], f32[i]["out"], f32[i]["commit"])
            assert (div is None) == (a[i]["out"] == f32[i]["out"] and a[i]["commit"] == f32[i]["commit"])

        run_arms(m, _Tok(), qs, [Arm("sentinel", max_batches=1)], gen, d, 3, "sys", grade=grade)
        assert len(_read_jsonl(os.path.join(d, "sentinel.jsonl"))) == 3


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"[pass] {t.__name__}")
    print(f"{len(tests)}/{len(tests)} passed")


if __name__ == "__main__":
    main()
