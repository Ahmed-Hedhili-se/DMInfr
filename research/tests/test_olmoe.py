"""CPU checks for the OLMoE router hooks (tiny random OlmoeConfig, bf16).

    python -m research.tests.test_olmoe

Pinned: the control hook is inert; fp32_all reproduces a model whose gate is
genuinely computed in fp32; fp32_prefill touches only the first forward of
generate(); noise is seeded and bounded; telemetry counts exactly the
non-padding prompt tokens (prefill) and the still-running rows (decode); the
runner writes one row per question and a repeated control matches exactly.
"""

import os
import sys
import tempfile

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from transformers import OlmoeConfig, OlmoeForCausalLM

from research import metrics
from research.olmoe import OlmoeProbe, run_arms
from research.runner import _read_jsonl

CFG = OlmoeConfig(vocab_size=300, hidden_size=64, intermediate_size=32, num_hidden_layers=3,
                  num_attention_heads=4, num_key_value_heads=4, num_experts=16, num_experts_per_tok=4,
                  norm_topk_prob=False, pad_token_id=1, eos_token_id=2, bos_token_id=2,
                  max_position_embeddings=256)


def _model():
    torch.manual_seed(0)
    m = OlmoeForCausalLM(CFG).to(torch.bfloat16).eval()
    with torch.no_grad():
        # near-uniform router at a large logit magnitude, like the real ones:
        # every expert row shares one direction plus a small spread, so the
        # logits sit in a band a few bf16 ULPs wide
        for layer in m.model.layers:
            w = layer.mlp.gate.weight
            common = torch.randn(1, w.shape[1]) * 0.5
            w.copy_(common + 0.02 * torch.randn_like(w.float()))
    return m


def _inputs():
    g = torch.Generator().manual_seed(1)
    a = torch.randint(3, 300, (7,), generator=g).tolist()
    b = torch.randint(3, 300, (4,), generator=g).tolist()
    ids = torch.tensor([a, [1] * 3 + b])
    am = torch.tensor([[1] * 7, [0] * 3 + [1] * 4])
    return ids, am


def _gen(m, arm=None, telemetry=False, n=12):
    ids, am = _inputs()
    p = None
    if arm is not None:
        p = OlmoeProbe(CFG.num_hidden_layers, arm, telemetry=telemetry).install(m)
        p.begin_batch(0)
    try:
        out = m.generate(input_ids=ids, attention_mask=am, max_new_tokens=n, do_sample=False,
                         pad_token_id=1, eos_token_id=2)
    finally:
        if p is not None:
            p.uninstall()
    return out, p


def test_control_inert():
    m = _model()
    base, _ = _gen(m)
    assert torch.equal(_gen(m, "control")[0], base)
    assert torch.equal(_gen(m, "control_tel", telemetry=True)[0], base)
    assert torch.equal(_gen(m)[0], base), "hooks not removed"


def test_fp32_all_matches_true_fp32_gate():
    m = _model()
    hooked, _ = _gen(m, "fp32_all")
    # reference: the gate weight kept in fp32 and fed fp32 input
    import types
    origs = []
    for layer in m.model.layers:
        g = layer.mlp.gate
        origs.append(g.forward)
        g.forward = types.MethodType(lambda self, x: torch.nn.functional.linear(x.float(), self.weight.float()), g)
    try:
        ref, _ = _gen(m)
    finally:
        for layer, f in zip(m.model.layers, origs):
            layer.mlp.gate.forward = f
    assert torch.equal(hooked, ref)
    # a random tiny model's argmax tokens are robust, so check the forward
    # logits and the routing itself rather than the generated tokens
    ids, am = _inputs()
    with torch.no_grad():
        base_logits = m(input_ids=ids, attention_mask=am).logits
        p = OlmoeProbe(CFG.num_hidden_layers, "fp32_all").install(m)
        try:
            f32_logits = m(input_ids=ids, attention_mask=am).logits
        finally:
            p.uninstall()
    assert not torch.equal(base_logits, f32_logits), "fp32 gate did not change the forward"
    _, tel = _gen(m, "control_tel", telemetry=True)
    assert tel.acc[..., metrics.STAT_COLS.index("flip_top8")].sum() > 0, "no bf16/fp32 flips in tiny model"


def test_prefill_only_and_counts():
    m = _model()
    out, p = _gen(m, "fp32_prefill")
    assert p.counts["prefill_forwards"] == 1
    assert p.counts["rows_fp32"] == 2 * 7                      # only the prefill rows (B x W)
    assert p.counts["forwards"] > 1


def test_single_decode_step():
    m = _model()
    out, p = _gen(m, "fp32_dec0")
    assert p.counts["rows_fp32"] == 2, p.counts                # one decode forward, 2 rows x 1 token
    assert p.injected is not None and p.injected.shape == (2,)
    out, p = _gen(m, "fp32_dec3", n=12)
    assert p.counts["rows_fp32"] == 2
    out, p = _gen(m, "fp32_dec50", n=12)                       # never reached
    assert p.counts["rows_fp32"] == 0 and p.injected is None
    base, _ = _gen(m, n=12)
    assert torch.equal(out, base)


def test_noise_seeded_bounded():
    m = _model()
    a, _ = _gen(m, "noise_s0")
    assert torch.equal(a, _gen(m, "noise_s0")[0])
    p = OlmoeProbe(CFG.num_hidden_layers, "noise_s3")
    l16 = (torch.randn(256, 16) * 0.05 + 2.0).to(torch.bfloat16)
    d = (p._noisy(l16) - l16.float()).abs()
    assert (d <= metrics.bf16_ulp(l16)).all() and d.max() > 0


def test_telemetry_counts_valid_tokens():
    m = _model()
    out, p = _gen(m, "control_tel", telemetry=True, n=6)
    cnt = p.acc[..., metrics.STAT_COLS.index("count")]
    assert torch.all(cnt[0] == 7 + 4), cnt[0]                  # prompt tokens, pads excluded
    gen = out[:, 7:]                                           # both prompts left-padded to W = 7
    # decode forwards see the previous step's token for each still-running row
    running = 0
    for t in range(gen.shape[1] - 1):
        running += int((gen[:, t] != 1).sum())
    assert torch.all(cnt[1] == running), (cnt[1], running)


class _Tok:
    pad_token_id, eos_token_id = 1, 2

    def apply_chat_template(self, msgs, add_generation_prompt, tokenize):
        return msgs[-1]["content"]

    def __call__(self, text, add_special_tokens=True):
        return {"input_ids": [3 + (ord(c) % 200) for c in text]}

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(i % 10) for i in ids)


def test_runner_rows_and_self_churn():
    m = _model()
    qs = [{"idx": i, "prompt": "q" * (3 + i % 4) + str(i), "expected": "1", "qhash": str(i)} for i in range(5)]
    with tempfile.TemporaryDirectory() as d:
        run_arms(m, _Tok(), qs, ["control", "control_rep", "fp32_all"], d, 2, "sys", max_new_tokens=8,
                 grade=lambda raw, exp: False)
        a = _read_jsonl(os.path.join(d, "control.jsonl"))
        b = _read_jsonl(os.path.join(d, "control_rep.jsonl"))
        assert len(a) == 5 and all(a[i]["out"] == b[i]["out"] for i in a)
        assert {"answer", "answer_strict", "capped", "out", "prompt_tokens"} <= set(a[0])


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"[pass] {t.__name__}")
    print(f"{len(tests)}/{len(tests)} passed")


if __name__ == "__main__":
    main()
