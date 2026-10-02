"""SQ3: the same router perturbations on an autoregressive MoE.

allenai/OLMoE-1B-7B-0924-Instruct through HF transformers (4.53.2), greedy.
Its MoE block routes exactly like LLaDA-MoE's (config.json: 64 experts, top-8,
norm_topk_prob=false; modeling_olmoe.OlmoeSparseMoeBlock: bf16 gate Linear ->
fp32 softmax -> topk), so the perturbation is applied the same way: a forward
hook on every layer's `mlp.gate` replaces its output (documented choice: a
hook, not a patch -- HF's code is untouched).

Arms (env-free; the hook is installed by this module only):
  control        hook returns the gate's own bf16 output (bit-identical)
  fp32_all       gate recomputed in fp32 on the same input, every forward
  fp32_prefill   fp32 only on the prefill forward (the whole prompt; it fills
                 the KV cache, the counterpart of LLaDA's prime pass); decode
                 steps stay bf16
  noise_s<k>     bf16 logits + U(-1, +1) bf16 ULP, generator reseeded per
                 (seed, batch) as in research/probe.py

Telemetry (control_tel): research.metrics.router_stats per (phase, layer),
phase = prefill (prompt tokens, left-padding excluded) or decode (one new
token per still-running row; finished rows, which HF feeds pad tokens, are
excluded).

Prompt: the same system prompt and 4-shot user prompt as the LLaDA runs,
rendered with OLMoE's own chat template (Tulu style: '<|endoftext|><|system|>
...<|user|>...<|assistant|>'); the template inserts BOS itself, so the text
is tokenized with add_special_tokens=False.
"""

from __future__ import annotations

import json
import os
import time
from functools import partial
from typing import Callable, Optional

import torch
import torch.nn.functional as F

from research.metrics import (COLLAPSE_RUN, STAT_COLS, bf16_ulp, gsm8k_final_answer,
                              gsm8k_strict_answer, max_repeat_run, router_stats)
from research.runner import _read_jsonl, make_batches

PHASES = ("prefill", "decode")


class OlmoeProbe:
    def __init__(self, num_layers: int, arm: str, telemetry: bool = False, pad_id: int = 1):
        self.num_layers = num_layers
        self.arm = arm
        self.telemetry = telemetry
        self.pad_id = pad_id
        self.noise_seed = int(arm.split("_s")[1]) if arm.startswith("noise_s") else None
        # fp32_dec<k>: fp32 on exactly one decode forward, the one that routes
        # generated token k (k = 0 is the first generated token)
        self.dec_k = int(arm[len("fp32_dec"):]) if arm.startswith("fp32_dec") else None
        if arm not in ("control", "control_rep", "control_tel", "control_sentinel", "fp32_all", "fp32_prefill") \
                and self.noise_seed is None and self.dec_k is None:
            raise ValueError(arm)
        self._dec = -1
        self.injected = None
        self._phase = "prefill"
        self._valid = None
        self._gen = None
        self._batch_id = 0
        self._handles = []
        self.acc = None
        self.counts = {"forwards": 0, "prefill_forwards": 0, "rows": 0, "rows_fp32": 0}

    def install(self, model):
        self._handles.append(model.register_forward_pre_hook(self._pre, with_kwargs=True))
        for i, layer in enumerate(model.model.layers):
            self._handles.append(layer.mlp.gate.register_forward_hook(partial(self._gate, i)))
        return self

    def uninstall(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    def begin_batch(self, batch_id: int):
        self._batch_id = batch_id
        self._gen = None
        self.injected = None

    def _pre(self, module, args, kwargs):
        ids = kwargs.get("input_ids", args[0] if args else None)
        B, T = ids.shape
        self._phase = "prefill" if T > 1 else "decode"
        self._dec = -1 if self._phase == "prefill" else self._dec + 1
        self._B = B
        if self._phase == "prefill":
            am = kwargs.get("attention_mask")
            valid = am[:, -T:].bool() if am is not None else torch.ones_like(ids, dtype=torch.bool)
        else:
            valid = ids != self.pad_id
        self._valid = valid.reshape(-1)
        self.counts["forwards"] += 1
        self.counts["prefill_forwards"] += self._phase == "prefill"
        n = int(B * T)
        self.counts["rows"] += n
        if self._fp32():
            self.counts["rows_fp32"] += n

    def _fp32(self) -> bool:
        if self.dec_k is not None:
            return self._phase == "decode" and self._dec == self.dec_k
        return self.arm == "fp32_all" or (self.arm == "fp32_prefill" and self._phase == "prefill")

    def _noisy(self, l16):
        if self._gen is None:
            self._gen = torch.Generator(device=l16.device)
            self._gen.manual_seed(self.noise_seed * 1_000_003 + self._batch_id)
        u = torch.rand(l16.shape, generator=self._gen, device=l16.device, dtype=torch.float32)
        return l16.float() + (2.0 * u - 1.0) * bf16_ulp(l16)

    def _gate(self, layer, module, inputs, out):
        x = inputs[0]
        l32 = None
        if self.telemetry or self._fp32():
            l32 = F.linear(x.float(), module.weight.float())
        if self.telemetry:
            if self.acc is None:
                self.acc = torch.zeros(len(PHASES), self.num_layers, len(STAT_COLS), dtype=torch.float64,
                                       device=x.device)
            st = router_stats(l32, out)
            w = self._valid.to(st.dtype).unsqueeze(0)
            self.acc[PHASES.index(self._phase), layer] += (w @ st).squeeze(0).double()
        if self._fp32():
            if self.dec_k is not None:
                from research.metrics import topk_flips
                f = (topk_flips(out, l32, 8) & self._valid).float().view(self._B, -1).sum(1)
                self.injected = f if self.injected is None else self.injected + f
            return l32
        if self.noise_seed is not None:
            return self._noisy(out)
        return None          # keep the gate's own output


def load_olmoe(path: str, device: str = "cuda"):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16,
                                                 attn_implementation="sdpa").to(device).eval()
    cfg = model.config
    assert (cfg.num_experts, cfg.num_experts_per_tok, cfg.norm_topk_prob) == (64, 8, False), cfg
    return model, tok


def encode(tok, prompt: str, system_prompt: str) -> list[int]:
    text = tok.apply_chat_template(
        [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
        add_generation_prompt=True, tokenize=False)
    return tok(text, add_special_tokens=False)["input_ids"]


@torch.no_grad()
def run_arms(model, tok, questions: list[dict], arms: list[str], out_dir: str, batch_size: int,
             system_prompt: str, max_new_tokens: int = 256,
             on_batch: Optional[Callable[[], None]] = None,
             should_stop: Optional[Callable[[], bool]] = None,
             grade: Optional[Callable[[str, str], bool]] = None) -> dict:
    if grade is None:
        from benchmarks.correctness.run_math_reasoning_code import _grade_gsm8k, extract_final_answer
        grade = lambda raw, exp: bool(_grade_gsm8k(extract_final_answer(raw), exp, raw))
    os.makedirs(out_dir, exist_ok=True)
    qmap = {q["idx"]: q for q in questions}
    encoded = {q["idx"]: encode(tok, q["prompt"], system_prompt) for q in questions}
    batches = make_batches(encoded, batch_size)
    device = next(model.parameters()).device
    pad_id, eos_id = tok.pad_token_id, tok.eos_token_id
    summary = {}
    for arm in arms:
        path = os.path.join(out_dir, f"{arm}.jsonl")
        done = _read_jsonl(path)
        probe = OlmoeProbe(model.config.num_hidden_layers, arm, telemetry=arm == "control_tel",
                           pad_id=pad_id).install(model)
        segment, n_new, t_arm = None, 0, time.time()
        try:
            for b_id, idxs in enumerate(batches):
                if arm == "control_sentinel" and b_id >= 1:
                    break                      # first batch only: cross-container reproducibility check
                if all(i in done for i in idxs):
                    continue
                if should_stop is not None and should_stop():
                    print(f"[{arm}] stop requested before batch {b_id}", flush=True)
                    break
                segment = b_id if segment is None else segment
                rows = [encoded[i] for i in idxs]
                W = max(len(r) for r in rows)
                ids = torch.tensor([[pad_id] * (W - len(r)) + r for r in rows], device=device)
                am = torch.tensor([[0] * (W - len(r)) + [1] * len(r) for r in rows], device=device)
                probe.begin_batch(b_id)
                t0 = time.time()
                out = model.generate(input_ids=ids, attention_mask=am, max_new_tokens=max_new_tokens,
                                     do_sample=False, temperature=None, top_p=None, top_k=None,
                                     pad_token_id=pad_id, eos_token_id=eos_id)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                dt = time.time() - t0
                gen = out[:, W:].tolist()
                with open(path, "a") as f:
                    for r, i in enumerate(idxs):
                        g = gen[r]
                        hit = eos_id in g
                        g_ids = g[: g.index(eos_id)] if hit else [t for t in g if t != pad_id]
                        text = tok.decode(g_ids, skip_special_tokens=True)
                        q = qmap[i]
                        f.write(json.dumps({
                            "arm": arm, "idx": i, "qhash": q["qhash"], "batch": b_id, "expected": q["expected"],
                            "answer": gsm8k_final_answer(text), "answer_strict": gsm8k_strict_answer(text),
                            "correct": grade(text, q["expected"]),
                            "capped": not hit,     # hit max_new_tokens without EOS
                            "n_tokens": len(g_ids), "max_run": max_repeat_run(g_ids),
                            "collapse": max_repeat_run(g_ids) >= COLLAPSE_RUN,
                            "out": g, "text": text, "batch_seconds": round(dt, 3), "batch_size": len(idxs),
                            "prompt_tokens": len(encoded[i]),
                            "injected_flips": None if probe.injected is None else float(probe.injected[r]),
                        }) + "\n")
                n_new += len(idxs)
                if probe.acc is not None:
                    torch.save({"acc": probe.acc.cpu(), "phases": list(PHASES), "stat_cols": list(STAT_COLS),
                                "counts": probe.counts, "batches_done": b_id},
                               os.path.join(out_dir, f"{arm}.telemetry.seg{segment}.pt"))
                print(f"[{arm}] batch {b_id + 1}/{len(batches)} B={len(idxs)} W={W} {dt:.1f}s "
                      f"({sum(len(x) for x in gen) / dt:.0f} tok/s incl. pads)", flush=True)
                if on_batch is not None:
                    on_batch()
        finally:
            probe.uninstall()
        summary[arm] = {"new_rows": n_new, "seconds": round(time.time() - t_arm, 1), "counts": probe.counts}
        if segment is not None:
            with open(os.path.join(out_dir, f"{arm}.counts.seg{segment}.json"), "w") as f:
                json.dump(summary[arm], f)
        if on_batch is not None:
            on_batch()
    return summary
