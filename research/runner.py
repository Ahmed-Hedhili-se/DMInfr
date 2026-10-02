"""Paired-arm GSM8K runner for the routing-fragility study.

One process loads the model once, then runs each arm over the SAME questions in
the SAME fixed batches (batch composition is held identical across arms: the
tuned Triton tile config is chosen from the total token count M, so a row's
numerics may depend on its batch-mates, and only the eager fp32 path is covered
by tests/test_variable_length_batch). Rows are appended to
<out_dir>/<arm>.jsonl after every batch, one row per question; a rerun skips
batches that are already complete, so a timeout loses at most one batch.

Arms are configured through the probe's env flags (see research/probe.py).
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import random
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Optional

import torch

from research.metrics import COLLAPSE_RUN, gsm8k_final_answer, gsm8k_strict_answer, max_repeat_run
from research.probe import RoutingProbe

MASK_ID = 156895


@dataclass(frozen=True)
class GenConfig:
    max_tokens: int = 256
    block_length: int = 32
    steps: int = 256
    threshold: Optional[float] = None      # None = fixed schedule
    low_threshold: Optional[float] = None

    @property
    def tag(self) -> str:
        t = "fixed" if self.threshold is None else f"thr{self.threshold:g}-{self.low_threshold:g}"
        return f"L{self.max_tokens}_b{self.block_length}_s{self.steps}_{t}"


@dataclass
class Arm:
    name: str
    env: dict = field(default_factory=dict)   # probe env flags, see research/probe.py
    max_batches: Optional[int] = None         # run only the first k batches (cross-container sentinel)


# ── model ───────────────────────────────────────────────────────────────────

def load_engine(weight_dir: str, device: str = "cuda"):
    """LLaDA-MoE with fused Triton MoE, loaded straight to the GPU.

    The server's load_model builds the full model in fp32 on the host first
    (~28 GB of RAM) and converts afterwards. Here the skeleton is created on
    the meta device, materialised in bf16 on the GPU, filled from the
    safetensors shards, and converted to fused blocks one layer at a time.
    Every parameter in the model must be covered by the checkpoint, or this
    raises -- the same 3219/3219 check the README reports.
    """
    from safetensors import safe_open
    from dminfr.engine.model import FULL_CFG, LLaDAMoEKV, TritonFusedMoEBlock

    with torch.device("meta"):
        model = LLaDAMoEKV(FULL_CFG, use_fused_moe=False)
    model = model.to(torch.bfloat16).to_empty(device=device).eval()
    params = dict(model.named_parameters())
    loaded = set()
    with open(os.path.join(weight_dir, "model.safetensors.index.json")) as f:
        wmap = json.load(f)["weight_map"]
    by_shard = {}
    for hk, shard in wmap.items():
        by_shard.setdefault(shard, []).append(hk)
    with torch.no_grad():
        for shard, keys in sorted(by_shard.items()):
            with safe_open(os.path.join(weight_dir, shard), framework="pt", device=device) as f:
                for hk in keys:
                    mk = hk[len("model."):] if hk.startswith("model.") else hk
                    if mk not in params:
                        continue
                    t = f.get_tensor(hk)
                    if t.shape != params[mk].shape:
                        raise ValueError(f"shape mismatch {hk}: {t.shape} vs {params[mk].shape}")
                    params[mk].copy_(t)
                    loaded.add(mk)
    missing = set(params) - loaded
    if missing:
        raise ValueError(f"{len(missing)} parameters not in checkpoint, e.g. {sorted(missing)[:3]}")
    for layer in model.layers:
        fused = TritonFusedMoEBlock(layer.mlp.cfg).to(torch.bfloat16).to(device)
        fused.load_state_dict_from_unfused(layer.mlp)
        layer.mlp = fused
        gc.collect()
    torch.cuda.empty_cache()
    print(f"loaded {len(loaded)}/{len(params)} tensors", flush=True)
    return model


# ── questions ───────────────────────────────────────────────────────────────

def gsm8k_questions(n: int, seed: int = 42) -> list[dict]:
    """The repo's GSM8K selection: 4-shot prompts, _stable_subset under `seed`.

    The selection is a prefix of one seeded permutation, so the n=200 set is
    the first 200 of the n=1000 set.
    """
    from benchmarks.correctness.run_math_reasoning_code import SYSTEM_PROMPT_GSM8K, load_task

    random.seed(seed)
    items, system_prompt = load_task("gsm8k", n, 4)
    assert system_prompt == SYSTEM_PROMPT_GSM8K
    return [
        {"idx": i, "prompt": it.prompt, "expected": it.expected,
         "qhash": hashlib.sha1(it.prompt.encode()).hexdigest()[:12]}
        for i, it in enumerate(items)
    ]


def gsm8k_all_questions(seed: int = 42) -> list[dict]:
    """The whole GSM8K test set (1319), in _stable_subset's seeded order.

    _stable_subset returns the dataset in its raw row order when limit >= the
    dataset size (its early return), which is exactly the order dependence it
    exists to remove. This applies its sort-then-shuffle to the full set, so
    the first 1000 here are the n=1000 selection and the first 200 the n=200
    one (checked against the saved selection when it is built).
    """
    from benchmarks.correctness.run_math_reasoning_code import _load_gsm8k

    items = _load_gsm8k(0, 4)                       # limit 0: every item, raw order
    items = sorted(items, key=lambda it: hashlib.sha1(it.prompt.encode("utf-8")).hexdigest())
    random.seed(seed)
    random.shuffle(items)
    return [
        {"idx": i, "prompt": it.prompt, "expected": it.expected,
         "qhash": hashlib.sha1(it.prompt.encode()).hexdigest()[:12]}
        for i, it in enumerate(items)
    ]


def encode_chat(tokenizer, prompt: str, system_prompt: str) -> list[int]:
    """Same template call as dminfr/serving/server.py's chat endpoint."""
    text = tokenizer.apply_chat_template(
        [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
        add_generation_prompt=True, tokenize=False,
    )
    return tokenizer(text)["input_ids"]


def decode_output(tokenizer, out_ids: list[int]) -> tuple[list[int], str]:
    """Same post-processing as the server: cut at EOS, strip trailing MASK."""
    gen = list(out_ids)
    eos = tokenizer.eos_token_id
    if eos in gen:
        gen = gen[: gen.index(eos)]
    while gen and gen[-1] == MASK_ID:
        gen.pop()
    return gen, tokenizer.decode(gen, skip_special_tokens=True)


def make_batches(encoded: dict, batch_size: int) -> list[list[int]]:
    """Fixed batches: sort by (prompt length, idx), then chunk. Similar lengths
    share a batch to keep left-padding small; the result depends only on the
    question set and batch_size, never on the arm or on a resume."""
    order = sorted(encoded, key=lambda i: (len(encoded[i]), i))
    return [order[k:k + batch_size] for k in range(0, len(order), batch_size)]


def _read_jsonl(path: str) -> dict:
    """Rows by question idx. For analysis, RESEARCH_ANSWER selects which
    answer definition fills row["answer_strict"] (every analysis compares
    that field), re-derived from the stored text:
      strict     (default) the value recorded at run time: gsm8k_strict_answer (v1)
      strict_v2  gsm8k_strict_answer_v2 (broader answer-line markers)
      strict_v3  gsm8k_strict_answer_v3 (every format observed; first marker with a number)
      lenient    the repo grader's extraction (row["answer"])
    The original v1 value stays available as row["answer_strict_v1"]."""
    from research.metrics import gsm8k_strict_answer_v2, gsm8k_strict_answer_v3
    mode = os.environ.get("RESEARCH_ANSWER", "strict")
    if mode not in ("strict", "strict_v2", "strict_v3", "lenient"):
        raise ValueError(f"RESEARCH_ANSWER={mode!r}")
    rows = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    if "answer_strict" in r:
                        r["answer_strict_v1"] = r["answer_strict"]
                        if mode == "strict_v2":
                            r["answer_strict"] = gsm8k_strict_answer_v2(r["text"])
                        elif mode == "strict_v3":
                            r["answer_strict"] = gsm8k_strict_answer_v3(r["text"])
                        elif mode == "lenient":
                            r["answer_strict"] = r["answer"]
                    rows[r["idx"]] = r
    return rows


# ── runner ──────────────────────────────────────────────────────────────────

def run_arms(model, tokenizer, questions: list[dict], arms: list[Arm], gen: GenConfig,
             out_dir: str, batch_size: int, system_prompt: str,
             on_batch: Optional[Callable[[], None]] = None,
             should_stop: Optional[Callable[[], bool]] = None,
             grade: Optional[Callable[[str, str], bool]] = None) -> dict:
    """Run every arm over every batch. Arm-major order, so a cut-off run
    leaves whole arms complete rather than all arms partial."""
    from dminfr.engine.generate import generate_cached

    if grade is None:
        from benchmarks.correctness.run_math_reasoning_code import _grade_gsm8k, extract_final_answer
        grade = lambda raw, exp: bool(_grade_gsm8k(extract_final_answer(raw), exp, raw))

    os.makedirs(out_dir, exist_ok=True)
    qmap = {q["idx"]: q for q in questions}
    encoded = {q["idx"]: encode_chat(tokenizer, q["prompt"], system_prompt) for q in questions}
    batches = make_batches(encoded, batch_size)
    device = next(model.parameters()).device
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else (tokenizer.eos_token_id or 0)
    summary = {}

    for arm in arms:
        path = os.path.join(out_dir, f"{arm.name}.jsonl")
        done = _read_jsonl(path)
        saved_env = {k: os.environ.get(k) for k in arm.env}
        os.environ.update(arm.env)
        probe = RoutingProbe.from_env(model.cfg.NL).install(model)
        t_arm = time.time()
        n_new = 0
        segment = None   # first batch this invocation runs; names its telemetry/counts files
        try:
            for b_id, idxs in enumerate(batches):
                if arm.max_batches is not None and b_id >= arm.max_batches:
                    break
                if all(i in done for i in idxs):
                    continue
                if should_stop is not None and should_stop():
                    print(f"[{arm.name}] stop requested before batch {b_id}", flush=True)
                    break
                if segment is None:
                    segment = b_id
                rows_ids = [encoded[i] for i in idxs]
                width = max(len(r) for r in rows_ids)
                input_ids = torch.tensor([[pad_id] * (width - len(r)) + r for r in rows_ids], device=device)
                lens = torch.tensor([len(r) for r in rows_ids], device=device)
                probe.begin_batch(b_id)
                t0 = time.time()
                out = generate_cached(
                    model, input_ids, gen_length=gen.max_tokens, steps=gen.steps,
                    block_length=gen.block_length, temperature=0.0,
                    confidence_threshold=gen.threshold, low_confidence_threshold=gen.low_threshold,
                    prompt_lens=lens if len(set(lens.tolist())) > 1 else None,
                )
                if device.type == "cuda":
                    torch.cuda.synchronize()
                dt = time.time() - t0
                commit = probe.commit_steps()
                conf = probe.commit_confidence()
                inj = probe.injected_flips()
                with open(path, "a") as f:
                    for row_i, i in enumerate(idxs):
                        full = out[row_i].tolist()
                        gen_ids, text = decode_output(tokenizer, full)
                        q = qmap[i]
                        f.write(json.dumps({
                            "arm": arm.name, "idx": i, "qhash": q["qhash"], "batch": b_id,
                            "gen": gen.tag, "expected": q["expected"],
                            "answer": gsm8k_final_answer(text),
                            "answer_strict": gsm8k_strict_answer(text),
                            "hit_eos": tokenizer.eos_token_id in full,
                            "correct": grade(text, q["expected"]),
                            "max_run": max_repeat_run(gen_ids),
                            "collapse": max_repeat_run(gen_ids) >= COLLAPSE_RUN,
                            "n_tokens": len(gen_ids), "tokens": gen_ids, "text": text,
                            "out": full, "commit": commit[row_i].tolist(),
                            "conf": [round(v, 4) for v in conf[row_i].tolist()],
                            "injected_flips": None if inj is None else float(inj[row_i]),
                            "batch_seconds": round(dt, 3), "batch_size": len(idxs),
                        }) + "\n")
                n_new += len(idxs)
                tel = probe.telemetry_state()
                if tel is not None:
                    torch.save({**tel, "counts": probe.counts, "batches_done": b_id},
                               os.path.join(out_dir, f"{arm.name}.telemetry.seg{segment}.pt"))
                print(f"[{arm.name}] batch {b_id + 1}/{len(batches)} B={len(idxs)} W={width} "
                      f"{dt:.1f}s ({len(idxs) * gen.max_tokens / dt:.0f} tok/s)", flush=True)
                if on_batch is not None:
                    on_batch()
        finally:
            RoutingProbe.uninstall()
            for k, v in saved_env.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        summary[arm.name] = {"new_rows": n_new, "seconds": round(time.time() - t_arm, 1),
                             "window": str(probe.window), "counts": probe.counts}
        if segment is not None:
            with open(os.path.join(out_dir, f"{arm.name}.counts.seg{segment}.json"), "w") as f:
                json.dump(summary[arm.name], f)
        if on_batch is not None:
            on_batch()
    return summary
