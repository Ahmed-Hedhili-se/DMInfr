"""Modal entry points for the routing-fragility study (research/).

    modal run research/modal_app.py::fetch          # CPU: weights + GSM8K selection into the volume
    modal run research/modal_app.py::setup          # GPU: autotune MoE for this GPU, run GPU test suite
    modal run research/modal_app.py::phase --phase p0_smoke --n 5 --arms control

Weights, the HF cache, the tuned kernel config and every result live in one
modal.Volume ("llada-routing"), so nothing is downloaded twice. Every GPU
function has a hard timeout, writes JSONL after each batch, commits the volume
after each batch, and keeps a heartbeat ledger entry (/vol/ledger/<run>.json)
with its elapsed seconds and cost, so a timeout loses at most one batch and
still leaves an accurate bill. The local entrypoint appends each run's cost to
research/BUDGET.md and refuses to start a run that the remaining budget
cannot cover.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import modal

APP_NAME = "llada-routing"
VOL = "/vol"
PROJ = "/root/proj"
LLADA_REPO = "inclusionAI/LLaDA-MoE-7B-A1B-Instruct"
OLMOE_REPO = "allenai/OLMoE-1B-7B-0924-Instruct"
GPU = "A100-40GB"

# Modal list prices (modal.com/pricing, read 2026-10-01), USD per second.
RATES = {"A100-40GB": 0.000583, "L40S": 0.000542, "cpu_core": 0.0000131, "mem_gib": 0.00000222}
GPU_CPU, GPU_MEM_GIB = 2.0, 16
CPU_ONLY_CPU, CPU_ONLY_MEM_GIB = 2.0, 8
# Container boot + image pull happen before our clock starts; billed anyway.
BOOT_ALLOWANCE_S = 60
BUDGET_CAP_USD = 22.0

REPO_ROOT = Path(__file__).resolve().parent.parent

app = modal.App(APP_NAME)
vol = modal.Volume.from_name(APP_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.6.0",            # bundles triton 3.2
        "transformers==4.53.2",    # the project's pin; also has OLMoE
        "datasets==3.6.0",
        "safetensors", "accelerate", "huggingface_hub", "numpy", "requests", "tqdm",
    )
    .env({"HF_HOME": f"{VOL}/hf", "PYTHONPATH": PROJ, "LLADA_RESEARCH_HOOKS": "1",
          "TOKENIZERS_PARALLELISM": "false"})
    .add_local_dir(REPO_ROOT / "dminfr", f"{PROJ}/dminfr")
    .add_local_dir(REPO_ROOT / "benchmarks", f"{PROJ}/benchmarks")
    .add_local_dir(REPO_ROOT / "tests", f"{PROJ}/tests")
    .add_local_dir(REPO_ROOT / "research", f"{PROJ}/research",
                   ignore=["data/**", "plots/**", "__pycache__/**", "*.md"])
)


def _rate_per_s(gpu: str | None, cpu: float, mem_gib: float) -> float:
    r = cpu * RATES["cpu_core"] + mem_gib * RATES["mem_gib"]
    return r + (RATES[gpu] if gpu else 0.0)


class _Ledger:
    """Heartbeat bill for one invocation, rewritten on every tick."""

    def __init__(self, run_id: str, gpu: str | None, cpu: float, mem: float, note: str = ""):
        self.path = f"{VOL}/ledger/{run_id}.json"
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self.t0 = time.time()
        self.rate = _rate_per_s(gpu, cpu, mem)
        self.row = {"run": run_id, "gpu": gpu, "cpu": cpu, "mem_gib": mem, "rate_usd_s": self.rate,
                    "note": note, "boot_allowance_s": BOOT_ALLOWANCE_S}
        self.tick(status="running")

    def seconds(self) -> float:
        return time.time() - self.t0 + BOOT_ALLOWANCE_S

    def cost(self) -> float:
        return self.seconds() * self.rate

    def tick(self, status: str = "running", **extra):
        self.row.update(extra, status=status, seconds=round(self.seconds(), 1),
                        cost_usd=round(self.cost(), 4))
        with open(self.path, "w") as f:
            json.dump(self.row, f)
        vol.commit()


def _install_tuned_config():
    """The engine loads moe_tune_config.device_name=<gpu>.json from the repo root.

    Modal may serve an "A100-40GB" request on an A100-SXM4-80GB. Same SM80
    chip, but a different device name, so the 40GB-keyed file would silently
    not load and the kernel would fall back to hardcoded tiles -- different
    tiles, different numerics. Every run therefore installs the one tuned
    config under whatever name this container's GPU has.
    """
    import glob
    import shutil
    import torch
    files = sorted(glob.glob(f"{VOL}/tune/moe_tune_config*.json"))
    assert len(files) == 1, f"expected exactly one tuned config, found {files}"
    tag = torch.cuda.get_device_name(0).replace(" ", "_").replace("/", "_")
    dest = f"{PROJ}/moe_tune_config.device_name={tag}.json"
    shutil.copy(files[0], dest)
    print(f"tuned config: {os.path.basename(files[0])} installed as {os.path.basename(dest)}", flush=True)
    return os.path.basename(files[0])


# ── CPU: downloads ──────────────────────────────────────────────────────────

@app.function(image=image, volumes={VOL: vol}, timeout=60 * 60, cpu=CPU_ONLY_CPU,
              memory=CPU_ONLY_MEM_GIB * 1024)
def fetch_remote(run_id: str, olmoe: bool = False):
    import sys
    sys.path.insert(0, PROJ)
    from huggingface_hub import snapshot_download

    ledger = _Ledger(run_id, None, CPU_ONLY_CPU, CPU_ONLY_MEM_GIB, "weights + GSM8K selection")
    ignore = ["*.msgpack", "*.h5", "flax_model*", "tf_model*", "rust_model.ot", "coreml*", "onnx*"]
    repos = [(LLADA_REPO, f"{VOL}/weights/llada")] + ([(OLMOE_REPO, f"{VOL}/weights/olmoe")] if olmoe else [])
    for repo, dest in repos:
        if os.path.exists(os.path.join(dest, "config.json")) and any(Path(dest).glob("*.safetensors")):
            print("already present:", dest)
            continue
        snapshot_download(repo_id=repo, local_dir=dest, ignore_patterns=ignore)
        print("downloaded", repo, "->", dest, flush=True)
        ledger.tick()

    sel = f"{VOL}/data/gsm8k_seed42_n1000.json"
    if not os.path.exists(sel):
        from research.runner import gsm8k_questions
        qs = gsm8k_questions(1000, seed=42)
        os.makedirs(os.path.dirname(sel), exist_ok=True)
        with open(sel, "w") as f:
            json.dump(qs, f)
        print("GSM8K selection:", len(qs), "questions; first qhash", qs[0]["qhash"])
    full = f"{VOL}/data/gsm8k_seed42_n1319.json"
    if not os.path.exists(full):
        from research.runner import gsm8k_all_questions
        qs_all = gsm8k_all_questions(42)
        with open(sel) as f:
            qs1000 = json.load(f)
        assert len(qs_all) == 1319, len(qs_all)
        assert [q["qhash"] for q in qs_all[:1000]] == [q["qhash"] for q in qs1000],             "full-set order does not extend the n=1000 selection"
        with open(full, "w") as f:
            json.dump(qs_all, f)
        print("GSM8K full set: 1319 questions; first 1000 identical to the n=1000 selection")
    ledger.tick(status="done")
    return ledger.row


# ── GPU: autotune + test suite ──────────────────────────────────────────────

@app.function(image=image, gpu=GPU, volumes={VOL: vol}, timeout=40 * 60, cpu=GPU_CPU,
              memory=GPU_MEM_GIB * 1024)
def setup_remote(run_id: str, tune: bool = True):
    import glob
    import shutil
    import subprocess
    import torch

    ledger = _Ledger(run_id, GPU, GPU_CPU, GPU_MEM_GIB, "autotune + GPU tests")
    dev = torch.cuda.get_device_name(0)
    print("GPU:", dev, torch.cuda.get_device_properties(0).total_memory / 2**30, "GiB", flush=True)

    def run(cmd, env=None):
        print("+", " ".join(cmd), flush=True)
        p = subprocess.run(cmd, cwd=PROJ, text=True, capture_output=True, env={**os.environ, **(env or {})})
        print(p.stdout[-6000:], p.stderr[-3000:], flush=True)
        return p.returncode

    results = {}
    if tune and not glob.glob(f"{VOL}/tune/moe_tune_config*.json"):
        rc = run(["python", "-m", "dminfr.tuning.autotune_moe", "--model", "FULL_CFG"])
        os.makedirs(f"{VOL}/tune", exist_ok=True)
        for f in glob.glob(f"{PROJ}/moe_tune_config*.json"):
            if "H100" not in f:
                shutil.copy(f, f"{VOL}/tune/")
                print("saved", f)
        results["autotune"] = rc
        ledger.tick()
    _install_tuned_config()
    for t in ["tests.test_fusions", "tests.test_fused_rope", "tests.test_fused_silu_epilogue",
              "tests.test_moe_align_block_size", "tests.test_num_logits_slice",
              "tests.test_select_transfer_indices", "tests.test_threshold_decoding",
              "tests.test_variable_length_batch", "research.tests.test_probe"]:
        results[t] = run(["python", "-m", t])
        ledger.tick()
    ledger.tick(status="done", device=dev, results=results)
    return ledger.row


# ── GPU: paired arms ────────────────────────────────────────────────────────

W = "LLADA_FP32_ROUTER_WINDOW"
#: name -> (probe env, max_batches). See research/probe.py for the window kinds.
ARM_PRESETS = {
    "control": ({}, None),
    "control_rep": ({}, None),
    "control_tel": ({"LLADA_ROUTER_TELEMETRY": "1"}, None),
    # first two batches of control, re-run in a second container to check that
    # arms split across containers still pair against one control
    "control_sentinel": ({}, 2),
    "fp32_all": ({W: "all"}, None),
    "fp32_prime": ({W: "prime"}, None),
    "fp32_finalize": ({W: "finalize"}, None),
    "fp32_steps_within_early": ({W: "steps_within:0:0.25"}, None),
    "fp32_steps_within_late": ({W: "steps_within:0.25:1"}, None),
    "fp32_steps_global_early": ({W: "steps_global:0:0.25"}, None),
    "fp32_steps_global_late": ({W: "steps_global:0.25:1"}, None),
    # equal-size (25%) late windows: churn may saturate after the first
    # divergence, so a 25% window is only comparable with another 25% window
    "fp32_steps_within_last25": ({W: "steps_within:0.75:1"}, None),
    "fp32_steps_global_last25": ({W: "steps_global:0.75:1"}, None),
    # causal layer arms, all passes
    "fp32_layer0": ({"LLADA_FP32_ROUTER_LAYERS": "0"}, None),
    "fp32_layers11_14": ({"LLADA_FP32_ROUTER_LAYERS": "11-14"}, None),
    "fp32_layer15": ({"LLADA_FP32_ROUTER_LAYERS": "15"}, None),
    # matched-dose: fp32 on exactly one denoising-step pass
    "fp32_single_b0s0": ({W: "single:0:0"}, None),
    "fp32_single_b0s16": ({W: "single:0:16"}, None),
    "noise_s0": ({"LLADA_ROUTER_NOISE_SEED": "0"}, None),
    "noise_s1": ({"LLADA_ROUTER_NOISE_SEED": "1"}, None),
    "noise_s2": ({"LLADA_ROUTER_NOISE_SEED": "2"}, None),
}


@app.function(image=image, gpu=GPU, volumes={VOL: vol}, timeout=3 * 60 * 60, cpu=GPU_CPU,
              memory=GPU_MEM_GIB * 1024)
def phase_remote(run_id: str, phase: str, n: int, arms: list[str], gen: dict, batch_size: int,
                 max_usd: float):
    import torch
    tuned = _install_tuned_config()
    import dminfr.engine.fused_moe_triton as fmt
    assert fmt.TUNED_CONFIGS, "tuned MoE config did not load"
    from transformers import AutoTokenizer
    from benchmarks.correctness.run_math_reasoning_code import SYSTEM_PROMPT_GSM8K
    from research.runner import Arm, GenConfig, load_engine, run_arms

    ledger = _Ledger(run_id, GPU, GPU_CPU, GPU_MEM_GIB, f"{phase} n={n} arms={','.join(arms)}")
    dev = torch.cuda.get_device_name(0)
    gcfg = GenConfig(**gen)
    qfile = f"{VOL}/data/gsm8k_seed42_n1319.json"
    if not os.path.exists(qfile):
        qfile = f"{VOL}/data/gsm8k_seed42_n1000.json"
    with open(qfile) as f:
        questions = json.load(f)
    assert n <= len(questions), (n, qfile)
    questions = questions[:n]
    tok = AutoTokenizer.from_pretrained(f"{VOL}/weights/llada", trust_remote_code=True)
    t_load = time.time()
    model = load_engine(f"{VOL}/weights/llada")
    print(f"model loaded in {time.time() - t_load:.0f}s; GPU {dev}; "
          f"{torch.cuda.memory_allocated() / 2**30:.1f} GiB", flush=True)
    out_dir = f"{VOL}/results/{phase}/{gcfg.tag}_B{batch_size}"
    summary = run_arms(
        model, tok, questions, [Arm(a, *ARM_PRESETS[a]) for a in arms], gcfg, out_dir,
        batch_size, SYSTEM_PROMPT_GSM8K,
        on_batch=lambda: ledger.tick(peak_gib=round(torch.cuda.max_memory_allocated() / 2**30, 2)),
        should_stop=lambda: ledger.cost() > max_usd,
    )
    ledger.tick(status="done", device=dev, tuned=tuned, out_dir=out_dir, summary=summary)
    return ledger.row


@app.function(image=image, gpu=GPU, volumes={VOL: vol}, timeout=3 * 60 * 60, cpu=GPU_CPU,
              memory=GPU_MEM_GIB * 1024)
def olmoe_remote(run_id: str, phase: str, n: int, arms: list[str], batch_size: int, max_usd: float,
                 ids: list[int] | None = None):
    """SQ3: OLMoE arms (research/olmoe.py). `ids` restricts to a question subset."""
    import torch
    from benchmarks.correctness.run_math_reasoning_code import SYSTEM_PROMPT_GSM8K
    from research.olmoe import load_olmoe, run_arms

    ledger = _Ledger(run_id, GPU, GPU_CPU, GPU_MEM_GIB, f"{phase} OLMoE n={n} arms={','.join(arms)}")
    with open(f"{VOL}/data/gsm8k_seed42_n1319.json") as f:
        questions = json.load(f)[:n]
    if ids is not None:
        keep = set(ids)
        questions = [q for q in questions if q["idx"] in keep]
    t = time.time()
    model, tok = load_olmoe(f"{VOL}/weights/olmoe")
    dev = torch.cuda.get_device_name(0)
    print(f"OLMoE loaded in {time.time() - t:.0f}s on {dev}; {torch.cuda.memory_allocated() / 2**30:.1f} GiB; "
          f"{len(questions)} questions", flush=True)
    sample = tok.apply_chat_template([{"role": "system", "content": "S"}, {"role": "user", "content": "U"}],
                                     add_generation_prompt=True, tokenize=False)
    print("chat template renders as:", repr(sample), flush=True)
    out_dir = f"{VOL}/results/{phase}/olmoe_L256_greedy_B{batch_size}"
    summary = run_arms(
        model, tok, questions, arms, out_dir, batch_size, SYSTEM_PROMPT_GSM8K, max_new_tokens=256,
        on_batch=lambda: ledger.tick(peak_gib=round(torch.cuda.max_memory_allocated() / 2**30, 2)),
        should_stop=lambda: ledger.cost() > max_usd,
    )
    ledger.tick(status="done", device=dev, out_dir=out_dir, summary=summary, template=sample)
    return ledger.row


# ── local side: budget bookkeeping ──────────────────────────────────────────

BUDGET_MD = Path(__file__).resolve().parent / "BUDGET.md"
SPENT_MARK = "<!-- ledger -->"


def _spent_so_far() -> float:
    """Sum of the cost column of the run log in BUDGET.md."""
    total = 0.0
    if BUDGET_MD.exists():
        for line in BUDGET_MD.read_text(encoding="utf-8").splitlines():
            if line.startswith("| run-"):
                total += float(line.split("|")[6].strip().lstrip("$"))
    return total


def _record(row: dict):
    spent = _spent_so_far() + row["cost_usd"]
    line = (f"| {row['run']} | {row.get('note', '')} | {row.get('gpu') or 'CPU'} | {row['seconds']:.0f} | "
            f"{row['rate_usd_s']:.6f} | ${row['cost_usd']:.4f} | ${spent:.4f} | {row.get('status')} |\n")
    with open(BUDGET_MD, "a", encoding="utf-8") as f:
        f.write(line)
    print(f"cost ${row['cost_usd']:.4f}; cumulative ${spent:.4f} of ${BUDGET_CAP_USD}")


def _ledger_row(run_id: str) -> dict | None:
    try:
        data = b"".join(vol.read_file(f"ledger/{run_id}.json"))
        return json.loads(data)
    except Exception:
        return None


def _call(fn, run_id: str, *args, **kwargs):
    """Run a remote function; bill it from its return value, or from the
    heartbeat ledger if it died (timeout, preemption, crash)."""
    try:
        row = fn.remote(run_id, *args, **kwargs)
    except Exception as e:
        print("remote failed:", repr(e))
        row = _ledger_row(run_id)
        if row is None:
            raise
        row["status"] = f"failed: {type(e).__name__}"
    _record(row)
    return row


def _spawn(fn, run_id: str, *args):
    """Start a run that survives the local client: use with `modal run --detach`.
    Nothing is billed into BUDGET.md here; record the run afterwards from its
    ledger (`python -m research.ledger <run_id>`)."""
    call = fn.spawn(run_id, *args)
    print(f"spawned {run_id} as {call.object_id}; follow /vol/ledger/{run_id}.json", flush=True)


def _run_id(tag: str) -> str:
    return f"run-{time.strftime('%Y%m%d-%H%M%S')}-{tag}"


@app.local_entrypoint()
def fetch(olmoe: bool = False):
    _call(fetch_remote, _run_id("fetch"), olmoe)


@app.local_entrypoint()
def setup(tune: bool = True):
    _call(setup_remote, _run_id("setup"), tune)


@app.local_entrypoint()
def phase(phase: str, n: int, arms: str, max_tokens: int = 256, block_length: int = 32,
          steps: int = 0, threshold: float = -1.0, low_threshold: float = 0.4,
          batch_size: int = 32, max_usd: float = 3.0, spawn: bool = False):
    remaining = BUDGET_CAP_USD - _spent_so_far()
    if max_usd > remaining:
        raise SystemExit(f"run cap ${max_usd} exceeds remaining budget ${remaining:.2f}")
    gen = dict(max_tokens=max_tokens, block_length=block_length, steps=steps or max_tokens,
               threshold=None if threshold < 0 else threshold,
               low_threshold=None if threshold < 0 else low_threshold)
    if spawn:
        _spawn(phase_remote, _run_id(phase), phase, n, arms.split(","), gen, batch_size, max_usd)
        return
    row = _call(phase_remote, _run_id(phase), phase, n, arms.split(","), gen, batch_size, max_usd)
    print(json.dumps(row.get("summary", {}), indent=1)[:3000])


@app.local_entrypoint()
def olmoe(phase: str, n: int, arms: str, batch_size: int = 32, max_usd: float = 1.0, ids_file: str = "",
          spawn: bool = False):
    remaining = BUDGET_CAP_USD - _spent_so_far()
    if max_usd > remaining:
        raise SystemExit(f"run cap ${max_usd} exceeds remaining budget ${remaining:.2f}")
    ids = json.load(open(ids_file)) if ids_file else None
    if spawn:
        _spawn(olmoe_remote, _run_id(phase), phase, n, arms.split(","), batch_size, max_usd, ids)
        return
    row = _call(olmoe_remote, _run_id(phase), phase, n, arms.split(","), batch_size, max_usd, ids)
    print(json.dumps(row.get("summary", {}), indent=1)[:3000])
