"""Run the single-GPU measurements on Modal.

Modal is serverless containers, not a machine you SSH into, so the multi-GPU
serving work (start_dp.sh, the router, hybrid TP) does not port here without
rework. What DOES port cleanly is everything single-GPU and single-process,
which is exactly what this project still needs:

  1. a same-machine baseline, so the speedup ladder stops crossing hardware
  2. the historical profiling study, which is single-GPU batch-1 by design

Setup (once):

    pip install modal
    modal token new

Then, in order:

    modal run modal_app.py::fetch_weights     # CPU only, ~10 min, cents
    modal run modal_app.py::tune              # 1x H100, ~10 min
    modal run modal_app.py::baseline          # 1x H100, ~10 min   <- the number
    modal run modal_app.py::profile           # 1x H100, ~45 min   <- optional

Weights live in a Modal Volume, so they are downloaded once and reused. The
tuned kernel config is written to the same Volume, because Modal's H100 is an
SXM5 part whose device name differs from the PCIe cards used so far -- without
its own tuning run the kernel silently falls back to hardcoded tile shapes and
every number below would understate the engine.

NOTE: this file has not been executed. There is no Modal token on the machine
it was written on, so treat the first run as a smoke test and expect to adjust
the image pins.
"""

import modal

REPO = "https://github.com/RLS-ResearchLab/LLaDA_infr_hedhili.git"
MODEL = "inclusionAI/LLaDA-MoE-7B-A1B-Instruct"

app = modal.App("dminfr")

# Weights (~15 GB) and tuned configs persist here between runs.
vol = modal.Volume.from_name("dminfr-data", create_if_missing=True)
DATA = "/data"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch",
        "numpy",
        "safetensors",
        "transformers==4.53.2",   # pinned by the project
        "accelerate",
        "huggingface_hub",
        "triton",
        "tqdm",
    )
)


def _clone():
    """Fresh clone per run: the profiling study needs real git history."""
    import subprocess
    subprocess.run(["git", "clone", "--quiet", REPO, "/repo"], check=True)
    # The engine looks for the tuned config in the repo root.
    import glob, shutil, os
    for f in glob.glob(f"{DATA}/moe_tune_config*.json"):
        shutil.copy(f, os.path.join("/repo", os.path.basename(f)))
    return "/repo"


def _run(cmd, cwd="/repo"):
    import subprocess, sys
    print("+", " ".join(cmd), flush=True)
    p = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)
    sys.stdout.write(p.stdout)
    sys.stderr.write(p.stderr)
    return p.stdout


@app.function(image=image, volumes={DATA: vol}, timeout=60 * 60, cpu=4)
def fetch_weights():
    """CPU only -- no reason to pay for a GPU to download 15 GB."""
    import os
    from huggingface_hub import snapshot_download
    dest = f"{DATA}/weights"
    if os.path.exists(os.path.join(dest, "config.json")):
        print("weights already present, skipping")
        return
    snapshot_download(repo_id=MODEL, local_dir=dest)
    vol.commit()
    print("weights downloaded to", dest)


@app.function(image=image, gpu="H100", volumes={DATA: vol}, timeout=60 * 60)
def tune():
    """Autotune the MoE kernel for THIS GPU. Skipping this makes every
    measurement below an under-estimate -- tile shapes are hardware-specific,
    and running the tuner was the largest single speed win in this project."""
    import glob, shutil, torch
    repo = _clone()
    print("GPU:", torch.cuda.get_device_name(0))
    _run(["python", "-m", "dminfr.tuning.autotune_moe", "--model", "FULL_CFG"])
    for f in glob.glob(f"{repo}/moe_tune_config*.json"):
        shutil.copy(f, DATA)
        print("saved", f)
    vol.commit()


@app.function(image=image, gpu="H100", volumes={DATA: vol}, timeout=60 * 60)
def baseline():
    """The missing number: baseline and optimized measured on ONE machine.

    Every speedup ratio in the README currently rests on a baseline taken on
    different hardware from the throughput figures. This closes that."""
    import torch
    repo = _clone()
    print("GPU:", torch.cuda.get_device_name(0))
    import dminfr.engine.fused_moe_triton as f  # noqa
    out = _run([
        "python", "-m", "benchmarks.check_time_inference",
        "--weight-dir", f"{DATA}/weights",
        "--mode", "both",
        "--gen-length", "128", "--steps", "128", "--block-length", "32",
        "--num-runs", "3", "--num-warmup", "1",
    ])
    return out


@app.function(image=image, gpu="H100", volumes={DATA: vol}, timeout=3 * 60 * 60)
def profile():
    """The historical profiling study: five milestones, one fixed workload.

    Milestones and their rationale are in profiling/README.md. Runs without
    nsys here -- Modal images do not ship Nsight Systems, so this captures
    timings only. That still gives the speed progression; the kernel timelines
    would need nsys installed into the image."""
    repo = _clone()
    milestones = [
        ("m1_fused_moe_kv", "a5f6ebe"),
        ("m2_host_sync",    "5b2220d"),
        ("m3_mem_traffic",  "b954121"),
        ("m4_launch_count", "c2196ba"),
        ("m5_rope_final",   "1b21e25"),
    ]
    results = {}
    for label, commit in milestones:
        print(f"\n{'=' * 60}\n{label}  ({commit})\n{'=' * 60}", flush=True)
        _run(["git", "worktree", "add", "--detach", f"/wt/{label}", commit])
        import glob, shutil, os
        for f in glob.glob(f"{DATA}/moe_tune_config*.json"):
            shutil.copy(f, os.path.join(f"/wt/{label}", os.path.basename(f)))
        # Pre-restructure commits use eval/, current ones use benchmarks/.
        mod = ("benchmarks.check_time_inference"
               if os.path.exists(f"/wt/{label}/benchmarks/check_time_inference.py")
               else "eval.check_time_inference")
        results[label] = _run([
            "python", "-m", mod,
            "--weight-dir", f"{DATA}/weights",
            "--mode", "both",
            "--gen-length", "128", "--steps", "128", "--block-length", "32",
            "--num-runs", "3", "--num-warmup", "1",
        ], cwd=f"/wt/{label}")
    return results


@app.function(image=image, gpu="H100", volumes={DATA: vol}, timeout=90 * 60)
def trace():
    """Capture a baseline-vs-optimized trace pair for the report.

    Chrome-format traces, not nsys: Modal's image does not ship Nsight Systems,
    and the PyTorch profiler is already a dependency. The output opens directly
    in ui.perfetto.dev, which is where the existing report figure came from, so
    the new pair will match it visually.
    """
    import glob, shutil, os, torch
    repo = _clone()
    print("GPU:", torch.cuda.get_device_name(0))
    out = f"{DATA}/traces"
    os.makedirs(out, exist_ok=True)
    _run([
        "python", "-m", "benchmarks.trace_compare",
        "--weight-dir", f"{DATA}/weights",
        "--out-dir", out,
        "--gen-length", "128", "--steps", "128", "--block-length", "32",
    ])
    vol.commit()
    for f in sorted(glob.glob(f"{out}/*")):
        print("  %-42s %8.1f MB" % (os.path.basename(f), os.path.getsize(f) / 1e6))
    return sorted(os.path.basename(f) for f in glob.glob(f"{out}/*"))


@app.local_entrypoint()
def main():
    print("Run one function at a time, in order:")
    print("  modal run modal_app.py::fetch_weights")
    print("  modal run modal_app.py::tune")
    print("  modal run modal_app.py::baseline")
    print("  modal run modal_app.py::trace     <- traces for the report")
    print("  modal run modal_app.py::profile")
    print()
    print("Download the traces afterwards with:")
    print("  modal volume get dminfr-data traces/ ./traces")
