"""Capture a before/after execution trace pair: baseline vs optimized engine.

Same machine, same prompt, same generation settings -- only the engine differs.
Writes two Chrome-format traces that open directly in https://ui.perfetto.dev
(or chrome://tracing), which is what makes the difference visible rather than
merely tabulated.

    python -m benchmarks.trace_compare --weight-dir weights --out-dir traces

The server already has a profiling hook (PROFILE_BATCHES), but it only fires
inside the batching path, so it needs a running server and it only ever traces
the optimized engine. For a report figure the useful thing is the pair, which
is why this is a separate script.

CUDA activity only by default. Adding CPU activity traces every Python-level
dispatch across every denoising step, which produced 500-640 MB files in
earlier runs of this project -- impractical to move and slow to open. Pass
--with-cpu if you specifically want the launch-side view.
"""

import argparse
import json
import os
import time

import torch
from transformers import AutoTokenizer

from benchmarks.check_time_inference import load_baseline, load_optimized

PROMPT = "Explain in a few sentences why matrix multiplication is central to neural networks."


def _trace(label, fn, out_dir, with_cpu):
    acts = [torch.profiler.ProfilerActivity.CUDA]
    if with_cpu:
        acts.insert(0, torch.profiler.ProfilerActivity.CPU)

    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad(), torch.profiler.profile(activities=acts) as prof:
        fn()
        torch.cuda.synchronize()
    elapsed = time.time() - t0

    path = os.path.join(out_dir, f"trace_{label}.json")
    prof.export_chrome_trace(path)
    size = os.path.getsize(path) / 1e6

    # A short summary next to the trace, so the numbers in the report caption
    # do not have to be read off the timeline by hand.
    ev = [e for e in prof.key_averages() if e.self_device_time_total > 0]
    ev.sort(key=lambda e: -e.self_device_time_total)
    total = sum(e.self_device_time_total for e in ev) or 1
    summary = {
        "label": label,
        "wall_seconds": round(elapsed, 3),
        "gpu_ms": round(total / 1000, 1),
        "distinct_kernels": len(ev),
        "launches": int(sum(e.count for e in ev)),
        "top": [
            {
                "name": e.key[:70],
                "share_pct": round(100 * e.self_device_time_total / total, 1),
                "ms": round(e.self_device_time_total / 1000, 1),
                "count": int(e.count),
            }
            for e in ev[:8]
        ],
    }
    with open(os.path.join(out_dir, f"summary_{label}.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== {label} ===")
    print(f"  wall {elapsed:.2f}s | GPU {total/1000:.1f}ms | "
          f"{len(ev)} distinct kernels | {summary['launches']} launches")
    print(f"  trace: {path} ({size:.1f} MB)")
    for k in summary["top"][:5]:
        print(f"    {k['share_pct']:5.1f}%  {k['ms']:8.1f}ms  x{k['count']:<6d} {k['name'][:52]}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weight-dir", required=True)
    ap.add_argument("--out-dir", default="traces")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--gen-length", type=int, default=128)
    ap.add_argument("--steps", type=int, default=128)
    ap.add_argument("--block-length", type=int, default=32)
    ap.add_argument("--with-cpu", action="store_true")
    ap.add_argument("--skip-baseline", action="store_true",
                    help="the baseline takes ~35s per run; skip it for a quick check")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print("GPU:", torch.cuda.get_device_name(0))

    import dminfr.engine.fused_moe_triton as fmt
    print("tuned kernel configs loaded:", len(fmt.TUNED_CONFIGS))
    if not fmt.TUNED_CONFIGS:
        print("  WARNING: none found -- the optimized run will use hardcoded tile")
        print("  shapes and will understate the engine. Run the autotuner first.")

    tok = AutoTokenizer.from_pretrained(args.weight_dir, trust_remote_code=True)
    ids = tok(PROMPT, return_tensors="pt")["input_ids"].to(args.device)

    gen_kw = dict(gen_length=args.gen_length, steps=args.steps,
                  block_length=args.block_length, temperature=0.0)
    out = {}

    if not args.skip_baseline:
        from dminfr.reference.generate import generate
        model = load_baseline(args.weight_dir, args.device)
        # One warm-up so the trace is not dominated by lazy initialisation.
        with torch.no_grad():
            generate(model, ids, gen_length=32, steps=32, block_length=32, temperature=0.0)
        out["baseline"] = _trace("baseline",
                                 lambda: generate(model, ids, **gen_kw),
                                 args.out_dir, args.with_cpu)
        del model
        torch.cuda.empty_cache()

    from dminfr.engine.generate import generate_cached
    model = load_optimized(args.weight_dir, args.device)
    with torch.no_grad():
        generate_cached(model, ids, gen_length=32, steps=32, block_length=32, temperature=0.0)
    out["optimized"] = _trace("optimized",
                              lambda: generate_cached(model, ids, **gen_kw),
                              args.out_dir, args.with_cpu)

    if "baseline" in out:
        b, o = out["baseline"], out["optimized"]
        print("\n=== comparison ===")
        print(f"  wall      {b['wall_seconds']:8.2f}s -> {o['wall_seconds']:8.2f}s"
              f"   {b['wall_seconds']/max(o['wall_seconds'],1e-9):.2f}x")
        print(f"  GPU time  {b['gpu_ms']:8.1f}ms -> {o['gpu_ms']:8.1f}ms")
        print(f"  launches  {b['launches']:8d} -> {o['launches']:8d}"
              f"   {b['launches']/max(o['launches'],1):.1f}x fewer")

    print(f"\nOpen the .json files at https://ui.perfetto.dev (drag and drop).")


if __name__ == "__main__":
    main()
