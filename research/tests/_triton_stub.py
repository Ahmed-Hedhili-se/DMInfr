"""Import-time stand-in for Triton on machines without it (e.g. Windows CPU).

dminfr.engine.model imports fused_moe_triton unconditionally, so on a box
with no Triton the engine cannot even be imported, though the eager MoE path
the CPU tests use never launches a kernel. This registers a stub that
satisfies the import-time references (@triton.jit, tl.constexpr annotations)
and raises if anything actually tries to run a kernel.
Only installed when `import triton` fails; on a GPU box the real one is used.
"""

import sys
import types


def install():
    import torch  # noqa: F401 -- torch probes for triton on import; let it see the real absence
    try:
        import triton  # noqa: F401
        return False
    except ImportError:
        pass

    def _no_kernel(*a, **k):
        raise RuntimeError("Triton stub: no kernels on this machine")

    class _Any:
        def __getattr__(self, name):
            return _Any()

        def __call__(self, *a, **k):
            return _no_kernel()

    triton = types.ModuleType("triton")
    triton.jit = lambda fn=None, **kw: fn if fn is not None else (lambda f: f)
    triton.cdiv = _no_kernel
    triton.next_power_of_2 = _no_kernel
    lang = types.ModuleType("triton.language")
    lang.__getattr__ = lambda name: _Any()
    triton.language = lang
    sys.modules["triton"] = triton
    sys.modules["triton.language"] = lang
    print("[note] triton not installed: using import stub (eager CPU paths only)")
    return True
