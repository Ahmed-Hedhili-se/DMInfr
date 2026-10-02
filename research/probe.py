"""Routing telemetry and router-precision windows for generate_cached.

Installed through dminfr.engine.generate.set_research_hooks (which requires
LLADA_RESEARCH_HOOKS=1). Configured from two env flags, both off by default:

  LLADA_ROUTER_TELEMETRY=1
      Compute the gate in both bf16 and fp32 on the same hidden states, at
      every MoE call, and accumulate research.metrics.router_stats per
      (pass, layer, position type). One extra 2048x64 matmul per layer and
      no second forward pass. The logits handed back to the model are the
      ones the precision policy selects, so telemetry does not change outputs.

  LLADA_FP32_ROUTER_WINDOW=<kind>[:<lo>:<hi>]
      Use fp32 router logits only for the forward passes the window selects;
      bf16 elsewhere. generate_cached runs three kinds of pass: one cache-prime
      pass (prompt + all-MASK suffix; commits the prompt's K/V), denoising
      steps, and one finalize pass per block (commits that block's K/V). The
      prime and finalize passes write the KV cache, so their effect persists
      into every later pass; the step-window kinds therefore never touch them.
        all            every pass (equivalent to LLADA_FP32_ROUTER=1)
        prime          the cache-prime pass only
        finalize       every block's finalize pass only
        steps_within   denoising steps only, by within-block fraction: step s
                       of a block with S steps sits at s/S
        steps_global   denoising steps only, by block: block b of N sits at b/N
        within, global the first design, kept for reference: like the two
                       above but also covering cache writes (prime at 0.0,
                       finalize at 1.0 / its block). Confounds step timing
                       with cache writes; not used for the reported arms.
      A coordinate is in [lo, hi) when lo <= c < hi (hi >= 1 also admits 1.0).
      prime + finalize + steps_*:0:a + steps_*:a:1 partition every pass.

  LLADA_FP32_ROUTER_LAYERS=<i>[-<j>]
      Restrict the fp32 window to MoE layers i..j (inclusive). Without a
      window it implies LLADA_FP32_ROUTER_WINDOW=all. Used for the causal
      layer arms: does fp32 routing at only layer 0 (95% flip rate, entropy
      ~0.9995) or only layer 15 change answers?

  LLADA_ROUTER_NOISE_SEED=<int>
      Perturbation-family control: at every pass, add uniform noise in
      [-1, +1] bf16 ULP (at each logit's own magnitude) to the bf16 router
      logits and route on the result (kept in fp32). The generator is reseeded
      per batch from (seed, batch id), so a resumed run draws the same noise.

Commit tracking (always on; one compare per step): records, for every
generated position, the global denoising step at which its token was
committed (block * steps_per_block + step), so paired arms can be compared on
where their trajectories first diverge, and the model's confidence in that
token at the moment it was committed (the remasking score: softmax
probability of the argmax).

Position types (which rows of the MoE input a statistic came from):
  prompt           prompt tokens (cache-prime pass only; left-padding excluded)
  active_unmasked  already-revealed tokens of the block being denoised
  active_masked    still-[MASK] tokens of the block being denoised
  future_masked    [MASK] tokens of later blocks (always masked)
During denoising steps the cached prefix does not pass through the MoE at all,
so "unmasked" positions at a step are only the revealed tokens of the active
block.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from research.metrics import STAT_COLS, bf16_ulp, router_stats, topk_flips

MASK_ID = 156895
TYPES = ("prompt", "active_unmasked", "active_masked", "future_masked")


_RANGED = ("within", "global", "steps_within", "steps_global")
_WHOLE = ("all", "prime", "finalize")
_SINGLE = "single"     # single:<block>:<step> -- exactly one denoising-step pass


@dataclass(frozen=True)
class Window:
    kind: str
    lo: float = 0.0
    hi: float = 1.0

    @classmethod
    def parse(cls, spec: Optional[str]) -> Optional["Window"]:
        if not spec or spec in ("0", "none", "off"):
            return None
        parts = spec.split(":")
        if parts[0] in _WHOLE and len(parts) == 1:
            return cls(parts[0])
        if parts[0] == _SINGLE and len(parts) == 3:
            return cls(_SINGLE, float(int(parts[1])), float(int(parts[2])))
        if parts[0] not in _RANGED or len(parts) != 3:
            raise ValueError(f"bad LLADA_FP32_ROUTER_WINDOW {spec!r}")
        return cls(parts[0], float(parts[1]), float(parts[2]))

    def contains(self, phase: str, within: float, global_: float,
                 block: Optional[int] = None, step: Optional[int] = None) -> bool:
        k = self.kind
        if k == _SINGLE:
            return phase == "step" and block == int(self.lo) and step == int(self.hi)
        if k == "all":
            return True
        if k in ("prime", "finalize"):
            return phase == k
        if k.startswith("steps_") and phase != "step":
            return False
        c = within if k in ("within", "steps_within") else global_
        return self.lo <= c < self.hi or (self.hi >= 1.0 and c >= 1.0)

    def __str__(self):
        if self.kind == _SINGLE:
            return f"single:{int(self.lo)}:{int(self.hi)}"
        return self.kind if self.kind in _WHOLE else f"{self.kind}:{self.lo:g}:{self.hi:g}"


class RoutingProbe:
    def __init__(self, num_layers: int, telemetry: bool = False, window: Optional[Window] = None,
                 noise_seed: Optional[int] = None, layers: Optional[tuple] = None,
                 max_blocks: int = 64, max_steps: int = 256):
        if window is not None and noise_seed is not None:
            raise ValueError("an arm perturbs by fp32 window or by noise, not both")
        if layers is not None and window is None:
            raise ValueError("a layer restriction needs a window")
        self.layers = None if layers is None else frozenset(range(layers[0], layers[1] + 1))
        self.num_layers = num_layers
        self.telemetry = telemetry
        self.window = window
        self.noise_seed = noise_seed
        self.top_k = 8
        self._injected = None
        self._B = 1
        self._gen = None
        self._batch_id = 0
        self._commit = None
        self._conf = None
        self._prev_active = None
        self._last_step = 0
        self.max_blocks = max_blocks
        self.max_steps = max_steps
        self._acc = None        # [max_blocks+1, max_steps+1, L, 4, K] float64; slot 0 = prime
        self._mask_acc = None   # [max_blocks+1, max_steps+1, 3]: rows, masked in block, masked in suffix
        self.counts = {"passes": 0, "passes_fp32": 0, "rows": 0, "rows_fp32": 0,
                       "step_passes": 0, "step_passes_fp32": 0}
        self._reset_pass()

    @classmethod
    def from_env(cls, num_layers: int) -> "RoutingProbe":
        if os.environ.get("LLADA_FP32_ROUTER", "0") != "0":
            raise RuntimeError("LLADA_FP32_ROUTER would be overridden by the probe; unset it")
        layers = os.environ.get("LLADA_FP32_ROUTER_LAYERS", "")
        layers = tuple(int(v) for v in (layers.split("-") * 2)[:2]) if layers else None
        window = Window.parse(os.environ.get("LLADA_FP32_ROUTER_WINDOW"))
        if layers is not None and window is None:
            window = Window("all")
        return cls(
            num_layers,
            telemetry=os.environ.get("LLADA_ROUTER_TELEMETRY", "0") != "0",
            window=window,
            layers=layers,
            noise_seed=(int(os.environ["LLADA_ROUTER_NOISE_SEED"])
                        if os.environ.get("LLADA_ROUTER_NOISE_SEED", "") != "" else None),
        )

    def begin_batch(self, batch_id: int):
        """Called by the runner before each generate_cached call."""
        self._batch_id = batch_id
        self._gen = None
        self._injected = None

    def injected_flips(self) -> Optional[torch.Tensor]:
        """[B] float: top-8 flips (bf16 vs fp32 on the same hidden states,
        summed over layers and positions) in the fp32 pass of a single-pass
        arm; None for every other arm."""
        return None if self._injected is None else self._injected.cpu()

    # ── installation ──
    def install(self, model):
        from dminfr.engine.generate import set_research_hooks
        self.top_k = model.cfg.TOPK
        for i, layer in enumerate(model.layers):
            layer.mlp.gate._probe_layer = i
        set_research_hooks(router=self.router_hook, step=self.step_hook)
        return self

    @staticmethod
    def uninstall():
        from dminfr.engine.generate import set_research_hooks
        set_research_hooks()

    # ── per-pass context ──
    def _reset_pass(self):
        self._phase = None
        self._block = 0
        self._num_blocks = 1
        self._spb = 1
        self._slot = (0, 0)
        self._types = None
        self._use32 = self.window is not None and self.window.kind == "all"

    def step_hook(self, phase: str, **ctx):
        if phase == "commit":
            col = self._block * self._bl
            t = ctx["transfer"]
            self._conf[:, col:col + self._bl][t] = ctx["confidence"][t].float()
            return
        if phase == "block":
            self._block = ctx["block_idx"]
            self._prev_active = None
            return
        ids = ctx["ids"]
        B, T = ids.shape
        self._B = B
        bl = ctx["block_length"]
        if phase == "prime":
            self._num_blocks = ctx["num_blocks"]
            self._spb = ctx["steps_per_block"]
            self._bl = bl
            if self._num_blocks > self.max_blocks or self._spb > self.max_steps:
                raise ValueError("probe accumulator too small for this generation config")
            self._block = 0
            within, glob = 0.0, 0.0
            self._slot = (0, 0)
            self._commit = torch.full((B, self._num_blocks * bl), -1, dtype=torch.int32, device=ids.device)
            self._conf = torch.full((B, self._num_blocks * bl), float("nan"), device=ids.device)
            self._prev_active = None
        elif phase == "step":
            within = ctx["step"] / self._spb
            glob = self._block / self._num_blocks
            self._slot = (self._block + 1, ctx["step"])
            self._record_commits(ids[:, :bl])
            self._last_step = ctx["step"]
        elif phase == "finalize":
            within = 1.0
            glob = self._block / self._num_blocks
            self._slot = (self._block + 1, self._spb)
            self._record_commits(ids[:, :bl])
            self._prev_active = None
        else:
            raise ValueError(phase)
        self._phase = phase
        self._use32 = self.window is not None and self.window.contains(phase, within, glob,
                                                                            block=self._block, step=ctx.get("step"))

        self.counts["passes"] += 1
        self.counts["rows"] += B * T
        if phase == "step":
            self.counts["step_passes"] += 1
        if self._use32:
            self.counts["passes_fp32"] += 1
            self.counts["rows_fp32"] += B * T
            if phase == "step":
                self.counts["step_passes_fp32"] += 1

        if not self.telemetry:
            return
        dev = ids.device
        is_mask = ids == MASK_ID
        col = torch.arange(T, device=dev).unsqueeze(0).expand(B, T)
        if phase == "prime":
            P = ctx["prompt_width"]
            types = torch.where(col < P, 0, 3)
            pad = ctx.get("pad_len")
            if pad is not None:
                types = torch.where(col < pad.to(dev).unsqueeze(1), -1, types)
        else:
            active = col < bl
            types = torch.where(active, torch.where(is_mask, 2, 1), 3)
            types = torch.where(~active & ~is_mask, -1, types)  # cannot happen; excluded if it does
        self._types = F.one_hot((types + 1).reshape(-1), len(TYPES) + 1)[:, 1:].float()

        if self._acc is None:
            # Sized to the generation config seen at the first prime; a later
            # call with a different config is refused below rather than mixed in.
            self._shape = (self._num_blocks + 1, self._spb + 1)
            self._acc = torch.zeros(*self._shape, self.num_layers,
                                    len(TYPES), len(STAT_COLS), dtype=torch.float64, device=dev)
            self._mask_acc = torch.zeros(*self._shape, 3, dtype=torch.float64, device=dev)
        elif self._shape != (self._num_blocks + 1, self._spb + 1):
            raise ValueError("generation config changed within one telemetry probe")
        b, s = self._slot
        self._mask_acc[b, s, 0] += B
        self._mask_acc[b, s, 1] += is_mask[:, :bl].sum()
        self._mask_acc[b, s, 2] += is_mask.sum()

    def _record_commits(self, active: torch.Tensor):
        """Positions unmasked since the previous step's snapshot were committed
        by that previous step."""
        if self._prev_active is not None:
            newly = (self._prev_active == MASK_ID) & (active != MASK_ID)
            col = self._block * self._bl
            step_id = self._block * self._spb + self._last_step
            self._commit[:, col:col + self._bl].masked_fill_(newly, step_id)
        self._prev_active = active.clone()

    def _noisy(self, l16: torch.Tensor) -> torch.Tensor:
        if self._gen is None:
            self._gen = torch.Generator(device=l16.device)
            self._gen.manual_seed(self.noise_seed * 1_000_003 + self._batch_id)
        u = torch.rand(l16.shape, generator=self._gen, device=l16.device, dtype=torch.float32)
        return l16.float() + (2.0 * u - 1.0) * bf16_ulp(l16)

    def router_hook(self, gate, x):
        use32 = self._use32 and (self.layers is None or gate._probe_layer in self.layers)
        if not self.telemetry:
            if use32:
                l32 = F.linear(x.float(), gate.weight.float())
                if self.window.kind == _SINGLE:
                    f = topk_flips(gate(x), l32, self.top_k).float().view(self._B, -1).sum(1)
                    self._injected = f if self._injected is None else self._injected + f
                return l32
            l16 = gate(x)
            return self._noisy(l16) if self.noise_seed is not None else l16
        l16 = gate(x)
        l32 = F.linear(x.float(), gate.weight.float())
        stats = router_stats(l32, l16)                      # [M, K]
        b, s = self._slot
        self._acc[b, s, gate._probe_layer] += (self._types.T @ stats).double()
        if use32:
            return l32
        return self._noisy(l16) if self.noise_seed is not None else l16

    def commit_steps(self) -> Optional[torch.Tensor]:
        """[B, gen_length] int32: the global step that committed each
        generated position (-1 if no step committed it)."""
        return None if self._commit is None else self._commit.cpu()

    def commit_confidence(self) -> Optional[torch.Tensor]:
        """[B, gen_length] float32: confidence of each token when committed."""
        return None if self._conf is None else self._conf.cpu()

    # ── results ──
    def telemetry_state(self) -> Optional[dict]:
        if self._acc is None:
            return None
        return {
            "acc": self._acc.cpu(),
            "mask_acc": self._mask_acc.cpu(),
            "stat_cols": list(STAT_COLS),
            "types": list(TYPES),
            "num_blocks": self._num_blocks,
            "steps_per_block": self._spb,
        }
