r"""
optim.py — LAMB, so the learning rate stops being the thing that decides whether
a run lives or dies.

WHY THIS EXISTS
---------------
Four AdamW runs on this project, four bad outcomes that all trace to one knob:

    3e-4  -> diverged instantly (step 350)
    5e-5  -> clean for ~600 steps, then grad norm ramped 0.7 -> 5.4 -> 10.9 and
             BOTH losses turned up. Reproduced at beta2 0.95 AND 0.999.
    1e-5  -> never diverged, but stalled at loss 9.39, ~0.007 progress per 50
             steps. Too cold. A slower way to get nothing.

The pattern -- works, then blows up hundreds of steps later -- is NOT what a
too-high LR looks like (that breaks immediately, like 3e-4 did). It is what an
UNBOUNDED update looks like: at effective batch 4, AdamW's second-moment
estimate v (governed by beta2) is itself noisy and drifts, so the update
magnitude slowly wanders until it destabilizes.

THE FIX (You, Ginsburg et al. 2019, "Large Batch Optimization for Deep
Learning: Training BERT in 76 minutes" -- the LAMB paper):

Adam gives you a per-parameter direction with a roughly unit-RMS magnitude.
LAMB then rescales each LAYER's whole update by the TRUST RATIO

        trust = ||weight|| / ||adam_update||

before applying the LR. The effect is exact and provable: every layer moves by

        lr * ||weight||

per step -- a fixed fraction of its own size -- REGARDLESS of how big the raw
gradient got. A gradient spike that would have pushed the AdamW update to 10x
its usual size gets divided back down by a trust ratio that shrank 10x. The
late gn ramp cannot compound, because the step size is renormalized every step.

This is the principled version of "measure the update/weight ratio and keep it
near 1e-3": instead of measuring it once and hoping, LAMB ENFORCES it, per
layer, every step. It is why LAMB is famously insensitive to the exact LR --
which is exactly the property this project needs.

NOTES THAT MATTER FOR CORRECTNESS
---------------------------------
- Weight decay is DECOUPLED (added to the update before the trust ratio, like
  AdamW, not mixed into the gradient like classic L2). This is what the paper
  specifies and what makes wd behave the same across LRs.
- The trust ratio is only applied to tensors with >1 element (i.e. matrices).
  Biases, norm gains, and any scalar are stepped WITHOUT the trust ratio -- the
  original LARS/LAMB convention, because ||w|| for a 1-D gain is not a
  meaningful "layer scale" and dividing by it destabilizes the very parameters
  that are supposed to be stable. This project has no biases (no-bias linears)
  and RMSNorm gains, so this branch mostly guards the norm weights.
- bias correction is applied to m and v (same as Adam). Skipping it makes the
  first ~1/(1-beta2) steps take oversized trust ratios -- exactly the early
  window warmup is meant to protect, so getting it right matters less, but it
  is cheap and correct so we keep it.
- No fused kernel: LAMB needs a per-tensor norm reduction that the fused AdamW
  path doesn't expose. The cost is small next to a 16384-token forward/backward.
"""

import torch
from torch.optim.optimizer import Optimizer


class Lamb(Optimizer):
    r"""LAMB (You et al. 2019). AdamW's update, rescaled per layer by the trust
    ratio ||w|| / ||update|| so the step size is a fixed fraction of each
    layer's norm regardless of gradient scale.

    Args mirror AdamW so it is a drop-in swap:
        lr, betas=(0.9, 0.999), eps=1e-6, weight_decay=0.0

    eps defaults to 1e-6, not AdamW's 1e-8: the LAMB paper uses 1e-6, and a
    larger eps damps the trust ratio when ||update|| is tiny (early steps),
    which is steadier here.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-6,
                 weight_decay=0.0):
        if lr <= 0.0:
            raise ValueError(f"invalid lr: {lr}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"invalid betas: {betas}")
        if eps <= 0.0:
            raise ValueError(f"invalid eps: {eps}")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr, eps, wd = group["lr"], group["eps"], group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if g.is_sparse:
                    raise RuntimeError("Lamb does not support sparse gradients")

                state = self.state[p]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)

                m, v = state["exp_avg"], state["exp_avg_sq"]
                state["step"] += 1
                t = state["step"]

                # Adam moments + bias correction -> a unit-RMS-ish direction.
                m.mul_(beta1).add_(g, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(g, g, value=1 - beta2)
                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)
                update = m_hat / (v_hat.sqrt().add_(eps))

                # Decoupled weight decay: part of the step, before the trust
                # ratio, so wd scales with the layer like everything else.
                if wd != 0.0:
                    update = update.add(p, alpha=wd)

                # Trust ratio ||w|| / ||update||, per tensor. Skip 1-D params
                # (norm gains, any bias): a scalar-ish "layer norm" is not a
                # meaningful scale and dividing by it is destabilizing.
                if p.dim() > 1:
                    w_norm = p.norm()
                    u_norm = update.norm()
                    # both-nonzero guard: a fresh zero-init tensor (||w||=0) or a
                    # zero gradient (||u||=0) must fall back to trust=1, never
                    # 0/0. This is the branch that keeps the zero-init cross-attn
                    # out.weight from being frozen forever.
                    trust = torch.where(
                        (w_norm > 0) & (u_norm > 0),
                        w_norm / u_norm,
                        torch.ones_like(w_norm),
                    )
                else:
                    trust = 1.0

                p.add_(update * trust, alpha=-lr)

        return loss
