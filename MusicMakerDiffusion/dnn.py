"""
dnn.py — shared neural primitives for the diffusion model.

These are deliberate COPIES of the equivalents in the MusicMakerTransformer
project, not imports. The two projects are independent: either must be movable,
editable, or deleted without touching the other. The duplication is the point.

Keeping the primitives byte-identical across both projects is what makes the
AR-vs-diffusion comparison honest -- if the diffusion model used a different
RMSNorm or RoPE, a quality difference could not be attributed to the
architecture.
"""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    Cheaper than LayerNorm: no mean subtraction, no bias.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return x / rms * self.weight


def build_rope(seq_len: int, d_head: int, device, base: float = 10000.0):
    """Precompute (seq_len, d_head/2) cos/sin tables. Build once, slice forever."""
    inv = 1.0 / (base ** (torch.arange(0, d_head, 2, device=device).float() / d_head))
    f = torch.outer(torch.arange(seq_len, device=device).float(), inv)
    return torch.cos(f), torch.sin(f)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
               offset: int = 0) -> torch.Tensor:
    """
    Rotary embedding on (B, H, S, d_head).

    In the AR project `offset` tracks KV-cache position during incremental
    decode. The diffusion model has no incremental decode -- it denoises the
    whole clip at once -- so offset is always 0 here. Kept in the signature so
    the function stays identical to its AR counterpart.
    """
    S = x.shape[2]
    c = cos[offset:offset + S][None, None]
    s = sin[offset:offset + S][None, None]
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1).flatten(-2)
