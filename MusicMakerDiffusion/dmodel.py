"""
dmodel.py — Diffusion Transformer (DiT) over EnCodec latents.

The diffusion counterpart to the AR project's transformer.py. Same family,
three deliberate differences -- each one is the diffusion architecture
asserting itself:

    AR transformer                  dmodel.py (diffusion)
    -----------------------------   ---------------------------------------
    causal mask (is_causal=True)    NO MASK -- every frame sees every frame
    predicts NEXT token             predicts the NOISE in ALL frames at once
    RoPE (relative, unbounded)      RoPE too, but context is fixed-length
    embedding: id -> vector         Linear: 128-dim latent -> d_model
    no timestep                     timestep t conditions EVERY block (adaLN)
    output: vocab logits            output: 128-dim v/eps prediction

WHY NO CAUSAL MASK -- this is the whole point of the comparison. The AR model
only sees backwards, so bar 8 cannot influence bar 1; long-range structure is
genuinely hard for it. The DiT sees the entire 10.24s clip at every denoising
step, so structure is decided globally. That is diffusion's real advantage, and
it costs the ability to continue past the window.

ADALN-ZERO conditioning: the timestep does not get concatenated as a token, it
MODULATES every block (scale/shift/gate on each sub-layer). The gate is
zero-initialized, so at init every block is an exact identity function and the
model starts as a clean no-op -- the same trick as zero-init cross-attn in the
AR project, for the same reason: don't inject noise into residuals before training.

KEPT from the AR project (so the comparison isolates architecture, not tricks):
    RMSNorm, fused QKV, flash attention via SDPA, no-bias linears, pre-norm,
    global std=0.02 init, frozen-T5 cross-attention, RoPE. These live in dnn.py
    -- copied, not imported: the projects stay independent.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from dnn import RMSNorm, apply_rope, build_rope


def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000):
    """Sinusoidal embedding of the diffusion timestep. (B,) -> (B, dim)."""
    half = dim // 2
    freqs = torch.exp(-math.log(max_period)
                      * torch.arange(half, device=t.device).float() / half)
    a = t[:, None].float() * freqs[None]
    emb = torch.cat([torch.cos(a), torch.sin(a)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class SelfAttention(nn.Module):
    """
    Bidirectional self-attention. Identical to the AR twin's fused-QKV design
    minus the causal mask -- that single flag is the architectural difference.
    """
    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model, self.n_heads = d_model, n_heads
        self.d_head = d_model // n_heads
        self.dropout_p = dropout
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, cos, sin):
        B, T, D = x.shape
        q, k, v = self.qkv(x).split(self.d_model, dim=-1)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        # is_causal=False: a denoiser must see the whole clip. No KV cache
        # either -- there is no incremental decode, so the AR project's
        # is_causal trap simply cannot occur here.
        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=False)
        return self.out_proj(out.transpose(1, 2).reshape(B, T, D))


class CrossAttention(nn.Module):
    """To frozen T5 lyric/tag embeddings. Same design as the AR project."""
    def __init__(self, d_model, n_heads, d_text, dropout=0.0):
        super().__init__()
        self.d_model, self.n_heads = d_model, n_heads
        self.d_head = d_model // n_heads
        self.dropout_p = dropout
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.kv_proj = nn.Linear(d_text, 2 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, ctx, ctx_mask=None):
        B, T, D = x.shape
        L = ctx.shape[1]
        q = self.q_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k, v = self.kv_proj(ctx).split(self.d_model, dim=-1)
        k = k.view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        m = ctx_mask[:, None, None, :] if ctx_mask is not None else None
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=m,
            dropout_p=self.dropout_p if self.training else 0.0, is_causal=False)
        return self.out_proj(out.transpose(1, 2).reshape(B, T, D))


class FFN(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff, bias=False)
        self.fc2 = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.dropout(F.gelu(self.fc1(x))))


class DiTBlock(nn.Module):
    """
    Pre-norm block with adaLN-Zero timestep modulation.

    Each sub-layer gets (shift, scale, gate) from the timestep embedding. The
    gates are zero-init'd (see DiT.__init__), so at step 0 every block is the
    identity and the network starts as a no-op rather than injecting noise.
    """
    def __init__(self, d_model, n_heads, d_ff, dropout=0.0, d_text=None):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = SelfAttention(d_model, n_heads, dropout)
        self.cross = None
        if d_text is not None:
            self.norm_x = RMSNorm(d_model)
            self.cross = CrossAttention(d_model, n_heads, d_text, dropout)
        self.norm2 = RMSNorm(d_model)
        self.ffn = FFN(d_model, d_ff, dropout)
        # 6 = (shift, scale, gate) x (attn, ffn)
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 6 * d_model, bias=True))

    def forward(self, x, c, cos, sin, ctx=None, ctx_mask=None):
        sh1, sc1, g1, sh2, sc2, g2 = self.ada(c).chunk(6, dim=-1)
        h = self.norm1(x) * (1 + sc1[:, None]) + sh1[:, None]
        x = x + g1[:, None] * self.attn(h, cos, sin)
        if self.cross is not None and ctx is not None:
            x = x + self.cross(self.norm_x(x), ctx, ctx_mask)
        h = self.norm2(x) * (1 + sc2[:, None]) + sh2[:, None]
        x = x + g2[:, None] * self.ffn(h)
        return x


class DiT(nn.Module):
    """
    Diffusion transformer over (B, latent_dim, T) EnCodec latents.
    Predicts v (or eps) of the same shape.
    """
    def __init__(self, latent_dim=128, d_model=768, n_heads=12, n_layers=12,
                 d_ff=3072, n_frames=512, dropout=0.1, d_text=None,
                 rope_base=10000.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.d_model = d_model
        self.n_frames = n_frames
        self.d_head = d_model // n_heads

        # A continuous latent, not a token id -- so Linear, not Embedding.
        self.in_proj = nn.Linear(latent_dim, d_model, bias=False)
        self.t_embed = nn.Sequential(
            nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        self.blocks = nn.ModuleList([
            DiTBlock(d_model, n_heads, d_ff, dropout, d_text) for _ in range(n_layers)])
        self.norm_out = RMSNorm(d_model)
        self.out_proj = nn.Linear(d_model, latent_dim, bias=False)
        # final adaLN modulation (shift, scale)
        self.ada_out = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 2 * d_model, bias=True))

        # Same global std=0.02 init as the AR project.
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # adaLN-Zero, applied AFTER the global loop (which would overwrite it):
        # zero the modulation outputs so every block starts as identity and the
        # final projection starts at 0. Same lesson as the AR project's cross-attn.
        for b in self.blocks:
            nn.init.zeros_(b.ada[-1].weight); nn.init.zeros_(b.ada[-1].bias)
            if b.cross is not None:
                nn.init.zeros_(b.cross.out_proj.weight)
        nn.init.zeros_(self.ada_out[-1].weight); nn.init.zeros_(self.ada_out[-1].bias)
        nn.init.zeros_(self.out_proj.weight)

        cos, sin = build_rope(n_frames, self.d_head, "cpu", rope_base)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, z, t, ctx=None, ctx_mask=None):
        """
        z:   (B, latent_dim, T) noisy latents
        t:   (B,) timesteps
        ->   (B, latent_dim, T) predicted v (or eps)
        """
        B, D, T = z.shape
        assert T <= self.n_frames, f"{T} frames exceeds RoPE table {self.n_frames}"
        x = self.in_proj(z.transpose(1, 2))                  # (B, T, d_model)
        c = self.t_embed(timestep_embedding(t, self.d_model))  # (B, d_model)
        for blk in self.blocks:
            x = blk(x, c, self.rope_cos, self.rope_sin, ctx, ctx_mask)
        sh, sc = self.ada_out(c).chunk(2, dim=-1)
        x = self.norm_out(x) * (1 + sc[:, None]) + sh[:, None]
        return self.out_proj(x).transpose(1, 2)              # (B, latent_dim, T)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    import ui
    from dconfig import DCFG

    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = DiT(latent_dim=DCFG.latent_dim, d_model=DCFG.d_model, n_heads=DCFG.n_heads,
            n_layers=DCFG.n_layers, d_ff=DCFG.d_ff, n_frames=DCFG.n_frames,
            d_text=768).to(dev)

    ui.rule("dmodel — DiT regression tests")
    z = torch.randn(2, DCFG.latent_dim, 256, device=dev)
    t = torch.randint(0, 1000, (2,), device=dev)
    ctx = torch.randn(2, 24, 768, device=dev)
    msk = torch.ones(2, 24, dtype=torch.bool, device=dev)
    out = m(z, t, ctx, msk)
    ui.kv_table([
        ("parameters", f"{m.count_parameters()/1e6:.1f}M  [dim](AR project: 138.4M)[/]"),
        ("latent", f"{DCFG.latent_dim}-dim x {DCFG.n_frames} frames = "
                   f"{DCFG.n_frames/DCFG.frame_rate:.2f}s"),
        ("forward", f"{tuple(z.shape)} + t -> {tuple(out.shape)}"),
    ])
    assert out.shape == z.shape, "output must match latent shape"

    # (a) adaLN-Zero: at init every block is identity and out_proj is zero, so
    #     the model outputs exactly 0 -- a clean no-op start, no noise injected.
    assert out.abs().max().item() == 0.0, "adaLN-Zero init broken (global loop overwrote it?)"
    ui.check(True, "adaLN-Zero: model is an exact no-op at init",
             "global std=0.02 loop did not overwrite the zero-init")

    # (b) BIDIRECTIONAL: changing the LAST frame must change the FIRST frame's
    #     output. This is the defining difference from the AR twin, where it
    #     provably cannot happen. If this fails, we built a causal model.
    m.eval()
    with torch.no_grad():
        m.out_proj.weight.normal_(0, 0.02)      # un-zero so outputs are visible
        for b in m.blocks:
            b.ada[-1].weight.normal_(0, 0.02)
        z1 = torch.randn(1, DCFG.latent_dim, 64, device=dev)
        z2 = z1.clone(); z2[:, :, -1] += 5.0    # perturb ONLY the last frame
        t1 = torch.zeros(1, dtype=torch.long, device=dev)
        d = (m(z1, t1) - m(z2, t1))[:, :, 0].abs().max().item()
    ui.check(d > 1e-4, "bidirectional: last frame influences first frame",
             f"delta {d:.4f} — an AR model structurally cannot do this")

    # (c) timestep must actually matter, or it's not a diffusion model
    with torch.no_grad():
        a = m(z1, torch.zeros(1, dtype=torch.long, device=dev))
        b = m(z1, torch.full((1,), 999, dtype=torch.long, device=dev))
        dt = (a - b).abs().max().item()
    ui.check(dt > 1e-4, "timestep conditions the network", f"t=0 vs t=999 delta {dt:.4f}")

    ui.panel("DiT verified — bidirectional, timestep-conditioned, zero-init", style="green")
