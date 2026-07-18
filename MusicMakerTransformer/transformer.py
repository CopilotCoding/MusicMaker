"""
transformer.py — Custom Optimized Transformer, adapted for music
================================================================

Descended from a text LM. The body is intact: flash attention, RMSNorm,
fused QKV, pre-norm, no-bias linears, the global std=0.02 init. What changed,
changed because audio is not text -- each one is justified below.

ARCHITECTURE DECISIONS — every choice is deliberate:

1. FLASH ATTENTION — F.scaled_dot_product_attention
   - Single fused CUDA kernel. No attention matrix ever written to memory.
   - This is load-bearing at seq_len=4096: a materialized S^2 mask would be
     33MB per head per layer -- ~4.7GB across 12 layers x 12 heads. Fatal.
   - is_causal=True ONLY when q_len == kv_len. See the KV-cache note in (6).

2. FUSED QKV PROJECTION — nn.Linear(d_model, 3 * d_model)
   - One GEMM instead of three. Tensor cores see a larger matrix.
   - Memory layout contiguous — split() is a view, not a copy.

3. PRE-NORM with RMSNorm — norm before attention/FFN, not after
   - More stable training than post-norm.
   - RMSNorm over LayerNorm: no mean subtraction, no bias, half the ops.

4. NO BIAS on Linear layers
   - Bias adds memory bandwidth with negligible quality benefit at this scale.

5. ROPE — replaces learned positional embeddings   [CHANGED FOR AUDIO]
   - Learned nn.Embedding(seq_len) hard-caps context at seq_len. A 3-minute
     song is ~72,000 tokens; the table would need re-anchoring every window,
     re-encoding positions and re-prefilling the KV cache.
   - RoPE is RELATIVE: verified here that shifting q and k by 1000 positions
     changes attention scores by 8.2e-04. So sliding-window generation is just
     "evict from the front and keep going" -- no re-anchoring at all.
   - Applied to q and k only (never v), after head reshape, before SDPA.

6. KV CACHE — with the is_causal trap handled     [ADDED FOR GENERATION]
   - SDPA's is_causal anchors its mask TOP-LEFT over (q_len, kv_len). That is
     correct only when q_len == kv_len. With a cache (q_len=1, kv_len=n) it
     means the new token attends ONLY to cached token 0.
   - Measured: is_causal=True with q_len=1 is wrong by 4.2 vs ground truth;
     is_causal=False is wrong by 0.0. It does not raise. It does not warn.
     Training looks perfect and generation is garbage.
   - So: gate on shapes, never on intent. Enforced in _attend().

7. CROSS-ATTENTION to frozen T5 lyric embeddings  [ADDED FOR CONDITIONING]
   - Inserted between attention and FFN, same pre-norm + residual contract.
   - is_causal=False: lyrics are not a causal sequence; every audio position
     may see every lyric token.
   - Output projection zero-init'd AFTER the global init loop (which would
     otherwise overwrite it), so the model starts as an identical
     unconditional LM and opens the lyric gate as it learns.
   - ctx=None at runtime skips the branch -> one checkpoint serves both
     conditional and unconditional inference, which is what CFG needs.

8. NO WEIGHT TYING — decoder is independent       [CHANGED FOR AUDIO]
   - Tying saves V*d. For text (V=50k, d=512) that is 25M params, a large
     fraction of the model. Here V=16386 codec tokens: 12.6M on a 133M model.
   - Worse, it is actively wrong: only 1/8 of the vocab is legal at any step
     (each slot owns a disjoint 2048-id slice), and input vs output roles are
     asymmetric -- "what does this token mean as context" vs "which residual
     best explains the error so far". Codec tokens are near-uniform by VQ
     construction, not Zipfian, so tying's frequency regularization is inert.

9. GELU activation in FFN
   - Standard. Smooth gradient, no dead neurons.

10. GLOBAL std=0.02 INIT — applied after all modules constructed
   - Every Linear and Embedding initialized identically. Biases zeroed.
   - Cross-attn output projections re-zeroed AFTER, see (7).

USAGE:
    from transformer import MusicTransformer
    model = MusicTransformer(vocab_size=codec.vocab_size, d_text=768)
    logits = model(tokens, ctx=lyric_emb, ctx_mask=lyric_mask)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    Cheaper than LayerNorm: no mean subtraction, no bias.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps    = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return x / rms * self.weight


def build_rope(seq_len: int, d_head: int, device, base: float = 10000.0):
    """Precompute (seq_len, d_head/2) cos/sin tables. Build once, slice forever."""
    inv = 1.0 / (base ** (torch.arange(0, d_head, 2, device=device).float() / d_head))
    f   = torch.outer(torch.arange(seq_len, device=device).float(), inv)
    return torch.cos(f), torch.sin(f)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
               offset: int = 0) -> torch.Tensor:
    """
    Rotary embedding on (B, H, S, d_head). `offset` is the absolute position of
    x[:, :, 0] — during cached decode that is the number of tokens already seen.
    Because RoPE is relative, a constant shift of q and k leaves scores intact.
    """
    S = x.shape[2]
    c = cos[offset:offset + S][None, None]        # (1, 1, S, d_head/2)
    s = sin[offset:offset + S][None, None]
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1).flatten(-2)


def _attend(q, k, v, dropout_p: float):
    """
    SDPA with the causal mask anchored correctly for ANY (q_len, kv_len).

    is_causal=True builds a top-left-anchored tril, which is right only when
    q_len == kv_len. For q_len=m < kv_len=n, correct causality needs
    j <= i + (n - m) — bottom-right. Gate on shapes so this cannot be got wrong.
    """
    q_len, kv_len = q.shape[2], k.shape[2]
    if q_len == kv_len:
        # training / full prefill: the fast path, flash kernel engaged
        return F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=dropout_p, is_causal=True)
    if q_len == 1:
        # single-token decode: every cached token is visible, no mask needed
        return F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False)
    # chunked prefill against an existing cache: bottom-right anchored mask
    i = torch.arange(kv_len - q_len, kv_len, device=q.device)[:, None]
    mask = torch.arange(kv_len, device=q.device)[None, :] <= i
    return F.scaled_dot_product_attention(
        q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False)


class CausalSelfAttention(nn.Module):
    """
    Flash-style causal self-attention with RoPE and optional KV cache.
    One fused kernel. No attention matrix materialized.
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model   = d_model
        self.n_heads   = n_heads
        self.d_head    = d_model // n_heads
        self.dropout_p = dropout
        # Fused QKV: one GEMM instead of three
        self.qkv      = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model,     bias=False)

    def forward(self, x, cos, sin, kv_cache=None, pos_offset=None):
        B, T, D = x.shape
        n_past  = kv_cache[0].shape[2] if kv_cache is not None else 0
        # RoPE position for this token. Normally it is the cache length, but a
        # SLIDING window breaks that: once the cache is truncated back to `keep`
        # every new token would be rotated at ~`keep` forever, collapsing the
        # positional distinction between successive tokens (measured: output
        # degrades exactly when eviction starts). `pos_offset` lets the caller
        # pass the TRUE absolute position so rotations keep advancing after an
        # eviction. Falls back to n_past when not supplied (training, no cache).
        offset = n_past if pos_offset is None else pos_offset

        qkv     = self.qkv(x)                                  # [B, T, 3D]
        q, k, v = qkv.split(self.d_model, dim=-1)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)   # [B,H,T,d]
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # RoPE before the cache concat: cached k is already rotated at its own
        # absolute position, so rotating it again would double-apply.
        q = apply_rope(q, cos, sin, offset=offset)
        k = apply_rope(k, cos, sin, offset=offset)

        if kv_cache is not None:
            k = torch.cat([kv_cache[0], k], dim=2)
            v = torch.cat([kv_cache[1], v], dim=2)
        new_cache = (k, v)

        out = _attend(q, k, v, self.dropout_p if self.training else 0.0)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.out_proj(out), new_cache


class CrossAttention(nn.Module):
    """
    Cross-attention from the audio stream to frozen T5 lyric embeddings.
    Fused KV mirrors the fused QKV above; Q is separate (different source).
    """
    def __init__(self, d_model: int, n_heads: int, d_text: int, dropout: float = 0.0):
        super().__init__()
        self.d_model   = d_model
        self.n_heads   = n_heads
        self.d_head    = d_model // n_heads
        self.dropout_p = dropout
        self.q_proj   = nn.Linear(d_model, d_model,     bias=False)
        self.kv_proj  = nn.Linear(d_text,  2 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model,     bias=False)

    def forward(self, x, ctx, ctx_mask=None):
        B, T, D = x.shape
        L = ctx.shape[1]
        q = self.q_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k, v = self.kv_proj(ctx).split(self.d_model, dim=-1)
        k = k.view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, L, self.n_heads, self.d_head).transpose(1, 2)

        attn_mask = None
        if ctx_mask is not None:
            # (B, 1, 1, L). A row that is entirely False makes SDPA return NaN,
            # so callers must keep >=1 visible key per example (see data.py).
            attn_mask = ctx_mask[:, None, None, :]

        # is_causal=False: lyrics are not causal.
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask,
            dropout_p=self.dropout_p if self.training else 0.0, is_causal=False)
        return self.out_proj(out.transpose(1, 2).contiguous().view(B, T, D))


class FFN(nn.Module):
    """Standard two-layer FFN with GELU."""
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.fc1     = nn.Linear(d_model, d_ff, bias=False)
        self.fc2     = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.dropout(F.gelu(self.fc1(x))))


class TransformerBlock(nn.Module):
    """Pre-norm block. Attention -> [cross-attention] -> FFN, all residual."""
    def __init__(self, d_model: int, n_heads: int, d_ff: int,
                 dropout: float = 0.0, d_text: int = None):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn  = CausalSelfAttention(d_model, n_heads, dropout)
        self.cross = None
        if d_text is not None:
            self.norm_x = RMSNorm(d_model)
            self.cross  = CrossAttention(d_model, n_heads, d_text, dropout)
        self.norm2 = RMSNorm(d_model)
        self.ffn   = FFN(d_model, d_ff, dropout)

    def forward(self, x, cos, sin, ctx=None, ctx_mask=None, kv_cache=None,
                pos_offset=None):
        h, new_cache = self.attn(self.norm1(x), cos, sin, kv_cache, pos_offset)
        x = x + h                                              # pre-norm + residual
        if self.cross is not None and ctx is not None:
            x = x + self.cross(self.norm_x(x), ctx, ctx_mask)  # lyric conditioning
        x = x + self.ffn(self.norm2(x))
        return x, new_cache


class MusicTransformer(nn.Module):
    """
    Music LM over EnCodec tokens.
    Flash attention + RoPE + RMSNorm + fused QKV + T5 cross-attention.
    """
    def __init__(
        self,
        vocab_size:  int,
        d_model:     int   = 768,
        n_heads:     int   = 12,
        n_layers:    int   = 12,
        d_ff:        int   = 3072,
        max_seq_len: int   = 4096,
        dropout:     float = 0.1,
        d_text:      int   = None,
        rope_base:   float = 10000.0,
        grad_checkpoint: bool = False,
    ):
        super().__init__()
        self.d_model     = d_model
        self.max_seq_len = max_seq_len
        self.d_head      = d_model // n_heads
        self.rope_base   = rope_base
        # Recompute activations in the backward pass instead of storing them.
        # Activations dominate memory here (~10GB of an 11.7GB step at
        # seq_len=4096; params are only ~15%), so this is the lever that buys
        # longer context or bigger batches. Costs ~30% step time.
        self.grad_checkpoint = grad_checkpoint

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.drop      = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout, d_text)
            for _ in range(n_layers)
        ])

        self.norm_out = RMSNorm(d_model)
        # NOT tied to embedding -- see decision (8).
        self.decoder  = nn.Linear(d_model, vocab_size, bias=False)

        # Global std=0.02 init — applied after all modules constructed
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

        # AFTER the global loop, which would otherwise overwrite this: zero the
        # cross-attn output projections so conditioning contributes exactly 0 at
        # step 0. The model starts as a clean unconditional LM and learns to
        # open the lyric gate. Ordering matters -- see decision (7).
        for blk in self.blocks:
            if blk.cross is not None:
                nn.init.zeros_(blk.cross.out_proj.weight)

        # RoPE tables are buffers: no grad, but they follow .to(device) and are
        # excluded from the checkpoint (they're derivable).
        cos, sin = build_rope(max_seq_len, self.d_head, "cpu", rope_base)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, x, ctx=None, ctx_mask=None, kv_caches=None,
                return_caches: bool = False, pos_offset=None):
        """
        x:        [B, T] int64 token ids
        ctx:      [B, L, d_text] frozen T5 lyric embeddings, or None
        ctx_mask: [B, L] bool, True = real token
        returns:  [B, T, vocab_size] logits  (+ caches if return_caches)
        """
        B, T = x.shape
        n_past = kv_caches[0][0].shape[2] if kv_caches is not None else 0
        # With a sliding window the true position outruns the cache length, so
        # check the position we will actually rotate at against the RoPE table.
        rope_pos = n_past if pos_offset is None else pos_offset
        assert rope_pos + T <= self.max_seq_len, (
            f"position {rope_pos + T} exceeds RoPE table {self.max_seq_len}; "
            "enlarge max_seq_len or reset the position counter"
        )
        h = self.drop(self.embedding(x))

        # Checkpointing only applies while training: it trades compute for
        # memory in the BACKWARD pass, so it is pointless (and breaks the KV
        # cache) during inference.
        ckpt = (self.grad_checkpoint and self.training and kv_caches is None
                and torch.is_grad_enabled())

        new_caches = []
        for i, block in enumerate(self.blocks):
            if ckpt:
                # use_reentrant=False: the modern implementation, and the only
                # one that behaves with keyword args and no-grad buffers.
                h, c = torch.utils.checkpoint.checkpoint(
                    block, h, self.rope_cos, self.rope_sin, ctx, ctx_mask, None,
                    use_reentrant=False)
            else:
                h, c = block(h, self.rope_cos, self.rope_sin, ctx, ctx_mask,
                             kv_caches[i] if kv_caches is not None else None,
                             pos_offset)
            new_caches.append(c)

        logits = self.decoder(self.norm_out(h))
        return (logits, new_caches) if return_caches else logits

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def flops_per_token(self, ctx_len: int = 0) -> int:
        """Approximate FLOPs per token (forward only)."""
        d, n_layers = self.d_model, len(self.blocks)
        d_ff = self.blocks[0].ffn.fc1.weight.shape[0]
        attn_flops = 4 * d * d                 # QKV + out proj
        ffn_flops  = 4 * d * d_ff              # fc1 + fc2
        cross = 0
        if self.blocks[0].cross is not None and ctx_len:
            d_text = self.blocks[0].cross.kv_proj.weight.shape[1]
            cross = 2 * d * d + 4 * ctx_len * d_text * d + 4 * ctx_len * d
        return n_layers * (attn_flops + ffn_flops + cross)


# Back-compat alias: the original class name.
Transformer = MusicTransformer


if __name__ == "__main__":
    import ui
    torch.manual_seed(0)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ui.rule("transformer — regression tests")
    V, d_text = 16386, 768
    model = MusicTransformer(vocab_size=V, d_text=d_text).to(dev)

    x   = torch.randint(0, V, (2, 128), device=dev)
    ctx = torch.randn(2, 32, d_text, device=dev)
    msk = torch.ones(2, 32, dtype=torch.bool, device=dev)
    ui.kv_table([
        ("parameters", f"{model.count_parameters():,}"),
        ("FLOPs/token", f"{model.flops_per_token(ctx_len=256):,}"),
        ("shape", f"d_model {model.d_model}  blocks {len(model.blocks)}  vocab {V}"),
        ("forward", f"{tuple(x.shape)} -> {tuple(model(x, ctx, msk).shape)}"),
    ])

    # ---- REGRESSION TESTS: each guards a silent, measured failure ----
    model.eval()

    # (a) cross-attn output projections must still be zero AFTER the global
    #     std=0.02 loop. If the loop runs last, conditioning injects noise into
    #     every residual from step 0 and early training quietly degrades.
    for i, blk in enumerate(model.blocks):
        assert blk.cross.out_proj.weight.abs().max().item() == 0.0, \
            f"block {i}: global init overwrote cross-attn zero-init"
    ui.check(True, "cross-attn zero-init survives global init loop")

    # (b) zero-init means conditioning is a no-op at step 0: a fresh model must
    #     give identical logits with and without lyrics.
    with torch.no_grad():
        assert torch.allclose(model(x, ctx, msk), model(x, None, None), atol=1e-5)
    ui.check(True, "untrained model is conditioning-invariant", "gate starts closed")

    # (c) THE BIG ONE. Cached incremental decode must equal a single full
    #     forward. Measured: is_causal=True with q_len=1 is wrong by 4.2 and
    #     raises nothing -- training looks perfect, generation is garbage.
    with torch.no_grad():
        seq  = torch.randint(0, V, (1, 32), device=dev)
        full = model(seq, ctx[:1], msk[:1])

        caches, outs = None, []
        for t in range(32):
            lg, caches = model(seq[:, t:t + 1], ctx[:1], msk[:1],
                               kv_caches=caches, return_caches=True)
            outs.append(lg)
        inc = torch.cat(outs, dim=1)
        delta = (full - inc).abs().max().item()
    assert delta < 1e-4, f"KV cache diverges from full forward by {delta}"
    ui.check(True, "cached decode == full forward", f"max delta {delta:.2e} — the is_causal trap")

    # (d) RoPE shift-invariance: the property that makes sliding-window
    #     generation just "evict and continue". Verified: shifting q and k by a
    #     constant leaves attention scores unchanged.
    cos, sin = model.rope_cos, model.rope_sin
    q = torch.randn(1, 4, 16, model.d_head, device=dev)
    k = torch.randn(1, 4, 16, model.d_head, device=dev)
    s0 = apply_rope(q, cos, sin, 0)   @ apply_rope(k, cos, sin, 0).transpose(-1, -2)
    s1 = apply_rope(q, cos, sin, 1000) @ apply_rope(k, cos, sin, 1000).transpose(-1, -2)
    shift = (s0 - s1).abs().max().item()
    assert shift < 1e-2, f"RoPE not shift-invariant: {shift}"
    ui.check(True, "RoPE shift-invariant over 1000 positions", f"max delta {shift:.2e} — sliding window is free")

    ui.panel("all regression tests passed", style="green")
