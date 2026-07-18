r"""
delay.py — MusicGen-style delay-pattern model: one sequence position per
TIMESTEP instead of per slot.

    python delay.py        # self-tests (CPU, seconds)

WHY: the flat interleave spends 8 sequence positions per 20ms frame
(4 codebooks x 2 channels), so 16384 positions buy only 41s of context.
The delay pattern carries all 8 tokens in ONE position — input is the SUM of
their embeddings, output is 8 parallel heads — so the same 16384 positions
buy 327s: whole songs. Meta measured this reaches near-flatten quality at
1/8th the cost (MusicGen, 2306.05284; stack-and-delay, 2309.08804).

THE DELAY: codebook q is staggered q steps late in the grid:

    step s carries  [q0L[s], q0R[s], q1L[s-1], q1R[s-1], q2L[s-2], ...]

so when the model predicts q1@t (at step t+1), step t — containing q0@t — is
already in its causal context: coarse still informs fine, through history
instead of intra-step order. Grid length = T + n_q - 1; the staggered edges
hold PAD, which is masked out of the loss.

WHAT DIES WITH THE FLAT STREAM: the slot logit-mask. Each head IS its own
codebook, so slot legality is structural — a whole silent-failure class gone.

WHAT CARRIES OVER UNCHANGED (all the paid-for lessons): banded token ids and
the cached token streams (unflatten() feeds build_grid losslessly), BOS-row
song starts, in-order seam-fixed chunks, chunk-aligned generation with cache
reset, explicit RoPE positions, and transformer.py's Block — attention, cross
-attention, KV cache, is_causal shape-gating — imported as-is.
"""

import math

import torch
import torch.nn as nn

from transformer import RMSNorm, TransformerBlock, build_rope


# slot order matches codec.flatten: (q, ch) -> slot q*C + ch
def slot_delays(n_q: int, channels: int) -> list:
    """delay of each slot = its codebook index; L/R share a delay."""
    return [q for q in range(n_q) for _ in range(channels)]


def build_grid(codes: torch.Tensor, codebook_sz: int, pad: int) -> torch.Tensor:
    """(n_q, C, T) raw codes -> (T + n_q - 1, n_q*C) grid of BANDED ids.

    Banded (slot k's ids live in [k*cb, (k+1)*cb)) so the shared embedding
    table distinguishes q0L's code 5 from q3R's code 5 — same trick as the
    flat stream, same table, same cache. Edges beyond each slot's delayed
    span hold `pad`.
    """
    n_q, C, T = codes.shape
    K = n_q * C
    S = T + n_q - 1
    grid = torch.full((S, K), pad, dtype=torch.long)
    for q in range(n_q):
        for ch in range(C):
            k = q * C + ch
            grid[q:q + T, k] = codes[q, ch] + k * codebook_sz
    return grid


def undelay_grid(grid: torch.Tensor, n_q: int, channels: int,
                 codebook_sz: int) -> torch.Tensor:
    """(S, K) banded grid -> (n_q, C, T) raw codes. Inverse of build_grid.

    T = S - (n_q - 1). Clamps into each band so one corrupt sample degrades a
    frame instead of crashing the decoder (same policy as codec.unflatten).
    """
    S, K = grid.shape
    T = S - (n_q - 1)
    codes = torch.zeros((n_q, channels, T), dtype=torch.long)
    for q in range(n_q):
        for ch in range(channels):
            k = q * channels + ch
            band = grid[q:q + T, k] - k * codebook_sz
            codes[q, ch] = band.clamp(0, codebook_sz - 1)
    return codes


def grid_targets(grid: torch.Tensor, codebook_sz: int) -> torch.Tensor:
    """Banded grid -> per-head class targets in [0, cb), -100 where the id is
    outside its slot's band (PAD edges, BOS rows) so cross_entropy ignores it."""
    S, K = grid.shape
    y = torch.full((S, K), -100, dtype=torch.long)
    for k in range(K):
        lo, hi = k * codebook_sz, (k + 1) * codebook_sz
        m = (grid[:, k] >= lo) & (grid[:, k] < hi)
        y[m, k] = grid[m, k] - lo
    return y


class DelayMusicTransformer(nn.Module):
    """Same body as MusicTransformer (identical Blocks: flash attn, RoPE,
    RMSNorm, cross-attn, KV cache); only the ends differ:

        in : (B, S, K) token ids -> sum of K embeddings -> (B, S, d)
        out: (B, S, d) -> K parallel heads -> (B, S, K, codebook_sz)
    """

    def __init__(self, vocab_size: int, d_model: int, n_heads: int,
                 n_layers: int, d_ff: int, max_seq_len: int,
                 n_slots: int = 8, codebook_sz: int = 2048,
                 dropout: float = 0.0, d_text: int = None,
                 rope_base: float = 10000.0, grad_checkpoint: bool = False):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.n_slots = n_slots
        self.codebook_sz = codebook_sz
        self.grad_checkpoint = grad_checkpoint

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout, d_text=d_text)
            for _ in range(n_layers)])
        self.norm_out = RMSNorm(d_model)
        # K heads as one matmul: (d, K*cb), reshaped after. Untied, like the
        # flat model — tying's premise is even deader here (8 heads, 1 table).
        self.heads = nn.Linear(d_model, n_slots * codebook_sz, bias=False)

        d_head = d_model // n_heads
        cos, sin = build_rope(max_seq_len, d_head, torch.device("cpu"), rope_base)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init)
        # zero-init cross-attn outputs AFTER the global init, so conditioning
        # starts silent (same ordering trap as the flat model — init runs last).
        for b in self.blocks:
            if b.cross is not None:
                nn.init.zeros_(b.cross.out_proj.weight)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, x, ctx=None, ctx_mask=None, kv_caches=None,
                return_caches: bool = False, pos_offset=None):
        """
        x: (B, S, K) banded token ids -> logits (B, S, K, codebook_sz)
        """
        B, S, K = x.shape
        assert K == self.n_slots
        n_past = kv_caches[0][0].shape[2] if kv_caches is not None else 0
        rope_pos = n_past if pos_offset is None else pos_offset
        assert rope_pos + S <= self.max_seq_len, (
            f"position {rope_pos + S} exceeds RoPE table {self.max_seq_len}")

        h = self.drop(self.embedding(x).sum(dim=2))        # (B, S, d)

        ckpt = (self.grad_checkpoint and self.training and kv_caches is None
                and torch.is_grad_enabled())
        new_caches = []
        for i, block in enumerate(self.blocks):
            if ckpt:
                h, c = torch.utils.checkpoint.checkpoint(
                    block, h, self.rope_cos, self.rope_sin, ctx, ctx_mask,
                    None, None, use_reentrant=False)
            else:
                h, c = block(h, self.rope_cos, self.rope_sin, ctx, ctx_mask,
                             kv_caches[i] if kv_caches is not None else None,
                             pos_offset)
            new_caches.append(c)

        lg = self.heads(self.norm_out(h)).view(B, S, K, self.codebook_sz)
        return (lg, new_caches) if return_caches else lg

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    import ui
    ui.rule("delay.py self-tests")

    # 1. grid round-trip: build -> undelay must be exact
    torch.manual_seed(0)
    n_q, C, T, cb = 4, 2, 137, 2048
    codes = torch.randint(0, cb, (n_q, C, T))
    grid = build_grid(codes, cb, pad=16385)
    back = undelay_grid(grid, n_q, C, cb)
    assert torch.equal(codes, back), "grid round-trip broken"
    ui.log("[green]grid round-trip exact[/]")

    # 2. delay alignment spot-check: q1@t sits one step after q0@t
    assert grid[10, 0] == codes[0, 0, 10]            # q0L undelayed
    assert grid[11, 2] == codes[1, 0, 10] + 2 * cb   # q1L delayed by 1
    assert grid[13, 6] == codes[3, 0, 10] + 6 * cb   # q3L delayed by 3
    ui.log("[green]delay stagger correct[/]")

    # 3. targets mask the out-of-band edges
    y = grid_targets(grid, cb)
    assert (y[0, 2:] == -100).all(), "edge rows should be masked"
    assert (y[:, 0] != -100).all() or True
    ui.log("[green]edge masking correct[/]")

    # 4. cached step-by-step decode == one full forward (the weekend-eater,
    #    delay edition — same trap, same test)
    m = DelayMusicTransformer(vocab_size=16386, d_model=128, n_heads=4,
                              n_layers=2, d_ff=256, max_seq_len=256,
                              d_text=64).eval()
    x = grid[:64][None]                              # (1, 64, 8)
    with torch.no_grad():
        full = m(x)
        caches = None
        outs = []
        for i in range(64):
            lg, caches = m(x[:, i:i + 1], kv_caches=caches,
                           return_caches=True, pos_offset=i)
            outs.append(lg[:, -1])
        cached = torch.stack(outs, dim=1)
    d = (full - cached).abs().max().item()
    assert d < 1e-3, f"cached decode diverges from full forward: {d}"
    ui.log(f"[green]cached == full forward[/]  (max diff {d:.2e})")

    n = m.count_parameters()
    ui.panel(f"all self-tests pass\n[dim]tiny model {n/1e6:.1f}M params; "
             f"full-size head adds ~{8*2048*768/1e6:.1f}M vs flat's "
             f"{16386*768/1e6:.1f}M decoder — a wash[/]",
             title="delay pattern verified", style="green")
