"""
codec.py — EnCodec wrapper: audio <-> flat discrete token stream.

This is the highest-risk file in the project. Everything downstream trains on
whatever this produces, and every failure mode here is SILENT: a wrong
interleave order does not raise, it just produces noise the model dutifully
learns.

WHICH CODEC, AND WHY
--------------------
facebook/encodec_32khz -- the codec MusicGen is built on. Measured against the
alternatives on this machine:

                      48khz          24khz bw=3.0    32khz (chosen)
    duration          ~1s drift      exact           exact
    chunking          1s + overlap   none            none
    scales needed     yes            no              no
    recon MAE         --             0.0435          0.0224
    stereo @4096tok   --             6.83s           10.24s

It dominates on every axis at once. The 48khz model is natively stereo, which
looked attractive, but its 1s internal chunking with 1% overlap makes encode ->
decode drift ~1s and forces per-chunk scale bookkeeping -- a silent-failure
surface with no upside once 32khz exists.

STEREO
------
32khz is a MONO model. Stereo is done the way MusicGen does it: encode L and R
as a batch of two mono clips, then interleave the two token streams. Channels
stay coupled because they are adjacent in the sequence, so the model sees
[..q_L, q_R..] and learns the correlation directly.

THE CONTRACT
------------
    audio (C,N) --encode--> codes (C, n_q, T) --flatten--> flat (T*n_q*C,)
    flat (S,)   --unflatten--> codes (C, n_q, T) --decode--> audio (C,N)

VOCAB -- each (codebook, channel) pair gets a disjoint slice, so a token id
implies both which codebook and which channel it came from. An illegal
prediction is detectable instead of silently decoding as garbage:

    token(q, c, code) = (q * C + c) * codebook_size + code
    n_slots           = n_q * C
    vocab_size        = n_slots * codebook_size + 2      # + BOS, PAD

INTERLEAVE -- timestep-major, then codebook, then channel:

    [q0L@t0, q0R@t0, q1L@t0, q1R@t0, ..., q0L@t1, q0R@t1, ...]

Matches MusicGen's [1_L, 1_R, 2_L, 2_R, ...] and mirrors RVQ's residual
structure: predicting q1@t conditions on q0@t, exactly as the quantizer encoded
it. Codebook-major (all q0 for the whole clip, then all q1) would destroy
locality and make sliding-window generation incoherent.

MEASURED FACTS (facebook/encodec_32khz, verified here):
  - sr=32000, mono, frame_rate=50, codebook_size=2048, n_q=4
  - chunk_length_s=None -> encode() returns n_chunks=1, no chunk-axis games
  - normalize=False -> audio_scales are [None]; nothing to cache
  - stereo: 50 * 4 * 2 = 400 tok/s -> 4096 tokens = 10.24s of audio
"""

import numpy as np
import torch
from transformers import EncodecModel

import audio_io
from config import CFG

BOS_OFFSET = 0
PAD_OFFSET = 1


class Codec:
    def __init__(self, model_id: str = None, bandwidth: float = None,
                 stereo: bool = None, device: str = "cuda"):
        self.model_id = model_id or CFG.codec_id
        self.bandwidth = bandwidth if bandwidth is not None else CFG.bandwidth
        self.stereo = CFG.stereo if stereo is None else stereo
        self.device = device

        self.model = EncodecModel.from_pretrained(self.model_id).to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        c = self.model.config
        self.sr = c.sampling_rate
        self.codec_channels = c.audio_channels      # 1 for the 32khz model
        self.channels = 2 if self.stereo else 1     # what WE store/produce
        self.codebook_sz = c.codebook_size
        self.frame_rate = c.frame_rate
        self.normalize = c.normalize
        self.chunked = c.chunk_length_s is not None

        # n_q is DERIVED from a real encode. Hardcoding it is how you end up
        # training on a stream whose stride doesn't match the interleave.
        self.n_q = self._probe_n_q()

        self.n_slots = self.n_q * self.channels     # tokens per timestep
        self.vocab_size = self.n_slots * self.codebook_sz + 2
        self.bos = self.n_slots * self.codebook_sz + BOS_OFFSET
        self.pad = self.n_slots * self.codebook_sz + PAD_OFFSET
        self.tokens_per_sec = self.frame_rate * self.n_slots

        if self.codec_channels != 1 and self.stereo:
            raise ValueError(
                f"{self.model_id} is not mono; the batch-of-2 stereo trick "
                "assumes a mono codec. Use encodec_32khz."
            )
        # (n_q, C) offsets, flattened to slot order q-major then channel
        self._offsets = (
            torch.arange(self.n_slots, device=device) * self.codebook_sz
        ).view(1, self.n_q, self.channels, 1)

    def _probe_n_q(self) -> int:
        sil = torch.zeros(1, self.codec_channels, self.sr, device=self.device)
        with torch.no_grad():
            out = self.model.encode(sil, bandwidth=self.bandwidth)
        assert out.audio_codes.shape[0] == 1, (
            f"expected unchunked codec, got {out.audio_codes.shape[0]} chunks"
        )
        return int(out.audio_codes.shape[2])

    # ---- the flatten contract -------------------------------------------

    def flatten(self, codes: torch.Tensor) -> torch.Tensor:
        """(B, n_q, C, T) in [0, codebook_sz) -> (B, T*n_q*C) offset ids"""
        B, n_q, C, T = codes.shape
        assert n_q == self.n_q and C == self.channels, codes.shape
        assert codes.min() >= 0 and codes.max() < self.codebook_sz, "code out of range"
        x = codes + self._offsets.to(codes.device)
        # -> (B, T, n_q, C) then flatten: timestep-major, codebook, channel.
        # permute THEN reshape; reshape (not view) since permute breaks contiguity.
        return x.permute(0, 3, 1, 2).reshape(B, T * self.n_slots)

    def unflatten(self, flat: torch.Tensor) -> torch.Tensor:
        """(B, S) offset ids -> (B, n_q, C, T) raw codes. Inverse of flatten()."""
        B, S = flat.shape
        assert S % self.n_slots == 0, f"length {S} not divisible by n_slots={self.n_slots}"
        T = S // self.n_slots
        x = flat.view(B, T, self.n_q, self.channels).permute(0, 2, 3, 1)
        x = x - self._offsets.to(flat.device)
        # A model can still emit an id from the wrong slice; clamp so a bad
        # sample degrades one frame instead of crashing the decoder.
        return x.clamp_(0, self.codebook_sz - 1)

    def slot_at(self, step: int) -> int:
        return step % self.n_slots

    def logit_mask(self, step: int, device=None) -> torch.Tensor:
        """
        (vocab_size,) additive mask: 0 inside the slot legal at `step`, -inf
        elsewhere. Position i is always slot i % n_slots -- free information
        the sampler should never waste.
        """
        s = step % self.n_slots
        m = torch.full((self.vocab_size,), float("-inf"),
                       device=device or self.device)
        m[s * self.codebook_sz:(s + 1) * self.codebook_sz] = 0.0
        return m

    # ---- audio <-> tokens ------------------------------------------------

    # Seconds per encode call. Chunks are fed SEQUENTIALLY and concatenated:
    # chunk1 -> chunk2 -> chunk3 -> ... in order, so the token stream is the
    # song in order, start to finish, with no gaps.
    #
    # WHY CHUNK AT ALL: encoder memory scales linearly with track length.
    # MEASURED single-shot peaks: 152s track -> 12.77GB, 287s -> 23.89GB,
    # 350s -> 29.10GB. A 6-minute song cannot be encoded whole on a 16GB card,
    # full stop. EnCodec's own repo says it "does not implement smart handling
    # for long files... will not support this use case", and the 48kHz variant
    # chunks internally at 1s. Chunking is how this codec is meant to run.
    #
    # WHY 80s: accuracy is purely a function of boundary count. MEASURED token
    # match against a whole-track encode of a 152s song:
    #     20s chunks -> 83.9%   (7 boundaries)   1.57GB
    #     40s chunks -> 89.9%   (3 boundaries)   2.89GB
    #     80s chunks -> 95.8%   (1 boundary)     6.17GB
    #     whole      -> 100%    (0 boundaries)  12.77GB
    # 80s is the largest chunk that comfortably fits alongside everything else.
    #
    # The remaining ~4% are NOT errors in any audible sense: the residual
    # quantizer picks codes greedily, so a tiny float difference at a boundary
    # flips a code to a NEIGHBOURING centroid. Decoded audio differs by MAE
    # 0.0145 vs the codec's own 0.0901 reconstruction error -- 16% of loss we
    # already accept. Left-padding chunks with context only reaches ~98%, which
    # confirms quantizer sensitivity rather than missing receptive field.
    #
    # And it barely matters: training windows are seq_len tokens (41s), so with
    # ~4 boundaries in a 6-minute song, almost every window never spans one.
    ENCODE_CHUNK_S = 80

    @torch.no_grad()
    def encode_wav(self, wav: torch.Tensor, chunk_s: float = None) -> torch.Tensor:
        """(C, N) float32 -> flat (S,) int64 on cpu. Sequential chunks, in order."""
        assert wav.shape[0] == self.channels, f"expected {self.channels}ch, got {wav.shape}"
        chunk_s = chunk_s or self.ENCODE_CHUNK_S
        spf = self.sr // self.frame_rate                 # samples per frame
        step = int(chunk_s * self.frame_rate) * spf      # whole frames only
        N = wav.shape[1]

        parts = []
        for i in range(0, N, step):
            seg = wav[:, i:i + step]
            # Trailing scrap shorter than one frame (<20ms) is dropped, NOT
            # padded. Padding to fill a chunk would append silence, and the
            # model would dutifully learn "songs are followed by silence" at
            # every track boundary -- a real artifact for <20ms of nothing.
            if seg.shape[1] < spf:
                break
            # (C, N) -> (C, 1, N): a batch of C mono clips. MusicGen's stereo
            # path -- L and R encoded independently, interleaved after.
            x = seg[:, None, :].to(self.device)
            out = self.model.encode(x, bandwidth=self.bandwidth)
            codes = out.audio_codes                      # (1, C, n_q, T)
            assert codes.shape[0] == 1, f"unexpected chunking: {codes.shape}"
            parts.append(codes[0].permute(1, 0, 2).cpu())    # (n_q, C, T)
            del x, out, codes
        if not parts:
            raise ValueError(f"audio too short to encode: {N} samples")
        allc = torch.cat(parts, dim=-1)[None]            # (1, n_q, C, T)
        return self.flatten(allc.to(self.device))[0].cpu()

    @torch.no_grad()
    def encode_file(self, path) -> torch.Tensor:
        return self.encode_wav(audio_io.load(path, self.sr, self.channels))

    @torch.no_grad()
    def decode_flat(self, flat: torch.Tensor) -> torch.Tensor:
        """(S,) or (1,S) int64 -> (C, N) float32 cpu waveform."""
        if flat.dim() == 1:
            flat = flat[None]
        codes = self.unflatten(flat.to(self.device))    # (1, n_q, C, T)
        codes = codes[0].permute(1, 0, 2)[None]         # (1, C, n_q, T)
        wav = self.model.decode(codes, [None])[0]       # (C, 1, N)
        return wav[:, 0, :].cpu()                       # (C, N)

    def __repr__(self):
        return (f"Codec({self.model_id}, n_q={self.n_q}x{self.channels}ch, "
                f"vocab={self.vocab_size}, {self.sr}Hz, "
                f"{self.tokens_per_sec} tok/s)")


def _selftest():
    import ui
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    c = Codec(device=dev)
    ui.rule("codec — contract tests")
    ui.kv_table([
        ("model", c.model_id),
        ("stream", f"n_q={c.n_q} x {c.channels}ch = {c.n_slots} slots/timestep"),
        ("vocab", f"{c.vocab_size}  ({c.codebook_sz}/slot + BOS/PAD)"),
        ("rate", f"{c.frame_rate}Hz frames -> {c.tokens_per_sec} tok/s"),
        ("context", f"{CFG.max_seq_len} tok = {CFG.max_seq_len/c.tokens_per_sec:.2f}s audio"),
        ("clean", f"chunked={c.chunked}  normalize={c.normalize}"),
    ])
    assert not c.chunked, "chunked codec -- the whole point of 32khz is no chunking"
    assert not c.normalize, "normalizing codec would need scale bookkeeping"

    # 1. flatten/unflatten exact inverse, at a length that is not a round number
    codes = torch.randint(0, c.codebook_sz, (2, c.n_q, c.channels, 137), device=dev)
    flat = c.flatten(codes)
    assert flat.shape == (2, 137 * c.n_slots), flat.shape
    assert torch.equal(c.unflatten(flat), codes), "flatten/unflatten not inverse"
    ui.check(True, "flatten/unflatten exact inverse", f"{tuple(codes.shape)} <-> {tuple(flat.shape)}")

    # 2. interleave ORDER, element-wise. Catches a codebook-major or
    #    channel-major mistake, which test 1 cannot.
    for t in (0, 1, 42, 136):
        for q in range(c.n_q):
            for ch in range(c.channels):
                i = t * c.n_slots + q * c.channels + ch
                want = codes[0, q, ch, t].item() + (q * c.channels + ch) * c.codebook_sz
                assert flat[0, i].item() == want, f"interleave wrong t={t} q={q} ch={ch}"
    ui.check(True, "interleave order verified element-wise", "timestep -> codebook -> channel")

    # 3. every id lands in its own slot's slice
    for s in range(c.n_slots):
        ids = flat[:, s::c.n_slots]
        assert (ids >= s * c.codebook_sz).all() and (ids < (s + 1) * c.codebook_sz).all()
    assert flat.max() < c.vocab_size
    ui.check(True, f"vocab slicing disjoint across {c.n_slots} slots")

    # 4. logit mask selects exactly one slot
    for step in (0, 1, 5, 4095):
        m = c.logit_mask(step, dev)
        assert (m == 0).sum().item() == c.codebook_sz
        assert m[(step % c.n_slots) * c.codebook_sz] == 0
    ui.check(True, "logit mask selects exactly one slot")

    # 5. real audio round-trip, distinct L/R so a channel collapse is visible
    secs = 4.0
    t = torch.linspace(0, secs, int(c.sr * secs))
    wav = torch.stack([
        0.4 * torch.sin(2 * torch.pi * 440 * t) + 0.15 * torch.sin(2 * torch.pi * 880 * t),
        0.4 * torch.sin(2 * torch.pi * 554 * t) + 0.15 * torch.sin(2 * torch.pi * 1108 * t),
    ])[:c.channels]
    flat = c.encode_wav(wav)
    assert flat.dim() == 1 and flat.shape[0] % c.n_slots == 0
    back = c.decode_flat(flat)
    n = min(back.shape[1], wav.shape[1])
    mae = (back[:, :n] - wav[:, :n]).abs().mean().item()
    drift_ms = abs(back.shape[1] - wav.shape[1]) / c.sr * 1000
    ui.check(True, "audio round-trip",
             f"{secs}s -> {flat.shape[0]} tok -> {back.shape[1]/c.sr:.2f}s | "
             f"MAE {mae:.4f} vs absmean {wav.abs().mean():.4f} | drift {drift_ms:.1f}ms")
    assert back.shape[0] == c.channels, "lost channels"
    assert drift_ms < 20, f"duration drift {drift_ms:.0f}ms"
    assert back.abs().max() < 1.05, "near-clipping"
    assert mae < wav.abs().mean() * 0.4, f"reconstruction too poor: {mae}"

    if c.channels == 2:
        lr = (back[0, :n] - back[1, :n]).abs().mean().item()
        assert lr > 0.05, f"channels collapsed (L-R={lr:.4f})"
        ui.check(True, "stereo preserved", f"L-R delta {lr:.3f}")

    # 6. token counts must be exactly predictable -- data.py chunking depends on it
    for s in (1.0, 2.5, 7.0):
        w = torch.zeros(c.channels, int(c.sr * s))
        got = c.encode_wav(w).shape[0]
        want = int(s * c.frame_rate) * c.n_slots
        assert abs(got - want) <= c.n_slots, f"{s}s: got {got} tokens, expected ~{want}"
    ui.check(True, "token count exactly predictable", f"{c.tokens_per_sec} tok/s")

    ui.panel("codec contract verified", style="green")


if __name__ == "__main__":
    _selftest()
