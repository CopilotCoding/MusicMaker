r"""
dac_codec.py — Descript Audio Codec wrapped in the exact interface codec.py
proved out: flatten contract, banded vocab, BOS/PAD, chunked encode.

    python dac_codec.py       # self-tests (flatten round-trip + audio A/B)

WHY DAC: measured by ear on this corpus — EnCodec 2.2kbps makes vocals warble
("almost autotuned"); DAC ~8kbps was "basically the same as the original".
The codec is the ceiling on everything trained above it, so the ceiling moved.

DERIVED, not assumed (the n_q lesson): codebooks, codebook size, sample rate
and frame rate are probed from a real encode at init. Expected: 9 codebooks x
1024 entries at 44.1kHz, ~86.13 frames/sec -> 18 stereo slots,
vocab 18*1024+2 = 18434. Note frame_rate is NOT an integer (44100/512);
keep arithmetic in frames and only convert to seconds for display.

Stereo = L/R as a batch of two mono clips, interleaved timestep-major ->
codebook -> channel: [q0L, q0R, q1L, q1R, ...] — the identical contract to
the EnCodec path, so delay.py's grid builder works unchanged (it is fully
parameterized by n_slots/codebook_sz).
"""

import torch

import audio_io
import ui


class DacCodec:
    """Same public surface as codec.Codec: n_q, channels, n_slots,
    codebook_sz, vocab_size, bos, pad, sr, frame_rate, tokens_per_sec,
    flatten/unflatten, encode_wav/encode_file, decode_flat."""

    # Seconds per encode call. Conv codecs scale activation memory with
    # length (EnCodec lesson: 350s single-shot = 29GB). 20s chunks were
    # MEASURED fine on this card at 44.1kHz in the ceiling test.
    ENCODE_CHUNK_S = 20

    def __init__(self, model_id: str = "descript/dac_44khz",
                 device: str = "cuda"):
        import transformers
        transformers.logging.set_verbosity_error()
        from transformers import DacModel

        self.model_id = model_id
        self.device = device
        self.model = DacModel.from_pretrained(model_id).to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        c = self.model.config
        self.sr = c.sampling_rate
        self.codebook_sz = c.codebook_size
        # probe a real encode: n_q and frame rate are facts, not config reads
        with torch.no_grad():
            enc = self.model.encode(torch.zeros(1, 1, self.sr, device=device))
        self.n_q = enc.audio_codes.shape[1]
        self.frame_rate = self.sr / (self.sr // enc.audio_codes.shape[-1]) \
            if enc.audio_codes.shape[-1] else 0
        # exact hop from the probe: frames per 1s of samples
        self._hop = round(self.sr / enc.audio_codes.shape[-1])
        self.frame_rate = self.sr / self._hop            # 86.1328125 for 44.1k

        self.channels = 2
        self.stereo = True
        self.n_slots = self.n_q * self.channels
        self.vocab_size = self.n_slots * self.codebook_sz + 2
        self.bos = self.n_slots * self.codebook_sz
        self.pad = self.bos + 1
        self.tokens_per_sec = self.frame_rate * self.n_slots
        self._offsets = (torch.arange(self.n_slots) * self.codebook_sz) \
            .view(1, self.n_q, self.channels, 1)

    # ---- the flatten contract (identical math to codec.Codec) -----------
    def flatten(self, codes: torch.Tensor) -> torch.Tensor:
        """(B, n_q, C, T) -> (B, T*n_slots), timestep-major, banded ids."""
        B, n_q, C, T = codes.shape
        x = codes + self._offsets.to(codes.device)
        return x.permute(0, 3, 1, 2).reshape(B, T * self.n_slots)

    def unflatten(self, flat: torch.Tensor) -> torch.Tensor:
        B, S = flat.shape
        assert S % self.n_slots == 0
        T = S // self.n_slots
        x = flat.view(B, T, self.n_q, self.channels).permute(0, 2, 3, 1)
        return (x - self._offsets.to(flat.device)).clamp_(0, self.codebook_sz - 1)

    def slot_at(self, step: int) -> int:
        return step % self.n_slots

    # ---- audio <-> tokens ------------------------------------------------
    @torch.no_grad()
    def encode_wav(self, wav: torch.Tensor) -> torch.Tensor:
        """(2, N) float at self.sr -> flat (S,) int64. Chunked sequentially."""
        assert wav.shape[0] == self.channels
        chunk = int(self.ENCODE_CHUNK_S * self.sr)
        parts = []
        for lo in range(0, wav.shape[1], chunk):
            seg = wav[:, lo:lo + chunk]
            if seg.shape[1] < self._hop:            # sub-frame tail: nothing
                break
            x = seg[:, None, :].to(self.device)     # (2, 1, n) L/R batch
            enc = self.model.encode(x)
            parts.append(enc.audio_codes)           # (2, n_q, t)
            if self.device == "cuda":
                torch.cuda.empty_cache()
        codes = torch.cat(parts, dim=-1)            # (2, n_q, T)
        codes = codes.permute(1, 0, 2)[None]        # (1, n_q, C, T)
        return self.flatten(codes.to("cpu"))[0]

    def encode_file(self, path) -> torch.Tensor:
        wav = audio_io.load(path, target_sr=self.sr, target_ch=self.channels)
        return self.encode_wav(wav)

    @torch.no_grad()
    def decode_flat(self, flat: torch.Tensor) -> torch.Tensor:
        """(B, S) or (S,) flat ids -> (C, N) float waveform."""
        if flat.dim() == 1:
            flat = flat[None]
        codes = self.unflatten(flat.to(self.device))    # (1, n_q, C, T)
        lr = codes[0].permute(1, 0, 2)                  # (C, n_q, T)
        chunk_t = int(self.ENCODE_CHUNK_S * self.frame_rate)
        outs = []
        for lo in range(0, lr.shape[-1], chunk_t):
            dec = self.model.decode(audio_codes=lr[:, :, lo:lo + chunk_t])
            y = dec.audio_values if hasattr(dec, "audio_values") else dec[0]
            outs.append(y.squeeze(1) if y.dim() == 3 else y)
            if self.device == "cuda":
                torch.cuda.empty_cache()
        return torch.cat(outs, dim=-1).cpu()            # (C, N)

    def __repr__(self):
        return (f"DacCodec({self.model_id}, n_q={self.n_q}x{self.channels}ch, "
                f"cb={self.codebook_sz}, vocab={self.vocab_size}, "
                f"{self.frame_rate:.2f} fps, {self.tokens_per_sec:.0f} tok/s)")


def make_codec(kind: str, device: str = "cuda"):
    """'dac' -> DacCodec, 'encodec' -> the original codec.Codec."""
    if kind == "dac":
        return DacCodec(device=device)
    from codec import Codec
    return Codec(device=device)


if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ui.rule("dac_codec self-tests")
    c = DacCodec(device=dev)
    ui.log(repr(c))

    # 1. flatten contract: exact round-trip on random codes
    torch.manual_seed(0)
    codes = torch.randint(0, c.codebook_sz, (2, c.n_q, c.channels, 137))
    assert torch.equal(c.unflatten(c.flatten(codes)), codes)
    # spot-check banding: flat[t*K+k] == codes[q,ch,t] + k*cb
    f = c.flatten(codes)[0]
    t, q, ch = 5, 3, 1
    k = q * c.channels + ch
    assert f[t * c.n_slots + k] == codes[0, q, ch, t] + k * c.codebook_sz
    ui.log("[green]flatten contract exact[/]")

    # 2. audio round-trip on a real second of noise-burst audio
    wav = torch.randn(2, c.sr * 2) * 0.1
    flat = c.encode_wav(wav)
    assert flat.shape[0] % c.n_slots == 0
    assert int(flat.max()) < c.vocab_size - 2
    back = c.decode_flat(flat)
    ui.log(f"[green]encode->decode OK[/]  {flat.shape[0]} tokens -> "
           f"{back.shape[-1]/c.sr:.2f}s ({back.shape[0]}ch)")
    ui.panel(f"n_q={c.n_q} cb={c.codebook_sz} slots={c.n_slots} "
             f"vocab={c.vocab_size}\nframe_rate={c.frame_rate:.4f} "
             f"(NOT integer — keep arithmetic in frames)\n"
             f"16384-step window = {16384/c.frame_rate:.0f}s audio",
             title="derived facts", style="green")
