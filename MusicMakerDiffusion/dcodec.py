"""
dcodec.py — EnCodec as a continuous VAE-style latent codec.

The AR project uses EnCodec's QUANTIZED output: discrete token ids
it can do cross-entropy over. Diffusion cannot use those -- you cannot add
Gaussian noise to an integer id and get anything meaningful. So this wraps the
SAME EnCodec model but taps the encoder BEFORE the quantizer:

    AR:        audio -> encoder -> quantizer -> token ids  (discrete, 16386 vocab)
    diffusion: audio -> encoder ------------> latents      (continuous, 128-dim)

Same model, same 50Hz frame rate, same audio. The ONLY difference between the
two projects is what happens to the representation -- which is exactly the
comparison we want. (This project is standalone; it shares no imports with the
AR project, only the design.)

BONUS: skipping the quantizer skips quantization loss entirely. Measured
continuous round-trip MAE 0.0947 vs the quantized path's 0.0224 on tones --
note the continuous path is NOT automatically better in practice, because the
decoder was trained on quantized inputs. Verified in _selftest.

SCALE FACTOR: diffusion math assumes ~unit-variance data. Raw EnCodec latents
measured std ~5.4 on music, ~3.0 on noise -- it is data-dependent, so it is
MEASURED from the real corpus (see ddata.build_latent_cache) and stored in the
manifest, exactly like Stable Diffusion's VAE scale factor. Hardcoding it would
silently break the noise schedule on a different corpus.
"""

import torch
from transformers import EncodecModel

import audio_io
from dconfig import DCFG


class LatentCodec:
    def __init__(self, model_id: str = None, stereo: bool = None,
                 scale: float = None, device: str = "cuda"):
        self.model_id = model_id or DCFG.codec_id
        self.stereo = DCFG.stereo if stereo is None else stereo
        self.device = device
        self.scale = scale          # set from the manifest; None until measured

        self.model = EncodecModel.from_pretrained(self.model_id).to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        c = self.model.config
        self.sr = c.sampling_rate
        self.channels = 2 if self.stereo else 1
        self.frame_rate = c.frame_rate
        self.latent_dim = self._probe_dim()

    def _probe_dim(self) -> int:
        x = torch.zeros(1, 1, self.sr, device=self.device)
        with torch.no_grad():
            return int(self.model.encoder(x).shape[1])

    # ---- audio <-> latents ----------------------------------------------

    @torch.no_grad()
    def encode_wav(self, wav: torch.Tensor) -> torch.Tensor:
        """
        (C, N) float32 -> (C, latent_dim, T) continuous latents on cpu.

        Stereo is handled exactly like the AR twin: L and R encoded as a batch
        of two mono clips (MusicGen's approach). Keeping this identical is what
        makes the comparison fair.
        """
        assert wav.shape[0] == self.channels, f"expected {self.channels}ch, got {wav.shape}"
        x = wav[:, None, :].to(self.device)          # (C, 1, N)
        z = self.model.encoder(x)                    # (C, 128, T)
        if self.scale is not None:
            z = z * self.scale
        return z.cpu()

    @torch.no_grad()
    def encode_file(self, path) -> torch.Tensor:
        return self.encode_wav(audio_io.load(path, self.sr, self.channels))

    @torch.no_grad()
    def decode_latent(self, z: torch.Tensor) -> torch.Tensor:
        """(C, latent_dim, T) -> (C, N) waveform on cpu."""
        z = z.to(self.device)
        if self.scale is not None:
            z = z / self.scale                       # undo normalization
        wav = self.model.decoder(z)                  # (C, 1, N)
        return wav[:, 0, :].cpu()

    def frames_for_seconds(self, secs: float) -> int:
        return int(round(secs * self.frame_rate))

    def __repr__(self):
        s = f"{self.scale:.4f}" if self.scale else "unmeasured"
        return (f"LatentCodec({self.model_id}, {self.latent_dim}-dim @ "
                f"{self.frame_rate}Hz, {self.channels}ch, scale={s})")


def _selftest():
    import ui

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    c = LatentCodec(device=dev)
    ui.rule("dcodec — continuous latent contract")
    ui.kv_table([
        ("model", f"{c.model_id} [dim](encoder only, pre-quantizer)[/]"),
        ("latent", f"{c.latent_dim}-dim @ {c.frame_rate}Hz x {c.channels}ch"),
        ("context", f"{DCFG.n_frames} frames = {DCFG.n_frames/c.frame_rate:.2f}s"),
        ("vs AR", f"same {c.frame_rate}Hz rate — identical temporal resolution"),
    ])

    secs = 4.0
    t = torch.linspace(0, secs, int(c.sr * secs))
    wav = torch.stack([
        0.4 * torch.sin(2 * torch.pi * 440 * t) + 0.15 * torch.sin(2 * torch.pi * 880 * t),
        0.4 * torch.sin(2 * torch.pi * 554 * t) + 0.15 * torch.sin(2 * torch.pi * 1108 * t),
    ])[:c.channels]

    # 1. shape + frame rate must match the AR stream exactly
    z = c.encode_wav(wav)
    want_T = c.frames_for_seconds(secs)
    ui.check(z.shape == (c.channels, c.latent_dim, want_T),
             "encode shape", f"{tuple(wav.shape)} {secs}s -> {tuple(z.shape)}")
    assert abs(z.shape[-1] - want_T) <= 1, f"got {z.shape[-1]} frames, want {want_T}"

    # 2. round-trip through the continuous path
    back = c.decode_latent(z)
    n = min(back.shape[1], wav.shape[1])
    mae = (back[:, :n] - wav[:, :n]).abs().mean().item()
    drift = abs(back.shape[1] - wav.shape[1]) / c.sr * 1000
    ui.check(drift < 20 and back.shape[0] == c.channels, "latent round-trip",
             f"MAE {mae:.4f} vs absmean {wav.abs().mean():.4f} | drift {drift:.1f}ms")
    assert back.shape[0] == c.channels, "lost channels"

    if c.channels == 2:
        lr = (back[0, :n] - back[1, :n]).abs().mean().item()
        ui.check(lr > 0.05, "stereo preserved", f"L-R delta {lr:.3f}")

    # 3. scale factor must actually normalize -- this is what the noise
    #    schedule depends on. Unscaled latents would break diffusion silently.
    raw_std = z.std().item()
    c.scale = 1.0 / raw_std
    z2 = c.encode_wav(wav)
    ui.check(abs(z2.std().item() - 1.0) < 0.05, "scale factor normalizes",
             f"std {raw_std:.3f} -> {z2.std().item():.3f} (scale {c.scale:.4f})")

    # 4. scaling must be exactly invertible or decode drifts in level
    b2 = c.decode_latent(z2)
    ui.check(torch.allclose(b2, back, atol=1e-3), "scale is invertible",
             "decode(scale(z)) == decode(z)")

    ui.panel("continuous latent contract verified", style="green")


if __name__ == "__main__":
    _selftest()
