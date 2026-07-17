"""
audio_io.py — audio load/save without torchaudio.

torchaudio 2.11 reworked its backend API (list_audio_backends() is gone), so
dispatch is a moving target. soundfile handles wav/flac directly; ffmpeg
handles mp3 and all resampling. Both are confirmed present.

Convention: waveforms are torch float32 (C, N) in [-1, 1], channels-first.
soundfile is channels-LAST (N, C) -- everything transposes at the boundary.
"""

import subprocess
import numpy as np
import soundfile as sf
import torch

from dconfig import DCFG as CFG

_FFMPEG = "ffmpeg"
_FFPROBE = "ffprobe"


def _run(cmd: list, stdin_bytes: bytes | None = None) -> bytes:
    p = subprocess.run(
        cmd, input=stdin_bytes, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    if p.returncode != 0:
        tail = p.stderr.decode("utf-8", "replace")[-800:]
        raise RuntimeError(f"{cmd[0]} failed ({p.returncode}):\n{tail}")
    return p.stdout


def load(path, target_sr: int, target_ch: int = 2) -> torch.Tensor:
    """
    Decode any format to (C, N) float32 in [-1, 1], resampled and channel-forced.

    Routes everything through ffmpeg: it reads mp3/wav/flac uniformly and does
    resampling + channel mapping in the same pass, so there is no separate
    resampler to disagree with.
    """
    raw = _run([
        _FFMPEG, "-v", "error",
        "-i", str(path),
        "-f", "f32le",              # raw float32 little-endian on stdout
        "-acodec", "pcm_f32le",
        "-ar", str(target_sr),      # resample
        "-ac", str(target_ch),      # mono->stereo dup, >2ch downmix
        "pipe:1",
    ])
    if len(raw) == 0:
        raise RuntimeError(f"decoded 0 bytes from {path}")
    x = np.frombuffer(raw, dtype=np.float32)
    # ffmpeg emits interleaved frames -> (N, C) -> transpose to (C, N)
    x = x.reshape(-1, target_ch).T.copy()
    return torch.from_numpy(x)


def save(wav: torch.Tensor, path, sr: int, fmt: str | None = None,
         peak_norm: float | None = None) -> None:
    """
    Write (C, N) float32 to wav / mp3 / flac. Format inferred from suffix
    unless given explicitly.

    peak_norm defaults to OFF: normalizing on every save silently rewrites
    levels and would erase dynamics if it ever touched training audio. Pass
    peak_norm=CFG.peak_norm explicitly for generated output, where the decoder
    has no reason to land at a sane level.
    """
    path = str(path)
    fmt = (fmt or path.rsplit(".", 1)[-1]).lower()
    if wav.dim() == 1:
        wav = wav[None, :]
    wav = wav.detach().float().cpu()

    if peak_norm:
        peak = wav.abs().max()
        if peak > 1e-8:
            wav = wav / peak * peak_norm
    wav = wav.clamp(-1.0, 1.0)

    C = wav.shape[0]
    arr = wav.numpy()

    if fmt in ("wav", "flac"):
        # soundfile wants (N, C) -- passing (C, N) silently writes an
        # N-channel file, which is the classic version of this bug.
        sf.write(path, arr.T, sr, subtype="PCM_16", format=fmt.upper())
    elif fmt == "mp3":
        _run([
            _FFMPEG, "-v", "error", "-y",
            "-f", "f32le", "-ar", str(sr), "-ac", str(C),
            "-i", "pipe:0",
            "-c:a", "libmp3lame", "-b:a", CFG.mp3_bitrate,
            path,
        ], stdin_bytes=arr.T.tobytes())   # interleave before piping
    else:
        raise ValueError(f"unsupported format {fmt!r} (want wav/mp3/flac)")


def duration_s(path) -> float:
    out = _run([
        _FFPROBE, "-v", "error", "-show_entries", "format=duration",
        "-of", "default=nw=1:nk=1", str(path),
    ])
    return float(out.strip())


if __name__ == "__main__":
    import tempfile, os

    sr = 48000
    t = torch.linspace(0, 2.0, sr * 2)
    # distinct L/R so a channel swap or collapse is visible
    tone = torch.stack([
        0.5 * torch.sin(2 * torch.pi * 440 * t),
        0.5 * torch.sin(2 * torch.pi * 660 * t),
    ])
    print(f"source: {tuple(tone.shape)} @ {sr}Hz, 2.0s")

    d = tempfile.mkdtemp()
    # wav/flac are lossless -> only 16-bit quantization should show up.
    # mp3 is lossy and perceptual, so it gets a far looser bound.
    tol = {"wav": 1e-3, "flac": 1e-3, "mp3": 0.05}
    for fmt in ("wav", "flac", "mp3"):
        p = os.path.join(d, f"t.{fmt}")
        save(tone, p, sr, fmt)                 # no peak_norm -> levels preserved
        back = load(p, sr, 2)
        n = min(back.shape[1], tone.shape[1])
        err = (back[:, :n] - tone[:, :n]).abs().mean().item()
        # channel identity: L and R must stay distinct and in order
        lr_delta = (back[0, :n] - back[1, :n]).abs().mean().item()
        size_kb = os.path.getsize(p) / 1024
        print(f"  {fmt:4s} {size_kb:7.1f}KB  shape={tuple(back.shape)}  "
              f"dur={back.shape[1]/sr:.2f}s  peak={back.abs().max():.3f}  "
              f"MAE={err:.6f}  L-R={lr_delta:.3f}")
        assert back.shape[0] == 2, f"{fmt}: lost stereo"
        assert lr_delta > 0.1, f"{fmt}: channels collapsed or swapped"
        assert err < tol[fmt], f"{fmt}: MAE {err:.5f} exceeds {tol[fmt]}"
        assert abs(back.abs().max() - 0.5) < 0.02, f"{fmt}: level altered"

    # peak_norm is opt-in and must actually work when asked for
    p = os.path.join(d, "n.wav")
    save(tone, p, sr, "wav", peak_norm=0.95)
    assert abs(load(p, sr, 2).abs().max() - 0.95) < 0.01
    print("OK — lossless stays lossless, levels preserved, peak_norm opt-in")
