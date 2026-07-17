r"""
ceiling.py — hear what the model actually hears.

    python ceiling.py "..\Grunge\Alice in Chains - Facelift\02 - Alice in Chains - Man in the Box.mp3"
    python ceiling.py <track.mp3> --seconds 30 --start 60 --out ceiling.mp3

Run with no arguments to pick a track from your library at random.

Encodes a track through EnCodec and decodes it straight back -- no model
involved. The result is the QUALITY CEILING: the best your trained model could
ever sound, because it can only ever predict these tokens.

Your mp3s are 320kbps. The codec compresses to ~2.2kbps -- a further ~145x --
turning one second of stereo into 400 integers, each an index into a 2048-entry
codebook. Measured reconstruction error is MAE ~0.022 against a signal
averaging ~0.25, i.e. ~9%.

So if generated audio sounds a bit smeared -- cymbals losing shimmer, guitar
grain, soft sibilance -- compare it to THIS file before blaming the model.
Anything this file loses is gone before training even starts.

Writes both the original slice and the round-trip so you can A/B them.
"""

import argparse
from pathlib import Path

import torch

import audio_io
import ui


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("track", nargs="?", default=None,
                    help="audio file; omit to pick one from --music-dir at random")
    ap.add_argument("--music-dir", default="../Grunge",
                    help="library to pick from when no track is given")
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--start", type=float, default=0.0, help="skip N seconds in")
    ap.add_argument("--out", default="ceiling.mp3")
    a = ap.parse_args()

    import transformers
    transformers.logging.set_verbosity_error()
    from codec import Codec

    if a.track:
        src = Path(a.track)
        if not src.exists():
            raise SystemExit(f"not found: {src}")
    else:
        import random
        pool = [p for p in Path(a.music_dir).rglob("*")
                if p.suffix.lower() in (".mp3", ".wav", ".flac")]
        if not pool:
            raise SystemExit(f"no audio in {a.music_dir} — pass a track, or --music-dir")
        src = random.choice(pool)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    c = Codec(device=dev)

    ui.rule("codec ceiling — what the model actually hears")
    wav = audio_io.load(src, c.sr, c.channels)
    i0 = int(a.start * c.sr)
    wav = wav[:, i0:i0 + int(a.seconds * c.sr)]

    flat = c.encode_wav(wav)
    back = c.decode_flat(flat)

    n = min(back.shape[1], wav.shape[1])
    mae = (back[:, :n] - wav[:, :n]).abs().mean().item()
    ref = wav.abs().mean().item()

    orig_p = Path(a.out).with_name(Path(a.out).stem + "_original.mp3")
    audio_io.save(wav, orig_p, c.sr)
    audio_io.save(back, a.out, c.sr)

    secs = wav.shape[1] / c.sr
    mp3_bits = src.stat().st_size * 8 / (audio_io.duration_s(src) or 1)
    ui.kv_table([
        ("track", src.name[:48]),
        ("slice", f"{a.start:.0f}s -> {a.start+secs:.0f}s  ({secs:.1f}s)"),
        ("your mp3", f"~{mp3_bits/1000:.0f} kbps  =  "
                     f"{wav.shape[1]*c.channels:,} samples"),
        ("what the model sees", f"{len(flat):,} integers  "
                                f"[dim]({c.tokens_per_sec}/s, each 0-{c.codebook_sz-1})[/]"),
        ("compression", f"[bold]{wav.shape[1]*c.channels/len(flat):.0f}x[/] fewer numbers"),
        ("reconstruction", f"MAE {mae:.4f} vs signal {ref:.4f}  "
                           f"= [bold]{mae/ref*100:.1f}%[/] error"),
    ])
    ui.panel(f"[bold]{orig_p.name}[/]  — your audio\n"
             f"[bold]{Path(a.out).name}[/]  — after the codec\n\n"
             f"[dim]A/B them. The difference is the ceiling: your model can\n"
             f"never sound better than the second file.[/]",
             title="wrote", style="green")


if __name__ == "__main__":
    main()
