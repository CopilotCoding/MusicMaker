r"""
dac_ceiling.py — hear the codec ceiling: EnCodec 2.2kbps vs DAC ~8kbps, A/B/C.

    python dac_ceiling.py --song "..\Grunge\...\track.mp3"
    python dac_ceiling.py --song "..." --start 60 --seconds 60

Writes three wavs of the SAME slice:
    <stem>__orig.wav      the rip, untouched (just resampled)
    <stem>__encodec.wav   round-trip through the current codec (2.2 kbps)
    <stem>__dac.wav       round-trip through Descript DAC 44.1kHz (~8 kbps)

Neither round-trip involves any trained model of ours — this is purely the
ceiling either codec imposes on everything downstream. If DAC's vocals are
clearly better, upgrading the codec (9x1024 codebooks, 86Hz, 18 stereo slots)
buys that ceiling for the whole project — at the cost of shorter context per
position and a full re-verification pass. If it's marginal, we keep 328s
windows with zero regret. Ears decide.

DAC is mono; stereo = encode L/R as a batch of two, exactly the trick the
32kHz EnCodec path already uses.
"""

import argparse
from pathlib import Path

import torch

import audio_io
import ui


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--song", required=True)
    ap.add_argument("--start", type=float, default=60.0,
                    help="slice start (sec) — default 60 to land on vocals")
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    a = ap.parse_args()

    song = Path(a.song)
    if not song.exists():
        raise SystemExit(f"no such file: {song}")
    dev = ("cuda" if torch.cuda.is_available() else "cpu") \
        if a.device == "auto" else a.device
    out = Path(__file__).parent
    stem = "".join(c if c.isalnum() or c in "-_" else "_" for c in song.stem)[:40]

    import transformers
    transformers.logging.set_verbosity_error()

    ui.rule("codec ceiling A/B/C — same slice, three worlds")

    # ---- original slice at 44.1k (the reference) ----
    wav44 = audio_io.load(song, target_sr=44100, target_ch=2)   # (2, N)
    s0, s1 = int(a.start * 44100), int((a.start + a.seconds) * 44100)
    slice44 = wav44[:, s0:s1]
    p_orig = out / f"{stem}__orig.wav"
    audio_io.save(slice44, p_orig, 44100)

    # ---- EnCodec 32k round-trip (the current ceiling) ----
    from codec import Codec
    codec = Codec(device=dev)
    wav32 = audio_io.load(song, target_sr=codec.sr, target_ch=2)
    t0, t1 = int(a.start * codec.sr), int((a.start + a.seconds) * codec.sr)
    flat = codec.encode_wav(wav32[:, t0:t1]) if hasattr(codec, "encode_wav") \
        else None
    if flat is None:
        # fall back: encode via file path API on a temp slice
        import tempfile
        tmp = Path(tempfile.mkstemp(suffix=".wav")[1])
        audio_io.save(wav32[:, t0:t1], tmp, codec.sr)
        flat = codec.encode_file(tmp)
        tmp.unlink(missing_ok=True)
    enc_wav = codec.decode_flat(flat[None].to(dev) if flat.dim() == 1 else flat)
    p_enc = out / f"{stem}__encodec.wav"
    audio_io.save(enc_wav.cpu(), p_enc, codec.sr)
    del codec
    if dev == "cuda":
        torch.cuda.empty_cache()

    # ---- DAC 44.1k round-trip (the candidate ceiling) ----
    from transformers import DacModel
    dac = DacModel.from_pretrained("descript/dac_44khz").to(dev).eval()
    for p in dac.parameters():
        p.requires_grad_(False)
    x = slice44.to(dev)[:, None, :]              # (2, 1, N): L/R as batch of 2
    # chunk to keep conv activations sane on long slices (measured lesson from
    # EnCodec: conv codecs scale memory with length)
    CH = 20 * 44100
    outs = []
    with torch.no_grad():
        for lo in range(0, x.shape[-1], CH):
            seg = x[:, :, lo:lo + CH]
            enc = dac.encode(seg)
            dec = dac.decode(quantized_representation=enc.quantized_representation)
            outs.append(dec.audio_values if hasattr(dec, "audio_values") else dec[0])
            if dev == "cuda":
                torch.cuda.empty_cache()
    y = torch.cat(outs, dim=-1)                  # (2, 1, N) or (2, N)
    y = y.squeeze(1) if y.dim() == 3 else y
    p_dac = out / f"{stem}__dac.wav"
    audio_io.save(y.cpu(), p_dac, 44100)

    # DAC facts, derived not assumed (the n_q lesson)
    with torch.no_grad():
        probe = dac.encode(torch.zeros(1, 1, 44100, device=dev))
    n_q = probe.audio_codes.shape[1]
    fr = probe.audio_codes.shape[-1]
    ui.kv_table([
        ("original", str(p_orig.name)),
        ("encodec 2.2kbps", str(p_enc.name)),
        ("dac ~8kbps", str(p_dac.name)),
        ("dac derived", f"{n_q} codebooks, {fr} frames/sec "
                        f"-> {n_q*2} stereo slots, "
                        f"16384-step window = {16384/fr:.0f}s"),
    ], title="listen in this order")
    ui.panel("Focus on the VOCALS — that's where 2.2kbps bleeds. If __dac is "
             "clearly closer to __orig, the upgrade is real and we weigh the "
             "context trade. If you have to strain to tell, keep the 328s "
             "window and move on.", title="what to listen for", style="cyan")


if __name__ == "__main__":
    main()
