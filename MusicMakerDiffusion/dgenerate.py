"""
dgenerate.py — sample latents from noise, decode to wav/mp3/flac.

    python dgenerate.py --text "genre: blues" --out out.mp3
    python dgenerate.py --text-file song.txt --seconds 10 --out out.wav

Compare to the AR project's generate.py:
  - no KV cache, no sliding window, no slot logit mask, no BOS
  - ~50 denoising passes over the WHOLE clip instead of 4096 sequential steps
  - length is FIXED at sample time; it cannot continue past the window, which
    is the real cost of diffusion's global view

Length is capped at n_frames (the RoPE table). The AR model has no such cap --
it just keeps going. That trade is the comparison in one line.
"""

import argparse

import torch

import audio_io
import ui
from dconfig import CKPT_DIR, DCFG
from diffusion import Diffusion


def load_model(ckpt_path, device="cuda", use_ema=True):
    from dmodel import DiT
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    s = ck["sig"]
    m = DiT(latent_dim=s["latent_dim"], d_model=s["d_model"], n_heads=s["n_heads"],
            n_layers=s["n_layers"], d_ff=s["d_ff"], n_frames=s["n_frames"],
            dropout=0.0, d_text=s["d_text"]).to(device)
    # EMA weights, not raw: diffusion sample quality depends on this.
    m.load_state_dict(ck["ema" if use_ema and "ema" in ck else "model"])
    m.eval()
    return m, s, ck.get("step", -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(CKPT_DIR / "final.pt"))
    ap.add_argument("--text", type=str, default=None)
    ap.add_argument("--text-file", type=str, default=None)
    ap.add_argument("--seconds", type=float, default=None)
    ap.add_argument("--out", type=str, default="out.wav")
    ap.add_argument("--steps", type=int, default=DCFG.sample_steps)
    ap.add_argument("--guidance", type=float, default=DCFG.guidance)
    ap.add_argument("--eta", type=float, default=DCFG.ddim_eta)
    ap.add_argument("--raw", action="store_true", help="use raw (non-EMA) weights")
    a = ap.parse_args()

    import transformers
    transformers.logging.set_verbosity_error()
    from dcodec import LatentCodec

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, sig, step = load_model(a.ckpt, dev, use_ema=not a.raw)
    codec = LatentCodec(scale=sig["scale"], device=dev)

    assert sig["codec_id"] == codec.model_id and sig["latent_dim"] == codec.latent_dim, \
        f"checkpoint codec {sig['codec_id']} != {codec.model_id}"

    secs = a.seconds or sig["n_frames"] / sig["frame_rate"]
    n_frames = min(int(round(secs * sig["frame_rate"])), sig["n_frames"])
    if a.seconds and n_frames < int(round(a.seconds * sig["frame_rate"])):
        ui.log(f"[yellow]capped at {n_frames/sig['frame_rate']:.2f}s[/] "
               f"[dim]— diffusion is fixed-length; the AR model has no such cap[/]")

    ui.rule("MusicMakerDiffusion — generate")
    ui.kv_table([
        ("checkpoint", f"{a.ckpt}  [dim]step {step}"
                       f"{' (raw)' if a.raw else ' (EMA)'}[/]"),
        ("model", f"DiT {model.count_parameters()/1e6:.1f}M"),
        ("output", f"{n_frames} frames = {n_frames/sig['frame_rate']:.2f}s, "
                   f"{sig['channels']}ch @ {sig['sr']}Hz"),
        ("sampling", f"{a.steps} DDIM steps [dim](AR would need "
                     f"{n_frames*8} sequential steps)[/]"),
    ])

    text = a.text
    if a.text_file:
        text = open(a.text_file, encoding="utf-8").read().strip()
    ctx = msk = None
    if text:
        from textenc import LyricEncoder
        te = LyricEncoder(model_id=sig["t5_id"], device=dev)
        ctx, msk = te.encode([text])
        # one row per channel: each channel is denoised as its own batch row
        ctx = ctx.repeat_interleave(sig["channels"], dim=0)
        msk = msk.repeat_interleave(sig["channels"], dim=0)
        ui.panel(text[:300], title="conditioning", style="magenta")

    diff = Diffusion(sig["timesteps"], sig["schedule"], sig["prediction"], dev)
    prog = ui.gen_progress()
    task = prog.add_task("denoising", total=a.steps, stats="") if prog else None
    if prog:
        prog.start()

    z = diff.sample(
        model, (sig["channels"], sig["latent_dim"], n_frames), ctx, msk,
        steps=a.steps, eta=a.eta, guidance=a.guidance, device=dev,
        progress=(lambda i, n: prog.update(task, completed=i,
                                           stats=f"t={1000-int(1000*i/n):4d}"))
        if prog else None)
    if prog:
        prog.stop()

    wav = codec.decode_latent(z)
    audio_io.save(wav, a.out, sig["sr"], peak_norm=DCFG.peak_norm)
    ui.panel(f"[bold]{a.out}[/]\n{wav.shape[1]/sig['sr']:.2f}s  "
             f"{wav.shape[0]}ch @ {sig['sr']}Hz  peak {wav.abs().max():.2f}",
             title="wrote", style="green")


if __name__ == "__main__":
    main()
