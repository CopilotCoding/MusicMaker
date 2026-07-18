r"""
generate_delay.py — sample the delay-pattern model, decode to wav/mp3/flac.

    python generate_delay.py --lyrics-file song.txt --seconds 60 --out out.mp3
    python generate_delay.py --seconds 30 --out out.wav       # unconditional
    python generate_delay.py --ckpt checkpoints_delay/best.pt --device cpu ...

Rows are generated chunk-aligned (cache reset every seq_steps rows, exactly
as training reads songs), all 8 heads sampled per row — ~8x fewer forward
passes per second of audio than the flat model, i.e. faster than realtime on
GPU. CFG doubles the forwards when lyrics are given.
"""

import argparse
from pathlib import Path

import torch

import audio_io
import ui
from config import CFG
from delay import DelayMusicTransformer, slot_delays, undelay_grid

CKPT_DIR = Path(__file__).parent / "checkpoints_delay"


def load_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    s = ck["codec_sig"]
    model = DelayMusicTransformer(
        vocab_size=s["vocab_size"], d_model=s["d_model"], n_heads=s["n_heads"],
        n_layers=s["n_layers"], d_ff=s["d_ff"], max_seq_len=s["seq_steps"],
        n_slots=s["n_slots"], codebook_sz=s["codebook_sz"], dropout=0.0,
        d_text=s["d_text"], rope_base=CFG.rope_base,
    ).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, s, ck.get("step", -1)


def sample_row(logits, temperature, top_k, top_p):
    """(K, cb) -> (K,) class ids. Per-head temp -> top-k -> top-p."""
    if temperature <= 0.02:
        return logits.argmax(-1)
    lg = logits / temperature
    if top_k and top_k < lg.shape[-1]:
        kth = lg.topk(top_k, dim=-1).values[..., -1:]
        lg = lg.masked_fill(lg < kth, float("-inf"))
    if top_p and top_p < 1.0:
        srt, idx = lg.sort(dim=-1, descending=True)
        cum = srt.softmax(-1).cumsum(-1)
        cut = cum - srt.softmax(-1) > top_p
        srt = srt.masked_fill(cut, float("-inf"))
        lg = torch.full_like(lg, float("-inf")).scatter(-1, idx, srt)
    return torch.multinomial(lg.softmax(-1), 1)[:, 0]


@torch.no_grad()
def generate_rows(model, sig, n_rows, ctx=None, ctx_mask=None,
                  temperature=1.0, top_k=250, top_p=0.95, guidance=3.0,
                  device="cuda", verbose=True):
    K, cb = sig["n_slots"], sig["codebook_sz"]
    S = sig["seq_steps"]
    delays = slot_delays(sig["n_q"], sig["channels"])
    use_cfg = guidance and guidance != 1.0 and ctx is not None
    bos_row = torch.full((1, 1, K), sig["bos"], dtype=torch.long, device=device)

    lg, caches = model(bos_row, ctx, ctx_mask, kv_caches=None,
                       return_caches=True, pos_offset=0)
    lgu = ccaches = None
    if use_cfg:
        lgu, ccaches = model(bos_row, None, None, kv_caches=None,
                             return_caches=True, pos_offset=0)
    cache_len = 1

    offs = torch.arange(K) * cb
    rows = []
    prog = ui.gen_progress() if verbose else None
    task = prog.add_task("rows", total=n_rows, stats="") if prog else None
    if prog:
        prog.start()
    for s in range(n_rows):
        l = lg[0, -1].float()
        if use_cfg:
            l = lgu[0, -1].float() + guidance * (l - lgu[0, -1].float())
        cls = sample_row(l.cpu(), temperature, top_k, top_p)
        row = cls + offs
        for k in range(K):                       # structural stagger pads
            if s < delays[k]:
                row[k] = sig["pad"]
        rows.append(row)
        if s + 1 < n_rows:
            if cache_len >= S:                   # chunk boundary, like training
                caches = ccaches = None
                cache_len = 0
            feed = row[None, None].to(device)
            lg, caches = model(feed, ctx, ctx_mask, kv_caches=caches,
                               return_caches=True, pos_offset=cache_len)
            if use_cfg:
                lgu, ccaches = model(feed, None, None, kv_caches=ccaches,
                                     return_caches=True, pos_offset=cache_len)
            cache_len += 1
        if prog:
            prog.update(task, completed=s + 1,
                        stats=f"[bold]{(s+1)/sig['frame_rate']:.1f}s[/] audio")
    if prog:
        prog.stop()
    return torch.stack(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(CKPT_DIR / "best.pt"))
    ap.add_argument("--lyrics", type=str, default=None)
    ap.add_argument("--lyrics-file", type=str, default=None)
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--out", default="out_delay.wav")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=250)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--guidance", type=float, default=3.0)
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto",
                    help="'cpu' to sample safely while training holds the GPU")
    a = ap.parse_args()

    import transformers
    transformers.logging.set_verbosity_error()

    dev = ("cuda" if torch.cuda.is_available() else "cpu") \
        if a.device == "auto" else a.device
    model, sig, step = load_model(a.ckpt, dev)
    ui.rule("MusicMaker — generate (delay pattern)")
    ui.kv_table([
        ("checkpoint", f"{a.ckpt}  step {step}"),
        ("model", f"{model.count_parameters()/1e6:.1f}M, "
                  f"{sig['n_slots']} heads, window {sig['seq_steps']} steps"),
    ])

    ctx = msk = None
    text = a.lyrics
    if a.lyrics_file:
        text = Path(a.lyrics_file).read_text(encoding="utf-8",
                                             errors="replace").strip()
    if text:
        from textenc import LyricEncoder
        te = LyricEncoder(device=dev)
        c, m = te.encode([text])
        ctx, msk = c.to(dev), m.to(dev)
        del te
        ui.panel(text[:300], title="lyrics")

    n_rows = int(a.seconds * sig["frame_rate"]) + sig["n_q"] - 1
    rows = generate_rows(model, sig, n_rows, ctx, msk, a.temperature,
                         a.top_k, a.top_p, a.guidance, dev)

    codes = undelay_grid(rows, sig["n_q"], sig["channels"], sig["codebook_sz"])
    del model
    if dev == "cuda":
        torch.cuda.empty_cache()
    # decode with the codec the checkpoint was TRAINED on — recorded in the
    # sig, never assumed. DAC tokens through an EnCodec decoder are noise.
    from dac_codec import make_codec
    codec = make_codec(sig.get("codec_kind", "encodec"), dev)
    flat = codec.flatten(codes[None].to(dev))
    wav = codec.decode_flat(flat)
    audio_io.save(wav.cpu(), a.out, codec.sr, peak_norm=CFG.peak_norm)
    ui.panel(f"[bold]{a.out}[/]  {wav.shape[-1]/codec.sr:.2f}s "
             f"{wav.shape[0]}ch @ {codec.sr}Hz", title="wrote", style="green")


if __name__ == "__main__":
    main()
