r"""
lrcheck.py — measure whether an LR is sane, in minutes, without a sweep.

    python lrcheck.py --music-dir "...\Grunge"
    python lrcheck.py --music-dir "..." --lrs 1e-5 3e-5 5e-5 1e-4

THE METRIC: update-to-weight ratio.

    mean(|actual parameter update|) / mean(|parameter|)

This asks the only question that matters: is the optimizer moving the weights
by a sensible fraction of their own size? The widely-used target is ~1e-3, i.e.
each step nudges a weight by about a tenth of a percent.

    >> 1e-2   the LR is swinging the model around; it will look like it works
              and then destabilize (exactly what happened here: gn crept
              0.7 -> 2.4 -> 3.6 -> 5.4 over 600 steps at 5e-5)
    ~  1e-3   healthy
    << 1e-4   crawling; you will wait forever for nothing

WHY THIS AND NOT AN LR RANGE TEST: a range test needs a readable loss curve,
which needs a large effective batch and many steps -- an hour, minimum, and the
first attempt here swept 1e-7..1e-2 at effective batch 4 and produced pure
noise (loss moved 0.4 total) while confidently reporting an argmin HIGHER than
the LR that had already failed. The update ratio needs neither: it reads the
optimizer directly, on real data, in a few steps.

Measures the REAL update -- weights before vs after opt.step() -- so it
includes AdamW's preconditioning and weight decay, not just the raw gradient.
Clips exactly as train.py does, or it would measure dynamics that never occur.
"""

import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import ui
from config import CFG
from data import MusicDataset, build_cache, collate

TARGET = 1e-3   # the well-known healthy update/weight ratio


def measure(lr, model_fn, dl, dev, steps, accum, warm=2):
    """Median update/weight ratio over `steps` real optimizer steps at `lr`."""
    torch.manual_seed(CFG.seed)          # same init for every LR
    model = model_fn()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=CFG.betas,
                            weight_decay=CFG.weight_decay, fused=(dev == "cuda"))
    it = iter(dl)
    ratios, gns, losses = [], [], []
    model.train()
    for s in range(steps + warm):
        opt.zero_grad(set_to_none=True)
        tot = 0.0
        for _ in range(accum):
            try:
                b = next(it)
            except StopIteration:
                it = iter(dl); b = next(it)
            x, y = b["x"].to(dev), b["y"].to(dev)
            ctx = b["ctx"].to(dev) if b["ctx"] is not None else None
            msk = b["ctx_mask"].to(dev) if b["ctx_mask"] is not None else None
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
                lg = model(x, ctx, msk)
                loss = F.cross_entropy(lg.reshape(-1, lg.shape[-1]), y.reshape(-1))
            (loss / accum).backward()
            tot += loss.item() / accum
        gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.grad_clip))

        # snapshot the weights, step, then diff: this is the ACTUAL update,
        # after AdamW's preconditioning and decay -- not the raw gradient.
        before = [p.detach().clone() for p in model.parameters() if p.requires_grad]
        opt.step()
        after = [p.detach() for p in model.parameters() if p.requires_grad]

        if s >= warm:                    # skip AdamW's bias-correction warmup
            upd = sum((a - b_).abs().sum().item() for a, b_ in zip(after, before))
            mag = sum(b_.abs().sum().item() for b_ in before)
            ratios.append(upd / max(mag, 1e-12))
            gns.append(gn); losses.append(tot)
    del model, opt
    if dev == "cuda":
        torch.cuda.empty_cache()
    ratios.sort()
    return ratios[len(ratios)//2], sum(gns)/len(gns), losses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--music-dir", type=str, default=None)
    ap.add_argument("--lrs", type=float, nargs="+",
                    default=[1e-5, 3e-5, 5e-5, 1e-4])
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--seq-len", type=int, default=CFG.seq_len)
    ap.add_argument("--batch-size", type=int, default=CFG.batch_size)
    ap.add_argument("--grad-accum", type=int, default=CFG.grad_accum)
    a = ap.parse_args()

    import transformers
    transformers.logging.set_verbosity_error()
    from codec import Codec
    from textenc import LyricEncoder
    from transformer import MusicTransformer

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    codec = Codec(device=dev)
    te = LyricEncoder(device=dev)
    paths = build_cache(codec, te, raw_dir=a.music_dir)
    n_slots, vocab, d_text = codec.n_slots, codec.vocab_size, te.d_text
    del codec, te
    import gc; gc.collect()
    if dev == "cuda":
        torch.cuda.empty_cache()

    ds = MusicDataset(paths, a.seq_len, n_slots, random_crop=True)

    def model_fn():
        return MusicTransformer(
            vocab_size=vocab, d_model=CFG.d_model, n_heads=CFG.n_heads,
            n_layers=CFG.n_layers, d_ff=CFG.d_ff,
            max_seq_len=max(a.seq_len, CFG.max_seq_len), dropout=CFG.dropout,
            d_text=d_text, rope_base=CFG.rope_base, grad_checkpoint=True,
        ).to(dev)

    ui.rule("update/weight ratio — is the optimizer sane?")
    ui.kv_table([
        ("metric", "mean(|update|) / mean(|weight|) per optimizer step"),
        ("target", f"~{TARGET:.0e}   [dim](>>1e-2 swings the model, <<1e-4 crawls)[/]"),
        ("config", f"batch {a.batch_size} x {a.grad_accum} accum = "
                   f"{a.batch_size*a.grad_accum} effective, seq_len {a.seq_len}"),
        ("control", "same seed + same init for every LR — only the LR changes"),
    ])

    rows, results = [], {}
    for lr in a.lrs:
        # identical data order per LR: the only variable is the LR itself
        g = torch.Generator(); g.manual_seed(CFG.seed)
        dl = DataLoader(ds, batch_size=a.batch_size, shuffle=True,
                        collate_fn=collate, drop_last=True, generator=g)
        r, gn, losses = measure(lr, model_fn, dl, dev, a.steps, a.grad_accum)
        results[lr] = r
        off = r / TARGET
        if off > 5:
            verdict = f"[bold red]{off:.0f}x too hot[/]"
        elif off < 0.2:
            verdict = f"[yellow]{1/off:.0f}x too cold[/]"
        else:
            verdict = "[bold green]healthy[/]"
        rows.append((f"{lr:.0e}", f"ratio [bold]{r:.2e}[/]  gn {gn:4.1f}  "
                                  f"loss {losses[0]:.2f}->{losses[-1]:.2f}   {verdict}"))
        ui.log(f"  [dim]{lr:.0e} done[/]")
    ui.log("")
    ui.kv_table(rows, title="measured on your data")

    # closest to target in LOG space -- 1e-4 and 1e-2 are equally wrong, and
    # only a log distance says so.
    import math
    best = min(results, key=lambda lr: abs(math.log10(results[lr] / TARGET)))
    ui.panel(f"closest to the {TARGET:.0e} target: [bold green]{best:.0e}[/]  "
             f"[dim](ratio {results[best]:.2e})[/]\n\n"
             f"[dim]If two are close, prefer the larger -- it trains faster and\n"
             f"warmup + clipping cover the early steps. Set config.lr to it.[/]",
             title="measured answer", style="green")


if __name__ == "__main__":
    main()
