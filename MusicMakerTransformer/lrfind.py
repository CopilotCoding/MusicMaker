r"""
lrfind.py — measure the learning rate instead of guessing it.

    python lrfind.py --music-dir "...\Grunge"

WHY THIS EXISTS: this project has now guessed the LR twice and been wrong twice.
  - 3e-4 (the standard default for effective batch ~32): diverged outright at
    step 350 -- loss went 8.18 -> 9.39, unlearning everything.
  - 5e-5 (3e-4 / 8, by the linear scaling rule for effective batch 4): looked
    stable for 600 steps, then gradient norm climbed 0.7 -> 2.4 -> 3.6 -> 5.4,
    i.e. clipped on every step and heading the same way.
Both were arithmetic. This measures.

METHOD (Smith 2015, "Cyclical Learning Rates"): ramp the LR exponentially from
tiny to absurd over a few hundred steps, recording loss at each. The curve has
three regions:

    loss
     |  \                        /
     |   \___________          /      <- explodes: LR too big
     |               \____/           <- the good band
     +----------------------------> log(lr)
        too small     minimum

Take the LR roughly 1/10th of where loss hits its minimum -- NOT the minimum
itself, which is already at the edge of instability.

This trains a throwaway model on the real data and the real config. It does not
touch your checkpoints.
"""

import argparse
import math

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import ui
from config import CFG
from data import MusicDataset, build_cache, collate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--music-dir", type=str, default=None)
    # Sweep at EFFECTIVE batch 32 (2 x 16 accum), not the training config's
    # effective 4. The whole point of a range test is a readable loss curve,
    # and at effective 4 the gradient variance drowns the signal -- the first
    # version of this test swept 1e-7..1e-2 and loss moved 9.855 -> 9.433, i.e.
    # nothing. Accumulation is mathematically identical to a true large batch,
    # so this measures the real thing.
    #
    # Narrow window: 1e-7..1e-2 wasted ~90% of its steps in a range where
    # nothing happens. Start where the last runs actually lived.
    ap.add_argument("--start-lr", type=float, default=1e-6)
    ap.add_argument("--end-lr", type=float, default=1e-3)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--seq-len", type=int, default=CFG.seq_len)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=16)
    a = ap.parse_args()

    import transformers
    transformers.logging.set_verbosity_error()
    from codec import Codec
    from textenc import LyricEncoder
    from transformer import MusicTransformer

    torch.manual_seed(CFG.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    codec = Codec(device=dev)
    te = LyricEncoder(device=dev)
    paths = build_cache(codec, te, raw_dir=a.music_dir)
    n_slots, vocab = codec.n_slots, codec.vocab_size
    d_text = te.d_text
    del codec, te
    import gc; gc.collect()
    if dev == "cuda":
        torch.cuda.empty_cache()

    ds = MusicDataset(paths, a.seq_len, n_slots, random_crop=True)
    dl = DataLoader(ds, batch_size=a.batch_size, shuffle=True, collate_fn=collate,
                    drop_last=True)

    model = MusicTransformer(
        vocab_size=vocab, d_model=CFG.d_model, n_heads=CFG.n_heads,
        n_layers=CFG.n_layers, d_ff=CFG.d_ff,
        max_seq_len=max(a.seq_len, CFG.max_seq_len), dropout=0.0,
        d_text=d_text, rope_base=CFG.rope_base, grad_checkpoint=True,
    ).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.start_lr, betas=CFG.betas,
                            weight_decay=CFG.weight_decay, fused=(dev == "cuda"))

    ui.rule("LR range test — measuring, not guessing")
    ui.kv_table([
        ("sweep", f"{a.start_lr:.0e} -> {a.end_lr:.0e} over {a.steps} steps"),
        ("config", f"batch {a.batch_size} x {a.grad_accum} accum = "
                   f"{a.batch_size*a.grad_accum} effective, seq_len {a.seq_len}"),
        ("note", "[dim]exponential ramp; loss explodes past the usable band[/]"),
    ])

    mult = (a.end_lr / a.start_lr) ** (1 / max(1, a.steps - 1))
    lr = a.start_lr
    it = iter(dl)
    hist = []
    best = float("inf")

    model.train()
    prog = ui.train_progress()
    task = prog.add_task("lr sweep", total=a.steps, stats="") if prog else None
    if prog:
        prog.start()

    for i in range(a.steps):
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad(set_to_none=True)
        tot = 0.0
        for _ in range(a.grad_accum):
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
            (loss / a.grad_accum).backward()
            tot += loss.item() / a.grad_accum
        # Clip EXACTLY as train.py does. The first version passed 1e9 here to
        # "measure without clipping" -- which meant the sweep ran unclipped
        # while real training clips at 1.0, so it measured dynamics that never
        # occur. clip_grad_norm_ returns the PRE-clip norm either way, so we
        # still see the true gradient magnitude.
        gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.grad_clip))
        opt.step()
        hist.append((lr, tot, gn))
        best = min(best, tot)
        if prog:
            prog.update(task, completed=i + 1,
                        stats=f"lr [bold]{lr:.2e}[/] loss [bold]{tot:.3f}[/] gn {gn:.1f}")
        # stop once it has clearly blown up
        if tot > best * 3 or not math.isfinite(tot):
            if prog:
                prog.update(task, completed=a.steps)
            break
        lr *= mult
    if prog:
        prog.stop()

    _report(hist)


def _report(hist):
    if len(hist) < 10:
        raise SystemExit("too few steps to judge -- widen the sweep")
    # smooth the loss so a single noisy step doesn't pick the minimum
    sm, beta, avg = [], 0.8, hist[0][1]
    for _, l, _ in hist:
        avg = beta * avg + (1 - beta) * l
        sm.append(avg)
    i_min = min(range(len(sm)), key=lambda i: sm[i])
    lr_min = hist[i_min][0]
    # the usable LR is ~1/10th of the minimum: the minimum is already the edge
    suggested = lr_min / 10

    # A sweep that barely moved has NOT found a minimum -- it found noise on a
    # flat line, and reporting its argmin as an answer is worse than useless.
    # The first version of this test did exactly that: loss went 9.855 -> 9.433
    # (0.4 total) and it confidently reported "minimum at 6.66e-04", a number
    # HIGHER than the LR that had just failed. Refuse to answer instead.
    span = max(sm) - min(sm)
    if span < 1.0:
        ui.panel(
            f"[bold red]INCONCLUSIVE[/] — loss only moved {span:.3f} across the "
            f"whole sweep\n({max(sm):.3f} -> {min(sm):.3f}).\n\n"
            f"[dim]That is noise on a flat line, not a minimum. The model needs\n"
            f"more steps per LR, or a larger effective batch, before the curve\n"
            f"means anything. Do NOT trust an argmin from this.[/]",
            title="no answer", style="red")
        return

    ui.log("")
    rows = []
    step = max(1, len(hist) // 14)
    for i in range(0, len(hist), step):
        lr, l, gn = hist[i]
        bar = "#" * int(max(0, min(40, (12 - sm[i]) * 5)))
        mark = "  <- min" if i == i_min else ""
        rows.append((f"{lr:.2e}", f"{sm[i]:6.3f}  gn {gn:5.1f}  {bar}{mark}"))
    ui.kv_table(rows, title="lr vs smoothed loss")

    ui.panel(
        f"loss minimum at lr [bold]{lr_min:.2e}[/]\n"
        f"suggested lr:      [bold green]{suggested:.2e}[/]  [dim](min / 10)[/]\n\n"
        f"[dim]the minimum is the edge of instability -- training there is what\n"
        f"made gn climb 0.7 -> 5.4. Set config.lr to the suggested value.[/]",
        title="measured", style="green")


if __name__ == "__main__":
    main()
