"""
train.py — train the music LM.

    python train.py                      # full run
    python train.py --overfit 3 --steps 500 --seq-len 2048
                                         # correctness check: 3 tracks to ~0 loss

bf16, not fp16: bf16 carries fp32's exponent range, so no GradScaler -- one
fewer moving part and one fewer failure mode.

PER-SLOT LOSS is the best diagnostic in the project. The stream is interleaved
[q0L, q0R, q1L, q1R, ...]; slicing loss by position % n_slots gives one number
per (codebook, channel). Coarse codebooks (q0) should fall fast and far; fine
ones (q3) stay high because fine residuals are close to noise. If every slot
reports the SAME loss, the interleave is broken -- the model is seeing a stream
whose structure doesn't match what we think we cached.

VRAM: Windows silently spills to shared host memory instead of raising OOM. A
21GB "success" on a 16GB card trains at unusable speed and looks fine, so we
assert against a hard budget rather than trusting the absence of an exception.
"""

import argparse
import math
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import ui
from config import CFG, CKPT_DIR
from data import MusicDataset, build_cache, collate


def lr_lambda(step, warmup, total):
    if step < warmup:
        return step / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    return 0.1 + 0.45 * (1 + math.cos(math.pi * min(1.0, p)))   # -> 0.1x


def per_slot_loss(logits, y, n_slots, start_pos: int = 0):
    """Mean CE per interleave slot. Diverging values = healthy RVQ structure."""
    out = []
    for s in range(n_slots):
        # position i in x predicts y[i], whose slot is (start+i+1) % n_slots
        idx = torch.arange(logits.shape[1], device=logits.device)
        m = ((idx + start_pos + 1) % n_slots) == s
        if m.any():
            out.append(F.cross_entropy(
                logits[:, m].reshape(-1, logits.shape[-1]),
                y[:, m].reshape(-1)).item())
        else:
            out.append(float("nan"))
    return out


def evaluate(model, dl, device, n_slots, max_batches=None):
    """
    Mean val loss. max_batches=None reads the WHOLE val set.

    It used to cap at 8 batches, which on a 72-window val split measured only
    22% of it -- and a different random 22% each time, so val loss jittered for
    no reason and best.pt could be chosen off noise. The split is small; just
    read all of it.
    """
    model.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for i, b in enumerate(dl):
            if max_batches is not None and i >= max_batches:
                break
            x, y = b["x"].to(device), b["y"].to(device)
            ctx = b["ctx"].to(device) if b["ctx"] is not None else None
            msk = b["ctx_mask"].to(device) if b["ctx_mask"] is not None else None
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg = model(x, ctx, msk)
                tot += F.cross_entropy(lg.reshape(-1, lg.shape[-1]),
                                       y.reshape(-1)).item()
            n += 1
    model.train()
    return tot / max(1, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=CFG.max_steps)
    ap.add_argument("--seq-len", type=int, default=CFG.seq_len)
    ap.add_argument("--batch-size", type=int, default=CFG.batch_size)
    ap.add_argument("--grad-accum", type=int, default=CFG.grad_accum)
    ap.add_argument("--lr", type=float, default=CFG.lr)
    ap.add_argument("--overfit", type=int, default=0,
                    help="use only N tracks, disable dropout/crop: correctness check")
    ap.add_argument("--resume", type=str, default=None)
    ap.add_argument("--music-dir", type=str, default=None,
                    help="train from ANY folder (recursive). Nothing is copied.")
    ap.add_argument("--grad-checkpoint", action="store_true",
                    help="recompute activations in backward: ~40%% less VRAM, "
                         "~30%% slower. Buys longer seq_len or bigger batch.")
    a = ap.parse_args()

    import transformers
    transformers.logging.set_verbosity_error()
    from codec import Codec
    from textenc import LyricEncoder
    from transformer import MusicTransformer

    torch.manual_seed(CFG.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # Free throughput on Ada/Blackwell, both OFF by default in torch:
    #  - TF32 lets fp32 matmuls use tensor cores. Harmless under bf16 autocast
    #    (the layers that stay fp32 are the ones that don't need the mantissa).
    #  - cudnn.benchmark autotunes conv algorithms. Our shapes are fixed after
    #    the first step, so the one-off autotune cost pays back immediately.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    codec = Codec(device=dev)
    te = LyricEncoder(device=dev)
    paths = build_cache(codec, te, raw_dir=a.music_dir)
    if a.overfit:
        paths = paths[:a.overfit]

    # Everything the codec and T5 produced is now on disk. Keeping them
    # resident costs ~1.5GB of VRAM for the entire training run and they are
    # never called again -- the dataset reads tokens and ctx from the cache.
    # Hold only the scalars the model needs.
    codec_sig = {"vocab_size": codec.vocab_size, "n_slots": codec.n_slots,
                 "channels": codec.channels, "n_q": codec.n_q,
                 "tokens_per_sec": codec.tokens_per_sec, "sr": codec.sr,
                 "model_id": codec.model_id}
    d_text, t5_id = te.d_text, te.model_id
    del codec, te
    import gc; gc.collect()
    if dev == "cuda":
        torch.cuda.empty_cache()

    dropout = 0.0 if a.overfit else CFG.dropout
    ds = MusicDataset(paths, a.seq_len, codec_sig["n_slots"], random_crop=not a.overfit)
    n_val = max(1, int(len(ds) * CFG.val_frac)) if not a.overfit else 0
    if n_val and len(ds) > n_val * 2:
        tr, va = torch.utils.data.random_split(ds, [len(ds) - n_val, n_val])
    else:
        tr, va = ds, None

    dl = DataLoader(tr, batch_size=a.batch_size, shuffle=True, collate_fn=collate,
                    drop_last=len(tr) >= a.batch_size)
    vdl = DataLoader(va, batch_size=a.batch_size, collate_fn=collate) if va else None

    model = MusicTransformer(
        vocab_size=codec_sig["vocab_size"], d_model=CFG.d_model, n_heads=CFG.n_heads,
        n_layers=CFG.n_layers, d_ff=CFG.d_ff, max_seq_len=max(a.seq_len, CFG.max_seq_len),
        dropout=dropout, d_text=d_text, rope_base=CFG.rope_base,
        grad_checkpoint=(a.grad_checkpoint or CFG.grad_checkpoint),
    ).to(dev)

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=CFG.betas,
                            weight_decay=CFG.weight_decay, fused=(dev == "cuda"))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: lr_lambda(s, CFG.warmup_steps, a.steps))

    step0 = 0
    if a.resume:
        # weights_only=False: torch>=2.6 defaults this to True, which refuses
        # to unpickle the cfg/codec_sig dicts we store alongside the tensors.
        # Without it, resume raises on load. (generate.py already passes it.)
        ck = torch.load(a.resume, map_location=dev, weights_only=False)
        # A checkpoint trained at a different seq_len/vocab will load its
        # weights happily and then produce nonsense -- RoPE tables and the
        # decoder are sized from these. Fail loudly instead.
        old = ck.get("codec_sig", {})
        if old.get("vocab_size") not in (None, codec_sig["vocab_size"]):
            raise SystemExit(
                f"checkpoint vocab {old['vocab_size']} != current "
                f"{codec_sig['vocab_size']} — the cache was built with a "
                "different codec config")
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"]); step0 = ck["step"]
        ui.log(f"[green]resumed[/] {a.resume} @ step {step0}")

    ui.rule("MusicMaker — training")
    ui.kv_table([
        ("model", f"{model.count_parameters()/1e6:.1f}M params  "
                  f"d={CFG.d_model} L={CFG.n_layers} H={CFG.n_heads}"),
        ("codec", f"{codec_sig['model_id']}  n_q={codec_sig['n_q']}x{codec_sig['channels']}ch  "
                  f"vocab={codec_sig['vocab_size']}  {codec_sig['tokens_per_sec']} tok/s"),
        ("context", f"{a.seq_len} tokens = "
                    f"{a.seq_len/codec_sig['tokens_per_sec']:.2f}s audio"),
        ("data", f"{len(tr)} windows over {len(paths)} tracks"),
        ("batch", f"{a.batch_size} x {a.grad_accum} accum = {a.batch_size*a.grad_accum} effective"),
        ("lyrics", f"T5 {t5_id} (frozen, d={d_text})"),
        ("grad ckpt", "[green]on[/] — ~40% less VRAM, ~30% slower"
                      if (a.grad_checkpoint or CFG.grad_checkpoint) else "off"),
        ("mode", "[bold yellow]OVERFIT[/] — no dropout, no crop" if a.overfit
                 else f"train — dropout {dropout}, cond-dropout {CFG.cond_dropout}"),
    ])
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    # Track the best val loss so best.pt always holds the best model seen.
    # Seeded from an existing best.pt so a resumed run cannot clobber a better
    # checkpoint with a worse one just because it restarted at inf.
    best_val, best_step = float("inf"), -1
    _bp = CKPT_DIR / "best.pt"
    if _bp.exists() and a.resume:
        try:
            _b = torch.load(_bp, map_location="cpu", weights_only=False)
            if _b.get("val_loss") is not None:
                best_val, best_step = _b["val_loss"], _b.get("step", -1)
                ui.log(f"[dim]existing best: val {best_val:.3f} @ step {best_step}[/]")
        except Exception:
            pass

    model.train()
    it = iter(dl)
    t0 = time.time()
    prog = ui.train_progress()
    task = prog.add_task("train", total=a.steps, completed=step0, stats="") if prog else None
    if prog:
        prog.start()
    for step in range(step0, a.steps):
        opt.zero_grad(set_to_none=True)
        tot = 0.0
        for _ in range(a.grad_accum):
            try:
                b = next(it)
            except StopIteration:
                it = iter(dl); b = next(it)
            x, y = b["x"].to(dev, non_blocking=True), b["y"].to(dev, non_blocking=True)
            ctx = b["ctx"].to(dev) if b["ctx"] is not None else None
            msk = b["ctx_mask"].to(dev) if b["ctx_mask"] is not None else None
            # Conditioning dropout: trains the unconditional path in the same
            # run, which is exactly what classifier-free guidance samples from.
            if ctx is not None and not a.overfit and torch.rand(1).item() < CFG.cond_dropout:
                ctx, msk = None, None
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
                lg = model(x, ctx, msk)
                loss = F.cross_entropy(lg.reshape(-1, lg.shape[-1]), y.reshape(-1))
            (loss / a.grad_accum).backward()
            tot += loss.item() / a.grad_accum

        # clip_grad_norm_ returns the PRE-clip norm -- the single best early
        # warning of instability. If it sits far above grad_clip the optimizer
        # is being clipped every step, which means the LR is too high for this
        # batch size and the run is heading for divergence.
        gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.grad_clip))
        opt.step(); sched.step()

        peak = torch.cuda.max_memory_allocated() / 1e9 if dev == "cuda" else 0
        if prog:
            dt = time.time() - t0
            tps = (step - step0 + 1) * a.batch_size * a.grad_accum * a.seq_len / max(dt, 1e-9)
            prog.update(task, completed=step + 1, stats=(
                f"loss [bold]{tot:.3f}[/] ppl [bold]{math.exp(min(tot,20)):.1f}[/] "
                f"lr {sched.get_last_lr()[0]:.1e} "
                f"gn {'[red]' if gnorm > CFG.grad_clip * 5 else ''}{gnorm:.1f}"
                f"{'[/]' if gnorm > CFG.grad_clip * 5 else ''} "
                f"{tps/1e3:.0f}k tok/s vram {peak:.1f}GB"))
        elif step % 10 == 0:
            print(f"step {step:5d} | loss {tot:6.3f} | ppl {math.exp(min(tot,20)):8.1f} | "
                  f"vram {peak:.1f}GB")

        # Windows won't OOM -- it spills to host memory and crawls. Assert.
        if step % 10 == 0:
            assert peak < CFG.vram_budget_gb, (
                f"peak VRAM {peak:.1f}GB exceeds {CFG.vram_budget_gb}GB budget; "
                "on Windows this silently spills to shared memory instead of "
                "raising OOM. Lower batch_size or seq_len."
            )

        # Fire on the FIRST step too, not just every eval_interval. The
        # per-slot table is the only check that the interleave is intact, and
        # that is worth knowing 10 seconds in rather than 9 minutes in -- a
        # broken interleave means the whole run is training on noise.
        if step == step0 or (step > step0 and step % CFG.eval_interval == 0):
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16,
                                                 enabled=(dev == "cuda")):
                ps = per_slot_loss(model(x[:1], ctx[:1] if ctx is not None else None,
                                         msk[:1] if msk is not None else None),
                                   y[:1], codec_sig["n_slots"])
            names = [f"q{s//codec_sig['channels']}{'LR'[s%codec_sig['channels']]}"
                     for s in range(codec_sig["n_slots"])]
            if prog:
                prog.stop()
            title = f"per-slot loss @ step {step}"
            if step == step0:
                # ln(16386)=9.70 -- an untrained model guessing uniformly lands
                # here, so this row is the baseline every later table is read
                # against, not a warning sign.
                title += "  [dim](baseline: untrained ~= ln(vocab) = " \
                         f"{math.log(codec_sig['vocab_size']):.2f})[/]"
            ui.slot_table(names, ps, title=title)
            # Coarse slots should sit well below fine ones -- that spread IS the
            # RVQ residual structure showing up in the loss. Flat slots CAN mean
            # a broken interleave, but only in the middle of the range:
            #   - at step ~0 everything is ~ln(vocab)=9.7 because the model is
            #     untrained and guessing uniformly. Flat is CORRECT here.
            #   - after memorizing, everything collapses to ~0. Flat again, also
            #     correct.
            # So only warn once training has had a real chance to differentiate
            # the slots AND the loss is still stuck near the uniform-guess line.
            warmed = step >= max(200, CFG.warmup_steps)
            if warmed and max(ps) - min(ps) < 0.05 and min(ps) > 0.5:
                ui.log("[bold red]!! slots flat and still high after warmup — "
                       "interleave may be broken[/]")
            if vdl:
                vl = evaluate(model, vdl, dev, codec_sig["n_slots"])
                # Track the BEST model, not just the last one. With 1444
                # windows over ~22 epochs this WILL overfit: train loss keeps
                # falling while val loss bottoms out and turns back up. Without
                # this, final.pt could be hours past the good model and nothing
                # would tell you which step to go back to.
                if vl < best_val:
                    best_val, best_step = vl, step
                    _save(CKPT_DIR / "best.pt", model, opt, sched, step, a,
                          codec_sig, t5_id, d_text, val=vl, quiet=True)
                    ui.log(f"   [dim]val loss[/] [bold green]{vl:.3f}[/] "
                           f"[dim]<- new best, saved best.pt[/]")
                else:
                    # Val rising has TWO causes that need opposite responses:
                    #  - OVERFITTING: train loss keeps falling, val climbs. The
                    #    model is memorizing. best.pt already has the good one.
                    #  - DIVERGING: train loss climbs TOO, back toward
                    #    ln(vocab). The optimizer is unstable -- LR too high for
                    #    the batch size. Nothing later will be better; stop and
                    #    lower the LR.
                    # Calling both "overfitting" sent exactly the wrong signal
                    # once already, so distinguish them.
                    tag = ""
                    if tot > best_val * 1.05 and vl > best_val * 1.05:
                        tag = " — [bold red]DIVERGING: train loss rising too, " \
                              "lower --lr[/]"
                    elif vl > best_val * 1.05:
                        tag = " — [yellow]overfitting[/]"
                    ui.log(f"   [dim]val loss[/] [bold]{vl:.3f}[/] "
                           f"[dim](best {best_val:.3f} @ step {best_step})[/]{tag}")
            if prog:
                prog.start()

        if step > step0 and step % CFG.ckpt_interval == 0:
            p = CKPT_DIR / f"step{step}.pt"
            _save(p, model, opt, sched, step, a, codec_sig, t5_id, d_text)
            ui.log(f"   [dim]saved[/] {p}")

    if prog:
        prog.stop()
    p = CKPT_DIR / "final.pt"
    _save(p, model, opt, sched, a.steps, a, codec_sig, t5_id, d_text)
    body = (f"final checkpoint: [bold]{p}[/]\n"
            f"last train loss: [bold]{tot:.3f}[/]  "
            f"ppl [bold]{math.exp(min(tot, 20)):.1f}[/]")
    if best_step >= 0:
        body += (f"\n\n[bold green]best val loss {best_val:.3f} @ step {best_step}[/]"
                 f"\n[dim]-> generate from checkpoints/best.pt, not final.pt,"
                 f" unless val was still falling at the end[/]")
    ui.panel(body, title="done", style="green")


def _save(path, model, opt, sched, step, a, codec_sig, t5_id, d_text,
          val=None, quiet=False):
    """One place that knows the checkpoint format. generate.py reads codec_sig
    to rebuild the model, so these keys are a contract -- don't drift them."""
    torch.save({
        "model": model.state_dict(), "opt": opt.state_dict(),
        "sched": sched.state_dict(), "step": step, "cfg": vars(a),
        "val_loss": val,
        "codec_sig": {
            "codec_id": codec_sig["model_id"], "n_q": codec_sig["n_q"],
            "n_slots": codec_sig["n_slots"], "channels": codec_sig["channels"],
            "vocab_size": codec_sig["vocab_size"], "sr": codec_sig["sr"],
            "t5_id": t5_id, "d_text": d_text,
            "d_model": CFG.d_model, "n_heads": CFG.n_heads,
            "n_layers": CFG.n_layers, "d_ff": CFG.d_ff,
            "max_seq_len": max(a.seq_len, CFG.max_seq_len),
        }}, path)


if __name__ == "__main__":
    main()
