r"""
train_delay.py — train the delay-pattern music LM (1 position per timestep).

    python train_delay.py --music-dir "..\Grunge"          # grunge (+ lyrics)
    python train_delay.py --pretrain-cache data/cache_pretrain
                                                           # Jamendo, unconditional
    python train_delay.py --music-dir "..." --init-from checkpoints_delay/pre_final.pt
                                                           # fine-tune from pretrain

CONFIG: seq 16384 STEPS = 327.6s of audio per window — whole songs for 84% of
the grunge corpus (mean track 4.4min). MEASURED: batch 2 -> 9.7GB, 5.16s/step,
consuming 656s of audio per step (~127x realtime). The same positions bought
41s under the flat interleave; the delay pattern is why whole-song structure
is in context at all.

Everything else is the proven stack: BOS-row song starts, in-order seam-fixed
windows (short tracks pad + mask, so nothing is dropped), per-song val split,
LAMB + grad-norm backoff (now with the absolute tripwire the 1e-3 run showed
was needed), bf16, grad checkpointing, per-head loss table, metrics.csv,
best/interrupted/final checkpoints, auto-save on any exit.

Cross-attention is ALWAYS built (d_text=768) even for the unconditional
pretrain — zero-init keeps it silent, and it means the pretrain checkpoint's
shapes match the fine-tune's, so --init-from is a plain weight load.
"""

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import ui
from config import CFG, ROOT
from delay import DelayMusicTransformer, build_grid, grid_targets
from train import GradNormBackoff, _split_by_song, lr_lambda

CKPT_DIR = ROOT / "checkpoints_delay"


class DelayDataset(Dataset):
    """In-order, non-overlapping windows of grid ROWS, per track.

        window 0:  [BOSrow, g0..g_{S-2}] -> targets rows [0, S)
        window k:  [g_{kS-1}..g_{kS+S-2}] -> targets rows [kS, (k+1)S)   (seam)

    Tracks are padded with PAD rows to whole windows; padded targets are -100.
    NOTHING is dropped: a 30s track is one mostly-padded window — the loss
    mask makes the padding free, and short tracks still teach their content.
    """

    def __init__(self, cache_paths, seq_steps, n_q, channels, codebook_sz,
                 bos, pad):
        self.S = seq_steps
        self.K = n_q * channels
        self.bos_row = torch.full((self.K,), bos, dtype=torch.int16)
        self.tracks, self.index = [], []
        skipped = 0
        for p in cache_paths:
            z = np.load(p)
            flat = torch.from_numpy(z["codes"].astype(np.int64))
            T = flat.shape[0] // self.K
            if T < 50:                      # < 1s of audio: nothing to learn
                skipped += 1
                continue
            codes = flat.view(T, n_q, channels).permute(1, 2, 0)  # unflatten,
            codes = codes - (torch.arange(self.K).view(n_q, channels, 1)
                             * codebook_sz)                        # de-band
            grid = build_grid(codes, codebook_sz, pad)             # (G, K)
            tgt = grid_targets(grid, codebook_sz)                  # (G, K)
            G = grid.shape[0]
            W = max(1, math.ceil(G / self.S))
            L = W * self.S
            if L > G:
                grid = torch.cat([grid, torch.full((L - G, self.K), pad,
                                                   dtype=torch.long)])
                tgt = torch.cat([tgt, torch.full((L - G, self.K), -100,
                                                 dtype=torch.long)])
            ctx = torch.from_numpy(z["ctx"].astype(np.float32)) \
                if "ctx" in z else None
            msk = torch.from_numpy(z["ctx_mask"]) if "ctx_mask" in z else None
            ti = len(self.tracks)
            # int16 storage (ids <= 16385, targets >= -100): 4x less RAM than
            # int64 across a 1000h cache. Cast per-item in __getitem__.
            self.tracks.append({"grid": grid.to(torch.int16),
                                "tgt": tgt.to(torch.int16),
                                "ctx": ctx, "mask": msk, "name": p.stem})
            self.index += [(ti, w) for w in range(W)]
        if skipped:
            ui.log(f"[dim]{skipped} sub-second tracks skipped[/]")
        if not self.tracks:
            raise SystemExit("no usable tracks in the cache")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        ti, w = self.index[i]
        tr = self.tracks[ti]
        grid = tr["grid"].long()
        tgt = tr["tgt"].long()
        S = self.S
        if w == 0:
            x = torch.cat([self.bos_row.long()[None], grid[:S - 1]])
        else:
            x = grid[w * S - 1: w * S + S - 1]
        y = tgt[w * S: w * S + S]
        return {"x": x, "y": y, "ctx": tr["ctx"], "mask": tr["mask"],
                "name": tr["name"]}


def collate(batch):
    x = torch.stack([b["x"] for b in batch])
    y = torch.stack([b["y"] for b in batch])
    if all(b["ctx"] is None for b in batch):
        return {"x": x, "y": y, "ctx": None, "ctx_mask": None}
    d = next(b["ctx"].shape[-1] for b in batch if b["ctx"] is not None)
    L = max((b["ctx"].shape[0] for b in batch if b["ctx"] is not None), default=1)
    B = len(batch)
    ctx = torch.zeros(B, L, d)
    msk = torch.zeros(B, L, dtype=torch.bool)
    for i, b in enumerate(batch):
        if b["ctx"] is not None:
            n = b["ctx"].shape[0]
            ctx[i, :n] = b["ctx"]
            msk[i, :n] = b["mask"] if b["mask"] is not None else True
        else:
            msk[i, 0] = True                # NaN guard: never all-False
    return {"x": x, "y": y, "ctx": ctx, "ctx_mask": msk}


def per_head_loss(logits, y):
    """(B,S,K,cb) vs (B,S,K) -> K masked mean CE values: the interleave
    diagnostic, delay edition. Coarse heads should fall faster and lower."""
    out = []
    for k in range(logits.shape[2]):
        out.append(F.cross_entropy(
            logits[:, :, k].reshape(-1, logits.shape[-1]),
            y[:, :, k].reshape(-1), ignore_index=-100).item())
    return out


@torch.no_grad()
def evaluate(model, vdl, dev):
    model.eval()
    tot, n = 0.0, 0
    for b in vdl:
        x, y = b["x"].to(dev), b["y"].to(dev)
        ctx = b["ctx"].to(dev) if b["ctx"] is not None else None
        msk = b["ctx_mask"].to(dev) if b["ctx_mask"] is not None else None
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
            lg = model(x, ctx, msk)
            tot += F.cross_entropy(lg.reshape(-1, lg.shape[-1]),
                                   y.reshape(-1), ignore_index=-100).item()
        n += 1
    model.train()
    return tot / max(1, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--music-dir", type=str, default=None)
    ap.add_argument("--pretrain-cache", type=str, default=None,
                    help="train straight from a cache dir (Jamendo pretrain); "
                         "unconditional, no T5")
    ap.add_argument("--init-from", type=str, default=None,
                    help="load model WEIGHTS from this checkpoint, fresh "
                         "optimizer/schedule (the fine-tune path)")
    ap.add_argument("--resume", type=str, default=None)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--seq-steps", type=int, default=28672,
                    help="window in TIMESTEPS. DAC @86fps: 28672 = 334s — "
                         "whole songs for ~84%% of the grunge corpus. "
                         "MEASURED: batch 1 -> 9.4GB, 6.9s/step.")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--overfit", type=int, default=0)
    ap.add_argument("--codec", choices=["dac", "encodec"], default="dac",
                    help="dac: 44.1kHz ~8kbps, 18 slots (measured: 'basically "
                         "the original'). encodec: the old 2.2kbps ceiling.")
    a = ap.parse_args()

    import transformers
    transformers.logging.set_verbosity_error()
    from dac_codec import make_codec

    torch.manual_seed(CFG.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    codec = make_codec(a.codec, dev)
    sig = {"model_id": codec.model_id, "n_q": codec.n_q,
           "channels": codec.channels, "n_slots": codec.n_slots,
           "codebook_sz": codec.codebook_sz, "vocab_size": codec.vocab_size,
           "bos": codec.bos, "pad": codec.pad, "sr": codec.sr,
           "tokens_per_sec": codec.tokens_per_sec,
           "frame_rate": codec.frame_rate, "codec_kind": a.codec}
    if a.pretrain_cache:
        paths = sorted(Path(a.pretrain_cache).glob("*.npz"))
        t5_id, d_text = CFG.t5_id, 768      # layers built, dormant
    else:
        from textenc import LyricEncoder
        from data import build_cache
        te = LyricEncoder(device=dev)
        # per-codec cache dir: DAC and EnCodec tokens are different worlds
        cdir = ROOT / "data" / ("cache" if a.codec == "encodec" else
                                f"cache_{a.codec}")
        paths = build_cache(codec, te, raw_dir=a.music_dir, cache_dir=cdir)
        t5_id, d_text = te.model_id, te.d_text
        del te
    del codec
    import gc; gc.collect()
    if dev == "cuda":
        torch.cuda.empty_cache()

    if a.overfit:
        paths = paths[:a.overfit]
    ds = DelayDataset(paths, a.seq_steps, sig["n_q"], sig["channels"],
                      sig["codebook_sz"], sig["bos"], sig["pad"])
    n_val = max(1, int(len(ds) * CFG.val_frac)) if not a.overfit else 0
    if n_val and len(ds) > n_val * 2:
        tr, va = _split_by_song(ds, CFG.val_frac, CFG.seed)
    else:
        tr, va = ds, None
    dl = DataLoader(tr, batch_size=a.batch_size, shuffle=True, collate_fn=collate,
                    drop_last=len(tr) >= a.batch_size)
    vdl = DataLoader(va, batch_size=a.batch_size, collate_fn=collate) if va else None

    dropout = 0.0 if a.overfit else CFG.dropout
    model = DelayMusicTransformer(
        vocab_size=sig["vocab_size"], d_model=CFG.d_model, n_heads=CFG.n_heads,
        n_layers=CFG.n_layers, d_ff=CFG.d_ff, max_seq_len=a.seq_steps,
        n_slots=sig["n_slots"], codebook_sz=sig["codebook_sz"],
        dropout=dropout, d_text=d_text, rope_base=CFG.rope_base,
        grad_checkpoint=True,
    ).to(dev)

    if a.init_from:
        ck = torch.load(a.init_from, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"])
        ui.log(f"[green]initialized from[/] {a.init_from} "
               f"[dim](weights only — fresh optimizer/schedule)[/]")

    from optim import Lamb
    opt = Lamb(model.parameters(), lr=a.lr, betas=CFG.betas,
               weight_decay=CFG.weight_decay)
    lr_scale = {"v": 1.0}
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: lr_scale["v"] * lr_lambda(s, CFG.warmup_steps, a.steps))
    backoff = GradNormBackoff(lr_scale, floor=2.0, abs_fire=8.0)

    step0 = 0
    if a.resume:
        ck = torch.load(a.resume, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"]); step0 = ck["step"]
        lr_scale["v"] = ck.get("lr_scale", 1.0)
        backoff.since_fire = 0
        ui.log(f"[green]resumed[/] {a.resume} @ step {step0}")

    names = [f"q{s//sig['channels']}{'LR'[s % sig['channels']]}"
             for s in range(sig["n_slots"])]
    ui.rule("MusicMaker — DELAY-PATTERN training")
    ui.kv_table([
        ("model", f"{model.count_parameters()/1e6:.1f}M  d={CFG.d_model} "
                  f"L={CFG.n_layers} heads={sig['n_slots']}x{sig['codebook_sz']}"),
        ("codec", f"{a.codec}: {sig['n_q']}x{sig['codebook_sz']} @ "
                  f"{sig['frame_rate']:.1f} fps, vocab {sig['vocab_size']}"),
        ("window", f"{a.seq_steps} steps = "
                   f"{a.seq_steps/sig['frame_rate']:.0f}s audio"),
        ("data", f"{len(tr)} windows over {len(ds.tracks)} tracks"
                 + (f"  [dim](val: whole songs held out)[/]" if va else "")),
        ("batch", f"{a.batch_size} x {a.grad_accum} accum — "
                  f"{a.batch_size*a.grad_accum*a.seq_steps/sig['frame_rate']:.0f}s "
                  f"audio/step"),
        ("mode", "[bold]PRETRAIN (unconditional)[/]" if a.pretrain_cache
                 else f"conditioned — dropout {dropout}, "
                      f"cond-dropout {CFG.cond_dropout}"),
        ("optimizer", f"LAMB {a.lr:.0e}, backoff floor 2.0 + abs 8.0"),
    ])
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    best_val, best_step = float("inf"), -1
    metrics_p = CKPT_DIR / "metrics.csv"
    new_file = not metrics_p.exists() or step0 == 0
    metrics_f = open(metrics_p, "a" if not new_file else "w", buffering=1)
    if new_file:
        metrics_f.write("step,loss,ppl,lr,grad_norm,val_loss,vram_gb,elapsed_s\n")

    def save(path, step, val=None):
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "step": step, "cfg": vars(a),
                    "val_loss": val, "lr_scale": lr_scale["v"],
                    "codec_sig": {**sig, "t5_id": t5_id, "d_text": d_text,
                                  "d_model": CFG.d_model, "n_heads": CFG.n_heads,
                                  "n_layers": CFG.n_layers, "d_ff": CFG.d_ff,
                                  "seq_steps": a.seq_steps}}, path)

    model.train()
    it = iter(dl)
    t0 = time.time()
    prog = ui.train_progress()
    task = prog.add_task("train", total=a.steps, completed=step0, stats="") if prog else None
    if prog:
        prog.start()
    step, tot = step0, float("nan")
    interrupted = False
    try:
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
                if ctx is not None and not a.overfit \
                        and torch.rand(1).item() < CFG.cond_dropout:
                    ctx, msk = None, None
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
                    lg = model(x, ctx, msk)
                    loss = F.cross_entropy(lg.reshape(-1, lg.shape[-1]),
                                           y.reshape(-1), ignore_index=-100)
                (loss / a.grad_accum).backward()
                tot += loss.item() / a.grad_accum
            gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                         CFG.grad_clip))
            opt.step(); sched.step()
            msg = backoff.update(gnorm)
            if msg:
                if prog:
                    prog.stop()
                ui.log(f"[bold yellow]!! {msg}[/]")
                if prog:
                    prog.start()

            peak = torch.cuda.max_memory_allocated() / 1e9 if dev == "cuda" else 0
            metrics_f.write(f"{step},{tot:.4f},{math.exp(min(tot,20)):.1f},"
                            f"{sched.get_last_lr()[0]:.3e},{gnorm:.4f},,"
                            f"{peak:.2f},{time.time()-t0:.0f}\n")
            if prog:
                prog.update(task, completed=step + 1, stats=(
                    f"loss [bold]{tot:.3f}[/] lr {sched.get_last_lr()[0]:.1e} "
                    f"gn {gnorm:.1f} vram {peak:.1f}GB"))
            if step % 10 == 0:
                assert peak < CFG.vram_budget_gb, \
                    f"peak {peak:.1f}GB over budget — Windows spills silently"

            if step == step0 or (step > step0 and step % CFG.eval_interval == 0):
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16,
                                                     enabled=(dev == "cuda")):
                    ps = per_head_loss(model(x[:1], ctx[:1] if ctx is not None else None,
                                             msk[:1] if msk is not None else None),
                                       y[:1])
                if prog:
                    prog.stop()
                title = f"per-head loss @ step {step}"
                if step == step0:
                    title += f"  [dim](untrained ~= ln(cb) = " \
                             f"{math.log(sig['codebook_sz']):.2f})[/]"
                ui.slot_table(names, ps, title=title)
                if vdl:
                    vl = evaluate(model, vdl, dev)
                    metrics_f.write(f"{step},{tot:.4f},{math.exp(min(tot,20)):.1f},"
                                    f"{sched.get_last_lr()[0]:.3e},{gnorm:.4f},"
                                    f"{vl:.4f},{peak:.2f},{time.time()-t0:.0f}\n")
                    if vl < best_val:
                        best_val, best_step = vl, step
                        save(CKPT_DIR / "best.pt", step, val=vl)
                        ui.log(f"   [dim]val[/] [bold green]{vl:.3f}[/] "
                               f"[dim]<- new best[/]")
                    else:
                        ui.log(f"   [dim]val[/] [bold]{vl:.3f}[/] "
                               f"[dim](best {best_val:.3f} @ {best_step})[/]")
                if prog:
                    prog.start()

            if step > step0 and step % CFG.ckpt_interval == 0:
                save(CKPT_DIR / f"step{step}.pt", step)
                ui.log(f"   [dim]saved step{step}.pt[/]")

    except KeyboardInterrupt:
        interrupted = True
        if prog:
            prog.stop()
        ui.log("[yellow]interrupted — saving resumable checkpoint...[/]")
    except BaseException:
        if prog:
            prog.stop()
        tmp = CKPT_DIR / "interrupted.pt.tmp"
        save(tmp, step)
        tmp.replace(CKPT_DIR / "interrupted.pt")
        ui.log(f"[red]crashed — saved resumable interrupted.pt @ step {step}[/]")
        raise
    finally:
        try:
            metrics_f.flush()
        except Exception:
            pass

    if interrupted:
        tmp = CKPT_DIR / "interrupted.pt.tmp"
        save(tmp, step)
        tmp.replace(CKPT_DIR / "interrupted.pt")
        metrics_f.close()
        ui.panel(f"stopped at step [bold]{step}[/]/{a.steps}. Resume:\n\n"
                 f"  [bold]python train_delay.py ... --resume "
                 f"checkpoints_delay/interrupted.pt[/]\n\n"
                 f"[dim]best: checkpoints_delay/best.pt "
                 f"(val {best_val:.3f} @ {best_step})[/]",
                 title="saved — resumable", style="yellow")
        return

    if prog:
        prog.stop()
    metrics_f.close()
    save(CKPT_DIR / "final.pt", a.steps)
    ui.panel(f"final: checkpoints_delay/final.pt\n"
             f"best val [bold]{best_val:.3f}[/] @ step {best_step} "
             f"(checkpoints_delay/best.pt)", title="done", style="green")


if __name__ == "__main__":
    main()
