"""
dtrain.py — train the diffusion music model.

    python dtrain.py                                   # full run
    python dtrain.py --overfit 2 --steps 400           # correctness check

Compare to the AR project's train.py. The loop is SIMPLER: no per-slot loss, no
interleave to keep aligned, no vocab. Just "noise it, predict the noise, MSE".
All the subtlety moved into diffusion.py's schedule algebra.

EMA is not optional here. Diffusion sample quality depends heavily on averaged
weights -- the raw weights bounce around enough that sampling from them looks
noticeably worse. The AR model does not need this.

PER-TIMESTEP LOSS is this project's diagnostic, mirroring the AR project's
per-slot loss. What matters is that it VARIES with t; flat loss means the
timestep conditioning is not reaching the network and the model is denoising
blind.

The direction is counter-intuitive under v-prediction, and measured here it
runs HIGH at low t (~0.94) and LOW at high t (~0.62). That is correct: at
t~999 the target v ~= noise, which the model can largely read off its own
noisy input; at t~0 the target leans on z0, which requires actually having
learned the data. Under eps-prediction the slope inverts -- which is exactly
why eps-pred gives a near-free loss at high t and why v-pred is the better
objective for music, where global structure is decided at high noise.
"""

import argparse
import copy
import math
import time

import torch
from torch.utils.data import DataLoader

import ui
from dconfig import CKPT_DIR, DCFG
from ddata import LatentDataset, build_latent_cache, collate
from diffusion import Diffusion


def lr_lambda(step, warmup, total):
    if step < warmup:
        return step / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    return 0.1 + 0.45 * (1 + math.cos(math.pi * min(1.0, p)))


@torch.no_grad()
def per_timestep_loss(model, diff, z0, ctx, msk, buckets=4):
    """Loss in timestep buckets. Must VARY with t; flat means t isn't wired in.
    Under v-prediction it falls as t rises -- see the module docstring."""
    out = []
    edges = torch.linspace(0, diff.T, buckets + 1).long()
    for i in range(buckets):
        t = torch.randint(int(edges[i]), int(max(edges[i] + 1, edges[i + 1])),
                          (z0.shape[0],), device=z0.device)
        noise = torch.randn_like(z0)
        pred = model(diff.q_sample(z0, t, noise), t, ctx, msk)
        out.append(torch.nn.functional.mse_loss(
            pred, diff.target(z0, noise, t)).item())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=DCFG.max_steps)
    ap.add_argument("--n-frames", type=int, default=DCFG.n_frames)
    ap.add_argument("--batch-size", type=int, default=DCFG.batch_size)
    ap.add_argument("--grad-accum", type=int, default=DCFG.grad_accum)
    ap.add_argument("--lr", type=float, default=DCFG.lr)
    ap.add_argument("--overfit", type=int, default=0)
    ap.add_argument("--resume", type=str, default=None)
    ap.add_argument("--music-dir", type=str, default=None,
                    help="train from ANY folder (recursive). Nothing is copied.")
    a = ap.parse_args()

    import transformers
    transformers.logging.set_verbosity_error()
    from dcodec import LatentCodec
    from dmodel import DiT
    from textenc import LyricEncoder

    torch.manual_seed(DCFG.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    codec = LatentCodec(device=dev)
    te = LyricEncoder(device=dev)
    paths, scale = build_latent_cache(codec, te, raw_dir=a.music_dir)
    if a.overfit:
        paths = paths[:a.overfit]

    dropout = 0.0 if a.overfit else DCFG.dropout
    ds = LatentDataset(paths, a.n_frames, scale, random_crop=not a.overfit)
    dl = DataLoader(ds, batch_size=a.batch_size, shuffle=True, collate_fn=collate,
                    drop_last=len(ds) >= a.batch_size)

    model = DiT(latent_dim=codec.latent_dim, d_model=DCFG.d_model,
                n_heads=DCFG.n_heads, n_layers=DCFG.n_layers, d_ff=DCFG.d_ff,
                n_frames=max(a.n_frames, DCFG.n_frames), dropout=dropout,
                d_text=te.d_text).to(dev)
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)

    diff = Diffusion(DCFG.timesteps, DCFG.schedule, DCFG.prediction, dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=DCFG.betas,
                            weight_decay=DCFG.weight_decay, fused=(dev == "cuda"))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: lr_lambda(s, DCFG.warmup_steps, a.steps))

    step0 = 0
    if a.resume:
        ck = torch.load(a.resume, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"]); ema.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        step0 = ck["step"]

    ui.rule("MusicMakerDiffusion — training")
    ui.kv_table([
        ("model", f"DiT {model.count_parameters()/1e6:.1f}M  d={DCFG.d_model} "
                  f"L={DCFG.n_layers} [dim](AR project: 138.4M — param-matched)[/]"),
        ("latent", f"{codec.latent_dim}-dim @ {codec.frame_rate}Hz, scale {scale:.4f}"),
        ("context", f"{a.n_frames} frames = {a.n_frames/codec.frame_rate:.2f}s "
                    f"[dim](bidirectional — sees it all at once)[/]"),
        ("diffusion", f"{DCFG.schedule} schedule, {DCFG.prediction}-prediction, "
                      f"{DCFG.timesteps} steps"),
        ("data", f"{len(ds)} windows over {len(ds.tracks)} tracks"),
        ("batch", f"{a.batch_size} x {a.grad_accum} accum "
                  f"[dim](x{2 if DCFG.stereo else 1} channels as rows)[/]"),
        ("mode", "[bold yellow]OVERFIT[/]" if a.overfit else "train"),
    ])
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

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
            z0 = b["z"].to(dev, non_blocking=True)
            ctx = b["ctx"].to(dev) if b["ctx"] is not None else None
            msk = b["ctx_mask"].to(dev) if b["ctx_mask"] is not None else None
            if ctx is not None and not a.overfit and torch.rand(1).item() < DCFG.cond_dropout:
                ctx, msk = None, None          # CFG's unconditional path
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
                loss, _ = diff.loss(model, z0, ctx, msk)
            (loss / a.grad_accum).backward()
            tot += loss.item() / a.grad_accum

        torch.nn.utils.clip_grad_norm_(model.parameters(), DCFG.grad_clip)
        opt.step(); sched.step()

        # EMA: diffusion sampling quality leans on this heavily.
        with torch.no_grad():
            d = DCFG.ema_decay
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.mul_(d).add_(pm.detach(), alpha=1 - d)
            for be, bm in zip(ema.buffers(), model.buffers()):
                be.copy_(bm)

        peak = torch.cuda.max_memory_allocated() / 1e9 if dev == "cuda" else 0
        if prog:
            dt = time.time() - t0
            sps = (step - step0 + 1) / max(dt, 1e-9)
            prog.update(task, completed=step + 1, stats=(
                f"loss [bold]{tot:.4f}[/] lr {sched.get_last_lr()[0]:.1e} "
                f"{sps:.1f} it/s vram {peak:.1f}GB"))
        elif step % 10 == 0:
            print(f"step {step:5d} | loss {tot:.4f} | vram {peak:.1f}GB")

        if step % 10 == 0:
            assert peak < DCFG.vram_budget_gb, (
                f"peak VRAM {peak:.1f}GB exceeds {DCFG.vram_budget_gb}GB; on "
                "Windows this silently spills to shared memory instead of "
                "raising OOM. Lower batch_size or n_frames.")

        if step > step0 and step % DCFG.eval_interval == 0:
            pl = per_timestep_loss(ema, diff, z0[:2],
                                   ctx[:2] if ctx is not None else None,
                                   msk[:2] if msk is not None else None)
            if prog:
                prog.stop()
            names = ["t 0-250", "t 250-500", "t 500-750", "t 750-1000"]
            ui.slot_table(names, pl, title=f"per-timestep loss @ step {step}")
            # Under v-prediction, loss falling as t rises is CORRECT (see the
            # module docstring). Only flatness is a red flag.
            if max(pl) - min(pl) < 0.01:
                ui.log("[bold red]!! loss flat across t — timestep conditioning "
                       "may not be reaching the network[/]")
            if prog:
                prog.start()

        if step > step0 and step % DCFG.ckpt_interval == 0:
            _save(CKPT_DIR / f"step{step}.pt", model, ema, opt, sched, step, a,
                  codec, te, scale)

    if prog:
        prog.stop()
    _save(CKPT_DIR / "final.pt", model, ema, opt, sched, a.steps, a, codec, te, scale)
    ui.panel(f"final checkpoint: [bold]{CKPT_DIR / 'final.pt'}[/]\n"
             f"last loss: [bold]{tot:.4f}[/]", title="done", style="green")


def _save(path, model, ema, opt, sched, step, a, codec, te, scale):
    torch.save({
        "model": model.state_dict(), "ema": ema.state_dict(),
        "opt": opt.state_dict(), "sched": sched.state_dict(),
        "step": step, "cfg": vars(a),
        "sig": {
            "codec_id": codec.model_id, "latent_dim": codec.latent_dim,
            "channels": codec.channels, "sr": codec.sr,
            "frame_rate": codec.frame_rate, "scale": scale,
            "t5_id": te.model_id, "d_text": te.d_text,
            "d_model": DCFG.d_model, "n_heads": DCFG.n_heads,
            "n_layers": DCFG.n_layers, "d_ff": DCFG.d_ff,
            "n_frames": max(a.n_frames, DCFG.n_frames),
            "timesteps": DCFG.timesteps, "schedule": DCFG.schedule,
            "prediction": DCFG.prediction,
        }}, path)
    ui.log(f"   [dim]saved[/] {path}")


if __name__ == "__main__":
    main()
