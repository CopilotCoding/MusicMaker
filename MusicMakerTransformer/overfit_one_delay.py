r"""
overfit_one_delay.py — the delay-pattern proof: memorize ONE song, regenerate
it, and measure the speedup over the flat interleave.

    python overfit_one_delay.py --song "..\Grunge\...\track.mp3"

Same test, same standard as overfit_one.py (the flat version hit 100% free-run
match): feed the song in order as step-chunks, memorize, regenerate greedily
from BOS, report token match + wavs + steps/s. A 10.2s window here is 515
sequence positions instead of 4096 — the whole point being measured.

Chunk layout (identical rules to the flat model, in STEPS):
    chunk 0:  [BOSrow, g0 .. g_{S-2}]  ->  [g0 .. g_{S-1}]
    chunk k:  [g_{kS-1} .. g_{kS+S-2}] ->  [g_{kS} .. g_{kS+S-1}]   (seam fix)
Generation resets the KV cache every S rows, exactly as training reads.
"""

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import ui
from config import CFG
from delay import (DelayMusicTransformer, build_grid, grid_targets,
                   slot_delays, undelay_grid)


def sample_heads(logits, temperature: float, top_k: int):
    """(K, cb) logits -> (K,) class ids. Per-head temp + top-k; no slot mask
    needed — each head IS its codebook."""
    if temperature and temperature > 0.02:
        lg = logits / temperature
        if top_k and top_k < lg.shape[-1]:
            kth = lg.topk(top_k, dim=-1).values[..., -1:]
            lg = lg.masked_fill(lg < kth, float("-inf"))
        return torch.multinomial(lg.softmax(-1), 1)[:, 0]
    return logits.argmax(-1)


@torch.no_grad()
def gen_rows(model, n_rows, S, bos_row, delays, pad, codebook_sz,
             prime=None, temperature=0.01, top_k=1, device="cuda",
             frame_rate=50.0):
    """Chunk-aligned row generation mirroring training layout exactly.

    prime: (P, K) real grid rows to continue from (song start), or None.
    Returns (n_total_rows, K) banded grid including any prime.
    """
    K = len(delays)
    model.eval()
    prime = prime if prime is not None else torch.zeros((0, K), dtype=torch.long)
    P = prime.shape[0]

    # prefill in training-layout pieces, fresh cache per piece
    pieces = [torch.cat([bos_row[None], prime[:min(P, S - 1)]], dim=0)]
    p = S - 1
    while p < P:
        pieces.append(prime[p:p + S])
        p += S
    # bf16 autocast to MATCH training numerics: the model memorized under
    # autocast, and at 18 heads/row a borderline fp32-vs-bf16 argmax flip
    # anywhere derails the whole autoregressive rollout.
    amp = torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda"))
    caches = None
    for piece in pieces:
        with amp:
            lg, caches = model(piece[None].to(device), kv_caches=None,
                               return_caches=True, pos_offset=0)
    cache_len = pieces[-1].shape[0]

    rows = [r for r in prime]
    prog = ui.gen_progress()
    task = prog.add_task("rows", total=n_rows, stats="") if prog else None
    if prog:
        prog.start()
    for i in range(n_rows):
        s = P + i                                   # absolute row index
        cls = sample_heads(lg[0, -1].float(), temperature, top_k).cpu()  # (K,)
        row = cls + torch.arange(K) * codebook_sz   # band the ids (all CPU)
        # structural PADs at the song-start stagger: slot k has no token
        # before row delay_k. Training masked these; generation imposes them.
        for k in range(K):
            if s < delays[k]:
                row[k] = pad
        rows.append(row)
        if i + 1 < n_rows:
            if cache_len >= S:                      # chunk boundary: reset,
                caches = None                       # exactly like training
                cache_len = 0
            with amp:
                lg, caches = model(row[None, None].to(device), kv_caches=caches,
                                   return_caches=True, pos_offset=cache_len)
            cache_len += 1
        if prog:
            prog.update(task, completed=i + 1,
                        stats=f"[bold]{(P+i+1)/frame_rate:.1f}s[/] audio")
    if prog:
        prog.stop()
    return torch.stack(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--song", required=True)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--seq-steps", type=int, default=512,
                    help="window in TIMESTEPS. 512 = 10.24s = the same audio "
                         "the flat test's 4096 tokens covered, in 1/8 the "
                         "sequence positions.")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--gen-seconds", type=float, default=15.0)
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    ap.add_argument("--codec", choices=["dac", "encodec"], default="dac",
                    help="dac = 44.1kHz ~8kbps (measured: 'basically the "
                         "original'); encodec = the old 2.2kbps ceiling")
    ap.add_argument("--target-acc", type=float, default=99.99,
                    help="the audit bar: teacher-forced accuracy every chunk "
                         "should reach")
    ap.add_argument("--stop-loss", type=float, default=0.002,
                    help="default path: end the constant-LR phase once the "
                         "MEDIAN loss over one full chunk cycle drops below "
                         "this (median, not mean — one alias-fighting outlier "
                         "chunk held the mean hostage forever)")
    ap.add_argument("--cycles", action="store_true",
                    help="optional SGDR mode: high/medium/low LR passes "
                         "repeating until the audit clears --target-acc. The "
                         "default (constant LR + median stop + one annealed "
                         "consolidation) measured 100%% free-run; use this if "
                         "a song's audit won't converge on the default path")
    ap.add_argument("--auto", action="store_true",
                    help="experimental probe-up/back-off LR controller: rides "
                         "the LR up 5%% while full-pass mean loss keeps "
                         "improving, backs off 30%% when it doesn't — anneals "
                         "itself at convergence. Racing against the default; "
                         "not yet the champion")
    a = ap.parse_args()

    song = Path(a.song)
    if not song.exists():
        raise SystemExit(f"no such file: {song}")
    dev = ("cuda" if torch.cuda.is_available() else "cpu") \
        if a.device == "auto" else a.device

    import transformers
    transformers.logging.set_verbosity_error()
    from dac_codec import make_codec
    from optim import Lamb
    import audio_io

    out_dir = Path(__file__).parent
    stem = "".join(c if c.isalnum() or c in "-_" else "_" for c in song.stem)[:40]

    ui.rule(f"overfit-one, DELAY PATTERN — codec: {a.codec}")
    codec = make_codec(a.codec, dev)
    n_q, C, cb = codec.n_q, codec.channels, codec.codebook_sz
    K = n_q * C
    fr = codec.frame_rate
    delays = slot_delays(n_q, C)

    # per-codec cache: DAC and EnCodec tokens are different worlds — never mix
    from data import cache_name
    cdir = out_dir / "_overfit_cache"
    cdir.mkdir(exist_ok=True)
    suffix = "" if a.codec == "encodec" else f"__{a.codec}"
    npz = cdir / f"{cache_name(song, song.parent)}{suffix}.npz"
    if npz.exists():
        ui.log("[dim]song already cached — reusing[/]")
        flat = torch.from_numpy(np.load(npz)["codes"].astype(np.int64))
    else:
        flat = codec.encode_file(song).long()
        np.savez(npz, codes=flat.numpy().astype(np.int16))
    codes = codec.unflatten(flat[None])[0]            # (n_q, C, T)
    T = codes.shape[-1]

    grid = build_grid(codes, cb, codec.pad)           # (T+n_q-1, K)
    tgt = grid_targets(grid, cb)                      # class ids / -100
    S = a.seq_steps
    G = grid.shape[0]

    # Pad the grid out to whole windows: a song SHORTER than the window (the
    # whole-song-in-one-context case, and every short track in the pretrain)
    # fills its tail with PAD rows, masked (-100) out of the loss. This is the
    # exact path the real training data pipeline uses for short tracks.
    W = max(1, math.ceil(G / S))
    L = W * S
    if L > G:
        pad_rows = torch.full((L - G, K), codec.pad, dtype=torch.long)
        grid = torch.cat([grid, pad_rows])
        tgt = torch.cat([tgt, torch.full((L - G, K), -100, dtype=torch.long)])

    # in-order chunks with BOS + seam, in steps
    bos_row = torch.full((K,), codec.bos, dtype=torch.long)
    chunks = []
    x0 = torch.cat([bos_row[None], grid[:S - 1]], dim=0)
    chunks.append((x0, tgt[:S]))
    for k in range(1, W):
        chunks.append((grid[k * S - 1: k * S + S - 1], tgt[k * S: k * S + S]))

    model = DelayMusicTransformer(
        vocab_size=codec.vocab_size, d_model=CFG.d_model, n_heads=CFG.n_heads,
        n_layers=CFG.n_layers, d_ff=CFG.d_ff,
        max_seq_len=max(S, 2048), n_slots=K, codebook_sz=cb,
        dropout=0.0, d_text=None, rope_base=CFG.rope_base,
        grad_checkpoint=CFG.grad_checkpoint,
    ).to(dev)
    opt = Lamb(model.parameters(), lr=a.lr, betas=CFG.betas, weight_decay=0.0)

    ui.kv_table([
        ("song", f"{song.name}  ({T/fr:.0f}s, {T} frames)"),
        ("codec", f"{a.codec}: {n_q}x{cb} @ {fr:.1f} fps, vocab {codec.vocab_size}"),
        ("window", f"{S} steps = {S/fr:.1f}s audio  "
                   f"[dim](flat needed {S*K} positions for this)[/]"),
        ("chunks", f"{len(chunks)} in-order"),
        ("model", f"{model.count_parameters()/1e6:.1f}M params, {K} heads"),
        ("device", dev),
    ])

    # ---- memorize: HIGH -> MEDIUM -> LOW cycles until done (SGDR-style) ----
    # WHY CYCLES, not one long constant-LR run + cleanup: cycling one chunk
    # per step at constant high LR never converges — every update partially
    # overwrites the other chunks, and the model ORBITS the joint optimum at
    # a radius set by the step size (measured: 26 chunks stuck just below
    # target after 8000 flat steps; a targeted-polish phase made it WORSE by
    # concentrating interference — 5000 steps on one stuck chunk collapsed
    # free-run from 89% to 1%). Warm-restart cycles are the textbook answer
    # (Smith's CLR; Loshchilov & Hutter's SGDR): the HIGH pass buys plasticity
    # to fix what small steps can't reach, the LOW pass collapses the orbit so
    # all chunks settle JOINTLY, and repeating alternates the two until the
    # audit — not a loss printout — says every chunk is memorized.
    def audit(only=None):
        model.eval()
        idxs = range(len(chunks)) if only is None else only
        out = {}
        with torch.no_grad():
            for ci in idxs:
                cx, cy = chunks[ci]
                with torch.autocast("cuda", dtype=torch.bfloat16,
                                    enabled=(dev == "cuda")):
                    lg = model(cx[None].to(dev))
                pred = lg[0].float().argmax(-1)
                yy = cy.to(dev)
                m = yy != -100
                out[ci] = 100.0 * (pred[m] == yy[m]).float().mean().item()
        model.train()
        return out

    last = 99.0

    def train_chunk(ci):
        nonlocal last
        cx, cy = chunks[ci]
        x = cx[None].to(dev); y = cy[None].to(dev)
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
            lg = model(x)
            loss = F.cross_entropy(lg.reshape(-1, cb), y.reshape(-1),
                                   ignore_index=-100)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.grad_clip)
        opt.step()
        last = loss.item()

    model.train()
    t0 = time.time()
    steps_done = 0
    accs = {}

    # Chunks the free-run test will actually regenerate. A weak chunk INSIDE
    # this span derails the match from that point on (measured: cycle mode
    # accepted chunk #0 at 99.5% as "residue" -> free-run 1.4%, primed 100%);
    # a weak chunk OUTSIDE it is provably harmless to the test. No accept-
    # and-stop path may fire while an in-span chunk is below target.
    span = set(range((min(int(a.gen_seconds * fr), T) + n_q - 1) // S + 1))

    def bad_in_span(ac):
        return sorted(ci for ci, v in ac.items()
                      if v < a.target_acc and ci in span)

    prog = ui.train_progress()
    task = prog.add_task("memorizing", total=a.steps, stats="") if prog else None
    if prog:
        prog.start()

    if a.cycles:
        # OPTIONAL SGDR mode: high/medium/low warm-restart cycles, audit
        # between cycles, repeat until the audit clears — with three guards
        # all MEASURED necessary:
        #  1. DECAYING RESTARTS: once the audit is nearly converged, a full
        #     high-LR restart destroys the converged state (measured: cycle 24
        #     was 1-below-target, cycle 25's 1e-3 pass wrecked 31 chunks).
        #     Near convergence the high phase is skipped; very near, medium
        #     too — late cycles ARE anneals.
        #  2. NEVER END MID-HIGH: a cycle only starts if the step budget can
        #     finish it (the cap once landed mid-high-phase and the run ended
        #     with the model at its most disrupted -> free-run 1.5%).
        #  3. STAGNATION STOP: if audit MEAN ACCURACY stops improving for 3
        #     cycles AND no below-target chunk sits in the free-run span, the
        #     residue is alias-unfittable — accept and stop. (Judging on the
        #     below-target COUNT was measured wrong twice: it saturates early,
        #     and it once accepted chunk #0 — in-span — at 99.5%, which
        #     produced free-run 1.4% with primed 100%.)
        PHASES = ((a.lr, 6), (a.lr / 10, 2), (a.lr / 100, 2))
        cycle = 0
        prev_acc, stagnant = None, 0
        while steps_done < a.steps:
            cycle += 1
            phases = PHASES
            if accs:
                mean_acc = sum(accs.values()) / len(accs)
                if mean_acc >= 99.98 and not bad_in_span(accs):
                    phases = PHASES[2:]      # consolidation only
                elif mean_acc >= 99.9:
                    # no high restart near convergence; medium stays available
                    # while any in-span chunk is weak — lr/100 alone was
                    # measured too small to unstick chunk #0
                    phases = PHASES[1:]
            need = sum(p for _, p in phases) * len(chunks)
            if steps_done + need > a.steps:
                ui.log(f"[dim]stopping: {a.steps - steps_done} steps left "
                       f"< one full cycle ({need}) — refusing to end "
                       f"mid-phase[/]")
                break
            for lr_c, passes in phases:
                for g in opt.param_groups:
                    g["lr"] = lr_c
                for _ in range(passes):
                    for ci in range(len(chunks)):
                        train_chunk(ci)
                        steps_done += 1
                        if prog:
                            prog.update(task, completed=steps_done, stats=(
                                f"cyc {cycle} lr {lr_c:.0e} "
                                f"loss [bold]{last:.4f}[/] "
                                f"[dim]{steps_done/(time.time()-t0):.1f} st/s[/]"))
            accs = audit()
            n_below = sum(1 for v in accs.values() if v < a.target_acc)
            mean_acc = sum(accs.values()) / len(accs)
            if prog:
                prog.stop()
            ui.log(f"[dim]cycle {cycle} ({len(phases)} phases): mean "
                   f"{mean_acc:.3f}% "
                   f"worst {min(accs.values()):.3f}%  {n_below} below target  "
                   f"({steps_done} steps)[/]")
            if prog:
                prog.start()
            if n_below == 0:
                break
            if prev_acc is not None and mean_acc <= prev_acc + 0.001:
                stagnant += 1
            else:
                stagnant = 0
            prev_acc = mean_acc
            if stagnant >= 3:
                stuck = bad_in_span(accs)
                if not stuck:
                    ui.log(f"[yellow]{n_below} chunk(s) stagnant for 3 cycles, "
                           f"all outside the free-run span — accepting as "
                           f"alias-unfittable residue[/]")
                    break
                stagnant = 0
                ui.log(f"[yellow]stagnant but chunk(s) {stuck} are IN the "
                       f"free-run span — cannot accept, continuing[/]")
    elif a.auto:
        # EXPERIMENTAL probe-up/back-off controller. The unit of comparison is
        # one full in-order pass over ALL chunks — same chunks, same order, so
        # consecutive pass-means are like-for-like. (Per-step loss is NOT: it
        # reflects chunk difficulty, the confound that breaks naive
        # smoothness controllers.) Rule: pass-mean improved -> lr *= 1.05;
        # didn't -> lr *= 0.7. At convergence passes stop improving by
        # definition, so the controller backs off repeatedly: it ANNEALS
        # ITSELF. Audit every 5 passes decides done/stagnant.
        # GUARDS (measured necessary, first --auto run):
        #  1. Stagnation is judged on audit MEAN ACCURACY, never the
        #     below-target count — that count sits saturated at len(chunks)
        #     all through early training (fired at pass 20 with worst-chunk
        #     64% and still climbing fast).
        #  2. "Accept as residue" is gated on mean acc >= 99.9. Stagnant far
        #     below that is an LR problem, not aliasing — kick the LR down
        #     and keep training instead of quitting.
        #  3. The probe ceiling decays with accuracy (cycle mode's decaying-
        #     restart lesson): high LR is destructive once nearly converged.
        lr_now = a.lr
        LO, HI = a.lr / 1000, a.lr * 3
        hi_now = HI
        prev_mean = None
        pass_n = 0
        prev_acc, stagnant = None, 0
        while steps_done < a.steps:
            pass_n += 1
            for g in opt.param_groups:
                g["lr"] = lr_now
            tot = 0.0
            for ci in range(len(chunks)):
                train_chunk(ci)
                steps_done += 1
                tot += last
                if prog:
                    prog.update(task, completed=steps_done, stats=(
                        f"pass {pass_n} lr [bold]{lr_now:.2e}[/] "
                        f"loss {last:.4f} "
                        f"[dim]{steps_done/(time.time()-t0):.1f} st/s[/]"))
            mean = tot / len(chunks)
            if prev_mean is None or mean < prev_mean:
                lr_now = min(lr_now * 1.05, hi_now)  # improving: probe up
            else:
                lr_now = max(lr_now * 0.7, LO)       # worse: back off hard
            prev_mean = mean
            if pass_n % 5 == 0 or lr_now <= LO * 1.001:
                accs = audit()
                n_below = sum(1 for v in accs.values() if v < a.target_acc)
                mean_acc = sum(accs.values()) / len(accs)
                if prog:
                    prog.stop()
                ui.log(f"[dim]pass {pass_n}: lr {lr_now:.2e} "
                       f"pass-mean {mean:.4f}  mean acc {mean_acc:.3f}% "
                       f"worst {min(accs.values()):.3f}% "
                       f"{n_below} below  ({steps_done} steps)[/]")
                if prog:
                    prog.start()
                if n_below == 0:
                    break
                if lr_now <= LO * 1.001:
                    stuck = bad_in_span(accs)
                    if not stuck:
                        ui.log("[yellow]LR self-annealed to floor — "
                               "accepting[/]")
                        break
                    lr_now = a.lr / 100
                    ui.log(f"[yellow]LR at floor but chunk(s) {stuck} are IN "
                           f"the free-run span — bouncing lr to "
                           f"{lr_now:.2e}[/]")
                # decaying probe ceiling: no hot LR near convergence
                if mean_acc >= 99.98:
                    hi_now = min(hi_now, a.lr / 10)
                elif mean_acc >= 99.9:
                    hi_now = min(hi_now, a.lr)
                lr_now = min(lr_now, hi_now)
                # stagnation on accuracy, gated by accuracy
                if prev_acc is not None and mean_acc <= prev_acc + 0.001:
                    stagnant += 1
                else:
                    stagnant = 0
                prev_acc = mean_acc
                if stagnant >= 3:
                    stuck = bad_in_span(accs)
                    if mean_acc >= 99.9 and not stuck:
                        ui.log(f"[yellow]{n_below} chunk(s) stagnant for 3 "
                               f"audits at {mean_acc:.3f}%, all outside the "
                               f"free-run span — accepting as unfittable "
                               f"residue[/]")
                        break
                    stagnant = 0
                    if stuck and mean_acc >= 99.9:
                        # converged everywhere except in-span stragglers: the
                        # annealed LR is too small to unstick them — bounce up
                        lr_now = min(max(lr_now * 10, a.lr / 100), hi_now)
                        ui.log(f"[yellow]stagnant with chunk(s) {stuck} IN "
                               f"the free-run span — bouncing lr to "
                               f"{lr_now:.2e}[/]")
                    else:
                        # far from target and not moving: the LR is the
                        # problem, not the data — kick down and keep going
                        lr_now = max(lr_now * 0.5, LO)
                        ui.log(f"[yellow]stagnant at {mean_acc:.3f}% (too low "
                               f"to accept) — kicking lr down to "
                               f"{lr_now:.2e}[/]")
        if not accs:
            accs = audit()
    else:
        # DEFAULT path — MEASURED 100% free-run on the DAC test: constant LR
        # with a median early-stop, then one annealed consolidation.
        recent = []
        for step in range(a.steps):
            train_chunk(step % len(chunks))
            steps_done += 1
            recent.append(last)
            if len(recent) > len(chunks):
                recent.pop(0)
            if prog:
                prog.update(task, completed=step + 1, stats=(
                    f"loss [bold]{last:.4f}[/] "
                    f"[dim]{steps_done/(time.time()-t0):.1f} st/s[/]"))
            if len(recent) == len(chunks) and \
                    sorted(recent)[len(recent) // 2] < a.stop_loss:
                if prog:
                    prog.stop()
                ui.log(f"[green]early stop[/] @ step {step + 1}: median loss "
                       f"over a full chunk cycle < {a.stop_loss}")
                if prog:
                    prog.start()
                break
        # Annealed consolidation: constant-LR cycling ORBITS the joint optimum
        # (each chunk's update partially overwrites the rest); shrinking the LR
        # collapses the orbit so all chunks settle jointly. ~250 steps.
        for lr_c, cycles_n in ((a.lr / 10, 3), (a.lr / 100, 2)):
            for g in opt.param_groups:
                g["lr"] = lr_c
            for _ in range(cycles_n):
                for ci in range(len(chunks)):
                    train_chunk(ci)
                    steps_done += 1
        accs = audit()

    if prog:
        prog.stop()
    dt = time.time() - t0
    ui.log(f"[green]memorized[/] — {steps_done} steps in {dt:.0f}s "
           f"= [bold]{steps_done/dt:.2f} steps/s[/]")
    sps = steps_done / dt

    worst = min(accs, key=accs.get)
    n_below = sum(1 for v in accs.values() if v < a.target_acc)
    ui.kv_table([
        ("mean acc", f"{sum(accs.values())/len(accs):.3f}%  "
                     f"(teacher-forced, full audit)"),
        ("worst chunk", f"#{worst}: {accs[worst]:.3f}%  "
                        f"(target {a.target_acc}%)"),
        ("below target", f"{n_below} chunk(s)"),
        ("verdict", "[green]all chunks at target[/]" if n_below == 0 else
                    f"[yellow]step cap hit; residue is likely reset-boundary "
                    f"aliasing (unfittable tokens) — if outside the generated "
                    f"span it cannot affect the match[/]"),
    ], title="memorization audit")

    # ---- regenerate ----
    ui.rule("regenerating from the real start")
    n_frames = min(int(a.gen_seconds * fr), T)
    n_rows = n_frames + n_q - 1

    free = gen_rows(model, n_rows, S, bos_row, delays, codec.pad, cb,
                    prime=None, device=dev, frame_rate=fr)
    codes_f = undelay_grid(free[:n_rows], n_q, C, cb)
    m_free = 100.0 * (codes_f == codes[:, :, :n_frames]).float().mean().item()

    P = int(5 * fr)                                   # 5s prime, in rows
    cont = gen_rows(model, n_rows - P, S, bos_row, delays, codec.pad, cb,
                    prime=grid[:P], device=dev, frame_rate=fr)
    codes_c = undelay_grid(cont, n_q, C, cb)
    new_f = codes_c.shape[-1] - (P - (n_q - 1))       # frames beyond the prime
    cc = codes_c[:, :, -new_f:]
    rr = codes[:, :, codes_c.shape[-1] - new_f: codes_c.shape[-1]]
    m_cont = 100.0 * (cc == rr).float().mean().item()

    # ---- decode to wavs ----
    def to_flat(c):
        return codec.flatten(c[None].to(codec.device))[0]
    real_wav = codec.decode_flat(to_flat(codes[:, :, :n_frames])[None])
    free_wav = codec.decode_flat(to_flat(codes_f)[None])
    cont_wav = codec.decode_flat(to_flat(codes_c)[None])
    p_real = out_dir / f"{stem}__dly_real.wav"
    p_free = out_dir / f"{stem}__dly_freerun.wav"
    p_cont = out_dir / f"{stem}__dly_primed.wav"
    audio_io.save(real_wav.cpu(), p_real, codec.sr)
    audio_io.save(free_wav.cpu(), p_free, codec.sr)
    audio_io.save(cont_wav.cpu(), p_cont, codec.sr)

    chance = 100.0 / cb
    ui.rule("verdict")
    ui.kv_table([
        ("free-run match", f"[bold]{m_free:.1f}%[/]  (chance {chance:.3f}%)"),
        ("primed match", f"[bold]{m_cont:.1f}%[/]  (continuation only)"),
        ("throughput", f"[bold]{sps:.2f} steps/s[/] at a {S}-step window"),
        ("real", str(p_real.name)),
        ("free-run", str(p_free.name)),
        ("primed", str(p_cont.name)),
    ], title="delay pattern: did it memorize?")
    if m_free > 80:
        ui.panel("[bold green]DELAY PATTERN VERIFIED[/] — memorized and "
                 "regenerated through the parallel heads, delay stagger, "
                 "chunk resets and undelay. The architecture is sound; "
                 "full-song context (16384 steps = 327s) is now on the table.",
                 title="green light", style="green")
    else:
        ui.panel("[bold red]DID NOT REPRODUCE[/] — loss low but match poor "
                 "means the generation path (row layout / delay alignment / "
                 "cache resets) disagrees with training. Check the stagger "
                 "and the structural PADs first.", title="red flag",
                 style="red")


if __name__ == "__main__":
    main()
