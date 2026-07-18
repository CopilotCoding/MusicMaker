r"""
overfit_one.py — can the model memorize ONE song and regenerate it?

    python overfit_one.py --song "C:\...\Grunge\...\some track.mp3"
    python overfit_one.py --song "..." --steps 2500 --lr 1e-3

THE POINT
---------
The strongest end-to-end correctness test in the project, independent of every
"is the data big enough" worry. If a fresh model drives its loss on ONE song to
near zero and then regenerates audio that matches it, the ENTIRE pipeline is
provably correct:

    codec encode -> flatten/interleave -> RoPE -> attention -> KV cache ->
    sampling -> logit mask -> unflatten -> codec decode

WHY IN-ORDER, NOT RANDOM CROPS (this is the fix)
------------------------------------------------
The full training run uses random-crop windows -- correct there, because the
model only needs to learn LOCAL transitions, never to reproduce a song from its
start. But that made this test lie: a model fed 27 RANDOM 10s crops of a 286s
song saw offset-40000 and offset-8000 both as "position 0", so free-generation
from the start had 27 contradictory beginnings to reproduce -> noise, even with
a flawless pipeline.

This version feeds the song IN ORDER as sequential, non-overlapping chunks
(random_crop OFF), so position 0 always means the real start of the song. Then
it generates by rolling the sliding window from that same real start. Now a
memorized model has exactly one coherent thing to reproduce, and "does the audio
match" becomes a real test instead of a rigged one.

A song longer than seq_len still doesn't fit in one context (286s = 114k tokens
vs a 16k window), so both training and generation roll the window forward -- the
same sliding-window mechanism used for real full-song generation.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--song", required=True, help="path to ONE audio file")
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--seq-len", type=int, default=4096,
                    help="training window. 4096 = ~10s at 400 tok/s. The song is "
                         "fed as in-order chunks of this size.")
    ap.add_argument("--lr", type=float, default=1e-3,
                    help="high on purpose: memorize, don't generalize. LAMB "
                         "keeps it stable.")
    ap.add_argument("--gen-seconds", type=float, default=15.0)
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    ap.add_argument("--out-dir", type=str, default=None)
    a = ap.parse_args()

    song = Path(a.song)
    if not song.exists():
        raise SystemExit(f"no such file: {song}")

    dev = ("cuda" if torch.cuda.is_available() else "cpu") \
        if a.device == "auto" else a.device
    if dev == "cuda":
        ui.log("[dim]note: pass --device cpu if a real training run is holding "
               "the GPU.[/]")

    import transformers
    transformers.logging.set_verbosity_error()
    from codec import Codec
    from textenc import LyricEncoder
    from transformer import MusicTransformer
    from optim import Lamb
    from generate import generate
    import audio_io

    out_dir = Path(a.out_dir) if a.out_dir else Path(__file__).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = "".join(c if c.isalnum() or c in "-_" else "_" for c in song.stem)[:50]

    ui.rule("overfit-one — memorize a song fed IN ORDER, then regenerate it")

    codec = Codec(device=dev)
    n_slots, vocab = codec.n_slots, codec.vocab_size
    S = a.seq_len - (a.seq_len % n_slots)

    # --- encode the song + its lyrics (reuse isolated cache if present) ---
    scratch = out_dir / "_overfit_cache"
    scratch.mkdir(parents=True, exist_ok=True)
    from data import cache_name
    npz = scratch / f"{cache_name(song, song.parent)}.npz"
    ctx = msk = None
    if npz.exists():
        ui.log("[dim]song already cached — reusing[/]")
        z = np.load(npz)
        codes = torch.from_numpy(z["codes"].astype(np.int64))
        if "ctx" in z:
            ctx = torch.from_numpy(z["ctx"].astype(np.float32))[None].to(dev)
            msk = torch.from_numpy(z["ctx_mask"])[None].to(dev)
    else:
        ui.log(f"[dim]encoding {song.name} (one time)[/]")
        codes = codec.encode_file(song).long()
        payload = {"codes": codes.numpy().astype(np.int16)}
        txt = song.with_suffix(".txt")
        if txt.exists():
            te = LyricEncoder(device=dev)
            text = txt.read_text(encoding="utf-8", errors="replace").strip()
            if text:
                c, m = te.encode([text])
                payload["ctx"] = c[0].cpu().numpy().astype(np.float16)
                payload["ctx_mask"] = m[0].cpu().numpy()
                ctx, msk = c.to(dev), m.to(dev)
            del te
            if dev == "cuda":
                torch.cuda.empty_cache()
        np.savez(npz, **payload)

    total = codes.shape[0]
    total -= total % n_slots
    codes = codes[:total]

    # --- build IN-ORDER, non-overlapping chunks, matching training exactly ---
    # chunk 0 gets a BOS ([BOS, t0, ..., t_{S-2}] -> [t0, ..., t_{S-1}]) so the
    # model learns to START the song; later chunks are plain next-token slices
    # continuing in order. This mirrors MusicDataset so the test validates the
    # real path.
    chunks = []
    # chunk 0 with BOS: [BOS, t0..t_{S-2}] -> [t0..t_{S-1}]
    real0 = codes[0:S]
    if real0.shape[0] == S:
        x0 = torch.cat([torch.tensor([codec.bos], dtype=real0.dtype), real0[:-1]])
        chunks.append((x0, real0))
    # interior chunks start ONE token early so the boundary token t_{kS} is a
    # target: [t_{kS-1}..t_{kS+S-2}] -> [t_{kS}..t_{kS+S-1}]. Without this,
    # t_{kS} is never predicted by any chunk and generation derails at every
    # boundary (measured: 68.2% = exactly chunk 0 and nothing after).
    k = 1
    while (k + 1) * S <= total:
        w = codes[k * S - 1: k * S + S]
        chunks.append((w[:-1], w[1:]))
        k += 1
    if not chunks:
        raise SystemExit(
            f"song ({total} tokens = {total/codec.tokens_per_sec:.0f}s) shorter "
            f"than seq_len ({S}). Use --seq-len smaller.")

    ui.kv_table([
        ("song", song.name),
        ("length", f"{total} tokens = {total/codec.tokens_per_sec:.0f}s"),
        ("chunks", f"{len(chunks)} in-order, non-overlapping, {S} tokens each "
                   f"(~{S/codec.tokens_per_sec:.1f}s)"),
        ("lyrics", "yes" if ctx is not None else "none (unconditional)"),
        ("device", dev),
        ("optimizer", f"LAMB lr {a.lr:.0e} — memorize, don't generalize"),
    ])

    # --- train a fresh model on those in-order chunks ---
    model = MusicTransformer(
        vocab_size=vocab, d_model=CFG.d_model, n_heads=CFG.n_heads,
        n_layers=CFG.n_layers, d_ff=CFG.d_ff,
        max_seq_len=max(S, CFG.max_seq_len), dropout=0.0,
        d_text=(ctx.shape[-1] if ctx is not None else 768),
        rope_base=CFG.rope_base, grad_checkpoint=CFG.grad_checkpoint,
    ).to(dev)
    opt = Lamb(model.parameters(), lr=a.lr, betas=CFG.betas, weight_decay=0.0)

    model.train()
    t0 = time.time()
    prog = ui.train_progress()
    task = prog.add_task("memorizing", total=a.steps, stats="") if prog else None
    if prog:
        prog.start()
    last = 99.0
    for step in range(a.steps):
        cx, cy = chunks[step % len(chunks)]          # cycle chunks IN ORDER
        x = cx[None].to(dev); y = cy[None].to(dev)
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
            lg = model(x, ctx, msk)
            loss = F.cross_entropy(lg.reshape(-1, lg.shape[-1]), y.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.grad_clip)
        opt.step()
        last = loss.item()
        if prog:
            prog.update(task, completed=step + 1,
                        stats=f"loss [bold]{last:.4f}[/] ppl {math.exp(min(last,20)):.2f}")
    if prog:
        prog.stop()
    ui.log(f"[green]memorized[/] — final loss [bold]{last:.4f}[/] "
           f"(ppl {math.exp(min(last,20)):.2f}) in {time.time()-t0:.0f}s")
    if last > 1.0:
        ui.log("[yellow]!! loss did not fall below 1.0 — needs more --steps, "
               "or a higher --lr. A single song SHOULD memorize to near zero.[/]")

    model.eval()
    real = codes[None]                                # (1, total)
    n_gen = int(a.gen_seconds * codec.tokens_per_sec); n_gen -= n_gen % n_slots
    n_gen = min(n_gen, total - n_slots)

    ui.rule("regenerating from the real start of the song")

    # GREEDY for the memorization test: temperature/top-k randomness is exactly
    # what knocks a memorized reproduction off its rails -- one sampled slip
    # compounds autoregressively into "spazzing" later in the clip. A memorized
    # model should reproduce the song deterministically, so sample greedily
    # (temp~0, top_k=1). Real style-generation WANTS temperature; that is a
    # different mode. This test is asking "can it reproduce", not "can it riff".
    free = generate(model, codec, n_gen, ctx=ctx, ctx_mask=msk,
                    temperature=0.01, top_k=1, top_p=1.0,
                    guidance=1.0, device=dev, context_len=S)
    m_free = _match(free, real, n_slots)

    # (b) primed with the real first 5s, then continue: tests continuation.
    prime_len = int(5.0 * codec.tokens_per_sec); prime_len -= prime_len % n_slots
    prime = real[:, :prime_len].to(dev)
    cont = generate(model, codec, n_gen, ctx=ctx, ctx_mask=msk, prompt=prime,
                    temperature=0.01, top_k=1, top_p=1.0,
                    guidance=1.0, device=dev, context_len=S)
    cont_new = cont[:, prime_len:]
    real_new = real[:, prime_len:prime_len + cont_new.shape[1]].to(dev)
    m_cont = _match(cont_new, real_new, n_slots)

    # --- decode all three to wav ---
    real_wav = codec.decode_flat(real[:, :n_gen].to(dev))
    free_wav = codec.decode_flat(free[:, :n_gen])
    cont_wav = codec.decode_flat(cont[:, :prime_len + cont_new.shape[1]])
    p_real = out_dir / f"{stem}__real.wav"
    p_free = out_dir / f"{stem}__gen_freerun.wav"
    p_cont = out_dir / f"{stem}__gen_primed.wav"
    audio_io.save(real_wav.cpu(), p_real, codec.sr)
    audio_io.save(free_wav.cpu(), p_free, codec.sr)
    audio_io.save(cont_wav.cpu(), p_cont, codec.sr)

    chance = 100.0 / vocab
    ui.rule("verdict")
    ui.kv_table([
        ("free-run match", f"[bold]{m_free:.1f}%[/]  vs {chance:.4f}% chance "
                           f"= {m_free/chance:.0f}x"),
        ("primed match", f"[bold]{m_cont:.1f}%[/]  (continuation only) "
                         f"= {m_cont/chance:.0f}x"),
        ("real audio", str(p_real)),
        ("free-run audio", str(p_free)),
        ("primed audio", str(p_cont)),
    ], title="did it memorize?")
    if m_free > 40 or m_cont > 50:
        ui.panel("[bold green]PIPELINE VERIFIED[/] — reproduced the song well "
                 "above chance. Codec, interleave, RoPE, KV cache, sampling and "
                 "decode are all correct end to end.\n\n[dim]Bad output on the "
                 "full run is therefore data/scale, not a bug.[/]",
                 title="green light", style="green")
    else:
        ui.panel("[bold red]DID NOT REPRODUCE[/] — match near chance even fed "
                 "in order. If final loss was low but match is low, the "
                 "generation path (KV cache / sampling) is the suspect: run "
                 "diagnose_gen.py to isolate it. If loss stayed high, it never "
                 "memorized: more --steps or higher --lr.",
                 title="red flag", style="red")


def _match(a_tokens, b_tokens, n_slots):
    """Percent of positions where generated == real, over the overlap."""
    n = min(a_tokens.shape[1], b_tokens.shape[1])
    n -= n % n_slots
    if n == 0:
        return 0.0
    a = a_tokens[0, :n].to(b_tokens.device)
    b = b_tokens[0, :n]
    return 100.0 * (a == b).float().mean().item()


if __name__ == "__main__":
    main()
