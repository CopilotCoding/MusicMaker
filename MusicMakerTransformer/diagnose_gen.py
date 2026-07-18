r"""
diagnose_gen.py — the memorized model makes noise. WHERE is it broken?

"Trains fine, generates garbage" has a handful of classic causes. This isolates
them in order, so we stop guessing from ears. It memorizes ONE window of ONE
song (fast, deterministic) and runs three tests, each ruling out one stage:

  TEST 1 — decode round-trip: cached tokens -> audio -> is it the song?
           Rules out the codec/decode path. (If this is noise, encoding is the
           bug, not generation.)

  TEST 2 — teacher-forced accuracy: feed the REAL tokens, does argmax predict
           the REAL next token? A memorized model should be ~100%. This tests
           the MODEL in isolation, with NO sampling and NO KV cache.

  TEST 3 — cached vs uncached logits: run the same tokens through one full
           forward, then token-by-token WITH the KV cache. If they disagree,
           the KV-cache/RoPE-offset path is broken (the is_causal trap) — which
           looks perfect in training and destroys generation.

Whichever test first fails is the bug. If all three pass, the problem is not a
bug at all — it is sampling settings or the seq_len/positioning mismatch (a
model trained on random 10s crops has no coherent position-0 to free-run from).
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

import ui
from config import CFG


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--song", required=True)
    ap.add_argument("--seq-len", type=int, default=4096)
    ap.add_argument("--steps", type=int, default=300,
                    help="one window memorizes to ~0 loss in a few hundred "
                         "steps; no need to wait longer")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    a = ap.parse_args()

    dev = ("cuda" if torch.cuda.is_available() else "cpu") \
        if a.device == "auto" else a.device

    import transformers
    transformers.logging.set_verbosity_error()
    from codec import Codec
    from transformer import MusicTransformer
    from optim import Lamb
    import audio_io
    import numpy as np

    song = Path(a.song)
    codec = Codec(device=dev)
    n_slots, vocab = codec.n_slots, codec.vocab_size

    # encode the song (reuse the isolated overfit cache if present)
    cache = Path(__file__).parent / "_overfit_cache"
    from data import cache_name
    npz = cache / f"{cache_name(song, song.parent)}.npz"
    if npz.exists():
        codes = torch.from_numpy(np.load(npz)["codes"].astype(np.int64))
    else:
        flat = codec.encode_file(song)
        codes = flat.long()
    ui.rule("diagnose — where does generation break?")

    # Memorize the window at the START of the song, WITH BOS prepended, exactly
    # as training now does. Then TEST 4's free-run-from-BOS has a coherent thing
    # to reproduce and validates the BOS fix end to end.
    #   input  = [BOS, t0, ..., t_{S-2}]   target = [t0, ..., t_{S-1}]
    S = a.seq_len
    S -= S % n_slots
    real = codes[:S]                        # S real tokens from the true start
    if real.shape[0] < S:
        raise SystemExit(f"song shorter than seq_len {S}")
    x = torch.cat([torch.tensor([codec.bos], dtype=real.dtype), real[:-1]])[None].to(dev)
    y = real[None].to(dev)                  # (1, S)
    win = torch.cat([real, codes[S:S+1]]) if codes.shape[0] > S else real  # for TEST 1 decode

    # ---- TEST 1: decode round-trip (is the codec/decode path fine?) ----
    real_wav = codec.decode_flat(real[None].to(dev))
    p_real = Path(__file__).parent / "_diag_real.wav"
    audio_io.save(real_wav.cpu(), p_real, codec.sr)
    ui.kv_table([
        ("test 1", "decode round-trip"),
        ("wrote", f"{p_real}  ({real_wav.shape[-1]/codec.sr:.1f}s)"),
        ("listen", "if THIS is noise, encoding/decoding is the bug, not "
                   "generation. If it's the song, decode is fine -> test 2."),
    ], title="TEST 1 — codec")

    # ---- memorize this one window ----
    model = MusicTransformer(
        vocab_size=vocab, d_model=CFG.d_model, n_heads=CFG.n_heads,
        n_layers=CFG.n_layers, d_ff=CFG.d_ff,
        max_seq_len=max(S, CFG.max_seq_len), dropout=0.0,
        d_text=768, rope_base=CFG.rope_base, grad_checkpoint=False,
    ).to(dev)
    opt = Lamb(model.parameters(), lr=a.lr, betas=CFG.betas, weight_decay=0.0)
    model.train()
    prog = ui.train_progress()
    task = prog.add_task("memorize 1 window", total=a.steps, stats="") if prog else None
    if prog:
        prog.start()
    for step in range(a.steps):
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
            lg = model(x, None, None)
            loss = F.cross_entropy(lg.reshape(-1, lg.shape[-1]), y.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.grad_clip)
        opt.step()
        if prog:
            prog.update(task, completed=step + 1, stats=f"loss [bold]{loss.item():.4f}[/]")
    if prog:
        prog.stop()
    ui.log(f"[green]memorized 1 window[/] — loss {loss.item():.4f}")

    model.eval()

    # ---- TEST 2: teacher-forced argmax accuracy (is the MODEL right?) ----
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
        lg = model(x, None, None)                      # (1, S, V)
    pred = lg.float().argmax(-1)                        # greedy next-token
    acc = (pred[0] == y[0]).float().mean().item() * 100
    # also apply the per-position slot mask, since generation always does
    # lg[0,i] predicts y[0,i]. With BOS training, x = [BOS, t0, ..., t_{S-2}] and
    # y = [t0, ..., t_{S-1}], so the target y[0,i] is stream token i, whose slot
    # is i % n_slots. (Without BOS the target would be i+1.) Mask with the
    # TARGET's slot -- exactly what generation does at each position it fills --
    # or the test is off by one slot and reports 0% on a perfect model.
    masked_correct = 0
    for i in range(0, S, max(1, S // 2000)):           # sample positions for speed
        m = codec.logit_mask(i, dev)                  # BOS layout: target slot = i%n_slots
        pm = (lg[0, i].float() + m).argmax().item()
        masked_correct += int(pm == y[0, i].item())
    n_sampled = len(range(0, S, max(1, S // 2000)))
    acc_masked = 100 * masked_correct / n_sampled
    ui.kv_table([
        ("test 2", "teacher-forced argmax (no sampling, no cache)"),
        ("raw accuracy", f"[bold]{acc:.1f}%[/]  (memorized -> should be ~100%)"),
        ("with slot mask", f"[bold]{acc_masked:.1f}%[/]"),
        ("verdict", "[green]MODEL OK[/] -> test 3" if acc > 90 else
                    "[bold red]MODEL DID NOT MEMORIZE[/] — it can't even predict "
                    "the tokens it was trained on. Bug is in the model/training, "
                    "NOT sampling. Stop here."),
    ], title="TEST 2 — model")

    # ---- TEST 3: cached == uncached? (is the KV-cache path correct?) ----
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
        full = model(x, None, None).float()            # (1, S, V) one shot
        # token-by-token with the cache, feeding the SAME real tokens
        caches = None
        cached_logits = []
        cur = x[:, :1]
        for i in range(S):
            lg_i, caches = model(cur, None, None, kv_caches=caches, return_caches=True)
            cached_logits.append(lg_i[:, -1].float())
            if i + 1 < S:
                cur = x[:, i + 1:i + 2]                 # feed the real next token
        cached = torch.stack(cached_logits, dim=1)     # (1, S, V)
    # compare where it matters: the argmax token, position by position
    full_pred = full.argmax(-1)[0]
    cached_pred = cached.argmax(-1)[0]
    agree = (full_pred == cached_pred).float().mean().item() * 100
    max_logit_diff = (full - cached).abs().max().item()
    ui.kv_table([
        ("test 3", "cached decode vs one full forward, same tokens"),
        ("argmax agreement", f"[bold]{agree:.1f}%[/]  (should be ~100%)"),
        ("max logit diff", f"{max_logit_diff:.3f}  (should be < ~0.1)"),
        ("verdict", "[green]KV CACHE OK[/]" if agree > 99 else
                    "[bold red]KV-CACHE BROKEN[/] — cached decode disagrees with "
                    "the full forward. This is the is_causal/RoPE-offset trap. "
                    "Training looks perfect; generation is garbage. THIS is the bug."),
    ], title="TEST 3 — KV cache")

    # ---- TEST 4: the REAL generate() path, teacher-forced. Does generation
    #      land the right SLOT at each position? This is the one that catches
    #      the BOS/position-phase bug: training put a real slot-0 token at
    #      position 0, but generate() prepends BOS. If the phase is off, every
    #      sampled token is masked to the wrong codebook -> noise. ----
    from generate import generate
    # Greedy (temp tiny, top_k 1) so it is deterministic and directly comparable
    # to the real tokens. No CFG. Free-run from BOS, exactly like real use.
    gen = generate(model, codec, S, ctx=None, ctx_mask=None, prompt=None,
                   temperature=0.01, top_k=1, top_p=1.0, guidance=1.0,
                   device=dev, verbose=False, context_len=S)
    g = gen[0]
    real0 = real.to(g.device)                          # the real tokens (no BOS)
    n = min(g.shape[0], real0.shape[0]); n -= n % n_slots
    gen_match = (g[:n] == real0[:n]).float().mean().item() * 100
    # slot-phase check: is generated token at position p actually in slot p's band?
    slot_ok = 0
    for p in range(0, n):
        band = (p % n_slots) * codec.codebook_sz
        if band <= int(g[p]) < band + codec.codebook_sz:
            slot_ok += 1
    slot_pct = 100 * slot_ok / max(1, n)
    ui.kv_table([
        ("test 4", "real generate() free-run vs the memorized window (greedy)"),
        ("token match", f"[bold]{gen_match:.1f}%[/]  (memorized+correct gen "
                        f"-> should be very high)"),
        ("slot-phase ok", f"[bold]{slot_pct:.1f}%[/]  (generated token lands in "
                          f"the RIGHT codebook for its position)"),
        ("verdict", "[green]GENERATION OK[/]" if gen_match > 80 else
                    "[bold red]GENERATION MISALIGNED[/] — the model memorized "
                    "perfectly (test 2) but generate() reproduces ~nothing. If "
                    "slot-phase is also low, it is the BOS/position-phase bug: "
                    "training has a real token at position 0, generate() puts "
                    "BOS there, shifting every slot."),
    ], title="TEST 4 — real generation path")

    ui.rule("summary")
    if acc_masked > 90 and agree > 99 and gen_match > 80:
        ui.panel("[bold green]ALL PASS[/] — codec, model, KV cache AND the real "
                 "generation path all correct. The garbage on the full run is "
                 "genuinely data/scale, not a bug.", title="clean", style="green")
    elif acc_masked <= 90:
        ui.panel("[bold red]MASK/SLOT MISALIGNMENT (test 2).[/] The model's "
                 "correct prediction is being masked to the wrong codebook. "
                 "logit_mask slot phase disagrees with the token stream.",
                 title="bug: masking", style="red")
    elif gen_match <= 80:
        ui.panel("[bold red]GENERATION-PATH BUG (test 4).[/] Model memorized "
                 "(test 2 ok), cache ok (test 3), but generate() reproduces "
                 "nothing. This is the BOS/position-phase mismatch: fix "
                 "generate.py to match how training positions the stream.",
                 title="bug: generation", style="red")
    else:
        ui.panel(f"[yellow]Inconclusive — masked acc {acc_masked:.0f}%, "
                 f"agree {agree:.0f}%, gen {gen_match:.0f}%. "
                 "Read the tables above.[/]", title="?", style="yellow")


if __name__ == "__main__":
    main()
