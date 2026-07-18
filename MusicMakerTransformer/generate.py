"""
generate.py — sample tokens, decode to wav/mp3/flac.

    python generate.py --lyrics "genre: blues" --seconds 20 --out out.mp3
    python generate.py --lyrics-file song.txt --seconds 30 --out out.wav
    python generate.py --prompt-audio seed.wav --seconds 15 --out cont.flac

LONG OUTPUT is chunk-aligned, not sliding-window: training feeds every chunk
at RoPE positions 0..S-1 with its context starting AT the chunk boundary, so
generation resets the KV cache every S tokens and continues -- the exact
context pattern the model learned. (A sliding window was measured to derail a
memorized song at precisely the first chunk boundary: the model never saw a
context that starts mid-chunk.)

SAMPLING ORDER: slot mask -> temperature -> top-k -> top-p. The mask goes FIRST
because position i can only legally hold a token from slot i % n_slots; letting
illegal mass into the nucleus would distort top-p's cutoff.

CFG: logits_uncond + g * (logits_cond - logits_uncond). Costs one extra forward
per step and is what makes conditioning audibly matter. Enabled by the runtime
ctx=None gate, which is why one checkpoint serves both paths.
"""

import argparse

import torch
import torch.nn.functional as F

import audio_io
import ui
from config import CFG, CKPT_DIR


def sample_from(logits, slot_mask, temperature, top_k, top_p):
    """logits (B,V) -> ids (B,1). Mask first, then temperature, k, p."""
    logits = logits + slot_mask                       # illegal slots -> -inf
    if temperature != 1.0:
        logits = logits / max(temperature, 1e-5)
    if top_k:
        k = min(top_k, logits.shape[-1])
        kth = logits.topk(k, dim=-1).values[..., -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p and top_p < 1.0:
        s, idx = logits.sort(dim=-1, descending=True)
        cum = s.softmax(-1).cumsum(-1)
        cut = cum - s.softmax(-1) > top_p             # keep the crossing token
        s = s.masked_fill(cut, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(-1, idx, s)
    return torch.multinomial(logits.softmax(-1), 1)


@torch.no_grad()
def generate(model, codec, n_tokens, ctx=None, ctx_mask=None, prompt=None,
             temperature=None, top_k=None, top_p=None, guidance=None,
             device="cuda", verbose=True, context_len=None):
    """Returns flat token ids (1, S) including any prompt (never the BOS).

    CHUNK-ALIGNED, mirroring training exactly. Training feeds each song as
    fixed windows, every one placed at RoPE positions 0..S-1:

        chunk 0:   [BOS,       t0        .. t_{S-2}  ] -> [t0     .. t_{S-1} ]
        chunk k:   [t_{kS-1},  t_{kS}    .. t_{kS+S-2}] -> [t_{kS} .. t_{kS+S-1}]

    So every training prediction's context BEGINS at a chunk boundary. A
    sliding window hands the model contexts that begin mid-chunk -- something
    it never once saw -- and it derails. MEASURED: a memorized song reproduced
    exactly its first chunk (68.2% of a 6000-token run = 4096/6000) and turned
    to noise at the first boundary. The fix is to generate the way training
    reads: every S cache entries, RESET the cache and continue with the last
    token at position 0, exactly like the next training chunk does.

    The prompt is prefilled in the same chunk layout (BOS + first S-1 tokens,
    then fresh S-token pieces), so a prime of any length lands the model in
    precisely the state training would have it in at that point in the song.
    """
    temperature = CFG.temperature if temperature is None else temperature
    top_k = CFG.top_k if top_k is None else top_k
    top_p = CFG.top_p if top_p is None else top_p
    guidance = CFG.guidance if guidance is None else guidance
    n_slots = codec.n_slots
    use_cfg = guidance and guidance != 1.0 and ctx is not None

    model.eval()
    S = context_len or model.max_seq_len          # the TRAINED chunk length
    S = min(S, model.max_seq_len)
    S -= S % n_slots

    bos = torch.full((1, 1), codec.bos, dtype=torch.long, device=device)
    if prompt is None or prompt.numel() == 0:
        known = bos[:, :0]                        # empty (1, 0)
    else:
        known = prompt.to(device)
        assert known.shape[1] % n_slots == 0, "prompt must be slot-aligned"
    P = known.shape[1]
    n_tokens -= n_tokens % n_slots                # emit whole timesteps only

    def fwd(tokens, caches, ccaches, rope0):
        lg, caches = model(tokens, ctx, ctx_mask, kv_caches=caches,
                           return_caches=True, pos_offset=rope0)
        lgu = None
        if use_cfg:
            lgu, ccaches = model(tokens, None, None, kv_caches=ccaches,
                                 return_caches=True, pos_offset=rope0)
        return lg, lgu, caches, ccaches

    # ---- prefill the prompt in training-layout pieces (fresh cache each) ----
    pieces = [torch.cat([bos, known[:, :min(P, S - 1)]], dim=1)]
    p = S - 1
    while p < P:
        pieces.append(known[:, p:p + S])          # starts at t_{kS-1}
        p += S
    caches = ccaches = None
    for piece in pieces:
        lg, lgu, caches, ccaches = fwd(piece, None, None, 0)
    cache_len = pieces[-1].shape[1]

    prog = ui.gen_progress() if verbose else None
    task = prog.add_task("sampling", total=n_tokens, stats="") if prog else None
    if prog:
        prog.start()

    seq = known                                   # real tokens only, no BOS
    for i in range(n_tokens):
        sp = P + i                                # stream position being sampled
        l = lg[:, -1]
        if use_cfg:
            l = lgu[:, -1] + guidance * (l - lgu[:, -1])
        nxt = sample_from(l.float(), codec.logit_mask(sp, device),
                          temperature, top_k, top_p)
        seq = torch.cat([seq, nxt], dim=1)

        if i + 1 < n_tokens:
            if cache_len >= S:                    # chunk complete -> new chunk,
                caches = ccaches = None           # exactly like training
                cache_len = 0
            lg, lgu, caches, ccaches = fwd(nxt, caches, ccaches, cache_len)
            cache_len += 1

        if prog:
            prog.update(task, completed=i + 1,
                        stats=f"[bold]{(i+1)/codec.tokens_per_sec:.1f}s[/] audio")
    if prog:
        prog.stop()

    return seq[:, : seq.shape[1] - seq.shape[1] % n_slots]


def load_model(ckpt_path, device="cuda"):
    from transformer import MusicTransformer
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    s = ck["codec_sig"]
    model = MusicTransformer(
        vocab_size=s["vocab_size"], d_model=s["d_model"], n_heads=s["n_heads"],
        n_layers=s["n_layers"], d_ff=s["d_ff"], max_seq_len=s["max_seq_len"],
        dropout=0.0, d_text=s["d_text"], rope_base=CFG.rope_base,
    ).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    # The seq_len the run actually TRAINED at. Generation must roll its KV
    # window at this length, not at max_seq_len (the RoPE table size) -- a model
    # trained at 4096 produces clean audio for exactly 4096 tokens and then
    # degrades, because it never saw a position past 4095.
    s = dict(s)
    s["trained_seq_len"] = (ck.get("cfg") or {}).get("seq_len") or s["max_seq_len"]
    return model, s, ck.get("step", -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(CKPT_DIR / "final.pt"))
    ap.add_argument("--lyrics", type=str, default=None)
    ap.add_argument("--lyrics-file", type=str, default=None)
    ap.add_argument("--prompt-audio", type=str, default=None)
    ap.add_argument("--prompt-seconds", type=float, default=3.0)
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--out", type=str, default="out.wav")
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--top-p", type=float, default=None)
    ap.add_argument("--guidance", type=float, default=None)
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto",
                    help="'cpu' to sample safely while training holds the GPU")
    a = ap.parse_args()

    import transformers
    transformers.logging.set_verbosity_error()
    from codec import Codec

    dev = ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" \
        else a.device
    if dev == "cuda" and torch.cuda.is_available():
        # Sampling alongside a training run means TWO 138M models plus two KV
        # caches on one card. At seq_len=16384 training already holds ~15/16GB,
        # and the second process does not get a clean OOM -- it gets a corrupted
        # CUDA context and dies with "illegal memory access" somewhere
        # unrelated (observed: inside apply_rope). Warn before that happens.
        free, total = torch.cuda.mem_get_info()
        if free / 1e9 < 4.0:
            ui.log(f"[yellow]only {free/1e9:.1f}GB of {total/1e9:.0f}GB free on the "
                   f"GPU — something else (training?) is using it.[/]\n"
                   f"[dim]Two models on one card corrupts the CUDA context rather "
                   f"than OOMing cleanly. Use --device cpu to sample safely.[/]")
    model, sig, step = load_model(a.ckpt, dev)
    codec = Codec(device=dev)

    # A checkpoint decoded with a different codec is silent garbage.
    assert sig["codec_id"] == codec.model_id and sig["n_slots"] == codec.n_slots \
        and sig["vocab_size"] == codec.vocab_size, \
        f"checkpoint codec {sig['codec_id']}/{sig['n_slots']} != {codec.model_id}/{codec.n_slots}"
    ui.rule("MusicMaker — generate")
    ui.kv_table([
        ("checkpoint", f"{a.ckpt}  [dim]step {step}[/]"),
        ("model", f"{model.count_parameters()/1e6:.1f}M params"),
        ("codec", f"{codec.model_id}  {codec.tokens_per_sec} tok/s  {codec.channels}ch"),
    ])

    text = a.lyrics
    if a.lyrics_file:
        text = open(a.lyrics_file, encoding="utf-8").read().strip()
    ctx = msk = None
    if text:
        from textenc import LyricEncoder
        ctx, msk = LyricEncoder(model_id=sig["t5_id"], device=dev).encode([text])
        ui.panel(text[:300] + ("..." if len(text) > 300 else ""), title="lyrics", style="magenta")

    prompt = None
    if a.prompt_audio:
        wav = audio_io.load(a.prompt_audio, codec.sr, codec.channels)
        wav = wav[:, : int(a.prompt_seconds * codec.sr)]
        prompt = codec.encode_wav(wav)[None].to(dev)
        ui.log(f"[dim]prompt[/] {a.prompt_audio} {a.prompt_seconds}s -> {prompt.shape[1]} tokens")

    n = int(a.seconds * codec.tokens_per_sec)
    ui.log(f"[dim]sampling[/] {n} tokens = {a.seconds}s  "
           f"[dim]temp[/] {a.temperature or CFG.temperature} "
           f"[dim]top_k[/] {a.top_k or CFG.top_k} [dim]top_p[/] {a.top_p or CFG.top_p} "
           f"[dim]cfg[/] {a.guidance or CFG.guidance}")
    flat = generate(model, codec, n, ctx, msk, prompt,
                    a.temperature, a.top_k, a.top_p, a.guidance, dev,
                    context_len=sig.get("trained_seq_len"))

    wav = codec.decode_flat(flat[0])
    # peak_norm here (unlike training data): the decoder has no reason to land
    # at a sane level, so normalize for listenable output.
    audio_io.save(wav, a.out, codec.sr, peak_norm=CFG.peak_norm)
    ui.panel(f"[bold]{a.out}[/]\n"
             f"{wav.shape[1]/codec.sr:.2f}s  {wav.shape[0]}ch @ {codec.sr}Hz  "
             f"peak {wav.abs().max():.2f}",
             title="wrote", style="green")


if __name__ == "__main__":
    main()
