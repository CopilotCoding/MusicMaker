"""
generate.py — sample tokens, decode to wav/mp3/flac.

    python generate.py --lyrics "genre: blues" --seconds 20 --out out.mp3
    python generate.py --lyrics-file song.txt --seconds 30 --out out.wav
    python generate.py --prompt-audio seed.wav --seconds 15 --out cont.flac

SLIDING WINDOW is trivial here because positions are RoPE, not learned: keep
the last max_seq_len KV entries, evict from the front, continue. No
re-anchoring, no position re-encoding, no periodic re-prefill. Eviction length
must stay a multiple of n_slots or every later token lands in the wrong slot.

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
             device="cuda", verbose=True):
    """Returns flat token ids (1, S) including any prompt."""
    temperature = CFG.temperature if temperature is None else temperature
    top_k = CFG.top_k if top_k is None else top_k
    top_p = CFG.top_p if top_p is None else top_p
    guidance = CFG.guidance if guidance is None else guidance
    n_slots = codec.n_slots
    use_cfg = guidance and guidance != 1.0 and ctx is not None

    model.eval()
    if prompt is None or prompt.numel() == 0:
        # BOS is outside every slot's slice, so the first real token is slot 0.
        seq = torch.full((1, 1), codec.bos, dtype=torch.long, device=device)
        start = 0
    else:
        seq = prompt.to(device)
        assert seq.shape[1] % n_slots == 0, "prompt must be slot-aligned"
        start = seq.shape[1]

    n_tokens -= n_tokens % n_slots                    # emit whole timesteps only
    caches = ccaches = None
    cur = seq
    prog = ui.gen_progress() if verbose else None
    task = prog.add_task("sampling", total=n_tokens, stats="") if prog else None
    if prog:
        prog.start()

    for i in range(n_tokens):
        pos = start + i
        # Roll the window: RoPE is relative, so we can just drop the oldest
        # tokens. Evict in whole timesteps to preserve slot phase.
        if caches is not None and caches[0][0].shape[2] >= model.max_seq_len:
            keep = model.max_seq_len - n_slots
            keep -= keep % n_slots
            caches = [(k[:, :, -keep:], v[:, :, -keep:]) for k, v in caches]
            if ccaches is not None:
                ccaches = [(k[:, :, -keep:], v[:, :, -keep:]) for k, v in ccaches]

        lg, caches = model(cur, ctx, ctx_mask, kv_caches=caches, return_caches=True)
        lg = lg[:, -1]
        if use_cfg:
            lgu, ccaches = model(cur, None, None, kv_caches=ccaches,
                                 return_caches=True)
            lg = lgu[:, -1] + guidance * (lg - lgu[:, -1])

        nxt = sample_from(lg.float(), codec.logit_mask(pos, device),
                          temperature, top_k, top_p)
        seq = torch.cat([seq, nxt], dim=1)
        cur = nxt
        if prog:
            prog.update(task, completed=i + 1,
                        stats=f"[bold]{(i+1)/codec.tokens_per_sec:.1f}s[/] audio")
    if prog:
        prog.stop()

    # Drop BOS so the stream starts on a clean slot-0 boundary.
    out = seq[:, 1:] if prompt is None else seq
    return out[:, : out.shape[1] - out.shape[1] % n_slots]


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
    a = ap.parse_args()

    import transformers
    transformers.logging.set_verbosity_error()
    from codec import Codec

    dev = "cuda" if torch.cuda.is_available() else "cpu"
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
                    a.temperature, a.top_k, a.top_p, a.guidance, dev)

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
