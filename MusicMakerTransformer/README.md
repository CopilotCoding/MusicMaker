# MusicMaker

A music LM built on a hand-written transformer. Trains on mp3/wav/flac,
generates stereo wav/mp3/flac, conditions on lyrics from sibling `.txt` files.

## How it works

Raw audio isn't discrete — stereo 32kHz is 64,000 floats/sec. EnCodec turns it
into 400 tokens/sec of discrete codes (4 codebooks × 2 channels, interleaved
into 8 slots per timestep, vocab 16,386); the transformer models those;
EnCodec turns them back into sound.

```
audio.wav ──EnCodec──> tokens ──[TRANSFORMER]──> tokens ──EnCodec──> audio.mp3
lyrics.txt ──T5 (frozen)──> embeddings ──cross-attention──┘
```

138M params: d=768, 12 layers, 12 heads, RoPE, RMSNorm, fused QKV, flash
attention via SDPA. Context 16,384 tokens = **41 seconds** of stereo audio,
reachable on a 16GB card only via gradient checkpointing.

## Quickstart

```bash
python train.py --music-dir "D:/Music/Grunge"        # train on YOUR folder
python plot.py                                        # live loss/gn/lr window
python generate.py --lyrics "your lyrics" --seconds 20 --out out.mp3
```

Training auto-saves on Ctrl-C or crash (`checkpoints/interrupted.pt`) and
resumes exactly where it stopped:

```bash
python train.py --music-dir "..." --resume checkpoints/interrupted.pt
```

`best.pt` always holds the best-validation model — generate from that one.

## Using your own music

Rip to FLAC or mp3, then let `tag.py` read whatever metadata your ripper
embedded, or write lyrics into sidecar `.txt` files by hand:

```bash
python tag.py --music-dir "D:/Music/Grunge" --audit    # what can it teach?
python tag.py --music-dir "D:/Music/Grunge" --write    # metadata -> .txt sidecars
python train.py --music-dir "D:/Music/Grunge"
```

**`--music-dir` points at ANY folder and copies nothing.** It scans
recursively, so `Artist/Album/track.mp3` works as-is. Only the token cache
(~150MB for ~250 songs) lives in this project. Pairing and cache names are
keyed on the FULL PATH, not the filename — every album has an `01 - Intro`.
Caching is incremental and an interrupted encode resumes.

**Everything in your files gets written.** `tag.py` reports on your tags; it
never discards them (`--drop-constant` exists, off by default). Lyrics are
free-form text — mood/instrument/tempo descriptors work too, and at
generation you can recombine them even if that pairing never appeared in
training. `text_max_len` is 768 T5 tokens, sized so the longest hand-written
lyric in a 257-track corpus survives untruncated.

## How training reads the data

Songs are fed **in order, start to finish**, as fixed non-overlapping
seq_len windows. Window 0 of every song is `[BOS, t0, …]` so the model
*learns to begin a song from nothing*; every interior window starts one token
early so target coverage is contiguous — every token in the song is predicted
exactly once. Validation holds out **whole songs**, so no track straddles the
train/val split.

Generation mirrors that layout exactly: chunk-aligned, resetting the KV cache
every seq_len tokens the way training chunks do. This was measured, not
assumed — a sliding window reproduced a memorized song for *exactly* its
first training chunk and then derailed (68.2% of a 6000-token run =
4096/6000), because the model had never seen a context that starts mid-chunk.

## Optimizer: LAMB, at effective batch 4

AdamW failed at every learning rate on this hardware: 3e-4 diverged at step
350, 5e-5 diverged at ~600 (under two β₂ values), 1e-5 stalled. At effective
batch 4 its variance estimate drifts over hundreds of steps — the run looks
healthy, then the gradient norm ramps 0.7 → 10.9 and everything unlearns.

LAMB's per-layer trust ratio re-pins every step to `lr × ‖weight‖`, so the
drift can't compound. Three LRs spanning 33× (3e-5 → 1e-3) all trained stably
on the first attempt. A grad-norm-trend backoff (auto-halve on sustained ramp)
is wired in as a safety net; on the full 1e-3 run it fired four times as the
loss surface sharpened, annealing the LR 16× while val set a record at every
eval — an empirically driven decay schedule. The full story, with receipts:
[LAMB_AT_BATCH_FOUR.md](LAMB_AT_BATCH_FOUR.md).

## Verified on this machine (RTX 5060 Ti, 16GB)

| check | result |
|---|---|
| codec round-trip | MAE 0.0224, **0.0ms** duration drift, stereo intact |
| cached decode == full forward | max delta 5.66e-07 |
| RoPE depends only on relative distance | ~6e-04 at pos 100 vs 7000 |
| memorize one song (27 chunks, 2000 steps) | loss 0.005 |
| **regenerate that song from BOS, free-run** | **100.0% token match** (chance: 0.006%) |
| primed continuation across chunk boundaries | 98.7% |
| VRAM @ seq_len 16384, batch 2, grad ckpt | ~10.2 GB |
| seq_len headroom measured (batch 1) | 49,152 tokens (123s) at 12.5 GB |
| full-corpus run (257 songs, LAMB 1e-3) | val **9.86 → 7.74** in 800 steps / 2.5h |

The 100% free-run row is the load-bearing one: it exercises codec, interleave,
BOS, RoPE, KV cache, chunk-aligned generation, slot-masked sampling, and
decode simultaneously. Any one of them subtly wrong puts it near zero — and
three of them *were* subtly wrong until this test caught them (see below).

## Files

| file | role |
|---|---|
| `transformer.py` | the model — flash attn, RoPE, RMSNorm, cross-attn, KV cache |
| `codec.py` | EnCodec wrap; the flatten/interleave contract (highest-risk file) |
| `optim.py` | LAMB — the trust-ratio optimizer that ended the divergences |
| `textenc.py` | frozen T5 lyric encoder |
| `data.py` | scan, offline token cache, in-order BOS windows, per-song split |
| `train.py` | bf16, LAMB + backoff, per-slot loss, VRAM assert, auto-save |
| `generate.py` | chunk-aligned sampling, CFG, slot-masked decode to audio |
| `plot.py` | live matplotlib window over `checkpoints/metrics.csv` |
| `tag.py` | metadata → `.txt` sidecars via ffprobe |
| `overfit_one.py` | memorize ONE song, regenerate it — the end-to-end proof |
| `diagnose_gen.py` | four-stage bisect: codec / model / KV cache / generation |
| `audio_io.py` | soundfile + ffmpeg. No torchaudio (2.11 reworked its backends) |
| `ceiling.py`, `lrcheck.py`, `lrfind.py` | hear the codec ceiling; probe the optimizer |
| `bootstrap.py` | fetch a tiny public dataset |
| `ui.py` | Rich console |

## Design notes

**encodec_32khz, not 48khz.** The 48kHz model is natively stereo, which looked
ideal, but its 1s internal chunking makes encode→decode drift ~2000ms. 32kHz
(what MusicGen uses) is exact, needs no scales, and reconstructs better.
Stereo = encode L/R as two mono streams and interleave, as MusicGen does.

**RoPE, not learned positions** — but with the position counter passed
explicitly during generation. Deriving it from KV-cache length breaks the
moment the cache is truncated: every later token rotates at the same offset
and loses its positional identity. Measured, fixed, regression-tested.

**BOS in training, because generation needs it.** A model trained with a real
token at position 0 cannot start a song from nothing — feeding it an untrained
BOS shifts every RoPE position by one and produces confident garbage. Trains
fine, generates noise, no error raised. Found by `diagnose_gen.py`'s
teacher-forced-vs-generated bisect.

**No weight tying.** Saves 12.6M on a 138M model, and is actively wrong here:
only 1/8 of the vocab is legal at any position, and input/output roles differ.

**`is_causal` is gated on shapes.** SDPA anchors its causal mask top-left,
correct only when `q_len == kv_len`. With a KV cache it silently makes each
new token attend only to cached token 0 — measured wrong by 4.2, no error
raised. `_attend()` gates on shape, never intent.

**Windows lies about OOM.** It silently spills VRAM into shared host memory
instead of raising — a 21GB "success" on a 16GB card trains at unusable speed.
`train.py` asserts against a 14GB budget instead of trusting the absence of an
exception.

## Expectations

Textured audio in your training set's character first; song-like structure is
a data question — real music LMs train on thousands of hours, this starts
from ~19. Confirmed by ear at the grunge-only floor (val 7.74): convincing
texture, not yet music. That number is the baseline the
pretrain-on-public-data-then-finetune path (MTG-Jamendo, unconditional, then
grunge + lyrics) has to beat — and the pipeline is verified ready for it.
