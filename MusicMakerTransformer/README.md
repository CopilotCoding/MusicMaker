# MusicMaker

A music LM built on a hand-written transformer. Trains on mp3/wav/flac,
generates stereo wav/mp3/flac, conditions on lyrics from sibling `.txt` files.

## How it works

Raw audio isn't discrete — stereo 32kHz is 64,000 floats/sec. EnCodec turns it
into 400 tokens/sec of discrete codes; the transformer models those; EnCodec
turns them back into sound.

```
audio.wav ──EnCodec──> tokens ──[YOUR TRANSFORMER]──> tokens ──EnCodec──> audio.mp3
lyrics.txt ──T5 (frozen)──> embeddings ──cross-attention──┘
```

## Quickstart

```bash
python bootstrap.py --tracks 8        # fetch GTZAN -> data/raw/*.wav + *.txt
python train.py --overfit 3 --steps 260 --seq-len 2048 --batch-size 2 --grad-accum 2
python generate.py --lyrics "genre: blues" --seconds 20 --out out.mp3
```

## Using your own music

Rip to FLAC (lossless, and the pipeline reads it natively), drop the files in
`data/raw/`, then let `tag.py` read the metadata your ripper embedded:

```bash
python tag.py --music-dir "D:/Music/Grunge" --audit    # what can it teach?
python tag.py --music-dir "D:/Music/Grunge" --write    # metadata -> .txt sidecars
python train.py --music-dir "D:/Music/Grunge"
```

**`--music-dir` points at ANY folder and copies nothing.** It scans
recursively, so `Artist/Album/track.mp3` works as-is. Only the token cache
(~150MB for ~250 songs) lives in this project; your library stays where it is.
Point BOTH projects at the same folder — that is what makes the comparison
fair.

Pairing and cache names are keyed on the FULL PATH, not the filename: every
album has an `01 - Intro`, and stem-keyed matching would silently collapse
them into one track. Caching is incremental, so an interrupted encode resumes.

`tag.py` reads artist/album/year/genre via ffprobe — works identically for
flac, mp3, and wav.

**Everything in your files gets written.** `tag.py` reports on your tags; it
never discards them. `--audit` is information to act on however you like:

```
  artist   8/8 tracks  3 values, ~2.7 tracks each
   genre   8/8 tracks  constant — no contrast to learn from yet
   title   8/8 tracks  unique per track — tends toward memorization
```

Cross-attention learns correlations that *recur*, so tags that **vary but
repeat** (usually `artist`) carry the most conditioning signal. A tag that's
constant across the corpus has no contrast to learn from *today* — but it stops
being constant the moment you add one differing record, and then it's exactly
the distinction that makes conditioning work. That's why nothing is dropped.
(`--drop-constant` exists if you want it; off by default.)

You can hand-edit any `.txt` afterwards — it's free-form text, so
mood/instrument/tempo descriptors work too, and at generation you can recombine
them (`"artist: alice in chains, tempo: slow"`) even if that pairing never
appeared in training.

Note `text_max_len` in `config.py` is 256 T5 tokens (~200 words); raise it if
you're writing long lyrics.

## Verified on this machine (RTX 5060 Ti, 16GB)

| check | result |
|---|---|
| codec round-trip | MAE 0.0224, **0.0ms** duration drift, stereo intact |
| cached decode == full forward | max delta 3.81e-06 |
| RoPE shift-invariance (1000 pos) | max delta 2.89e-04 |
| overfit 3 tracks, 260 steps | loss 9.86 → **0.008** (ppl 1.0) |
| memorized model recalls training audio | **78.1%** token match (random: 0.05%) |
| VRAM @ 138M params, seq_len 2048 | 6.4 GB |

## Files

| file | role |
|---|---|
| `transformer.py` | the model — flash attn, RoPE, RMSNorm, fused QKV, cross-attn |
| `codec.py` | EnCodec wrap; the flatten/unflatten contract (highest-risk file) |
| `textenc.py` | frozen T5 lyric encoder |
| `data.py` | scan, offline token cache, slot-aligned windows |
| `train.py` | bf16, grad accum, per-slot loss, VRAM assert |
| `generate.py` | sliding-window sampling, CFG, decode to audio |
| `audio_io.py` | soundfile + ffmpeg. No torchaudio (2.11 reworked its backends) |
| `bootstrap.py` | fetch a tiny public dataset |
| `ui.py` | Rich console |

Every file runs standalone as its own self-test: `python codec.py`,
`python transformer.py`, etc.

## Design notes

**encodec_32khz, not 48khz.** The 48kHz model is natively stereo, which looked
ideal, but its 1s internal chunking with 1% overlap makes encode→decode drift
~2000ms. 32kHz (what MusicGen uses) is exact, needs no scales, and has *better*
reconstruction. Stereo = encode L/R as two mono clips and interleave, which is
what MusicGen's stereo models do.

**RoPE, not learned positions.** Learned `nn.Embedding(seq_len)` caps context
and forces window re-anchoring. RoPE is relative, so sliding-window generation
is just "evict from the front and continue".

**No weight tying.** Saves 12.6M on a 138M model, and is actively wrong here:
only 1/8 of the vocab is legal at any position, and input/output roles differ.

**`is_causal` is gated on shapes.** SDPA anchors its causal mask top-left, which
is correct only when `q_len == kv_len`. With a KV cache it silently makes each
new token attend only to cached token 0 — measured wrong by 4.2, with no error
raised. `_attend()` in `transformer.py` gates on shape, never intent.

## Scaling up

Defaults (`config.py`) are `seq_len=4096` (10.24s), `batch_size=4`. Measured:
B=4 → 11.33GB, B=8 → 21GB (does *not* fit; Windows silently spills to shared
memory instead of raising OOM, hence the VRAM assert in `train.py`).

Expect textured audio in your training set's character, not radio-ready songs —
real music LMs use thousands of hours. The pipeline is correct and will scale.
