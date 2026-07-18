# MusicMaker

A music LM built on a hand-written transformer. Trains on mp3/wav/flac,
generates stereo wav/mp3/flac, conditions on lyrics from sibling `.txt` files.

Two generations of architecture live in this repo:

- **Delay-pattern model (current)** — `delay.py` / `train_delay.py`: one
  sequence position per audio frame, all codebooks predicted in parallel by
  18 heads. Whole songs (334s) fit in one context window.
- **Flat model (legacy, fully working)** — `transformer.py` / `train.py`:
  one position per codebook token. Proven end to end, kept as the baseline;
  its best full-corpus run reached val 7.74 on the grunge set.

## How it works

Raw audio isn't discrete. A neural codec turns it into tokens; the
transformer models the tokens; the codec turns them back into sound.

```
audio ──DAC 44.1kHz──> 86 frames/sec x 18 tokens ──[TRANSFORMER]──> tokens ──DAC──> audio
lyrics.txt ──T5 (frozen)──> embeddings ──cross-attention──┘
```

**Codec: Descript DAC 44.1kHz (~8kbps)** — 9 codebooks x 1024 entries, stereo
as L/R batch-of-two, 18 slots per frame, vocab 18,434. Chosen by ear: the old
EnCodec 2.2kbps ceiling made vocals warble ("almost autotuned"); DAC was
"basically the same as the original" on the same slice (`dac_ceiling.py`
reproduces that A/B/C). The codec is the ceiling on everything above it.

**Delay pattern (MusicGen-style)** — each position carries all 18 tokens of
one frame: input is the sum of their embeddings, output is 18 parallel
heads. Codebook q is staggered q steps so coarse still informs fine through
the causal history. Consequences, all measured:

- 16384 positions = 41s under the flat interleave -> **191s** under delay
  (DAC) — and **28672 positions = 334s: whole songs** at 9.4GB, 6.9s/step.
- 3.2x training throughput at equal audio window; ~13x faster generation
  (faster than realtime on GPU).
- The slot logit-mask is structurally impossible to need — each head IS its
  codebook. One silent-failure class deleted.

Model: 113M params — d=768, 12 layers, 12 heads, RoPE, RMSNorm, flash
attention, cross-attention to frozen T5 (zero-init: silent until lyrics
matter, and always built so pretrain/fine-tune checkpoints stay
shape-compatible).

## The pipeline (current runbook)

```bash
python pretrain_fetch.py --shards 0 0     # 1. one-shard smoke test
python pretrain_fetch.py                  # 2. MTG-Jamendo moodtheme: ~1050h
                                          #    streamed shard-at-a-time, audio
                                          #    deleted after encoding, ~3GB cache
python train_delay.py --pretrain-cache data/cache_pretrain --steps 8000
                                          # 3. unconditional pretrain
python train_delay.py --music-dir "..\Grunge" --init-from checkpoints_delay/best.pt
                                          # 4. fine-tune on grunge + lyrics
python generate_delay.py --lyrics-file song.txt --seconds 120 --out song.mp3
                                          # 5. hear it
```

Training auto-saves on Ctrl-C or crash (`checkpoints_delay/interrupted.pt`)
and resumes exactly where it stopped; `best.pt` always holds the
best-validation model.

## Using your own music

```bash
python tag.py --music-dir "D:/Music/Grunge" --audit    # what can it teach?
python tag.py --music-dir "D:/Music/Grunge" --write    # metadata -> .txt sidecars
python train_delay.py --music-dir "D:/Music/Grunge"
```

**`--music-dir` points at ANY folder and copies nothing.** Recursive scan,
full-path-keyed pairing and cache names (every album has an `01 - Intro`),
incremental resumable caching. Lyrics are free-form sidecar `.txt`;
`text_max_len` 768 T5 tokens keeps hand-written lyrics whole. Nothing in your
tags is ever dropped (`--drop-constant` exists, off by default).

**Caches are per-codec and never mix** — DAC tokens live in `data/cache_dac`
(grunge) and `data/cache_pretrain` (Jamendo); the old EnCodec cache stays in
`data/cache` for the legacy model. The cache signature refuses a mismatch.

## How training reads a song

In-order, non-overlapping windows of frames, whole songs first-class:

- window 0 is `[BOS-row, frames...]` — the model *learns to start a song*
- interior windows begin one frame early, so every frame in the song is a
  target exactly once (no unlearnable boundary tokens)
- short tracks pad with masked PAD rows — nothing is dropped, padding is
  free to the loss
- validation holds out **whole songs**, never windows
- generation is chunk-aligned: the KV cache resets every window, exactly the
  context pattern training taught. (A sliding window was measured to derail
  a memorized song at precisely the first boundary — the model had never
  seen a mid-chunk context start.)

At 28672 steps (334s), ~84% of the grunge corpus (mean track 4.4 min) trains
as complete songs — both occurrences of every chorus inside one window,
which is the cheapest long-range structure a transformer learns
(copy/induction) and the whole point of the window size.

## Optimizer: LAMB, at effective batch 1–4

AdamW failed at every LR on this hardware (diverged at 3e-4 and 5e-5,
stalled at 1e-5) — at tiny effective batch its variance estimate drifts for
hundreds of steps, then the run unlearns itself. LAMB's per-layer trust
ratio re-pins every step to `lr x ||w||`, and three LRs spanning 33x all
trained stably on the first try. A grad-norm backoff auto-halves the LR on a
sustained ramp; on the flat model's full run it fired four times, walking
1e-3 -> 6.2e-5 like an empirically-driven annealing schedule while val set a
record at every eval. (It also exposed its own design flaw — the relative
trigger deafens as its fires raise the baseline — so it now carries an
absolute tripwire too.) Full story with receipts:
[LAMB_AT_BATCH_FOUR.md](LAMB_AT_BATCH_FOUR.md).

## Verified on this machine (RTX 5060 Ti 16GB, 32GB RAM)

| check | result |
|---|---|
| DAC flatten contract + audio round-trip | exact / clean (`dac_codec.py`) |
| DAC vs EnCodec by ear, same slice | DAC "basically the original"; EnCodec warbles |
| delay grid round-trip + stagger + edge masks | exact (`delay.py` self-tests) |
| cached decode == full forward (delay) | 5.96e-07 |
| flat model, one song memorized -> regenerated | **100.0%** free-run token match |
| delay model, whole song in ONE window (323s) | **99.8%** free-run, audibly the song |
| delay+DAC memorize/regenerate | 89.1% -> polish-to-target added (99.99% TF per chunk) |
| delay training cost @ 28672 steps (334s), batch 1 | **9.4GB, 6.9s/step** |
| delay throughput vs flat, same audio window | **3.2x** train, ~13x generation |
| flat full-corpus baseline (LAMB 1e-3) | val 9.86 -> **7.74** in 800 steps |

The memorize-one-song test is the load-bearing verification: it exercises
codec, grid/stagger, BOS, RoPE, KV cache, chunk resets, sampling and decode
simultaneously — three separate silent bugs were caught by exactly this test
and nothing else.

## Files

| file | role |
|---|---|
| `delay.py` | delay-pattern model + grid build/undelay (current architecture) |
| `dac_codec.py` | DAC 44.1kHz wrapper; `make_codec()` factory (dac/encodec) |
| `train_delay.py` | delay training: pretrain / fine-tune / resume, LAMB+backoff |
| `generate_delay.py` | delay sampling: CFG, per-head sampling, decode to audio |
| `pretrain_fetch.py` | MTG-Jamendo -> token cache, streamed, sha256, resumable |
| `overfit_one_delay.py` | memorize one song + audit + polish-to-target + regen |
| `dac_ceiling.py` | the codec A/B/C that decided the DAC swap |
| `transformer.py` `train.py` `generate.py` | flat model (legacy, proven) |
| `overfit_one.py` `diagnose_gen.py` | flat-model verification suite |
| `codec.py` | EnCodec 32kHz wrapper (legacy codec) |
| `optim.py` | LAMB |
| `data.py` | scan, cache, windows, per-song split (flat); shared helpers |
| `textenc.py` | frozen T5 lyric encoder |
| `plot.py` | live matplotlib window over metrics.csv |
| `tag.py` | metadata -> `.txt` sidecars via ffprobe |
| `audio_io.py` | soundfile + ffmpeg (no torchaudio) |
| `ceiling.py` `lrcheck.py` `lrfind.py` `bootstrap.py` `ui.py` | tools |

## Design notes (each one paid for)

**The codec is the ceiling.** A model that reproduces its training tokens at
100% still sounds exactly as good as the codec round-trip — no better. Judge
codecs by ear on YOUR material before training anything (`dac_ceiling.py`).

**Positions = frames, so codebook count is free but frame rate is not.**
DAC's 86fps vs EnCodec's 50fps is why the same window buys 191s not 327s.
`frame_rate` is not an integer (44100/512); all arithmetic stays in frames.

**BOS in training, because generation needs it.** A model trained with a
real token at position 0 cannot start a song from nothing — an untrained BOS
shifts every position and produces confident garbage. Trains fine, generates
noise, no error raised.

**RoPE positions passed explicitly during generation.** Deriving them from
cache length breaks the moment the cache is truncated or reset.

**`is_causal` gated on shapes.** SDPA anchors its causal mask top-left,
correct only when `q_len == kv_len`; with a KV cache it silently attends
each new token to cached token 0 only. Measured wrong by 4.2, no warning.

**Generation must match training numerics.** bf16-trained, fp32-sampled cost
~2x the argmax flips at 18 heads/row; rollouts cascade. Sample under the
same autocast.

**Teacher-forced audit before trusting any rollout.** The training loss only
shows the last batch; one under-memorized chunk out of 49 derails free-run
generation from that point on. Audit per chunk, polish to target.

**Windows lies about OOM.** It spills VRAM to shared host memory instead of
raising — a 21GB "success" on a 16GB card trains at unusable speed. Assert
against a hard budget (`vram_budget_gb`).

## Status

Verified: everything in the table above. Pending, in order: fetch smoke test
(one shard) -> full Jamendo fetch (~1050h, ~3GB cache) -> unconditional
pretrain (~21h/epoch at whole-song windows) -> grunge+lyrics fine-tune ->
the first generated songs with structure. The flat model's val-7.74 grunge
baseline is the number to beat.
