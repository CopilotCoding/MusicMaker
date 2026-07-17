# MusicMakerDiffusion

A diffusion music model, built as the controlled counterpart to
`../MusicMakerTransformer` (autoregressive). Same codec, same audio, same tags,
same parameter count — so a quality difference is attributable to the
**architecture**, not to anything else.

**These are two independent projects.** No shared imports, no shared files.
Either can be moved, edited, or deleted without touching the other. The
duplication (`audio_io.py`, `ui.py`, `textenc.py`, `dnn.py`) is deliberate.

## The comparison

| | MusicMakerTransformer (AR) | MusicMakerDiffusion (this) |
|---|---|---|
| representation | discrete tokens (16386 vocab) | continuous latents (128-dim) |
| attention | causal — sees backwards only | **bidirectional — sees the whole clip** |
| objective | cross-entropy, next token | MSE, predict the noise |
| generation | 400 sequential steps **per second** | **50 steps for any length** |
| length | unbounded (sliding window) | **fixed** at sample time |
| params | 138.4M | 138.9M (+0.4%) |
| codec fidelity | MAE 0.0224 (quantized) | **MAE 0.0177** (no quantization) |

**Diffusion should win on** global structure (bar 8 can influence bar 1 —
structurally impossible for the AR model), sampling speed (~82× fewer
sequential steps), and codec fidelity (skips quantization entirely).

**AR should win on** continuation ("here's 5s, keep going"), unbounded length,
and tight temporal conditioning — which is why every serious lyrics-conditioned
model (YuE, MusicGen) is autoregressive.

That's the hypothesis. The point of building both is to find out.

## Quickstart

```bash
python tag.py --music-dir "D:/Music/Grunge" --write   # metadata -> .txt sidecars
python dtrain.py --music-dir "D:/Music/Grunge"
python dgenerate.py --text "artist: soundgarden" --seconds 10 --out out.mp3
```

**`--music-dir` points at ANY folder and copies nothing.** It scans
recursively (`Artist/Album/track.mp3`), pairs `.txt` sidecars by full path, and
caches incrementally. Only the latent cache lives here.

**Point BOTH projects at the SAME `--music-dir`.** Different audio means you
are comparing datasets, not architectures.

## Verified on this machine (RTX 5060 Ti)

| check | result |
|---|---|
| latent round-trip | MAE 0.0177, 0.0ms drift, stereo intact |
| adaLN-Zero init | model is an exact no-op at step 0 |
| **bidirectionality** | last frame shifts first frame's output by 0.0047 |
| timestep conditioning | t=0 vs t=999 output delta 0.478 |
| schedule monotonic | a_bar 1.0000 → 0.000000 |
| **perfect pred recovers z0** | 4.77e-07 (validates the v↔eps↔z0 algebra) |
| scale factor | corpus RMS 3.17 → batch std 1.002 |
| overfit 2 tracks, 300 steps | loss → 0.63, 8 it/s, 4.6GB |
| sampling speed | **50 steps in ~1s** (AR: 59s for 6s of audio) |

## Files

| file | role |
|---|---|
| `dmodel.py` | DiT — bidirectional, adaLN-Zero timestep conditioning |
| `diffusion.py` | schedule, forward process, DDIM sampler, CFG |
| `dcodec.py` | EnCodec **encoder only** — continuous latents, pre-quantizer |
| `ddata.py` | latent cache + **measured** scale factor |
| `dtrain.py` | bf16, EMA, per-timestep loss, VRAM assert |
| `dgenerate.py` | DDIM sampling → audio |
| `dnn.py` | RMSNorm/RoPE — copies of the AR project's, kept identical |
| `audio_io.py`, `ui.py`, `textenc.py`, `tag.py`, `bootstrap.py` | copies, independent |

Every file runs standalone as its own self-test: `python dcodec.py`,
`python diffusion.py`, etc.

## Design notes

**Continuous latents, not tokens.** You cannot add Gaussian noise to an integer
token id. This taps `EncodecModel.encoder` before the quantizer: `(128, T)` at
50Hz — the same frame rate as the AR token stream, so both models see identical
temporal resolution.

**The scale factor is measured, never hardcoded.** Diffusion assumes ~unit
variance. Raw EnCodec latents measured RMS **3.17 on real music** but 4.58 on
synthetic tones — it's data-dependent. `ddata.build_latent_cache` computes it
over your corpus and stores it in the manifest (same idea as Stable Diffusion's
VAE scale factor). Getting it wrong doesn't raise; it silently trains the model
against the wrong signal-to-noise curve.

**v-prediction, not eps.** At t≈999 the input is nearly pure noise, so "predict
the noise" is trivial — the loss signal collapses exactly where global musical
structure is decided. v-prediction stays well-conditioned across the whole
schedule.

**Per-timestep loss runs *high at low t*** (0.94) **and low at high t** (0.62).
That's correct under v-prediction and the opposite of the eps-pred intuition:
at high t the target `v ≈ noise`, which the model can partly read off its input.
Only *flat* loss across t is a red flag — it means timestep conditioning isn't
reaching the network.

**adaLN-Zero.** The timestep modulates every block (shift/scale/gate) rather
than being concatenated as a token. Gates are zero-init'd **after** the global
std=0.02 loop — which would otherwise overwrite them — so the model starts as an
exact identity. Same lesson as the AR project's zero-init cross-attention.

**EMA is not optional.** Diffusion sample quality depends heavily on averaged
weights; `dgenerate.py` loads EMA by default (`--raw` to compare).

**Param matching.** adaLN costs `6*d²` per block, so 12 layers @ d_ff=3072 would
be 158.4M (+20M vs the AR model). 11 layers @ d_ff=2688 lands at 138.9M — within
0.4%. Measured, not guessed.

## Expectations

Same as the AR project: with a personal-scale library on one 5060 Ti, expect
textured audio in the character of your training set, not radio-ready songs.
Diffusion will likely sound *better sooner* on limited data, but won't do
continuation or lyric-alignment as naturally.
