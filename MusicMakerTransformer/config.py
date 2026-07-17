"""
config.py — every tunable in one place.

Values marked MEASURED were established by running the real thing on this
machine, not by arithmetic. Do not "optimize" them without re-measuring.
"""

from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).parent
DATA_RAW = ROOT / "data" / "raw"
DATA_CACHE = ROOT / "data" / "cache"
CKPT_DIR = ROOT / "checkpoints"


@dataclass
class Config:
    # ---- codec ----------------------------------------------------------
    # encodec_32khz -- the codec MusicGen uses, and it dominates the
    # alternatives on every axis we measured:
    #
    #                     48khz         24khz bw=3.0   32khz (this)
    #   duration          ~1s drift     exact          exact
    #   chunking/scales   1s + scales   none           none
    #   recon MAE         --            0.0435         0.0224
    #   stereo @4096 tok  --            6.83s          10.24s
    #
    # Mono model: stereo = encode L/R as a batch of 2, interleave the streams.
    # That is exactly what MusicGen's stereo models do ([1_L, 1_R, 2_L, 2_R...]).
    # frame_rate 50Hz, n_q=4, normalize=False (so NO scales to track).
    codec_id:  str   = "facebook/encodec_32khz"
    bandwidth: float = 2.2      # the model's only bandwidth; n_q still DERIVED
    stereo:    bool  = True

    # ---- text encoder ---------------------------------------------------
    t5_id:       str = "t5-base"    # d_model=768
    # MEASURED against the real lyric corpus: median 279 T5 tokens, max 714.
    # At 256 this truncated 61% of files and discarded ~15k tokens of hand-
    # written lyrics -- silently, mid-verse. 768 keeps 100% of them whole and
    # costs only +0.12GB VRAM (cross-attn over lyrics is cheap next to
    # self-attn over 4096 audio tokens) plus ~1.1MB/track of cached ctx.
    # T5 is frozen, so longer text costs no training compute.
    text_max_len: int = 768

    # ---- model ----------------------------------------------------------
    d_model:     int   = 768
    n_heads:     int   = 12
    n_layers:    int   = 12
    d_ff:        int   = 3072
    max_seq_len: int   = 16384      # 41s of audio at 400 tok/s
    dropout:     float = 0.1
    rope_base:   float = 10000.0

    # ---- training -------------------------------------------------------
    # 16384 tokens = 41s of audio -- a verse and a chorus, not just a riff.
    # 4x the old 4096 (10.2s), which could not hold musical structure at all.
    #
    # MEASURED with grad_checkpoint=True, text_max_len=768, real cross-attn:
    #   seq_len  audio  batch  peak GB  s/step
    #     4096   10.2s     4     5.15    1.38
    #     8192   20.5s     2     5.14    1.82
    #    16384   41.0s     1     5.14    2.69
    #    16384   41.0s     2     8.57    5.41   <-- chosen
    #    16384   41.0s     4    15.43   OVER budget
    #
    # For contrast, WITHOUT checkpointing batch=4 @ 4096 was 14.99GB. Trading
    # ~30% step time for ~3x less memory is what makes 41s reachable at all.
    seq_len:     int   = 16384
    batch_size:  int   = 2
    # accum 2 -> effective batch 4. NOT the textbook 32, deliberately.
    #
    # Accumulation does not change throughput -- it only decides how many
    # micro-steps are bundled into one weight update. MEASURED ~5.5s per
    # micro-step, so:
    #   accum 16 (eff 32):  88s/step,  45 steps/epoch
    #   accum  4 (eff  8):  22s/step, 180 steps/epoch
    #   accum  2 (eff  4):  11s/step, 361 steps/epoch   <-- chosen
    # Same tokens/second in every row; the difference is update frequency.
    #
    # With only 1444 windows, eff batch 32 gives a miserly 45 updates per
    # epoch. Small batches are noisier per step but buy 8x more updates from
    # the same compute, which matters far more on a small corpus.
    grad_accum:  int   = 2
    # 3e-5 with the LAMB optimizer (see optim.py), NOT AdamW. The history:
    #
    #   AdamW 3e-4  -> diverged at step 350 (instant: too-high-LR signature).
    #   AdamW 5e-5  -> clean ~600 steps, then grad norm ramped 0.7 -> 5.4 -> 10.9
    #                  and BOTH losses turned up. Same at beta2 0.95 AND 0.999.
    #   AdamW 1e-5  -> never diverged, but STALLED at loss 9.39 (~0.007 progress
    #                  per 50 steps). Too cold. A slower way to get nothing.
    #
    # The "works then blows up 600 steps later" pattern is NOT a too-high LR
    # (that breaks immediately). It is an UNBOUNDED update: at effective batch 4
    # AdamW's second-moment estimate drifts and the step size wanders until it
    # destabilizes. LAMB fixes this at the source -- its trust ratio renormalizes
    # every layer's step to a fixed fraction of the layer's own norm each step,
    # so a late gn ramp cannot compound. LAMB is famously LR-insensitive, which
    # is the property four AdamW guesses proved this project needs.
    #
    # 3e-5 sits between the too-cold 1e-5 and the (under AdamW) too-hot 5e-5.
    # Under LAMB the exact value matters far less; a grad-norm backoff halves it
    # automatically if it still ramps (train.py: GradNormBackoff). This is the
    # measured mechanism attacked directly, not a fifth guess at a magic number.
    lr:          float = 3e-5
    # (0.9, 0.999) -- the LAMB paper's default. beta2 is the half-life of the
    # squared-gradient memory (0.999 ~= 1000 steps). Under AdamW at effective
    # batch 4 this estimate drifts and was implicated in the late gn ramp; LAMB's
    # trust ratio renormalizes on top of it every step, so the drift no longer
    # sets the step size and 0.999 is the safe, standard choice.
    betas:       tuple = (0.9, 0.999)
    weight_decay: float = 0.1
    warmup_steps: int  = 200
    # MEASURED ~5.5s/step at seq_len=16384, batch 2x16. The old 100k default
    # would be 6.4 DAYS and ~2200 epochs over a 257-track library -- it would
    # overfit into the ground long before finishing. 8000 steps is ~12 hours
    # and ~180 epochs, which is already generous for 18.7 hours of audio.
    # Raise it if val loss is still falling when it ends; the cosine schedule
    # is sized from this, so changing it changes the LR curve.
    max_steps:    int  = 8_000
    grad_clip:    float = 1.0
    # ON by default: seq_len=16384 (41s) is simply not reachable without it.
    # Activations are ~85% of memory here, so recomputing them in the backward
    # pass instead of storing them is worth the ~30% step-time cost.
    grad_checkpoint: bool = True
    cond_dropout: float = 0.1       # drop lyrics 10% -> enables CFG at sampling
    val_frac:     float = 0.05
    # Every 50 steps ~= every 9 minutes at 10.8s/step. The per-slot table is
    # the only signal that the interleave is intact, so waiting 45 minutes
    # (the old 250) to find out is a bad trade for ~2.8% overhead:
    #   every 250: 0.6% overhead, first table at 45 min
    #   every  50: 2.8% overhead, first table at  9 min   <-- chosen
    #   every  25: 5.6% overhead, first table at  4.5 min
    eval_interval: int = 50
    ckpt_interval: int = 1000
    seed:          int = 1337

    # Hard VRAM ceiling. Windows silently spills to shared host memory instead
    # of raising OOM -- a 21GB "success" on a 16GB card trains at unusable
    # speed and looks fine. Assert, don't trust the absence of an exception.
    vram_budget_gb: float = 14.0

    # ---- generation -----------------------------------------------------
    temperature: float = 1.0
    top_k:       int   = 250        # NOT 40 -- codec vocabs are high-entropy;
    top_p:       float = 0.95       # aggressive truncation buzzes and loops.
    guidance:    float = 3.0        # classifier-free guidance scale

    # ---- audio io -------------------------------------------------------
    mp3_bitrate: str = "192k"       # "regular quality"
    peak_norm:   float = 0.95

    def __post_init__(self):
        assert self.d_model % self.n_heads == 0
        assert (self.d_model // self.n_heads) % 2 == 0, "RoPE needs even d_head"


CFG = Config()
