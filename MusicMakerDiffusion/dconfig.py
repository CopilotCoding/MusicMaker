"""
dconfig.py — config for the diffusion music model.

This project is STANDALONE: it shares no code with MusicMakerTransformer. The
values below intentionally MIRROR that project's config so a head-to-head
comparison is honest, but they are independent copies -- changing one project
must never silently alter the other.

Keep these identical to the AR project or the comparison is meaningless:
    codec, sample rate, stereo, T5 text encoder, data/raw layout, tags,
    context length in seconds, model size

Different by necessity (this IS the experiment):
    continuous latents instead of discrete tokens
    bidirectional DiT instead of causal transformer
    ~50 denoising steps over a whole clip instead of 400 AR steps per second

Point both projects at the same data/raw/ (copy or symlink your tracks in) to
train them on identical audio.
"""

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).parent
DATA_RAW = ROOT / "data" / "raw"                 # this project's own audio
DATA_CACHE = ROOT / "data" / "cache"             # latents, not tokens
CKPT_DIR = ROOT / "checkpoints"


@dataclass
class DConfig:
    # ---- SHARED with ../config.py — keep identical -----------------------
    codec_id:  str   = "facebook/encodec_32khz"
    bandwidth: float = 2.2
    stereo:    bool  = True
    t5_id:     str   = "t5-base"
    # MEASURED against the real lyric corpus: median 279 T5 tokens, max 714.
    # At 256 this silently truncated 61% of files mid-verse. 768 keeps all of
    # them whole. Must match the AR project or the comparison is unfair.
    text_max_len: int = 768

    # ---- latent space ---------------------------------------------------
    # EnCodec's encoder output BEFORE quantization: (128, T) at 50Hz.
    # Same frame rate as the AR token stream, so both models see identical
    # temporal resolution -- that is what makes this a fair comparison.
    latent_dim:  int = 128
    frame_rate:  int = 50
    # Diffusion assumes ~unit-variance data; raw latents measured std ~5.4 on
    # music (and ~3.0 on noise -- it is data-dependent). MEASURED from the real
    # corpus at cache-build time and stored in the manifest. Never hardcode.
    latent_scale: float = None

    # ---- model (DiT) ----------------------------------------------------
    # PARAM-MATCHED to the AR model: 138.9M vs its 138.4M (+0.4%). This matters
    # -- a quality difference must be attributable to the architecture, not to
    # one model simply being bigger. adaLN modulation costs 6*d^2 per block, so
    # 12 layers @ d_ff=3072 would land at 158.4M (+20M). 11 layers @ d_ff=2688
    # is the closest match. MEASURED, not guessed.
    d_model:  int   = 768
    n_heads:  int   = 12
    n_layers: int   = 11         # 12 would be +20M vs the AR model
    d_ff:     int   = 2688       # trimmed from 3072 to absorb adaLN's cost
    dropout:  float = 0.1
    # 512 frames = 10.24s -- exactly the AR model's 4096-token context.
    n_frames: int   = 512

    # ---- diffusion ------------------------------------------------------
    # v-prediction + cosine schedule: better behaved than eps-pred at the
    # high-noise end, which is where music structure gets decided.
    timesteps:    int   = 1000
    schedule:     str   = "cosine"
    prediction:   str   = "v"          # "v" | "eps"
    sample_steps: int   = 50           # DDIM steps at inference
    ddim_eta:     float = 0.0          # 0 = deterministic

    # ---- training -------------------------------------------------------
    batch_size:   int   = 8            # latents are far lighter than tokens
    grad_accum:   int   = 4
    lr:           float = 1e-4         # diffusion likes lower lr than AR
    betas:        tuple = (0.9, 0.999)
    weight_decay: float = 0.01
    warmup_steps: int   = 200
    max_steps:    int   = 100_000
    grad_clip:    float = 1.0
    cond_dropout: float = 0.1          # enables CFG, same as AR
    ema_decay:    float = 0.999        # EMA matters a lot for diffusion quality
    val_frac:     float = 0.05
    eval_interval: int  = 250
    ckpt_interval: int  = 1000
    seed:          int  = 1337
    vram_budget_gb: float = 14.0

    # ---- generation -----------------------------------------------------
    guidance:   float = 3.0            # same default as AR for comparability
    peak_norm:  float = 0.95
    mp3_bitrate: str  = "192k"

    def __post_init__(self):
        assert self.d_model % self.n_heads == 0


DCFG = DConfig()
