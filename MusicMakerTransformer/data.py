"""
data.py — scan data/raw, cache tokens offline, serve aligned chunks.

THE OFFLINE CACHE IS NOT OPTIONAL. EnCodec + T5 are far too slow to re-run
every epoch; encoding once to disk is the difference between a training step
measured in milliseconds and one measured in seconds.

LAYOUT -- point --music-dir at ANY folder; nothing is copied.

    <your library>/Artist/Album/track.mp3  +  track.txt   (sidecar, optional)
    data/cache/    <stem>_<pathhash>.npz   {codes int16, ctx fp16, ctx_mask bool}
                   manifest.json           identity of codec/T5/dir that built it

Scanning is RECURSIVE and pairs by full path, not bare filename: every album
has an "01 - Intro", and stem-keyed pairing would silently cross-match them.
Cache names carry a path hash for the same reason.

ALIGNMENT IS THE WHOLE GAME. The flat stream is interleaved
[q0L, q0R, q1L, q1R, ...] with stride n_slots. Any chunk that does not start on
a multiple of n_slots shifts every token into the wrong slot -- the model then
trains on noise, silently, with a loss curve that looks plausible. Every start
offset is snapped to n_slots and asserted.

CACHE INVALIDATION: the manifest records codec id, n_q, n_slots, vocab, T5 id,
and each file's mtime/size. Reusing a cache built with a different codec would
change n_slots and desynchronize the interleave -- same silent noise. Any
mismatch forces a re-encode.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

import ui
from config import CFG, DATA_CACHE, DATA_RAW

AUDIO_EXT = (".mp3", ".wav", ".flac")


def scan(raw_dir=None, recursive: bool = True):
    """
    [(audio_path, lyric_path|None)] paired by sidecar: song.mp3 -> song.txt in
    the same folder.

    RECURSIVE by default, because a real library is Artist/Album/track.mp3.

    Pairing is by FULL PATH, not by bare filename. Nested libraries are full of
    colliding stems -- every album has an "01 - Intro" -- and matching on stem
    alone would silently pair one album's audio with another's text, or collapse
    them into one entry. Path-keyed pairing makes that impossible.
    """
    raw_dir = Path(raw_dir) if raw_dir else DATA_RAW
    if not raw_dir.exists():
        return []
    it = raw_dir.rglob("*") if recursive else raw_dir.iterdir()
    audio = sorted(p for p in it if p.suffix.lower() in AUDIO_EXT)
    out = []
    for p in audio:
        txt = p.with_suffix(".txt")                  # sidecar, same folder
        out.append((p, txt if txt.exists() else None))
    return out


def cache_name(audio_path, root=None) -> str:
    """
    Collision-proof cache filename for a track.

    "Nirvana/Nevermind/01 - Intro.mp3" and "Pearl Jam/Ten/01 - Intro.mp3" must
    not both want "01 - Intro.npz". Prefix the stem with a hash of the full
    path so every track gets its own slot no matter how the library nests.
    """
    p = Path(audio_path).resolve()
    h = hashlib.md5(str(p).encode()).hexdigest()[:8]
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in p.stem)[:60]
    return f"{safe}_{h}"


def _sig(codec, textenc) -> dict:
    return {
        # getattr: DacCodec has no bandwidth knob (one fixed rate); the field
        # stays in the sig so an EnCodec-built cache can never be mistaken for
        # a DAC one — their vocab/n_slots differ anyway, belt and suspenders.
        "codec_id": codec.model_id, "bandwidth": getattr(codec, "bandwidth", None),
        "n_q": codec.n_q, "channels": codec.channels, "n_slots": codec.n_slots,
        "vocab_size": codec.vocab_size, "sr": codec.sr,
        "t5_id": textenc.model_id if textenc else None,
        "text_max_len": textenc.max_len if textenc else None,
    }


def _file_key(p) -> str:
    st = p.stat()
    return hashlib.md5(f"{p.name}:{st.st_size}:{int(st.st_mtime)}".encode()).hexdigest()[:16]


def build_cache(codec, textenc=None, raw_dir=None, force: bool = False,
                cache_dir=None, quiet: bool = False):
    """
    Encode every track once. Returns list of cache paths.

    raw_dir may be ANY folder -- point it at your music library and nothing is
    copied. Only the cache (tokens, ~50KB/track) lives in this project.
    """
    cache = Path(cache_dir) if cache_dir else DATA_CACHE
    cache.mkdir(parents=True, exist_ok=True)
    pairs = scan(raw_dir)
    if not pairs:
        raise SystemExit(f"no audio found in {raw_dir or DATA_RAW} — "
                         "pass --music-dir, or run bootstrap.py")

    sig = _sig(codec, textenc)
    sig["raw_dir"] = str(Path(raw_dir).resolve()) if raw_dir else None
    man_path = cache / "manifest.json"
    man = {}
    if man_path.exists() and not force:
        old = json.loads(man_path.read_text())
        if old.get("sig") == sig:
            man = old.get("files", {})
        else:
            print("  cache signature changed (codec/T5/dir differs) — rebuilding")

    # Work out what actually needs encoding BEFORE starting, so the progress
    # bar has a real total and a resumed run says "12 to go", not "256".
    out, todo = [], []
    for audio_p, txt_p in pairs:
        name = cache_name(audio_p, raw_dir)
        npz = cache / f"{name}.npz"
        if man.get(name) == _file_key(audio_p) and npz.exists():
            out.append(npz)
        else:
            todo.append((audio_p, txt_p, name, npz))

    if todo and not quiet:
        ui.log(f"[dim]encoding {len(todo)} track(s) through EnCodec + T5"
               + (f", {len(out)} already cached" if out else "")
               + " — one time, resumable[/]")

    # A single-shot encode of a 4-minute stereo track peaks at ~12.8GB. That
    # fits in 16GB only if the GPU is otherwise EMPTY. T5 is ~1GB and is not
    # needed while audio is encoding, so park it on the CPU and bring it back
    # only for the text. Without this the two together (plus the allocator's
    # cached blocks) overflow into Windows shared memory -- which never raises,
    # it just drops throughput ~10x.
    t5_parked = False
    if textenc is not None and str(getattr(textenc, "device", "cpu")).startswith("cuda"):
        textenc.enc.to("cpu")
        t5_parked = True
        torch.cuda.empty_cache()

    prog = ui.cache_progress() if (todo and not quiet) else None
    task = prog.add_task("encoding", total=len(todo), stats="") if prog else None
    if prog:
        prog.start()
    try:
        for i, (audio_p, txt_p, name, npz) in enumerate(todo):
            flat = codec.encode_wav_file(audio_p) if hasattr(codec, "encode_wav_file") \
                else codec.encode_file(audio_p)
            assert flat.shape[0] % codec.n_slots == 0, "encode produced unaligned stream"
            assert int(flat.max()) < codec.vocab_size, "token id exceeds vocab"

            payload = {"codes": flat.numpy().astype(np.int16)}  # max id 16385 < 32767
            if txt_p and textenc:
                text = txt_p.read_text(encoding="utf-8", errors="replace").strip()
                if text:
                    # Audio activations are freed; T5 can have the GPU now.
                    if t5_parked:
                        torch.cuda.empty_cache()
                        textenc.enc.to(textenc.device)
                    ctx, msk = textenc.encode([text])
                    payload["ctx"] = ctx[0].cpu().numpy().astype(np.float16)
                    payload["ctx_mask"] = msk[0].cpu().numpy()
                    del ctx, msk
                    if t5_parked:
                        textenc.enc.to("cpu")
            # Hand every byte back before the next 12.8GB encode. The allocator
            # caches freed blocks by default, and that cache is exactly what
            # tips the next track over the edge.
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            np.savez(npz, **payload)   # not savez_compressed: 11x slower for
                                       # ~20% off a cache that is only ~150MB
            man[name] = _file_key(audio_p)
            out.append(npz)

            # Flush the manifest as we go. Writing it only at the end means a
            # Ctrl-C loses the record of every track already encoded, and the
            # next run redoes all of them despite the .npz files being right
            # there. Cheap insurance on a 20-minute job.
            man_path.write_text(json.dumps({"sig": sig, "files": man}, indent=2))

            if prog:
                secs = flat.shape[0] / codec.tokens_per_sec
                prog.update(task, completed=i + 1,
                            stats=f"[dim]{audio_p.name[:34]}[/] {secs:.0f}s")
    finally:
        if prog:
            prog.stop()

    man_path.write_text(json.dumps({"sig": sig, "files": man}, indent=2))
    return out


class MusicDataset(Dataset):
    """
    Serves each song IN ORDER as fixed, non-overlapping seq_len windows --
    start to finish, never random chunks. Window 0 of every song carries a BOS,
    so the model learns to begin a song from nothing; every later window is a
    plain next-token slice continuing where the previous one ended.

    WHY IN-ORDER, NON-OVERLAPPING (deliberate, replaces random-crop):
      - Random cropping fed a different random slice each step -- fine for
        learning LOCAL transitions, but the model never saw a song's true
        progression start to finish, and free-run generation (which begins at
        the song start) had no coherent position-0 to reproduce.
      - Fixed in-order windows teach the actual song: window k is [k*S, (k+1)*S),
        so concatenating the model's continuations walks the whole track.
      - Tradeoff: no crop augmentation, so the same windows repeat every epoch;
        on a small corpus this memorizes faster and can overfit sooner. best.pt
        (best val) matters more as a result.

    BOS: input for window 0 is [BOS, t0, ..., t_{S-2}], target [t0, ..., t_{S-1}]
    -- BOS at position 0 predicts the real first token. Generation feeds BOS the
    same way, so training and generation agree on what position 0 holds. Without
    it, position 0 is a real token in training but BOS in generation, shifting
    every token one RoPE position -> silent garbage. If bos is None, window 0 is
    a plain slice too (only primed generation works then).
    """
    def __init__(self, cache_paths, seq_len: int, n_slots: int,
                 random_crop: bool = False, bos: int | None = None):
        assert seq_len % n_slots == 0, f"seq_len {seq_len} must be a multiple of n_slots {n_slots}"
        self.seq_len = seq_len
        self.n_slots = n_slots
        self.bos = bos
        # random_crop is accepted for call-compatibility but IGNORED: training is
        # always strictly in-order now. Warn rather than silently do the wrong
        # thing if someone still passes True expecting the old behaviour.
        if random_crop:
            ui.log("[dim]note: random_crop is ignored — training is strictly "
                   "in-order, non-overlapping windows now.[/]")

        self.tracks, self.index = [], []
        for p in cache_paths:
            z = np.load(p)
            codes = torch.from_numpy(z["codes"].astype(np.int64))
            ctx = torch.from_numpy(z["ctx"].astype(np.float32)) if "ctx" in z else None
            msk = torch.from_numpy(z["ctx_mask"]) if "ctx_mask" in z else None
            min_len = seq_len if bos is not None else seq_len + 1
            if codes.shape[0] < min_len:
                continue
            ti = len(self.tracks)
            self.tracks.append({"codes": codes, "ctx": ctx, "mask": msk,
                                "name": p.stem})
            # Fixed, non-overlapping TARGET coverage: chunk 0 targets t0..t_{S-1}
            # (from BOS), interior chunk w targets t_{wS}..t_{wS+S-1} (its input
            # starts ONE token early, at t_{wS-1}). Without that one-early start
            # the boundary token t_{wS} is never anyone's target -- chunk 0's
            # targets stop at t_{S-1} and chunk 1's used to begin at t_{S+1} --
            # so generation hit position S with a token the model was never
            # trained to produce, and derailed exactly there (measured: 68.2%
            # free-run match = 4096/6000 = chunk 0 perfect, then noise).
            if bos is not None:
                n_win = codes.shape[0] // seq_len         # chunk w needs (w+1)S
            else:
                n_win = (codes.shape[0] - 1) // seq_len   # legacy, no BOS
            self.index += [(ti, w) for w in range(max(1, n_win))]
        if not self.tracks:
            raise SystemExit("no track long enough for seq_len — lower seq_len")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        ti, w = self.index[i]
        tr = self.tracks[ti]
        codes = tr["codes"]
        start = w * self.seq_len                           # fixed, in-order
        assert start % self.n_slots == 0                   # multiple of S, so OK

        if self.bos is not None and w == 0:
            # First window of the song: mark the start with BOS.
            # input  = [BOS, t0, ..., t_{S-2}]   target = [t0, ..., t_{S-1}]
            real = codes[0:self.seq_len]
            x = torch.cat([torch.tensor([self.bos], dtype=real.dtype), real[:-1]])
            y = real
        elif self.bos is not None:
            # Interior window, starting ONE token early so the boundary token
            # t_{wS} is a target:  input [t_{wS-1} .. t_{wS+S-2}]
            #                      target [t_{wS}   .. t_{wS+S-1}]
            # Target coverage across chunks is then contiguous: every token in
            # the song is predicted exactly once. y still begins on slot 0
            # (wS % n_slots == 0), so the per-slot loss table stays valid.
            win = codes[start - 1:start + self.seq_len]
            x, y = win[:-1], win[1:]
        else:
            # Legacy (no BOS): plain next-token slice.
            win = codes[start:start + self.seq_len + 1]
            x, y = win[:-1], win[1:]
        return {"x": x, "y": y, "ctx": tr["ctx"], "mask": tr["mask"],
                "name": tr["name"]}


def collate(batch):
    """
    Stack windows; pad lyric ctx to the batch max.

    NaN GUARD: a cross-attn row that is entirely masked makes SDPA emit NaN.
    Tracks without lyrics get a single visible zero-vector key rather than an
    all-False mask, so the row is a no-op instead of a poison pill.
    """
    # x/y are pre-split in __getitem__ (a BOS window has x != y[:-1], so we can
    # no longer slice a single "tokens" tensor here).
    x = torch.stack([b["x"] for b in batch]).contiguous()
    y = torch.stack([b["y"] for b in batch]).contiguous()

    if all(b["ctx"] is None for b in batch):
        return {"x": x, "y": y, "ctx": None, "ctx_mask": None,
                "names": [b["name"] for b in batch]}

    d = next(b["ctx"].shape[-1] for b in batch if b["ctx"] is not None)
    L = max((b["ctx"].shape[0] for b in batch if b["ctx"] is not None), default=1)
    B = len(batch)
    ctx = torch.zeros(B, L, d)
    msk = torch.zeros(B, L, dtype=torch.bool)
    for i, b in enumerate(batch):
        if b["ctx"] is not None:
            n = b["ctx"].shape[0]
            ctx[i, :n] = b["ctx"]
            msk[i, :n] = b["mask"] if b["mask"] is not None else True
        else:
            msk[i, 0] = True          # keep one key visible: never all-False
    return {"x": x, "y": y, "ctx": ctx, "ctx_mask": msk,
            "names": [b["name"] for b in batch]}


if __name__ == "__main__":
    import argparse
    import transformers
    transformers.logging.set_verbosity_error()
    from codec import Codec
    from textenc import LyricEncoder

    _ap = argparse.ArgumentParser()
    _ap.add_argument("--music-dir", default=None,
                     help="scan ANY folder (recursive) instead of data/raw")
    _a = _ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    codec = Codec(device=dev)
    te = LyricEncoder(device=dev)
    print(codec); print(te)

    paths = build_cache(codec, te, raw_dir=_a.music_dir)
    print(f"cache: {len(paths)} tracks")

    seq = 2048
    ds = MusicDataset(paths, seq_len=seq, n_slots=codec.n_slots)
    print(f"dataset: {len(ds)} windows of {seq} tokens "
          f"({seq/codec.tokens_per_sec:.2f}s each) over {len(ds.tracks)} tracks")

    from torch.utils.data import DataLoader
    dl = DataLoader(ds, batch_size=4, shuffle=True, collate_fn=collate)
    b = next(iter(dl))
    print(f"batch: x {tuple(b['x'].shape)} y {tuple(b['y'].shape)} "
          f"ctx {tuple(b['ctx'].shape)} mask {tuple(b['ctx_mask'].shape)}")
    print(f"  tracks in batch: {b['names']}")

    # x/y must be shifted by exactly one token
    assert torch.equal(b["x"][:, 1:], b["y"][:, :-1]), "x/y not next-token aligned"
    assert b["x"].max() < codec.vocab_size and b["x"].min() >= 0
    assert b["ctx_mask"].any(dim=1).all(), "an all-masked row would NaN the SDPA"

    # Slot identity: token at flat position i must live in slot i % n_slots.
    # This is the assertion that catches an unaligned crop -- the failure that
    # otherwise just looks like "the model won't converge".
    for bi in range(b["x"].shape[0]):
        for i in range(0, 64):
            slot = i % codec.n_slots
            tok = b["x"][bi, i].item()
            lo, hi = slot * codec.codebook_sz, (slot + 1) * codec.codebook_sz
            assert lo <= tok < hi, f"batch {bi} pos {i}: token {tok} not in slot {slot}"
    print("  slot alignment verified over 64 positions x 4 examples")
    print("OK — cache built, windows aligned, lyrics attached")
