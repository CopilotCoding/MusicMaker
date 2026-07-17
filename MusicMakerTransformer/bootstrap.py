"""
bootstrap.py — fetch a tiny public dataset so the pipeline can be proven today.

Writes data/raw/<stem>.wav + <stem>.txt pairs, which is the exact layout the
real library uses. Your own mp3/wav/flac + lyric .txt files drop into the same
folder later with no code change -- that is the point of the convention.

GTZAN: 30s clips, 10 genres, research-standard, freely available.

Two landmines found and handled here:
  - `marsyas/gtzan` is a SCRIPT dataset; datasets>=4 refuses to run it
    ("Dataset scripts are no longer supported"). `sanchit-gandhi/gtzan` is the
    same corpus as plain data files.
  - datasets>=4 decodes audio through torchcodec, which isn't installed. We
    don't need it: cast the column to Audio(decode=False) to get raw bytes and
    let our own ffmpeg path (audio_io) do the decoding it already does well.

LYRICS: GTZAN has none. That is fine -- the goal here is proving the
cross-attention path TRAINS, so we write genre/tempo metadata as pseudo-lyrics.
It carries real mutual information with the audio, so the smoke test is honest:
swap two tracks' .txt files and generated audio must change.
"""

import argparse
import io

import torch

import audio_io
from config import CFG, DATA_RAW

DATASET = "sanchit-gandhi/gtzan"


def _describe(genre: str, idx: int) -> str:
    """Pseudo-lyrics: real, per-track, correlated with the audio."""
    return (
        f"genre: {genre}\n"
        f"style: {genre} music\n"
        f"instrumental track, {genre} character\n"
        f"track id: {genre}.{idx:05d}"
    )


def fetch(n_tracks: int = 8, per_genre: int = 1, sr: int = None,
          channels: int = None) -> int:
    from datasets import Audio, load_dataset

    sr = sr or CFG.__class__.__dict__.get("_sr", None) or 32000
    channels = channels or (2 if CFG.stereo else 1)
    DATA_RAW.mkdir(parents=True, exist_ok=True)

    print(f"streaming {DATASET} -> {DATA_RAW}")
    ds = load_dataset(DATASET, split="train", streaming=True)
    # decode=False: hand us the original bytes; ffmpeg does the rest.
    ds = ds.cast_column("audio", Audio(decode=False))

    names = ds.features["genre"].names if hasattr(ds.features["genre"], "names") else None
    seen, written = {}, 0

    for ex in ds:
        if written >= n_tracks:
            break
        g = ex["genre"]
        genre = names[g] if names and isinstance(g, int) else str(g)
        if seen.get(genre, 0) >= per_genre:
            continue

        raw = ex["audio"]["bytes"]
        src = DATA_RAW / f"_tmp_{genre}_{seen.get(genre,0)}.wav"
        src.write_bytes(raw)
        try:
            # Normalize to the codec's rate/channels once, here, so the token
            # cache never re-does it and every track is uniform.
            wav = audio_io.load(src, sr, channels)
        finally:
            src.unlink(missing_ok=True)

        stem = f"{genre}_{seen.get(genre, 0):02d}"
        audio_io.save(wav, DATA_RAW / f"{stem}.wav", sr)      # levels preserved
        (DATA_RAW / f"{stem}.txt").write_text(
            _describe(genre, seen.get(genre, 0)), encoding="utf-8")

        seen[genre] = seen.get(genre, 0) + 1
        written += 1
        print(f"  {stem}.wav  {wav.shape[1]/sr:5.1f}s  {tuple(wav.shape)}  + .txt")

    print(f"wrote {written} track+lyric pairs to {DATA_RAW}")
    return written


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", type=int, default=8)
    ap.add_argument("--per-genre", type=int, default=1)
    a = ap.parse_args()

    from codec import Codec  # only to read sr/channels the codec expects
    import transformers
    transformers.logging.set_verbosity_error()

    n = fetch(a.tracks, a.per_genre, sr=32000, channels=2 if CFG.stereo else 1)
    if n == 0:
        raise SystemExit("no tracks written")
