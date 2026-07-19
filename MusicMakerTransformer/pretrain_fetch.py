r"""
pretrain_fetch.py — stream the MTG-Jamendo moodtheme set into a token cache.

    python pretrain_fetch.py                  # full set: 100 shards, ~13h
    python pretrain_fetch.py --shards 0 4     # just shards 0-4 (test run)

WHAT IT DOES, per shard (100 tar shards, ~1.5GB each, ~152GB total wire):
    download (resumable, sha256-verified)  ->  unpack  ->  encode every mp3
    through EnCodec into data/cache_pretrain/*.npz (int16 codes ONLY — the
    pretrain is unconditional, no T5, which is what keeps the cache ~3GB)
    ->  delete the mp3s and the tar  ->  next shard.

Disk high-water is ~3 buffered shards (~5GB) + the growing cache. Nothing
else survives. MEASURED on this machine: single connection to the Finland
mirror is ~2.8MB/s but it scales linearly with connections (4 conns saturated
the line at 11.7MB/s), so shards download 3-at-a-time ahead of the encoder.
The GPU encoder (~2.8s/track) is the bottleneck: ~13h for the full set.

RESUMABLE at track granularity: the manifest is flushed after every track
(the same Ctrl-C lesson build_cache learned the hard way). Re-running skips
completed shards instantly and picks up mid-shard where it stopped.

License note: MTG-Jamendo audio is Creative Commons, restricted to
non-commercial research use — which is what this is.
"""

import argparse
import hashlib
import json
import queue
import subprocess
import tarfile
import threading
from pathlib import Path

import numpy as np
import torch

import ui
from config import ROOT

CACHE = ROOT / "data" / "cache_pretrain"
TMP = CACHE / "_shards"

MIRROR = "https://cdn.freesound.org/mtg-jamendo/autotagging_moodtheme/audio"
SHA_URL = ("https://raw.githubusercontent.com/MTG/mtg-jamendo-dataset/master/"
           "data/download/autotagging_moodtheme_audio_sha256_tars.txt")
N_SHARDS = 100


def shard_name(i: int) -> str:
    return f"autotagging_moodtheme_audio-{i:02d}.tar"


def fetch_checksums() -> dict:
    """{tar_name: sha256}. Cached locally so re-runs work offline."""
    local = CACHE / "sha256_tars.txt"
    if not local.exists():
        subprocess.run(["curl", "-sSf", "-o", str(local), SHA_URL], check=True)
    out = {}
    for line in local.read_text().splitlines():
        h, name = line.split()
        out[name] = h
    return out


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_shard(i: int, expect_sha: str) -> Path:
    """curl with resume + retry; verify sha256; one re-download on mismatch."""
    TMP.mkdir(parents=True, exist_ok=True)
    name = shard_name(i)
    tar_p = TMP / name
    for attempt in (1, 2):
        subprocess.run(
            ["curl", "-sS", "-C", "-", "--retry", "5", "--retry-delay", "5",
             "-o", str(tar_p), f"{MIRROR}/{name}"],
            check=True)
        if sha256_file(tar_p) == expect_sha:
            return tar_p
        tar_p.unlink()                     # corrupt: delete and re-pull once
        if attempt == 2:
            raise RuntimeError(f"{name}: sha256 mismatch twice — mirror issue?")
    raise AssertionError


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", type=int, nargs=2, default=[0, N_SHARDS - 1],
                    metavar=("FIRST", "LAST"),
                    help="inclusive shard range (default: all 100)")
    ap.add_argument("--dl-workers", type=int, default=3,
                    help="parallel shard downloads (3 saturates the line)")
    ap.add_argument("--buffer", type=int, default=3,
                    help="max downloaded-but-unencoded shards on disk")
    ap.add_argument("--codec", choices=["dac", "encodec"], default="dac",
                    help="dac is the project codec now (measured: 'basically "
                         "the original'); tokens are NOT interchangeable "
                         "between codecs")
    a = ap.parse_args()

    import transformers
    transformers.logging.set_verbosity_error()
    from dac_codec import make_codec

    CACHE.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    codec = make_codec(a.codec, dev)

    man_p = CACHE / "manifest.json"
    man = json.loads(man_p.read_text()) if man_p.exists() else {
        "sig": {"codec_id": codec.model_id, "n_q": codec.n_q,
                "n_slots": codec.n_slots, "vocab_size": codec.vocab_size,
                "sr": codec.sr},
        "shards_done": [], "tracks": {}, "hours": 0.0}
    # A cache built under a different codec config is silent poison — refuse.
    assert man["sig"]["vocab_size"] == codec.vocab_size, \
        "existing pretrain cache was built with a different codec config"

    shard_ids = [i for i in range(a.shards[0], a.shards[1] + 1)
                 if i not in man["shards_done"]]
    checks = fetch_checksums()
    ui.rule("MTG-Jamendo moodtheme -> token cache (unconditional)")
    ui.kv_table([
        ("shards", f"{len(shard_ids)} to do "
                   f"({len(man['shards_done'])} already done)"),
        ("cached so far", f"{len(man['tracks'])} tracks, {man['hours']:.1f}h"),
        ("cache dir", str(CACHE)),
        ("flow", f"{a.dl_workers} parallel downloads ahead of one GPU encoder; "
                 f"audio deleted after encoding"),
    ])
    if not shard_ids:
        ui.log("[green]nothing to do — all requested shards cached[/]")
        return

    # ---- downloader threads: keep the encoder fed, never > buffer on disk ---
    ready: "queue.Queue[tuple[int, Path]]" = queue.Queue()
    slots = threading.Semaphore(a.buffer)     # cap buffered shards
    dl_iter = iter(shard_ids)
    dl_lock = threading.Lock()
    stop = threading.Event()

    def downloader():
        while not stop.is_set():
            with dl_lock:
                try:
                    i = next(dl_iter)
                except StopIteration:
                    return
            slots.acquire()
            try:
                tar_p = download_shard(i, checks[shard_name(i)])
                ready.put((i, tar_p))
            except Exception as e:
                slots.release()
                ui.log(f"[red]shard {i:02d} download failed: {e}[/]")
                ready.put((i, None))

    threads = [threading.Thread(target=downloader, daemon=True)
               for _ in range(a.dl_workers)]
    for t in threads:
        t.start()

    # ---- encoder (main thread, owns the GPU) ----
    prog = ui.cache_progress()
    task = prog.add_task("shards", total=len(shard_ids), stats="") if prog else None
    if prog:
        prog.start()
    done_shards = 0
    try:
        for _ in range(len(shard_ids)):
            i, tar_p = ready.get()
            if tar_p is None:
                continue                      # failed shard: logged, skip
            ex_dir = TMP / f"x{i:02d}"
            with tarfile.open(tar_p) as tf:
                tf.extractall(ex_dir, filter="data")
            mp3s = sorted(ex_dir.rglob("*.mp3"))

            for j, mp3 in enumerate(mp3s):
                # per-TRACK progress: a shard is ~185 tracks / ~20 min of
                # encoding, and a bar that only moves per shard looks frozen
                # the whole time (measured: it did).
                if prog:
                    prog.update(task, completed=done_shards + j / max(1, len(mp3s)),
                                stats=f"shard {i:02d}: track {j+1}/{len(mp3s)} "
                                      f"| [bold]{man['hours']:.1f}h[/] cached")
                name = f"jam_{mp3.stem}"
                npz = CACHE / f"{name}.npz"
                if name in man["tracks"] and npz.exists():
                    continue                  # resumed mid-shard
                try:
                    flat = codec.encode_file(mp3)
                except Exception as e:
                    ui.log(f"[yellow]skip {mp3.name}: {e}[/]")
                    continue
                assert flat.shape[0] % codec.n_slots == 0
                assert int(flat.max()) < codec.vocab_size
                np.savez(npz, codes=flat.numpy().astype(np.int16))
                man["tracks"][name] = True
                man["hours"] += flat.shape[0] / codec.tokens_per_sec / 3600
                # flush EVERY track: Ctrl-C never loses more than one encode
                man_p.write_text(json.dumps(man))
                if dev == "cuda":
                    torch.cuda.empty_cache()

            # shard fully encoded: delete audio, free the buffer slot
            import shutil
            shutil.rmtree(ex_dir, ignore_errors=True)
            tar_p.unlink(missing_ok=True)
            man["shards_done"].append(i)
            man_p.write_text(json.dumps(man))
            slots.release()
            done_shards += 1
            if prog:
                prog.update(task, completed=done_shards,
                            stats=f"[bold]{man['hours']:.0f}h[/] cached, "
                                  f"{len(man['tracks'])} tracks")
    except KeyboardInterrupt:
        stop.set()
        ui.log(f"\n[yellow]interrupted[/] — {man['hours']:.1f}h cached across "
               f"{len(man['tracks'])} tracks. Re-run to resume; completed "
               "shards are skipped instantly.")
        return
    finally:
        if prog:
            prog.stop()

    ui.panel(f"[bold green]{man['hours']:.1f} hours[/] of audio cached as "
             f"{len(man['tracks'])} tracks in {CACHE}\n"
             f"[dim]cache size ~{sum(f.stat().st_size for f in CACHE.glob('*.npz'))/1e9:.2f}GB[/]",
             title="pretrain cache ready", style="green")


if __name__ == "__main__":
    main()
