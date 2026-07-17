"""
ddata.py — scan data/raw, cache continuous latents, serve fixed-length windows.

Same folder convention as the AR project (track.mp3 + track.txt sidecar), so
BOTH projects can point --music-dir at the SAME library with no copying. That
is what makes the comparison fair. Only the cache differs: latents (float16)
instead of token ids.

THE SCALE FACTOR IS MEASURED HERE. Diffusion assumes ~unit-variance data, and
raw EnCodec latents measured std ~4.6-5.4 on music (data-dependent -- ~3.0 on
noise). We compute it over the real corpus, store it in the manifest, and reuse
it at generation. Getting this wrong does not raise; it just quietly breaks the
noise schedule so the model trains against the wrong signal-to-noise curve.

FIXED-LENGTH WINDOWS: unlike the AR project there is no slot alignment to worry
about (no interleave), but every window MUST be exactly n_frames -- diffusion
operates on a fixed-shape tensor. Short tracks are skipped, not padded: padding
would teach the model that silence follows every song.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

import ui
from dconfig import DATA_CACHE, DATA_RAW, DCFG

AUDIO_EXT = (".mp3", ".wav", ".flac")


def scan(raw_dir=None, recursive: bool = True):
    """
    [(audio_path, lyric_path|None)] paired by sidecar: song.mp3 -> song.txt in
    the same folder. RECURSIVE -- a real library is Artist/Album/track.mp3.

    Pairs by FULL PATH, never bare stem: every album has an "01 - Intro", and
    stem-keyed pairing would silently cross-match them across albums.
    """
    raw_dir = Path(raw_dir) if raw_dir else DATA_RAW
    if not raw_dir.exists():
        return []
    it = raw_dir.rglob("*") if recursive else raw_dir.iterdir()
    audio = sorted(p for p in it if p.suffix.lower() in AUDIO_EXT)
    out = []
    for p in audio:
        txt = p.with_suffix(".txt")
        out.append((p, txt if txt.exists() else None))
    return out


def cache_name(audio_path, root=None) -> str:
    """Collision-proof cache filename: stem + hash of the full path."""
    p = Path(audio_path).resolve()
    h = hashlib.md5(str(p).encode()).hexdigest()[:8]
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in p.stem)[:60]
    return f"{safe}_{h}"


def _sig(codec, textenc) -> dict:
    return {
        "codec_id": codec.model_id, "latent_dim": codec.latent_dim,
        "channels": codec.channels, "sr": codec.sr,
        "frame_rate": codec.frame_rate,
        "t5_id": textenc.model_id if textenc else None,
        "text_max_len": textenc.max_len if textenc else None,
    }


def _file_key(p) -> str:
    st = p.stat()
    return hashlib.md5(f"{p.name}:{st.st_size}:{int(st.st_mtime)}".encode()).hexdigest()[:16]


def build_latent_cache(codec, textenc=None, raw_dir=None, force=False,
                       cache_dir=None, quiet=False):
    """
    Encode every track to latents once. Returns (paths, scale).

    raw_dir may be ANY folder -- point it at your music library and nothing is
    copied. Only the latent cache lives in this project.

    Encodes UNSCALED, then measures the corpus RMS and stores the scale in the
    manifest, so the scale can be recomputed without re-encoding.
    """
    cache = Path(cache_dir) if cache_dir else DATA_CACHE
    cache.mkdir(parents=True, exist_ok=True)
    pairs = scan(raw_dir)
    if not pairs:
        raise SystemExit(f"no audio in {raw_dir or DATA_RAW} — "
                         "pass --music-dir, or run bootstrap.py")

    sig = _sig(codec, textenc)
    sig["raw_dir"] = str(Path(raw_dir).resolve()) if raw_dir else None
    man_path = cache / "manifest.json"
    man, scale = {}, None
    if man_path.exists() and not force:
        old = json.loads(man_path.read_text())
        if old.get("sig") == sig:
            man = old.get("files", {})
            scale = old.get("scale")
        else:
            print("  cache signature changed — rebuilding")

    # Work out what needs encoding BEFORE starting, so the bar has a real
    # total and a resumed run says "12 to go", not "256".
    out, todo, fresh = [], [], False
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

    prog = ui.cache_progress() if (todo and not quiet) else None
    task = prog.add_task("encoding", total=len(todo), stats="") if prog else None
    if prog:
        prog.start()
    try:
        for i, (audio_p, txt_p, name, npz) in enumerate(todo):
            codec.scale = None                    # cache UNSCALED latents
            z = codec.encode_file(audio_p)        # (C, 128, T)
            payload = {"latent": z.numpy().astype(np.float16)}
            if txt_p and textenc:
                text = txt_p.read_text(encoding="utf-8", errors="replace").strip()
                if text:
                    ctx, msk = textenc.encode([text])
                    payload["ctx"] = ctx[0].cpu().numpy().astype(np.float16)
                    payload["ctx_mask"] = msk[0].cpu().numpy()
            np.savez_compressed(npz, **payload)
            man[name] = _file_key(audio_p)
            out.append(npz)
            fresh = True
            # Flush as we go: writing the manifest only at the end means a
            # Ctrl-C loses the record of everything already encoded.
            man_path.write_text(json.dumps(
                {"sig": sig, "files": man, "scale": scale}, indent=2))
            if prog:
                secs = z.shape[-1] / codec.frame_rate
                prog.update(task, completed=i + 1,
                            stats=f"[dim]{audio_p.name[:30]}[/] {secs:.0f}s")
    finally:
        if prog:
            prog.stop()

    # Measure the scale factor over the real corpus. Diffusion needs ~unit
    # variance; hardcoding this would silently break the schedule on new data.
    if scale is None or fresh:
        acc, n = 0.0, 0
        for p in out:
            z = np.load(p)["latent"].astype(np.float32)
            acc += float((z ** 2).sum()); n += z.size
        rms = (acc / max(n, 1)) ** 0.5
        scale = 1.0 / max(rms, 1e-8)
        print(f"  measured latent RMS {rms:.4f} over {n/1e6:.1f}M values "
              f"-> scale {scale:.4f}")

    man_path.write_text(json.dumps(
        {"sig": sig, "files": man, "scale": scale}, indent=2))
    return out, scale


class LatentDataset(Dataset):
    """Fixed-length latent windows. Each item: (C, latent_dim, n_frames)."""

    def __init__(self, cache_paths, n_frames, scale, random_crop=True):
        self.n_frames = n_frames
        self.scale = scale
        self.random_crop = random_crop
        self.tracks, self.index = [], []
        for p in cache_paths:
            z = np.load(p)
            lat = torch.from_numpy(z["latent"].astype(np.float32))
            if lat.shape[-1] < n_frames:
                continue                      # skip, never pad -- see module docstring
            ctx = torch.from_numpy(z["ctx"].astype(np.float32)) if "ctx" in z else None
            msk = torch.from_numpy(z["ctx_mask"]) if "ctx_mask" in z else None
            ti = len(self.tracks)
            self.tracks.append({"latent": lat, "ctx": ctx, "mask": msk, "name": p.stem})
            n_win = max(1, lat.shape[-1] // n_frames)
            self.index += [(ti, w * n_frames) for w in range(n_win)]
        if not self.tracks:
            raise SystemExit(f"no track has >= {n_frames} frames "
                             f"({n_frames/DCFG.frame_rate:.1f}s) — lower n_frames")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        ti, start = self.index[i]
        tr = self.tracks[ti]
        lat = tr["latent"]
        hi = lat.shape[-1] - self.n_frames
        if self.random_crop and hi > 0:
            start = int(torch.randint(0, hi + 1, (1,)).item())
        start = min(start, max(0, hi))
        z = lat[..., start:start + self.n_frames] * self.scale
        return {"latent": z, "ctx": tr["ctx"], "mask": tr["mask"], "name": tr["name"]}


def collate(batch):
    """
    Stack windows; pad text ctx to batch max.

    Same NaN guard as the AR project: a fully-masked cross-attn row makes SDPA
    emit NaN, so a track without text keeps one visible zero key rather than an
    all-False mask.
    """
    z = torch.stack([b["latent"] for b in batch])          # (B, C, D, T)
    B, C, D, T = z.shape
    z = z.view(B * C, D, T)                                # channels -> batch

    if all(b["ctx"] is None for b in batch):
        return {"z": z, "ctx": None, "ctx_mask": None,
                "names": [b["name"] for b in batch], "n_ch": C}

    d = next(b["ctx"].shape[-1] for b in batch if b["ctx"] is not None)
    L = max((b["ctx"].shape[0] for b in batch if b["ctx"] is not None), default=1)
    ctx = torch.zeros(B, L, d)
    msk = torch.zeros(B, L, dtype=torch.bool)
    for i, b in enumerate(batch):
        if b["ctx"] is not None:
            n = b["ctx"].shape[0]
            ctx[i, :n] = b["ctx"]
            msk[i, :n] = b["mask"] if b["mask"] is not None else True
        else:
            msk[i, 0] = True                               # never all-False
    # each channel is its own batch row, so text repeats per channel
    ctx = ctx.repeat_interleave(C, dim=0)
    msk = msk.repeat_interleave(C, dim=0)
    return {"z": z, "ctx": ctx, "ctx_mask": msk,
            "names": [b["name"] for b in batch], "n_ch": C}


if __name__ == "__main__":
    import argparse
    import transformers
    transformers.logging.set_verbosity_error()
    import ui
    from dcodec import LatentCodec
    from textenc import LyricEncoder

    _ap = argparse.ArgumentParser()
    _ap.add_argument("--music-dir", default=None,
                     help="scan ANY folder (recursive) instead of data/raw")
    _a = _ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    codec = LatentCodec(device=dev)
    te = LyricEncoder(device=dev)
    ui.rule("ddata — latent cache")

    paths, scale = build_latent_cache(codec, te, raw_dir=_a.music_dir)
    codec.scale = scale
    ds = LatentDataset(paths, DCFG.n_frames, scale)
    ui.kv_table([
        ("tracks", f"{len(ds.tracks)} cached, {len(paths)} files"),
        ("windows", f"{len(ds)} x {DCFG.n_frames} frames "
                    f"({DCFG.n_frames/DCFG.frame_rate:.2f}s each)"),
        ("scale", f"{scale:.4f}  [dim](measured from corpus)[/]"),
    ])

    from torch.utils.data import DataLoader
    dl = DataLoader(ds, batch_size=2, shuffle=True, collate_fn=collate)
    b = next(iter(dl))
    ui.log(f"batch: z {tuple(b['z'].shape)} [dim](B*C rows)[/] "
           f"ctx {tuple(b['ctx'].shape)}")

    # The scale factor's whole job: make the data ~unit variance so the noise
    # schedule is meaningful. If this drifts far from 1.0, diffusion trains
    # against the wrong signal-to-noise curve and nobody tells you.
    s = b["z"].std().item()
    ui.check(0.5 < s < 2.0, "scaled latents are ~unit variance",
             f"batch std {s:.3f} — the noise schedule assumes this")
    ui.check(b["ctx_mask"].any(1).all().item(), "no all-masked text row",
             "would NaN the cross-attention")
    ui.check(b["z"].shape[-1] == DCFG.n_frames, "fixed-length windows",
             f"diffusion needs a fixed shape, got {b['z'].shape[-1]}")
    ui.panel("latent cache verified", style="green")
