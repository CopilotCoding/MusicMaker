"""
tag.py — read embedded metadata from audio files, write conditioning .txt files.

    python tag.py                      # dry run: show what WOULD be written
    python tag.py --write              # actually write the .txt files
    python tag.py --write --tempo      # also estimate BPM (slower)
    python tag.py --audit              # just report what's learnable

Your ripper already wrote artist/album/title/year into the files. ffprobe reads
those uniformly across flac/mp3/wav, so there is nothing to type by hand.

EVERYTHING GETS WRITTEN. This tool reports on your tags; it does not throw any
away. --audit is information to act on however you like, not a filter.

WHAT THE AUDIT MEANS
--------------------
Cross-attention learns correlations that RECUR across the dataset. A tag with
the same value on every track is unlikely to condition much on its own -- there
is no contrast to learn from. A tag unique to every track tends toward
memorization rather than generalization (song titles are the classic case).
Tags that vary but repeat -- artist, year -- are where conditioning usually
bites.

"Unlikely" is not "never", which is exactly why nothing is dropped by default.
A constant tag still anchors the whole corpus in T5's semantic space, and if
you later add a non-grunge record, "genre: grunge" retroactively becomes the
contrast that makes the distinction learnable. Discarding it now would cost you
that. --drop-constant exists if you ever want the old behavior.
"""

import argparse
import json
import subprocess
from collections import Counter, defaultdict

import ui
from dconfig import DATA_RAW

AUDIO_EXT = (".mp3", ".wav", ".flac")

# Tags worth conditioning on, in the order they'll appear in the .txt.
# "year" not "date" -- read_tags renames it. title/track deliberately excluded:
# unique per track means the model memorizes text->track instead of learning.
USE_TAGS = ["artist", "album", "year", "genre"]
RENAME = {"date": "year", "album_artist": "artist"}


def read_tags(path) -> dict:
    """
    Embedded metadata via ffprobe. Same call works for flac/mp3/wav.

    Bytes in, UTF-8 out -- NOT text=True. subprocess's text mode decodes using
    the locale encoding, which on Windows is cp1252, and real music tags are
    full of characters cp1252 cannot represent: accented artist names, smart
    quotes, en-dashes. ffprobe always emits UTF-8, so decode it as UTF-8.
    errors="replace" means one exotic tag degrades to a funny character instead
    of taking down a 263-file scan.
    """
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format_tags",
         "-of", "json", str(path)],
        capture_output=True,
    ).stdout
    if not out:
        return {}
    try:
        raw = json.loads(out.decode("utf-8", "replace")) \
            .get("format", {}).get("tags", {}) or {}
    except (json.JSONDecodeError, UnicodeDecodeError):
        return {}
    # tag case varies by format and ripper (TITLE vs title vs Title)
    tags = {}
    for k, v in raw.items():
        k = k.lower().strip()
        if k in ("encoder", "comment"):
            continue
        v = str(v).strip()
        if v:
            tags[RENAME.get(k, k)] = v
    if "year" in tags:                       # "1994-09-13" -> "1994"
        tags["year"] = tags["year"][:4]
    return tags


def estimate_tempo(path, sr=22050) -> str | None:
    """Coarse BPM bucket. Buckets, not exact BPM: 'mid' recurs, '112' doesn't."""
    try:
        import numpy as np
        import audio_io
        w = audio_io.load(path, sr, 1)[0].numpy()
    except Exception:
        return None
    # onset strength via spectral flux, then autocorrelation peak in 60-180bpm
    import numpy as np
    hop = 512
    n = (len(w) - 1024) // hop
    if n < 100:
        return None
    spec = np.abs(np.fft.rfft(
        np.lib.stride_tricks.as_strided(
            w, (n, 1024), (w.strides[0] * hop, w.strides[0])) * np.hanning(1024),
        axis=1))
    flux = np.maximum(0, np.diff(spec, axis=0)).sum(1)
    flux = (flux - flux.mean()) / (flux.std() + 1e-9)
    ac = np.correlate(flux, flux, "full")[len(flux) - 1:]
    fps = sr / hop
    lo, hi = int(fps * 60 / 180), int(fps * 60 / 60)     # 180bpm .. 60bpm
    if hi >= len(ac):
        return None
    bpm = 60 * fps / (lo + int(np.argmax(ac[lo:hi])))
    return "slow" if bpm < 90 else "mid" if bpm < 130 else "fast"


def to_text(tags: dict, tempo: str = None, drop: set = ()) -> str:
    """`drop` is empty unless --drop-constant is passed: by default every tag
    the file carries gets written."""
    lines = [f"{k}: {tags[k].lower()}" for k in USE_TAGS
             if k in tags and k not in drop]
    if tempo:
        lines.append(f"tempo: {tempo}")
    return "\n".join(lines)


def constant_tags(rows) -> set:
    """Tags with the same value on every track that has them."""
    vals = defaultdict(set)
    for _, tags, _ in rows:
        for k, v in tags.items():
            vals[k].add(v.lower())
    return {k for k, s in vals.items() if len(s) == 1 and len(rows) > 1}


def audit(rows):
    """
    Report, per tag: how many tracks have it, how many distinct values, and
    whether it can actually teach the model anything.
    """
    n = len(rows)
    vals = defaultdict(list)
    for _, tags, _ in rows:
        for k, v in tags.items():
            vals[k].append(v.lower())

    table = []
    for k in sorted(vals, key=lambda k: -len(vals[k])):
        c = Counter(vals[k])
        cov, uniq = len(vals[k]), len(c)
        avg = cov / uniq
        if uniq == 1:
            note = "[dim]constant — no contrast to learn from yet[/]"
        elif uniq == cov and cov > 3:
            note = "[dim]unique per track — tends toward memorization[/]"
        else:
            note = (f"[green]{uniq} values, ~{avg:.1f} tracks each[/]" if avg >= 2
                    else f"[yellow]{uniq} values, only ~{avg:.1f} tracks each[/]")
        top = ", ".join(f"{v}({n_})" for v, n_ in c.most_common(3))
        table.append((k, f"{cov}/{n} tracks  {note}\n   [dim]{top}[/]"))
    ui.kv_table(table, title="tag distribution across your corpus")

    strong = [k for k in vals
              if 1 < len(set(vals[k])) and len(vals[k]) / len(set(vals[k])) >= 2]
    if strong:
        ui.log(f"\n[bold green]Strongest conditioning signal:[/] {', '.join(strong)} "
               "[dim]— these vary across tracks and repeat.[/]")
    else:
        ui.log("\n[yellow]No tag both varies and repeats yet.[/] [dim]Conditioning "
               "will lean on memorization until the corpus grows. Everything is "
               "still written — more tracks may make these learnable.[/]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", "--music-dir", dest="dir", default=str(DATA_RAW),
                    help="your music folder; scanned recursively")
    ap.add_argument("--write", action="store_true", help="write .txt (default: dry run)")
    ap.add_argument("--tempo", action="store_true", help="estimate BPM bucket (slow)")
    ap.add_argument("--audit", action="store_true", help="only report learnability")
    ap.add_argument("--overwrite", action="store_true", help="replace existing .txt")
    ap.add_argument("--drop-constant", action="store_true",
                    help="omit tags identical on every track (default: keep everything)")
    a = ap.parse_args()

    from pathlib import Path
    d = Path(a.dir)
    # rglob: a real library is Artist/Album/track.mp3
    files = sorted(p for p in d.rglob("*") if p.suffix.lower() in AUDIO_EXT) \
        if d.exists() else []
    if not files:
        raise SystemExit(f"no audio in {d}")

    ui.rule(f"tag — {len(files)} files in {d}")
    rows = []
    for p in files:
        tags = read_tags(p)
        tempo = estimate_tempo(p) if (a.tempo and not a.audit) else None
        rows.append((p, tags, tempo))

    untagged = [p.name for p, t, _ in rows if not t]
    if untagged:
        ui.log(f"[yellow]{len(untagged)} file(s) have no metadata[/] — "
               f"e.g. {', '.join(untagged[:3])}")
        ui.log("[dim]  a ripper usually embeds these; otherwise write the .txt by hand[/]")

    audit(rows)
    if a.audit:
        return

    # Everything the file carries gets written. The audit above tells you what
    # is likely to condition well, but that is information, not a decision --
    # tags are the user's data and this tool does not silently discard them.
    drop = constant_tags(rows) if a.drop_constant else set()
    if drop:
        ui.log(f"\n[yellow]--drop-constant:[/] omitting {', '.join(sorted(drop))} "
               f"[dim](identical on every track)[/]")

    ui.log("")
    wrote = skipped = 0
    for p, tags, tempo in rows:
        txt = p.with_suffix(".txt")
        body = to_text(tags, tempo, drop)
        if not body:
            continue
        if txt.exists() and not a.overwrite:
            skipped += 1
            continue
        if a.write:
            txt.write_text(body, encoding="utf-8")
            wrote += 1
        ui.log(f"[dim]{txt.name}[/]\n{body}\n")

    if a.write:
        ui.panel(f"wrote [bold]{wrote}[/] .txt files"
                 + (f"\nskipped {skipped} existing (use --overwrite)" if skipped else ""),
                 title="done", style="green")
    else:
        ui.panel("dry run — nothing written\nre-run with [bold]--write[/] to create these",
                 title="preview", style="yellow")


if __name__ == "__main__":
    main()
