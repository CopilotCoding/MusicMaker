"""
ui.py — shared Rich console helpers.

Every entry point here is a CLI, and the interesting output is either a live
training curve or a pass/fail wall of self-tests. Both read far better as
structured panels than as scrolling print().

Degrades gracefully: if rich isn't importable, everything falls back to plain
print rather than taking the project down with it.
"""

import sys

# Windows console defaults to cp1252, which cannot encode the Braille glyphs
# rich uses for spinners -- it raises UnicodeEncodeError mid-render and takes
# the training run down with it. Force UTF-8 on the streams before rich reads
# them. Must happen before Console() is constructed.
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):               # pragma: no cover
        pass

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import (BarColumn, Progress, SpinnerColumn, TextColumn,
                               TimeElapsedColumn, TimeRemainingColumn)
    from rich.table import Table
    _RICH = True
except ImportError:                                    # pragma: no cover
    _RICH = False

# legacy_windows=False opts out of the cp1252 render path even when rich
# detects an old console host.
console = Console(legacy_windows=False, soft_wrap=False) if _RICH else None

OK = "[bold green]PASS[/]"
BAD = "[bold red]FAIL[/]"


def log(msg=""):
    console.print(msg) if _RICH else print(msg)


def rule(title):
    console.rule(f"[bold cyan]{title}[/]") if _RICH else print(f"\n=== {title} ===")


def panel(body, title=None, style="cyan"):
    if _RICH:
        console.print(Panel(body, title=title, border_style=style, expand=False))
    else:
        if title:
            print(f"--- {title} ---")
        print(body)


def check(passed: bool, label: str, detail: str = ""):
    """One self-test line. Returns `passed` so callers can assert on it."""
    tag = (OK if passed else BAD) if _RICH else ("PASS" if passed else "FAIL")
    d = f"  [dim]{detail}[/]" if (_RICH and detail) else (f"  {detail}" if detail else "")
    log(f"{tag}  {label}{d}")
    return passed


def kv_table(rows, title=None):
    """rows: [(key, value)] -> aligned two-column table."""
    if not _RICH:
        if title:
            print(f"--- {title} ---")
        for k, v in rows:
            print(f"  {k:22s} {v}")
        return
    t = Table(show_header=False, box=None, padding=(0, 2), title=title,
              title_style="bold cyan", title_justify="left")
    t.add_column(style="dim", justify="right")
    t.add_column(style="bold")
    for k, v in rows:
        t.add_row(str(k), str(v))
    console.print(t)


def train_progress():
    """Live training bar: step count, rate, ETA."""
    if not _RICH:
        return None
    return Progress(
        SpinnerColumn(spinner_name="line"),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=None),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("[dim]|[/]"),
        TextColumn("{task.fields[stats]}"),
        TimeElapsedColumn(),
        TextColumn("[dim]eta[/]"),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    )


def gen_progress():
    if not _RICH:
        return None
    return Progress(
        SpinnerColumn(spinner_name="line"),
        TextColumn("[bold magenta]{task.description}"),
        BarColumn(bar_width=None),
        TextColumn("{task.completed}/{task.total} tok"),
        TextColumn("[dim]|[/]"),
        TextColumn("{task.fields[stats]}"),
        TimeElapsedColumn(),
        console=console,
    )


def cache_progress():
    """
    Encoding progress for the offline cache build.

    This is the slowest step in the whole project (EnCodec + T5 over every
    track) and it runs BEFORE the training UI appears -- without a bar it just
    looks frozen. Shows count, rate and ETA so a 20-minute encode is legible.
    """
    if not _RICH:
        return None
    return Progress(
        SpinnerColumn(spinner_name="line"),
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(bar_width=None),
        TextColumn("{task.completed}/{task.total}"),
        TextColumn("[dim]|[/]"),
        TextColumn("{task.fields[stats]}"),
        TimeElapsedColumn(),
        TextColumn("[dim]eta[/]"),
        TimeRemainingColumn(),
        console=console,
    )


def slot_table(names, values, title="per-slot loss"):
    """
    Per-slot CE. Coarse codebooks should sit well below fine ones; if every
    slot matches, the interleave is broken and this table is how you see it.
    """
    if not _RICH:
        print("   " + title + ": " + "  ".join(f"{n}={v:.2f}" for n, v in zip(names, values)))
        return
    t = Table(title=title, title_style="bold yellow", box=None, padding=(0, 1))
    for n in names:
        t.add_column(n, justify="right")
    lo, hi = min(values), max(values)
    cells = []
    for v in values:
        # green = learned most, red = closest to noise
        c = "green" if v <= lo + (hi - lo) * 0.34 else \
            "yellow" if v <= lo + (hi - lo) * 0.67 else "red"
        cells.append(f"[{c}]{v:.2f}[/]")
    t.add_row(*cells)
    console.print(t)
