r"""
plot.py — LIVE graph of checkpoints\metrics.csv. Real axes, real window.

    python plot.py                 # live window, redraws every 5s
    python plot.py --png           # write training.png instead (headless)
    python plot.py --last 500      # only the most recent 500 steps
    python plot.py --every 15      # redraw interval in seconds

Reads the CSV train.py writes every step. CPU only, touches nothing, safe to
run while training. Leave the window open next to the terminal.

WHAT TO LOOK FOR:
  val falling             -> healthy, ignore everything else
  val flat, train falling -> overfitting; best.pt already holds your model
  BOTH rising             -> diverging; lower the LR, nothing later improves
  gn spiky but val down    -> normal; that is what gradient clipping is for
  gn trending up + val up  -> the real failure
"""

import argparse
import csv
import time
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

import ui
from config import CKPT_DIR


def load(path, last=None):
    tr, va = [], []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            try:
                s = int(r["step"])
                tr.append((s, float(r["loss"]), float(r["lr"]),
                           float(r["grad_norm"])))
                if r.get("val_loss"):
                    va.append((s, float(r["val_loss"])))
            except (ValueError, KeyError):
                continue
    if last:
        tr = tr[-last:]
        lo = tr[0][0] if tr else 0
        va = [v for v in va if v[0] >= lo]
    return tr, va


def smooth(ys, k=21):
    """Rolling mean. Raw per-step loss is too noisy to read a trend from."""
    if len(ys) < k:
        return ys
    out, acc = [], 0.0
    from collections import deque
    win = deque()
    for y in ys:
        win.append(y); acc += y
        if len(win) > k:
            acc -= win.popleft()
        out.append(acc / len(win))
    return out


def make_fig():
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True,
                             gridspec_kw={"height_ratios": [3, 2, 1.4]})
    fig.canvas.manager.set_window_title("MusicMaker — training (live)")
    fig.patch.set_facecolor("#12141a")
    return fig, axes


def render(tr, va, fig, axes, vocab_ln=9.70):
    if len(tr) < 3:
        return False
    steps = [r[0] for r in tr]
    loss = [r[1] for r in tr]
    lrs = [r[2] for r in tr]
    gns = [r[3] for r in tr]

    for ax in axes:
        ax.clear()                      # redraw in place: one window, not N
        ax.set_facecolor("#171a21")
        ax.grid(alpha=0.18, color="#8892a4", linewidth=0.6)
        ax.tick_params(colors="#9aa4b5", labelsize=9)
        for sp in ax.spines.values():
            sp.set_color("#2c313c")

    # ---- loss: the whole story ----
    ax = axes[0]
    ax.plot(steps, loss, color="#3d5a80", lw=0.7, alpha=0.45, label="train (raw)")
    ax.plot(steps, smooth(loss), color="#5eb3f6", lw=1.9, label="train (smoothed)")
    if va:
        vs, vl = zip(*va)
        ax.plot(vs, vl, color="#ff7ab6", lw=2.0, marker="o", ms=3.5,
                label="val")
        bi = vl.index(min(vl))
        ax.plot(vs[bi], vl[bi], marker="*", ms=16, color="#7CFFB2",
                zorder=5, label=f"best val {vl[bi]:.3f} @ {vs[bi]}")
    ax.axhline(vocab_ln, color="#e05263", ls="--", lw=1.1, alpha=0.7)
    ax.annotate(f"random guessing (ln vocab = {vocab_ln:.2f})",
                xy=(steps[0], vocab_ln), xytext=(4, 4), textcoords="offset points",
                color="#e05263", fontsize=8)
    ax.set_ylabel("cross-entropy loss", color="#c8d0dd", fontsize=10)
    ax.set_title("MusicMaker training", color="#e8edf5", fontsize=13, pad=10)
    ax.legend(facecolor="#171a21", edgecolor="#2c313c", labelcolor="#c8d0dd",
              fontsize=9, loc="upper right")

    # ---- gradient norm: the instability tell ----
    ax = axes[1]
    ax.plot(steps, gns, color="#5a4a7a", lw=0.7, alpha=0.5)
    ax.plot(steps, smooth(gns), color="#c792ea", lw=1.9, label="grad norm")
    ax.axhline(1.0, color="#ffb454", ls="--", lw=1.1, alpha=0.8)
    ax.annotate("clip threshold 1.0", xy=(steps[0], 1.0), xytext=(4, 4),
                textcoords="offset points", color="#ffb454", fontsize=8)
    hi = max(gns)
    if hi > 6:                       # keep one spike from flattening the rest
        ax.set_ylim(0, min(hi * 1.1, 12))
    ax.set_ylabel("gradient norm", color="#c8d0dd", fontsize=10)
    ax.legend(facecolor="#171a21", edgecolor="#2c313c", labelcolor="#c8d0dd",
              fontsize=9, loc="upper left")

    # ---- lr schedule ----
    ax = axes[2]
    ax.plot(steps, lrs, color="#7CFFB2", lw=1.8)
    ax.set_ylabel("lr", color="#c8d0dd", fontsize=10)
    ax.set_xlabel("step", color="#c8d0dd", fontsize=10)

    fig.tight_layout()
    return True


def clipped_pct(gns, n=100):
    return 100 * sum(1 for g in gns[-n:] if g > 1.0) / min(n, len(gns))


def verdict(tr, va):
    """The read, in words, so the graph is not the only interpretation."""
    loss = [r[1] for r in tr]
    gns = [r[3] for r in tr]
    n = min(40, len(loss) // 2)
    if n < 5:
        return
    lt = (sum(loss[-n:]) / n) / max(sum(loss[-2*n:-n]) / n, 1e-9)
    gt = (sum(gns[-n:]) / n) / max(sum(gns[-2*n:-n]) / n, 1e-9)
    rows = [("train", f"{'[red]rising[/]' if lt > 1.01 else '[green]falling[/]' if lt < 0.99 else '[yellow]flat[/]'}  ({lt:.3f}x over last {n} steps)")]
    if va:
        vl = [v for _, v in va]
        best = min(vl)
        # Two signals, either is enough. The earlier version required 4 val
        # points past the turn before it would say anything -- so on a graph
        # that showed an unmistakable divergence it printed "healthy". The eye
        # saw it instantly; the heuristic needed more evidence than the run
        # would ever survive to produce.
        #   1. the LAST val is meaningfully worse than the best  -> turned
        #   2. best is not recent                                -> stalled
        # 1.01 was too loose: a val of 9.090 against a best of 9.000 is a
        # clear turn, and 1% of 9.0 is 0.09 -- exactly at the threshold, so it
        # read as "improving". Loss is already a log scale; any consistent
        # rise matters. Use a small absolute margin instead of a ratio.
        last_worse = vl[-1] > best + 0.02
        since_best = len(vl) - 1 - vl.index(best)
        v_rising = last_worse or since_best >= 3
        rows.append(("val", f"{'[red]past its best[/]' if v_rising else '[green]improving[/]'}"
                            f"  best {best:.3f}, last {vl[-1]:.3f}"
                            f"{f', {since_best} evals ago' if since_best else ''}"))
        if v_rising and lt > 1.0:
            rows.append(("verdict", "[bold red]DIVERGING — train AND val rising. "
                                    "Lower the LR; nothing later gets better.[/]"))
        elif v_rising:
            rows.append(("verdict", "[yellow]overfitting — best.pt has your model[/]"))
        elif gt > 1.5 and clipped_pct(gns) > 50:
            rows.append(("verdict", "[yellow]val still improving but gradients are "
                                    "clipping hard — watch it[/]"))
        else:
            rows.append(("verdict", "[bold green]healthy — let it run[/]"))
    rows.append(("grad norm", f"{gt:.2f}x trend, clipped on {clipped_pct(gns):.0f}% "
                              f"of last {min(100,len(gns))} steps"))
    ui.kv_table(rows, title="the read")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=str(CKPT_DIR / "metrics.csv"))
    ap.add_argument("--png", action="store_true",
                    help="write training.png once instead of a live window")
    ap.add_argument("--out", default="training.png")
    ap.add_argument("--last", type=int, default=None)
    ap.add_argument("--every", type=float, default=5.0,
                    help="redraw interval, seconds")
    a = ap.parse_args()

    p = Path(a.file)
    if not p.exists():
        raise SystemExit(f"no metrics at {p} — start train.py first")

    if a.png:
        matplotlib.use("Agg")
        tr, va = load(p, a.last)
        fig, axes = make_fig()
        if render(tr, va, fig, axes):
            fig.savefig(a.out, dpi=110, facecolor=fig.get_facecolor())
            ui.rule(f"steps {tr[0][0]}..{tr[-1][0]}  ->  {a.out}")
            verdict(tr, va)
        return

    # live window: redraw the SAME figure in place as the CSV grows
    plt.ion()
    fig, axes = make_fig()
    fig.show()
    last_n = -1
    ui.log(f"[dim]live plot — reading {p.name} every {a.every:.0f}s. "
           f"Close the window or Ctrl-C to stop.[/]")
    try:
        while plt.fignum_exists(fig.number):
            tr, va = load(p, a.last)
            if len(tr) != last_n:
                last_n = len(tr)
                render(tr, va, fig, axes)
                fig.canvas.draw_idle()
            fig.canvas.flush_events()
            plt.pause(a.every)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
