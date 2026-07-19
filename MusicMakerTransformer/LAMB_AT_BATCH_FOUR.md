# Large Batch Optimization for Deep Learning, Except the Batch Is Four

*Training a music LM on one consumer GPU with the optimizer built for 32,768 of them.*

---

## Abstract

LAMB (You et al., 2019) was designed to scale BERT pretraining to batch sizes
of 32,768, for which it earned its title, "Training BERT in 76 Minutes." We
report its use at effective batch size **4** — a regime approximately 8,000×
smaller than intended, achieved not through ambition but through owning exactly
one RTX 5060 Ti. We find that LAMB's layer-wise trust ratio, designed to govern
step magnitudes when gradients are too *clean*, works equally well when
gradients are too *noisy*, because the trust ratio does not ask why a step
magnitude is untrustworthy before overruling it. Across four AdamW baseline
runs, every learning rate tried either diverged (3e-4 at step 350; 5e-5 at
step ~600, reproduced under two β₂ values) or stalled (1e-5, −0.007 loss per
50 steps, a slower way of achieving nothing). Under LAMB, three learning rates
spanning 33× (3e-5, 1e-4, 1e-3) all trained stably on the first attempt, with
pre-clip gradient norms flat at 0.3–0.7 where AdamW's had ramped 0.7 → 10.9.
We conclude that the optimizer market has an underserved customer segment at
the bottom, and that the trust ratio is a magnitude governor, not a batch-size
technology.

---

## 1. The Problem Nobody Writes Papers About

The large-batch literature exists because well-funded labs hit a wall going
*up*: past a few thousand samples per step, gradients become so precise that
the learning rate must grow to match, and Adam's per-parameter magnitudes go
wrong. LARS and LAMB fixed this with a per-layer **trust ratio**:

```
update = lr · (‖W‖ / ‖adam_update‖) · adam_update
```

Every layer moves by a fixed fraction of its own norm. Every step. No appeals.

We hit the same wall going *down*. At effective batch 4 (batch 2 × grad-accum
2 — the most a 16GB card allows at 41 seconds of audio context), AdamW's
second-moment estimate is built from almost no evidence. It drifts. The
failure is distinctive and cruel: **the run looks healthy for 400–600 steps,
then the gradient norm ramps and both losses turn upward.** It survives long
enough that you have hope. It reproduced at β₂ = 0.95 and β₂ = 0.999, ruling
out the moment decay as the cause, and an update/weight ratio measured
"textbook healthy" (1.18e-03) six steps from init diverged anyway at step 480
— the pathology takes hundreds of steps to accumulate and no short probe sees
it coming.

## 2. Results

| Optimizer | LR | Outcome |
|---|---|---|
| AdamW | 3e-4 | diverged, step 350 |
| AdamW | 5e-5 (β₂ 0.95) | gn 0.7 → 5.4, dead by step ~650 |
| AdamW | 5e-5 (β₂ 0.999) | diverged *earlier*, step ~480 |
| AdamW | 1e-5 | stable; also motionless (−0.007 / 50 steps) |
| **LAMB** | 3e-5 | stable, slow |
| **LAMB** | 1e-4 | stable; val 9.62 @ step 250, 9.17 @ 1000, 8.85 @ 1400 |
| **LAMB** | 1e-3 | stable; memorized one song to loss 0.005 |
| **LAMB** | 1e-3 (full corpus) | **val 9.86 → 7.74 in 800 steps**; a record at every one of 16 evals; stopped by hand at its floor |

The AdamW column is a bracketing proof that no learning rate exists for it in
this regime: every value is either too hot (dies late, expensively) or too
cold (lives pointlessly). The LAMB column spans 33× and contains no failures.
The measured update/weight ratio under LAMB pins to the learning rate itself
(3.1e-5 at lr 3e-5, held across 400 steps), which is the entire point: the
knob that killed four runs became a knob that merely sets the pace.

A gradient-norm-trend backoff was implemented as a safety net (halve the LR if
gn trends 1.5× over 40 steps above the clip threshold). Over the full-corpus
run it fired **four times** — steps 336, 478, 596, 794 — walking the LR down
1e-3 → 5e-4 → 2.5e-4 → 1.25e-4 → 6.2e-5 as the loss surface sharpened. Every
fire occurred while validation loss was setting a record low, and the first
one landed within spitting distance of the LR the project's human had
predicted as optimal an hour earlier by looking at a graph. The authors
decline to adjudicate who gets credit. The net effect was an *empirically
driven annealing schedule*: the cosine decay anneals on a timetable; the
backoff annealed on the terrain.

One design flaw surfaced in production: the trigger is purely *relative*
(recent 40-mean > 1.5× the previous 40-mean), so each fire raises the baseline
and partially deafens the next — by the end, gradient-norm spikes of 15.6 sat
below the ~15 mean the fourth-fired baseline demanded, and detection lagged
the spikes by ~40 steps. Under LAMB this was harmless (the trust ratio pins
the step size regardless of gn), but a future version should OR-in an
absolute threshold.

## 3. Why This Works, Probably

Both ends of the batch spectrum break the same invariant: *the raw update
magnitude stops being trustworthy.* At batch 32k it is untrustworthy because
gradients are cleaner than Adam's calibration assumes. At batch 4 it is
untrustworthy because the variance estimate is itself a random number that
wanders over hundreds of steps. The trust ratio renormalizes the magnitude
every step against the one quantity that is always well-defined — the layer's
own norm — and is therefore indifferent to *which* failure produced the bad
magnitude. AdamW's comfort zone is the middle of the spectrum, where noise
averages out and nothing needs governing. One consumer GPU at long context is
simply not in the middle, and never will be.

## 4. Conclusion

We used the flagship large-batch optimizer at the smallest effective batch we
have seen reported, on the grounds that it was the only thing that worked. It
worked. The authors of LAMB intended to democratize training by making it fit
into 76 minutes on a TPU pod; they appear to have also democratized it in the
other direction, for people with one graphics card and 18.7 hours of grunge.

*No hyperparameters were tuned in the making of this optimizer swap, which
was, in fact, the point.*

## Postscript (final)

The full-corpus run at lr 1e-3 — 3.3× above AdamW's *instant-death* LR — ran
800 steps (~2.5 hours) and was stopped by hand at its floor. Validation loss
went **9.857 → 7.737**, setting a new record at all sixteen evaluations
without exception. It crossed the uniform-guess line (ln 16,386 = 9.70)
during warmup, before the learning rate had finished ramping; it passed the
previous run's 1400-step best (8.85) at step 400. Mid-run, the model found
the residual-codebook structure (slot q1L collapsed 9.1 → 7.0 while gradient
norms climbed from 0.6 to means of 11 with spikes to 15.6), and the backoff
annealed the LR 16× in response while the descent continued uninterrupted —
the failure mode that killed four AdamW runs, occurring under LAMB as a
non-event with paperwork.

The floor itself (~7.7) is a data ceiling, not an optimizer failure: 18.7
hours of audio contains what it contains. The optimizer designed for 32,768
samples per step was, in the end, never the problem — and at batch four,
never troubled.

## Post-postscript: the schedule that schedules itself

Having established that the learning rate barely matters at the start, we
then removed the human from choosing it at all. A two-rule controller on the
memorization harness — *pass-mean improved: lr ×1.05; didn't: lr ×0.7* —
was set loose at batch one, LAMB underneath. It probed from 1e-3 up to
2.65e-3 (a rate no hand would dare), detonated a pass (loss 18.1, one chunk
knocked from ~98% to 11%), backed itself off through its own crater,
recovered everything, and then annealed 1e-3 → 4e-6 with no schedule,
no warmup, and no cosine — because at convergence, passes stop improving *by
definition*, so the back-off rule **is** the anneal. It reached a 100.0%
free-run reproduction in 4,165 steps; the hand-tuned constant-LR-plus-
consolidation baseline needed ~5,240. The controller's only inputs are "did
that help" and "was that a disaster," which we note is also the complete
methodology of this paper.

---

*Method details: 138M-parameter decoder-only transformer (d=768, 12L, 12H,
RoPE, RMSNorm), EnCodec-32kHz token stream (8 interleaved slots, vocab 16,386),
seq_len 16,384 (~41s stereo audio), bf16, gradient checkpointing, LAMB
β=(0.9, 0.999), wd 0.1, 200-step warmup, cosine decay to 0.1×, clip 1.0.*
