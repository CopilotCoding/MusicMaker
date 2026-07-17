"""
diffusion.py — the noise schedule, forward process, and DDIM sampler.

This file is the actual "diffusion" part; dmodel.py is just a denoiser.

FORWARD (training): take clean latents z0, pick a random timestep t, mix in
Gaussian noise according to the schedule, and ask the model to recover what was
added. One forward pass per training step -- no sequential unrolling.

    z_t = sqrt(a_bar_t) * z0 + sqrt(1 - a_bar_t) * noise

REVERSE (sampling): start from pure noise and walk t from 999 -> 0, subtracting
the model's predicted noise a bit at a time. ~50 DDIM steps for the WHOLE clip,
regardless of length -- versus the AR model's 400 steps per second of audio.

V-PREDICTION over eps-prediction. At t near 999 the input is almost pure noise,
so "predict the noise" is trivial (just echo the input) and the loss signal
collapses exactly where global musical structure gets decided. v-prediction
    v = sqrt(a_bar) * noise - sqrt(1 - a_bar) * z0
stays well-conditioned across the entire schedule. This is the single most
important choice in this file.

COSINE SCHEDULE over linear: linear spends too many steps at high noise where
nothing is learnable. Cosine allocates more resolution to the mid/low-noise
range where the actual content appears.
"""

import math

import torch


class Diffusion:
    def __init__(self, timesteps=1000, schedule="cosine", prediction="v",
                 device="cuda"):
        self.T = timesteps
        self.prediction = prediction
        self.device = device

        if schedule == "cosine":
            # Nichol & Dhariwal: a_bar(t) = cos^2((t/T + s)/(1+s) * pi/2)
            s = 0.008
            x = torch.linspace(0, timesteps, timesteps + 1, device=device)
            ab = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
            ab = ab / ab[0]
            betas = (1 - (ab[1:] / ab[:-1])).clamp(1e-8, 0.999)
        elif schedule == "linear":
            betas = torch.linspace(1e-4, 0.02, timesteps, device=device)
        else:
            raise ValueError(schedule)

        self.betas = betas
        self.alphas = 1.0 - betas
        self.a_bar = torch.cumprod(self.alphas, dim=0)
        self.sqrt_ab = self.a_bar.sqrt()
        self.sqrt_1mab = (1 - self.a_bar).sqrt()

    def _g(self, arr, t, ndim):
        """Gather per-sample schedule values and broadcast to (B,1,1)."""
        return arr[t].view(-1, *([1] * (ndim - 1)))

    # ---- forward (training) ---------------------------------------------

    def q_sample(self, z0, t, noise=None):
        """Add noise: z0 -> z_t. This IS the forward process."""
        noise = torch.randn_like(z0) if noise is None else noise
        return (self._g(self.sqrt_ab, t, z0.dim()) * z0
                + self._g(self.sqrt_1mab, t, z0.dim()) * noise)

    def target(self, z0, noise, t):
        """What the model should predict at timestep t."""
        if self.prediction == "eps":
            return noise
        # v-prediction
        return (self._g(self.sqrt_ab, t, z0.dim()) * noise
                - self._g(self.sqrt_1mab, t, z0.dim()) * z0)

    def loss(self, model, z0, ctx=None, ctx_mask=None):
        """Sample t, noise z0, predict, MSE. One forward pass -- that's it."""
        B = z0.shape[0]
        t = torch.randint(0, self.T, (B,), device=z0.device)
        noise = torch.randn_like(z0)
        z_t = self.q_sample(z0, t, noise)
        pred = model(z_t, t, ctx, ctx_mask)
        return torch.nn.functional.mse_loss(pred, self.target(z0, noise, t)), t

    # ---- reverse (sampling) ---------------------------------------------

    def _to_z0_eps(self, z_t, pred, t):
        """Convert the model's output to (z0_hat, eps_hat) regardless of param."""
        sab = self._g(self.sqrt_ab, t, z_t.dim())
        s1 = self._g(self.sqrt_1mab, t, z_t.dim())
        if self.prediction == "eps":
            eps = pred
            z0 = (z_t - s1 * eps) / sab.clamp(min=1e-8)
        else:
            z0 = sab * z_t - s1 * pred
            eps = (z_t - sab * z0) / s1.clamp(min=1e-8)
        return z0, eps

    @torch.no_grad()
    def sample(self, model, shape, ctx=None, ctx_mask=None, steps=50, eta=0.0,
               guidance=1.0, device=None, progress=None):
        """
        DDIM sampling. Pure noise -> clean latents in `steps` steps.

        guidance > 1 runs classifier-free guidance: one conditional and one
        unconditional pass per step, pushed apart. Same mechanism as the AR
        model's CFG, and it relies on the same ctx=None runtime gate.
        """
        device = device or self.device
        z = torch.randn(shape, device=device)
        ts = torch.linspace(self.T - 1, 0, steps, device=device).long()
        use_cfg = guidance != 1.0 and ctx is not None

        for i, t in enumerate(ts):
            tb = t.repeat(shape[0])
            pred = model(z, tb, ctx, ctx_mask)
            if use_cfg:
                un = model(z, tb, None, None)
                pred = un + guidance * (pred - un)

            z0, eps = self._to_z0_eps(z, pred, tb)
            t_prev = ts[i + 1] if i + 1 < len(ts) else None
            ab_prev = (self.a_bar[t_prev] if t_prev is not None
                       else torch.tensor(1.0, device=device))

            # DDIM update. eta=0 -> deterministic; eta=1 -> DDPM-like.
            sigma = 0.0
            if eta > 0 and t_prev is not None:
                ab_t = self.a_bar[t]
                sigma = eta * (((1 - ab_prev) / (1 - ab_t))
                               * (1 - ab_t / ab_prev)).sqrt()
            dir_zt = (1 - ab_prev - sigma ** 2).clamp(min=0).sqrt() * eps
            z = ab_prev.sqrt() * z0 + dir_zt
            if eta > 0 and t_prev is not None:
                z = z + sigma * torch.randn_like(z)
            if progress:
                progress(i + 1, len(ts))
        return z


if __name__ == "__main__":
    import ui
    from dconfig import DCFG
    from dmodel import DiT

    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ui.rule("diffusion — process tests")

    for pred in ("v", "eps"):
        d = Diffusion(DCFG.timesteps, DCFG.schedule, pred, dev)

        # 1. schedule sanity: a_bar must run 1 -> 0 monotonically. If it does
        #    not, the model is asked to denoise a signal that was never noised.
        ok = (d.a_bar[0] > 0.99 and d.a_bar[-1] < 0.02
              and (d.a_bar.diff() <= 0).all())
        ui.check(ok, f"[{pred}] cosine schedule monotonic 1->0",
                 f"a_bar: {d.a_bar[0]:.4f} -> {d.a_bar[-1]:.6f}")

        # 2. q_sample at t=0 is nearly clean; at t=T-1 nearly pure noise.
        z0 = torch.randn(4, 128, 64, device=dev)
        near = d.q_sample(z0, torch.zeros(4, dtype=torch.long, device=dev))
        far = d.q_sample(z0, torch.full((4,), d.T - 1, dtype=torch.long, device=dev))
        c0 = torch.nn.functional.cosine_similarity(
            near.flatten(1), z0.flatten(1), dim=1).mean().item()
        c1 = torch.nn.functional.cosine_similarity(
            far.flatten(1), z0.flatten(1), dim=1).mean().item()
        ui.check(c0 > 0.99 and abs(c1) < 0.2, f"[{pred}] forward process",
                 f"cos(z_t, z0): t=0 -> {c0:.3f} | t=999 -> {c1:.3f}")

        # 3. THE round-trip: given a PERFECT prediction, the reverse conversion
        #    must recover z0 exactly. This validates the v<->eps<->z0 algebra,
        #    which is where sign errors hide silently and ruin sampling.
        t = torch.randint(0, d.T, (4,), device=dev)
        noise = torch.randn_like(z0)
        z_t = d.q_sample(z0, t, noise)
        perfect = d.target(z0, noise, t)
        z0_hat, eps_hat = d._to_z0_eps(z_t, perfect, t)
        e0 = (z0_hat - z0).abs().max().item()
        ee = (eps_hat - noise).abs().max().item()
        ui.check(e0 < 1e-3 and ee < 1e-2, f"[{pred}] perfect pred recovers z0",
                 f"z0 err {e0:.2e} | eps err {ee:.2e}")

    # 4. end-to-end sampler shape + finiteness (untrained model -> noise, but
    #    it must be FINITE noise of the right shape)
    d = Diffusion(DCFG.timesteps, DCFG.schedule, "v", dev)
    m = DiT(latent_dim=DCFG.latent_dim, d_model=256, n_heads=4, n_layers=2,
            d_ff=512, n_frames=128, d_text=768).to(dev).eval()
    out = d.sample(m, (2, DCFG.latent_dim, 64), steps=10, device=dev)
    ui.check(out.shape == (2, DCFG.latent_dim, 64) and torch.isfinite(out).all(),
             "DDIM sampler", f"10 steps -> {tuple(out.shape)}, all finite")

    # 5. the efficiency claim, stated concretely
    ui.kv_table([
        ("diffusion", f"{DCFG.sample_steps} steps for ANY length"),
        ("AR (this project's twin)", f"{DCFG.n_frames * 8} steps for "
                                     f"{DCFG.n_frames/DCFG.frame_rate:.1f}s"),
        ("ratio", f"~{DCFG.n_frames*8/DCFG.sample_steps:.0f}x fewer sequential steps"),
    ], title="sampling cost")

    ui.panel("diffusion process verified", style="green")
