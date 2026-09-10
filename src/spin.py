"""A diagonal-complex linear recurrent carrier, implemented here rather than taken from anywhere.

WHAT IT IS. Each channel holds a complex state that is multiplied by a fixed complex number every step:

    h_t = lambda * h_{t-1} + B x_t ,      lambda = exp(-nu + i*theta),   nu > 0
    y_t = Re(C h_t)

The magnitude exp(-nu) is a per-channel forgetting rate and theta is a per-channel rotation, so each channel
is a damped oscillator with its own timescale and frequency. Because lambda is diagonal and CONSTANT in time,
the recurrence is a convolution and can be evaluated in parallel.

WHY BOTH A REFERENCE AND A FAST PATH. The parallel form needs lambda^(-s), which is exp(nu*s) and overflows
for any useful sequence length. The fix is to chunk, so the largest power is the chunk size rather than the
sequence length. That is an optimisation with a real chance of being subtly wrong, so the sequential
definition is kept and the fast path is checked against it rather than assumed to match.

WHAT IS DELIBERATELY NOT HERE. No gating, no input-dependent lambda (that would make it selective, i.e. an
SSM of the Mamba kind), no attention. This is the smallest thing that carries state across tokens, so that
what is being measured is the carrier and not a hybrid.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn


class SpinCarrier(nn.Module):
    def __init__(self, d: int, chunk: int = 64, nu_min: float = 1e-3, nu_max: float = 0.3):
        super().__init__()
        self.d, self.chunk = d, chunk
        # nu is parameterised through softplus so it stays positive: |lambda| < 1 is what keeps the
        # recurrence stable, and a sign flip here is the difference between forgetting and exploding.
        nu0 = torch.linspace(math.log(math.expm1(nu_min)), math.log(math.expm1(nu_max)), d)
        self.nu_raw = nn.Parameter(nu0)
        # spread the rotation rates over the unit circle so channels start with distinct frequencies
        self.theta = nn.Parameter(torch.linspace(0.0, math.pi, d))
        self.B = nn.Linear(d, d, bias=False)
        self.C = nn.Linear(d, d, bias=False)
        self.ln = nn.LayerNorm(d)
        self.gate = nn.Parameter(torch.zeros(()))

    NU_MIN = 1e-3            # bounds |lambda| <= exp(-1e-3) = 0.999

    def lam(self):
        # The floor is load-bearing. gamma = sqrt(1 - |lambda|^2) goes to zero as nu goes to zero and its
        # derivative diverges there, so an optimizer that pushes nu down finds an infinite gradient and the run
        # dies. Measured without the floor: |lambda| driven 0.999026 -> 0.999758 over 30 steps, then NaN.
        nu = self.NU_MIN + torch.nn.functional.softplus(self.nu_raw)
        return torch.exp(-nu), self.theta                      # magnitude, angle

    def spectral_parameters(self):
        """Parameters that live on a MEANINGFUL scale and must not be trust-ratio stepped.

        A trust ratio moves a parameter by a fixed fraction of its own norm, which is right for a weight matrix
        whose norm is arbitrary. nu and theta are not arbitrary: they ARE the per-channel timescale and
        frequency. With ||nu_raw|| about 104, a trust fraction of 6e-3 moves the whole spectrum by 0.6 every
        step, which is how the run above reached |lambda| = 0.9998 in thirty steps. These want a plain, small,
        absolute step."""
        return [self.nu_raw, self.theta, self.gate]

    # ------------------------------------------------------------------ reference: the definition itself
    def scan_reference(self, u: torch.Tensor, h0=None):
        """h_t = lambda h_{t-1} + u_t, done literally. O(T) python steps: slow, and correct by construction."""
        mag, ang = self.lam()
        lr, li = mag * torch.cos(ang), mag * torch.sin(ang)
        B_, T, D = u.shape
        hr = torch.zeros(B_, D, device=u.device, dtype=u.dtype) if h0 is None else h0[0]
        hi = torch.zeros(B_, D, device=u.device, dtype=u.dtype) if h0 is None else h0[1]
        out = []
        for t in range(T):
            nr = hr * lr - hi * li + u[:, t]
            ni = hr * li + hi * lr
            hr, hi = nr, ni
            out.append(hr)
        return torch.stack(out, 1), (hr, hi)

    # ------------------------------------------------------------------ fast: chunked, same recurrence
    def scan(self, u: torch.Tensor, h0=None):
        """Chunked parallel scan. Within a chunk the largest power of lambda is `chunk`, so exp(nu*chunk)
        stays finite; across chunks the state is threaded sequentially, which is O(T/chunk) steps."""
        mag, ang = self.lam()
        B_, T, D = u.shape
        C = self.chunk
        pad = (C - T % C) % C
        if pad:
            u = torch.nn.functional.pad(u, (0, 0, 0, pad))
        N = u.shape[1] // C
        uc = u.view(B_, N, C, D)

        k = torch.arange(C, device=u.device).view(1, 1, C, 1)
        logmag = torch.log(mag).view(1, 1, 1, D)
        # decay^(t) and its inverse, both bounded because t < chunk
        dec = torch.exp(logmag * k)                                 # |lambda|^k
        ph = ang.view(1, 1, 1, D) * k                               # k*theta
        wr, wi = dec * torch.cos(ph), dec * torch.sin(ph)           # lambda^k
        inv = torch.exp(-logmag * k)
        ir, ii = inv * torch.cos(-ph), inv * torch.sin(-ph)         # lambda^-k

        # within-chunk prefix sums of lambda^-s u_s, then multiplied by lambda^t
        ar = torch.cumsum(uc * ir, dim=2)
        ai = torch.cumsum(uc * ii, dim=2)
        yr = ar * wr - ai * wi
        yi = ar * wi + ai * wr

        # thread chunk-final states forward
        hr = torch.zeros(B_, D, device=u.device, dtype=u.dtype) if h0 is None else h0[0]
        hi = torch.zeros(B_, D, device=u.device, dtype=u.dtype) if h0 is None else h0[1]
        outs = []
        cr, ci = wr[:, :, :, :], wi[:, :, :, :]
        for n in range(N):
            # carried state decays by lambda^(k+1) across the chunk
            gr = cr[0, 0] * hr.unsqueeze(1) - ci[0, 0] * hi.unsqueeze(1)
            gi = cr[0, 0] * hi.unsqueeze(1) + ci[0, 0] * hr.unsqueeze(1)
            # lambda^(k+1) not lambda^k, since the carried state is one step older
            mr = mag * torch.cos(ang); mi = mag * torch.sin(ang)
            br = gr * mr - gi * mi
            bi = gr * mi + gi * mr
            r = yr[:, n] + br
            i_ = yi[:, n] + bi
            outs.append(r)
            hr, hi = r[:, -1], i_[:, -1]
        y = torch.cat(outs, 1)
        return (y[:, :T] if pad else y), (hr, hi)

    def gamma(self):
        """Input normalisation, and it is not optional.

        For h_t = lambda h_{t-1} + gamma u_t with unit-variance u, Var(h) = gamma^2 / (1 - |lambda|^2). Without
        the factor, a channel with |lambda| near 1 accumulates the whole sequence with no attenuation and its
        gain grows with T. Measured before this was added: initial loss 301 against a random baseline of 9.0,
        |lambda| then driven from 0.999026 toward 1 by the optimizer, NaN by step 24. With it, every channel
        contributes unit variance regardless of its timescale, so slow channels are long-memory rather than
        loud."""
        mag, _ = self.lam()
        return torch.sqrt(1.0 - mag * mag).clamp_min(1e-4)

    def forward(self, x: torch.Tensor, h0=None):
        u = self.B(self.ln(x)) * self.gamma()
        y, h = self.scan(u, h0)
        return x + torch.sigmoid(self.gate) * self.C(y), h
