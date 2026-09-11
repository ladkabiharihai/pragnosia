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
    def __init__(self, d: int, chunk: int = 128, nu_min: float = 1e-3, nu_max: float = 0.3):
        # chunk=128 measured: 53.3 ms/step compiled against 61.5 at chunk=64 and 204 eager, because the
        # cross-chunk loop is sequential and halving its length halves the kernel-launch rounds. Accuracy is
        # unaffected: relative error in float32 is 9.6e-6 at chunk 32, 64, 128 and 256 alike, since within a
        # chunk the sum is dominated by its own largest term and the large lambda^-k cancels cleanly.
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

    # ------------------------------------------------------------------ fast: FFT, no inverse powers
    def scan_chunked(self, u: torch.Tensor, h0=None):
        """RETIRED. Kept only so the reason survives.

        This computed within-chunk prefix sums of lambda^(-s) u_s and then multiplied by lambda^t. The inverse
        power lambda^(-s) is exp(nu*s), and nu is LEARNED with no upper bound: the optimizer pushed nu_max from
        0.345 to 0.382 in a thousand steps, taking lambda^-128 from 1.5e19 to 1.7e21. Under bf16 autocast an
        8-bit mantissa cannot hold a cumsum whose terms span twenty orders of magnitude, and the run went NaN
        twice at around step 14,500. Capping nu would fix it and would also be a hand-set bound on what the
        model may learn, which vision goal 8 forbids. The FFT path below needs no cap."""
        raise RuntimeError("scan_chunked is retired; it overflows once nu grows. use scan()")

    @torch.compiler.disable
    def scan(self, u: torch.Tensor, h0=None):
        """Causal convolution with kernel lambda^t, by REAL FFT. Excluded from torch.compile.

        Only the real part of the complex state is emitted within a window; the imaginary part exists solely
        to carry the recurrence across windows. So the per-window output is Re(h_t) = sum Re(lambda^(t-s)) u_s,
        a REAL convolution, which rfft/irfft evaluates at half the transforms and half the intermediate memory
        of the complex version this replaced. Measured before: spin 1.83x slower than FlashAttention at T=512
        with memory growing in T because complex64 intermediates were saved for backward.

        Every kernel entry has magnitude |lambda|^t <= 1, so nothing here can overflow at any forgetting rate.
        The final imaginary state is recovered separately as one weighted sum, only when a caller carries it.

        Excluded from torch.compile: inductor's backward through FFT produced non-finite gradients while eager
        was correct, in fp32 and under autocast alike."""
        mag, ang = self.lam()
        B_, T, D = u.shape
        rt = torch.float64 if u.dtype == torch.float64 else torch.float32
        uf = u.to(rt); mag = mag.to(rt); ang = ang.to(rt)
        t = torch.arange(T, device=u.device, dtype=rt).unsqueeze(1)                  # (T,1)
        logk = t * torch.log(mag).unsqueeze(0)                                       # (T,D), all <= 0
        decay = torch.exp(logk)
        kr = decay * torch.cos(t * ang.unsqueeze(0))                                 # Re(lambda^t)
        n = 2 * T
        U = torch.fft.rfft(uf, n=n, dim=1)
        Kr = torch.fft.rfft(kr, n=n, dim=0).unsqueeze(0)
        hr = torch.fft.irfft(U * Kr, n=n, dim=1)[:, :T]
        need_state = True
        if h0 is not None:
            hr0, hi0 = h0[0].to(rt), h0[1].to(rt)
            d1 = torch.exp(logk + torch.log(mag).unsqueeze(0))                      # |lambda|^(t+1)
            a1 = (t + 1) * ang.unsqueeze(0)
            # Re(lambda^(t+1) * h0) = Re*hr0 - Im*hi0, all real arithmetic
            hr = hr + (d1 * torch.cos(a1)).unsqueeze(0) * hr0.unsqueeze(1) \
                    - (d1 * torch.sin(a1)).unsqueeze(0) * hi0.unsqueeze(1)
        # final imaginary state: Im(h_{T-1}) = sum_s Im(lambda^(T-1-s)) u_s, one weighted sum, real ops
        rev = torch.flip(decay * torch.sin(t * ang.unsqueeze(0)), dims=[0])          # Im(lambda^(T-1-s))
        hi_last = (uf * rev.unsqueeze(0)).sum(1)
        if h0 is not None:
            dT = mag ** T; aT = T * ang
            hi_last = hi_last + dT * torch.sin(aT) * hr0 + dT * torch.cos(aT) * hi0
        return hr.to(u.dtype), (hr[:, -1].to(u.dtype), hi_last.to(u.dtype))

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
