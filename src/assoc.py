"""The associative carrier: a state that stores ASSOCIATIONS and is read by content.

Why it exists. Four matched runs at 77M showed the diagonal carrier, gated or not, retrieving a fact from 256
tokens back at +0.05 to +0.08 nats of lift where attention managed +4.8, and losing 0.07 to 0.14 bits per byte
on web text, where copying and induction carry much of the loss. A diagonal state is a compressed linear
summary of the past; it cannot look anything up by content. Content addressing needs a state that stores the
outer product of a key and a value and is read by a query. That is Hebbian plasticity, the brain's own
mechanism, and it is also the mathematics of the linear-attention family (RetNet, GLA, DeltaNet are its
efficient forms). The user was told this plainly and delegated the decision; this file is that decision.

What is ours. Per head, the memory S in R^{dk x dv} evolves as
    S_t = alpha_t S_{t-1} + i_t k_t v_t^T,    o_t = S_t^T q_t,
with q, k, v unit vectors, i_t in (0, 1) an input-dependent per-head write gate (so |S| is bounded by the number
of tokens remembered), alpha_t in (0, 1] an input-dependent per-head forgetting (a token can hold or clear a head's memory, as
fspin's gate did per channel) with a learned per-head timescale, and the keys and queries carrying the spin:
a per-channel-pair rotation R(theta_i t), learned frequencies, so that q_t . k_s picks up cos(theta_i (t - s))
per pair. The state is constant in size (H x dk x dv per layer, 360 KB at d704), carries across windows, and
retrieves by content at any distance subject to the forgetting it learned. Reads have no softmax and no
denominator; the per-head output is RMS-normalised instead.

Computation. The recurrence is time-varying, so it runs chunked: within a chunk of C tokens the read is a
C x C masked attention with the decay mask exp(L_t - L_s), L the cumulative log-decay, every exponent <= 0;
across chunks the state is carried with exp(L_C) S. Nothing can overflow at any forgetting rate. The chunked
form is checked against the literal recurrence in tests/test_assoc.py.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint


def rotate(x, theta, pos):
    """RoPE-style rotation of channel pairs by angle theta_i * pos. x: (B, H, T, dk); theta: (H, dk/2); pos: (T,)."""
    ang = pos.view(1, 1, -1, 1).to(theta.dtype) * theta.view(theta.shape[0], 1, theta.shape[1]).unsqueeze(0)  # (1,H,T,dk/2)
    c, s = torch.cos(ang), torch.sin(ang)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1).flatten(-2)


def _chunk(qc, kc, vc, lc, S):
    """One chunk: outputs and the next state. Factored out so it can be recomputed in backward."""
    C = qc.shape[2]
    L = lc.cumsum(2)                                                              # (B,H,C), <= 0 and decreasing
    mask = torch.ones(C, C, device=qc.device, dtype=torch.bool).tril()
    dec = torch.exp((L.unsqueeze(3) - L.unsqueeze(2)).clamp(max=0.0)) * mask     # exp(L_t - L_s), s <= t
    A = (qc @ kc.transpose(2, 3)) * dec                                          # (B,H,C,C)
    eL = torch.exp(L).unsqueeze(3)                                                # exp(L_t): decay from chunk start
    o = A @ vc + (qc * eL) @ S                                                    # intra + carried
    # S_new = exp(L_C) S + sum_s exp(L_C - L_s) k_s v_s^T
    w = torch.exp((L[:, :, -1:] - L).clamp(max=0.0)).unsqueeze(3)                 # (B,H,C,1)
    S = eL[:, :, -1:] * S + (kc * w).transpose(2, 3) @ vc                        # eL[:, :, -1:] is (B,H,1,1)
    return o, S


def assoc_scan(q, k, v, logalpha, S0=None, chunk: int = 64, recompute: bool = False):
    """o_t = sum_{s<=t} exp(L_t - L_s) (q_t . k_s) v_s  [+ exp(L_t) q_t . S0], chunked and stable.

    q, k, v: (B, H, T, D); logalpha: (B, H, T) <= 0 (log of the per-token forgetting); S0: (B, H, D, D) or None.
    Returns o: (B, H, T, D) and the final state S: (B, H, D, D). fp32 inside."""
    B, H, T, D = q.shape
    rt = torch.float64 if q.dtype == torch.float64 else torch.float32
    q, k, v, la = q.to(rt), k.to(rt), v.to(rt), logalpha.to(rt)
    S = torch.zeros(B, H, D, D, device=q.device, dtype=rt) if S0 is None else S0.to(rt)
    outs = []
    for s0 in range(0, T, chunk):
        qc, kc, vc, lc = q[:, :, s0:s0 + chunk], k[:, :, s0:s0 + chunk], v[:, :, s0:s0 + chunk], la[:, :, s0:s0 + chunk]
        if recompute and torch.is_grad_enabled():
            # activation checkpointing per chunk: the C x C decay and score matrices of every chunk of every
            # layer otherwise stay alive for backward (the 219M model at B32 T1024 ran a 36 GiB cap out)
            o, S = torch.utils.checkpoint.checkpoint(_chunk, qc, kc, vc, lc, S, use_reentrant=False)
        else:
            o, S = _chunk(qc, kc, vc, lc, S)
        outs.append(o)
    return torch.cat(outs, 2).to(q.dtype), S


class AssocCarrier(nn.Module):
    def __init__(self, d: int, heads: int | None = None, chunk: int = 64, recompute: bool = False):
        super().__init__()
        self.d = d; self.recompute = recompute
        self.h = heads or max(1, d // 64)
        assert d % self.h == 0 and (d // self.h) % 2 == 0
        self.dk = d // self.h
        self.chunk = chunk
        self.ln = nn.LayerNorm(d)
        self.q = nn.Linear(d, d, bias=False); self.k = nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, d, bias=False); self.o = nn.Linear(d, d, bias=False)
        # per-head forgetting: log alpha_t = -softplus(w . z + b) * exp(A_h). A_h spreads the heads over
        # timescales at init (half-lives from ~7 to ~7000 tokens at the gate's resting point); the gate moves it.
        self.wa = nn.Linear(d, self.h)
        nn.init.zeros_(self.wa.weight); nn.init.constant_(self.wa.bias, 0.0)      # softplus(0) = 0.69
        # write gate per head: how much of this token goes into the memory. With k and v both unit-normalised
        # every write has magnitude sigmoid(.) <= 1, so |S| is bounded by the number of tokens remembered.
        # Measured without it (run7, step 20k): |S| per head up to 193,000, most tokens' forgetting learned to
        # zero, gradient spikes of 24x the median on 5% of batches, and divergence at 23k.
        self.wi = nn.Linear(d, self.h)
        self.A = nn.Parameter(torch.linspace(math.log(0.1 / 0.69), math.log(1e-4 / 0.69), self.h))
        # the spin: rotation frequencies per channel pair, learned; geometric init like RoPE (base 10000)
        base = 10000.0 ** (-torch.arange(0, self.dk, 2).float() / self.dk)
        self.theta = nn.Parameter(base.unsqueeze(0).repeat(self.h, 1))
        self.on = nn.RMSNorm(self.dk)
        self.gate = nn.Parameter(torch.zeros(()))

    def spectral_parameters(self):
        """Timescales and frequencies take small absolute steps, as in the diagonal carrier."""
        return [self.A, self.theta, self.gate]

    def logalpha(self, z):
        return -F.softplus(self.wa(z)) * torch.exp(self.A)                        # (B,T,H), <= 0

    def forward(self, x, h0=None, pos0: int = 0):
        """h0: (state S (B,H,dk,dk), position offset) from a previous window, or None."""
        B, T, D = x.shape
        z = self.ln(x)
        q, k, v = [t(z).view(B, T, self.h, self.dk).transpose(1, 2) for t in (self.q, self.k, self.v)]
        q, k, v = F.normalize(q, dim=-1), F.normalize(k, dim=-1), F.normalize(v, dim=-1)   # scores in [-1, 1], unit writes
        k = k * torch.sigmoid(self.wi(z)).transpose(1, 2).unsqueeze(-1)            # write gate per head
        S0, p0 = (None, pos0) if h0 is None else h0
        pos = torch.arange(p0, p0 + T, device=x.device)
        q, k = rotate(q, self.theta, pos), rotate(k, self.theta, pos)
        la = self.logalpha(z).transpose(1, 2)                                     # (B,H,T)
        o, S = assoc_scan(q, k, v, la, S0, chunk=self.chunk, recompute=self.recompute)
        o = self.on(o).transpose(1, 2).reshape(B, T, D)
        return x + torch.sigmoid(self.gate) * self.o(o), (S, p0 + T)
