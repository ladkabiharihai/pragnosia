"""The language model: a spin carrier for token mixing, an MLP for channel mixing.

Deliberately small in surface area. The carrier is the only unusual part and it lives in spin.py; everything
here is the ordinary scaffolding around it, kept ordinary so that when a measurement moves it is attributable.

ALLOCATION. Per-layer cost is (2 + 2*mlp_mult) * d^2, so mlp_mult trades directly against width, and width is
also the carrier's state size. Measured at 45M with matched parameters and depth: carrier share 17.8% gives
loss 4.850, 5.5% gives 4.921, 3.4% gives 4.943. Loss rises monotonically as the carrier is diluted, so
mlp_mult defaults to 4 rather than to the 27 the previous lineage grew to.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from .spin import SpinCarrier


class MLP(nn.Module):
    def __init__(self, d: int, mult: int = 4):
        super().__init__()
        self.n = nn.LayerNorm(d)
        self.a = nn.Linear(d, mult * d)
        self.b = nn.Linear(mult * d, d)

    def forward(self, x):
        return x + self.b(F.gelu(self.a(self.n(x))))


class LM(nn.Module):
    def __init__(self, vocab=8192, d=768, layers=12, mlp_mult=4, ctx=512):
        super().__init__()
        self.vocab, self.d, self.layers, self.ctx = vocab, d, layers, ctx
        self.e = nn.Embedding(vocab, d)
        self.p = nn.Embedding(ctx, d)
        self.mix = nn.ModuleList([SpinCarrier(d) for _ in range(layers)])
        self.mlp = nn.ModuleList([MLP(d, mlp_mult) for _ in range(layers)])
        self.n = nn.LayerNorm(d)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        # nn.Embedding defaults to N(0,1). Tied to the output that gives logits of std ~sqrt(d) and a first
        # loss of 515 against a random baseline of 9.0, which is easy to mistake for an architecture problem.
        if isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, 0.0, 0.02)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x, return_hidden=False, hidden_layer=None):
        """hidden_layer selects WHICH representation to return. The final one is what the output head reads,
        so it is the least informative place to look for signal the head has not already used."""
        h = self.e(x) + self.p(torch.arange(x.shape[1], device=x.device))
        grab = None
        for i, (mx, mp) in enumerate(zip(self.mix, self.mlp)):
            h = mp(mx(h)[0])
            if hidden_layer is not None and i == hidden_layer:
                grab = h
        h = self.n(h)
        logits = F.linear(h, self.e.weight)          # tied
        if not return_hidden:
            return logits
        return logits, (h if grab is None else grab)

    def param_groups(self, lr=6e-3, spectral_lr=1e-3):
        """Spectral parameters take small ABSOLUTE steps, everything else takes trust-ratio steps.

        nu and theta are the per-channel timescale and frequency, not arbitrary weights, so moving them by a
        fraction of their own norm is meaningless. Measured: with ||nu_raw|| about 104, a 6e-3 trust fraction
        moved the spectrum by 0.6 per step and drove |lambda| from 0.999 to 0.9998 in thirty steps, then NaN."""
        spec = [p for mx in self.mix for p in mx.spectral_parameters()]
        sid = {id(p) for p in spec}
        rest = [p for p in self.parameters() if id(p) not in sid]
        return [{"params": rest, "lr": lr, "trust": True},
                {"params": spec, "lr": spectral_lr, "trust": False, "factored": False}]
