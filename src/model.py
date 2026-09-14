"""The language model: a spin carrier for token mixing, an MLP for channel mixing.

Deliberately small in surface area. The carrier is the only unusual part and it lives in spin.py; everything
here is the ordinary scaffolding around it, kept ordinary so that when a measurement moves it is attributable.

ALLOCATION. Per-layer cost is (2 + 2*mlp_mult) * d^2, so mlp_mult trades directly against width, and width is
also the carrier's state size. Measured at 45M with matched parameters and depth: carrier share 17.8% gives
loss 4.850, 5.5% gives 4.921, 3.4% gives 4.943. Loss rises monotonically as the carrier is diluted, so
mlp_mult defaults to 4 rather than to the 27 the previous lineage grew to.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from .spin import SpinCarrier
from .assoc import AssocCarrier


class Attn(nn.Module):
    """Causal self-attention. BASELINE ONLY.

    The vision rules out attention as the architecture. It is here because "spin is better" is unfalsifiable
    without a matched control trained on identical data with identical everything else, and the 45M allocation
    test was too short to settle it. Costs 4d^2 per layer against the carrier's 2d^2, so at matched parameters
    the attention control is slightly narrower.

    QK-norm (RMS-normalised queries and keys per head, learned scale) is standard in current transformers and
    is here because the first matched run at the sweep-optimal rate 6e-3 diverged at step ~11,000: gradient
    norm median rose 1.15 -> 5.0 and every step from 15,400 on was skipped by the spike guard while the spin
    run at identical settings took 57 skips in 120,000. The control must be the strongest attention I can
    field, or "spin wins" means nothing."""
    def __init__(self, d: int, heads: int = 8):
        super().__init__()
        self.n = nn.LayerNorm(d); self.h = heads
        self.q = nn.Linear(d, d); self.k = nn.Linear(d, d); self.v = nn.Linear(d, d); self.o = nn.Linear(d, d)
        self.qn = nn.RMSNorm(d // heads); self.kn = nn.RMSNorm(d // heads)

    def forward(self, x, h0=None):
        """h0 is a KV cache (k, v) from earlier tokens of the SAME window, for incremental decoding.

        Without it, decoding one token at a time gives attention a context of exactly that one token, which is
        not attention at all: the control's generations were pure loops (96% repeated 4-grams) and its decode
        speed was measured on a model that could not see. The cache is what a real transformer does, and it is
        also the thing that grows with every token while a recurrent state does not."""
        B_, T_, D = x.shape; z = self.n(x)
        q, k, v = [t(z).view(B_, T_, self.h, D // self.h).transpose(1, 2) for t in (self.q, self.k, self.v)]
        q, k = self.qn(q), self.kn(k)
        if h0 is not None:
            k = torch.cat([h0[0], k], dim=2); v = torch.cat([h0[1], v], dim=2)
        Lk = k.shape[2]
        if Lk == T_:
            a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            # is_causal aligns its mask to the top-left, so with a cache (k longer than q) it masks the WRONG
            # pairs: the control's bits per byte went from 1.12 to 2.99 before this was caught. Build the mask
            # against absolute positions instead.
            i = torch.arange(Lk - T_, Lk, device=x.device).unsqueeze(1)
            j = torch.arange(Lk, device=x.device).unsqueeze(0)
            a = F.scaled_dot_product_attention(q, k, v, attn_mask=(j <= i).view(1, 1, T_, Lk))
        a = a.transpose(1, 2).reshape(B_, T_, D)
        return x + self.o(a), (k, v)

    def spectral_parameters(self):
        return []


class MLP(nn.Module):
    def __init__(self, d: int, mult: int = 4):
        super().__init__()
        self.n = nn.LayerNorm(d)
        self.a = nn.Linear(d, mult * d)
        self.b = nn.Linear(mult * d, d)

    def forward(self, x):
        return x + self.b(F.gelu(self.a(self.n(x))))


class LM(nn.Module):
    def __init__(self, vocab=8192, d=768, layers=12, mlp_mult=4, ctx=512, mixer="spin"):
        """mixer: "spin" (plain carrier), "gspin" (input and output gates), "fspin" (input and recurrence
        gates: per-token forgetting), "aspin" (associative: content-addressed memory), "attn" (control)."""
        super().__init__()
        self.vocab, self.d, self.layers, self.ctx, self.mixer = vocab, d, layers, ctx, mixer
        self.e = nn.Embedding(vocab, d)
        self.p = nn.Embedding(ctx, d)
        mk = {"spin": lambda d: SpinCarrier(d), "gspin": lambda d: SpinCarrier(d, gated="io"),
              "fspin": lambda d: SpinCarrier(d, gated="rf"), "aspin": AssocCarrier, "attn": Attn}[mixer]
        self.mix = nn.ModuleList([mk(d) for _ in range(layers)])
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

    def trunk(self, x):
        """Everything but the output head: the normalised final representation, (B, T, d)."""
        h = self.e(x) + self.p(torch.arange(x.shape[1], device=x.device))
        for mx, mp in zip(self.mix, self.mlp):
            h = mp(mx(h)[0])
        return self.n(h)

    def loss(self, h, y, chunks=4):
        """Cross-entropy from the trunk output WITHOUT materialising all the logits at once (goal 6).

        The logits are B x T x V; at B32 T1024 V16384 that is 2.1 GB in fp32 and several copies of it live
        through softmax and its backward, which is what ran a 36 GiB cap out of memory. The head is applied to
        the batch in chunks under activation checkpointing, so at most one chunk's logits exist at a time and
        they are recomputed in backward. The trunk is compiled; this is not, and need not be."""
        y = y.reshape(-1); hf = h.reshape(-1, h.shape[-1]); n = hf.shape[0]
        E = self.e.weight
        def piece(hc, yc):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg = F.linear(hc, E)
            return F.cross_entropy(lg.float(), yc, reduction="sum")
        total = 0.0
        for i in range(chunks):
            sl = slice(i * n // chunks, (i + 1) * n // chunks)
            total = total + torch.utils.checkpoint.checkpoint(piece, hf[sl], y[sl], use_reentrant=False)
        return total / n

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

    @torch.no_grad()
    def forward_carried(self, x, window=None):
        """Process a sequence LONGER than ctx by carrying the carrier state across windows.

        This is the path that makes the carrier worth having: attention would need a KV cache that grows with
        the sequence, whereas the carrier's state is a fixed (B, d) complex vector per layer, so a 4096-token
        input costs the same memory as a 512-token one. Positional embeddings restart at each window, which is
        the only positional signal the model was trained with.

        Returns logits for the FINAL window only, which is what a retrieval probe scores."""
        window = window or self.ctx
        B_, T = x.shape
        states = [None] * self.layers
        pos = torch.arange(window, device=x.device)
        logits = None
        for s in range(0, T, window):
            xw = x[:, s:s + window]
            h = self.e(xw) + self.p(pos[: xw.shape[1]])
            for i, (mx, mp) in enumerate(zip(self.mix, self.mlp)):
                h, states[i] = mx(h, states[i])
                h = mp(h)
            logits = F.linear(self.n(h), self.e.weight)
        return logits

    @torch.no_grad()
    def grow(self, d_new: int):
        """Widen the model IN PLACE from d to d_new, preserving the function it computes (vision goal 7).

        Every grown parameter becomes a NEW Parameter object. Replacing .data in place looked cleaner and was
        wrong: autograd caches an accumulator per parameter with the shape at its creation, and the next
        backward after an in-place widening failed with "invalid gradient ... got [8192, 256] but expected
        [8192, 192]". Returns {new_param: (old_param, old_shape)} for Fabric.grew, which moves the optimizer
        state from the old object to the new one and resizes it.

        The rule that makes it work: in the residual stream, every WRITER of a new dimension starts at zero and
        every READER starts random. Writers zero means the new dimensions are exactly zero everywhere, so the
        logits do not change. Readers random means the writers receive a non-zero gradient on the very first
        step (grad of a writer is the upstream sensitivity, which flows through the readers), so the new
        capacity begins learning immediately instead of sitting at a zero-gradient fixed point, which is what
        zero-zero initialisation produces. Writers into the stream: the token and position embeddings' new
        columns, C's new rows, the MLP down-projection's new rows. Readers: LayerNorm entries, B's new columns,
        the MLP up-projection's new columns. Inside a layer the same rule holds for new carrier channels and
        new MLP units: B rows and MLP-up rows random (writers into a private space), C columns and MLP-down
        columns zero (readers of that space into the stream).

        LayerNorm is the one place preservation is approximate rather than exact: it normalises over all d_new
        entries, and the appended zeros shrink the variance by d/d_new, so the old affine weights are scaled by
        sqrt(d/d_new) to compensate. The residual error is the shift in the mean, second order for a stream
        whose mean is already near zero; the test measures it. (RMSNorm would make it exact; LayerNorm is kept
        so the grown model stays the same architecture as the runs it is compared with.)"""
        d, dn = self.d, d_new
        assert dn > d and self.mixer in ("spin", "gspin", "fspin")
        fresh = LM(self.vocab, dn, self.layers, self.mlp[0].a.weight.shape[0] // d, self.ctx, self.mixer)
        fresh = fresh.to(self.e.weight.device)
        grown = {}
        def put(mod, name, new):
            old = getattr(mod, name)
            newp = nn.Parameter(new)
            setattr(mod, name, newp)
            grown[newp] = (old, tuple(old.shape))
        def widen(mod, name, src, zero_rows=False, zero_cols=False):
            old = getattr(mod, name).data
            new = src.detach().clone().to(old.dtype)
            sl = tuple(slice(0, s) for s in old.shape)
            if zero_rows: new[old.shape[0]:] = 0
            if zero_cols and new.dim() == 2: new[:, old.shape[1]:] = 0
            new[sl] = old
            put(mod, name, new)
        def widen_ln(ln):
            old_w, old_b = ln.weight.data, ln.bias.data
            # new affine weights ZERO: LayerNorm maps the zero new dimensions to -mean/std, not to zero, and
            # random readers would see that. With the weight at zero the readers see exactly zero; the weight
            # itself gets a gradient on step one (readers are random), and the writers on step two.
            w = torch.zeros(dn, device=old_w.device, dtype=old_w.dtype); w[:d] = old_w * math.sqrt(d / dn)
            b = torch.zeros(dn, device=old_b.device, dtype=old_b.dtype); b[:d] = old_b
            put(ln, "weight", w); put(ln, "bias", b); ln.normalized_shape = (dn,)
        widen(self.e, "weight", fresh.e.weight, zero_cols=True)              # writer (and tied reader): zero
        widen(self.p, "weight", fresh.p.weight, zero_cols=True)              # writer: zero
        self.e.embedding_dim = self.p.embedding_dim = dn
        for mx, fx in zip(self.mix, fresh.mix):
            # new channels span the whole timescale and frequency range, not the tail of a longer linspace
            k = dn - d
            nu_new = torch.linspace(float(mx.nu_raw.data.min()), float(mx.nu_raw.data.max()), k, device=mx.nu_raw.device)
            th_new = torch.linspace(0.0, math.pi, k, device=mx.theta.device)
            put(mx, "nu_raw", torch.cat([mx.nu_raw.data, nu_new])); put(mx, "theta", torch.cat([mx.theta.data, th_new]))
            widen(mx.B, "weight", fx.B.weight)                                # rows: writers into new state, random;
                                                                              # cols: readers of the stream, random
            widen(mx.C, "weight", fx.C.weight, zero_rows=True, zero_cols=True)  # rows: writers into stream; cols: readers of new state
            if mx.gated:
                second = ("Wo", "Wr")[mx.gated == "rf"]
                for g_, fg in ((mx.Wi, fx.Wi), (getattr(mx, second), getattr(fx, second))):  # readers: random
                    widen(g_, "weight", fg.weight); widen(g_, "bias", fg.bias, zero_rows=True)
                    g_.in_features = g_.out_features = dn
            widen_ln(mx.ln)
            mx.B.in_features = mx.B.out_features = mx.C.in_features = mx.C.out_features = dn
            mx.d = dn
        for mp, fp in zip(self.mlp, fresh.mlp):
            widen(mp.a, "weight", fp.a.weight)                                # new units random, new stream readers random
            widen(mp.a, "bias", fp.a.bias, zero_rows=True)
            widen(mp.b, "weight", fp.b.weight, zero_rows=True, zero_cols=True)  # writers into stream zero, readers of new units zero
            widen(mp.b, "bias", fp.b.bias, zero_rows=True)
            widen_ln(mp.n)
            mp.a.in_features = dn; mp.a.out_features = fp.a.out_features
            mp.b.in_features = fp.b.in_features; mp.b.out_features = dn
        widen_ln(self.n)
        self.d = dn
        return grown

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
