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


_SWA_MASKS, _SWA_FLEX = {}, None
def _swa(q, k, v, w):
    """TRUE SLIDING-WINDOW causal attention: query i (the last len(q) of the len(k) positions) sees the w positions
    ending at itself, in training and at inference alike. flex_attention with a block mask keeps it fused and linear
    in length; one compiled kernel, one mask per (query length, key length, window)."""
    global _SWA_FLEX
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask
    # INSIDE a compiled region call flex_attention itself (the compiler lowers it to the fused kernel); OUTSIDE, call a
    # compiled copy made outside any trace. The first version created that copy lazily on the first call, which in
    # training happened INSIDE the compiled trunk: the copy never became a compiled kernel, and the trials' eager
    # held-out pass then ran flex's reference maths, a full 8 x 8,192 x 8,192 score matrix, 24 GiB (run80, 2026-10-02).
    if torch.compiler.is_compiling():
        fa = flex_attention
    else:
        if _SWA_FLEX is None: _SWA_FLEX = torch.compile(flex_attention)
        fa = _SWA_FLEX
    Lq, Lk = q.shape[2], k.shape[2]; off = Lk - Lq
    key = (Lq, Lk, w, q.device)
    if key not in _SWA_MASKS:
        def mm(b, h, qi, ki): return (qi + off >= ki) & (qi + off - ki < w)
        _SWA_MASKS[key] = create_block_mask(mm, B=None, H=None, Q_LEN=Lq, KV_LEN=Lk, device=q.device)
    return fa(q, k, v, block_mask=_SWA_MASKS[key])


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
    def __init__(self, d: int, heads: int = 8, window: int = 0, read_state: int = 0, sliding: bool = False):
        super().__init__()
        # sliding: a TRUE sliding window everywhere (_swa). Without it the window is BLOCK-LOCAL in training (independent
        # 512-token blocks: a token sees 0-511 back depending on where it sits in its block), sliding at decode, and
        # FULL causal over the window for a prefill whose length is not a multiple of the window -- three different
        # functions, found 2026-10-02 when retrieval stopped dead at the window. Kept for checkpoints trained that way.
        self.sliding = sliding
        self.n = nn.LayerNorm(d); self.h = heads
        # read_state: head dimension of the carrier memory this layer is allowed to read, 0 for none. The
        # 1:2 hybrid recovered three quarters of attention's exact recall with a 512-token window but its
        # context gain went POSITIVE (+0.0837 over 512 -> 32k, 2026-09-17): block attention cannot see past
        # its own block and half the carrier layers cannot hold what twelve held. This lets the attention
        # layer put its OWN query into the previous carrier layer's state, which is where everything older
        # than the block actually lives. The read is at chunk granularity, which is all the state has.
        self.read_state = read_state
        self.q = nn.Linear(d, d); self.k = nn.Linear(d, d); self.v = nn.Linear(d, d); self.o = nn.Linear(d, d)
        self.qn = nn.RMSNorm(d // heads); self.kn = nn.RMSNorm(d // heads)
        if read_state:
            self.rq = nn.Linear(d, d, bias=False)          # the query this layer puts into the carrier's state
            self.rn = nn.RMSNorm(read_state); self.rg = nn.Parameter(torch.zeros(()))
            self.rs_chunk = 64
        # window > 0 restricts each token to the last `window` positions. The cost is then linear in length and
        # the cache is bounded, which is the only form of attention that can sit in this project without
        # giving back what the carrier was built to win.
        self.window = window

    def forward(self, x, h0=None, mem=None):
        """h0 is a KV cache (k, v) from earlier tokens of the SAME window, for incremental decoding.

        Without it, decoding one token at a time gives attention a context of exactly that one token, which is
        not attention at all: the control's generations were pure loops (96% repeated 4-grams) and its decode
        speed was measured on a model that could not see. The cache is what a real transformer does, and it is
        also the thing that grows with every token while a recurrent state does not."""
        B_, T_, D = x.shape; z = self.n(x)
        import src.assoc as _A
        if _A.FUSED_PROJ:
            # one matmul for q, k and v: z is read from VRAM once and the product is 3d wide (see AssocCarrier)
            y = F.linear(z, torch.cat([self.q.weight, self.k.weight, self.v.weight]),
                         torch.cat([self.q.bias, self.k.bias, self.v.bias]))
            q, k, v = [t.reshape(B_, T_, self.h, D // self.h).transpose(1, 2) for t in y.split(D, dim=-1)]
        else:
            q, k, v = [t(z).view(B_, T_, self.h, D // self.h).transpose(1, 2) for t in (self.q, self.k, self.v)]
        q, k = self.qn(q), self.kn(k)
        if self.window and self.sliding:
            if h0 is not None:
                k = torch.cat([h0[0], k], dim=2); v = torch.cat([h0[1], v], dim=2)
            if T_ == 1:
                # decode: the ring, a fixed-size cache of the window (shapes constant for the captured decode step).
                # Slots the graphed decode pads with zeros are not keys: masked (a real key is RMS-normed, never zero)
                k = k[:, :, -self.window:]; v = v[:, :, -self.window:]
                # ONE QUERY AGAINST THE WINDOW AS TWO MATRIX-VECTOR PRODUCTS. Through SDPA this picked a float32
                # memory-efficient kernel built for long queries: 3.25 ms per token for three layers at an 8,192
                # window, 167 tok/s against 431 at 512 (2026-10-03).
                sc = (q.float() @ k.float().transpose(-1, -2)) * (q.shape[-1] ** -0.5)
                sc = sc.masked_fill(~(k.abs().amax(-1) > 0).unsqueeze(2), float("-inf"))
                a = (sc.softmax(-1) @ v.float()).to(q.dtype)
            else:
                a = _swa(q, k, v, self.window)
            return x + self.o(self._state_read(a, z, mem, B_, T_, D)), (k[:, :, -self.window:], v[:, :, -self.window:])
        if h0 is not None and self.window and T_ > self.window:
            # A WHOLE NEW WINDOW ARRIVING WITH A CACHE (long-context prefill: the carrier's state carries, attention's
            # cache came along). The ring branch below kept only the LAST `window` keys of cache + new, so every query
            # before the last `window` of the new tokens had NO visible key: its row was fully masked and the layer
            # output zeros for 7,680 of 8,192 tokens in every window after the first (found 2026-10-03). Every
            # block-local "more context hurts" measurement past 8,192 went through this. Block-local semantics: a
            # block-aligned window ignores the cache, exactly as in training; any other length sees a true sliding window.
            if T_ % self.window == 0:
                h0 = None
            else:
                k = torch.cat([h0[0], k], dim=2); v = torch.cat([h0[1], v], dim=2)
                a = _swa(q, k, v, self.window)
                return x + self.o(self._state_read(a, z, mem, B_, T_, D)), (k[:, :, -self.window:], v[:, :, -self.window:])
        if h0 is not None:
            if self.window and h0[0].shape[2] == self.window:
                # A RING, NOT A LIST. Once the cache is full, decoding rolls it: the shapes never change again,
                # so the whole decode step can be captured as a CUDA graph. Measured 2026-09-18: a hybrid
                # decoded at 16.8 ms per token against the pure carrier's 1.92 purely because its cache grew by
                # one token per step and the graph could not hold it. The oldest column is dropped, the newest
                # written at the end, and the mask below still reads left to right.
                # Drop as many of the oldest columns as there are new ones, not exactly one: this branch is
                # entered whenever the cache happens to be exactly one window long, and a prefill continuing
                # from a full cache arrives with T_ > 1. With T_ = 6 it handed back 511 + 6 = 517 and the
                # graphed decode's 512-slot ring could not take it (2026-09-22). For the decode step this is
                # T_ = 1 and the behaviour and the shapes are unchanged, so the graph still captures.
                k = torch.cat([h0[0], k], dim=2)[:, :, -self.window:]
                v = torch.cat([h0[1], v], dim=2)[:, :, -self.window:]
                Lk = k.shape[2]
                i = torch.arange(Lk - T_, Lk, device=x.device).unsqueeze(1)
                j = torch.arange(Lk, device=x.device).unsqueeze(0)
                a = F.scaled_dot_product_attention(q, k, v, attn_mask=(j <= i).view(1, 1, T_, Lk))
                a = self._state_read(a, z, mem, B_, T_, D)
                return x + self.o(a), (k, v)
            k = torch.cat([h0[0], k], dim=2); v = torch.cat([h0[1], v], dim=2)
            if self.window:
                # A WINDOWED LAYER MUST NOT KEEP AN UNBOUNDED CACHE. Trained on independent 512-token blocks,
                # this layer had never decoded before 2026-09-18, and the decode path was attending over the
                # whole cache: a different function from the one trained, and a cache that grows for ever,
                # which is precisely the cost the carrier exists to avoid. Sliding rather than block-aligned,
                # so a token sees the same span it saw in training and never more.
                k = k[:, :, -self.window:]; v = v[:, :, -self.window:]
        Lk = k.shape[2]
        if self.window and Lk == T_ and T_ % self.window == 0 and T_ > self.window:
            # BLOCK-LOCAL, not a masked full attention. An explicit sliding-window mask costs a T x T boolean
            # and turns off the flash kernel: measured 94 ktok/s against the plain attention's 300 at ctx 2048.
            # Reshaping into independent blocks of `window` and running the flash kernel causally inside each
            # keeps the fast path, is linear in length, and bounds the cache at one block. Tokens early in a
            # block see less; the carrier layers between these are what carry the rest.
            w = self.window; nb = T_ // w
            qb, kb, vb = [t.reshape(B_, self.h, nb, w, D // self.h).reshape(B_ * nb, self.h, w, D // self.h)
                          if False else t.view(B_, self.h, nb, w, D // self.h).transpose(1, 2).reshape(B_ * nb, self.h, w, D // self.h)
                          for t in (q, k, v)]
            a = F.scaled_dot_product_attention(qb, kb, vb, is_causal=True)
            a = a.view(B_, nb, self.h, w, D // self.h).transpose(1, 2).reshape(B_, self.h, T_, D // self.h)
        elif Lk == T_:
            a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            # is_causal aligns its mask to the top-left, so with a cache (k longer than q) it masks the WRONG
            # pairs: the control's bits per byte went from 1.12 to 2.99 before this was caught. Build the mask
            # against absolute positions instead.
            i = torch.arange(Lk - T_, Lk, device=x.device).unsqueeze(1)
            j = torch.arange(Lk, device=x.device).unsqueeze(0)
            a = F.scaled_dot_product_attention(q, k, v, attn_mask=(j <= i).view(1, 1, T_, Lk))
        if self.window:
            # the cache this layer hands back is never longer than its window, whichever path built it: a
            # prefill of 1,006 tokens handed back all 1,006 and the graphed decode's 512-slot ring could not
            # take it (2026-09-21)
            k = k[:, :, -self.window:]; v = v[:, :, -self.window:]
        return x + self.o(self._state_read(a, z, mem, B_, T_, D)), (k, v)

    def _state_read(self, a, z, mem, B_, T_, D):
        """Fold in this layer's read of the carrier's memory, if it has one, and flatten the heads."""
        C = self.rs_chunk if self.read_state else 0
        NC = min(T_ // C, mem.shape[2]) if (C and mem is not None) else 0
        if NC:
            # r_t = q_t S_c for the chunk c that token t starts in: one batched matmul of (NC,C,dk) by
            # (NC,dk,dk) per head. The read is normalised and gated so an untrained layer adds nothing.
            # A sequence whose length is not a multiple of the chunk leaves a TAIL with no entering state of
            # its own; those tokens read nothing and rely on this layer's own block, which covers exactly the
            # recent past they sit in. Training always used multiples of 64, so this path first mattered at
            # inference, where it raised rather than silently misaligning (2026-09-18).
            hh, dk = mem.shape[1], mem.shape[3]; n = NC * C
            qm = self.rq(z[:, :n]).view(B_, n, hh, dk).transpose(1, 2).reshape(B_, hh, NC, C, dk)
            qm = F.normalize(qm, dim=-1).to(mem.dtype)
            rr = torch.matmul(qm, mem[:, :, :NC].to(mem.dtype)).view(B_, hh, n, dk)
            rr = self.rn(rr.float()).to(a.dtype).transpose(1, 2).reshape(B_, n, hh * dk)
            a = a.transpose(1, 2).reshape(B_, T_, D)
            a = a + torch.sigmoid(self.rg) * F.pad(rr, (0, 0, 0, T_ - n))
        else:
            a = a.transpose(1, 2).reshape(B_, T_, D)
        return a

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


def _head_piece_eager(hc, yc, E):
    """One chunk of the output head: logits in bf16, cross-entropy in fp32."""
    with torch.autocast("cuda", dtype=torch.bfloat16):
        lg = F.linear(hc, E)
    return F.cross_entropy(lg.float(), yc, reduction="sum")


# The same chunk compiled, once, at module level (a closure compiled inside loss() would recompile per call).
# Eager, the fp32 cast of a (2048, 8192) logits chunk was a separate copy kernel, twelve per micro-step with
# the checkpoint's recompute and the backward, and the softmax ran as cuDNN's generic forward and backward.
# Compiled, the cast fuses into a single log-sum-exp reduction over the bf16 logits and the backward into one
# kernel: attention 325 -> 343 ktok/s at ctx 2048 micro-batch 4, reserved memory 4.52 -> 4.16 GiB. Chunk
# count 1, 2, 4 or 8 makes no difference to either at this vocabulary.
_head_piece = torch.compile(_head_piece_eager, dynamic=False)


def _head_piece_ul_eager(hc, yc, negc, mskc, E, alpha):
    """One chunk of the head, with an UNLIKELIHOOD term on the tokens just emitted.

    Maximum likelihood never conditions on the model's own output: every position is scored against ground
    truth, so nothing in the objective ever discourages repeating what was just said. At generation the model
    conditions on itself, and the measured result is a stream that stays in a judge's band for about 4,000
    tokens and then loops. The data, the window, the kernel, the carried state and the sampler have each been
    ruled out by measurement (see PROGRESS.md, 2026-09-22), which leaves the objective.

    So for every position, the tokens that occurred in the previous `ul_window` positions are made LESS likely,
    except the one that is actually correct there: loss += -alpha * sum log(1 - p(previous token)). It needs
    only the probabilities of those few tokens, not the whole row, so it costs one gather of W columns and
    leaves the chunked head's memory behaviour untouched."""
    with torch.autocast("cuda", dtype=torch.bfloat16):
        lg = F.linear(hc, E)
    lp = lg.float().log_softmax(-1)
    ce = F.nll_loss(lp, yc, reduction="sum")
    p_neg = lp.gather(-1, negc).exp()
    ul = -(torch.log1p(-p_neg.clamp(max=1.0 - 1e-4)) * mskc).sum()
    return ce + alpha * ul


_head_piece_ul = torch.compile(_head_piece_ul_eager, dynamic=False)


class GatedLinear(nn.Module):
    """A linear layer whose output is scaled by a learnable scalar gate g (ReZero, Bachlechner et al. 2020). Grown
    layers are wrapped with g = 0: the model is exactly what it was, yet dLoss/dg is non-zero from the first step
    (the layer computes real features), so the model opens the gate as fast as the layer proves useful -- its
    ramp-in is its own. `.weight` / `.bias` return the GATED tensors, so code that reads them directly (the carrier
    applies F.linear(o, self.o.weight * ...)) sees the gate too."""
    def __init__(self, lin: nn.Linear):
        super().__init__()
        self.lin = lin
        self.g = nn.Parameter(torch.zeros((), device=lin.weight.device))
    @property
    def weight(self): return self.lin.weight * self.g
    @property
    def bias(self): return None if self.lin.bias is None else self.lin.bias * self.g
    def forward(self, x): return F.linear(x, self.weight, self.bias)


class LM(nn.Module):
    def __init__(self, vocab=8192, d=768, layers=12, mlp_mult=4, ctx=512, mixer="spin", norm="ln", short_conv=0,
                 attn_window=512, attn_sliding=False, attn_windows=None):
        """mixer: "spin" (plain carrier), "gspin" (input and output gates), "fspin" (input and recurrence
        gates: per-token forgetting), "aspin" (associative: content-addressed memory), "attn" (control)."""
        super().__init__()
        self.vocab, self.d, self.layers, self.ctx, self.mixer, self.norm = vocab, d, layers, ctx, mixer, norm
        self.attn_window, self.attn_sliding = attn_window, attn_sliding
        self.e = nn.Embedding(vocab, d)
        self.p = nn.Embedding(ctx, d)
        mk = {"spin": lambda d: SpinCarrier(d), "gspin": lambda d: SpinCarrier(d, gated="io"),
              "fspin": lambda d: SpinCarrier(d, gated="rf"), "aspin": AssocCarrier,
              "dspin": lambda d: AssocCarrier(d, delta=True),
              # FEWER, WIDER heads: the state per layer is h*(d/h)^2 = d^2/h, so four heads hold 2.7x the memory
              # of eleven at the same parameter count, for 12% of the speed (196 against 222 ktok/s).
              "wspin": lambda d: AssocCarrier(d, heads=4),
              # both fixes at once: wide heads and the delta write rule, which replaces a key's stored value
              # instead of accumulating onto it.
              "xspin": lambda d: AssocCarrier(d, delta=True, heads=4),
              # two heads: d^2/2 of state per layer, 5.5x the eleven-head carrier's. The wide (four-head) run
              # closed 18% of the long-context gap and the gain was larger at long context than at short, which
              # says the state is what is being exhausted; this is the next point on that curve.
              "w2spin": lambda d: AssocCarrier(d, heads=2),
              # HEAD DIM 128, the widest head the Triton kernels take. State per layer is d*128 against the
              # 64-dim carrier's d*64, so twice the memory AT KERNEL SPEED, where wspin's 176-dim heads fall
              # back to the torch path and run at 98 ktok/s against 335. At d768 that is six heads. These two
              # exist to ask, paired, whether the delta rule helps once the state is no longer the binding
              # constraint: at ctx 2048 the delta rule was 0.014 bits/byte ahead of the eleven-head carrier
              # and at ctx 8192 it was 0.026 behind the four-head wide one (2026-09-17).
              "hspin": lambda d: AssocCarrier(d, head_dim=128),
              "hdspin": lambda d: AssocCarrier(d, delta=True, head_dim=128),
              # SPARSE KEYS on top of the wide delta carrier: only the 8 largest coordinates of each 128-dim
              # key survive. The copy probe of 2026-09-17 put the carrier's exact recall at zero beyond 1,000
              # tokens against attention's +1.59 at 4,000, and the cause is key crosstalk, which sparsity
              # attacks directly. Everything else is identical to hdspin, so the pair isolates it.
              "sdspin": lambda d: AssocCarrier(d, delta=True, head_dim=128, key_topk=8),
              # the wide delta carrier with a structural floor under each head's forgetting
              "tspin": lambda d: AssocCarrier(d, delta=True, head_dim=128, hl_floor=True),
              # no write decision at all: every token replaces what its key held, at full strength
              "aspin1": lambda d: AssocCarrier(d, delta=True, head_dim=128, write_all=True),
              # the PLAIN rule with no write gate at all: every token is written, nothing is selected. No
              # chunk inverse, so nothing to make singular, which is what killed the delta version in 70 s.
              "hspin1": lambda d: AssocCarrier(d, head_dim=128, write_all=True),
              # and the same with one state-reading attention layer in four, which is where exact recall came
              # from; if the floor fixes general context use these two together close both halves
              "trspin": lambda d, i: (Attn(d, window=512, read_state=128) if i % 4 == 3 else
                                      AssocCarrier(d, delta=True, head_dim=128, hl_floor=True)),
              # ALTERNATING: odd layers are the wide delta carrier, even layers are attention over a 512-token
              # window. The copy probe of 2026-09-17 showed the carrier has no exact recall past 1,000 tokens
              # while attention holds +1.59 at 4,000; a bounded window buys back exact recall of the recent
              # past at linear cost and a fixed cache, and the carrier still carries everything older. Both
              # blocks are 4 d^2, so this has the same parameter count as twelve of either.
              "altspin": lambda d, i: (Attn(d, window=512) if i % 2 else
                                       AssocCarrier(d, delta=True, head_dim=128)),
              # ONE ATTENTION LAYER IN FOUR instead of one in two. The 1:2 hybrid recovered three quarters of
              # attention's exact recall but its context gain went POSITIVE (+0.0837 from 512 to 32k tokens,
              # measured 2026-09-17): six carrier layers cannot hold what twelve held. Nine carrier layers and
              # three windowed attention layers is the same question with the ratio moved back toward memory.
              "alt4spin": lambda d, i: (Attn(d, window=512) if i % 4 == 3 else
                                        AssocCarrier(d, delta=True, head_dim=128)),
              # the same 1:4 ratio, but the attention layers also put their own query into the state of the
              # carrier layer beneath them. This is the only variant so far that attacks long-range recall
              # rather than trading it for local recall.
              "rspin": lambda d, i: (Attn(d, window=512, read_state=128) if i % 4 == 3 else
                                     AssocCarrier(d, delta=True, head_dim=128)),
              # TEN carrier layers and two state-reading attention layers. The state-reading hybrid won exact
              # recall outright but its context gain was worse than the pure carrier's, and the five-model
              # table of 2026-09-18 showed context gain tracking the number of carrier layers exactly. This is
              # the same read-side mechanism with the ratio moved back toward memory.
              "r6spin": lambda d, i: (Attn(d, window=512, read_state=128) if i % 6 == 5 else
                                      AssocCarrier(d, delta=True, head_dim=128)),
              # GROWABLE: the rspin stack built so that widening is exactly function-preserving as well as
              # deepening. Attention heads are FIXED at 64 dimensions and multiply with d (8 at 512, 12 at 768,
              # 16 at 1024), so widening adds whole heads -- like the carrier -- instead of widening each head,
              # which changed every QK-normalised score (2026-09-23). No state read: it measured as dead weight.
              # Built with norm="rms": RMSNorm over zero-padded dimensions is an exact rescale, LayerNorm is not.
              "grspin": lambda d, i: (Attn(d, heads=d // 64, window=attn_window, sliding=attn_sliding) if i % 4 == 3 else
                                      AssocCarrier(d, delta=True, head_dim=128)),
              "attn": Attn}[mixer]
        self.mix = nn.ModuleList([mk(d, i) if mixer in ("altspin", "alt4spin", "rspin", "trspin", "r6spin", "grspin") else mk(d) for i in range(layers)])
        self.mlp = nn.ModuleList([MLP(d, mlp_mult) for _ in range(layers)])
        self.n = nn.LayerNorm(d)
        self.head_compile = True                    # tests flip it to compare against the eager form
        if norm == "rms":
            # every residual-stream LayerNorm becomes an RMSNorm of the same width (the per-head norms inside
            # the carrier and attention already are). This is what makes zero-padded widening exact.
            for mod in list(self.modules()):
                for nm, child in list(mod.named_children()):
                    if isinstance(child, nn.LayerNorm):
                        setattr(mod, nm, nn.RMSNorm(child.normalized_shape[0]))
        self.apply(self._init)
        # per-layer attention windows {layer index: window}, where they differ from attn_window (2026-10-03: one long-range
        # layer for retrieval while the others stay short, so the carrier still has to carry)
        self.attn_windows = {int(k_): int(v_) for k_, v_ in (attn_windows or {}).items()}
        for k_, v_ in self.attn_windows.items(): self.mix[k_].window = v_
        # short causal convolutions on every carrier's q, k, v (AssocCarrier.add_short_conv), identity at birth
        self.short_conv = 0
        if short_conv: self.add_short_conv(short_conv)

    def add_short_conv(self, K: int):
        """ADD A SHORT CAUSAL CONVOLUTION (K tokens) to every carrier layer, PRESERVING THE FUNCTION EXACTLY: each
        starts as the identity. See AssocCarrier.add_short_conv for why (in-context lookup, 2026-10-01)."""
        for mx in self.mix:
            if hasattr(mx, "add_short_conv"): mx.add_short_conv(K)
        self.short_conv = K
        return self

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

    def trunk(self, x, states=None, carry=False):
        """Everything but the output head: the normalised final representation, (B, T, d).

        `mem` carries the most recent carrier layer's per-chunk states forward, so an attention layer built
        with read_state can query the memory the layers beneath it wrote.
        carry=True (train.py --carry): `states` are the layers' states from the previous window of the same streams
        (None at a stream's start), and the new states are returned too -- training exactly as inference runs."""
        # modulo the table: a training window longer than the table wraps its positions, exactly as inference does
        # at every window boundary with the state carried (train.py --pos-table); identical for windows <= the table
        h = self.e(x) + self.p(torch.arange(x.shape[1], device=x.device) % self.p.num_embeddings)
        mem = None; new = []
        for i, (mx, mp) in enumerate(zip(self.mix, self.mlp)):
            if getattr(mx, "read_state", 0):
                o_, st_ = mx(h, mem=mem)
            else:
                o_, st_ = mx(h, states[i]) if states is not None else mx(h)
                if getattr(mx, "sent", None) is not None:
                    mem = mx.sent
            h = mp(o_); new.append(st_)
        return (self.n(h), new) if carry else self.n(h)

    def loss(self, h, y, chunks=4, ul_alpha=0.0, ul_window=128):
        """Cross-entropy from the trunk output WITHOUT materialising all the logits at once (goal 6).

        The logits are B x T x V; at B32 T1024 V16384 that is 2.1 GB in fp32 and several copies of it live
        through softmax and its backward, which is what ran a 36 GiB cap out of memory. The head is applied to
        the batch in chunks under activation checkpointing, so at most one chunk's logits exist at a time and
        they are recomputed in backward. The chunk itself is compiled (see _head_piece); the loop is not."""
        negf = mskf = None
        if ul_alpha:
            # the previous ul_window targets for each position, per sequence, with the true token masked out
            B_, T_ = y.shape
            W_ = min(ul_window, T_ - 1)
            off = torch.arange(1, W_ + 1, device=y.device)
            pos = torch.arange(T_, device=y.device).unsqueeze(1) - off.unsqueeze(0)      # (T, W)
            neg = y[:, pos.clamp(min=0)]                                                 # (B, T, W)
            msk = (pos >= 0).unsqueeze(0) & (neg != y.unsqueeze(-1))
            negf = neg.reshape(-1, W_); mskf = msk.reshape(-1, W_).to(torch.float32)
        y = y.reshape(-1); hf = h.reshape(-1, h.shape[-1]); n = hf.shape[0]
        E = self.e.weight
        cmp_ = h.is_cuda and self.head_compile
        piece = (_head_piece_ul if cmp_ else _head_piece_ul_eager) if ul_alpha else \
                (_head_piece if cmp_ else _head_piece_eager)
        total = 0.0
        for i in range(chunks):
            sl = slice(i * n // chunks, (i + 1) * n // chunks)
            args = (hf[sl], y[sl], negf[sl], mskf[sl], E, ul_alpha) if ul_alpha else (hf[sl], y[sl], E)
            total = total + torch.utils.checkpoint.checkpoint(piece, *args, use_reentrant=False)
        return total / n

    def forward(self, x, return_hidden=False, hidden_layer=None):
        """hidden_layer selects WHICH representation to return. The final one is what the output head reads,
        so it is the least informative place to look for signal the head has not already used."""
        # modulo the table: a training window longer than the table wraps its positions, exactly as inference does
        # at every window boundary with the state carried (train.py --pos-table); identical for windows <= the table
        h = self.e(x) + self.p(torch.arange(x.shape[1], device=x.device) % self.p.num_embeddings)
        grab = None; mem = None
        for i, (mx, mp) in enumerate(zip(self.mix, self.mlp)):
            if getattr(mx, "read_state", 0):
                h = mp(mx(h, mem=mem)[0])
            else:
                h = mp(mx(h)[0])
                if getattr(mx, "sent", None) is not None:
                    mem = mx.sent
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
    def _grow_assoc(self, dn: int):
        """Widen a model whose layers are associative carriers, attention blocks, or both (vision goal 7).

        The rule is the one the diagonal carrier's growth established: in the residual stream every WRITER of a
        new dimension starts at zero and every READER starts random, so the function is preserved on step zero
        and the new capacity has a gradient on step one. Here the private space inside a layer is a HEAD, and
        the head dimension is fixed by the kernels (64 or 128), so widening d adds whole new heads rather than
        widening the existing ones. New heads are new memory: their q, k, v rows are random writers into a
        private space, their output columns are zero readers of it, and their timescale and frequency
        parameters are drawn across the SAME range as the existing heads rather than appended past the end of
        a linspace, or the new memory would all be the fastest or all the slowest."""
        import math as _m
        d = self.d; k = dn - d
        fresh = LM(self.vocab, dn, self.layers, self.mlp[0].a.weight.shape[0] // d, self.ctx, self.mixer,
                   norm=getattr(self, "norm", "ln"), attn_window=getattr(self, "attn_window", 512), attn_sliding=getattr(self, "attn_sliding", False))
        fresh = fresh.to(self.e.weight.device)
        grown = {}
        def put(mod, name, new):
            old = getattr(mod, name); newp = nn.Parameter(new)
            setattr(mod, name, newp); grown[newp] = (old, tuple(old.shape))
        def widen(mod, name, src, zero_rows=False, zero_cols=False):
            old = getattr(mod, name).data
            new = src.detach().clone().to(old.dtype)
            if zero_rows: new[old.shape[0]:] = 0
            if zero_cols and new.dim() == 2: new[:, old.shape[1]:] = 0
            new[tuple(slice(0, s) for s in old.shape)] = old
            put(mod, name, new)
        def widen_ln(ln):
            ow = ln.weight.data
            w = torch.zeros(dn, device=ow.device, dtype=ow.dtype); w[:d] = ow * _m.sqrt(d / dn)
            if isinstance(ln, nn.RMSNorm):
                # EXACT: zero-padding lowers the RMS by exactly sqrt(d/dn), which the old weights absorb, and the
                # new dimensions are zero so they come out zero whatever their weight -- so they get the typical
                # scale rather than zero, and can pass signal as soon as something writes to them
                w[d:] = w[:d].median()
                put(ln, "weight", w); ln.normalized_shape = (dn,)
                return
            ob = ln.bias.data
            b = torch.zeros(dn, device=ob.device, dtype=ob.dtype); b[:d] = ob
            put(ln, "weight", w); put(ln, "bias", b); ln.normalized_shape = (dn,)
        widen(self.e, "weight", fresh.e.weight, zero_cols=True)
        widen(self.p, "weight", fresh.p.weight, zero_cols=True)
        self.e.embedding_dim = self.p.embedding_dim = dn
        for mx, fx in zip(self.mix, fresh.mix):
            if isinstance(mx, AssocCarrier):
                for nm in ("q", "k", "v"):                                    # writers into the new heads
                    widen(getattr(mx, nm), "weight", getattr(fx, nm).weight)
                widen(mx.o, "weight", fx.o.weight, zero_rows=True, zero_cols=True)   # readers of them: zero
                for nm in ("wa", "wi") + (("wb",) if mx.delta else ()):        # per-head gates: one row per head
                    widen(getattr(mx, nm), "weight", getattr(fx, nm).weight)
                    widen(getattr(mx, nm), "bias", getattr(fx, nm).bias)
                    getattr(mx, nm).in_features = dn; getattr(mx, nm).out_features = fx.h
                hn = fx.h - mx.h
                A_new = torch.linspace(float(mx.A.data.min()), float(mx.A.data.max()), hn, device=mx.A.device)
                put(mx, "A", torch.cat([mx.A.data, A_new]))
                put(mx, "theta", torch.cat([mx.theta.data, fx.theta.data[mx.h:]]))
                widen_ln(mx.ln)
                for nm in ("q", "k", "v", "o"):
                    getattr(mx, nm).in_features = getattr(mx, nm).out_features = dn
                mx.d = dn; mx.h = fx.h
            elif fx.h > mx.h:
                # A GROWABLE ATTENTION BLOCK (fixed 64-dimension heads): widening adds whole heads, exactly as the
                # carrier does. New heads' q, k, v rows are random, their output columns zero; every existing
                # head, and its QK-normalised scores, is untouched. The rows are laid out head by head, so the
                # new heads are exactly the appended rows.
                for nm in ("q", "k", "v"):
                    widen(getattr(mx, nm), "weight", getattr(fx, nm).weight)
                    widen(getattr(mx, nm), "bias", getattr(fx, nm).bias, zero_rows=True)
                widen(mx.o, "weight", fx.o.weight, zero_rows=True, zero_cols=True)
                widen(mx.o, "bias", fx.o.bias, zero_rows=True)
                widen_ln(mx.n)
                for nm in ("q", "k", "v", "o"):
                    getattr(mx, nm).in_features = getattr(mx, nm).out_features = dn
                mx.h = fx.h
            else:
                # AN ATTENTION BLOCK. Its heads stay at the same count and each head gets WIDER, so the new
                # dimensions land inside the dot product that makes the scores. Zero on both q and k would
                # never learn (each one's gradient is proportional to the other), so k's new rows are zero and
                # q's are random: the scores are unchanged on step zero and k has a gradient on step one.
                widen(mx.q, "weight", fx.q.weight); widen(mx.q, "bias", fx.q.bias, zero_rows=True)
                widen(mx.k, "weight", fx.k.weight, zero_rows=True)
                widen(mx.k, "bias", fx.k.bias, zero_rows=True)
                widen(mx.v, "weight", fx.v.weight); widen(mx.v, "bias", fx.v.bias, zero_rows=True)
                # QK-norm divides by the RMS over the head dimension, and appending zeros lowers that RMS, so
                # the surviving entries are scaled by sqrt(old/new) exactly as the LayerNorms are
                for nm, fn_ in (("qn", fx.qn), ("kn", fx.kn)):
                    ln_ = getattr(mx, nm); ow = ln_.weight.data; od = ow.shape[0]; nd = fn_.weight.shape[0]
                    w = torch.zeros(nd, device=ow.device, dtype=ow.dtype)
                    w[:od] = ow * _m.sqrt(od / nd)
                    put(ln_, "weight", w); ln_.normalized_shape = (nd,)
                widen(mx.o, "weight", fx.o.weight, zero_rows=True, zero_cols=True)
                widen(mx.o, "bias", fx.o.bias, zero_rows=True)
                if getattr(mx, "read_state", 0):
                    widen(mx.rq, "weight", fx.rq.weight)
                    mx.rq.in_features = mx.rq.out_features = dn
                widen_ln(mx.n)
                for nm in ("q", "k", "v", "o"):
                    getattr(mx, nm).in_features = getattr(mx, nm).out_features = dn
                mx.h = fx.h
        for mp, fp in zip(self.mlp, fresh.mlp):
            widen(mp.a, "weight", fp.a.weight); widen(mp.a, "bias", fp.a.bias, zero_rows=True)
            widen(mp.b, "weight", fp.b.weight, zero_rows=True, zero_cols=True)
            widen(mp.b, "bias", fp.b.bias, zero_rows=True)
            widen_ln(mp.n)
            mp.a.in_features = dn; mp.a.out_features = fp.a.out_features
            mp.b.in_features = fp.b.in_features; mp.b.out_features = dn
        widen_ln(self.n)
        self.d = dn
        return grown

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
        assert not getattr(self, "short_conv", 0), "growth of a model with short convolutions is not implemented yet"
        d, dn = self.d, d_new
        assert dn > d, (d, dn)
        assert self.mixer in ("spin", "gspin", "fspin", "aspin", "dspin", "hspin", "hdspin",
                              "altspin", "alt4spin", "rspin", "grspin", "attn"), self.mixer
        if self.mixer not in ("spin", "gspin", "fspin"):
            return self._grow_assoc(dn)
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

    def param_groups(self, lr=6e-3, spectral_lr=1e-3, new_lr=3e-4):
        """Spectral parameters take small ABSOLUTE steps, everything else takes trust-ratio steps.

        nu and theta are the per-channel timescale and frequency, not arbitrary weights, so moving them by a
        fraction of their own norm is meaningless. Measured: with ||nu_raw|| about 104, a 6e-3 trust fraction
        moved the spectrum by 0.6 per step and drove |lambda| from 0.999 to 0.9998 in thirty steps, then NaN.

        Layers added by deepen() form a third group, also ABSOLUTE steps: they start with their outputs at
        exactly zero, and a trust-ratio step is proportional to the parameter's own norm, floored at 1e-3, so a
        zero tensor would move by lr x 1e-3 per step and take ~100,000 steps to wake up."""
        new = set(getattr(self, "new_layers", ()))
        newp = [p for i in sorted(new) for p in list(self.mix[i].parameters()) + list(self.mlp[i].parameters())]
        nid = {id(p) for p in newp}
        spec = [p for mx in self.mix for p in mx.spectral_parameters() if id(p) not in nid]
        sid = {id(p) for p in spec}
        rest = [p for p in self.parameters() if id(p) not in sid and id(p) not in nid]
        groups = [{"params": rest, "lr": lr, "trust": True},
                  {"params": spec, "lr": spectral_lr, "trust": False, "factored": False}]
        if newp:
            # factored like every other weight matrix: copying the spectral group's dense second moment made the
            # captured optimizer step 431 ms instead of 21 (measured 2026-09-23), halving a deepened run's speed
            groups.append({"params": newp, "lr": new_lr, "trust": False, "factored": True})
        return groups

    def add_layers(self, k: int):
        """ADD k LAYERS ON TOP, PRESERVING THE FUNCTION EXACTLY (goal 7, continual learning). The same residual
        argument as deepen(): a layer whose output projections are zero is the identity, under any norm, so this
        works on the LayerNorm lineage that cannot be widened exactly. New knowledge can then be written into the
        new layers alone, with every old weight frozen, instead of over what was already known."""
        assert not getattr(self, "short_conv", 0), "growth of a model with short convolutions is not implemented yet"
        L = self.layers
        big = LM(self.vocab, self.d, L + k, self.mlp[0].a.weight.shape[0] // self.d, self.ctx, self.mixer,
                 norm=getattr(self, "norm", "ln"), attn_window=getattr(self, "attn_window", 512), attn_sliding=getattr(self, "attn_sliding", False)).to(self.e.weight.device)
        big.e.load_state_dict(self.e.state_dict()); big.p.load_state_dict(self.p.state_dict())
        big.n.load_state_dict(self.n.state_dict())
        for j in range(L):
            big.mix[j].load_state_dict(self.mix[j].state_dict()); big.mlp[j].load_state_dict(self.mlp[j].state_dict())
        big.new_layers = list(range(L, L + k))
        with torch.no_grad():
            for i in big.new_layers:
                big.mix[i].o.weight.zero_()
                if getattr(big.mix[i].o, "bias", None) is not None: big.mix[i].o.bias.zero_()
                big.mlp[i].b.weight.zero_(); big.mlp[i].b.bias.zero_()
        return big

    def deepen(self, init="zero"):
        """DOUBLE THE DEPTH, PRESERVING THE FUNCTION EXACTLY (vision goal 7).

        Widening a trained model is not function-preserving here (LayerNorm over padded zeros moved the loss
        1.980 -> 2.390 on run45, 2026-09-23). Depth can be: every layer is residual, x + f(x), so a new layer
        whose output projections are zero is exactly the identity, and the deeper model computes precisely what
        the shallower one did. It then learns from there.

        Layout for the 1-in-4 attention stack, chosen so the result is an ordinary 2L-layer model of the same
        mixer and reloads with LM(**cfg): the old layers keep their order and their types, landing on positions
        0,1,2,7,8,9,10,15,... of 24, and the new ones fill the gaps as blocks of attention + three carriers
        spread through the depth. Returns the new model; its .new_layers lists the added indices, which
        param_groups() trains with absolute steps."""
        assert not getattr(self, "short_conv", 0), "growth of a model with short convolutions is not implemented yet"
        L = self.layers
        assert self.mixer in ("rspin", "grspin"), self.mixer
        big = LM(self.vocab, self.d, 2 * L, self.mlp[0].a.weight.shape[0] // self.d, self.ctx, self.mixer,
                 norm=getattr(self, "norm", "ln"), attn_window=getattr(self, "attn_window", 512), attn_sliding=getattr(self, "attn_sliding", False))
        big = big.to(self.e.weight.device)
        # old layer j keeps its type: rspin puts attention at i % 4 == 3 in both stacks
        oldpos, p = [], 0
        for j in range(L):
            want_attn = (j % 4 == 3)
            while (p % 4 == 3) != want_attn or (want_attn and (p // 4) % 2 == 0):
                p += 1
            oldpos.append(p); p += 1
        assert len(set(oldpos)) == L and max(oldpos) < 2 * L, oldpos
        big.e.load_state_dict(self.e.state_dict()); big.p.load_state_dict(self.p.state_dict())
        big.n.load_state_dict(self.n.state_dict())
        for j, q in enumerate(oldpos):
            big.mix[q].load_state_dict(self.mix[j].state_dict())
            big.mlp[q].load_state_dict(self.mlp[j].state_dict())
        big.new_layers = sorted(set(range(2 * L)) - set(oldpos))
        big.copied_from = {}
        with torch.no_grad():
            if init in ("copy", "gate"):
                # STACKING (Du et al., NeurIPS 2024, "Stacking your transformers"): each new layer starts as a COPY of
                # the nearest trained layer of its own type before it (after it if none), so it carries useful
                # features from its first step. Not function-preserving; the growth trial measures whether it pays.
                # Measured need (2026-09-27): exact zero-output layers switch on so slowly that no growth trial in
                # run60 found growing worth its compute.
                for i in big.new_layers:
                    same = [q for q in oldpos if (q % 4 == 3) == (i % 4 == 3)]
                    before = [q for q in same if q < i]
                    q = max(before) if before else min(same)
                    big.mix[i].load_state_dict(big.mix[q].state_dict()); big.mlp[i].load_state_dict(big.mlp[q].state_dict())
                    big.copied_from[i] = q
                if init == "gate":
                    # GATED STACKING: the copies sit behind gates at zero -- exact at the growth (the plain copies
                    # jumped the loss 3.07 -> 4.81), useful features from the first step, opened by the gradient
                    for i in big.new_layers:
                        big.mix[i].o = GatedLinear(big.mix[i].o); big.mlp[i].b = GatedLinear(big.mlp[i].b)
            else:
                for i in big.new_layers:
                    big.mix[i].o.weight.zero_()
                    if getattr(big.mix[i].o, "bias", None) is not None: big.mix[i].o.bias.zero_()
                    big.mlp[i].b.weight.zero_(); big.mlp[i].b.bias.zero_()
        return big
