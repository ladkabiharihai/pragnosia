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


_BLOCK = 16      # rows per forward-substitution block in the delta solve


@torch.compiler.disable
def _delta_solve(rhs, N, C, dtype, device):
    """Excluded from torch.compile: the block loop unrolls into a graph inductor spent 30 minutes on and never
    finished. Eager matmuls are fast; only this function breaks the graph."""
    b = min(_BLOCK, C)
    W = torch.zeros_like(rhs)
    for i0 in range(0, C, b):
        i1 = min(i0 + b, C)
        r = rhs[:, :, i0:i1]
        if i0:
            r = r - N[:, :, i0:i1, :i0] @ W[:, :, :i0]
        R = -N[:, :, i0:i1, i0:i1]
        X = torch.eye(i1 - i0, device=device, dtype=dtype)
        m_ = 1
        while m_ < (i1 - i0):
            X = X + R @ X
            R = R @ R
            m_ *= 2
        W = torch.cat([W[:, :, :i0], X @ r, W[:, :, i1:]], dim=2)
    return W


def _chunk(qc, kc, vc, lc, S, mm=None, bc=None):
    """One chunk: outputs and the next state. Factored out so it can be recomputed in backward.

    mm is the dtype of the four matmuls. The decay arithmetic (cumulative sums, exponentials) and the state S
    stay in the input dtype, which is fp32 or fp64; only the products run in mm. In bf16 that is the difference
    between tensor cores and not: measured 2.3x slower than attention at 219M with everything in fp32, where
    the FLOP count says this should be CHEAPER than attention at a 1024-token window (intra-chunk 2TCd plus
    state 2Td*dk against attention's 2T^2d).

    S accumulates over the whole sequence and is never demoted; the C x C score matrix is transient and is."""
    C = qc.shape[2]
    L = lc.cumsum(2)                                                              # (B,H,C), <= 0 and decreasing
    mask = torch.ones(C, C, device=qc.device, dtype=torch.bool).tril()
    dec = torch.exp((L.unsqueeze(3) - L.unsqueeze(2)).clamp(max=0.0)) * mask     # exp(L_t - L_s), s <= t
    eL = torch.exp(L).unsqueeze(3)                                                # exp(L_t): decay from chunk start
    w = torch.exp((L[:, :, -1:] - L).clamp(max=0.0)).unsqueeze(3)                 # (B,H,C,1)
    if bc is not None:
        # DELTA WRITE. The plain rule adds k v^T for every token, so two values written under similar keys leave
        # their sum and neither reads back: the associative interference this architecture loses to attention on
        # exact-retrieval text. The delta rule writes v MINUS what the memory already returns for that key, so a
        # key's value is replaced rather than accumulated.
        #
        # Each token's correction has to see the writes of the tokens before it in the same chunk. Correcting
        # them all against the chunk's incoming state instead (Jacobi) overshoots when keys are correlated: it
        # is stable only for beta < 2/chunk, and at chunk 64 the loss went non-finite at step 128. Writing the
        # recursion out gives, with W the residuals actually written and A[t,s] = (k_t . k_s) exp(L_t - L_s),
        #     (I + diag(beta) strictly_lower(A)) W = diag(beta) (V - exp(L) K S),
        # a unit lower triangular solve: exact, parallel over the chunk, and contractive for any beta in (0,1).
        # The result does not depend on the chunk size, which is the test.
        rhs = bc.unsqueeze(-1) * (vc - eL * (kc @ S))
        N = bc.unsqueeze(-1) * ((kc @ kc.transpose(2, 3)) * dec).tril(-1)         # strictly lower, nilpotent
        # Solve (I + N) W = rhs by forward substitution in BLOCKS of b rows.
        #
        # Two faster-looking routes were tried and both failed. torch.linalg.solve_triangular issues one tiny
        # factorisation per chunk per head per layer, about 67,000 per step, and training crawled at 17 ktok/s.
        # Summing the Neumann series sum_k (-N)^k by doubling is exact in exact arithmetic, N being nilpotent,
        # but its intermediate powers reach norm C^k before they vanish and fp32 cancels: chunk 16 and chunk 256
        # disagreed by 1.1. Blocked substitution keeps every intermediate bounded, is exact, and is all matmuls.
        vc = _delta_solve(rhs, N, C, qc.dtype, qc.device)
    if mm is None:
        A = (qc @ kc.transpose(2, 3)) * dec
        o = A @ vc + (qc * eL) @ S
        S = eL[:, :, -1:] * S + (kc * w).transpose(2, 3) @ vc                    # eL[:, :, -1:] is (B,H,1,1)
        return o, S
    qm, km, vm = qc.to(mm), kc.to(mm), vc.to(mm)
    A = (qm @ km.transpose(2, 3)).to(dec.dtype) * dec                            # scores fp32-accumulated, decay exact
    o = (A.to(mm) @ vm).to(qc.dtype) + ((qc * eL).to(mm) @ S.to(mm)).to(qc.dtype)
    S = eL[:, :, -1:] * S + ((kc * w).transpose(2, 3).to(mm) @ vm).to(S.dtype)
    return o, S


def _scan_twopass(q, k, v, la, S, chunk, mm):
    """The plain rule as two BATCHED passes around one trivial loop.

    The chunk-by-chunk version did every matmul inside a Python loop over T/C chunks, so nothing batched across
    chunks and the GPU idled between launches: 154 ktok/s for the two-head carrier against attention's ~220 at
    the same shape. Only one thing in the recurrence is sequential, the state carried from chunk to chunk, and
    that step is S_c = a_c S_{c-1} + KV_c, a scaled ADD with no matmul in it. So:
      pass 0, batched over all chunks: the C x C decayed scores, the intra-chunk output, and each chunk's
              contribution KV_c = (K * w)^T V and total decay a_c = exp(L_C);
      pass 1, the loop: NC scaled adds producing the state ENTERING every chunk;
      pass 2, batched: O = intra + (Q * eL) @ S_entering.
    Identical result to the sequential form (tests), all matmuls batched, the loop nearly free."""
    B, H, T, D = q.shape
    C = min(chunk, T)
    pad = (-T) % C
    if pad:                                                                        # zero tokens with alpha = 1
        z = lambda t: F.pad(t, (0, 0, 0, pad)) if t.dim() == 4 else F.pad(t, (0, pad))
        q, k, v, la = z(q), z(k), z(v), z(la)
    NC = q.shape[2] // C
    qc, kc, vc = (t.view(B, H, NC, C, D) for t in (q, k, v))
    lc = la.view(B, H, NC, C)
    L = lc.cumsum(3)                                                               # (B,H,NC,C), <= 0
    mask = torch.ones(C, C, device=q.device, dtype=torch.bool).tril()
    dec = torch.exp((L.unsqueeze(4) - L.unsqueeze(3)).clamp(max=0.0)) * mask       # (B,H,NC,C,C)
    eL = torch.exp(L).unsqueeze(4)                                                 # (B,H,NC,C,1)
    w = torch.exp((L[..., -1:] - L).clamp(max=0.0)).unsqueeze(4)                   # (B,H,NC,C,1)
    a = torch.exp(L[..., -1])                                                      # (B,H,NC): decay over the chunk
    if mm is None:
        A = (qc @ kc.transpose(3, 4)) * dec
        intra = A @ vc
        KV = (kc * w).transpose(3, 4) @ vc                                        # (B,H,NC,D,D)
    else:
        qm, km, vm = qc.to(mm), kc.to(mm), vc.to(mm)
        A = (qm @ km.transpose(3, 4)).to(dec.dtype) * dec
        intra = (A.to(mm) @ vm).to(q.dtype)
        KV = ((kc * w).transpose(3, 4).to(mm) @ vm).to(q.dtype)
    # pass 1: the only sequential step, and it is a scaled add
    ent = []
    for c in range(NC):
        ent.append(S)
        S = a[:, :, c, None, None] * S + KV[:, :, c]
    Sent = torch.stack(ent, 2)                                                     # (B,H,NC,D,D): state entering each chunk
    if mm is None:
        carried = (qc * eL) @ Sent
    else:
        carried = ((qc * eL).to(mm) @ Sent.to(mm)).to(q.dtype)
    o = (intra + carried).reshape(B, H, NC * C, D)[:, :, :T]
    return o, S


def _unit_lower_inverse(N, b=16):
    """X = (I + N)^-1 for a batch of strictly-lower N (..., C, C), by blocked forward substitution.

    Every intermediate stays bounded (the Neumann series over the whole C x C did not: its powers reach norm
    C^k before vanishing and fp32 cancelled). Within a b x b diagonal block the series is summed by doubling,
    exact after log2(b) steps because the block is nilpotent; across blocks it is substitution.

    Batched over the BLOCKS as well as the leading dimensions: the diagonal blocks are inverted all at once,
    and the substitution runs one step per block row with a single matmul each. The earlier version looped
    over blocks in Python and, measured, was the delta rule's entire cost (1.43 ms of a 1.81 ms forward; the
    Triton loop kernel it feeds is 0.08 ms)."""
    C = N.shape[-1]
    if C % b:
        b = C                                                                   # tiny chunks (tests): one block
    nb = C // b
    lead = N.shape[:-2]
    Nb = N.reshape(*lead, nb, b, nb, b).transpose(-3, -2)                       # (..., nb, nb, b, b): [i, j]
    idx = torch.arange(nb, device=N.device)
    Ndiag = Nb[..., idx, idx, :, :]                                             # (..., nb, b, b)
    # all diagonal blocks at once: X_ii = sum_k (-N_ii)^k, doubled
    R = -Ndiag
    Xd = torch.eye(b, device=N.device, dtype=N.dtype).expand_as(R)
    m_ = 1
    while m_ < b:
        Xd = Xd + R @ Xd
        R = R @ R
        m_ *= 2
    # forward substitution over block rows: X[i, :] = X_ii (E_i - sum_{j<i} N_ij X[j, :])
    rows = []
    E = torch.eye(C, device=N.device, dtype=N.dtype).expand(*lead, C, C)
    for i in range(nb):
        rhs = E[..., i * b:(i + 1) * b, :]
        if i:
            below = torch.cat(rows, dim=-2)                                     # (..., i*b, C)
            rhs = rhs - N[..., i * b:(i + 1) * b, :i * b] @ below
        rows.append(Xd[..., i, :, :] @ rhs)
    return torch.cat(rows, dim=-2)


def _scan_twopass_delta(q, k, v, la, be, S, chunk, mm, kernel=False):
    """The DELTA rule in the same two-pass shape.

    S_t = a_t S_{t-1} + b_t k_t (v_t - k_t S_{t-1})^T: a key's stored value is replaced, not accumulated onto.
    Within a chunk the residuals actually written, W, satisfy (I + diag(b) strictly_lower(K K^T dec)) W =
    diag(b)(V - eL K S_entering). The matrix does NOT depend on the state, so its inverse X is computed for all
    chunks at once in pass 0; only the right-hand side needs S_entering, so pass 1 is, per chunk, two small
    matmuls (K S and X rhs) and the scaled add. This is what makes the rule affordable: the earlier per-token
    and per-chunk solves ran at 17 and 6 ktok/s."""
    B, H, T, D = q.shape
    C = min(chunk, T)
    pad = (-T) % C
    if pad:
        z = lambda t: F.pad(t, (0, 0, 0, pad)) if t.dim() == 4 else F.pad(t, (0, pad))
        q, k, v, la, be = z(q), z(k), z(v), z(la), z(be)          # padded beta = 0: writes nothing
    NC = q.shape[2] // C
    qc, kc, vc = (t.view(B, H, NC, C, D) for t in (q, k, v))
    lc, bc = la.view(B, H, NC, C), be.view(B, H, NC, C)
    L = lc.cumsum(3)
    mask = torch.ones(C, C, device=q.device, dtype=torch.bool).tril()
    dec = torch.exp((L.unsqueeze(4) - L.unsqueeze(3)).clamp(max=0.0)) * mask
    eL = torch.exp(L).unsqueeze(4)
    w = torch.exp((L[..., -1:] - L).clamp(max=0.0)).unsqueeze(4)
    a = torch.exp(L[..., -1])
    KK = (kc @ kc.transpose(3, 4)) * dec
    X = _unit_lower_inverse(bc.unsqueeze(-1) * KK.tril(-1))                       # (B,H,NC,C,C), batched
    A = None if (kernel and q.is_cuda and D in (64, 128) and C == 64) else (qc @ kc.transpose(3, 4)) * dec
    kw = (kc * w).transpose(3, 4)                                                  # (B,H,NC,D,C)
    ent = []; Ws = []
    for c in range(NC):
        ent.append(S)
        rhs = bc[:, :, c].unsqueeze(-1) * (vc[:, :, c] - eL[:, :, c] * (kc[:, :, c] @ S))
        Wc = X[:, :, c] @ rhs
        Ws.append(Wc)
        S = a[:, :, c, None, None] * S + kw[:, :, c] @ Wc
    Sent = torch.stack(ent, 2); W = torch.stack(Ws, 2)
    if kernel and q.is_cuda and D in (64, 128) and C == 64:
        # the C x C part through the Triton kernel, with the residuals W in the value slot: it recomputes the
        # decayed scores in backward instead of saving them, which is where the 27 GiB went
        from .assoc_kernel import _op_out
        o = _op_out(q.to(torch.bfloat16), k.to(torch.bfloat16), W.reshape(B, H, NC * C, D).to(torch.bfloat16),
                    la, Sent, C)[:, :, :T]
    else:
        o = (A @ W + (qc * eL) @ Sent).reshape(B, H, NC * C, D)[:, :, :T]
    return o, S


def assoc_scan(q, k, v, logalpha, S0=None, chunk: int = 64, recompute: bool = False, mm=None, beta=None, kernel=False):
    """o_t = sum_{s<=t} exp(L_t - L_s) (q_t . k_s) v_s  [+ exp(L_t) q_t . S0], chunked and stable.

    q, k, v: (B, H, T, D); logalpha: (B, H, T) <= 0 (log of the per-token forgetting); S0: (B, H, D, D) or None.
    Returns o: (B, H, T, D) and the final state S: (B, H, D, D). fp32 inside."""
    B, H, T, D = q.shape
    rt = torch.float64 if q.dtype == torch.float64 else torch.float32
    q, k, v, la = q.to(rt), k.to(rt), v.to(rt), logalpha.to(rt)
    S = torch.zeros(B, H, D, D, device=q.device, dtype=rt) if S0 is None else S0.to(rt)
    fn, args = (_scan_twopass, (q, k, v, la, S, chunk, mm)) if beta is None else \
               (_scan_twopass_delta, (q, k, v, la, beta.to(rt), S, chunk, mm, kernel))
    if recompute and torch.is_grad_enabled():
        o, S = torch.utils.checkpoint.checkpoint(fn, *args, use_reentrant=False)
    else:
        o, S = fn(*args)
    return o.to(q.dtype), S


# Use the fused flash-linear-attention scan where it is exact and nothing needs the per-chunk states. Off
# unless asked for, so every existing measurement keeps its kernel; see the note at the dispatch below.
FUSED_SCAN = False
# One matmul for the carrier's five input projections instead of five; see the note in forward().
FUSED_PROJ = False


class AssocCarrier(nn.Module):
    def __init__(self, d: int, heads: int | None = None, chunk: int = 64, recompute: bool = False,
                 delta: bool = False, head_dim: int | None = None, kernel: bool = True, key_topk: int = 0,
                 hl_floor: bool = False, write_all: bool = False):
        super().__init__()
        self.d = d; self.recompute = recompute; self.delta = delta
        # SPARSE KEYS. An outer-product memory with dense unit keys in d_k dimensions holds about d_k patterns
        # before crosstalk swamps the read: two random unit vectors in 128 dimensions have expected |cos| of
        # 0.088, so n stored pairs give signal 1 against noise growing as sqrt(n/d_k), and by 8,192 tokens the
        # noise is 8x the signal. Measured 2026-09-17: the carrier's copy gain halves by 1,000 tokens of
        # separation and goes NEGATIVE by 4,000, where attention still holds +1.59. Keeping only the largest
        # key_topk coordinates of each key means two keys collide only when they share an active coordinate,
        # so capacity rises combinatorially rather than linearly. 0 keeps the dense key.
        self.key_topk = key_topk
        # A FLOOR ON EACH HEAD'S HALF-LIFE. Measured on 2026-09-18, the trained model's heads had half-lives of
        # 2, 5, 19, 246, 250 and 217 tokens, and the slowest head anywhere in twelve layers was 1,502: the
        # initialisation spreads them from 7 to 7,000 tokens and training collapses every one of them, because
        # a model trained at 2,048 tokens is never rewarded for remembering further. The state was not full
        # when this was measured, it was using 18 of its 128 directions, so the limit on long context was
        # forgetting and not capacity. This makes the spread structural: head h may not forget faster than its
        # floor, so some heads hold a clause and others hold a chapter, and no amount of short-context training
        # can collapse them together.
        self.hl_floor = hl_floor
        # WRITE EVERYTHING. Measured 2026-09-18: the state uses 18 of its 128 directions, its heads forget
        # within 250 tokens, forcing them not to forget changes nothing, and the model writes into its slowest
        # heads at strength 0.15 whether or not they are slow. Capacity is not the constraint and decay is not
        # the constraint; the WRITE POLICY is. Attention decides what to retrieve at read time, with the query
        # in hand; this architecture has to decide what to keep at write time, thousands of tokens before it
        # learns whether it mattered. This removes the decision: every token is written at full strength, which
        # wastes capacity the model was not using anyway, and tests whether the policy was the thing holding
        # long context back.
        self.write_all = write_all
        # set False when no later layer reads this memory, which lets the fused scan run (it never
        # materialises the per-chunk states); True keeps our own kernel, which writes them
        self.need_sent = True
        # kernel: the Triton scan (src/assoc_kernel.py) when it applies (CUDA, plain rule, head dim 64 or 128,
        # more than one token); the PyTorch two-pass form otherwise, and always for the T=1 decode step, which
        # is CUDA-graphed already and has nothing for a chunk kernel to do.
        self.kernel = kernel
        self.bv = 32                                  # state columns per program in the delta loop kernel
        self.h = heads or max(1, d // (head_dim or 64))
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
        # the delta rule's write strength: 0 leaves the memory alone, 1 replaces this key's value outright
        if delta:
            self.wb = nn.Linear(d, self.h)
        self.A = nn.Parameter(torch.linspace(math.log(0.1 / 0.69), math.log(1e-4 / 0.69), self.h))
        if hl_floor:
            # geometric from 8 tokens to 131,072: the fastest head still tracks a clause, the slowest keeps
            # essentially everything within any window this project trains on
            hl = torch.logspace(math.log10(8.0), math.log10(131072.0), self.h)
            self.register_buffer("la_floor", -math.log(2.0) / hl, persistent=False)
        # the spin: rotation frequencies per channel pair, learned; geometric init like RoPE (base 10000)
        base = 10000.0 ** (-torch.arange(0, self.dk, 2).float() / self.dk)
        self.theta = nn.Parameter(base.unsqueeze(0).repeat(self.h, 1))
        self.on = nn.RMSNorm(self.dk)
        self.gate = nn.Parameter(torch.zeros(()))
        self.sent = None                            # set by forward on the kernel path; read by rspin layers

    def spectral_parameters(self):
        """Timescales and frequencies take small absolute steps, as in the diagonal carrier."""
        return [self.A, self.theta, self.gate]

    def logalpha(self, z):
        la = -F.softplus(self.wa(z)) * torch.exp(self.A)                          # (B,T,H), <= 0
        if self.hl_floor:
            la = la.clamp(min=self.la_floor)                                      # no faster than the floor
        return la

    def forward(self, x, h0=None, pos0: int = 0):
        """h0: (state S (B,H,dk,dk), position offset) from a previous window, or None."""
        B, T, D = x.shape
        z = self.ln(x)
        S0, p0 = (None, pos0) if h0 is None else h0
        # p0 may be a 0-d tensor: a CUDA-graphed decode step needs the position on the device, not baked in
        pos = p0 + torch.arange(T, device=x.device)
        kern = self.kernel and x.is_cuda and self.dk in (64, 128) and T > 1
        if FUSED_PROJ and self.delta and not self.key_topk:
            # ONE MATMUL FOR FIVE PROJECTIONS. q, k, v, the decay gate and the write strength all read z; as five
            # Linear calls z is read from VRAM five times and each product is only d or h wide, the shape tensor
            # cores handle worst. Concatenated, z is read once and the product is 3d + 2h wide. Same parameters,
            # same function (the separate modules are kept so checkpoints and analysis scripts are unchanged).
            W = torch.cat([self.q.weight, self.k.weight, self.v.weight, self.wa.weight, self.wb.weight])
            y = F.linear(z, W)
            qo, ko, vo, ao, bo = y.split([D, D, D, self.h, self.h], dim=-1)
            kp = ko.view(B, T, self.h, self.dk)
            self._fused = (qo, vo, ao + self.wa.bias, bo + self.wb.bias)
        else:
            self._fused = None
            kp = self.k(z).view(B, T, self.h, self.dk)
        if self.key_topk:
            # straight through: the mask selects, the surviving coordinates keep their gradient unchanged and
            # the dropped ones get none, which is what a selection should do
            m = torch.zeros_like(kp).scatter_(-1, kp.abs().topk(self.key_topk, dim=-1).indices, 1.0)
            kp = kp * m
        if kern:
            # normalise, gate, rotate and cast in one kernel each way, straight from the projections' layout
            # into the scan kernels' (B,H,T,D); measured against the torch form below in tests/test_assoc.py
            from .assoc_kernel import prep_kernel
            if self._fused is not None:
                qp, vp = self._fused[0].view(B, T, self.h, self.dk), self._fused[1].view(B, T, self.h, self.dk)
            else:
                qp, vp = self.q(z).view(B, T, self.h, self.dk), self.v(z).view(B, T, self.h, self.dk)
            gate = None if (self.delta or self.write_all) else self.wi(z)   # write_all: no gate at all
            q, k, v = prep_kernel(qp, kp, vp, gate, self.theta, pos)
        else:
            qz = self._fused[0] if self._fused is not None else self.q(z)
            vz = self._fused[1] if self._fused is not None else self.v(z)
            q, k, v = [t.view(B, T, self.h, self.dk).transpose(1, 2)
                       for t in (qz, kp.reshape(B, T, self.d), vz)]
            q, k, v = F.normalize(q, dim=-1), F.normalize(k, dim=-1), F.normalize(v, dim=-1)   # scores in [-1, 1], unit writes
            if not self.delta and not self.write_all:
                k = k * torch.sigmoid(self.wi(z)).transpose(1, 2).unsqueeze(-1)  # write gate per head
            q, k = rotate(q, self.theta, pos), rotate(k, self.theta, pos)
        if self._fused is not None:
            la = -F.softplus(self._fused[2]) * torch.exp(self.A)
            if self.hl_floor: la = la.clamp(min=self.la_floor)
            la = la.transpose(1, 2)                                               # (B,H,T)
            bt = torch.sigmoid(self._fused[3]).transpose(1, 2)
        else:
            la = self.logalpha(z).transpose(1, 2)                                 # (B,H,T)
            bt = torch.sigmoid(self.wb(z)).transpose(1, 2) if self.delta else None # (B,H,T)
        mm = torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled("cuda") and x.is_cuda else None
        self._fused = None                                                        # do not keep activations alive
        if self.delta and self.write_all:
            # beta = 1 is NOT a safe way to say "write everything": the delta rule's chunk inverse is
            # (I + N)^-1 with N = beta * strictly_lower(K K^T dec), and at beta = 1 two similar keys inside one
            # chunk make it singular. Measured 2026-09-18: the weights went non-finite in 70 seconds. Floored
            # well below 1 instead, which still removes most of the decision without breaking the solve.
            bt = bt.clamp(min=0.75)
        if kern:
            if self.delta and FUSED_SCAN and not self.need_sent:
                # THE FUSED KERNEL (flash-linear-attention's chunked gated delta rule). Identical maths to ours --
                # measured 2026-09-23 against the float32 reference, 0.48% relative on the output and 0.38% on
                # the final state, the same bf16 error as our own kernel -- and 1.50-1.67x faster forward and
                # backward at every shape we train, because it keeps the state on-chip and recomputes chunk
                # states in the backward pass instead of writing one to VRAM per chunk and reading it back. It
                # does not produce those per-chunk states, so it is used only when no later layer reads them.
                from fla.ops.gated_delta_rule import chunk_gated_delta_rule
                t = lambda x: x.transpose(1, 2).contiguous()
                o, S = chunk_gated_delta_rule(t(q), t(k), t(v), t(la).float(), t(bt).float(), scale=1.0,
                                              initial_state=S0, output_final_state=True)
                o = o.transpose(1, 2); sent = None
            elif self.delta:
                from .assoc_kernel import delta_scan_kernel
                o, S, sent = delta_scan_kernel(q, k, v, la, bt, S0, chunk=self.chunk if self.chunk in (64, 128) else 64,
                                               bv=self.bv)
            else:
                from .assoc_kernel import assoc_scan_kernel
                o, S, sent = assoc_scan_kernel(q, k, v, la, S0)
            o = o.to(qp.dtype)          # the projections' dtype: bf16 under autocast (no copy), fp32 otherwise
        else:
            o, S = assoc_scan(q, k, v, la, S0, chunk=self.chunk, recompute=self.recompute, mm=mm, beta=bt,
                              kernel=self.kernel and self.dk in (64, 128))
            sent = None
        o = self.on(o).transpose(1, 2).reshape(B, T, D)
        # the output gate on the projection's WEIGHT, not its output: as a multiply on the output it kept that
        # output alive for the gate's gradient, 11 MiB per layer at B4 T2048; on the weight it is 1 MiB, the
        # same function
        # the per-chunk entering states, kept for a later layer that reads this memory with its own queries
        self.sent = sent
        return x + F.linear(o, self.o.weight * torch.sigmoid(self.gate)), (S, p0 + T)
