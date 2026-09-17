"""The chunked associative scan as Triton kernels: the plain rule, forward and backward.

Why. With the scan written in PyTorch and fused by inductor, the 78M carrier trains at 275 ktok/s and 5.9 GiB
at ctx 2048; with the scan bypassed it would run at 472 ktok/s and 3.4 GiB, against attention's 312 and 4.1.
So the scan alone costs 42% of the step and 2.5 GiB, all of it fp32 intermediates that inductor materialises:
the C x C decay and score matrices, the state entering every chunk, fp32 copies of q, k and v. Computed in
registers, they cost nothing to store. The ceiling is a carrier 1.5x faster and leaner than attention at this
window, and more at longer ones. The delta rule needs the same kernel.

What. For one (batch, head), with per-token log-decay la_t <= 0 and L the within-chunk cumulative sum:
    o_t = sum_{s<=t, same chunk} exp(L_t - L_s) (q_t . k_s) v_s + exp(L_t) q_t S_c
    S_{c+1} = exp(L_C) S_c + sum_s exp(L_C - L_s) k_s v_s^T
Every exponent is <= 0. The state entering each chunk is NOT computed by a recurrence: with scalar per-chunk
decays its solution is a weighted prefix sum, S_ent[c] = sum_{j<c} exp(A_{c-1} - A_j) KV_j, one small
lower-triangular matmul over chunks per (batch, head), all weights <= 1; the reverse recurrence for dS is the
matching suffix sum. That leaves one Triton kernel per direction, both fully parallel over chunks: the output
(the C x C decayed scores against V, plus the read of the entering state) and the per-chunk gradients of q, k,
v and the decay, with dla the reverse cumulative sum of dL inside the chunk. The reference is src/assoc.py's two-pass form, itself exact against the sequential
definition; tests/test_assoc.py holds both to it.

Constraints. Chunk C = 64. Head dim D in {64, 128} (Triton block shapes are powers of two), which fixes the
number of heads at d/D. Inputs any float dtype; the matmuls run in bf16 with fp32 accumulation, the decay and
the state in fp32.
"""
from __future__ import annotations
import torch
import triton
import triton.language as tl

C = 64
_LOOP = {"nw_fwd": 4, "nw_bwd": 4, "ns_fwd": 3, "ns_bwd": 3}     # loop-kernel launch knobs, set by the sweep


_CFGS = [triton.Config({}, num_warps=w, num_stages=st) for w in (2, 4, 8) for st in (1, 2, 3)]


@triton.autotune(configs=_CFGS, key=["T", "D"])
@triton.jit
def _fwd_out(q_ptr, k_ptr, v_ptr, la_ptr, sent_ptr, o_ptr,
             T, NC, s_bh_t, s_bh_l, D: tl.constexpr, BC: tl.constexpr):
    """Output of one chunk, one program per (batch-head, chunk)."""
    bh = tl.program_id(0); c = tl.program_id(1)
    rt = tl.arange(0, BC); rd = tl.arange(0, D)
    t0 = c * BC
    tm = (t0 + rt) < T
    la = tl.load(la_ptr + bh * s_bh_l + t0 + rt, mask=tm, other=0.0).to(tl.float32)
    L = tl.cumsum(la, 0)
    off = bh * s_bh_t + (t0 + rt)[:, None] * D + rd[None, :]
    q = tl.load(q_ptr + off, mask=tm[:, None], other=0.0)
    k = tl.load(k_ptr + off, mask=tm[:, None], other=0.0)
    v = tl.load(v_ptr + off, mask=tm[:, None], other=0.0)
    S = tl.load(sent_ptr + (bh * NC + c) * D * D + rd[:, None] * D + rd[None, :])
    # decayed causal scores, in registers
    dec = tl.exp(tl.minimum(L[:, None] - L[None, :], 0.0))
    dec = tl.where(rt[:, None] >= rt[None, :], dec, 0.0)
    A = tl.dot(q.to(tl.bfloat16), tl.trans(k.to(tl.bfloat16)), out_dtype=tl.float32) * dec
    o = tl.dot(A.to(tl.bfloat16), v.to(tl.bfloat16), out_dtype=tl.float32)
    qe = (q.to(tl.float32) * tl.exp(L)[:, None]).to(tl.bfloat16)
    o += tl.dot(qe, S.to(tl.bfloat16), out_dtype=tl.float32)
    tl.store(o_ptr + off, o.to(o_ptr.dtype.element_ty), mask=tm[:, None])


@triton.autotune(configs=_CFGS, key=["T", "D"])
@triton.jit
def _bwd_chunk(q_ptr, k_ptr, v_ptr, la_ptr, do_ptr, sent_ptr,
               dq_ptr, dk_ptr, dv_ptr, dla_ptr, dsent_ptr,
               T, NC, s_bh_t, s_bh_l, D: tl.constexpr, BC: tl.constexpr):
    """Backward of _fwd_out for one chunk: gradients of q, k, v, the decay and the entering state, through
    the decayed scores and the state read. Everything on the path INTO the entering state (each chunk's KV
    contribution and the over-chunk decays) is compiled PyTorch, and autograd differentiates it."""
    bh = tl.program_id(0); c = tl.program_id(1)
    rt = tl.arange(0, BC); rd = tl.arange(0, D)
    t0 = c * BC
    tm = (t0 + rt) < T
    la = tl.load(la_ptr + bh * s_bh_l + t0 + rt, mask=tm, other=0.0).to(tl.float32)
    L = tl.cumsum(la, 0)
    eL = tl.exp(L)
    off = bh * s_bh_t + (t0 + rt)[:, None] * D + rd[None, :]
    q = tl.load(q_ptr + off, mask=tm[:, None], other=0.0).to(tl.float32)
    k = tl.load(k_ptr + off, mask=tm[:, None], other=0.0).to(tl.float32)
    v = tl.load(v_ptr + off, mask=tm[:, None], other=0.0).to(tl.float32)
    do = tl.load(do_ptr + off, mask=tm[:, None], other=0.0).to(tl.float32)
    S = tl.load(sent_ptr + (bh * NC + c) * D * D + rd[:, None] * D + rd[None, :])
    qb = q.to(tl.bfloat16); kb = k.to(tl.bfloat16); vb = v.to(tl.bfloat16); dob = do.to(tl.bfloat16)
    dec = tl.exp(tl.minimum(L[:, None] - L[None, :], 0.0))
    dec = tl.where(rt[:, None] >= rt[None, :], dec, 0.0)
    P = tl.dot(qb, tl.trans(kb), out_dtype=tl.float32)
    A = P * dec
    dA = tl.dot(dob, tl.trans(vb), out_dtype=tl.float32)
    dP = dA * dec
    dq = tl.dot(dP.to(tl.bfloat16), kb, out_dtype=tl.float32)
    Z = tl.dot(qb, S.to(tl.bfloat16), out_dtype=tl.float32)                       # q S, pre-decay
    dq += tl.dot(dob, tl.trans(S.to(tl.bfloat16)), out_dtype=tl.float32) * eL[:, None]
    dk = tl.dot(tl.trans(dP.to(tl.bfloat16)), qb, out_dtype=tl.float32)
    dv = tl.dot(tl.trans(A.to(tl.bfloat16)), dob, out_dtype=tl.float32)
    qe = (q * eL[:, None]).to(tl.bfloat16)
    dS = tl.dot(tl.trans(qe), dob, out_dtype=tl.float32)                          # d(entering state)
    M = dA * A
    dL = tl.sum(M, 1) - tl.sum(M, 0) + tl.sum(do * Z, 1) * eL
    tot = tl.sum(dL, 0)
    dla = tot - tl.cumsum(dL, 0) + dL                                             # reverse cumulative sum
    tl.store(dq_ptr + off, dq.to(dq_ptr.dtype.element_ty), mask=tm[:, None])
    tl.store(dk_ptr + off, dk.to(dk_ptr.dtype.element_ty), mask=tm[:, None])
    tl.store(dv_ptr + off, dv.to(dv_ptr.dtype.element_ty), mask=tm[:, None])
    tl.store(dla_ptr + bh * s_bh_l + t0 + rt, dla, mask=tm)
    tl.store(dsent_ptr + (bh * NC + c) * D * D + rd[:, None] * D + rd[None, :], dS)


# ------------------------------------------------------------------ the C x C part as a custom op
# Only the two Triton kernels live inside custom ops; a custom op's body runs eagerly, so putting the chunk
# bookkeeping in there cost more in launches than the sequential kernels it replaced (2.19 ms against 1.13).
# The chunk-level structure is in assoc_scan_kernel below, which torch.compile fuses with the rest of the layer,
# and autograd differentiates the state path through the prefix matmul on its own.
@torch.library.custom_op("membrae::assoc_out", mutates_args=())
def _op_out(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, la: torch.Tensor, sent: torch.Tensor, bc: int) -> torch.Tensor:
    B, H, T, D = q.shape; NC = sent.shape[2]
    # OUTPUTS IN THE CONSUMER'S DTYPE. o, dq, dk, dv, W and dW were all written in fp32 and cast to bf16 at the
    # next boundary (autograd casts the gradients back to the inputs' dtype, W was cast for the output kernel).
    # Writing them in that dtype from the kernel is the same numbers with half the bytes and no cast pass;
    # _bwd_chunk moves 170 MB per layer at micro-batch 4 and its floor is that traffic, not its 7 dots.
    o = torch.empty(B, H, T, D, device=q.device, dtype=q.dtype)
    _fwd_out[(B * H, NC)](q.contiguous(), k.contiguous(), v.contiguous(), la.contiguous(),
                          sent.contiguous(), o, T, NC, T * D, T, D=D, BC=bc)
    return o


@_op_out.register_fake
def _(q, k, v, la, sent, bc):
    return torch.empty(q.shape, device=q.device, dtype=q.dtype)


@torch.library.custom_op("membrae::assoc_out_bwd", mutates_args=())
def _op_out_bwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, la: torch.Tensor, sent: torch.Tensor,
                do: torch.Tensor, bc: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    B, H, T, D = q.shape; NC = sent.shape[2]
    dq = torch.empty_like(q); dk = torch.empty_like(k); dv = torch.empty_like(v)
    dla = torch.empty(B, H, T, device=q.device, dtype=torch.float32)
    dsent = torch.empty(B, H, NC, D, D, device=q.device, dtype=torch.float32)
    _bwd_chunk[(B * H, NC)](q.contiguous(), k.contiguous(), v.contiguous(), la.contiguous(), do.contiguous(),
                            sent.contiguous(), dq, dk, dv, dla, dsent, T, NC, T * D, T, D=D, BC=bc)
    return dq, dk, dv, dla, dsent


@_op_out_bwd.register_fake
def _(q, k, v, la, sent, do, bc):
    B, H, T, D = q.shape; NC = sent.shape[2]
    e = lambda *sh: torch.empty(*sh, device=q.device, dtype=torch.float32)
    return torch.empty_like(q), torch.empty_like(k), torch.empty_like(v), e(B, H, T), e(B, H, NC, D, D)


def _setup(ctx, inputs, output):
    q, k, v, la, sent, bc = inputs
    ctx.save_for_backward(q, k, v, la, sent); ctx.bc = bc


def _backward(ctx, do):
    q, k, v, la, sent = ctx.saved_tensors
    dq, dk, dv, dla, dsent = _op_out_bwd(q, k, v, la, sent, do, ctx.bc)
    return dq, dk, dv, dla, dsent, None


_op_out.register_autograd(_backward, setup_context=_setup)


# ------------------------------------------------------------------ the plain rule's state over chunks
# S_{c+1} = exp(a_c) S_c + KV_c with a scalar decay per chunk. As a weighted prefix sum it was one fp32 batched
# GEMM of shape 32 x 32 @ 32 x 4096 per layer forward (0.143 ms: 370 MFLOP in a shape cuBLAS runs with 32 x 32
# FFMA tiles) and two more in backward. As a recurrence it is one elementwise pass over a D x BV tile per chunk,
# memory-bound at 16 KB per chunk; the backward is the reverse recurrence plus one inner product per chunk for
# the decay's gradient. Partials over column blocks for that gradient, summed by the traced caller.
@triton.jit
def _prefix_fwd(kv_ptr, al_ptr, s0_ptr, sent_ptr, sfin_ptr, NC, HAS_S0: tl.constexpr, D: tl.constexpr, BV: tl.constexpr):
    bh = tl.program_id(0); jb = tl.program_id(1)
    rd = tl.arange(0, D); rj = jb * BV + tl.arange(0, BV)
    off = rd[:, None] * D + rj[None, :]
    if HAS_S0:
        S = tl.load(s0_ptr + bh * D * D + off).to(tl.float32)
    else:
        S = tl.zeros([D, BV], dtype=tl.float32)
    for c in range(NC):
        base = (bh * NC + c) * D * D
        tl.store(sent_ptr + base + off, S.to(sent_ptr.dtype.element_ty))
        e = tl.exp(tl.load(al_ptr + bh * NC + c))
        S = e * S + tl.load(kv_ptr + base + off)
    tl.store(sfin_ptr + bh * D * D + off, S)


@triton.jit
def _prefix_bwd(sent_ptr, al_ptr, dsent_ptr, dsfin_ptr, dkv_ptr, dal_ptr, ds0_ptr, NC, NBH,
                D: tl.constexpr, BV: tl.constexpr):
    bh = tl.program_id(0); jb = tl.program_id(1)
    rd = tl.arange(0, D); rj = jb * BV + tl.arange(0, BV)
    off = rd[:, None] * D + rj[None, :]
    R = tl.load(dsfin_ptr + bh * D * D + off).to(tl.float32)                     # gradient on S_{NC}
    for i in range(NC):
        c = NC - 1 - i
        base = (bh * NC + c) * D * D
        e = tl.exp(tl.load(al_ptr + bh * NC + c))
        S = tl.load(sent_ptr + base + off).to(tl.float32)
        tl.store(dkv_ptr + base + off, R)                                       # S_{c+1} = e S_c + KV_c
        tl.store(dal_ptr + (jb * NBH + bh) * NC + c, e * tl.sum(tl.sum(R * S, 1), 0))
        R = tl.load(dsent_ptr + base + off) + e * R                             # gradient on S_c
    tl.store(ds0_ptr + bh * D * D + off, R)


@torch.library.custom_op("membrae::prefix_scan", mutates_args=())
def _op_prefix(KV: torch.Tensor, a_log: torch.Tensor, S0: torch.Tensor, has_s0: bool, bv: int) -> tuple[torch.Tensor, torch.Tensor]:
    B, H, NC, D, _ = KV.shape; BH = B * H
    sent = torch.empty(B, H, NC, D, D, device=KV.device, dtype=torch.bfloat16)   # see _op_delta_fwd: bf16 saved states
    sfin = torch.empty(B, H, D, D, device=KV.device, dtype=torch.float32)
    _prefix_fwd[(BH, D // bv)](KV.contiguous(), a_log.contiguous(), S0.contiguous() if has_s0 else sfin,
                               sent, sfin, NC, HAS_S0=has_s0, D=D, BV=bv)
    return sent, sfin


@_op_prefix.register_fake
def _(KV, a_log, S0, has_s0, bv):
    B, H, NC, D, _ = KV.shape
    return (torch.empty(B, H, NC, D, D, device=KV.device, dtype=torch.bfloat16),
            torch.empty(B, H, D, D, device=KV.device, dtype=torch.float32))


@torch.library.custom_op("membrae::prefix_scan_bwd", mutates_args=())
def _op_prefix_bwd(sent: torch.Tensor, a_log: torch.Tensor, dsent: torch.Tensor, dsfin: torch.Tensor,
                   bv: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, H, NC, D, _ = sent.shape; BH = B * H; NB = D // bv
    dKV = torch.empty(sent.shape, device=sent.device, dtype=torch.float32)
    dal = torch.empty(NB, B, H, NC, device=sent.device, dtype=torch.float32)
    dS0 = torch.empty(B, H, D, D, device=sent.device, dtype=torch.float32)
    _prefix_bwd[(BH, NB)](sent.contiguous(), a_log.contiguous(), dsent.contiguous(), dsfin.contiguous().float(),
                          dKV, dal, dS0, NC, BH, D=D, BV=bv)
    return dKV, dal, dS0


@_op_prefix_bwd.register_fake
def _(sent, a_log, dsent, dsfin, bv):
    B, H, NC, D, _ = sent.shape; NB = D // bv
    return (torch.empty(sent.shape, device=sent.device, dtype=torch.float32),
            torch.empty(NB, B, H, NC, device=sent.device, dtype=torch.float32),
            torch.empty(B, H, D, D, device=sent.device, dtype=torch.float32))


def _prefix_setup(ctx, inputs, output):
    KV, a_log, S0, has_s0, bv = inputs
    ctx.save_for_backward(a_log, output[0]); ctx.has_s0 = has_s0; ctx.bv = bv


def _prefix_backward(ctx, dsent, dsfin):
    a_log, sent = ctx.saved_tensors
    if dsent is None: dsent = torch.zeros(sent.shape, device=sent.device, dtype=torch.float32)
    if dsfin is None: dsfin = torch.zeros(sent.shape[0], sent.shape[1], sent.shape[3], sent.shape[4], device=sent.device)
    dKV, dal, dS0 = _op_prefix_bwd(sent, a_log, dsent, dsfin, ctx.bv)
    return dKV, dal.sum(0), (dS0 if ctx.has_s0 else None), None, None


_op_prefix.register_autograd(_prefix_backward, setup_context=_prefix_setup)


def assoc_scan_kernel(q, k, v, logalpha, S0=None, bv: int = 16):
    """Same contract as src.assoc.assoc_scan for the plain rule: (o, final state).

    The chunk-level structure: per-chunk decay statistics and each chunk's KV contribution in compiled
    PyTorch, the state entering every chunk by the recurrence kernel above, then the C x C work in the
    Triton output kernel. bf16 inputs halve the kernels' memory traffic."""
    B, H, T, D = q.shape
    NC = triton.cdiv(T, C); pad = NC * C - T
    la = logalpha.float().contiguous(); v = v.contiguous()
    lc = torch.nn.functional.pad(la, (0, pad)).view(B, H, NC, C)
    L = lc.cumsum(-1)
    a_log = lc.sum(-1)                                                          # log of each chunk's total decay
    w = torch.exp(a_log.unsqueeze(-1) - L).unsqueeze(-1)                        # exp(L_C - L_s), (B,H,NC,C,1)
    kp = torch.nn.functional.pad(k, (0, 0, 0, pad)).view(B, H, NC, C, D)
    vp = torch.nn.functional.pad(v, (0, 0, 0, pad)).view(B, H, NC, C, D)
    KV = ((kp.float() * w).to(torch.bfloat16).transpose(-1, -2) @ vp.to(torch.bfloat16)).float()   # (B,H,NC,D,D)
    has = S0 is not None
    s0 = S0.float() if has else torch.zeros(B, H, D, D, device=q.device, dtype=torch.float32)
    sent, sfin = _op_prefix(KV, a_log, s0, has, bv)
    o = _op_out(q.to(torch.bfloat16), k.to(torch.bfloat16), v.to(torch.bfloat16), la, sent, C)
    return o.to(q.dtype), sfin


# ------------------------------------------------------------------ the delta rule's loop over chunks
# W_c depends on the state entering chunk c, and that state on W of every earlier chunk, so unlike the plain
# rule this recurrence cannot become a prefix matmul. It is one program per (batch-head, block of state
# columns) walking the chunks: three small matmuls in registers per chunk and the scaled add. In compiled
# PyTorch the same loop ran at 73 ktok/s at batch 4; the matrix X_c = (I + beta strictly_lower(K K^T dec))^-1
# does not depend on the state and is computed for all chunks at once outside.
@triton.jit
def _delta_loop_fwd(k_ptr, v_ptr, la_ptr, be_ptr, x_ptr, s0_ptr, w_ptr, sent_ptr, sfin_ptr,
                    T, NC, s_bh_t, s_bh_l, HAS_S0: tl.constexpr, D: tl.constexpr, BC: tl.constexpr, BV: tl.constexpr):
    bh = tl.program_id(0); jb = tl.program_id(1)
    rt = tl.arange(0, BC); rd = tl.arange(0, D)
    rj = jb * BV + tl.arange(0, BV)
    if HAS_S0:
        S = tl.load(s0_ptr + bh * D * D + rd[:, None] * D + rj[None, :]).to(tl.float32)
    else:
        S = tl.zeros([D, BV], dtype=tl.float32)
    for c in range(NC):
        t0 = c * BC
        tm = (t0 + rt) < T
        la = tl.load(la_ptr + bh * s_bh_l + t0 + rt, mask=tm, other=0.0).to(tl.float32)
        be = tl.load(be_ptr + bh * s_bh_l + t0 + rt, mask=tm, other=0.0).to(tl.float32)
        L = tl.cumsum(la, 0); LC = tl.sum(la, 0)
        eL = tl.exp(L); w = tl.exp(LC - L)
        kk = tl.load(k_ptr + bh * s_bh_t + (t0 + rt)[:, None] * D + rd[None, :], mask=tm[:, None], other=0.0)
        vv = tl.load(v_ptr + bh * s_bh_t + (t0 + rt)[:, None] * D + rj[None, :], mask=tm[:, None], other=0.0).to(tl.float32)
        X = tl.load(x_ptr + (bh * NC + c) * BC * BC + rt[:, None] * BC + rt[None, :])
        tl.store(sent_ptr + (bh * NC + c) * D * D + rd[:, None] * D + rj[None, :], S.to(sent_ptr.dtype.element_ty))
        # rhs = beta (v - eL k S);  W = X rhs;  S = a S + (k w)^T W
        ks = tl.dot(kk.to(tl.bfloat16), S.to(tl.bfloat16), out_dtype=tl.float32)
        rhs = be[:, None] * (vv - eL[:, None] * ks)
        W = tl.dot(X.to(tl.bfloat16), rhs.to(tl.bfloat16), out_dtype=tl.float32)
        tl.store(w_ptr + bh * s_bh_t + (t0 + rt)[:, None] * D + rj[None, :], W.to(w_ptr.dtype.element_ty), mask=tm[:, None])
        kw = (kk.to(tl.float32) * w[:, None]).to(tl.bfloat16)
        S = tl.exp(LC) * S + tl.dot(tl.trans(kw), W.to(tl.bfloat16), out_dtype=tl.float32)
    tl.store(sfin_ptr + bh * D * D + rd[:, None] * D + rj[None, :], S)


@torch.library.custom_op("membrae::delta_loop_fwd", mutates_args=())
def _op_delta_fwd(k: torch.Tensor, v: torch.Tensor, la: torch.Tensor, be: torch.Tensor, X: torch.Tensor,
                  S0: torch.Tensor, has_s0: bool, bc: int, bv: int, bvb: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, H, T, D = k.shape; NC = X.shape[2]; BH = B * H
    W = torch.empty(B, H, T, D, device=k.device, dtype=torch.bfloat16)     # the output kernel's value slot
    # THE SAVED STATES IN bf16. The forward's accumulator stays fp32 in registers; this is the copy of the
    # state entering each chunk that the backward reads, and every consumer feeds it to a bf16 dot except the
    # decay's per-chunk inner product <S, dS>. fp32 it was 22 MiB per layer at B4 T2048, the largest single
    # item of the carrier's 1 GiB over attention (the memory audit of 2026-09-17).
    sent = torch.empty(B, H, NC, D, D, device=k.device, dtype=torch.bfloat16)
    sfin = torch.empty(B, H, D, D, device=k.device, dtype=torch.float32)
    _delta_loop_fwd[(BH, D // bv)](k.contiguous(), v.contiguous(), la.contiguous(), be.contiguous(), X.contiguous(),
                                   S0.contiguous() if has_s0 else sfin, W, sent, sfin,
                                   T, NC, T * D, T, HAS_S0=has_s0, D=D, BC=bc, BV=bv,
                                   num_warps=_LOOP["nw_fwd"], num_stages=_LOOP["ns_fwd"])
    return W, sent, sfin


@_op_delta_fwd.register_fake
def _(k, v, la, be, X, S0, has_s0, bc, bv, bvb):
    B, H, T, D = k.shape; NC = X.shape[2]
    e = lambda *sh: torch.empty(*sh, device=k.device, dtype=torch.float32)
    return (torch.empty(B, H, T, D, device=k.device, dtype=torch.bfloat16),
            torch.empty(B, H, NC, D, D, device=k.device, dtype=torch.bfloat16), e(B, H, D, D))


@triton.jit
def _delta_loop_bwd(k_ptr, v_ptr, la_ptr, be_ptr, x_ptr, w_ptr, sent_ptr, gs_ptr, dw_ptr, dsfin_ptr,
                    dk_ptr, dv_ptr, dla_ptr, dbe_ptr, dx_ptr, ds0_ptr,
                    T, NC, NBH, s_bh_t, s_bh_l, D: tl.constexpr, BC: tl.constexpr, BV: tl.constexpr):
    """Reverse loop over chunks for the delta rule. Per chunk, with S the entering state, D_S the gradient on
    the state LEAVING the chunk, G the gradient on the entering state from the output kernel's read of it, and
    dW_out the gradient on the residuals from the output kernel:
        dW    = dW_out + Kw D_S                (S' = a S + Kw^T W)
        drhs  = X^T dW,  dX += dW rhs^T         (W = X rhs)
        dbeta = sum drhs (V - eL K S),  dV = beta drhs,  dks = -beta eL drhs,  d(eL) = -sum beta drhs ks
        dK   += dks S^T + (W D_S^T) w,   dS = G + a D_S + K^T dks
    and the decay terms fold into dla as in the plain kernel. dK, dbeta, dla and dX are shared across the
    column blocks of one batch-head and accumulate atomically; dV and dS are per block."""
    bh = tl.program_id(0); jb = tl.program_id(1)
    rt = tl.arange(0, BC); rd = tl.arange(0, D)
    rj = jb * BV + tl.arange(0, BV)
    DS = tl.load(dsfin_ptr + bh * D * D + rd[:, None] * D + rj[None, :]).to(tl.float32)   # gradient on the final state
    for i in range(NC):
        c = NC - 1 - i
        t0 = c * BC
        tm = (t0 + rt) < T
        la = tl.load(la_ptr + bh * s_bh_l + t0 + rt, mask=tm, other=0.0).to(tl.float32)
        be = tl.load(be_ptr + bh * s_bh_l + t0 + rt, mask=tm, other=0.0).to(tl.float32)
        L = tl.cumsum(la, 0); LC = tl.sum(la, 0)
        eL = tl.exp(L); w = tl.exp(LC - L); a = tl.exp(LC)
        kk = tl.load(k_ptr + bh * s_bh_t + (t0 + rt)[:, None] * D + rd[None, :], mask=tm[:, None], other=0.0).to(tl.float32)
        vv = tl.load(v_ptr + bh * s_bh_t + (t0 + rt)[:, None] * D + rj[None, :], mask=tm[:, None], other=0.0).to(tl.float32)
        W = tl.load(w_ptr + bh * s_bh_t + (t0 + rt)[:, None] * D + rj[None, :], mask=tm[:, None], other=0.0)
        S = tl.load(sent_ptr + (bh * NC + c) * D * D + rd[:, None] * D + rj[None, :]).to(tl.float32)
        G = tl.load(gs_ptr + (bh * NC + c) * D * D + rd[:, None] * D + rj[None, :])
        dWo = tl.load(dw_ptr + bh * s_bh_t + (t0 + rt)[:, None] * D + rj[None, :], mask=tm[:, None], other=0.0).to(tl.float32)
        X = tl.load(x_ptr + (bh * NC + c) * BC * BC + rt[:, None] * BC + rt[None, :])
        kb = kk.to(tl.bfloat16)
        kw = kk * w[:, None]
        # dW = dW_out + Kw D_S
        dW = dWo + tl.dot(kw.to(tl.bfloat16), DS.to(tl.bfloat16), out_dtype=tl.float32)
        # through S' = a S + Kw^T W: dKw = W D_S^T, da = <S, D_S>
        dKw = tl.dot(W.to(tl.bfloat16), tl.trans(DS.to(tl.bfloat16)), out_dtype=tl.float32)
        da = tl.sum(tl.sum(S * DS, 1), 0)
        # through W = X rhs
        ks = tl.dot(kb, S.to(tl.bfloat16), out_dtype=tl.float32)
        rhs = be[:, None] * (vv - eL[:, None] * ks)
        drhs = tl.dot(tl.trans(X.to(tl.bfloat16)), dW.to(tl.bfloat16), out_dtype=tl.float32)
        dX = tl.dot(dW.to(tl.bfloat16), tl.trans(rhs.to(tl.bfloat16)), out_dtype=tl.float32)
        # through rhs = beta (v - eL ks)
        inner = vv - eL[:, None] * ks
        dbe = tl.sum(drhs * inner, 1)
        dV = be[:, None] * drhs
        dks = -be[:, None] * eL[:, None] * drhs
        deL = -tl.sum(be[:, None] * drhs * ks, 1)
        # through ks = K S
        dK = tl.dot(dks.to(tl.bfloat16), tl.trans(S.to(tl.bfloat16)), out_dtype=tl.float32) + dKw * w[:, None]
        DSn = G + a * DS + tl.dot(tl.trans(kb), dks.to(tl.bfloat16), out_dtype=tl.float32)
        # decay: eL, w and a terms
        dw = tl.sum(dKw * kk, 1) * w
        dL = deL * eL - dw
        dLC = tl.sum(dw, 0) + da * a
        tot = tl.sum(dL, 0)
        dla = tot - tl.cumsum(dL, 0) + dL + dLC
        # shared across column blocks: each program writes its own partial (slot jb), summed outside.
        # Atomic adds here cost 7.85 ms per micro-step, eight times the forward loop, from the two programs of
        # one batch-head contending on the same 64 x 64 tiles.
        NB = D // BV
        tl.store(dk_ptr + (jb * NBH + bh) * s_bh_t + (t0 + rt)[:, None] * D + rd[None, :], dK.to(dk_ptr.dtype.element_ty), mask=tm[:, None])
        tl.store(dla_ptr + (jb * NBH + bh) * s_bh_l + t0 + rt, dla, mask=tm)
        tl.store(dbe_ptr + (jb * NBH + bh) * s_bh_l + t0 + rt, dbe, mask=tm)
        tl.store(dx_ptr + ((jb * NBH + bh) * NC + c) * BC * BC + rt[:, None] * BC + rt[None, :], dX.to(dx_ptr.dtype.element_ty))
        tl.store(dv_ptr + bh * s_bh_t + (t0 + rt)[:, None] * D + rj[None, :], dV.to(dv_ptr.dtype.element_ty), mask=tm[:, None])
        DS = DSn
    tl.store(ds0_ptr + bh * D * D + rd[:, None] * D + rj[None, :], DS)


@torch.library.custom_op("membrae::delta_loop_bwd", mutates_args=())
def _op_delta_bwd(k: torch.Tensor, v: torch.Tensor, la: torch.Tensor, be: torch.Tensor, X: torch.Tensor,
                  W: torch.Tensor, sent: torch.Tensor, gsent: torch.Tensor, dW: torch.Tensor,
                  dsfin: torch.Tensor, bc: int, bv: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    B, H, T, D = k.shape; NC = X.shape[2]; BH = B * H
    NB = D // bv
    # dK and dX partials in bf16: their sums are cast to bf16 (k's dtype) and fed to a bf16 dot respectively
    dk = torch.empty(NB, B, H, T, D, device=k.device, dtype=torch.bfloat16)
    dv = torch.empty_like(v)
    dla = torch.empty(NB, B, H, T, device=k.device, dtype=torch.float32)
    dbe = torch.empty(NB, B, H, T, device=k.device, dtype=torch.float32)
    dX = torch.empty(NB, B, H, NC, bc, bc, device=k.device, dtype=torch.bfloat16)
    dS0 = torch.empty(B, H, D, D, device=k.device, dtype=torch.float32)
    _delta_loop_bwd[(BH, NB)](k.contiguous(), v.contiguous(), la.contiguous(), be.contiguous(), X.contiguous(),
                              W.contiguous(), sent.contiguous(), gsent.contiguous(), dW.contiguous(),
                              dsfin.contiguous().float(), dk, dv, dla, dbe, dX, dS0,
                              T, NC, BH, T * D, T, D=D, BC=bc, BV=bv,
                              num_warps=_LOOP["nw_bwd"], num_stages=_LOOP["ns_bwd"])
    # the partials are summed by the caller, which torch.compile traces; inside this op they would be eager
    # reductions, 48 launches per micro-step that the profile showed at the top of the table
    return dk, dv, dla, dbe, dX, dS0


@_op_delta_bwd.register_fake
def _(k, v, la, be, X, W, sent, gsent, dW, dsfin, bc, bv):
    B, H, T, D = k.shape; NC = X.shape[2]; NB = D // bv
    e = lambda *sh: torch.empty(*sh, device=k.device, dtype=torch.float32)
    b = lambda *sh: torch.empty(*sh, device=k.device, dtype=torch.bfloat16)
    return b(NB, B, H, T, D), torch.empty_like(v), e(NB, B, H, T), e(NB, B, H, T), b(NB, B, H, NC, bc, bc), e(B, H, D, D)


def _delta_setup(ctx, inputs, output):
    k, v, la, be, X, S0, has_s0, bc, bv, bvb = inputs
    W, sent, sfin = output
    ctx.save_for_backward(k, v, la, be, X, W, sent)
    ctx.has_s0 = has_s0; ctx.bc = bc; ctx.bv = bvb


def _delta_backward(ctx, dW, gsent, dsfin):
    k, v, la, be, X, W, sent = ctx.saved_tensors
    if gsent is None: gsent = torch.zeros(sent.shape, device=sent.device, dtype=torch.float32)
    if dsfin is None: dsfin = torch.zeros(sent.shape[0], sent.shape[1], sent.shape[3], sent.shape[4], device=k.device)
    dk, dv, dla, dbe, dX, dS0 = _op_delta_bwd(k, v, la, be, X, W, sent, gsent, dW, dsfin, ctx.bc, ctx.bv)
    return (dk.float().sum(0).to(k.dtype), dv, dla.sum(0), dbe.sum(0), dX.float().sum(0).to(X.dtype),
            (dS0 if ctx.has_s0 else None), None, None, None, None)


_op_delta_fwd.register_autograd(_delta_backward, setup_context=_delta_setup)


def delta_scan_kernel(q, k, v, logalpha, beta, S0=None, chunk: int = 64, bv: int = 32, bv_bwd: int = 16):
    """The delta rule end to end: X from the torch-side inverse (compiled, autograd), the loop over chunks in
    the Triton kernel, the C x C output through _op_out with the residuals W in the value slot.

    bv / bv_bwd: state columns per program in the forward / backward loop. Swept at B4 H11 T2048 D64: forward
    32 (0.046 ms; 16 is 0.057, 64 is 0.085), backward 16 (0.160 ms; 32 is 0.188, 64 is 0.304) once the
    backward's partial buffers were bf16. 4 warps and 3 pipeline stages for both; 8 warps, 2 warps and chunk
    128 are all slower."""
    from .assoc import _unit_lower_inverse
    B, H, T, D = q.shape; Cc = chunk
    NC = triton.cdiv(T, Cc); pad = NC * Cc - T
    # contiguous HERE, inside the compiled region, where the copy fuses into the kernel that produces the
    # tensor; the ops' own .contiguous() then finds nothing to do. Left to the ops it was 156 eager copies per
    # micro-step (the decay and beta arrive as transposes of (B,T,H), v as a transpose of (B,T,H,D)).
    la = logalpha.float().contiguous(); be = beta.float().contiguous(); v = v.contiguous()
    lc = torch.nn.functional.pad(la, (0, pad)).view(B, H, NC, Cc)
    L = lc.cumsum(-1)
    mask = torch.ones(Cc, Cc, device=q.device, dtype=torch.bool).tril()
    dec = torch.exp((L.unsqueeze(-1) - L.unsqueeze(-2)).clamp(max=0.0)) * mask
    if Cc == C:
        X = _op_inv(k.to(torch.bfloat16), la, be, NC)                             # Triton, three kernels per chunk
    else:
        kp = torch.nn.functional.pad(k, (0, 0, 0, pad)).view(B, H, NC, Cc, D).float()
        bcc = torch.nn.functional.pad(be, (0, pad)).view(B, H, NC, Cc)
        X = _unit_lower_inverse(bcc.unsqueeze(-1) * ((kp @ kp.transpose(-1, -2)) * dec).tril(-1))
    has = S0 is not None
    s0 = S0.float() if has else torch.zeros(B, H, D, D, device=q.device, dtype=torch.float32)
    W, sent, sfin = _op_delta_fwd(k.to(torch.bfloat16), v, la, be, X, s0, has, Cc, bv, bv_bwd)
    o = _op_out(q.to(torch.bfloat16), k.to(torch.bfloat16), W, la, sent, Cc)
    return o.to(q.dtype), sfin


# ------------------------------------------------------------------ the delta rule's inverse, in Triton
# X_c = (I + N_c)^-1 with N_c = beta strictly_lower(K K^T dec), per chunk. Measured in torch it was the delta
# rule's whole cost (inverse 0.72 ms of a 1.8 ms forward; its autograd backward twice that; the loop kernel it
# feeds 0.08 ms). Three kernels, all parallel over chunks: build N; invert it by blocked substitution with
# 16 x 16 diagonal blocks summed by doubling (the whole-tile series overflows at C=64: N^32 can reach 1e29 in
# fp32 before N^64 vanishes); and the backward, dN = -X^T dX X^T chained into dK, dbeta and the decay.
@triton.jit
def _inv_build(k_ptr, la_ptr, be_ptr, n_ptr, T, NC, s_bh_t, s_bh_l, D: tl.constexpr, BC: tl.constexpr):
    bh = tl.program_id(0); c = tl.program_id(1)
    rt = tl.arange(0, BC); rd = tl.arange(0, D)
    t0 = c * BC; tm = (t0 + rt) < T
    la = tl.load(la_ptr + bh * s_bh_l + t0 + rt, mask=tm, other=0.0).to(tl.float32)
    be = tl.load(be_ptr + bh * s_bh_l + t0 + rt, mask=tm, other=0.0).to(tl.float32)
    L = tl.cumsum(la, 0)
    k = tl.load(k_ptr + bh * s_bh_t + (t0 + rt)[:, None] * D + rd[None, :], mask=tm[:, None], other=0.0)
    P = tl.dot(k.to(tl.bfloat16), tl.trans(k.to(tl.bfloat16)), out_dtype=tl.float32)
    dec = tl.exp(tl.minimum(L[:, None] - L[None, :], 0.0))
    N = tl.where(rt[:, None] > rt[None, :], be[:, None] * P * dec, 0.0)          # strictly lower
    tl.store(n_ptr + (bh * NC + c) * BC * BC + rt[:, None] * BC + rt[None, :], N)


@triton.jit
def _inv16(n_ptr, base, i, BC: tl.constexpr):
    """(I + N_ii)^-1 for the i-th 16 x 16 diagonal block, by doubling (exact after 4 steps: nilpotent)."""
    r = tl.arange(0, 16)
    Nd = tl.load(n_ptr + base + (i * 16 + r)[:, None] * BC + (i * 16 + r)[None, :])
    R = -Nd
    X = tl.where(r[:, None] == r[None, :], 1.0, 0.0) + R
    R = tl.dot(R.to(tl.bfloat16), R.to(tl.bfloat16), out_dtype=tl.float32)      # (-N)^2
    X = X + tl.dot(R.to(tl.bfloat16), X.to(tl.bfloat16), out_dtype=tl.float32)   # terms 0..3
    R = tl.dot(R.to(tl.bfloat16), R.to(tl.bfloat16), out_dtype=tl.float32)      # (-N)^4
    X = X + tl.dot(R.to(tl.bfloat16), X.to(tl.bfloat16), out_dtype=tl.float32)   # terms 0..7
    R = tl.dot(R.to(tl.bfloat16), R.to(tl.bfloat16), out_dtype=tl.float32)      # (-N)^8
    X = X + tl.dot(R.to(tl.bfloat16), X.to(tl.bfloat16), out_dtype=tl.float32)   # terms 0..15: exact
    return X


@triton.jit
def _blk(n_ptr, base, i, j, BC: tl.constexpr):
    r = tl.arange(0, 16)
    return tl.load(n_ptr + base + (i * 16 + r)[:, None] * BC + (j * 16 + r)[None, :])


@triton.jit
def _inv_solve(n_ptr, x_ptr, NC, BC: tl.constexpr):
    """X = (I + N)^-1 for one chunk, blocked forward substitution over four 16-row blocks."""
    bh = tl.program_id(0); c = tl.program_id(1)
    base = (bh * NC + c) * BC * BC
    r = tl.arange(0, 16); cc = tl.arange(0, BC)
    E0 = tl.where(cc[None, :] == r[:, None], 1.0, 0.0)                             # rows 0..15 of I
    E1 = tl.where(cc[None, :] == r[:, None] + 16, 1.0, 0.0)
    E2 = tl.where(cc[None, :] == r[:, None] + 32, 1.0, 0.0)
    E3 = tl.where(cc[None, :] == r[:, None] + 48, 1.0, 0.0)
    X0 = tl.dot(_inv16(n_ptr, base, 0, BC).to(tl.bfloat16), E0.to(tl.bfloat16), out_dtype=tl.float32)
    rhs = E1 - tl.dot(_blk(n_ptr, base, 1, 0, BC).to(tl.bfloat16), X0.to(tl.bfloat16), out_dtype=tl.float32)
    X1 = tl.dot(_inv16(n_ptr, base, 1, BC).to(tl.bfloat16), rhs.to(tl.bfloat16), out_dtype=tl.float32)
    rhs = E2 - tl.dot(_blk(n_ptr, base, 2, 0, BC).to(tl.bfloat16), X0.to(tl.bfloat16), out_dtype=tl.float32) \
             - tl.dot(_blk(n_ptr, base, 2, 1, BC).to(tl.bfloat16), X1.to(tl.bfloat16), out_dtype=tl.float32)
    X2 = tl.dot(_inv16(n_ptr, base, 2, BC).to(tl.bfloat16), rhs.to(tl.bfloat16), out_dtype=tl.float32)
    rhs = E3 - tl.dot(_blk(n_ptr, base, 3, 0, BC).to(tl.bfloat16), X0.to(tl.bfloat16), out_dtype=tl.float32) \
             - tl.dot(_blk(n_ptr, base, 3, 1, BC).to(tl.bfloat16), X1.to(tl.bfloat16), out_dtype=tl.float32) \
             - tl.dot(_blk(n_ptr, base, 3, 2, BC).to(tl.bfloat16), X2.to(tl.bfloat16), out_dtype=tl.float32)
    X3 = tl.dot(_inv16(n_ptr, base, 3, BC).to(tl.bfloat16), rhs.to(tl.bfloat16), out_dtype=tl.float32)
    tl.store(x_ptr + base + (r)[:, None] * BC + cc[None, :], X0.to(x_ptr.dtype.element_ty))
    tl.store(x_ptr + base + (r + 16)[:, None] * BC + cc[None, :], X1.to(x_ptr.dtype.element_ty))
    tl.store(x_ptr + base + (r + 32)[:, None] * BC + cc[None, :], X2.to(x_ptr.dtype.element_ty))
    tl.store(x_ptr + base + (r + 48)[:, None] * BC + cc[None, :], X3.to(x_ptr.dtype.element_ty))


@triton.jit
def _inv_bwd(k_ptr, la_ptr, be_ptr, x_ptr, dx_ptr, dk_ptr, dla_ptr, dbe_ptr,
             T, NC, s_bh_t, s_bh_l, D: tl.constexpr, BC: tl.constexpr):
    """dN = -X^T dX X^T, then through N = beta (P dec) strictly lower with P = K K^T."""
    bh = tl.program_id(0); c = tl.program_id(1)
    rt = tl.arange(0, BC); rd = tl.arange(0, D)
    t0 = c * BC; tm = (t0 + rt) < T
    base = (bh * NC + c) * BC * BC
    la = tl.load(la_ptr + bh * s_bh_l + t0 + rt, mask=tm, other=0.0).to(tl.float32)
    be = tl.load(be_ptr + bh * s_bh_l + t0 + rt, mask=tm, other=0.0).to(tl.float32)
    L = tl.cumsum(la, 0)
    k = tl.load(k_ptr + bh * s_bh_t + (t0 + rt)[:, None] * D + rd[None, :], mask=tm[:, None], other=0.0).to(tl.float32)
    kb = k.to(tl.bfloat16)
    X = tl.load(x_ptr + base + rt[:, None] * BC + rt[None, :])
    dX = tl.load(dx_ptr + base + rt[:, None] * BC + rt[None, :])
    Xb = X.to(tl.bfloat16)
    dN = -tl.dot(tl.trans(Xb), tl.dot(dX.to(tl.bfloat16), tl.trans(Xb), out_dtype=tl.float32).to(tl.bfloat16), out_dtype=tl.float32)
    dN = tl.where(rt[:, None] > rt[None, :], dN, 0.0)
    P = tl.dot(kb, tl.trans(kb), out_dtype=tl.float32)
    dec = tl.exp(tl.minimum(L[:, None] - L[None, :], 0.0))
    dbe = tl.sum(dN * P * dec, 1)
    M2 = be[:, None] * dN
    dP = M2 * dec
    dK = tl.dot(dP.to(tl.bfloat16), kb, out_dtype=tl.float32) + tl.dot(tl.trans(dP.to(tl.bfloat16)), kb, out_dtype=tl.float32)
    Md = M2 * P * dec                                                             # ddec * dec
    dL = tl.sum(Md, 1) - tl.sum(Md, 0)
    tot = tl.sum(dL, 0)
    dla = tot - tl.cumsum(dL, 0) + dL
    tl.store(dk_ptr + bh * s_bh_t + (t0 + rt)[:, None] * D + rd[None, :], dK, mask=tm[:, None])
    tl.store(dla_ptr + bh * s_bh_l + t0 + rt, dla, mask=tm)
    tl.store(dbe_ptr + bh * s_bh_l + t0 + rt, dbe, mask=tm)


@torch.library.custom_op("membrae::delta_inv", mutates_args=())
def _op_inv(k: torch.Tensor, la: torch.Tensor, be: torch.Tensor, nc: int) -> torch.Tensor:
    B, H, T, D = k.shape; BH = B * H
    N = torch.empty(B, H, nc, C, C, device=k.device, dtype=torch.float32)
    # X is consumed by three kernels and every one of them feeds it to a bf16 dot: stored in bf16 it is the
    # same operand with half the traffic and half the footprint (23 -> 11.5 MB per layer at micro-batch 4)
    X = torch.empty(B, H, nc, C, C, device=k.device, dtype=torch.bfloat16)
    _inv_build[(BH, nc)](k.contiguous(), la.contiguous(), be.contiguous(), N, T, nc, T * D, T, D=D, BC=C)
    _inv_solve[(BH, nc)](N, X, nc, BC=C)
    return X


@_op_inv.register_fake
def _(k, la, be, nc):
    B, H, T, D = k.shape
    return torch.empty(B, H, nc, C, C, device=k.device, dtype=torch.bfloat16)


@torch.library.custom_op("membrae::delta_inv_bwd", mutates_args=())
def _op_inv_bwd(k: torch.Tensor, la: torch.Tensor, be: torch.Tensor, X: torch.Tensor, dX: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, H, T, D = k.shape; nc = X.shape[2]; BH = B * H
    dk = torch.empty(B, H, T, D, device=k.device, dtype=torch.float32)
    dla = torch.empty(B, H, T, device=k.device, dtype=torch.float32); dbe = torch.empty_like(dla)
    _inv_bwd[(BH, nc)](k.contiguous(), la.contiguous(), be.contiguous(), X.contiguous(), dX.contiguous(),
                       dk, dla, dbe, T, nc, T * D, T, D=D, BC=C)
    return dk, dla, dbe


@_op_inv_bwd.register_fake
def _(k, la, be, X, dX):
    B, H, T, D = k.shape
    return (torch.empty(B, H, T, D, device=k.device, dtype=torch.float32),
            torch.empty(B, H, T, device=k.device, dtype=torch.float32), torch.empty(B, H, T, device=k.device, dtype=torch.float32))


def _inv_setup(ctx, inputs, output):
    k, la, be, nc = inputs
    ctx.save_for_backward(k, la, be, output)


def _inv_backward(ctx, dX):
    k, la, be, X = ctx.saved_tensors
    dk, dla, dbe = _op_inv_bwd(k, la, be, X, dX)
    return dk.to(k.dtype), dla, dbe, None


_op_inv.register_autograd(_inv_backward, setup_context=_inv_setup)
