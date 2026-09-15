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
    tl.store(o_ptr + off, o, mask=tm[:, None])


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
    tl.store(dq_ptr + off, dq, mask=tm[:, None])
    tl.store(dk_ptr + off, dk, mask=tm[:, None])
    tl.store(dv_ptr + off, dv, mask=tm[:, None])
    tl.store(dla_ptr + bh * s_bh_l + t0 + rt, dla, mask=tm)
    tl.store(dsent_ptr + (bh * NC + c) * D * D + rd[:, None] * D + rd[None, :], dS)


# ------------------------------------------------------------------ the C x C part as a custom op
# Only the two Triton kernels live inside custom ops; a custom op's body runs eagerly, so putting the chunk
# bookkeeping in there cost more in launches than the sequential kernels it replaced (2.19 ms against 1.13).
# The chunk-level structure is in assoc_scan_kernel below, which torch.compile fuses with the rest of the layer,
# and autograd differentiates the state path through the prefix matmul on its own.
@torch.library.custom_op("membrae::assoc_out", mutates_args=())
def _op_out(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, la: torch.Tensor, sent: torch.Tensor) -> torch.Tensor:
    B, H, T, D = q.shape; NC = sent.shape[2]
    o = torch.empty(B, H, T, D, device=q.device, dtype=torch.float32)
    _fwd_out[(B * H, NC)](q.contiguous(), k.contiguous(), v.contiguous(), la.contiguous(),
                          sent.contiguous(), o, T, NC, T * D, T, D=D, BC=C)
    return o


@_op_out.register_fake
def _(q, k, v, la, sent):
    return torch.empty(q.shape, device=q.device, dtype=torch.float32)


@torch.library.custom_op("membrae::assoc_out_bwd", mutates_args=())
def _op_out_bwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, la: torch.Tensor, sent: torch.Tensor,
                do: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    B, H, T, D = q.shape; NC = sent.shape[2]
    dq = torch.empty(B, H, T, D, device=q.device, dtype=torch.float32); dk = torch.empty_like(dq); dv = torch.empty_like(dq)
    dla = torch.empty(B, H, T, device=q.device, dtype=torch.float32)
    dsent = torch.empty(B, H, NC, D, D, device=q.device, dtype=torch.float32)
    _bwd_chunk[(B * H, NC)](q.contiguous(), k.contiguous(), v.contiguous(), la.contiguous(), do.contiguous().float(),
                            sent.contiguous(), dq, dk, dv, dla, dsent, T, NC, T * D, T, D=D, BC=C)
    return dq, dk, dv, dla, dsent


@_op_out_bwd.register_fake
def _(q, k, v, la, sent, do):
    B, H, T, D = q.shape; NC = sent.shape[2]
    e = lambda *sh: torch.empty(*sh, device=q.device, dtype=torch.float32)
    return e(B, H, T, D), e(B, H, T, D), e(B, H, T, D), e(B, H, T), e(B, H, NC, D, D)


def _setup(ctx, inputs, output):
    q, k, v, la, sent = inputs
    ctx.save_for_backward(q, k, v, la, sent)


def _backward(ctx, do):
    q, k, v, la, sent = ctx.saved_tensors
    dq, dk, dv, dla, dsent = _op_out_bwd(q, k, v, la, sent, do)
    return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), dla, dsent


_op_out.register_autograd(_backward, setup_context=_setup)


def assoc_scan_kernel(q, k, v, logalpha, S0=None):
    """Same contract as src.assoc.assoc_scan for the plain rule: (o, final state).

    The chunk-level structure, in compiled PyTorch: per-chunk decay statistics, each chunk's KV contribution,
    and the state entering every chunk as a weighted prefix sum over chunks (S_ent[c] = sum_{j<c}
    exp(A_{c-1} - A_j) KV_j, all weights <= 1: with scalar per-chunk decays the recurrence is a matmul, not a
    loop). Then the C x C work in the Triton kernel. bf16 inputs halve the kernel's memory traffic."""
    B, H, T, D = q.shape
    NC = triton.cdiv(T, C); pad = NC * C - T
    la = logalpha.float()
    lc = torch.nn.functional.pad(la, (0, pad)).view(B, H, NC, C)
    L = lc.cumsum(-1)
    a_log = lc.sum(-1)                                                          # log of each chunk's total decay
    A = a_log.cumsum(-1)
    Am1 = torch.nn.functional.pad(A, (1, 0))[..., :-1]                          # A_{c-1}, A_{-1} = 0
    w = torch.exp(a_log.unsqueeze(-1) - L).unsqueeze(-1)                        # exp(L_C - L_s), (B,H,NC,C,1)
    kp = torch.nn.functional.pad(k, (0, 0, 0, pad)).view(B, H, NC, C, D)
    vp = torch.nn.functional.pad(v, (0, 0, 0, pad)).view(B, H, NC, C, D)
    KV = ((kp.float() * w).to(torch.bfloat16).transpose(-1, -2) @ vp.to(torch.bfloat16)).float()   # (B,H,NC,D,D)
    Wt = torch.exp((Am1.unsqueeze(-1) - A.unsqueeze(-2)).clamp(max=0.0))
    Wt = Wt * torch.ones(NC, NC, device=q.device, dtype=Wt.dtype).tril(-1)      # [c, j], j < c
    sent = (Wt @ KV.reshape(B, H, NC, D * D)).view(B, H, NC, D, D)
    if S0 is not None:
        sent = sent + torch.exp(Am1)[..., None, None] * S0.float().unsqueeze(2)
    sfin = torch.exp(a_log[..., -1])[..., None, None] * sent[:, :, -1] + KV[:, :, -1]
    o = _op_out(q.to(torch.bfloat16), k.to(torch.bfloat16), v.to(torch.bfloat16), la, sent)
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
        tl.store(sent_ptr + (bh * NC + c) * D * D + rd[:, None] * D + rj[None, :], S)
        # rhs = beta (v - eL k S);  W = X rhs;  S = a S + (k w)^T W
        ks = tl.dot(kk.to(tl.bfloat16), S.to(tl.bfloat16), out_dtype=tl.float32)
        rhs = be[:, None] * (vv - eL[:, None] * ks)
        W = tl.dot(X.to(tl.bfloat16), rhs.to(tl.bfloat16), out_dtype=tl.float32)
        tl.store(w_ptr + bh * s_bh_t + (t0 + rt)[:, None] * D + rj[None, :], W, mask=tm[:, None])
        kw = (kk.to(tl.float32) * w[:, None]).to(tl.bfloat16)
        S = tl.exp(LC) * S + tl.dot(tl.trans(kw), W.to(tl.bfloat16), out_dtype=tl.float32)
    tl.store(sfin_ptr + bh * D * D + rd[:, None] * D + rj[None, :], S)


@torch.library.custom_op("membrae::delta_loop_fwd", mutates_args=())
def _op_delta_fwd(k: torch.Tensor, v: torch.Tensor, la: torch.Tensor, be: torch.Tensor, X: torch.Tensor,
                  S0: torch.Tensor, has_s0: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, H, T, D = k.shape; NC = X.shape[2]; BH = B * H
    W = torch.empty(B, H, T, D, device=k.device, dtype=torch.float32)
    sent = torch.empty(B, H, NC, D, D, device=k.device, dtype=torch.float32)
    sfin = torch.empty(B, H, D, D, device=k.device, dtype=torch.float32)
    BV = 32
    _delta_loop_fwd[(BH, D // BV)](k.contiguous(), v.contiguous(), la.contiguous(), be.contiguous(), X.contiguous(),
                                   S0.contiguous() if has_s0 else sfin, W, sent, sfin,
                                   T, NC, T * D, T, HAS_S0=has_s0, D=D, BC=C, BV=BV)
    return W, sent, sfin


@_op_delta_fwd.register_fake
def _(k, v, la, be, X, S0, has_s0):
    B, H, T, D = k.shape; NC = X.shape[2]
    e = lambda *sh: torch.empty(*sh, device=k.device, dtype=torch.float32)
    return e(B, H, T, D), e(B, H, NC, D, D), e(B, H, D, D)


@triton.jit
def _delta_loop_bwd(k_ptr, v_ptr, la_ptr, be_ptr, x_ptr, w_ptr, sent_ptr, gs_ptr, dw_ptr, dsfin_ptr,
                    dk_ptr, dv_ptr, dla_ptr, dbe_ptr, dx_ptr, ds0_ptr,
                    T, NC, s_bh_t, s_bh_l, D: tl.constexpr, BC: tl.constexpr, BV: tl.constexpr):
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
        S = tl.load(sent_ptr + (bh * NC + c) * D * D + rd[:, None] * D + rj[None, :])
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
        # shared across column blocks: accumulate
        tl.atomic_add(dk_ptr + bh * s_bh_t + (t0 + rt)[:, None] * D + rd[None, :], dK, mask=tm[:, None])
        tl.atomic_add(dla_ptr + bh * s_bh_l + t0 + rt, dla, mask=tm)
        tl.atomic_add(dbe_ptr + bh * s_bh_l + t0 + rt, dbe, mask=tm)
        tl.atomic_add(dx_ptr + (bh * NC + c) * BC * BC + rt[:, None] * BC + rt[None, :], dX)
        tl.store(dv_ptr + bh * s_bh_t + (t0 + rt)[:, None] * D + rj[None, :], dV, mask=tm[:, None])
        DS = DSn
    tl.store(ds0_ptr + bh * D * D + rd[:, None] * D + rj[None, :], DS)


@torch.library.custom_op("membrae::delta_loop_bwd", mutates_args=())
def _op_delta_bwd(k: torch.Tensor, v: torch.Tensor, la: torch.Tensor, be: torch.Tensor, X: torch.Tensor,
                  W: torch.Tensor, sent: torch.Tensor, gsent: torch.Tensor, dW: torch.Tensor,
                  dsfin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    B, H, T, D = k.shape; NC = X.shape[2]; BH = B * H
    dk = torch.zeros(B, H, T, D, device=k.device, dtype=torch.float32)
    dv = torch.empty(B, H, T, D, device=k.device, dtype=torch.float32)
    dla = torch.zeros(B, H, T, device=k.device, dtype=torch.float32)
    dbe = torch.zeros(B, H, T, device=k.device, dtype=torch.float32)
    dX = torch.zeros(B, H, NC, C, C, device=k.device, dtype=torch.float32)
    dS0 = torch.empty(B, H, D, D, device=k.device, dtype=torch.float32)
    BV = 32
    _delta_loop_bwd[(BH, D // BV)](k.contiguous(), v.contiguous(), la.contiguous(), be.contiguous(), X.contiguous(),
                                   W.contiguous(), sent.contiguous(), gsent.contiguous(), dW.contiguous().float(),
                                   dsfin.contiguous().float(), dk, dv, dla, dbe, dX, dS0,
                                   T, NC, T * D, T, D=D, BC=C, BV=BV)
    return dk, dv, dla, dbe, dX, dS0


@_op_delta_bwd.register_fake
def _(k, v, la, be, X, W, sent, gsent, dW, dsfin):
    B, H, T, D = k.shape; NC = X.shape[2]
    e = lambda *sh: torch.empty(*sh, device=k.device, dtype=torch.float32)
    return e(B, H, T, D), e(B, H, T, D), e(B, H, T), e(B, H, T), e(B, H, NC, C, C), e(B, H, D, D)


def _delta_setup(ctx, inputs, output):
    k, v, la, be, X, S0, has_s0 = inputs
    W, sent, sfin = output
    ctx.save_for_backward(k, v, la, be, X, W, sent)
    ctx.has_s0 = has_s0


def _delta_backward(ctx, dW, gsent, dsfin):
    k, v, la, be, X, W, sent = ctx.saved_tensors
    if gsent is None: gsent = torch.zeros_like(sent)
    if dsfin is None: dsfin = torch.zeros(sent.shape[0], sent.shape[1], sent.shape[3], sent.shape[4], device=k.device)
    dk, dv, dla, dbe, dX, dS0 = _op_delta_bwd(k, v, la, be, X, W, sent, gsent, dW, dsfin)
    return dk.to(k.dtype), dv.to(v.dtype), dla, dbe, dX, (dS0 if ctx.has_s0 else None), None


_op_delta_fwd.register_autograd(_delta_backward, setup_context=_delta_setup)


def delta_scan_kernel(q, k, v, logalpha, beta, S0=None):
    """The delta rule end to end: X from the torch-side inverse (compiled, autograd), the loop over chunks in
    the Triton kernel, the C x C output through _op_out with the residuals W in the value slot."""
    from .assoc import _unit_lower_inverse
    B, H, T, D = q.shape
    NC = triton.cdiv(T, C); pad = NC * C - T
    la = logalpha.float(); be = beta.float()
    lc = torch.nn.functional.pad(la, (0, pad)).view(B, H, NC, C)
    L = lc.cumsum(-1)
    mask = torch.ones(C, C, device=q.device, dtype=torch.bool).tril()
    dec = torch.exp((L.unsqueeze(-1) - L.unsqueeze(-2)).clamp(max=0.0)) * mask
    kp = torch.nn.functional.pad(k, (0, 0, 0, pad)).view(B, H, NC, C, D).float()
    bc = torch.nn.functional.pad(be, (0, pad)).view(B, H, NC, C)
    X = _unit_lower_inverse(bc.unsqueeze(-1) * ((kp @ kp.transpose(-1, -2)) * dec).tril(-1))
    has = S0 is not None
    s0 = S0.float() if has else torch.zeros(B, H, D, D, device=q.device, dtype=torch.float32)
    W, sent, sfin = _op_delta_fwd(k.to(torch.bfloat16), v.float(), la, be, X, s0, has)
    o = _op_out(q.to(torch.bfloat16), k.to(torch.bfloat16), W.to(torch.bfloat16), la, sent)
    return o.to(q.dtype), sfin
