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
from typing import Optional
import torch
import torch._functorch.config
import triton
import triton.language as tl

# THE COMPILE CACHE DOES NOT SEE THESE OPS' BACKWARD. AOTAutograd's on-disk cache keys on the forward graph
# dynamo hands it; the Python backward glue below (_backward, _delta_backward, ...) and the ops' layout tags
# are only read when the backward is traced, which a cache hit skips. Measured on 2026-09-17: an edit to
# _backward and a needs_contiguous_strides tag had no effect on the compiled program for a whole session,
# every process loading a backward compiled from the code as it was before the edit. The FX graph cache is
# keyed on the traced graphs themselves and stays on; only the tracing must happen every time.
torch._functorch.config.enable_autograd_cache = False

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
# needs_contiguous_strides on every op: without it inductor handed the output kernel's backward its incoming
# gradient in the transposed order of the RMSNorm after it (a 12-layer model only; a 2-layer one got it
# contiguous), and the op's own .contiguous() then ran as an eager copy of 11 MB per layer. With the tag the
# copy is the producer's, fused into the kernel that writes the tensor.
@torch.library.custom_op("membrae::assoc_out", mutates_args=(), tags=(torch.Tag.needs_contiguous_strides,))
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


@torch.library.custom_op("membrae::assoc_out_bwd", mutates_args=(), tags=(torch.Tag.needs_contiguous_strides,))
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
    # do arrives with the strides of the transpose after the output RMSNorm, (B,T,H,D) order; made contiguous
    # HERE, in the traced backward, the copy fuses into the RMSNorm backward kernel that produces it, where
    # inside the op it was an eager copy of 11 MB per layer (0.23 ms per micro-step at micro-batch 4)
    dq, dk, dv, dla, dsent = _op_out_bwd(q, k, v, la, sent, do.contiguous(), ctx.bc)
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


def _chunk_kv(k, v, w, NC):
    """Each chunk's KV contribution, (B,H,NC,D,D) fp32, from the bf16 k and v the output kernel reads and the
    per-token weight w = exp(L_C - L_s), (B,H,NC,C,1) fp32. The weighted k is rounded to bf16 for the dot."""
    B, H, T, D = k.shape
    kw = (k.view(B, H, NC, -1, D).float() * w).to(torch.bfloat16)
    return (kw.transpose(-1, -2) @ v.view(B, H, NC, -1, D)).float()


@torch.library.custom_op("membrae::prefix_scan", mutates_args=(), tags=(torch.Tag.needs_contiguous_strides,))
def _op_prefix(k: torch.Tensor, v: torch.Tensor, w: torch.Tensor, a_log: torch.Tensor, S0: torch.Tensor,
               has_s0: bool, bv: int) -> tuple[torch.Tensor, torch.Tensor]:
    # THE KV MATMUL LIVES INSIDE THE OP. Traced by autograd it kept the fp32 k for the weighting's backward
    # and both bf16 operands for its own, 44 MiB per layer at B4 T2048, on top of the bf16 k and v the output
    # kernel saves; inside the op nothing is saved but those same k and v, and the backward recomputes the
    # weighted k (one elementwise pass) from them.
    B, H, NC = w.shape[:3]; D = k.shape[-1]; BH = B * H
    KV = _chunk_kv(k, v, w, NC)
    sent = torch.empty(B, H, NC, D, D, device=k.device, dtype=torch.bfloat16)   # see _op_delta_fwd: bf16 saved states
    sfin = torch.empty(B, H, D, D, device=k.device, dtype=torch.float32)
    _prefix_fwd[(BH, D // bv)](KV, a_log.contiguous(), S0.contiguous() if has_s0 else sfin,
                               sent, sfin, NC, HAS_S0=has_s0, D=D, BV=bv)
    return sent, sfin


@_op_prefix.register_fake
def _(k, v, w, a_log, S0, has_s0, bv):
    B, H, NC = w.shape[:3]; D = k.shape[-1]
    return (torch.empty(B, H, NC, D, D, device=k.device, dtype=torch.bfloat16),
            torch.empty(B, H, D, D, device=k.device, dtype=torch.float32))


@torch.library.custom_op("membrae::prefix_scan_bwd", mutates_args=(), tags=(torch.Tag.needs_contiguous_strides,))
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
    k, v, w, a_log, S0, has_s0, bv = inputs
    ctx.save_for_backward(k, v, w, a_log, output[0]); ctx.has_s0 = has_s0; ctx.bv = bv


def _prefix_backward(ctx, dsent, dsfin):
    k, v, w, a_log, sent = ctx.saved_tensors
    B, H, NC = w.shape[:3]; D = k.shape[-1]
    if dsent is None: dsent = torch.zeros(sent.shape, device=sent.device, dtype=torch.float32)
    if dsfin is None: dsfin = torch.zeros(sent.shape[0], sent.shape[1], sent.shape[3], sent.shape[4], device=sent.device)
    # contiguous in the TRACED backward, where the copy fuses into the kernel that produced the gradient;
    # inside the op it was an eager copy (the 2026-09-17 finding on the other ops' glue)
    dKV, dal, dS0 = _op_prefix_bwd(sent, a_log, dsent.contiguous(), dsfin.contiguous(), ctx.bv)
    # through KV_c = (k w)^T v per chunk, the weighted k recomputed
    kc = k.view(B, H, NC, -1, D); vc = v.view(B, H, NC, -1, D)
    kw = (kc.float() * w).to(torch.bfloat16)
    dKVb = dKV.to(torch.bfloat16)
    dkw = (vc @ dKVb.transpose(-1, -2)).float()                                 # (B,H,NC,C,D)
    dv = (kw @ dKVb).float()
    dk = dkw * w
    dw = (dkw * kc.float()).sum(-1, keepdim=True)
    return (dk.view(k.shape).to(k.dtype), dv.view(v.shape).to(v.dtype), dw, dal.sum(0),
            (dS0 if ctx.has_s0 else None), None, None)


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
    kb = k.to(torch.bfloat16); vb = v.to(torch.bfloat16)
    if pad:                                                                     # the prefix op wants whole chunks
        kb = torch.nn.functional.pad(kb, (0, 0, 0, pad)); vb = torch.nn.functional.pad(vb, (0, 0, 0, pad))
    has = S0 is not None
    s0 = S0.float() if has else torch.zeros(B, H, D, D, device=q.device, dtype=torch.float32)
    sent, sfin = _op_prefix(kb, vb, w, a_log, s0, has, bv)
    o = _op_out(q.to(torch.bfloat16), kb[..., :T, :], vb[..., :T, :], la, sent, C)
    # sent is the state ENTERING each chunk, (B,H,NC,D,D), returned so a later layer can read this memory
    return o.to(q.dtype), sfin, sent


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


@torch.library.custom_op("membrae::delta_loop_fwd", mutates_args=(), tags=(torch.Tag.needs_contiguous_strides,))
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


@torch.library.custom_op("membrae::delta_loop_bwd", mutates_args=(), tags=(torch.Tag.needs_contiguous_strides,))
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
    dk, dv, dla, dbe, dX, dS0 = _op_delta_bwd(k, v, la, be, X, W, sent, gsent.contiguous(), dW.contiguous(),
                                              dsfin.contiguous(), ctx.bc, ctx.bv)
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
    # v in bf16: the loop kernel rounds the right-hand side beta (v - e^L k S) to bf16 for its dot in any
    # case, and the fp32 v the op saved for its backward was 22 MiB per layer at B4 T2048
    W, sent, sfin = _op_delta_fwd(k.to(torch.bfloat16), v.to(torch.bfloat16), la, be, X, s0, has, Cc, bv, bv_bwd)
    o = _op_out(q.to(torch.bfloat16), k.to(torch.bfloat16), W, la, sent, Cc)
    # sent is the state ENTERING each chunk, (B,H,NC,D,D). Returned so a later layer can read this memory with
    # its own queries: the 1:2 hybrid of 2026-09-17 showed attention recovers exact recall the carrier cannot,
    # and the state is the only place the carrier's long-range content actually lives.
    return o.to(q.dtype), sfin, sent


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
    # IEEE fp32, NOT bfloat16. The doubling sums terms that reach magnitude ~5,000 to produce an answer whose
    # entries are at most 1, so the answer sits about 13 bits below the largest term. bfloat16 keeps 8 and
    # tf32 keeps 10, and both lose it entirely once N's entries approach 1 -- which is exactly what a run of
    # one repeated token does, since its keys are near-identical and the learned beta reaches 0.99999.
    # Measured 2026-09-22 on 128 repeated newlines: the true inverse has max |X| = 1, this returned 61.5, and
    # one carrier's state reached 4.3e14 where the float64 recurrence gives 2.8. See tests/test_assoc.py.
    R = tl.dot(R, R, out_dtype=tl.float32, input_precision="ieee")               # (-N)^2
    X = X + tl.dot(R, X, out_dtype=tl.float32, input_precision="ieee")           # terms 0..3
    R = tl.dot(R, R, out_dtype=tl.float32, input_precision="ieee")               # (-N)^4
    X = X + tl.dot(R, X, out_dtype=tl.float32, input_precision="ieee")           # terms 0..7
    R = tl.dot(R, R, out_dtype=tl.float32, input_precision="ieee")               # (-N)^8
    X = X + tl.dot(R, X, out_dtype=tl.float32, input_precision="ieee")           # terms 0..15: exact
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
    # the substitution across blocks is in ieee fp32 for the same reason as the doubling inside one
    X0 = tl.dot(_inv16(n_ptr, base, 0, BC), E0, out_dtype=tl.float32, input_precision="ieee")
    rhs = E1 - tl.dot(_blk(n_ptr, base, 1, 0, BC), X0, out_dtype=tl.float32, input_precision="ieee")
    X1 = tl.dot(_inv16(n_ptr, base, 1, BC), rhs, out_dtype=tl.float32, input_precision="ieee")
    rhs = E2 - tl.dot(_blk(n_ptr, base, 2, 0, BC), X0, out_dtype=tl.float32, input_precision="ieee") \
             - tl.dot(_blk(n_ptr, base, 2, 1, BC), X1, out_dtype=tl.float32, input_precision="ieee")
    X2 = tl.dot(_inv16(n_ptr, base, 2, BC), rhs, out_dtype=tl.float32, input_precision="ieee")
    rhs = E3 - tl.dot(_blk(n_ptr, base, 3, 0, BC), X0, out_dtype=tl.float32, input_precision="ieee") \
             - tl.dot(_blk(n_ptr, base, 3, 1, BC), X1, out_dtype=tl.float32, input_precision="ieee") \
             - tl.dot(_blk(n_ptr, base, 3, 2, BC), X2, out_dtype=tl.float32, input_precision="ieee")
    X3 = tl.dot(_inv16(n_ptr, base, 3, BC), rhs, out_dtype=tl.float32, input_precision="ieee")
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


@torch.library.custom_op("membrae::delta_inv", mutates_args=(), tags=(torch.Tag.needs_contiguous_strides,))
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


@torch.library.custom_op("membrae::delta_inv_bwd", mutates_args=(), tags=(torch.Tag.needs_contiguous_strides,))
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


# ------------------------------------------------------------------ the plumbing before the scan, fused
# What the carrier does between its projections and the scan: unit-normalise q, k and v per head, scale k
# by the write gate (plain rule), rotate q and k by pos * theta per channel pair, cast to bf16, and lay the
# result out (B,H,T,D)-contiguous for the kernels. Under torch.compile that was three kernels per layer
# (two forward, one backward) at 0.19 ms per layer, B4 T2048, with fp32 intermediates of every tensor kept
# for the backward. Here one forward pass reads the three bf16 projections and the gate logits and writes
# the three bf16 outputs; one backward pass recomputes the norms, the angles and the gate from the same
# inputs (all cheap: one sum of squares per row, one cos and sin per pair) and writes the grads of the
# projections in THEIR layout, (B,T,H,D), which is what the projection GEMMs' backward reads without a
# transpose. The gate's grad is per (b,t,h); theta's is a reduction over batch and time, accumulated per
# program into a partial buffer and summed by the traced caller. The rotation is orthogonal, so the gate
# can be applied on either side of it; the write gate's sigmoid is taken in fp32 here where the torch form
# rounded it to bf16.
_PREP_CFGS = [triton.Config({"BT": bt}, num_warps=w) for bt in (16, 32, 64) for w in (2, 4, 8)]


@triton.jit
def _pairs(x):
    """(BT, D) -> the even and odd columns as two (BT, D/2) tiles: loaded as one contiguous tile, split in
    registers. Loading the columns with stride 2 instead cost 2x (no vector loads): 106 us per layer backward."""
    return tl.split(tl.reshape(x, (x.shape[0], x.shape[1] // 2, 2)))


@triton.jit
def _interleave(a, b):
    return tl.reshape(tl.join(a, b), (a.shape[0], a.shape[1] * 2))


@triton.jit
def _unit(x1, x2):
    return 1.0 / tl.maximum(tl.sqrt(tl.sum(x1 * x1 + x2 * x2, 1)), 1e-12)


@triton.autotune(configs=_PREP_CFGS, key=["T", "D", "HAS_GATE"])
@triton.jit
def _prep_fwd(qp_ptr, kp_ptr, vp_ptr, gl_ptr, th_ptr, pos_ptr, q_ptr, k_ptr, v_ptr, T, H,
              HAS_GATE: tl.constexpr, D: tl.constexpr, BT: tl.constexpr):
    bh = tl.program_id(0); pt = tl.program_id(1)
    b = bh // H; h = bh % H
    t = pt * BT + tl.arange(0, BT); tm = t < T
    d = tl.arange(0, D); i = tl.arange(0, D // 2)
    m = tm[:, None] & (d[None, :] >= 0)
    in_off = ((b * T + t)[:, None] * H + h) * D + d[None, :]                  # (B,T,H,D)
    out_off = (bh * T + t)[:, None] * D + d[None, :]                           # (B,H,T,D)
    pos = tl.load(pos_ptr + t, mask=tm, other=0).to(tl.float32)
    th = tl.load(th_ptr + h * (D // 2) + i)
    ang = pos[:, None] * th[None, :]
    c = tl.cos(ang); s = tl.sin(ang)
    # q: normalise, rotate
    x1, x2 = _pairs(tl.load(qp_ptr + in_off, mask=m, other=0.).to(tl.float32))
    inv = _unit(x1, x2)
    tl.store(q_ptr + out_off, (_interleave(x1 * c - x2 * s, x1 * s + x2 * c) * inv[:, None]).to(tl.bfloat16), mask=m)
    # k: normalise, gate, rotate
    x1, x2 = _pairs(tl.load(kp_ptr + in_off, mask=m, other=0.).to(tl.float32))
    inv = _unit(x1, x2)
    if HAS_GATE:
        g = tl.load(gl_ptr + (b * T + t) * H + h, mask=tm, other=0.).to(tl.float32)
        inv = inv * tl.sigmoid(g)
    tl.store(k_ptr + out_off, (_interleave(x1 * c - x2 * s, x1 * s + x2 * c) * inv[:, None]).to(tl.bfloat16), mask=m)
    # v: normalise
    x = tl.load(vp_ptr + in_off, mask=m, other=0.).to(tl.float32)
    inv = 1.0 / tl.maximum(tl.sqrt(tl.sum(x * x, 1)), 1e-12)
    tl.store(v_ptr + out_off, (x * inv[:, None]).to(tl.bfloat16), mask=m)


@triton.autotune(configs=_PREP_CFGS, key=["T", "D", "HAS_GATE"])
@triton.jit
def _prep_bwd(qp_ptr, kp_ptr, vp_ptr, gl_ptr, th_ptr, pos_ptr, dq_ptr, dk_ptr, dv_ptr,
              dqp_ptr, dkp_ptr, dvp_ptr, dgl_ptr, dth_ptr, T, H, NT,
              HAS_GATE: tl.constexpr, D: tl.constexpr, BT: tl.constexpr):
    """With u = x / |x|, r = R(ang) u and g the grad on r: du = R^T g, dx = (du - u (u . du)) / |x|, and
    d ang = g2 r1 - g1 r2 per pair (the derivative of a rotation is the rotation by a quarter turn)."""
    bh = tl.program_id(0); pt = tl.program_id(1)
    b = bh // H; h = bh % H
    t = pt * BT + tl.arange(0, BT); tm = t < T
    d = tl.arange(0, D); i = tl.arange(0, D // 2)
    m = tm[:, None] & (d[None, :] >= 0)
    in_off = ((b * T + t)[:, None] * H + h) * D + d[None, :]
    out_off = (bh * T + t)[:, None] * D + d[None, :]
    pos = tl.load(pos_ptr + t, mask=tm, other=0).to(tl.float32)
    th = tl.load(th_ptr + h * (D // 2) + i)
    ang = pos[:, None] * th[None, :]
    c = tl.cos(ang); s = tl.sin(ang)
    # q
    x1, x2 = _pairs(tl.load(qp_ptr + in_off, mask=m, other=0.).to(tl.float32))
    inv = _unit(x1, x2)
    u1 = x1 * inv[:, None]; u2 = x2 * inv[:, None]
    g1, g2 = _pairs(tl.load(dq_ptr + out_off, mask=m, other=0.).to(tl.float32))
    dang = g2 * (u1 * c - u2 * s) - g1 * (u1 * s + u2 * c)
    du1 = g1 * c + g2 * s; du2 = g2 * c - g1 * s
    dot = tl.sum(u1 * du1 + u2 * du2, 1)
    tl.store(dqp_ptr + in_off, _interleave(du1 - u1 * dot[:, None], du2 - u2 * dot[:, None]) * inv[:, None], mask=m)
    # k
    x1, x2 = _pairs(tl.load(kp_ptr + in_off, mask=m, other=0.).to(tl.float32))
    inv = _unit(x1, x2)
    u1 = x1 * inv[:, None]; u2 = x2 * inv[:, None]
    g1, g2 = _pairs(tl.load(dk_ptr + out_off, mask=m, other=0.).to(tl.float32))
    r1 = u1 * c - u2 * s; r2 = u1 * s + u2 * c                                 # R u, before the gate
    if HAS_GATE:
        sg = tl.sigmoid(tl.load(gl_ptr + (b * T + t) * H + h, mask=tm, other=0.).to(tl.float32))
        dsg = tl.sum(g1 * r1 + g2 * r2, 1)
        tl.store(dgl_ptr + (b * T + t) * H + h, dsg * sg * (1.0 - sg), mask=tm)
        g1 = g1 * sg[:, None]; g2 = g2 * sg[:, None]                           # the rest sees sg * g
    dang += g2 * r1 - g1 * r2
    du1 = g1 * c + g2 * s; du2 = g2 * c - g1 * s
    dot = tl.sum(u1 * du1 + u2 * du2, 1)
    tl.store(dkp_ptr + in_off, _interleave(du1 - u1 * dot[:, None], du2 - u2 * dot[:, None]) * inv[:, None], mask=m)
    # theta: this program's share of the reduction over batch and time
    tl.store(dth_ptr + ((b * NT + pt) * H + h) * (D // 2) + i, tl.sum(dang * pos[:, None], 0))
    # v
    x = tl.load(vp_ptr + in_off, mask=m, other=0.).to(tl.float32)
    inv = 1.0 / tl.maximum(tl.sqrt(tl.sum(x * x, 1)), 1e-12)
    u = x * inv[:, None]
    g = tl.load(dv_ptr + out_off, mask=m, other=0.).to(tl.float32)
    dot = tl.sum(u * g, 1)
    tl.store(dvp_ptr + in_off, (g - u * dot[:, None]) * inv[:, None], mask=m)


_PREP_BT_MIN = 16                                  # the smallest tile in _PREP_CFGS sizes the theta partials


@torch.library.custom_op("membrae::assoc_prep", mutates_args=(), tags=(torch.Tag.needs_contiguous_strides,))
def _op_prep(qp: torch.Tensor, kp: torch.Tensor, vp: torch.Tensor, gl: Optional[torch.Tensor],
             theta: torch.Tensor, pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, T, H, D = qp.shape
    e = lambda: torch.empty(B, H, T, D, device=qp.device, dtype=torch.bfloat16)
    q, k, v = e(), e(), e()
    has = gl is not None
    _prep_fwd[lambda meta: (B * H, triton.cdiv(T, meta["BT"]))](
        qp.contiguous(), kp.contiguous(), vp.contiguous(), gl.contiguous() if has else theta,
        theta.contiguous().float(), pos.contiguous(), q, k, v, T, H, HAS_GATE=has, D=D)
    return q, k, v


@_op_prep.register_fake
def _(qp, kp, vp, gl, theta, pos):
    B, T, H, D = qp.shape
    e = lambda: torch.empty(B, H, T, D, device=qp.device, dtype=torch.bfloat16)
    return e(), e(), e()


@torch.library.custom_op("membrae::assoc_prep_bwd", mutates_args=(), tags=(torch.Tag.needs_contiguous_strides,))
def _op_prep_bwd(qp: torch.Tensor, kp: torch.Tensor, vp: torch.Tensor, gl: Optional[torch.Tensor],
                 theta: torch.Tensor, pos: torch.Tensor, dq: torch.Tensor, dk: torch.Tensor,
                 dv: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    B, T, H, D = qp.shape; NT = triton.cdiv(T, _PREP_BT_MIN)
    dqp = torch.empty_like(qp); dkp = torch.empty_like(kp); dvp = torch.empty_like(vp)
    dgl = torch.empty(B, T, H, device=qp.device, dtype=torch.float32)
    dth = torch.zeros(B * NT, H, D // 2, device=qp.device, dtype=torch.float32)   # a larger tile fills fewer rows
    has = gl is not None
    _prep_bwd[lambda meta: (B * H, triton.cdiv(T, meta["BT"]))](
        qp.contiguous(), kp.contiguous(), vp.contiguous(), gl.contiguous() if has else theta,
        theta.contiguous().float(), pos.contiguous(), dq.contiguous(), dk.contiguous(), dv.contiguous(),
        dqp, dkp, dvp, dgl, dth, T, H, NT, HAS_GATE=has, D=D)
    return dqp, dkp, dvp, dgl, dth


@_op_prep_bwd.register_fake
def _(qp, kp, vp, gl, theta, pos, dq, dk, dv):
    B, T, H, D = qp.shape; NT = triton.cdiv(T, _PREP_BT_MIN)
    return (torch.empty_like(qp), torch.empty_like(kp), torch.empty_like(vp),
            torch.empty(B, T, H, device=qp.device, dtype=torch.float32),
            torch.empty(B * NT, H, D // 2, device=qp.device, dtype=torch.float32))


def _prep_setup(ctx, inputs, output):
    qp, kp, vp, gl, theta, pos = inputs
    ctx.has_gate = gl is not None
    ctx.save_for_backward(qp, kp, vp, theta, pos, *([gl] if ctx.has_gate else []))


def _prep_backward(ctx, dq, dk, dv):
    qp, kp, vp, theta, pos = ctx.saved_tensors[:5]
    gl = ctx.saved_tensors[5] if ctx.has_gate else None
    dqp, dkp, dvp, dgl, dth = _op_prep_bwd(qp, kp, vp, gl, theta, pos, dq, dk, dv)
    return dqp, dkp, dvp, (dgl.to(gl.dtype) if ctx.has_gate else None), dth.sum(0).to(theta.dtype), None


_op_prep.register_autograd(_prep_backward, setup_context=_prep_setup)


def prep_kernel(qp, kp, vp, gate_logits, theta, pos):
    """qp, kp, vp: the projections as (B,T,H,D) views; gate_logits: (B,T,H) or None (delta rule); theta:
    (H, D/2); pos: (T,) integer positions. Returns bf16 q, k, v as (B,H,T,D), ready for the scan kernels."""
    return _op_prep(qp, kp, vp, gate_logits, theta, pos)
