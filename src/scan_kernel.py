"""A sequential complex recurrence as one Triton kernel per direction: h_t = lambda_t h_{t-1} + u_t.

One program owns a block of channels of one sequence and walks t from 0 to T, so the recurrence is computed
literally, in the order it is defined, with |lambda_t| <= 1 at every step: nothing here can overflow at any
forgetting rate, chunked or not. Per-token lambda_t = exp(logmag_t + i ang_t) is the general case; the
time-invariant carrier is the special case where the same row is passed at every t.

The FFT path stays for the time-invariant carrier where it is faster on training windows. This kernel is what
the forget-gated carrier trains through (its chunked pure-torch form compiled in 550 s and ran 3x slower than
the FFT), and it is the natural inference form for every variant: state in, state out, one token at a time.

Backward is the adjoint recurrence run in reverse, derived in the real 2-vector form. With R_t the rotation-
scaling matrix of lambda_t and h_t = R_t h_{t-1} + (u_t, 0):
    a_t = (g_t, 0) + R_{t+1}^T a_{t+1}      adjoint state, reverse in time
    dL/du_t = a_t[0]
    dL/dlogmag_t = a_t . (R_t h_{t-1})       because dR/dlogmag = R
    dL/dang_t = a_t . (R'_t h_{t-1})         with R' = mag [[-sin, -cos], [cos, -sin]]
tests/test_spin.py checks all three against autograd through the sequential definition."""
import torch
import triton
import triton.language as tl


@triton.jit
def _fwd(u_ptr, lm_ptr, an_ptr, h0r_ptr, h0i_ptr, hr_ptr, hi_ptr,
         T, D, s_b, s_t, HAS_H0: tl.constexpr, BLOCK: tl.constexpr):
    b = tl.program_id(0)
    d = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    m = d < D
    if HAS_H0:
        hr = tl.load(h0r_ptr + b * D + d, mask=m, other=0.0).to(tl.float32)
        hi = tl.load(h0i_ptr + b * D + d, mask=m, other=0.0).to(tl.float32)
    else:
        hr = tl.zeros([BLOCK], dtype=tl.float32)
        hi = tl.zeros([BLOCK], dtype=tl.float32)
    base = b * s_b + d
    for t in range(T):
        off = base + t * s_t
        u = tl.load(u_ptr + off, mask=m, other=0.0).to(tl.float32)
        lm = tl.load(lm_ptr + off, mask=m, other=0.0).to(tl.float32)
        an = tl.load(an_ptr + off, mask=m, other=0.0).to(tl.float32)
        mag = tl.exp(lm)
        c = mag * tl.cos(an)
        s = mag * tl.sin(an)
        nr = c * hr - s * hi + u
        ni = s * hr + c * hi
        hr = nr
        hi = ni
        tl.store(hr_ptr + off, hr, mask=m)
        tl.store(hi_ptr + off, hi, mask=m)


@triton.jit
def _bwd(g_ptr, lm_ptr, an_ptr, hr_ptr, hi_ptr, h0r_ptr, h0i_ptr, gli_ptr,
         du_ptr, dlm_ptr, dan_ptr, dh0r_ptr, dh0i_ptr,
         T, D, s_b, s_t, HAS_H0: tl.constexpr, BLOCK: tl.constexpr):
    b = tl.program_id(0)
    d = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    m = d < D
    ar = tl.zeros([BLOCK], dtype=tl.float32)      # adjoint state, (real, imag) components
    ai = tl.load(gli_ptr + b * D + d, mask=m, other=0.0).to(tl.float32)   # gradient on the final Im state
    base = b * s_b + d
    for i in range(T):
        t = T - 1 - i
        off = base + t * s_t
        g = tl.load(g_ptr + off, mask=m, other=0.0).to(tl.float32)
        lm = tl.load(lm_ptr + off, mask=m, other=0.0).to(tl.float32)
        an = tl.load(an_ptr + off, mask=m, other=0.0).to(tl.float32)
        # a_t = (g_t, 0) + R_{t+1}^T a_{t+1}; the R^T part was folded in at the end of the previous iteration
        ar = ar + g
        if t > 0:
            pr = tl.load(hr_ptr + off - s_t, mask=m, other=0.0)
            pi = tl.load(hi_ptr + off - s_t, mask=m, other=0.0)
        else:
            if HAS_H0:
                pr = tl.load(h0r_ptr + b * D + d, mask=m, other=0.0).to(tl.float32)
                pi = tl.load(h0i_ptr + b * D + d, mask=m, other=0.0).to(tl.float32)
            else:
                pr = tl.zeros([BLOCK], dtype=tl.float32)
                pi = tl.zeros([BLOCK], dtype=tl.float32)
        mag = tl.exp(lm)
        c = mag * tl.cos(an)
        s = mag * tl.sin(an)
        # R h_{t-1} and R' h_{t-1}
        rr = c * pr - s * pi
        ri = s * pr + c * pi
        qr = -s * pr - c * pi
        qi = c * pr - s * pi
        tl.store(du_ptr + off, ar, mask=m)
        tl.store(dlm_ptr + off, ar * rr + ai * ri, mask=m)
        tl.store(dan_ptr + off, ar * qr + ai * qi, mask=m)
        # propagate: a_{t-1} = R_t^T a_t
        nr = c * ar + s * ai
        ni = -s * ar + c * ai
        ar = nr
        ai = ni
    if HAS_H0:
        tl.store(dh0r_ptr + b * D + d, ar, mask=m)
        tl.store(dh0i_ptr + b * D + d, ai, mask=m)


_BLOCK = 128    # channels per program; tuned below


class _Scan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, lm, an, h0r, h0i):
        B, T, D = u.shape
        u, lm, an = u.contiguous(), lm.contiguous(), an.contiguous()
        hr = torch.empty(B, T, D, device=u.device, dtype=torch.float32)
        hi = torch.empty_like(hr)
        has = h0r is not None
        BLOCK = _BLOCK
        grid = (B, triton.cdiv(D, BLOCK))
        _fwd[grid](u, lm, an, h0r if has else u, h0i if has else u, hr, hi, T, D, T * D, D,
                   HAS_H0=has, BLOCK=BLOCK)
        ctx.save_for_backward(lm, an, hr, hi, *( (h0r, h0i) if has else () ))
        ctx.has = has
        return hr, hr[:, -1], hi[:, -1]

    @staticmethod
    def backward(ctx, g, g_last_r, g_last_i):
        lm, an, hr, hi, *h0 = ctx.saved_tensors
        B, T, D = hr.shape
        # gradients on the final state fold into the adjoint at t=T-1: the real part joins g there, the
        # imaginary part initialises the adjoint's imaginary component.
        g = g.contiguous().to(torch.float32)
        if g_last_r is not None:
            g = g.clone(); g[:, -1] += g_last_r.to(torch.float32)
        gli = g_last_i.contiguous().to(torch.float32) if g_last_i is not None else torch.zeros(B, D, device=hr.device)
        du = torch.empty(B, T, D, device=hr.device, dtype=torch.float32)
        dlm = torch.empty_like(du); dan = torch.empty_like(du)
        has = ctx.has
        dh0r = torch.empty(B, D, device=hr.device, dtype=torch.float32) if has else du
        dh0i = torch.empty(B, D, device=hr.device, dtype=torch.float32) if has else du
        BLOCK = _BLOCK
        grid = (B, triton.cdiv(D, BLOCK))
        _bwd[grid](g, lm, an, hr, hi, h0[0] if has else hr, h0[1] if has else hr, gli,
                   du, dlm, dan, dh0r, dh0i, T, D, T * D, D, HAS_H0=has, BLOCK=BLOCK)
        return du, dlm, dan, (dh0r if has else None), (dh0i if has else None)


@torch.compiler.disable
def scan_triton(u, logmag, ang, h0=None):
    """Same contract as SpinCarrier.scan_gated: (Re h, (Re h_last, Im h_last)). fp32 inside."""
    h0r, h0i = (h0[0].contiguous().float(), h0[1].contiguous().float()) if h0 is not None else (None, None)
    y, hr_last, hi_last = _Scan.apply(u.float(), logmag.float(), ang.float(), h0r, h0i)
    return y.to(u.dtype), (hr_last.to(u.dtype), hi_last.to(u.dtype))
