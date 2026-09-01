"""WE IMPLEMENTED LINEAR ATTENTION QUADRATICALLY. core.Layer builds the full [B,T,T] matrix:

    A = einsum('btf,bsf->bts', qf, kf); A = A.tril(); num = einsum('bts,bsd->btd', A, v)

That is O(T^2) in both time and memory -- exactly what linear attention exists to avoid. Measured against a
transformer at matched shape and data: transformer 338 ktok/s, ours 147 ktok/s. We are 2.3x SLOWER than the thing
we are supposed to beat, and the whole efficiency claim rests on this.

THE FIX -- CHUNKED linear attention. Split the sequence into blocks of size c. Inside a block use the quadratic
form (cheap: c^2, not T^2); across blocks carry the running state M = sum kf (x) v exactly as the recurrence
already does between chunks. Mathematically identical, cost T*c*FE + T*FE*DV instead of T^2*(FE+DV).

Verifies exactness against the current implementation, then measures the speedup at the real training length.
"""
import sys, time
sys.path.insert(0, "/opt/code/membrae")
from util import setup_log
setup_log("chunked_attn")
import numpy as np, torch, torch.nn.functional as F
import core as C

DEV = C.DEV


def attn_quadratic(qf, kf, v, state):
    """what core.Layer does now: the full T x T masked form."""
    A = torch.einsum('btf,bsf->bts', qf, kf).tril()
    den = A.sum(-1, keepdim=True)
    num = torch.einsum('bts,bsd->btd', A, v)
    if state is not None:
        M, zM = state
        num = num + torch.einsum('btf,bfd->btd', qf, M)
        den = den + torch.einsum('btf,bf->bt', qf, zM).unsqueeze(-1)
    return num, den


def attn_chunked(qf, kf, v, state, c=128):
    """same result, computed blockwise so no T x T matrix is ever built.
    Inside a block: the c x c masked form. Across blocks: the running state, which is what the model already
    carries between chunks -- so this is the SAME mechanism applied one level down."""
    B, T, FE = qf.shape
    DV = v.shape[-1]
    if state is not None:
        M, z = state[0].clone(), state[1].clone()
    else:
        M = torch.zeros(B, FE, DV, device=qf.device, dtype=qf.dtype)
        z = torch.zeros(B, FE, device=qf.device, dtype=qf.dtype)
    nums, dens = [], []
    for i in range(0, T, c):
        j = min(i + c, T)
        q_b, k_b, v_b = qf[:, i:j], kf[:, i:j], v[:, i:j]
        # inter-block: everything written before this block, held in the running state
        num_b = torch.einsum('btf,bfd->btd', q_b, M)
        den_b = torch.einsum('btf,bf->bt', q_b, z).unsqueeze(-1)
        # intra-block: the small masked form
        A_b = torch.einsum('btf,bsf->bts', q_b, k_b).tril()
        num_b = num_b + torch.einsum('bts,bsd->btd', A_b, v_b)
        den_b = den_b + A_b.sum(-1, keepdim=True)
        nums.append(num_b); dens.append(den_b)
        # advance the state past this block
        M = M + torch.einsum('btf,btd->bfd', k_b, v_b)
        z = z + k_b.sum(1)
    return torch.cat(nums, 1), torch.cat(dens, 1)


if __name__ == "__main__":
    torch.manual_seed(0)
    B, FE, DV = 4, 325, 384
    print("1) EXACTNESS -- chunked vs the current quadratic form\n", flush=True)
    for T in (512, 2048):
        qf = torch.randn(B, T, FE, device=DEV) * 0.1
        kf = torch.randn(B, T, FE, device=DEV) * 0.1
        v = torch.randn(B, T, DV, device=DEV)
        st = (torch.randn(B, FE, DV, device=DEV) * 0.1, torch.randn(B, FE, device=DEV) * 0.1)
        n1, d1 = attn_quadratic(qf, kf, v, st)
        for c in (64, 128, 256):
            n2, d2 = attn_chunked(qf, kf, v, st, c)
            print(f"   T={T:5d} chunk={c:4d}: max |num diff| {(n1-n2).abs().max():.3e}   "
                  f"max |den diff| {(d1-d2).abs().max():.3e}", flush=True)
    print("\n2) SPEED AND MEMORY at the real training length\n", flush=True)
    print(f"   {'T':>6s} {'method':>18s} {'ms/call':>9s} {'peak MiB':>10s} {'speedup':>9s}", flush=True)
    for T in (512, 2048):
        qf = torch.randn(B, T, FE, device=DEV) * 0.1
        kf = torch.randn(B, T, FE, device=DEV) * 0.1
        v = torch.randn(B, T, DV, device=DEV)
        st = (torch.randn(B, FE, DV, device=DEV) * 0.1, torch.randn(B, FE, device=DEV) * 0.1)
        base = None
        for name, fn in (("quadratic (current)", lambda: attn_quadratic(qf, kf, v, st)),
                         ("chunked c=128", lambda: attn_chunked(qf, kf, v, st, 128)),
                         ("chunked c=256", lambda: attn_chunked(qf, kf, v, st, 256))):
            for _ in range(2): fn()
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            for _ in range(5): fn()
            torch.cuda.synchronize()
            dt = (time.time() - t0) / 5 * 1000
            pk = torch.cuda.max_memory_allocated() / 2 ** 20
            if base is None: base = dt
            print(f"   {T:6d} {name:>18s} {dt:9.2f} {pk:10.0f} {base/dt:8.2f}x", flush=True)
