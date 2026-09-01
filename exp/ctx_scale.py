"""RE-MEASURE THE CHUNK LENGTH, UP TO 200k CONTEXT. The old measurement (chunk 4096 = 2x chunk 2048) was taken on
the OLD taylor() with the gather backward, before the boundary key was threaded and before gating. All three change
the per-chunk cost, so the optimum may have moved.

What is being measured: a document of length T is streamed in chunks of CH with state carried between them, which
is exactly what the trainer does. The document length is unbounded in principle -- our whole claim -- so this also
tests whether throughput and memory really stay FLAT out to 200k tokens with the current architecture.

CH is both the attention window (cost is O(CH^2) inside a chunk) and the BACKPROP HORIZON (state is detached
between chunks), so a larger CH is better for credit assignment as long as it fits and stays fast.
"""
import sys, time
sys.path.insert(0, "/opt/code/membrae")
from util import setup_log
setup_log("ctx_scale")
import numpy as np, torch, torch.nn.functional as F
import core as C
torch._dynamo.config.recompile_limit = 64

DEV = C.DEV
V = 32768; D = 384; NL = 4; DIN = 8      # the infant now training


def build():
    cfg = C.Config(D=D, DIN=DIN, DV=D, NL=NL, T=2048, res_scale=16 ** -0.5, tie_emb=True)
    cfg.gated = True; cfg.qknorm = True
    return C.Core(cfg).to(DEV), cfg


def measure(T, CH, B=1, reps=2, compile_it=True):
    """stream a T-token document in CH-sized chunks, carrying state -- the real training loop."""
    torch.manual_seed(0)
    m, cfg = build(); cfg.T = CH
    opt = torch.optim.Adam(m.parameters(), lr=1e-4)
    fwd = torch.compile(m) if compile_it else m
    X = torch.randint(0, V, (B, T + 1), device=DEV)
    n = T // CH
    def doc():
        st = None; pv = None; opt.zero_grad()
        for i in range(n):
            xi = X[:, i * CH:(i + 1) * CH]; yi = X[:, i * CH + 1:(i + 1) * CH + 1]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg, st, pv = fwd(xi, states=st, return_states=True, prev=pv, return_prev=True)
                l = F.cross_entropy(lg.reshape(-1, V), yi.reshape(-1)) / n
            st = [(a.detach(), b.detach()) for a, b in st]; pv = [p.detach() for p in pv]
            l.backward()
        opt.step()
    doc()                                                     # warm up / compile
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for _ in range(reps): doc()
    torch.cuda.synchronize()
    tps = B * T * reps / (time.time() - t0) / 1e3
    pk = torch.cuda.max_memory_allocated() / 2 ** 30
    del m, opt, fwd; torch.cuda.empty_cache()
    return tps, pk


if __name__ == "__main__":
    print(f"CHUNK LENGTH, RE-MEASURED on the CURRENT architecture (gated, qk-norm, outer-product taylor,\n"
          f"boundary key threaded). Infant shape D={D}x{NL}L DIN={DIN}, compiled.\n", flush=True)
    print("A) WHICH CHUNK IS FASTEST? (fixed 32768-token document, as the trainer uses)\n", flush=True)
    print(f"  {'chunk':>7s} {'chunks/doc':>11s} {'ktok/s':>8s} {'peak GiB':>9s}", flush=True)
    best = (0, None)
    for CH in (1024, 2048, 4096, 8192, 16384):
        try:
            tps, pk = measure(32768, CH)
            if tps > best[0]: best = (tps, CH)
            print(f"  {CH:7d} {32768//CH:11d} {tps:8.0f} {pk:9.2f}", flush=True)
        except torch.OutOfMemoryError:
            print(f"  {CH:7d} {32768//CH:11d} {'OOM':>8s}", flush=True); torch.cuda.empty_cache()
    print(f"\n  fastest: chunk {best[1]} at {best[0]:.0f} ktok/s", flush=True)

    print(f"\nB) DOES IT STAY FLAT TO 200k? (chunk {best[1]}, document length growing)\n", flush=True)
    print(f"  {'document':>10s} {'chunks':>7s} {'ktok/s':>8s} {'peak GiB':>9s} {'vs 32k':>8s}", flush=True)
    base = None
    for T in (32768, 65536, 131072, 200704):
        try:
            tps, pk = measure(T, best[1], reps=1)
            if base is None: base = tps
            print(f"  {T:10d} {T//best[1]:7d} {tps:8.0f} {pk:9.2f} {tps/base:7.2f}x", flush=True)
        except torch.OutOfMemoryError:
            print(f"  {T:10d} {T//best[1]:7d} {'OOM':>8s}", flush=True); torch.cuda.empty_cache()
    print(f"\n  flat throughput and flat memory = the constant-cost claim holds at 200k tokens.", flush=True)
