"""TRAINING AND GENERATION AT 1,000,000 TOKENS. The vision's headline claim is unbounded context at constant cost.
We have verified 200k; this pushes to 1M in both directions.

  A) TRAINING at 1M -- stream a 1M-token document in chunks, carrying state, with backward on every chunk.
     If throughput and peak memory stay flat from 32k to 1M, training context really is unbounded.
  B) GENERATION at 1M -- read 1M tokens into the state, then generate. Per-token cost must not grow with how
     much has been read, because the state is fixed-size.

Both on the CURRENT architecture: gated, qk-normed (sum-of-squares), outer-product taylor, boundary key threaded.
"""
import sys, time
sys.path.insert(0, "/opt/code/membrae")
from util import setup_log
setup_log("million")
import numpy as np, torch, torch.nn.functional as F
import core as C
torch._dynamo.config.recompile_limit = 64

DEV = C.DEV
V = 32768; D = 384; NL = 4; DIN = 8; CH = 4096


def build(T=CH):
    cfg = C.Config(D=D, DIN=DIN, DV=D, NL=NL, T=T, res_scale=16 ** -0.5, tie_emb=True)
    cfg.gated = True; cfg.qknorm = True
    return C.Core(cfg).to(DEV), cfg


def train_at(T, B=1, compile_it=True):
    """the real training loop: chunked, state carried, backward per chunk."""
    torch.manual_seed(0)
    m, cfg = build()
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
    doc()
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    t0 = time.time(); doc(); torch.cuda.synchronize()
    tps = B * T / (time.time() - t0) / 1e3
    pk = torch.cuda.max_memory_allocated() / 2 ** 30
    st_mib = NL * B * m.cfg.FE * m.cfg.DV * 4 / 2 ** 20
    del m, opt, fwd; torch.cuda.empty_cache()
    return tps, pk, st_mib


@torch.no_grad()
def gen_after(T, ngen=64):
    """read T tokens into the state, then generate -- per-token cost must not depend on T."""
    torch.manual_seed(0)
    m, cfg = build(); m.eval()
    X = torch.randint(0, V, (1, T), device=DEV)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    st = None; pv = None
    for i in range(T // CH):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, st, pv = m(X[:, i * CH:(i + 1) * CH], states=st, return_states=True, prev=pv, return_prev=True)
    torch.cuda.synchronize()
    read_s = time.time() - t0
    cur = X[:, -1:]
    torch.cuda.synchronize(); t1 = time.time()
    for _ in range(ngen):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg, st, pv = m.step(cur, st, pv)
        cur = torch.tensor([[int(lg[0].float().argmax())]], device=DEV)
    torch.cuda.synchronize()
    ms = (time.time() - t1) / ngen * 1000
    pk = torch.cuda.max_memory_allocated() / 2 ** 30
    del m; torch.cuda.empty_cache()
    return T / read_s / 1e3, ms, pk


if __name__ == "__main__":
    print(f"A) TRAINING CONTEXT TO 1M (chunk {CH}, backward on every chunk, state carried)\n", flush=True)
    print(f"  {'document':>10s} {'chunks':>7s} {'ktok/s':>8s} {'peak GiB':>9s} {'state MiB':>10s}", flush=True)
    base = None
    for T in (32768, 131072, 262144, 524288, 1048576):
        try:
            tps, pk, st = train_at(T)
            if base is None: base = tps
            print(f"  {T:10d} {T//CH:7d} {tps:8.0f} {pk:9.2f} {st:10.3f}", flush=True)
        except torch.OutOfMemoryError:
            print(f"  {T:10d} {T//CH:7d} {'OOM':>8s}", flush=True); torch.cuda.empty_cache()

    print(f"\nB) GENERATION AFTER READING UP TO 1M TOKENS\n", flush=True)
    print(f"  {'read':>10s} {'read ktok/s':>12s} {'ms/token':>10s} {'tok/s':>8s} {'peak GiB':>9s}", flush=True)
    for T in (32768, 131072, 524288, 1048576):
        try:
            rt, ms, pk = gen_after(T)
            print(f"  {T:10d} {rt:12.0f} {ms:10.2f} {1000/ms:8.0f} {pk:9.2f}", flush=True)
        except torch.OutOfMemoryError:
            print(f"  {T:10d} {'OOM':>12s}", flush=True); torch.cuda.empty_cache()
    print(f"\n  flat ms/token across a 32x range = generation cost is independent of how much was read.", flush=True)
