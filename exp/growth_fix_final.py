"""THE FIX TEST. Evidence so far:
  - a SINGLE growth pays for itself quickly (crossover at ~0.7 tok/param, new dims reach +0.42 load)
  - but MANY growths do not: FAST 4x+128 = 4.3483 vs ONCE 1x+512 = 4.2732 at identical final size
  - three initialisation fixes (random / net2net / mirror) all FAILED to beat the current one
  - mechanism: each growth scales the incumbent dims up by c=sqrt(nD/D), entrenching them (1.63x over our 9 events)
So the defect COMPOUNDS with the NUMBER of growth events. Two candidate fixes, tested against the current pace:

  A MANY   : 4 growths of +128  (our real pace -- the baseline to beat)
  B FEW    : 1 growth of +512   (fewer, larger events)
  C PRUNE  : 4 growths of +128, but at each growth the least-useful 25% of EXISTING dims are RE-INITIALISED.
             Biologically this is synaptic pruning alongside neurogenesis: the brain does not only add, it also
             clears out. Mechanically it breaks the incumbent entrenchment that the c-scaling creates.
All reach D=768 and train for the same total steps, long enough for the capacity to pay off.
"""
import sys, copy
sys.path.insert(0, "/opt/code/membrae")
from util import setup_log
setup_log("growth_fix_final")
import numpy as np, torch, torch.nn.functional as F
import core as C, growth as G
from data_mix import DataMix

DEV = C.DEV; CH = 512; B = 8; PRE = 1500; D0 = 256; TOTAL = 20000
CHECKS = [5000, 10000, 15000, 20000]
dm = DataMix(ctx=CH + 1, verbose=False, only='window2')
np.random.seed(7)
EV = [torch.tensor(dm.batch(4), device=DEV) for _ in range(6)]


def evl(m, cfg):
    m.eval(); t = 0
    with torch.no_grad():
        for X in EV:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                t += F.cross_entropy(m(X[:, :CH]).reshape(-1, cfg.vocab).float(),
                                     X[:, 1:CH + 1].reshape(-1)).item()
    m.train(); return t / len(EV)


def contrib(m, cfg, lo, hi):
    base = evl(m, cfg); sl = slice(lo, hi); sav = {}
    with torch.no_grad():
        for i, L in enumerate(m.layers):
            sav[i] = (L.Wo.weight[sl, :].clone(), L.m2.weight[sl, :].clone())
            L.Wo.weight[sl, :] = 0; L.m2.weight[sl, :] = 0
    d = evl(m, cfg) - base
    with torch.no_grad():
        for i, L in enumerate(m.layers):
            a, b = sav[i]; L.Wo.weight[sl, :] = a; L.m2.weight[sl, :] = b
    return d


def prune_weakest(m, frac=0.25):
    """Re-initialise the least-useful existing stream dims (synaptic pruning). Usefulness = how much each dim is
    WRITTEN into the stream, measured by the norm of its incoming weights across all layers."""
    with torch.no_grad():
        D = m.cfg.D
        score = torch.zeros(D, device=DEV)
        for L in m.layers:
            score += L.Wo.weight.norm(dim=1) + L.m2.weight.norm(dim=1)
        score += m.emb.weight.norm(dim=0)
        k = int(frac * D)
        weak = torch.topk(score, k, largest=False).indices          # the k least-written dims
        for L in m.layers:
            L.Wo.weight[weak, :] = G._new_lin(L.Wo.weight.shape[1], k, DEV)
            L.m2.weight[weak, :] = G._new_lin(L.m2.weight.shape[1], k, DEV)
            for nm in ('Wk', 'Wq', 'Wv'):
                W = getattr(L, nm).weight
                W[:, weak] = G._new_lin(k, W.shape[0], DEV)
            L.m1.weight[:, weak] = G._new_lin(k, L.m1.weight.shape[0], DEV)
        m.emb.weight[:, weak].normal_(0, 0.02)
    return k


def run(tag, schedule, prune, sd0):
    torch.manual_seed(1); np.random.seed(1)
    cfg = C.Config(D=D0, DIN=8, DV=D0, NL=4, T=CH, res_scale=4 ** -0.5)
    m = C.Core(cfg).to(DEV); m.load_state_dict(sd0)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3); sched = dict(schedule)
    out = {}
    for step in range(1, TOTAL + 1):
        if step in sched:
            if prune and m.cfg.D > D0: prune_weakest(m, 0.25)
            G.grow_width(m, sched[step]); cfg = m.cfg
            opt = torch.optim.Adam(m.parameters(), lr=1e-3)
        X = torch.tensor(dm.batch(B), device=DEV)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            l = F.cross_entropy(m(X[:, :CH]).reshape(-1, cfg.vocab), X[:, 1:CH + 1].reshape(-1))
        opt.zero_grad(); l.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        if step in CHECKS:
            lo = evl(m, cfg); d = contrib(m, cfg, D0, cfg.D) if cfg.D > D0 else 0.0
            out[step] = (lo, d)
            print(f"    {tag:16s} @{step:6d}: loss {lo:.4f}  grown-dim load {d:+.4f}", flush=True)
    with torch.no_grad():
        a = m.emb(EV[0][:, :CH])
        for L in m.layers: a, _ = L(a)
        rms = a.float().pow(2).mean((0, 1)).sqrt()
    print(f"    {tag:16s} stream RMS seed {rms[:D0].mean():.2f} vs grown {rms[D0:].mean():.2f} "
          f"(entrenchment {rms[:D0].mean()/max(1e-9,rms[D0:].mean()):.2f}x)", flush=True)
    del m; torch.cuda.empty_cache()
    return out


if __name__ == "__main__":
    torch.manual_seed(0); np.random.seed(0)
    cfg0 = C.Config(D=D0, DIN=8, DV=D0, NL=4, T=CH, res_scale=4 ** -0.5)
    seed = C.Core(cfg0).to(DEV); o = torch.optim.Adam(seed.parameters(), lr=1e-3)
    for it in range(PRE):
        X = torch.tensor(dm.batch(B), device=DEV)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            l = F.cross_entropy(seed(X[:, :CH]).reshape(-1, cfg0.vocab), X[:, 1:CH + 1].reshape(-1))
        o.zero_grad(); l.backward(); o.step()
    sd0 = copy.deepcopy(seed.state_dict())
    print(f"GROWTH FIX: seed D={D0} loss {evl(seed,cfg0):.4f}; all arms end at D=768 after {TOTAL} steps\n", flush=True)
    del seed; torch.cuda.empty_cache()
    q = TOTAL // 4
    R = {}
    R['C many+prune'] = run("C many+prune", [(1, 128), (q, 128), (2 * q, 128), (3 * q, 128)], True, sd0)
    print("\n  already measured: A many (current) = 3.9846 (+0.4118) | B few = 3.9076 (+0.6277)", flush=True)
    print(f"\n  {'step':>7s} " + " ".join(f"{k:>26s}" for k in R), flush=True)
    for s in CHECKS:
        print(f"  {s:7d} " + " ".join(f"{R[k][s][0]:12.4f} ({R[k][s][1]:+.4f})" for k in R), flush=True)
    c = R['C many+prune'][TOTAL][0]
    print(f"  C many+prune  = {c:.4f} ({R['C many+prune'][TOTAL][1]:+.4f})", flush=True)
    print(f"\n  vs current pace (3.9846): prune {3.9846-c:+.4f} nats | few-larger {3.9846-3.9076:+.4f} nats", flush=True)
