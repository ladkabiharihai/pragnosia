"""THE STRUCTURAL FIX: stop growth from entrenching the incumbent dims.

DIAGNOSIS (measured): grow_width preserves the function by writing ZERO into every new dim and scaling everything
that WRITES the stream by c=sqrt(nD/D). That amplifies the OLD dims at every single growth -- cumulatively 1.63x
after our five width growths -- while each newcomer starts at zero magnitude. RMS-norm then divides by a norm the
incumbents dominate (measured 26.1 vs 7.3 RMS), so new dims fight an ever-steeper gradient. Self-entrenching.

MIRROR FIX: invert which side starts at zero.
  current : new dims WRITTEN with zero, READ with random  -> requires amplifying old dims by c
  mirror  : new dims WRITTEN with random at MATCHED RMS, READ with zero -> total RMS is unchanged by construction,
            so NO amplification of old dims is needed, and the function is still exactly preserved (nothing reads
            the new dims yet). The readers then start from zero and learn to pick up a signal that already exists.
"""
import sys, copy
sys.path.insert(0, "/opt/code/membrae")
from util import setup_log
setup_log("mirror_growth")
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import core as C, growth as G
from data_mix import DataMix

DEV = C.DEV; CH = 512; B = 8; PRE = 1200; D0 = 256; ADD = 128
CHECKS = [1500, 4000, 7000]
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


def grow_mirror(m, add):
    """New dims are WRITTEN at matched RMS and READ with zero. Old weights are NOT amplified."""
    cfg = m.cfg; D = cfg.D; nD = D + add; nDV = cfg.DV + add
    with torch.no_grad():
        # embedding: new columns matched to the existing per-dim RMS -> total RMS unchanged -> no c-scaling needed
        rms = m.emb.weight.float().pow(2).mean().sqrt().item()
        ne = nn.Embedding(cfg.vocab, nD).to(DEV)
        ne.weight[:, :D] = m.emb.weight                       # UNCHANGED (no amplification)
        ne.weight[:, D:].normal_(0, rms)
        m.emb = ne
        for L in m.layers:
            for nm in ('Wk', 'Wq'):                            # READ the stream -> new columns ZERO
                old = getattr(L, nm); outf = old.weight.shape[0]
                new = nn.Linear(nD, outf, bias=False).to(DEV)
                new.weight[:, :D] = old.weight; new.weight[:, D:] = 0.0
                setattr(L, nm, new)
            nv = nn.Linear(nD, nDV, bias=False).to(DEV)        # Wv READS stream, WRITES value space
            nv.weight[:cfg.DV, :D] = L.Wv.weight; nv.weight[:cfg.DV, D:] = 0.0
            nv.weight[cfg.DV:, :] = G._new_lin(nD, add, DEV)   # new value dims: real signal
            L.Wv = nv
            no = nn.Linear(nDV, nD).to(DEV)                    # Wo WRITES the stream
            no.weight[:D, :cfg.DV] = L.Wo.weight               # UNCHANGED
            no.weight[:D, cfg.DV:] = 0.0                       # new value dims not read yet
            no.weight[D:, :] = G._new_lin(nDV, add, DEV)       # new stream dims ARE written
            no.bias[:D] = L.Wo.bias; no.bias[D:] = 0.0
            L.Wo = no
            hid = L.m1.weight.shape[0]; mult = getattr(cfg, 'mlp_mult', 4)
            nhid = mult * nD; addh = nhid - hid
            n1 = nn.Linear(nD, nhid).to(DEV)                   # m1 READS the stream
            n1.weight[:hid, :D] = L.m1.weight; n1.weight[:hid, D:] = 0.0
            if addh > 0: n1.weight[hid:] = G._new_lin(nD, addh, DEV)
            n1.bias[:hid] = L.m1.bias; n1.bias[hid:] = 0.0
            n2 = nn.Linear(nhid, nD).to(DEV)                   # m2 WRITES the stream
            n2.weight[:D, :hid] = L.m2.weight; n2.weight[:D, hid:] = 0.0
            n2.weight[D:, :] = G._new_lin(nhid, add, DEV)
            n2.bias[:D] = L.m2.bias; n2.bias[D:] = 0.0
            L.m1, L.m2 = n1, n2
        nh = nn.Linear(nD, cfg.vocab).to(DEV)                  # head READS -> new columns ZERO
        nh.weight[:, :D] = m.head.weight; nh.weight[:, D:] = 0.0
        nh.bias.copy_(m.head.bias)
        m.head = nh
    cfg.D = nD; cfg.DV = nDV
    for L in m.layers: L.cfg = cfg; L.rs = cfg.res_scale
    return nD


def run(tag, growfn, sd0):
    torch.manual_seed(1); np.random.seed(1)
    cfg = C.Config(D=D0, DIN=8, DV=D0, NL=4, T=CH, res_scale=4 ** -0.5)
    m = C.Core(cfg).to(DEV); m.load_state_dict(sd0)
    growfn(m); cfg = m.cfg
    just = evl(m, cfg)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    print(f"    {tag:22s} loss immediately after grow: {just:.4f}", flush=True)
    out = {}
    for step in range(1, max(CHECKS) + 1):
        X = torch.tensor(dm.batch(B), device=DEV)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            l = F.cross_entropy(m(X[:, :CH]).reshape(-1, cfg.vocab), X[:, 1:CH + 1].reshape(-1))
        opt.zero_grad(); l.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        if step in CHECKS:
            lo = evl(m, cfg); d = contrib(m, cfg, D0, D0 + ADD); out[step] = (lo, d)
            print(f"    {tag:22s} @{step:5d}: loss {lo:.4f}  new-dim load {d:+.4f}", flush=True)
    # how badly are the incumbents amplified?
    with torch.no_grad():
        a = m.emb(EV[0][:, :CH])
        for L in m.layers: a, _ = L(a)
        rms = a.float().pow(2).mean((0, 1)).sqrt()
    print(f"    {tag:22s} stream RMS: seed dims {rms[:D0].mean():.2f}  new dims {rms[D0:].mean():.2f} "
          f"(ratio {rms[:D0].mean()/max(1e-9,rms[D0:].mean()):.2f}x)", flush=True)
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
    print(f"MIRROR GROWTH TEST: seed D={D0} loss {evl(seed,cfg0):.4f}, grow +{ADD}\n", flush=True)
    del seed; torch.cuda.empty_cache()
    R = {}
    R['current'] = run("A current", lambda m: G.grow_width(m, ADD), sd0)
    R['mirror'] = run("B mirror", lambda m: grow_mirror(m, ADD), sd0)
    print(f"\n  {'step':>6s} " + " ".join(f"{k:>24s}" for k in R), flush=True)
    for s in CHECKS:
        print(f"  {s:6d} " + " ".join(f"{R[k][s][0]:11.4f} ({R[k][s][1]:+.4f})" for k in R), flush=True)
    fin = {k: R[k][max(CHECKS)][0] for k in R}
    b = min(fin, key=fin.get)
    print(f"\n  BEST: {b} at {fin[b]:.4f}  ({fin['current']-fin[b]:+.4f} vs current)", flush=True)
