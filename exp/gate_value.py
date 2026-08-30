"""WHAT SHOULD THE TOK/PARAM GATE BE? Everything so far says PACE is the lever: fewer/later growths beat more/earlier
(FAST 4.3483 > SLOW 4.3178 > ONCE 4.2732 at identical final size), and three different initialisation fixes all
failed to beat the current one. Our gate is 22 tok/param -- near Chinchilla-optimal (20) -- but Chinchilla is the
COMPUTE-optimal point for a fixed budget, not the point where a size has been EXPLOITED. Models that are actually
good at small size are overtrained to hundreds of tok/param.

TEST: grow ONCE, then keep training, and watch (a) how much load the new dims carry and (b) whether the loss
advantage over never-growing has materialised, as a function of tok/param spent at the new size.
The gate should be set where the new capacity has actually started paying for itself.
"""
import sys, copy
sys.path.insert(0, "/opt/code/membrae")
from util import setup_log
setup_log("gate_value")
import numpy as np, torch, torch.nn.functional as F
import core as C, growth as G
from data_mix import DataMix

DEV = C.DEV; CH = 512; B = 8; PRE = 1500; D0 = 256; ADD = 256
TOTAL = 24000
CHECKS = [1000, 2000, 4000, 8000, 12000, 16000, 20000, 24000]
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


def train_to(m, cfg, opt, frm, to):
    for step in range(frm, to):
        X = torch.tensor(dm.batch(B), device=DEV)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            l = F.cross_entropy(m(X[:, :CH]).reshape(-1, cfg.vocab), X[:, 1:CH + 1].reshape(-1))
        opt.zero_grad(); l.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()


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
    print(f"WHAT SHOULD THE GATE BE? seed D={D0} loss {evl(seed,cfg0):.4f}\n", flush=True)
    del seed; torch.cuda.empty_cache()

    # ARM 1: grow +256 immediately, then train and watch recruitment vs tok/param
    torch.manual_seed(1); np.random.seed(1)
    cfgA = C.Config(D=D0, DIN=8, DV=D0, NL=4, T=CH, res_scale=4 ** -0.5)
    A = C.Core(cfgA).to(DEV); A.load_state_dict(sd0); G.grow_width(A, ADD); cfgA = A.cfg
    PA = sum(p.numel() for p in A.parameters())
    optA = torch.optim.Adam(A.parameters(), lr=1e-3)
    # ARM 2: control that never grows, trained identically
    torch.manual_seed(1); np.random.seed(1)
    cfgB = C.Config(D=D0, DIN=8, DV=D0, NL=4, T=CH, res_scale=4 ** -0.5)
    Bm = C.Core(cfgB).to(DEV); Bm.load_state_dict(sd0)
    PB = sum(p.numel() for p in Bm.parameters())
    optB = torch.optim.Adam(Bm.parameters(), lr=1e-3)
    print(f"  grown model {PA/1e6:.1f}M   never-grown control {PB/1e6:.1f}M\n", flush=True)
    print(f"  {'tok/param':>10s} {'tokens':>9s} {'grown loss':>11s} {'control loss':>13s} "
          f"{'grown wins by':>14s} {'new-dim load':>13s}", flush=True)
    prev = 0
    for ck in CHECKS:
        train_to(A, cfgA, optA, prev, ck)
        train_to(Bm, cfgB, optB, prev, ck)
        prev = ck
        tok = ck * B * CH
        la = evl(A, cfgA); lb = evl(Bm, cfgB); d = contrib(A, cfgA, D0, D0 + ADD)
        print(f"  {tok/PA:10.1f} {tok/1e6:8.1f}M {la:11.4f} {lb:13.4f} {lb-la:+14.4f} {d:+13.4f}", flush=True)
    print(f"\n  'grown wins by' > 0 means the extra {(PA-PB)/1e6:.1f}M params are finally paying for themselves.", flush=True)
    print(f"  The gate should be set where that crossover happens.", flush=True)
