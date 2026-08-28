"""THE REAL SUSPECT: we grow TOO FAST. fix_growth2 showed the current zero-init DOES recruit new capacity, steadily
(-0.005 -> +0.095 and still climbing) -- the init is fine. But in the real run each stage gets only ~22-27 tok/param
before the NEXT growth adds more dims, so no generation ever finishes being recruited, and the ablation showed the
newest band worth 0.001 while the seed carries 8.13.

TEST: same final width (D=768) and same total training, reached three ways.
  FAST : 4 growths of +128, one every quarter of the budget   (mimics the real run's pace)
  SLOW : 2 growths of +256, one every half                    (half the pace)
  ONCE : 1 growth of +512 at the start                        (no repeated growth at all)
  NONE : no growth, stays D=256                               (control)
If SLOW/ONCE beat FAST, the growth PACE is the defect -- and the 22 tok/param gate is far too permissive.
"""
import sys, copy
sys.path.insert(0, "/opt/code/membrae")
from util import setup_log
setup_log("grow_pace")
import numpy as np, torch, torch.nn.functional as F
import core as C, growth as G
from data_mix import DataMix

DEV = C.DEV; CH = 512; B = 8; PRE = 1200; D0 = 256; TOTAL = 8000
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
    """nats lost when dims [lo,hi) are silenced = how much load they actually carry"""
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


def run(tag, schedule, sd0):
    """schedule: list of (step_at_which_to_grow, dims_to_add)"""
    torch.manual_seed(1); np.random.seed(1)
    cfg = C.Config(D=D0, DIN=8, DV=D0, NL=4, T=CH, res_scale=4 ** -0.5)
    m = C.Core(cfg).to(DEV); m.load_state_dict(sd0)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3); sched = dict(schedule)
    for step in range(1, TOTAL + 1):
        if step in sched:
            G.grow_width(m, sched[step]); cfg = m.cfg
            opt = torch.optim.Adam(m.parameters(), lr=1e-3)   # fresh optimizer, as the real run does
        X = torch.tensor(dm.batch(B), device=DEV)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            l = F.cross_entropy(m(X[:, :CH]).reshape(-1, cfg.vocab), X[:, 1:CH + 1].reshape(-1))
        opt.zero_grad(); l.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    fin = evl(m, cfg)
    grown = contrib(m, cfg, D0, cfg.D) if cfg.D > D0 else 0.0
    seed = contrib(m, cfg, 0, D0)
    P = sum(p.numel() for p in m.parameters())
    print(f"  {tag:26s} D={cfg.D:4d} {P/1e6:5.1f}M  loss {fin:.4f}   "
          f"grown dims {grown:+.4f}  seed dims {seed:+.4f}", flush=True)
    del m; torch.cuda.empty_cache()
    return fin, grown


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
    print(f"IS THE GROWTH PACE THE DEFECT? seed D={D0} loss {evl(seed,cfg0):.4f}; "
          f"FAST/SLOW/ONCE all end at D=768 after {TOTAL} steps\n", flush=True)
    del seed; torch.cuda.empty_cache()
    q = TOTAL // 4
    run("FAST  4x+128 (real pace)", [(1, 128), (q, 128), (2 * q, 128), (3 * q, 128)], sd0)
    run("SLOW  2x+256", [(1, 256), (2 * q, 256)], sd0)
    run("ONCE  1x+512", [(1, 512)], sd0)
    run("NONE  no growth (D=256)", [], sd0)
