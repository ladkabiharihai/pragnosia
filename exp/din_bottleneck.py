"""THE RETRIEVAL BOTTLENECK. The user's point: 235M parameters should be enough for coherence -- GPT-2 small is
124M and coherent -- so if ours is not, something is wrong beyond token count.

Found it: DIN has been 8 since the infant, and grow_width deliberately holds it fixed ("keys/queries: output dim is
DIN (fixed)"). Every key/query association in the whole model passes through a 2nd-order Taylor map of an
8-dimensional projection = 45 features. The model grew 2.7x in width (384 -> 1024) and its RETRIEVAL SPACE NEVER
GREW. A transformer at D=1024 has 1024 dimensions of attention; we have 45.

DIN costs almost no PARAMETERS (Wk and Wq are only D x DIN) -- it costs state size and attention compute. So if
raising it improves loss substantially, we have been starving the model of its central mechanism for free.

Measured at matched D/NL/steps so only the retrieval width differs.
"""
import sys, time
sys.path.insert(0, "/opt/code/membrae")
from util import setup_log
setup_log("din_bottleneck")
import numpy as np, torch, torch.nn.functional as F
import core as C
from data_mix import DataMix

DEV = C.DEV; CH = 512; B = 8; D = 384; NL = 6; STEPS = 3000
dm = DataMix(ctx=CH + 1, verbose=False, only='window2')
np.random.seed(7)
EV = [torch.tensor(dm.batch(4), device=DEV) for _ in range(12)]


def evl(m, cfg):
    m.eval(); t = 0
    with torch.no_grad():
        for X in EV:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                t += F.cross_entropy(m(X[:, :CH]).reshape(-1, cfg.vocab).float(),
                                     X[:, 1:CH + 1].reshape(-1)).item()
    m.train(); return t / len(EV)


def run(DIN, steps=STEPS):
    torch.manual_seed(0); np.random.seed(0)
    cfg = C.Config(D=D, DIN=DIN, DV=D, NL=NL, T=CH, res_scale=NL ** -0.5)
    m = C.Core(cfg).to(DEV)
    P = sum(p.numel() for p in m.parameters())
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    t0 = time.time()
    for i in range(steps):
        X = torch.tensor(dm.batch(B), device=DEV)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            l = F.cross_entropy(m(X[:, :CH]).reshape(-1, cfg.vocab), X[:, 1:CH + 1].reshape(-1))
        opt.zero_grad(); l.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    dt = time.time() - t0
    out = evl(m, cfg)
    tps = steps * B * CH / dt
    del m, opt; torch.cuda.empty_cache()
    return out, P, cfg.FE, tps


if __name__ == "__main__":
    print(f"DOES THE RETRIEVAL WIDTH MATTER? D={D}, {NL} layers, {STEPS} steps, only DIN varies\n", flush=True)
    print(f"  {'DIN':>4s} {'FE':>5s} {'params':>8s} {'loss':>8s} {'vs DIN=8':>9s} "
          f"{'state/layer':>12s} {'ktok/s':>8s}", flush=True)
    base = None
    for DIN in (8, 12, 16, 24):
        l, P, FE, tps = run(DIN)
        if base is None: base = l
        st = FE * D * 4 / 2 ** 20
        print(f"  {DIN:4d} {FE:5d} {P/1e6:7.1f}M {l:8.4f} {l-base:+9.4f} {st:11.2f}M {tps/1e3:8.0f}", flush=True)
    print(f"\n  DIN adds almost no parameters (Wk,Wq are D x DIN) -- it buys retrieval capacity for free in weights.", flush=True)
    print(f"  If loss drops sharply with DIN, the model has been starved of its central mechanism since birth.", flush=True)
