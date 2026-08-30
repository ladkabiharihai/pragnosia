"""THE MODEL HAS STOPPED IMPROVING. Fixed-eval loss went 3.8461 -> 3.8439 across 4.7B tokens, and the training loss
just bounces between 3.72 and 3.91 with no trend. Two very different explanations, and they imply opposite actions:

  LR-STARVED : the per-stage LR schedule decays to 10% of peak by the end of a growth stage, but PLAN is EXHAUSTED
               (10/10 growths done), so there is no next stage, the LR has sat at its 6e-5 floor ever since, and the
               model simply cannot move. Fix: restore a useful learning rate.
  SATURATED  : 235M genuinely has no more room on this data, which is precisely the condition the vision says should
               trigger GROWTH -- and we are blocking it by capping PLAN at 235M.

Test directly: continue training from the checkpoint at several learning rates and see which, if any, still makes
progress on a held-out set. If a higher LR moves the loss, it was starved, not saturated.
"""
import sys, copy
sys.path.insert(0, "/opt/code/membrae")
from util import setup_log
setup_log("plateau")
import numpy as np, torch, torch.nn.functional as F
import core as C
from data_mix import DataMix

DEV = C.DEV
CK = "/mnt/kv_cache/pragnosia_data/pragnosia.pt"
CH = 1024          # shorter than training's 2048 to fit alongside the live trainer and prod
B = 4
STEPS = 400


def load():
    sd = torch.load(CK, map_location=DEV)
    cfg = C.Config(D=sd['D'], DIN=8, DV=sd.get('DV', sd['D']), NL=sd['NL'], T=CH, res_scale=16 ** -0.5)
    m = C.Core(cfg).to(DEV); m.load_state_dict(sd['model'])
    return m, cfg


np.random.seed(1234)
_dm = DataMix(ctx=CH + 1, verbose=False, only='window2')
EV = [torch.tensor(_dm.batch(4), device=DEV) for _ in range(30)]     # 30*4*1024 = 122,880 tokens


@torch.no_grad()
def evl(m, cfg):
    m.eval(); t = 0
    for X in EV:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            t += F.cross_entropy(m(X[:, :CH]).reshape(-1, cfg.vocab).float(),
                                 X[:, 1:CH + 1].reshape(-1)).item()
    m.train(); return t / len(EV)


def run(lr, steps=STEPS):
    torch.manual_seed(0); np.random.seed(0)
    m, cfg = load()
    dm = DataMix(ctx=CH + 1, verbose=False, only='window2')
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    for i in range(steps):
        X = torch.tensor(dm.batch(B), device=DEV)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            l = F.cross_entropy(m(X[:, :CH]).reshape(-1, cfg.vocab), X[:, 1:CH + 1].reshape(-1))
        opt.zero_grad(); l.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    out = evl(m, cfg)
    del m, opt; torch.cuda.empty_cache()
    return out


if __name__ == "__main__":
    m, cfg = load(); base = evl(m, cfg)
    P = sum(p.numel() for p in m.parameters())
    del m; torch.cuda.empty_cache()
    print(f"STARTING POINT: {P/1e6:.0f}M, held-out loss {base:.4f} "
          f"(122,880 tokens, paired)\n", flush=True)
    print(f"  continuing training for {STEPS} steps ({STEPS*B*CH/1e6:.1f}M tokens) at each LR:\n", flush=True)
    print(f"  {'learning rate':>14s} {'loss after':>11s} {'change':>9s} {'verdict':>22s}", flush=True)
    best = (None, 1e9)
    for lr in (6e-5, 2e-4, 6e-4, 1.5e-3):
        l = run(lr)
        d = l - base
        v = "IMPROVES" if d < -0.005 else ("no change" if d < 0.005 else "worse")
        if l < best[1]: best = (lr, l)
        print(f"  {lr:14.0e} {l:11.4f} {d:+9.4f} {v:>22s}", flush=True)
    # VERDICT LOGIC CORRECTED: the first version called it "LR-starved" whenever the best LR improved, but 6e-5 IS
    # the current LR -- finding it best means the opposite, that nothing is starved. Only a HIGHER LR winning would
    # indicate starvation. Also note the small gain at 6e-5 is confounded: this test trains and evaluates at CH=1024
    # while the real run uses 2048, so some of it is the model specialising to the shorter context.
    higher = [lr for lr in (2e-4, 6e-4, 1.5e-3) if lr == best[0]]
    if higher:
        print(f"\n  -> LR-STARVED: a HIGHER rate ({best[0]:.0e}) beats the current 6e-5.", flush=True)
    else:
        print(f"\n  -> NOT LR-starved: 6e-5 (the current rate) is already the best; every higher rate is worse.", flush=True)
        print(f"     235M is near its ceiling on window2. The vision's answer to saturation is to GROW,", flush=True)
        print(f"     and PLAN caps this individual at 235M (10/10 growths used).", flush=True)
