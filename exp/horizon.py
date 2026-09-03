"""DOES CARRIED-STATE TRAINING BUY THE RETENTION HORIZON? The experiment I should have run before removing it.

WHAT I DID WRONG. I ablated carried-state training, measured that it costs +0.126 BPB, and cut it from the base
run. I never measured what it BUYS. BPB is exactly the metric this project has repeatedly shown does not capture
what these models can actually do, and I used it to make an architectural decision.

WHY IT MATTERS NOW. The live 125M dies at ~2000 tokens on BOTH the needle and long generation. It trains on
independent 2048-token windows with no state carried, so it has never once had a reason to retain anything past
2048. The learned forget rate agrees: mean gate half-life 2941 tokens, and the shallow layers sit at ~1500.
The horizon may simply be the training context, in which case the fix is training, not architecture.

    A  WINDOWS  -- independent 2048-token windows, no carried state   (what the base run does)
    B  CARRIED  -- 16384-token documents in 2048 chunks, state threaded, gradients detached between

Same parameters, same total tokens, same data, same seed. JUDGED ON THE NEEDLE HORIZON, with BPB reported
alongside so the trade is visible rather than assumed.
"""
import sys, time, math
sys.path.insert(0, "/opt/code/membrae")
from util import setup_log
setup_log("horizon")
import numpy as np, torch, torch.nn.functional as F
import core as C
from data_mix import DataMix
from tokenizers import Tokenizer
import torch._dynamo as _d; _d.config.recompile_limit = 64
import torch._inductor.config as _ic; _ic.triton.prefer_nd_tiling = True; _ic.compile_threads = 6

DEV = C.DEV
TOK = Tokenizer.from_file("/opt/code/membrae/data/bpe32k.json")
V = 32768; CH = 2048; D = 384; NL = 8; DIN = 16
DOC = 16384; NCH = DOC // CH
TPP = float(sys.argv[1]) if len(sys.argv) > 1 else 14.0

dmw = DataMix(ctx=CH + 1, verbose=False, lang_frac=0.70)
dml = DataMix(ctx=DOC + 1, verbose=False, lang_frac=0.70)
np.random.seed(777)
EV = [torch.tensor(dmw.batch(2), device=DEV) for _ in range(8)]

NAMES = ["Aurelia", "Kestrel", "Vandermeer", "Okonkwo", "Thackeray", "Zoravar", "Lindqvist", "Mbeki"]
np.random.seed(3)
POOL = []
while len(POOL) < 40000: POOL.extend(dmw.batch(1)[0].tolist())


def bpb(m, cfg):
    m.eval(); nats = 0.0; nb = 0
    with torch.no_grad():
        for X in EV:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg = m(X[:, :CH]).float()
            y = X[:, 1:CH + 1]
            nats += F.cross_entropy(lg.reshape(-1, V), y.reshape(-1), reduction='sum').item()
            for b in range(X.shape[0]):
                nb += len(TOK.decode([int(z) for z in y[b]]).encode('utf-8'))
    m.train(); return nats / (np.log(2) * nb)


@torch.no_grad()
def needle(m, cfg, dists=(0, 200, 2000, 8000, 30000), n=12):
    """paired lift: same token with the fact present vs a different fact, so frequency priors cancel"""
    m.eval()
    def zs():
        return [(torch.zeros(1, cfg.FE, cfg.DV, device=DEV), torch.zeros(1, cfg.FE, device=DEV))
                for _ in range(cfg.NL)]
    def score(ids, cue, tok):
        full = ids + cue; st = zs(); pv = None
        for i in range(0, len(full) - 1, CH):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, st, pv = m(torch.tensor([full[i:i + CH]], device=DEV), states=st,
                              return_states=True, prev=pv, return_prev=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg, _, _ = m.step(torch.tensor([[full[-1]]], device=DEV), st, pv)
        return float(lg[0].float().log_softmax(-1)[tok])
    out = []
    for Dd in dists:
        lifts = []
        for t in range(n):
            tn = NAMES[t % len(NAMES)]; other = NAMES[(t + 3) % len(NAMES)]
            cue = TOK.encode(" The vault keeper is").ids; tok = TOK.encode(f" {tn}").ids[0]
            rng = np.random.RandomState(500 + t); s0 = rng.randint(0, len(POOL) - Dd - 64)
            fill = POOL[s0:s0 + Dd]
            a = score(TOK.encode(f" The vault keeper is {tn}. ").ids + fill, cue, tok)
            b = score(TOK.encode(f" The vault keeper is {other}. ").ids + fill, cue, tok)
            lifts.append(a - b)
        out.append(float(np.mean(lifts)))
    m.train(); return out


def run(tag, carried):
    torch.manual_seed(0); np.random.seed(0)
    cfg = C.Config(D=D, DIN=DIN, DV=D, NL=NL, T=CH, res_scale=NL ** -0.5, tie_emb=True)
    cfg.gated = True; cfg.qknorm = True
    m = C.Core(cfg).to(DEV)
    P = sum(p.numel() for p in m.parameters())
    per = 8 * DOC if carried else 16 * CH          # tokens per optimizer step, matched as closely as batch allows
    STEPS = max(1, int(TPP * P) // per)
    fwd = torch.compile(m); opt = torch.optim.Adam(m.parameters(), lr=1e-3, betas=(0.9, 0.95))
    warm = max(20, STEPS // 20); t0 = time.time()
    torch.manual_seed(1); np.random.seed(1)
    for i in range(STEPS):
        lr = 1e-3 * ((i + 1) / warm if i < warm else
                     0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (i - warm) / max(1, STEPS - warm))))
        for g in opt.param_groups: g['lr'] = lr
        opt.zero_grad(set_to_none=True)
        if carried:
            Xd = torch.tensor(dml.batch(8), device=DEV)
            st = [(torch.zeros(8, cfg.FE, cfg.DV, device=DEV), torch.zeros(8, cfg.FE, device=DEV))
                  for _ in range(cfg.NL)]
            pv = None
            for ci in range(NCH):
                xi = Xd[:, ci * CH:(ci + 1) * CH]; yi = Xd[:, ci * CH + 1:(ci + 1) * CH + 1]
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    lg, st, pv = fwd(xi, states=st, return_states=True, prev=pv, return_prev=True)
                    l = F.cross_entropy(lg.reshape(-1, V), yi.reshape(-1)) / NCH
                st = [(a.detach(), b.detach()) for a, b in st]; pv = [q.detach() for q in pv]
                l.backward()
        else:
            X = torch.tensor(dmw.batch(16), device=DEV)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                l = F.cross_entropy(fwd(X[:, :CH]).reshape(-1, V), X[:, 1:CH + 1].reshape(-1))
            l.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    bb = bpb(m, cfg); nd = needle(m, cfg)
    with torch.no_grad():
        hl = float(np.mean([np.log(0.5) / np.log(min(0.9999999, torch.sigmoid(L.Wg.bias).item()))
                            for L in m.layers]))
    print(f"\n  {tag}", flush=True)
    print(f"    {P/1e6:.1f}M, {STEPS*per/1e6:.0f}M tokens, BPB {bb:.4f}, gate half-life {hl:.0f} tok "
          f"({(time.time()-t0)/60:.1f} min)", flush=True)
    print(f"    needle lift  " + "  ".join(f"d={d}:{v:+.3f}" for d, v in
                                           zip((0, 200, 2000, 8000, 30000), nd)), flush=True)
    del m, fwd, opt; torch.cuda.empty_cache()
    return bb, nd


if __name__ == "__main__":
    print(f"DOES CARRIED-STATE TRAINING BUY THE HORIZON? {D}x{NL}L DIN{DIN}, {TPP:.0f} tok/param\n"
          f"  the live 125M dies at ~2000 tokens and trains on 2048-token windows. Coincidence, or cause?\n",
          flush=True)
    a = run("A) WINDOWS -- 2048, no carried state (the base run)", False)
    b = run("B) CARRIED -- 16384-token documents, state threaded", True)
    print(f"\n  BPB   windows {a[0]:.4f}  |  carried {b[0]:.4f}   (carried was measured to cost ~0.126)", flush=True)
    print(f"  If B holds the needle further out, the horizon is a TRAINING choice, not an architectural limit --",
          flush=True)
    print(f"  and I cut the wrong thing from the base run by judging it on BPB alone.", flush=True)
