"""FIND THE TRAINER'S 0.49 BPB HANDICAP. This is the gate on everything else.

Same architecture, same corpus mix, same eval set:
    plain loop, 24.5M @ 20 tok/param   -> BPB 1.2277
    TRAINER,    25.0M @ 25.7 tok/param -> BPB 1.7175      (more tokens, 0.49 WORSE)

That handicap is worth more than any amount of extra scale: applied to the ladder it would take the 159M from
1.3320 to roughly 0.85, which is better than GPT-2's 0.9335. Nothing else we could spend an hour on comes close.

Four things the trainer does that the plain loop does not. Each is isolated here, one at a time, from the SAME
plain baseline, so the culprit is identified rather than guessed:
  A) baseline          -- the plain loop (known good)
  B) + recall batches  -- 15% of steps are answer-masked synthetic QA on short sequences
  C) + carried state   -- train on 32768-token documents in 16 chunks, state threaded, gradients detached between
  D) + both            -- to catch an interaction that neither shows alone
LR schedule and batch size are held identical across all four, so they cannot confound the result.
"""
import sys, time, math
sys.path.insert(0, "/opt/code/membrae")
from util import setup_log
setup_log("trainer_ablate")
import numpy as np, torch, torch.nn.functional as F
import core as C
from data_mix import DataMix
from tokenizers import Tokenizer
import torch._dynamo as _d; _d.config.recompile_limit = 64
import torch._inductor.config as _ic; _ic.triton.prefer_nd_tiling = True; _ic.compile_threads = 8

DEV = C.DEV
TOK = Tokenizer.from_file("/opt/code/membrae/data/bpe32k.json")
V = 32768; CH = 2048; B = 8; D = 384; NL = 8; DIN = 16
TPP = 20.0
DOC = 32768; NCH = DOC // CH

dmix = DataMix(ctx=CH + 1, verbose=False, lang_frac=0.70)
dlong = DataMix(ctx=DOC + 1, verbose=False, lang_frac=0.70)
np.random.seed(777)
EV = [torch.tensor(dmix.batch(2), device=DEV) for _ in range(10)]

RD = np.memmap("/mnt/kv_cache/pragnosia_data/recall_32k.bin", dtype=np.uint16, mode="r")


def recall_batch(bs, L):
    """answer-masked QA, as the trainer builds it"""
    X = np.empty((bs, L + 1), dtype=np.int64)
    for b in range(bs):
        i = np.random.randint(0, len(RD) - L - 2)
        X[b] = RD[i:i + L + 1]
    x = torch.tensor(X, device=DEV)
    return x[:, :L], x[:, 1:L + 1]


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
    m.train()
    return nats / (np.log(2) * nb)


def run(tag, recall_frac, carry):
    torch.manual_seed(0); np.random.seed(0)
    cfg = C.Config(D=D, DIN=DIN, DV=D, NL=NL, T=CH, res_scale=NL ** -0.5, tie_emb=True)
    cfg.gated = True; cfg.qknorm = True
    m = C.Core(cfg).to(DEV)
    P = sum(p.numel() for p in m.parameters())
    TOKENS = int(TPP * P)
    per = B * DOC if carry else B * CH                 # tokens consumed per optimizer step
    STEPS = max(1, TOKENS // per)
    fwd = torch.compile(m); opt = torch.optim.Adam(m.parameters(), lr=1e-3, betas=(0.9, 0.95))
    warm = max(10, STEPS // 100); t0 = time.time()
    torch.manual_seed(1); np.random.seed(1)

    def zs(b):
        return [(torch.zeros(b, cfg.FE, cfg.DV, device=DEV), torch.zeros(b, cfg.FE, device=DEV))
                for _ in range(cfg.NL)]

    for i in range(STEPS):
        lr = 1e-3 * ((i + 1) / warm if i < warm else
                     0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (i - warm) / max(1, STEPS - warm))))
        for g in opt.param_groups: g['lr'] = lr
        if recall_frac > 0 and np.random.rand() < recall_frac:
            X, Y = recall_batch(B, CH)
            opt.zero_grad()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                l = F.cross_entropy(fwd(X, states=zs(B)).reshape(-1, V), Y.reshape(-1))
            l.backward()
        elif carry:
            Xd = torch.tensor(dlong.batch(B), device=DEV); st = zs(B); pv = None; opt.zero_grad()
            for ci in range(NCH):
                xi = Xd[:, ci * CH:(ci + 1) * CH]; yi = Xd[:, ci * CH + 1:(ci + 1) * CH + 1]
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    lg, st, pv = fwd(xi, states=st, return_states=True, prev=pv, return_prev=True)
                    l = F.cross_entropy(lg.reshape(-1, V), yi.reshape(-1)) / NCH
                st = [(a.detach(), b.detach()) for a, b in st]; pv = [q.detach() for q in pv]
                l.backward()
        else:
            X = torch.tensor(dmix.batch(B), device=DEV)
            opt.zero_grad()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                l = F.cross_entropy(fwd(X[:, :CH]).reshape(-1, V), X[:, 1:CH + 1].reshape(-1))
            l.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    b = bpb(m, cfg)
    print(f"  {tag:>36s} {P/1e6:6.1f}M {STEPS*per/1e6:7.0f}M tok  BPB {b:7.4f}  "
          f"({(time.time()-t0)/60:4.1f} min)", flush=True)
    del m, fwd, opt; torch.cuda.empty_cache()
    return b


if __name__ == "__main__":
    print(f"WHICH TRAINER FEATURE COSTS 0.49 BPB? {D}x{NL}L DIN{DIN} at {TPP:.0f} tok/param\n"
          f"  plain loop reference 1.2277   |   full trainer 1.7175\n", flush=True)
    a = run("A) baseline (plain loop)", 0.0, False)
    b = run("B) + 15% recall batches", 0.15, False)
    c = run("C) + carried state, 32k docs", 0.0, True)
    d = run("D) + both (the trainer)", 0.15, True)
    print(f"\n  {'feature':>36s} {'BPB':>8s} {'cost vs baseline':>18s}", flush=True)
    for tag, v in (("baseline", a), ("recall batches", b), ("carried state", c), ("both", d)):
        print(f"  {tag:>36s} {v:8.4f} {v-a:+18.4f}", flush=True)
    print(f"\n  the feature with the large positive cost is the handicap. Remove it and retrain.", flush=True)
