"""DOES THE GAP NARROW WITH CONTEXT? Every comparison so far ran at 512 tokens -- the regime where a transformer's
KV cache holds EVERY token and our fixed-size state is pure compression loss. That is the worst possible test for
us and the least relevant to the vision, which is about unbounded context.

If the gap narrows as context grows, the 512-token benchmark was the wrong yardstick all along. If it does not,
the deficit is real at every length and we need a different fix.

Same parameters, same tokens, same data, both compiled. Only the context length changes.
"""
import sys, time, math
sys.path.insert(0, "/opt/code/membrae")
from util import setup_log
setup_log("ctx_gap")
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import core as C
from data_mix import DataMix
torch._dynamo.config.recompile_limit = 64

DEV = C.DEV
V = 32768; D = 384; NL = 6
TOKENS = int(sys.argv[1]) if len(sys.argv) > 1 else 60_000_000


class Blk(nn.Module):
    def __init__(s, d, h=6):
        super().__init__(); s.h = h
        s.n1 = nn.LayerNorm(d); s.qkv = nn.Linear(d, 3 * d, bias=False); s.o = nn.Linear(d, d, bias=False)
        s.n2 = nn.LayerNorm(d); s.m1 = nn.Linear(d, 4 * d); s.m2 = nn.Linear(4 * d, d)
    def forward(s, x):
        Bs, T, Dm = x.shape
        q, k, v = s.qkv(s.n1(x)).chunk(3, -1)
        q, k, v = [t.view(Bs, T, s.h, Dm // s.h).transpose(1, 2) for t in (q, k, v)]
        x = x + s.o(F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(Bs, T, Dm))
        return x + s.m2(F.gelu(s.m1(s.n2(x))))


class TLM(nn.Module):
    def __init__(s, maxT):
        super().__init__(); s.emb = nn.Embedding(V, D); s.pos = nn.Embedding(maxT, D)
        s.bl = nn.ModuleList([Blk(D) for _ in range(NL)]); s.nf = nn.LayerNorm(D)
        s.head = nn.Linear(D, V, bias=False); s.head.weight = s.emb.weight
        nn.init.normal_(s.emb.weight, 0, 0.02); nn.init.normal_(s.pos.weight, 0, 0.02)
    def forward(s, X):
        x = s.emb(X) + s.pos(torch.arange(X.shape[1], device=X.device))[None]
        for b in s.bl: x = b(x)
        return s.head(s.nf(x))


def mk_ours(T):
    cfg = C.Config(D=D, DIN=24, DV=D, NL=NL, T=T, res_scale=NL ** -0.5, tie_emb=True)
    cfg.gated = True; cfg.qknorm = True
    return C.Core(cfg).to(DEV)


def run(CH, is_ours, tokens=TOKENS):
    B = max(1, 8192 // CH)                       # keep tokens/step constant across context lengths
    steps = tokens // (B * CH)
    dm = DataMix(ctx=CH + 1, verbose=False, only='window2')
    np.random.seed(7)
    EVL = [torch.tensor(dm.batch(2), device=DEV) for _ in range(12)]
    torch.manual_seed(0); np.random.seed(0)
    m = mk_ours(CH) if is_ours else TLM(CH).to(DEV)
    fwd = torch.compile(m)
    torch.manual_seed(1); np.random.seed(1)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3, betas=(0.9, 0.95))
    warm = min(200, steps // 10); t0 = time.time()
    for i in range(steps):
        lr = 1e-3 * ((i + 1) / warm if i < warm else
                     0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (i - warm) / max(1, steps - warm))))
        for g in opt.param_groups: g['lr'] = lr
        X = torch.tensor(dm.batch(B), device=DEV)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            l = F.cross_entropy(fwd(X[:, :CH]).reshape(-1, V), X[:, 1:CH + 1].reshape(-1))
        opt.zero_grad(); l.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    m.eval(); t = 0
    with torch.no_grad():
        for X in EVL:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                t += F.cross_entropy(m(X[:, :CH]).reshape(-1, V).float(), X[:, 1:CH + 1].reshape(-1)).item()
    out = t / len(EVL); tps = steps * B * CH / (time.time() - t0)
    del m, opt, fwd; torch.cuda.empty_cache()
    return out, tps


if __name__ == "__main__":
    print(f"DOES THE GAP NARROW WITH CONTEXT? {TOKENS/1e6:.0f}M tokens each, both compiled.\n"
          f"Each context length is a SEPARATE eval, so compare the GAP column, not losses across rows.\n", flush=True)
    print(f"  {'context':>8s} {'transformer':>12s} {'ours':>9s} {'GAP':>9s} {'tf ktok/s':>10s} {'our ktok/s':>11s}",
          flush=True)
    for CH in (512, 2048, 8192):
        lt, st = run(CH, False)
        lo, so = run(CH, True)
        print(f"  {CH:8d} {lt:12.4f} {lo:9.4f} {lo-lt:+9.4f} {st/1e3:10.0f} {so/1e3:11.0f}", flush=True)
    print(f"\n  a shrinking GAP means the 512-token benchmark was the wrong yardstick.", flush=True)
