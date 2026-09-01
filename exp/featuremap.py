"""THE TAYLOR FEATURE MAP IS THE LAST UNTESTED CHOICE. Gating closed the gap from +0.413 to +0.159, but the gap
WIDENED with more tokens (+0.055 at 100M -> +0.159 at 250M), which points at a capacity ceiling rather than a
training shortfall. The one component we picked that no competitive linear-attention architecture uses is the
2nd-order Taylor map: taylor(x) = [1, x_i, {x_i x_j}_{i<=j}].

WHY IT MAY BE THE PROBLEM. The map produces FE = 1 + d + d(d+1)/2 features from d dimensions, but they are not
independent -- they are all polynomials of the same d numbers, so the RANK of the retrieval space is bounded by
what d values can express, not by FE. It also makes A = qf.kf a QUARTIC function of the projections, which is why
we needed qk-norm to stop it exploding. GLA, Mamba-2 and RWKV-6 all use simple ELEMENTWISE maps (SiLU, exp, or
identity) with a LARGER key dimension instead, and they match transformers.

Compare, at matched parameters and matched FE budget (so the state size is equal):
  taylor d=32   -> FE=561    (ours)
  identity d=561, silu d=561, elu+1 d=561   -- same state, but 561 INDEPENDENT dimensions
All with gating and qk-norm, which are already established wins.
"""
import sys, time, math
sys.path.insert(0, "/opt/code/membrae")
from util import setup_log
setup_log("featuremap")
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import core as C
from data_mix import DataMix

DEV = C.DEV
V = 32768; CH = 512; B = 16; D = 384; NL = 6
TOKENS = int(sys.argv[1]) if len(sys.argv) > 1 else 100_000_000
STEPS = TOKENS // (B * CH)

dm = DataMix(ctx=CH + 1, verbose=False, only='window2')
np.random.seed(7)
EV = [torch.tensor(dm.batch(4), device=DEV) for _ in range(16)]


class Mixer(nn.Module):
    """gated linear attention with a swappable feature map. `fmap` in {taylor, identity, silu, elu}."""
    def __init__(s, D, dk, rs, fmap, mult=4):
        super().__init__(); s.rs = rs; s.fmap = fmap; s.dk = dk
        if fmap == 'taylor':
            s.iu = torch.triu_indices(dk, dk, device=DEV)
            s.fe = 1 + dk + s.iu.shape[1]
        else:
            s.fe = dk
        s.Wk = nn.Linear(D, dk, bias=False); s.Wq = nn.Linear(D, dk, bias=False)
        s.Wv = nn.Linear(D, D, bias=False); s.Wo = nn.Linear(D, D)
        s.m1 = nn.Linear(D, mult * D); s.m2 = nn.Linear(mult * D, D)
        s.Wg = nn.Linear(D, 1); nn.init.zeros_(s.Wg.weight); nn.init.constant_(s.Wg.bias, 8.0)

    def phi(s, x):
        if s.fmap == 'taylor':
            qd = x[..., s.iu[0]] * x[..., s.iu[1]]
            return torch.cat([torch.ones(*x.shape[:-1], 1, device=x.device, dtype=x.dtype), x, qd], -1)
        if s.fmap == 'silu':     return F.silu(x)
        if s.fmap == 'elu':      return F.elu(x) + 1.0        # the classic linear-attention map (Katharopoulos)
        return x                                              # identity

    def forward(s, a):
        Bs, T, _ = a.shape
        Hn, _ = C.rmsnorm(a)
        Hs = torch.cat([torch.zeros_like(Hn[:, :1]), Hn[:, :-1]], 1)
        xk = s.Wk(Hs); xq = s.Wq(Hn)
        xk = xk * torch.rsqrt(xk.pow(2).mean(-1, keepdim=True) + 1e-6)
        xq = xq * torch.rsqrt(xq.pow(2).mean(-1, keepdim=True) + 1e-6)
        kf = s.phi(xk); qf = s.phi(xq)
        v = s.Wv(Hn)
        A = torch.einsum('btf,bsf->bts', qf, kf)
        lg = F.logsigmoid(s.Wg(Hn)).squeeze(-1); L = lg.cumsum(-1)
        A = A * (L.unsqueeze(-1) - L.unsqueeze(-2)).clamp(max=0.0).exp()
        A = A.tril()
        den = A.sum(-1, keepdim=True) + 1e-5
        r = torch.einsum('bts,bsd->btd', A, v) / den
        a1 = a + s.rs * (r @ s.Wo.weight.T + s.Wo.bias)
        H2, _ = C.rmsnorm(a1)
        return a1 + s.rs * (F.gelu(H2 @ s.m1.weight.T + s.m1.bias) @ s.m2.weight.T + s.m2.bias)


class M(nn.Module):
    def __init__(s, dk, fmap, d=D, nl=NL):
        super().__init__(); s.emb = nn.Embedding(V, d)
        s.layers = nn.ModuleList([Mixer(d, dk, nl ** -0.5, fmap) for _ in range(nl)])
        s.head = nn.Linear(d, V); s.head.weight = s.emb.weight
        nn.init.normal_(s.emb.weight, 0, 0.02)
    def forward(s, X):
        a = s.emb(X)
        for L in s.layers: a = L(a)
        h, _ = C.rmsnorm(a); return s.head(h)


def evl(m):
    m.eval(); t = 0
    with torch.no_grad():
        for X in EV:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                t += F.cross_entropy(m(X[:, :CH]).reshape(-1, V).float(), X[:, 1:CH + 1].reshape(-1)).item()
    m.train(); return t / len(EV)


def run(dk, fmap):
    torch.manual_seed(0); np.random.seed(0)
    m = M(dk, fmap).to(DEV)
    torch.manual_seed(1); np.random.seed(1)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3, betas=(0.9, 0.95))
    warm = 200; t0 = time.time()
    for i in range(STEPS):
        lr = 1e-3 * ((i + 1) / warm if i < warm else
                     0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (i - warm) / max(1, STEPS - warm))))
        for g in opt.param_groups: g['lr'] = lr
        X = torch.tensor(dm.batch(B), device=DEV)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            l = F.cross_entropy(m(X[:, :CH]).reshape(-1, V), X[:, 1:CH + 1].reshape(-1))
        opt.zero_grad(); l.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    out = evl(m); fe = m.layers[0].fe
    P = sum({id(p): p.numel() for p in m.parameters()}.values())
    tps = STEPS * B * CH / (time.time() - t0)
    del m, opt; torch.cuda.empty_cache()
    return out, fe, P, tps


if __name__ == "__main__":
    print(f"FEATURE MAP COMPARISON, all gated + qk-normed. {TOKENS/1e6:.0f}M tokens each.\n"
          f"Reference at this budget: transformer 3.4617, ours-with-taylor 3.5165\n", flush=True)
    print(f"  {'feature map':>22s} {'d_k':>5s} {'FE':>5s} {'params':>8s} {'loss':>8s} {'ktok/s':>8s}", flush=True)
    for tag, dk, fm in (("taylor d=32 (ours)", 32, 'taylor'),
                        ("identity d=561", 561, 'identity'),
                        ("silu d=561", 561, 'silu'),
                        ("elu+1 d=561", 561, 'elu'),
                        ("silu d=256", 256, 'silu')):
        l, fe, P, tps = run(dk, fm)
        print(f"  {tag:>22s} {dk:5d} {fe:5d} {P/1e6:7.1f}M {l:8.4f} {tps/1e3:8.0f}", flush=True)
