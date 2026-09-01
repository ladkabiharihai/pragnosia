"""THE BENCHMARK WE NOW HAVE. A 21M transformer trained on 420M tokens (20 tok/param) of window2 reaches loss
3.1538 on a fixed eval. Our 202M model at 12.3B tokens reaches 4.1742 on the same eval -- 1.02 nats WORSE with 9.6x
the parameters and 29x the tokens. So the sequence mixer is the deficit, not the token count.

This harness trains variants of our mixer at MATCHED size and MATCHED tokens against that transformer, so every
change can be judged against a real target instead of against our own past.

WHAT WE ARE MISSING vs the linear-attention architectures that DO match transformers (Mamba-2, GLA, RWKV-6):
  - DATA-DEPENDENT GATING: each token computes its own forget rate. We use one global constant (decay=0.05).
    In the parallel form a per-token gate g_u becomes a pairwise factor exp(L_t - L_s) where L_t = sum_{u<=t} log g_u,
    which is <= 1 for t >= s, so it is numerically safe.
  - MULTI-HEAD: several independent retrieval channels per layer. We have one.
Both preserve linear cost and leave memory/growth/consolidation untouched -- they operate on the same state.

Usage:  python exp/bench.py [tokens]     (default 100M for fast iteration; use 420000000 to match the target)
"""
import sys, time, math
sys.path.insert(0, "/opt/code/membrae")
from util import setup_log
setup_log("bench")
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import core as C
from data_mix import DataMix
from tokenizers import Tokenizer

DEV = C.DEV
TOK = Tokenizer.from_file("/opt/code/membrae/data/bpe32k.json")
V = 32768; CH = 512; B = 16; D = 384; NL = 6
TOKENS = int(sys.argv[1]) if len(sys.argv) > 1 else 100_000_000
STEPS = TOKENS // (B * CH)

dm = DataMix(ctx=CH + 1, verbose=False, only='window2')
np.random.seed(7)
EV = [torch.tensor(dm.batch(4), device=DEV) for _ in range(16)]     # the SAME eval the transformer was scored on


# ---------------------------------------------------------------- transformer reference
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
    def __init__(s, d=D, nl=NL):
        super().__init__(); s.emb = nn.Embedding(V, d); s.pos = nn.Embedding(CH, d)
        s.bl = nn.ModuleList([Blk(d) for _ in range(nl)]); s.nf = nn.LayerNorm(d)
        s.head = nn.Linear(d, V, bias=False); s.head.weight = s.emb.weight
        nn.init.normal_(s.emb.weight, 0, 0.02); nn.init.normal_(s.pos.weight, 0, 0.02)
    def forward(s, X):
        x = s.emb(X) + s.pos(torch.arange(X.shape[1], device=X.device))[None]
        for b in s.bl: x = b(x)
        return s.head(s.nf(x))


# ---------------------------------------------------------------- our mixer, with switches
def taylor_h(x, iu):
    qd = (x.unsqueeze(-1) * x.unsqueeze(-2))[..., iu[0], iu[1]]
    return torch.cat([torch.ones(*x.shape[:-1], 1, device=x.device, dtype=x.dtype), x, qd], -1)


class Mixer(nn.Module):
    """our recurrent mixer. heads>1 = multi-head; gate=True = per-token data-dependent forgetting."""
    def __init__(s, D, dh, H, rs, gate=False, qknorm=False, mult=4):
        super().__init__(); s.H = H; s.dh = dh; s.rs = rs; s.gate = gate; s.qknorm = qknorm; s.dv = D // H
        s.Wk = nn.Linear(D, H * dh, bias=False); s.Wq = nn.Linear(D, H * dh, bias=False)
        s.Wv = nn.Linear(D, D, bias=False); s.Wo = nn.Linear(D, D)
        s.m1 = nn.Linear(D, mult * D); s.m2 = nn.Linear(mult * D, D)
        if gate:
            s.Wg = nn.Linear(D, H)                      # one forget rate per head per token
            nn.init.zeros_(s.Wg.weight); nn.init.constant_(s.Wg.bias, 4.0)   # start near g=1 (remember everything)
        s.register_buffer('iu', torch.triu_indices(dh, dh), persistent=False)

    def forward(s, a):
        Bs, T, _ = a.shape; H, dh = s.H, s.dh
        Hn, _ = C.rmsnorm(a)
        Hs = torch.cat([torch.zeros_like(Hn[:, :1]), Hn[:, :-1]], 1)     # key from the PREVIOUS token
        xk = s.Wk(Hs).view(Bs, T, H, dh); xq = s.Wq(Hn).view(Bs, T, H, dh)
        if s.qknorm:
            # BOUND THE FEATURE MAP. taylor() is QUADRATIC in its input, so A = qf.kf grows QUARTICALLY with the
            # projection magnitude. Measured over 3000 steps: |xq|max 2 -> 12 and |A|max 17 -> 1.3e4, still climbing,
            # and the long run went NaN. Normalising q and k to unit RMS bounds the features for good.
            xk = xk * torch.rsqrt(xk.pow(2).mean(-1, keepdim=True) + 1e-6)
            xq = xq * torch.rsqrt(xq.pow(2).mean(-1, keepdim=True) + 1e-6)
        kf = taylor_h(xk, s.iu)
        qf = taylor_h(xq, s.iu)
        v = s.Wv(Hn).view(Bs, T, H, s.dv)
        A = torch.einsum('bthf,bshf->bhts', qf, kf)
        if s.gate:
            # per-token forget rate -> pairwise factor exp(L_t - L_s), L_t = sum_{u<=t} log g_u.
            # The exponent is <= 0 only for t >= s. For t < s it is POSITIVE and .exp() overflows to inf; the
            # causal mask is applied AFTER, and inf * 0 = NaN. CLAMP at 0 -- those entries are masked away, so
            # nothing the model uses changes. This is what caused the NaN at step ~78, reproducibly across seeds.
            lg = F.logsigmoid(s.Wg(Hn)).transpose(1, 2)                  # [B,H,T], each <= 0
            L = lg.cumsum(-1)
            A = A * (L.unsqueeze(-1) - L.unsqueeze(-2)).clamp(max=0.0).exp()
        A = A.tril()
        den = A.sum(-1, keepdim=True) + 1e-5
        num = torch.einsum('bhts,bshd->bthd', A, v)
        r = (num / den.permute(0, 2, 1, 3)).reshape(Bs, T, -1)
        a1 = a + s.rs * (r @ s.Wo.weight.T + s.Wo.bias)
        H2, _ = C.rmsnorm(a1)
        return a1 + s.rs * (F.gelu(H2 @ s.m1.weight.T + s.m1.bias) @ s.m2.weight.T + s.m2.bias)


class Ours(nn.Module):
    def __init__(s, dh=24, H=1, gate=False, qknorm=False, d=D, nl=NL):
        super().__init__(); s.emb = nn.Embedding(V, d)
        s.layers = nn.ModuleList([Mixer(d, dh, H, nl ** -0.5, gate, qknorm) for _ in range(nl)])
        s.head = nn.Linear(d, V); s.head.weight = s.emb.weight
        nn.init.normal_(s.emb.weight, 0, 0.02)
    def forward(s, X):
        a = s.emb(X)
        for L in s.layers: a = L(a)
        h, _ = C.rmsnorm(a); return s.head(h)


# ---------------------------------------------------------------- harness
def evl(m):
    m.eval(); t = 0
    with torch.no_grad():
        for X in EV:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                t += F.cross_entropy(m(X[:, :CH]).reshape(-1, V).float(), X[:, 1:CH + 1].reshape(-1)).item()
    m.train(); return t / len(EV)


def train(m, tag):
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
    dt = time.time() - t0
    P = sum({id(p): p.numel() for p in m.parameters()}.values())
    return evl(m), P, STEPS * B * CH / dt


# already measured at this budget: transformer 3.4617 (756 ktok/s), ours-now 3.8747 (292 ktok/s)
# measured at this budget already: transformer 3.4617 (756 ktok/s), ours-now 3.8747 (292),
# multi-head-only 3.9149 (209) -- multi-head alone HURTS, so it is dropped from further arms.
REF = 3.4617
VARIANTS = [
    ("+ qk-norm only", lambda: Ours(dh=24, H=1, gate=False, qknorm=True)),
    ("+ gating + qk-norm", lambda: Ours(dh=24, H=1, gate=True, qknorm=True)),
    ("+ gating + qk-norm, dh=32", lambda: Ours(dh=32, H=1, gate=True, qknorm=True)),
]

if __name__ == "__main__":
    print(f"BENCHMARK vs the transformer. {TOKENS/1e6:.0f}M tokens each, D={D}, {NL} layers, "
          f"same data and eval\n", flush=True)
    print(f"  {'variant':>32s} {'params':>8s} {'loss':>8s} {'vs target':>10s} {'ktok/s':>8s}", flush=True)
    ref = REF
    print(f"  {'transformer (the target)':>32s} {23.4:7.1f}M {3.4617:8.4f} {0.0:+10.4f} {756:8.0f}", flush=True)
    print(f"  {'ours now: 1 head, no gate':>32s} {21.6:7.1f}M {3.8747:8.4f} {0.4130:+10.4f} {292:8.0f}", flush=True)
    print(f"  {'+ multi-head only (6 x dh=10)':>32s} {21.8:7.1f}M {3.9149:8.4f} {0.4532:+10.4f} {209:8.0f}", flush=True)
    for name, build in VARIANTS:
        torch.manual_seed(0); np.random.seed(0)
        m = build().to(DEV)
        torch.manual_seed(1); np.random.seed(1)
        l, P, tps = train(m, name)
        print(f"  {name:>32s} {P/1e6:7.1f}M {l:8.4f} {l-ref:+10.4f} {tps/1e3:8.0f}", flush=True)
        del m; torch.cuda.empty_cache()
    print(f"\n  GOAL: a variant with 'vs target' <= 0 at comparable speed. Anything above 0 is still behind.",
          flush=True)
