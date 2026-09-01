"""SURPASS THE TRANSFORMER -- on loss AND on generation. The benchmark got us from +0.413 to +0.055 nats, which is
close but still BEHIND, and it never tested what either model actually WRITES. Loss is not the deliverable.

Three things this settles:
  1. LOSS at a longer budget -- does the remaining +0.055 close, hold, or widen with more tokens?
  2. GENERATION -- side by side, same prompts, same decoding. This is the claim that matters.
  3. THE FACULTIES -- our memory gain must survive the new mixer, or the win is hollow.

Two changes since the benchmark, both verified: the boundary key is threaded (the chunked recurrence is now exact to
8.7e-16 relative, was 8.1e-02), and the gate is in core.Layer itself rather than a bench-only copy.
"""
import sys, time, math
sys.path.insert(0, "/opt/code/membrae")
from util import setup_log
setup_log("beat_it")
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import core as C
from data_mix import DataMix
from tokenizers import Tokenizer

DEV = C.DEV
TOK = Tokenizer.from_file("/opt/code/membrae/data/bpe32k.json")
V = 32768; CH = 512; B = 16; D = 384; NL = 6
TOKENS = int(sys.argv[1]) if len(sys.argv) > 1 else 250_000_000
STEPS = TOKENS // (B * CH)

dm = DataMix(ctx=CH + 1, verbose=False, only='window2')
np.random.seed(7)
EV = [torch.tensor(dm.batch(4), device=DEV) for _ in range(16)]


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


def ours():
    cfg = C.Config(D=D, DIN=32, DV=D, NL=NL, T=CH, res_scale=NL ** -0.5, tie_emb=True)
    cfg.gated = True; cfg.qknorm = True                # the winning configuration
    return C.Core(cfg).to(DEV)


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
        if (i + 1) % 4000 == 0:
            print(f"    {tag}: {i+1}/{STEPS}  loss {l.item():.3f}  "
                  f"{(i+1)*B*CH/(time.time()-t0)/1e3:.0f} ktok/s", flush=True)
    return evl(m)


@torch.no_grad()
def generate(m, ids, n, is_ours, temp=0.85):
    if is_ours:
        out, _ = m.generate(ids, n, temp=temp)
        return out
    seq = list(ids)
    for _ in range(n):
        lg = m(torch.tensor([seq[-CH:]], device=DEV))[0, -1].float()
        seq.append(int(torch.multinomial((lg / temp).softmax(-1), 1)))
    return seq[len(ids):]


def gstats(t):
    d1 = len(set(t)) / max(1, len(t))
    g = [tuple(t[i:i + 4]) for i in range(len(t) - 3)]
    return d1, 1 - len(set(g)) / max(1, len(g))


if __name__ == "__main__":
    print(f"SURPASS THE TRANSFORMER: {TOKENS/1e6:.0f}M tokens each, D={D}, {NL} layers\n", flush=True)
    res = {}
    for tag, build, is_ours in (("transformer", TLM, False), ("ours (gated+qknorm dh=32)", ours, True)):
        torch.manual_seed(0); np.random.seed(0)
        m = build().to(DEV) if not is_ours else build()
        torch.manual_seed(1); np.random.seed(1)
        l = train(m, tag)
        P = sum({id(p): p.numel() for p in m.parameters()}.values())
        print(f"\n  {tag}: {P/1e6:.1f}M params, FINAL LOSS {l:.4f} (ppl {math.exp(l):.1f})\n", flush=True)
        res[tag] = (l, m, is_ours)

    print("=" * 96, flush=True)
    print("GENERATION, SIDE BY SIDE -- same context, same decoding", flush=True)
    print("=" * 96, flush=True)
    np.random.seed(11); d2 = DataMix(ctx=CH + 1, verbose=False, only='window2')
    docs = [d2.batch(1)[0].tolist() for _ in range(2)]
    for di, doc in enumerate(docs):
        print(f"\n  [CONTEXT] ...{TOK.decode(doc[200:256])}\n", flush=True)
        for tag, (l, m, io) in res.items():
            m.eval(); torch.manual_seed(di)
            out = generate(m, doc[:256], 90, io)
            d1, r4 = gstats(out)
            print(f"  {tag:>26s} (d1 {d1:.2f}, rep {r4:.2f}): {TOK.decode(out)}\n", flush=True)
    lt = res["transformer"][0]; lo = res["ours (gated+qknorm dh=32)"][0]
    print("=" * 96, flush=True)
    print(f"  LOSS: transformer {lt:.4f}   ours {lo:.4f}   gap {lo-lt:+.4f} "
          f"{'-- WE SURPASS IT' if lo < lt else '-- still behind'}", flush=True)
