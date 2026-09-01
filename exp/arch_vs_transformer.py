"""IS OUR ARCHITECTURE THE PROBLEM? The user's point: models trained to only ~20 tokens/param generate coherently,
and ours is at 60 tok/param and cannot. So "needs more tokens" is the wrong explanation and something structural is
wrong. This is the control that decides it.

Train a standard TRANSFORMER and OUR CORE at matched parameters, matched tokens, same corpus, same tokenizer, same
optimizer, to roughly 20 tokens/param -- then read what each one writes.

  transformer coherent, ours not  -> our architecture is the defect, and we should find which part
  both incoherent                 -> the expectation is wrong at this scale, not the architecture

Small scale on purpose: ~20M params x 20 tok/param = 400M tokens is minutes, not hours. Compute is not free.
"""
import sys, time, math
sys.path.insert(0, "/opt/code/membrae")
from util import setup_log
setup_log("arch_vs_transformer")
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import core as C
from data_mix import DataMix
from tokenizers import Tokenizer

DEV = C.DEV
TOK = Tokenizer.from_file("/opt/code/membrae/data/bpe32k.json")
V = 32768; CH = 512; B = 16
D = 384; NL = 6                      # both models get the same shape
TOKENS = 420_000_000                 # ~20 tok/param at ~21M params
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
    """standard transformer, TIED embeddings so the parameter budget matches ours"""
    def __init__(s, d=D, nl=NL):
        super().__init__(); s.emb = nn.Embedding(V, d); s.pos = nn.Embedding(CH, d)
        s.bl = nn.ModuleList([Blk(d) for _ in range(nl)]); s.nf = nn.LayerNorm(d)
        s.head = nn.Linear(d, V, bias=False); s.head.weight = s.emb.weight
        nn.init.normal_(s.emb.weight, 0, 0.02); nn.init.normal_(s.pos.weight, 0, 0.02)
    def forward(s, X):
        x = s.emb(X) + s.pos(torch.arange(X.shape[1], device=X.device))[None]
        for b in s.bl: x = b(x)
        return s.head(s.nf(x))


def evl(fwd, cfgv=V):
    t = 0
    with torch.no_grad():
        for X in EV:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                t += F.cross_entropy(fwd(X[:, :CH]).reshape(-1, cfgv).float(),
                                     X[:, 1:CH + 1].reshape(-1)).item()
    return t / len(EV)


def train(m, tag):
    opt = torch.optim.Adam(m.parameters(), lr=1e-3, betas=(0.9, 0.95))
    warm = 200
    t0 = time.time()
    for i in range(STEPS):
        lr = 1e-3 * ((i + 1) / warm if i < warm else
                     0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (i - warm) / max(1, STEPS - warm))))
        for g in opt.param_groups: g['lr'] = lr
        X = torch.tensor(dm.batch(B), device=DEV)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            l = F.cross_entropy(m(X[:, :CH]).reshape(-1, V), X[:, 1:CH + 1].reshape(-1))
        opt.zero_grad(); l.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        if (i + 1) % 2000 == 0:
            print(f"    {tag}: step {i+1}/{STEPS}  loss {l.item():.3f}  {(i+1)*B*CH/(time.time()-t0)/1e3:.0f} ktok/s",
                  flush=True)
    return m


@torch.no_grad()
def show(m, tag, is_ours):
    m.eval()
    np.random.seed(11); d2 = DataMix(ctx=CH + 1, verbose=False, only='window2')
    print(f"\n  --- {tag} writes ---", flush=True)
    for i in range(2):
        doc = d2.batch(1)[0].tolist()
        torch.manual_seed(i)
        if is_ours:
            out, _ = m.generate(doc[:256], 90, temp=0.85)
        else:
            ids = doc[:256][:]
            for _ in range(90):
                lg = m(torch.tensor([ids[-CH:]], device=DEV))[0, -1].float()
                ids.append(int(torch.multinomial((lg / 0.85).softmax(-1), 1)))
            out = ids[256:]
        d1 = len(set(out)) / len(out)
        g = [tuple(out[j:j + 4]) for j in range(len(out) - 3)]
        r4 = 1 - len(set(g)) / max(1, len(g))
        print(f"    [ctx] ...{TOK.decode(doc[210:256])}", flush=True)
        print(f"    [gen] {TOK.decode(out)}", flush=True)
        print(f"    (distinct-1 {d1:.2f}, 4gram-rep {r4:.2f})\n", flush=True)
    m.train()


if __name__ == "__main__":
    print(f"MATCHED CONTROL: same params, same {TOKENS/1e6:.0f}M tokens (~20 tok/param), same data\n", flush=True)
    torch.manual_seed(0); np.random.seed(0)
    t = TLM().to(DEV)
    Pt = sum({id(p): p.numel() for p in t.parameters()}.values())
    torch.manual_seed(0); np.random.seed(0)
    cfg = C.Config(D=D, DIN=24, DV=D, NL=NL, T=CH, res_scale=NL ** -0.5, tie_emb=True)
    o = C.Core(cfg).to(DEV)
    Po = sum(p.numel() for p in o.parameters())
    print(f"  transformer {Pt/1e6:.1f}M   ours {Po/1e6:.1f}M   "
          f"({TOKENS/Pt:.0f} and {TOKENS/Po:.0f} tokens/param)\n", flush=True)

    torch.manual_seed(1); np.random.seed(1); train(t, "transformer")
    lt = evl(t); print(f"\n  TRANSFORMER final loss {lt:.4f} (ppl {math.exp(lt):.1f})", flush=True)
    show(t, "TRANSFORMER", False)
    del t; torch.cuda.empty_cache()

    torch.manual_seed(1); np.random.seed(1); train(o, "ours")
    lo = evl(o); print(f"\n  OURS final loss {lo:.4f} (ppl {math.exp(lo):.1f})", flush=True)
    show(o, "OURS", True)
    print(f"\n  GAP: ours - transformer = {lo-lt:+.4f} nats", flush=True)
