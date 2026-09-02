"""DID THE DATA CHANGE HELP? A/B at 24M, identical in every respect except the corpus mix.

window2 is 10.5% synthetic kinship-QA and sequence-completion boilerplate; dclm is 0.0%. Our models reproduce that
boilerplate verbatim ("Question: Who is Mary? Answer: Who is Lily? Answer:"), so the hypothesis is that the data,
not the architecture and not the token budget, is what caps generation quality.

Both arms: same shape, same tokens, same seed, same decoding, both compiled. Only `lang_frac` vs `only='window2'`.

Evaluated on FIXED held-out sets so the numbers are comparable across arms -- a model evaluated on its own
training distribution would flatter itself. Reported on BOTH corpora for exactly that reason.

The metric that answers the question is TEMPLATE ECHO: how often the model spontaneously emits the synthetic
boilerplate when continuing ordinary prose. Loss cannot show this; it is the whole failure mode.
"""
import sys, time, math, re
sys.path.insert(0, "/opt/code/membrae")
from util import setup_log
setup_log("data_ab")
import numpy as np, torch, torch.nn.functional as F
import core as C
from data_mix import DataMix
from tokenizers import Tokenizer
import torch._dynamo as _d; _d.config.recompile_limit = 64
import torch._inductor.config as _ic; _ic.triton.prefer_nd_tiling = True; _ic.compile_threads = 8

DEV = C.DEV
TOK = Tokenizer.from_file("/opt/code/membrae/data/bpe32k.json")
V = 32768; CH = 2048; B = 8; D = 384; NL = 8; DIN = 16
TOK_PER_PARAM = float(sys.argv[1]) if len(sys.argv) > 1 else 20.0

TEMPLATE = [r"Question: Who is .{1,30}\? Answer:", r"Find the next (number|letter)",
            r"is the (mother|father) of ", r"The access code is"]


def held_out(only=None, lang_frac=None, n=10):
    d = DataMix(ctx=CH + 1, verbose=False, only=only, lang_frac=lang_frac)
    np.random.seed(777)
    return [torch.tensor(d.batch(2), device=DEV) for _ in range(n)]


def stats(t):
    d1 = len(set(t)) / max(1, len(t))
    g = [tuple(t[i:i + 4]) for i in range(len(t) - 3)]
    return d1, 1 - len(set(g)) / max(1, len(g))


def letters(x):
    c = [y for y in x if not y.isspace()]
    return sum(y.isalpha() for y in c) / max(1, len(c))


def evaluate(m, EV):
    m.eval(); tot = 0.0; nats = 0.0; nb = 0
    with torch.no_grad():
        for X in EV:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg = m(X[:, :CH]).float()
            y = X[:, 1:CH + 1]
            tot += F.cross_entropy(lg.reshape(-1, V), y.reshape(-1)).item()
            nats += F.cross_entropy(lg.reshape(-1, V), y.reshape(-1), reduction='sum').item()
            for b in range(X.shape[0]):
                nb += len(TOK.decode([int(z) for z in y[b]]).encode('utf-8'))
    m.train()
    return tot / len(EV), nats / (np.log(2) * nb)


def run(tag, only, lang_frac, probes, EVW, EVD):
    torch.manual_seed(0); np.random.seed(0)
    cfg = C.Config(D=D, DIN=DIN, DV=D, NL=NL, T=CH, res_scale=NL ** -0.5, tie_emb=True)
    cfg.gated = True; cfg.qknorm = True
    m = C.Core(cfg).to(DEV)
    P = sum(p.numel() for p in m.parameters())
    STEPS = max(1, int(TOK_PER_PARAM * P) // (B * CH))
    dm = DataMix(ctx=CH + 1, verbose=False, only=only, lang_frac=lang_frac)
    fwd = torch.compile(m); opt = torch.optim.Adam(m.parameters(), lr=1e-3, betas=(0.9, 0.95))
    warm = 200; t0 = time.time()
    torch.manual_seed(1); np.random.seed(1)
    for i in range(STEPS):
        lr = 1e-3 * ((i + 1) / warm if i < warm else
                     0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (i - warm) / max(1, STEPS - warm))))
        for g in opt.param_groups: g['lr'] = lr
        X = torch.tensor(dm.batch(B), device=DEV)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            l = F.cross_entropy(fwd(X[:, :CH]).reshape(-1, V), X[:, 1:CH + 1].reshape(-1))
        opt.zero_grad(); l.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    lw, bw = evaluate(m, EVW); ld, bd = evaluate(m, EVD)
    m.eval()
    echo = 0; R = []; D1 = []; L = []; samples = []
    for di, doc in enumerate(probes):
        torch.manual_seed(di)
        o, _ = m.generate(doc[:512], 200, temp=0.85)
        txt = TOK.decode(o)
        if any(re.search(p, txt) for p in TEMPLATE): echo += 1
        a, b = stats(o); D1.append(a); R.append(b); L.append(letters(txt))
        if di < 2: samples.append(txt)
    print(f"\n  {tag}", flush=True)
    print(f"    {P/1e6:.1f}M params, {STEPS*B*CH/1e6:.0f}M tokens ({TOK_PER_PARAM:.0f} tok/param), "
          f"{STEPS*B*CH/(time.time()-t0)/1e3:.0f} ktok/s", flush=True)
    print(f"    held-out window2 : loss {lw:.4f}  BPB {bw:.4f}", flush=True)
    print(f"    held-out dclm    : loss {ld:.4f}  BPB {bd:.4f}   <- neither arm over-represents this", flush=True)
    print(f"    TEMPLATE ECHO    : {100*echo/len(probes):.0f}% of generations emit synthetic boilerplate", flush=True)
    print(f"    generation       : 4gram-rep {np.mean(R):.2f}  distinct-1 {np.mean(D1):.2f}  "
          f"letters {np.mean(L):.2f}", flush=True)
    for i, s in enumerate(samples):
        print(f"      sample {i+1}: {s[:300]!r}", flush=True)
    del m, fwd, opt; torch.cuda.empty_cache()


if __name__ == "__main__":
    EVW = held_out(only='window2'); EVD = held_out(only='dclm')
    np.random.seed(11)
    pd = DataMix(ctx=CH + 1, verbose=False, only='dclm')      # probe from NEUTRAL prose, not from either arm's bulk
    probes = [pd.batch(1)[0].tolist() for _ in range(12)]
    print(f"A/B AT {D}x{NL}L DIN{DIN}, {TOK_PER_PARAM:.0f} tok/param, identical but for the corpus mix\n"
          f"Probes are dclm prose so neither arm is prompted from its own bulk.\n", flush=True)
    run("A) window2 ONLY (what we had)", 'window2', None, probes, EVW, EVD)
    run("B) LANGUAGE 70 / REASONING 30 (the change)", None, 0.70, probes, EVW, EVD)
