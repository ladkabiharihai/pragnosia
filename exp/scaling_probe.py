"""IS THE 4-HOUR RUN WORTH IT? Validate by EXTRAPOLATION before spending the compute.

The open question: does our CURRENT architecture reach GPT-2-class coherence (BPB 0.9335 at 124M / 73 tok/param)
if we simply train it far past 20 Chinchilla? Training a 100M model to 70 tok/param to find out costs ~4 hours.

This answers it for ~45 minutes by measuring the TRAJECTORY instead of a single point. Two model sizes, each
evaluated at several token budgets, gives a surface over (params, tokens) that can be fitted and extrapolated to
100M / 73 tok/param. A confident prediction below ~1.0 BPB justifies the run; a plateau says do not bother.

Two metrics, because loss alone has already proved a poor guide to whether text MEANS anything:
  BPB          -- tokenizer-independent, directly comparable to GPT-2's 0.9335 on the same text
  CONTEXT REUSE -- of the rare (content) words the model generates, what fraction appeared in the prompt? GPT-2
                  picked up "Buck" and "Spitz" from a Call of the Wild passage; ours did not. This measures
                  exactly that, and it is what "means something in context" looks like numerically.
Real text scores both, as the reference every number is read against.
"""
import sys, time, math, re
sys.path.insert(0, "/opt/code/membrae")
from util import setup_log
setup_log("scaling_probe")
import numpy as np, torch, torch.nn.functional as F
import core as C
from data_mix import DataMix
from tokenizers import Tokenizer
import torch._dynamo as _d; _d.config.recompile_limit = 64
import torch._inductor.config as _ic; _ic.triton.prefer_nd_tiling = True; _ic.compile_threads = 8

DEV = C.DEV
TOK = Tokenizer.from_file("/opt/code/membrae/data/bpe32k.json")
V = 32768; CH = 2048; B = 8; LANG = 0.70
SIZES = [(256, 6, 12), (384, 8, 16)]          # (D, NL, DIN) -> roughly 12M and 24M
BUDGETS = [10, 20, 40]                         # tok/param checkpoints within each run

dmix = DataMix(ctx=CH + 1, verbose=False, lang_frac=LANG)
np.random.seed(777)
EV = [torch.tensor(dmix.batch(2), device=DEV) for _ in range(10)]
np.random.seed(11)
pd = DataMix(ctx=CH + 1, verbose=False, only='dclm')
PROBES = [pd.batch(1)[0].tolist() for _ in range(10)]

STOP = set("the a an and or but of to in for on at by with from as is are was were be been being this that these "
           "those it its he she they we you i not no so if then than which who whom whose what when where how "
           "have has had do does did will would can could should may might must there here s t re ve ll d m".split())


def content_words(txt):
    return {w for w in re.findall(r"[a-z']{4,}", txt.lower()) if w not in STOP}


def context_reuse(prompt_txt, gen_txt):
    """of the content words GENERATED, what share also appear in the prompt? real text sets the reference"""
    p = content_words(prompt_txt); g = content_words(gen_txt)
    return len(p & g) / max(1, len(g))


def bpb(m):
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


def reuse(m):
    m.eval(); rs = []
    for di, doc in enumerate(PROBES):
        torch.manual_seed(di)
        o, _ = m.generate(doc[:512], 150, temp=0.85)
        rs.append(context_reuse(TOK.decode(doc[:512]), TOK.decode(o)))
    m.train()
    return float(np.mean(rs))


# real-text reference for context reuse: how much does a genuine continuation reuse its own preceding context?
np.random.seed(11)
_ref = []
for _ in range(10):
    d = pd.batch(1)[0].tolist()
    _ref.append(context_reuse(TOK.decode(d[:512]), TOK.decode(d[512:662])))
REF_REUSE = float(np.mean(_ref))


def run(D, NL, DIN):
    torch.manual_seed(0); np.random.seed(0)
    cfg = C.Config(D=D, DIN=DIN, DV=D, NL=NL, T=CH, res_scale=NL ** -0.5, tie_emb=True)
    cfg.gated = True; cfg.qknorm = True
    m = C.Core(cfg).to(DEV)
    P = sum(p.numel() for p in m.parameters())
    marks = {int(b * P) // (B * CH): b for b in BUDGETS}
    STEPS = max(marks) + 1
    fwd = torch.compile(m); opt = torch.optim.Adam(m.parameters(), lr=1e-3, betas=(0.9, 0.95))
    warm = 200; t0 = time.time(); out = []
    torch.manual_seed(1); np.random.seed(1)
    for i in range(STEPS):
        lr = 1e-3 * ((i + 1) / warm if i < warm else
                     0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (i - warm) / max(1, STEPS - warm))))
        for g in opt.param_groups: g['lr'] = lr
        X = torch.tensor(dmix.batch(B), device=DEV)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            l = F.cross_entropy(fwd(X[:, :CH]).reshape(-1, V), X[:, 1:CH + 1].reshape(-1))
        opt.zero_grad(); l.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        if i in marks:
            bb = bpb(m); rr = reuse(m)
            out.append((P, marks[i], bb, rr))
            print(f"  {P/1e6:6.1f}M {marks[i]:5.0f} tok/param {(i+1)*B*CH/1e6:8.0f}M tokens "
                  f"BPB {bb:7.4f}  context-reuse {rr:5.3f}", flush=True)
    print(f"    ({STEPS*B*CH/(time.time()-t0)/1e3:.0f} ktok/s)", flush=True)
    del m, fwd, opt; torch.cuda.empty_cache()
    return out


if __name__ == "__main__":
    print(f"SCALING PROBE -- extrapolate instead of spending 4 hours on a guess\n"
          f"  GPT-2 reference: 124M, 73 tok/param, BPB 0.9335\n"
          f"  real-text context-reuse reference: {REF_REUSE:.3f}\n", flush=True)
    print(f"  {'params':>7s} {'tok/param':>14s} {'tokens':>13s} {'BPB':>11s}  {'context-reuse'}", flush=True)
    rows = []
    for D, NL, DIN in SIZES:
        rows += run(D, NL, DIN)

    # fit log(BPB) = a + b*log(P) + c*log(tok/param) and extrapolate to 100M at 73 tok/param
    A = np.array([[1.0, np.log(p), np.log(t)] for p, t, _, _ in rows])
    y = np.array([np.log(b) for _, _, b, _ in rows])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = lambda P, T: float(np.exp(coef @ np.array([1.0, np.log(P), np.log(T)])))
    resid = float(np.sqrt(np.mean((A @ coef - y) ** 2)))
    print(f"\n  FIT log(BPB) = {coef[0]:.3f} {coef[1]:+.4f}*log(params) {coef[2]:+.4f}*log(tok/param)"
          f"   rms residual {resid:.4f}", flush=True)
    print(f"\n  {'extrapolation':>34s} {'predicted BPB':>14s}", flush=True)
    for P, T in [(100e6, 20), (100e6, 73), (200e6, 73), (100e6, 200)]:
        print(f"  {f'{P/1e6:.0f}M at {T} tok/param':>34s} {pred(P,T):14.4f}", flush=True)
    print(f"\n  GPT-2 is 0.9335. If 100M at 73 tok/param is predicted well above that, the 4-hour run does NOT",
          flush=True)
    print(f"  close the gap and something other than scale is needed.", flush=True)
