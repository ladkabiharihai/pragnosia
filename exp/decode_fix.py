"""GENERATION IS BROKEN BY THE DECODER, NOT THE MODEL. Evidence from exp/gen_debug.py:
  - teacher-forced, the model repeats its previous prediction only 5.0% of positions (no looping tendency)
  - it reaches 26.9% exact next-token accuracy and 80% on factual pairs
  - but at many positions the distribution is nearly FLAT (top-1 only 0.09), and GREEDY on a flat distribution picks
    an arbitrary tiny peak, which then locks in at 0.92-0.97 and repeats forever
  - freezing the recurrent state does NOT help, so it is not a self-feedback loop

So the collapse is a decoding failure on an under-confident distribution, not absent knowledge. Test decoders that
are robust to flat distributions, and measure both DIVERSITY and whether the text stays on topic.
"""
import sys
sys.path.insert(0, "/opt/code/membrae")
from util import setup_log
setup_log("decode_fix")
import numpy as np, torch, torch.nn.functional as F
import core as C
from data_mix import DataMix
from tokenizers import Tokenizer

DEV = C.DEV
TOK = Tokenizer.from_file("/opt/code/membrae/data/bpe32k.json")
CK = "/mnt/kv_cache/pragnosia_data/pragnosia.pt"
sd = torch.load(CK, map_location=DEV)
cfg = C.Config(D=sd['D'], DIN=8, DV=sd.get('DV', sd['D']), NL=sd['NL'], T=2048, res_scale=16 ** -0.5)
m = C.Core(cfg).to(DEV); m.load_state_dict(sd['model']); m.eval()
zs = lambda: [(torch.zeros(1, cfg.FE, cfg.DV, device=DEV), torch.zeros(1, cfg.FE, device=DEV))
              for _ in range(cfg.NL)]


@torch.no_grad()
def generate(prompt, n=48, temp=1.0, top_p=1.0, top_k=0, rep_pen=1.0, greedy=False):
    ids = TOK.encode(prompt).ids
    x = torch.tensor([ids], device=DEV)
    st, pv = m.prefill(x, zs())
    out = []; cur = x[:, -1:]
    hist = list(ids)
    for _ in range(n):
        lg, st, pv = m.step(cur, st, pv)
        lg = lg[0].float()
        if rep_pen != 1.0:                                   # penalise tokens already produced (Keskar et al.)
            for t in set(hist[-64:]):
                lg[t] = lg[t] / rep_pen if lg[t] > 0 else lg[t] * rep_pen
        if greedy:
            nx = int(lg.argmax())
        else:
            p = (lg / temp).softmax(-1)
            if top_k > 0:
                v, i = torch.topk(p, top_k)
                p = torch.zeros_like(p).scatter_(0, i, v)
            if top_p < 1.0:
                sp, si = torch.sort(p, descending=True)
                keep = (sp.cumsum(0) - sp) < top_p
                p = torch.zeros_like(p).scatter_(0, si[keep], sp[keep])
            nx = int(torch.multinomial(p / p.sum(), 1))
        out.append(nx); hist.append(nx); cur = torch.tensor([[nx]], device=DEV)
    return out


def stats(toks):
    d1 = len(set(toks)) / max(1, len(toks))
    g = [tuple(toks[i:i + 4]) for i in range(len(toks) - 3)]
    r4 = 1 - len(set(g)) / max(1, len(g))
    return d1, r4


PROMPTS = ["The capital of France is", "Water freezes at", "The sun is a",
           "Photosynthesis is the process by which plants", "In the beginning of the story"]

CONFIGS = [
    ("greedy (what we used)", dict(greedy=True)),
    ("temp 1.0 (what eval used)", dict(temp=1.0)),
    ("top-p 0.9, temp 0.8", dict(temp=0.8, top_p=0.9)),
    ("top-k 40, temp 0.8", dict(temp=0.8, top_k=40)),
    ("top-p 0.9 + rep-pen 1.2", dict(temp=0.8, top_p=0.9, rep_pen=1.2)),
    ("greedy + rep-pen 1.3", dict(greedy=True, rep_pen=1.3)),
    ("top-k 40 + rep-pen 1.2, t=0.7", dict(temp=0.7, top_k=40, rep_pen=1.2)),
]

# reference: what real text from this corpus scores
np.random.seed(9); dm = DataMix(ctx=100, verbose=False, only='window2')
rd, rr = zip(*[stats(dm.batch(1)[0][:48].tolist()) for _ in range(20)])
print(f"REAL TEXT reference: distinct-1 {np.mean(rd):.2f}  4gram-rep {np.mean(rr):.2f}\n", flush=True)
print(f"  {'decoder':>30s} {'distinct-1':>11s} {'4gram-rep':>10s}", flush=True)
best = None
for name, kw in CONFIGS:
    torch.manual_seed(0)
    ds, rs = [], []
    for p in PROMPTS:
        t = generate(p, 48, **kw)
        d, r = stats(t); ds.append(d); rs.append(r)
    d, r = np.mean(ds), np.mean(rs)
    print(f"  {name:>30s} {d:11.2f} {r:10.2f}", flush=True)
    score = d - r
    if best is None or score > best[0]: best = (score, name, kw)

print(f"\n  BEST: {best[1]}\n", flush=True)
torch.manual_seed(0)
for p in PROMPTS:
    print(f"   {p!r}\n      -> {TOK.decode(generate(p,56,**best[2]))!r}", flush=True)
