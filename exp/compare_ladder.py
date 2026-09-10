"""WHICH RUNG OF THE LADDER IS ACTUALLY THE BEST MODEL?

The 405M run has been degrading, so "bigger" and "later" are not the same as "better" here. This compares
every checkpoint we still hold on one protocol, so the choice of what to ship is made on measurement rather
than on which number is largest.

All four scored on the SAME eval text, the SAME 2048-token windows, and the SAME bytes. Note this is our
internal metric and is NOT comparable to GPT-2, which is capped at 1024 context; exp/gpt2_bpb.py does that
comparison fairly and the harness runs it.

Generation is reported with real text on the same metrics, because a repetition number without a reference
cannot distinguish good text from mush.
"""
import sys, os, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/opt/code/membrae")
from util import setup_log, cfg_from_ckpt, cap_gpu
setup_log("compare_ladder")
cap_gpu(float(os.environ.get("CAP", "18")))
import numpy as np, torch, torch.nn.functional as F
import core as C
from data_mix import DataMix
from tokenizers import Tokenizer

DEV = C.DEV
TOK = Tokenizer.from_file("/opt/code/membrae/data/bpe32k.json")
CH = 2048
D = "/mnt/kv_cache/pragnosia_data"
MODELS = [("125M finished", f"{D}/p124f.pt"),
          ("191M pre-grow2", f"{D}/p400f_before_grow2.pt"),
          ("279M pre-grow3", f"{D}/p400f_before_grow3.pt"),
          ("405M current",   f"{D}/p400f_405M_snap.pt")]

dm = DataMix(ctx=CH + 1, verbose=False, lang_frac=0.70)
np.random.seed(777)
EV = [torch.tensor(dm.batch(2), device=DEV) for _ in range(12)]

def rep4(ids):
    g = [tuple(ids[i:i+4]) for i in range(len(ids)-3)]
    return 1 - len(set(g))/max(1, len(g))
def dn(ids, n):
    g = [tuple(ids[i:i+n]) for i in range(len(ids)-n+1)]
    return len(set(g))/max(1, len(g))
def longest_run(ids):
    b = c = 1
    for i in range(1, len(ids)):
        c = c + 1 if ids[i] == ids[i-1] else 1
        b = max(b, c)
    return b

np.random.seed(0)
rp = DataMix(ctx=3001, verbose=False, only='dclm')
R = [[int(t) for t in rp.batch(1)[0][:3000]] for _ in range(8)]
REF = (np.mean([rep4(r) for r in R]), np.mean([dn(r,1) for r in R]), np.mean([dn(r,3) for r in R]))
print(f"REAL WEB TEXT reference:  4g-repeat {REF[0]:.3f}   distinct-1 {REF[1]:.3f}   distinct-3 {REF[2]:.3f}\n",
      flush=True)

np.random.seed(11)
pdd = DataMix(ctx=1025, verbose=False, only='dclm')
PROMPT = [int(t) for t in pdd.batch(1)[0][:256]]

@torch.no_grad()
def bpb(m, cfg):
    nats = 0.0; nb = 0
    for X in EV:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg = m(X[:, :CH]).float()
        y = X[:, 1:CH+1]
        nats += F.cross_entropy(lg.reshape(-1, cfg.vocab), y.reshape(-1), reduction='sum').item()
        for b in range(X.shape[0]):
            nb += len(TOK.decode([int(z) for z in y[b]]).encode('utf-8'))
    return nats / (np.log(2) * nb)

@torch.no_grad()
def needle(m, cfg, dists=(0, 200, 1000, 4000)):
    """paired in-context lift: same query, fact present vs absent, so frequency priors cancel."""
    out = []
    np.random.seed(5); pool = []
    while len(pool) < 12000: pool.extend(dm.batch(1)[0].tolist())
    name, other = " Okonkwo", " Vandermeer"
    for d in dists:
        L = []
        for trial in range(6):
            fill = pool[trial*1000: trial*1000 + d]
            cue = TOK.encode(" The vault keeper is").ids
            ans = TOK.encode(name).ids
            def sc(ctx):
                x = torch.tensor([ctx + cue + ans], device=DEV)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    lp = m(x[:, :-1]).float().log_softmax(-1)
                return float(lp[0, -len(ans):].gather(-1, x[0, 1:][-len(ans):].unsqueeze(-1)).mean())
            with_fact = TOK.encode(f" The vault keeper is{name}. ").ids + fill
            without   = TOK.encode(f" The vault keeper is{other}. ").ids + fill
            L.append(sc(with_fact) - sc(without))
        out.append((d, float(np.mean(L)), float(np.std(L)/np.sqrt(len(L)))))
    return out

print(f"{'model':>16s}{'params':>9s}{'BPB':>9s}{'ppl/byte':>10s}{'4g rep':>9s}{'dist-1':>8s}"
      f"{'dist-3':>8s}{'run':>5s}{'needle d=0':>12s}{'d=1000':>9s}", flush=True)
print("-" * 106)
rows = []
for tag, path in MODELS:
    sd = torch.load(path, map_location=DEV)
    cfg = cfg_from_ckpt(sd, T=CH, res_scale=sd['res_scale'])
    m = C.Core(cfg).to(DEV); m.load_state_dict(sd['model']); m.eval()
    P = sum(p.numel() for p in m.parameters())
    b = bpb(m, cfg)
    torch.manual_seed(0)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        out, _ = m.generate(torch.tensor([PROMPT], device=DEV), 3000, temp=0.9, top_p=0.95)
    g = [int(t) for t in out]
    nd = needle(m, cfg)
    rows.append((tag, P, b, g, nd))
    print(f"{tag:>16s}{P/1e6:>8.1f}M{b:>9.4f}{2**b:>10.3f}{rep4(g):>9.3f}{dn(g,1):>8.3f}"
          f"{dn(g,3):>8.3f}{longest_run(g):>5d}{nd[0][1]:>+12.3f}{nd[2][1]:>+9.3f}", flush=True)
    del m; torch.cuda.empty_cache()

best = min(rows, key=lambda r: r[2])
print(f"\n  BEST BPB: {best[0]} at {best[2]:.4f}   (seed noise floor 0.0042, so 2 sigma is 0.0120)")
print("\n" + "=" * 106)
print("  GENERATION SAMPLES (temp 0.9, 3000 tokens, first 480 chars each)")
print("=" * 106)
for tag, P, b, g, nd in rows:
    print(f"\n  --- {tag} ({P/1e6:.0f}M, BPB {b:.4f}) ---")
    print(f"  {TOK.decode(g)[:480]!r}", flush=True)
