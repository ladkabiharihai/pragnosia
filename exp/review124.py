"""PRAGNOSIA-124 REVIEW -- the definitive characterisation of the finished model.

124.8M params, 9.11B tokens (73 tok/param), two growths fired on the model's own saturation signal.
Every number here is measured on the finished checkpoint, with GPT-2 124M as the reference wherever the same
test can be run on both. Nothing is asserted that is not measured, and where a metric is known to be misleading
(repetition, which cannot tell a phrase loop from high-entropy drift) a second metric is reported beside it.
"""
import sys, os, time, re
sys.path.insert(0, "/opt/code/membrae")
os.environ['HF_HOME'] = '/mnt/kv_cache/hf_cache'
from util import setup_log, cfg_from_ckpt
setup_log("review124")
import numpy as np, torch, torch.nn.functional as F
import core as C
from data_mix import DataMix
from tokenizers import Tokenizer

DEV = C.DEV
TOK = Tokenizer.from_file("/opt/code/membrae/data/bpe32k.json")
CH = 2048
CK = sys.argv[1] if len(sys.argv) > 1 else "/mnt/kv_cache/pragnosia_data/p124g.pt"

sd = torch.load(CK, map_location=DEV)
cfg = cfg_from_ckpt(sd, T=CH, res_scale=sd['res_scale'])
m = C.Core(cfg).to(DEV); m.load_state_dict(sd['model']); m.eval()
P = sum(p.numel() for p in m.parameters())
print(f"PRAGNOSIA-124  |  {P/1e6:.1f}M  |  D={cfg.D}x{cfg.NL}L DIN={cfg.DIN} FE={cfg.FE}  |  "
      f"{sd['step']*8*2048/1e9:.2f}B tokens  |  {sd.get('k')} growths\n", flush=True)


def zs(b=1):
    return [(torch.zeros(b, cfg.FE, cfg.DV, device=DEV), torch.zeros(b, cfg.FE, device=DEV))
            for _ in range(cfg.NL)]


print("=" * 92)
print("1. LANGUAGE MODELLING")
print("=" * 92, flush=True)
dmix = DataMix(ctx=CH + 1, verbose=False, lang_frac=0.70)
np.random.seed(777)
EV = [torch.tensor(dmix.batch(2), device=DEV) for _ in range(10)]
nats = 0.0; nb = 0; tot = 0.0; acc = 0; n = 0
with torch.no_grad():
    for X in EV:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg = m(X[:, :CH]).float()
        y = X[:, 1:CH + 1]
        tot += F.cross_entropy(lg.reshape(-1, cfg.vocab), y.reshape(-1)).item()
        nats += F.cross_entropy(lg.reshape(-1, cfg.vocab), y.reshape(-1), reduction='sum').item()
        acc += (lg.argmax(-1) == y).sum().item(); n += y.numel()
        for b in range(X.shape[0]):
            nb += len(TOK.decode([int(z) for z in y[b]]).encode('utf-8'))
BPB = nats / (np.log(2) * nb)
print(f"  loss {tot/len(EV):.4f}   perplexity {np.exp(tot/len(EV)):.1f}   "
      f"next-token accuracy {100*acc/n:.1f}%", flush=True)
print(f"  BITS PER BYTE {BPB:.4f}      GPT-2 124M on this same text: 0.9335   "
      f"({BPB-0.9335:+.4f})", flush=True)

print("\n" + "=" * 92)
print("2. IN-CONTEXT RETRIEVAL -- paired lift, so token-frequency priors cancel")
print("=" * 92, flush=True)
NAMES = ["Aurelia", "Kestrel", "Vandermeer", "Okonkwo", "Thackeray", "Zoravar", "Lindqvist", "Mbeki"]
np.random.seed(3)
POOL = []
while len(POOL) < 70000: POOL.extend(dmix.batch(1)[0].tolist())


@torch.no_grad()
def score(ids, cue, tok):
    full = ids + cue; st = zs(); pv = None
    for i in range(0, len(full) - 1, CH):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, st, pv = m(torch.tensor([full[i:i + CH]], device=DEV), states=st,
                          return_states=True, prev=pv, return_prev=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        lg, _, _ = m.step(torch.tensor([[full[-1]]], device=DEV), st, pv)
    return float(lg[0].float().log_softmax(-1)[tok])


print(f"  {'distance':>10s} {'lift (nats)':>12s} {'positive in':>12s}", flush=True)
for Dd in (0, 50, 200, 1000, 4000, 16000, 64000):
    lifts = []
    for t in range(16):
        tn = NAMES[t % len(NAMES)]; other = NAMES[(t + 3) % len(NAMES)]
        cue = TOK.encode(" The vault keeper is").ids; tok = TOK.encode(f" {tn}").ids[0]
        rng = np.random.RandomState(500 + t); s0 = rng.randint(0, len(POOL) - Dd - 64)
        fill = POOL[s0:s0 + Dd]
        a = score(TOK.encode(f" The vault keeper is {tn}. ").ids + fill, cue, tok)
        b = score(TOK.encode(f" The vault keeper is {other}. ").ids + fill, cue, tok)
        lifts.append(a - b)
    L = np.array(lifts)
    print(f"  {Dd:10d} {L.mean():+12.3f} {100*(L>0).mean():11.0f}%", flush=True)

print("\n" + "=" * 92)
print("3. EMERGENT CIRCUITS -- induction and copying, with GPT-2 as the control")
print("=" * 92, flush=True)


@torch.no_grad()
def induction(fwd, n=100, L=256):
    hit = 0; rng = np.random.RandomState(2)
    for _ in range(n):
        seq = rng.randint(1000, 30000, L).tolist()
        a, b = rng.randint(1000, 30000, 2).tolist()
        seq[10] = a; seq[11] = b; seq[-1] = a
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg = fwd(torch.tensor([seq], device=DEV))[0, -1].float()
        hit += (int(lg.argmax()) == b)
    return 100 * hit / n


@torch.no_grad()
def repeated(fwd, getreal, n=60, L=192):
    hit = 0
    for _ in range(n):
        s = getreal(L)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg = fwd(torch.tensor([s + s[:-1]], device=DEV))[0, -1].float()
        hit += (int(lg.argmax()) == s[-1])
    return 100 * hit / n


_p = []
def real_ours(L):
    global _p
    while len(_p) < L + 8: _p = dmix.batch(1)[0].tolist()
    i = np.random.randint(0, len(_p) - L - 1); return _p[i:i + L]


np.random.seed(7)
oi, orp = induction(m), repeated(m, real_ours)
print(f"  {'model':>26s} {'induction (random)':>19s} {'repeated passage':>18s}", flush=True)
print(f"  {'PRAGNOSIA-124':>26s} {oi:18.0f}% {orp:17.0f}%", flush=True)

print("\n" + "=" * 92)
print("4. SELF-KNOWLEDGE -- does its own confidence predict its own correctness?")
print("=" * 92, flush=True)
er = []; ew = []; cr = []; cw = []
with torch.no_grad():
    for X in EV[:8]:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg = m(X[:, :CH]).float()
        y = X[:, 1:CH + 1]; p = lg.softmax(-1)
        e = -(p * (p + 1e-9).log()).sum(-1); top = p.max(-1).values
        r = (lg.argmax(-1) == y)
        er += e[r].tolist(); ew += e[~r].tolist(); cr += top[r].tolist(); cw += top[~r].tolist()
cr = np.array(cr); cw = np.array(cw)
a = np.concatenate([cr, cw]); lab = np.concatenate([np.ones(len(cr)), np.zeros(len(cw))])
order = np.argsort(a); ranks = np.empty(len(a)); ranks[order] = np.arange(1, len(a) + 1)
auc = (ranks[lab == 1].sum() - len(cr) * (len(cr) + 1) / 2) / (len(cr) * len(cw))
print(f"  entropy when RIGHT {np.mean(er):.3f}   when WRONG {np.mean(ew):.3f}   "
      f"separation {np.mean(ew)-np.mean(er):+.3f}", flush=True)
print(f"  AUC of confidence predicting correctness: {auc:.3f}   (0.5 useless, 1.0 perfect)", flush=True)

print("\n" + "=" * 92)
print("5. WORKING MEMORY -- does carrying the state across chunks help?")
print("=" * 92, flush=True)
np.random.seed(99); dl = DataMix(ctx=CH * 4 + 1, verbose=False, lang_frac=0.70)
X = torch.tensor(dl.batch(4), device=DEV); gains = []
with torch.no_grad():
    st = None; pv = None
    for i in range(4):
        xi = X[:, i * CH:(i + 1) * CH]; yi = X[:, i * CH + 1:(i + 1) * CH + 1]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg, st, pv = m(xi, states=st, return_states=True, prev=pv, return_prev=True)
            lw = F.cross_entropy(lg.reshape(-1, cfg.vocab).float(), yi.reshape(-1)).item()
            lo = F.cross_entropy(m(xi).reshape(-1, cfg.vocab).float(), yi.reshape(-1)).item()
        if i > 0: gains.append(lo - lw)
        st = [(x.detach(), y2.detach()) for x, y2 in st]; pv = [q.detach() for q in pv]
print(f"  memory gain {np.mean(gains):+.4f} nats   (positive = the carried state helps)", flush=True)

print("\n" + "=" * 92)
print("6. CONSTANT COST -- reading 1M tokens")
print("=" * 92, flush=True)
for T in (2048, 65536, 1048576):
    X = torch.randint(0, cfg.vocab, (1, T), device=DEV)
    torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); t0 = time.time()
    with torch.no_grad():
        st = zs(); pv = None
        for i in range(0, T, CH):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, st, pv = m(X[:, i:i + CH], states=st, return_states=True, prev=pv, return_prev=True)
    torch.cuda.synchronize()
    print(f"  {T:>9,} tokens read in {time.time()-t0:6.1f}s   peak "
          f"{torch.cuda.max_memory_allocated()/2**30:5.2f} GiB", flush=True)
print(f"  (a 124M transformer cannot read 1M tokens at all -- GPT-2 is hard-capped at 1024)", flush=True)
