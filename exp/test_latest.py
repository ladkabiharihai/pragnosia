"""TEST THE LATEST CHECKPOINT: what does the growing 191M actually do right now, next to the finished 125M.

BPB here uses 2048-token windows on our mix. That tracks progress and compares OUR checkpoints to each other;
it is NOT the GPT-2-comparable number, which needs the matched 1024-token protocol (exp/gpt2_bpb.py). Quoting
this figure against GPT-2 would be the exact unfair comparison the claim sheet exists to avoid.
"""
import sys, os, warnings, time
warnings.filterwarnings("ignore")
sys.path.insert(0, "/opt/code/membrae")
from util import setup_log, cfg_from_ckpt, cap_gpu
setup_log("test_latest")
cap_gpu(8.0)
import numpy as np, torch, torch.nn.functional as F
import core as C
from data_mix import DataMix
from tokenizers import Tokenizer

DEV = C.DEV
TOK = Tokenizer.from_file("/opt/code/membrae/data/bpe32k.json")
CH = 2048
NEW = "/mnt/kv_cache/pragnosia_data/p400f.pt"
OLD = "/mnt/kv_cache/pragnosia_data/p124f.pt"

dm = DataMix(ctx=CH + 1, verbose=False, lang_frac=0.70)
np.random.seed(777)
EV = [torch.tensor(dm.batch(2), device=DEV) for _ in range(12)]

def load(p):
    sd = torch.load(p, map_location=DEV)
    cfg = cfg_from_ckpt(sd, T=CH, res_scale=sd['res_scale'])
    m = C.Core(cfg).to(DEV); m.load_state_dict(sd['model']); m.eval()
    return m, cfg, sd

@torch.no_grad()
def bpb(m, cfg):
    nats = 0.0; nb = 0
    for X in EV:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg = m(X[:, :CH]).float()
        y = X[:, 1:CH + 1]
        nats += F.cross_entropy(lg.reshape(-1, cfg.vocab), y.reshape(-1), reduction='sum').item()
        for b in range(X.shape[0]):
            nb += len(TOK.decode([int(z) for z in y[b]]).encode('utf-8'))
    return nats / (np.log(2) * nb)

def rep4(ids):
    g = [tuple(ids[i:i+4]) for i in range(len(ids)-3)]
    return 1 - len(set(g))/max(1, len(g))

print("=" * 100)
print("  LATEST CHECKPOINT vs THE FINISHED 125M   (same eval set, same 2048-token windows)")
print("=" * 100, flush=True)
rows = []
for tag, path in [("finished 125M (p124f)", OLD), ("growing (p400f, live)", NEW)]:
    m, cfg, sd = load(path)
    P = sum(p.numel() for p in m.parameters())
    b = bpb(m, cfg)
    rows.append((tag, P, sd.get('step', 0), b, sd.get('k', 0)))
    print(f"  {tag:>24s}   {P/1e6:6.1f}M   step {sd.get('step',0):>7,}   growths {sd.get('k',0)}   "
          f"BPB {b:.4f}   ppl(byte) {2**b:.3f}", flush=True)
    del m; torch.cuda.empty_cache()

d = rows[1][3] - rows[0][3]
print(f"\n  delta {d:+.4f} BPB  ({'the growing model is AHEAD' if d < 0 else 'the growing model is BEHIND'}"
      f"; seed noise floor is 0.0042, so 2 sigma is 0.0120)")

print("\n" + "=" * 100)
print("  GENERATION from the latest checkpoint, temperature 0.9, closed-loop refresh on (the shipped default)")
print("=" * 100, flush=True)
m, cfg, sd = load(NEW)
np.random.seed(11); pdd = DataMix(ctx=1025, verbose=False, only='dclm')
prompt = [int(t) for t in pdd.batch(1)[0][:96]]
print(f"  PROMPT: {TOK.decode(prompt)[-300:]!r}\n", flush=True)
for temp in (0.9, 0.8):
    torch.manual_seed(0)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        out, _ = m.generate(torch.tensor([prompt], device=DEV), 1200, temp=temp, top_p=0.95)
    g = [int(t) for t in out]
    txt = TOK.decode(g)
    print(f"  --- temp {temp}   4-gram repetition {rep4(g):.3f}   refresh fired "
          f"{getattr(m,'last_refresh_triggers',0)}x   (real web text 0.051)")
    print(f"  {txt[:700]!r}\n", flush=True)
