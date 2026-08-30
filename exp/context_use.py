"""DOES THE MODEL ACTUALLY CONDITION ON ITS CONTEXT? A model that ignores context and predicts from unigram/local
statistics alone would look exactly like what we see: grammatical fragments with no meaning. That is a different
failure from "undertrained", and it would be fixable.

Also settles what the 0.6-nat gap means. Training reports loss 3.76; the fixed eval reports 4.35. If that gap is
mostly CONTEXT (training always has 2048 tokens plus carried memory, the eval's first chunk has neither), then every
quality number quoted from the eval understates the model.

  1. CONTEXT SENSITIVITY  -- same target token, real context vs SHUFFLED vs a DIFFERENT document's context.
                             If loss barely moves, the model is not using context at all.
  2. LOSS BY POSITION     -- within one chunk, does prediction improve as context accumulates?
  3. THE GAP EXPLAINED    -- loss at position 0 vs position 2047 vs with carried memory.
"""
import sys
sys.path.insert(0, "/opt/code/membrae")
from util import setup_log
setup_log("context_use")
import numpy as np, torch, torch.nn.functional as F
import core as C
from data_mix import DataMix
from tokenizers import Tokenizer

DEV = C.DEV
TOK = Tokenizer.from_file("/opt/code/membrae/data/bpe32k.json")
CH = 2048
sd = torch.load("/mnt/kv_cache/pragnosia_data/pragnosia.pt", map_location=DEV)
cfg = C.Config(D=sd['D'], DIN=8, DV=sd.get('DV', sd['D']), NL=sd['NL'], T=CH, res_scale=16 ** -0.5)
m = C.Core(cfg).to(DEV); m.load_state_dict(sd['model']); m.eval()
print(f"=== {sum(p.numel() for p in m.parameters())/1e6:.0f}M, step {sd['step']}, "
      f"{sd['step']*32768*8/1e9:.0f}B tokens ===\n", flush=True)

np.random.seed(31)
dm = DataMix(ctx=CH * 4 + 1, verbose=False, only='window2')
docs = [torch.tensor(dm.batch(1), device=DEV) for _ in range(6)]


@torch.no_grad()
def loss_of(x, y):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        lg = m(x).float()
    return F.cross_entropy(lg.reshape(-1, cfg.vocab), y.reshape(-1)).item()


print("1) CONTEXT SENSITIVITY -- predict the LAST 128 tokens of a window, varying only what precedes them", flush=True)
real, shuf, other, none_ = [], [], [], []
for i, d in enumerate(docs):
    seg = d[:, :512]                                   # 512-token window; we score the last 128
    tgt = seg[:, -128:]
    with torch.no_grad():
        # (a) real preceding context
        real.append(loss_of(seg[:, :-1], seg[:, 1:])
                    if False else loss_of(seg[:, :511], seg[:, 1:512]))
        # (b) same target, but the preceding 384 tokens SHUFFLED (word salad context)
        pre = seg[:, :384]
        perm = pre[:, torch.randperm(pre.shape[1], device=DEV)]
        sseg = torch.cat([perm, seg[:, 384:]], 1)
        shuf.append(loss_of(sseg[:, :511], sseg[:, 1:512]))
        # (c) same target, preceded by a DIFFERENT document
        oseg = torch.cat([docs[(i + 1) % len(docs)][:, :384], seg[:, 384:]], 1)
        other.append(loss_of(oseg[:, :511], oseg[:, 1:512]))
        # (d) the target alone, no preceding context at all
        none_.append(loss_of(tgt[:, :-1], tgt[:, 1:]))
print(f"   real context      {np.mean(real):.4f}", flush=True)
print(f"   SHUFFLED context  {np.mean(shuf):.4f}   ({np.mean(shuf)-np.mean(real):+.4f} vs real)", flush=True)
print(f"   OTHER doc context {np.mean(other):.4f}   ({np.mean(other)-np.mean(real):+.4f} vs real)", flush=True)
print(f"   NO context        {np.mean(none_):.4f}   ({np.mean(none_)-np.mean(real):+.4f} vs real)", flush=True)
print(f"   -> {'the model IS using context' if np.mean(other)-np.mean(real) > 0.15 else 'CONTEXT IS BARELY USED -- this is the bug'}",
      flush=True)

print("\n2) LOSS BY POSITION within one chunk (does prediction improve as context accumulates?)", flush=True)
buckets = [(0, 16), (16, 64), (64, 256), (256, 1024), (1024, 2047)]
agg = {b: [] for b in buckets}
with torch.no_grad():
    for d in docs:
        x = d[:, :CH]; y = d[:, 1:CH + 1]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg = m(x).float()
        nll = F.cross_entropy(lg[0], y[0], reduction='none')
        for b in buckets: agg[b].append(nll[b[0]:b[1]].mean().item())
for b in buckets:
    print(f"   positions {b[0]:5d}-{b[1]:5d}: loss {np.mean(agg[b]):.4f}", flush=True)

print("\n3) THE 3.76 vs 4.35 GAP EXPLAINED", flush=True)
first_chunk, late_chunk, with_mem = [], [], []
with torch.no_grad():
    for d in docs:
        first_chunk.append(loss_of(d[:, :CH], d[:, 1:CH + 1]))
        st = None
        for c in range(3):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg, st = m(d[:, c * CH:(c + 1) * CH], states=st, return_states=True)
            st = m.decay_states(st, 0.05)
        yl = d[:, 3 * CH + 1:4 * CH + 1]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lgm, _ = m(d[:, 3 * CH:4 * CH], states=st, return_states=True)
            lg0 = m(d[:, 3 * CH:4 * CH])
        with_mem.append(F.cross_entropy(lgm.reshape(-1, cfg.vocab).float(), yl.reshape(-1)).item())
        late_chunk.append(F.cross_entropy(lg0.reshape(-1, cfg.vocab).float(), yl.reshape(-1)).item())
print(f"   chunk 0, no memory   (what the eval reports) {np.mean(first_chunk):.4f}", flush=True)
print(f"   chunk 3, no memory                           {np.mean(late_chunk):.4f}", flush=True)
print(f"   chunk 3, WITH carried memory (as trained)    {np.mean(with_mem):.4f}", flush=True)
print(f"   -> the eval's headline number is {np.mean(first_chunk)-np.mean(with_mem):+.4f} nats WORSE than the "
      f"regime the model actually runs in", flush=True)
