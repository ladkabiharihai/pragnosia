"""WHAT IS THE GROWING MODEL ACTUALLY WORTH RIGHT NOW?

The live 191M reads 0.9288 BPB against the finished 125M's 0.9083, and that comparison is unfair in a way that
matters for a go/no-go decision: the 125M was ANNEALED (its learning rate decayed to near zero at the end of a
WSD schedule) while the 191M is mid-flight at a constant 6e-4. A constant-LR model is bouncing around a basin
it has not settled into. Comparing the two and concluding "growth made it worse" would be reading the schedule,
not the model.

This anneals a COPY and re-measures. Nothing here touches the live checkpoint or the running trainer.

Read the result as a LOWER BOUND. The decay here is short (a real WSD tail is 10-20% of the run), so it
recovers part of what a full anneal would. If even this partial decay closes the gap, the growing model is
already ahead and the ladder is working. If it does not close it, we know how far there is to go while there
is still time to act on it, rather than discovering it on the last day of GPU access.
"""
import sys, os, math, time, shutil, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/opt/code/membrae")
from util import setup_log, cfg_from_ckpt, cap_gpu
setup_log("anneal_probe")
cap_gpu(9.0)
import numpy as np, torch, torch.nn.functional as F
import core as C
from data_mix import DataMix
from tokenizers import Tokenizer

DEV = C.DEV
TOK = Tokenizer.from_file("/opt/code/membrae/data/bpe32k.json")
CH = 2048
# Memory in _full_mix scales with B x T x T (the within-chunk read is quadratic in the chunk), and at B=4,
# T=2048 this OOMed under a 9 GiB cap on a card already holding prod plus the live trainer. Accumulate
# instead: B=1 per micro-step, ACC micro-steps per update, so the EFFECTIVE batch is unchanged and the
# context stays 2048 -- shrinking the context would have made the anneal happen at a different sequence
# length than the run it is probing, which is a different experiment.
B = int(os.environ.get("B", "1"))
ACC = int(os.environ.get("ACC", "4"))
STEPS = int(os.environ.get("STEPS", "2000"))
LR0 = 6e-4
SRC = "/mnt/kv_cache/pragnosia_data/p400f.pt"
TMP = "/mnt/kv_cache/pragnosia_data/_anneal_probe_copy.pt"

# COPY FIRST. The live trainer rewrites SRC every 2000 steps; reading it directly while it is being written is
# a torn read, and writing anywhere near it would corrupt the run.
shutil.copyfile(SRC, TMP)
sd = torch.load(TMP, map_location=DEV)
cfg = cfg_from_ckpt(sd, T=CH, res_scale=sd['res_scale'])
m = C.Core(cfg).to(DEV); m.load_state_dict(sd['model'])
P = sum(p.numel() for p in m.parameters())
print(f"copied checkpoint at step {sd.get('step',0):,}, {P/1e6:.1f}M, {sd.get('k',0)} growths", flush=True)

dm = DataMix(ctx=CH + 1, verbose=False, lang_frac=0.70)
np.random.seed(777)
EV = [torch.tensor(dm.batch(2), device=DEV) for _ in range(12)]

@torch.no_grad()
def bpb():
    m.eval(); nats = 0.0; nb = 0
    for X in EV:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg = m(X[:, :CH]).float()
        y = X[:, 1:CH + 1]
        nats += F.cross_entropy(lg.reshape(-1, cfg.vocab), y.reshape(-1), reduction='sum').item()
        for b in range(X.shape[0]):
            nb += len(TOK.decode([int(z) for z in y[b]]).encode('utf-8'))
    m.train(); return nats / (np.log(2) * nb)

b0 = bpb()
print(f"BEFORE anneal   BPB {b0:.4f}   (finished 125M, fully annealed, is 0.9083)\n", flush=True)

opt = torch.optim.Adam(m.parameters(), lr=LR0, betas=(0.9, 0.95), fused=True)
if 'opt' in sd:
    try: opt.load_state_dict(sd['opt'])
    except Exception as e: print(f"  (optimizer state not reused: {type(e).__name__})", flush=True)
del sd; torch.cuda.empty_cache()

t0 = time.time()
for i in range(STEPS):
    lr = LR0 * (1.0 - (i + 1) / STEPS)          # linear decay to zero: the WSD tail, compressed
    for g in opt.param_groups: g['lr'] = lr
    opt.zero_grad(set_to_none=True)
    for _a in range(ACC):
        X = torch.tensor(dm.batch(B), device=DEV)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            l = F.cross_entropy(m(X[:, :CH]).reshape(-1, cfg.vocab), X[:, 1:CH + 1].reshape(-1)) / ACC
        l.backward()
        del X
    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    if (i + 1) % 250 == 0:
        el = time.time() - t0
        print(f"  {i+1}/{STEPS}  lr {lr:.2e}  loss {l.item()*ACC:.3f}  {el/60:.1f}m  "
              f"eta {(STEPS-i-1)*el/(i+1)/60:.0f}m  BPB {bpb():.4f}", flush=True)

b1 = bpb()
tok = STEPS * B * ACC * CH
print(f"\n{'='*92}")
print(f"  AFTER a {tok/1e6:.0f}M-token linear decay to zero:  BPB {b0:.4f} -> {b1:.4f}   ({b1-b0:+.4f})")
print(f"  finished 125M (fully annealed):                    0.9083")
print(f"  gap to the 125M: {b1-0.9083:+.4f} BPB", flush=True)
if b1 < 0.9083:
    print("  The growing model is ALREADY AHEAD of the finished 125M once the schedule is accounted for.")
else:
    print("  Still behind. This decay is short, so treat it as a lower bound on what a full anneal recovers,")
    print("  but the remaining gap is the real number to plan against, not the un-annealed 0.9288.")
print("="*92, flush=True)
try: os.remove(TMP)
except OSError: pass
