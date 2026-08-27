"""Look at the model from several INDEPENDENT angles. Greedy loops could mean (a) a broken model, or (b) a normal
early-training model whose argmax is dominated by high-frequency tokens. These angles distinguish them."""
import sys; sys.path.insert(0,"/opt/code/membrae")
from util import setup_log; setup_log("deep_check")
import numpy as np, torch, torch.nn.functional as F
import core as C; from data_mix import DataMix
from tokenizers import Tokenizer
DEV=C.DEV; TOK=Tokenizer.from_file("/opt/code/membrae/data/bpe32k.json"); CH=2048
sd=torch.load("/mnt/kv_cache/pragnosia_data/pragnosia.pt",map_location=DEV)
cfg=C.Config(D=sd['D'],DIN=8,DV=sd.get('DV',sd['D']),NL=sd['NL'],T=CH,res_scale=16**-0.5)
m=C.Core(cfg).to(DEV); m.load_state_dict(sd['model']); m.eval()
print(f"@ step {sd['step']} | {sum(p.numel() for p in m.parameters())/1e6:.0f}M\n")

# ANGLE 1: is the model actually PREDICTING well on real text? (the only thing training optimizes)
np.random.seed(42); dm=DataMix(ctx=CH+1,verbose=False)
@torch.no_grad()
def teacher_forced(nb=8):
    tot=0; acc=0; n=0
    for _ in range(nb):
        X=torch.tensor(dm.batch(4),device=DEV)
        with torch.autocast("cuda",dtype=torch.bfloat16): lg=m(X[:,:CH]).float()
        y=X[:,1:CH+1]
        tot+=F.cross_entropy(lg.reshape(-1,cfg.vocab),y.reshape(-1)).item()
        acc+=(lg.argmax(-1)==y).float().mean().item(); n+=1
    return tot/n, 100*acc/n
l,a=teacher_forced()
print(f"1) TEACHER-FORCED on real text: loss {l:.3f} (ppl {np.exp(l):.0f}), next-token accuracy {a:.1f}%")
print(f"   -> the model DOES predict real text ({a:.0f}% exact next-token); greedy FREE-RUN is a different, harder task")

# ANGLE 2: what does the distribution look like? entropy + top-token mass
@torch.no_grad()
def dist_stats():
    X=torch.tensor(dm.batch(2),device=DEV)
    with torch.autocast("cuda",dtype=torch.bfloat16): lg=m(X[:,:CH]).float()
    p=lg.softmax(-1); ent=-(p*(p+1e-9).log()).sum(-1).mean().item()
    top1=p.max(-1).values.mean().item()
    return ent, top1
e,t1=dist_stats()
print(f"\n2) OUTPUT DISTRIBUTION: entropy {e:.2f} nats (uniform would be {np.log(cfg.vocab):.2f}), mean top-1 prob {t1:.3f}")
print(f"   -> {'peaked and confident' if t1>0.3 else 'FLAT/uncertain: argmax is nearly arbitrary, which is exactly why greedy loops'}")

# ANGLE 3: does the loop happen because argmax repeats the LAST token? (a known early-training failure)
@torch.no_grad()
def repeat_bias():
    X=torch.tensor(dm.batch(4),device=DEV)
    with torch.autocast("cuda",dtype=torch.bfloat16): lg=m(X[:,:CH]).float()
    pred=lg.argmax(-1)
    same_as_prev=(pred[:,1:]==X[:,1:CH]).float().mean().item()     # predicts the CURRENT token again
    correct=(pred==X[:,1:CH+1]).float().mean().item()
    return 100*same_as_prev,100*correct
sp,cr=repeat_bias()
print(f"\n3) REPEAT BIAS: argmax equals the CURRENT token {sp:.1f}% of the time (correct next-token {cr:.1f}%)")
print(f"   -> {'STRONG repeat bias: the model copies rather than predicts' if sp>25 else 'no pathological repeat bias'}")

# ANGLE 4: how much of window2 is NOT English? (multilingual dilution at 49M)
@torch.no_grad()
def lang_mix(n=60):
    import re
    lat=0; tot=0
    for _ in range(n):
        s=TOK.decode([int(t) for t in torch.tensor(dm.batch(1))[0,:200]])
        letters=[c for c in s if c.isalpha()]
        if not letters: continue
        lat+=sum(1 for c in letters if ord(c)<0x250)/len(letters); tot+=1
    return 100*lat/tot
print(f"\n4) CORPUS LANGUAGE: {lang_mix():.0f}% of alphabetic characters are Latin-script")
print(f"   -> non-Latin share dilutes a 49M model across scripts")

# ANGLE 5: is loss still improving, or has it stalled? (fixed eval, two checkpoints)
print(f"\n5) PROGRESS (fixed eval set, from logs/eval_history.tsv):")
import csv
rows=list(csv.DictReader(open('/opt/code/membrae/logs/eval_history.tsv'),delimiter='\t'))
for r in rows[-4:]: print(f"   step {r['step']:>6s}  {r['params_M']:>3s}M  loss {r['loss']}  ppl {r['ppl']}  mem_gain {r['mem_gain']}")
