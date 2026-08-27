"""WHY does needle fail, and what would fix it? Test the model on the EXACT distribution it was trained on,
then progressively move toward my synthetic needle, to find where it breaks."""
import sys; sys.path.insert(0,"/opt/code/membrae")
from util import setup_log; setup_log("needle_diag")
import numpy as np, torch, torch.nn.functional as F
import core as C
from tokenizers import Tokenizer
DEV=C.DEV; TOK=Tokenizer.from_file("/opt/code/membrae/data/bpe32k.json"); CH=2048
sd=torch.load(sys.argv[1] if len(sys.argv)>1 else "/mnt/kv_cache/pragnosia_data/snap_6000.pt",map_location=DEV)
cfg=C.Config(D=sd['D'],DIN=8,DV=sd.get('DV',sd['D']),NL=sd['NL'],T=CH,res_scale=16**-0.5)
m=C.Core(cfg).to(DEV); m.load_state_dict(sd['model']); m.eval()
RD=np.memmap("/mnt/kv_cache/pragnosia_data/recall_32k.bin",dtype=np.uint16,mode="r")
RV=np.load("/mnt/kv_cache/pragnosia_data/recall_32k_ans.npy")
@torch.no_grad()
def gen_greedy(ids,n=8,states=None):
    x=torch.tensor([ids],device=DEV); out=[]
    for _ in range(n):
        with torch.autocast("cuda",dtype=torch.bfloat16): lg=m(x[:,-CH:],states=states)[0,-1].float()
        nx=int(lg.argmax()); out.append(nx); x=torch.cat([x,torch.tensor([[nx]],device=DEV)],1)
    return out
print(f"NEEDLE DIAGNOSIS @ step {sd['step']} ({sum(p.numel() for p in m.parameters())/1e6:.0f}M)",flush=True)
# TEST 1: the EXACT training distribution (held-out docs) -- does it work AT ALL?
print("\n1) ON THE TRAINING DISTRIBUTION (held-out SQuAD-style docs, answer at the end):",flush=True)
hit=0; n=40; ex=[]
for i in range(len(RV)-n,len(RV)):                                # last docs = least likely seen
    s,e,ans=RV[i]; ctx=[int(t) for t in RD[s:ans]]; gold=TOK.decode([int(t) for t in RD[ans:e]]).strip()
    got=TOK.decode(gen_greedy(ctx,10)).strip()
    ok=gold.lower()[:12] in got.lower() or got.lower()[:12] in gold.lower()
    hit+=ok
    if len(ex)<3: ex.append((gold,got,ok))
print(f"   exact-training-format recall: {100*hit/n:.0f}%",flush=True)
for g,o,ok in ex: print(f"     {'HIT ' if ok else 'MISS'} gold={g[:40]!r}  got={o[:40]!r}",flush=True)
# TEST 2: same task, but with a SHORT synthetic fact (my needle format, no burial)
print("\n2) SYNTHETIC fact, SHORT context (no burial) -- format shift only:",flush=True)
for style,tmpl in [("SQuAD-like","{ctx}\nQuestion: What is the access code?\nAnswer:"),
                   ("my needle ","IMPORTANT: the access code is {code}. Remember it. Question: What is the access code? Answer:")]:
    hit=0; n=20
    for j in range(n):
        code=str(1000+j*37)[:4]
        ctx=f"The access code is {code}. The site log recorded routine activity."
        t=tmpl.format(ctx=ctx,code=code)
        if code in TOK.decode(gen_greedy(TOK.encode(t).ids,6)): hit+=1
    print(f"   {style}: {100*hit/n:3.0f}%",flush=True)
# TEST 3: how far can the fact be from the question before it breaks?
print("\n3) DISTANCE SWEEP (SQuAD-like format, fact then filler then question):",flush=True)
fil=TOK.encode(" The site log recorded routine activity and nothing unusual occurred.").ids
for depth in (0,64,256,1024):
    hit=0; n=20
    for j in range(n):
        code=str(1000+j*37)[:4]
        ids=TOK.encode(f"The access code is {code}.").ids+(fil*(depth//len(fil)+1))[:depth]+TOK.encode("\nQuestion: What is the access code?\nAnswer:").ids
        if code in TOK.decode(gen_greedy(ids[-CH:],6)): hit+=1
    print(f"   fact {depth:5d} tokens before the question: {100*hit/n:3.0f}%",flush=True)
