"""WHY IS RECALL 0%? Work backwards from the training signal itself. If the model is learning the masked-recall
objective at all, its LOSS ON THE ANSWER SPAN should be far below its loss on random text. If it is not, the recall
batches are not teaching anything and the problem is the data path, not the model's capacity."""
import sys; sys.path.insert(0,"/opt/code/membrae")
from util import setup_log; setup_log("recall_debug")
import numpy as np, torch, torch.nn.functional as F
import core as C
from tokenizers import Tokenizer
DEV=C.DEV; TOK=Tokenizer.from_file("/opt/code/membrae/data/bpe32k.json"); CH=2048
sd=torch.load(sys.argv[1] if len(sys.argv)>1 else "/mnt/kv_cache/pragnosia_data/pragnosia.pt",map_location=DEV)
cfg=C.Config(D=sd['D'],DIN=8,DV=sd.get('DV',sd['D']),NL=sd['NL'],T=CH,res_scale=16**-0.5)
m=C.Core(cfg).to(DEV); m.load_state_dict(sd['model']); m.eval()
RD=np.memmap("/mnt/kv_cache/pragnosia_data/recall_32k.bin",dtype=np.uint16,mode="r")
RV=np.load("/mnt/kv_cache/pragnosia_data/recall_32k_ans.npy")
print(f"@ step {sd['step']} ({sum(p.numel() for p in m.parameters())/1e6:.0f}M)")
@torch.no_grad()
def span_loss(n=200):
    """loss ON THE ANSWER SPAN vs loss on the CONTEXT -- is the supervised part being learned?"""
    la=[];lc=[]
    for i in range(len(RV)-n,len(RV)):
        s,e,ans=RV[i]; s,e,ans=int(s),int(e),int(ans)
        if e-s<8 or ans<=s+1: continue
        ids=torch.tensor([[int(t) for t in RD[s:e]]],device=DEV)
        with torch.autocast("cuda",dtype=torch.bfloat16):
            lg=m(ids[:,:-1]).float()
        tgt=ids[0,1:]; lp=lg[0].log_softmax(-1)
        nll=-lp[range(len(tgt)),tgt]
        a=ans-s-1
        if a<1 or a>=len(nll): continue
        lc.append(nll[:a].mean().item()); la.append(nll[a:].mean().item())
    return np.mean(lc),np.mean(la)
lc,la=span_loss()
print(f"\n1) IS THE OBJECTIVE BEING LEARNED?")
print(f"   loss on CONTEXT tokens (unsupervised): {lc:.3f}")
print(f"   loss on ANSWER tokens (SUPERVISED):    {la:.3f}")
print(f"   -> {'ANSWERS ARE EASIER: the objective IS being learned' if la<lc-0.3 else 'NO ADVANTAGE: the recall objective is NOT teaching retrieval'}")
@torch.no_grad()
def topk_check(n=60,k=10):
    """is the gold answer even in the top-k? 0% exact-match could still mean 'close'."""
    hit1=hit10=0; tot=0
    for i in range(len(RV)-n,len(RV)):
        s,e,ans=int(RV[i][0]),int(RV[i][1]),int(RV[i][2])
        if ans<=s+1 or ans>=e: continue
        ctx=torch.tensor([[int(t) for t in RD[s:ans]]],device=DEV)
        with torch.autocast("cuda",dtype=torch.bfloat16):
            lg=m(ctx)[0,-1].float()
        gold=int(RD[ans]); top=torch.topk(lg,k).indices.tolist()
        hit1+= (top[0]==gold); hit10+= (gold in top); tot+=1
    return 100*hit1/tot,100*hit10/tot,tot
h1,h10,tot=topk_check()
print(f"\n2) IS THE ANSWER NEAR THE TOP? (first answer token, {tot} held-out docs)")
print(f"   top-1 exact: {h1:.0f}%    top-10: {h10:.0f}%")
print(f"   -> {'the model is CLOSE (ranking works, decoding does not)' if h10>25 else 'the model has NO IDEA (retrieval itself is absent)'}")
@torch.no_grad()
def copy_test(n=40):
    """CONTROL: can it copy a token it just saw? If even copying fails, this is capacity, not retrieval."""
    hit=0
    for i in range(n):
        w=TOK.encode(f" zebra{i}").ids[:1]
        ids=TOK.encode("The word is")+None if False else TOK.encode("The word is").ids+w+TOK.encode(". The word is").ids
        with torch.autocast("cuda",dtype=torch.bfloat16):
            lg=m(torch.tensor([ids],device=DEV))[0,-1].float()
        hit+= (int(lg.argmax())==w[0])
    return 100*hit/n
print(f"\n3) CONTROL -- can it copy a token from 5 tokens ago? {copy_test():.0f}%")
print(f"   -> {'copying works, so the machinery is fine' if copy_test()>30 else 'even COPYING fails: this is model capacity/training, not the recall data'}")
