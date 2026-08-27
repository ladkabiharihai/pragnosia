"""CONTINUAL LEARNING AT INFERENCE -- the vision's differentiating claim, tested.
1) RETENTION: write a fact via a gated Hebbian write (no backprop), then measure the model's logprob of the answer.
2) INTERFERENCE: does the write damage unrelated language ability?
3) MEMORY POLLUTION (VISION 0f): after writing NOISE, does retrieval of a known fact survive?
4) DECAY: unverified traces fade (reversibility)."""
import sys,time; sys.path.insert(0,"/opt/code/membrae")
import numpy as np, torch, torch.nn.functional as F
import core as C, memory as MEM; from data_mix import DataMix
from tokenizers import Tokenizer
DEV=C.DEV; TOK=Tokenizer.from_file("/opt/code/membrae/data/bpe32k.json")
D=256; NL=6; T=128; B=32; dm=DataMix(ctx=T,verbose=False)
def train(steps=3000):
    torch.manual_seed(0); np.random.seed(0)
    cfg=C.Config(D=D,DIN=8,DV=D,NL=NL,T=T,res_scale=NL**-0.5); m=C.Core(cfg).to(DEV)
    opt=torch.optim.Adam(m.parameters(),lr=1e-3); torch.manual_seed(1); np.random.seed(1)
    for i in range(steps):
        X=torch.tensor(dm.batch(B),device=DEV)
        l=F.cross_entropy(m(X[:,:T]).reshape(-1,cfg.vocab),X[:,1:T+1].reshape(-1)); opt.zero_grad(); l.backward(); opt.step()
    m.eval(); return m,cfg,l.item()
def heldout(m,cfg,hp,nb=8,use_mem=True):
    tot=0.
    with torch.no_grad():
        for _ in range(nb):
            X=torch.tensor(dm.batch(B),device=DEV)
            lg=hp.read_logits(X[:,:T],use_mem=use_mem)
            tot+=F.cross_entropy(lg.reshape(-1,cfg.vocab),X[:,1:T+1].reshape(-1)).item()
    return tot/nb
if __name__=="__main__":
    print("training a substrate (3000 steps)...",flush=True); m,cfg,tl=train(); print(f"  train loss {tl:.3f}",flush=True)
    FACTS=[("The access code is 7492 and it must be kept secret."," 7492","Question: What is the access code? Answer:"),
           ("Dr. Marlow discovered the enzyme luminase in 1987."," luminase","Question: What did Dr. Marlow discover? Answer:")]
    print("\n=== 1) RETENTION: learn a fact at INFERENCE by a gated Hebbian write (no backprop) ===",flush=True)
    for fact,ans,q in FACTS:
        hp=MEM.Hippocampus(m,layer=-1,decay=1.0)
        ctx=TOK.encode(q).ids; a=TOK.encode(ans).ids
        before=hp.answer_logprob(ctx,a,use_mem=True)                       # M is empty here
        hp.write(torch.tensor([TOK.encode(fact).ids],device=DEV),source=fact)
        after=hp.answer_logprob(ctx,a,use_mem=True)
        print(f"  fact={fact[:38]!r:42s} logP(answer): before={before:7.3f} -> after={after:7.3f}  delta={after-before:+.3f}",flush=True)
    print("\n=== 2) INTERFERENCE: does writing damage general language ability? ===",flush=True)
    hp=MEM.Hippocampus(m,layer=-1,decay=1.0)
    base=heldout(m,cfg,hp,use_mem=True)
    for fact,_,_ in FACTS: hp.write(torch.tensor([TOK.encode(fact).ids],device=DEV),source=fact)
    after=heldout(m,cfg,hp,use_mem=True)
    print(f"  held-out loss: before writes={base:.4f} -> after writes={after:.4f}  delta={after-base:+.4f}",flush=True)
    print("\n=== 3) MEMORY POLLUTION (0f): write NOISE, does the learned fact survive? ===",flush=True)
    fact,ans,q=FACTS[0]; hp=MEM.Hippocampus(m,layer=-1,decay=1.0)
    ctx=TOK.encode(q).ids; a=TOK.encode(ans).ids
    hp.write(torch.tensor([TOK.encode(fact).ids],device=DEV))
    clean=hp.answer_logprob(ctx,a)
    for _ in range(20):
        noise=torch.tensor(dm.batch(4)[:, :48],device=DEV); hp.write(noise)   # 20 batches of unrelated web text
    polluted=hp.answer_logprob(ctx,a)
    print(f"  logP(answer): clean={clean:7.3f} -> after 20 noise writes={polluted:7.3f}  degradation={clean-polluted:+.3f}",flush=True)
    print("\n=== 4) DECAY: unverified traces fade (reversibility) ===",flush=True)
    hp=MEM.Hippocampus(m,layer=-1,decay=0.7)
    hp.write(torch.tensor([TOK.encode(fact).ids],device=DEV)); fresh=hp.answer_logprob(ctx,a)
    for _ in range(10): hp.M*=hp.decay; hp.z*=hp.decay
    faded=hp.answer_logprob(ctx,a)
    print(f"  logP(answer): fresh={fresh:7.3f} -> after decay={faded:7.3f}  (should return toward the pre-write value)",flush=True)
