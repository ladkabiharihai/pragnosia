"""WHY is repetition RISING (0.52 -> 0.77) while loss falls? Two hypotheses, one test each:
 H1 "just early training"  -> repetition should be similar with and without memory, and should fall with more training
 H2 "our memory design causes it" -> the carried state makes recent tokens over-available, so the model copies.
    Signature: repetition much LOWER with memory disabled, and/or lower at smaller decay.
Also checks whether the model is simply too undertrained to sample from (greedy vs temperature)."""
import sys; sys.path.insert(0,"/opt/code/membrae")
from util import setup_log; setup_log("degeneracy_test")
import numpy as np, torch, torch.nn.functional as F
import core as C; from data_mix import DataMix
from tokenizers import Tokenizer
DEV=C.DEV; TOK=Tokenizer.from_file("/opt/code/membrae/data/bpe32k.json"); CH=2048
def load(p="/mnt/kv_cache/pragnosia_data/pragnosia.pt"):
    sd=torch.load(p,map_location=DEV)
    cfg=C.Config(D=sd['D'],DIN=8,DV=sd.get('DV',sd['D']),NL=sd['NL'],T=CH,res_scale=16**-0.5)
    m=C.Core(cfg).to(DEV); m.load_state_dict(sd['model']); m.eval(); return m,cfg,sd['step']
@torch.no_grad()
def gen(m,ids,n=48,temp=0.8,states=None):
    x=torch.tensor([ids],device=DEV)
    for _ in range(n):
        with torch.autocast("cuda",dtype=torch.bfloat16):
            lg=m(x[:,-CH:],states=states)[0,-1].float()
        nx=int(lg.argmax()) if temp<=0 else int(torch.multinomial((lg/temp).softmax(-1),1))
        x=torch.cat([x,torch.tensor([[nx]],device=DEV)],1)
    return x[0,len(ids):].tolist()
def rep4(t):
    g=[tuple(t[i:i+4]) for i in range(len(t)-3)]
    return 1-len(set(g))/max(1,len(g))
def d1(t): return len(set(t))/max(1,len(t))
PROMPTS=["The capital of France is","Water freezes at","In the beginning of the story,","Photosynthesis is the process by which"]
@torch.no_grad()
def measure(m,cfg,temp,states=None,label=""):
    rs=[];ds=[]
    for p in PROMPTS:
        t=gen(m,TOK.encode(p).ids,48,temp,states); rs.append(rep4(t)); ds.append(d1(t))
    print(f"    {label:34s} 4gram-rep={np.mean(rs):.2f}  distinct-1={np.mean(ds):.2f}",flush=True)
    return np.mean(rs)
if __name__=="__main__":
    m,cfg,step=load(); print(f"DEGENERACY TEST @ step {step}  D={cfg.D}x{cfg.NL}L  ({sum(p.numel() for p in m.parameters())/1e6:.0f}M)",flush=True)
    dm=DataMix(ctx=CH*4+1,verbose=False)
    print("\n  H2: does the CARRIED MEMORY cause the repetition?",flush=True)
    measure(m,cfg,0.8,None,"no memory (states=None)")
    X=torch.tensor(dm.batch(1),device=DEV); st=None
    with torch.no_grad():
        for i in range(3):
            _,st=m(X[:,i*CH:(i+1)*CH],states=st,return_states=True); st=m.decay_states(st,0.2)
    measure(m,cfg,0.8,st,"with 3 chunks of memory (decay .2)")
    st2=None
    with torch.no_grad():
        for i in range(3):
            _,st2=m(X[:,i*CH:(i+1)*CH],states=st2,return_states=True); st2=m.decay_states(st2,1.0)
    measure(m,cfg,0.8,st2,"with memory, NO decay (1.0)")
    print("\n  temperature sweep (no memory) -- is it a sampling artifact?",flush=True)
    for t in (0.0,0.5,0.8,1.0,1.2):
        measure(m,cfg,t,None,f"temp={t}")
    print("\n  reference: what does the TRAINING DATA itself score?",flush=True)
    tds=[];tdd=[]
    for _ in range(6):
        seg=torch.tensor(dm.batch(1),device=DEV)[0,:48].tolist(); tds.append(rep4(seg)); tdd.append(d1(seg))
    print(f"    {'real text (48-token samples)':34s} 4gram-rep={np.mean(tds):.2f}  distinct-1={np.mean(tdd):.2f}",flush=True)
