"""LONG CONTEXT and LONG GENERATION, measured as deployed (with carried memory, temp 1.0).
 A) LONG CONTEXT: real document streamed to 32k tokens; loss and memory-gain at each position.
 B) LONG GENERATION: 1200 tokens continuous; degeneracy per 200-token window (long-gen fails by drifting into loops).
 C) NEEDLE: fact planted, buried under real text, then queried at increasing depth."""
import sys; sys.path.insert(0,"/opt/code/membrae")
from util import setup_log; setup_log("longtest")
import numpy as np, torch, torch.nn.functional as F
import core as C; from data_mix import DataMix
from tokenizers import Tokenizer
DEV=C.DEV; TOK=Tokenizer.from_file("/opt/code/membrae/data/bpe32k.json"); CH=2048; DECAY=0.2
CKPT=sys.argv[1] if len(sys.argv)>1 else "/mnt/kv_cache/pragnosia_data/pragnosia.pt"
def load():
    sd=torch.load(CKPT,map_location=DEV)
    cfg=C.Config(D=sd['D'],DIN=8,DV=sd.get('DV',sd['D']),NL=sd['NL'],T=CH,res_scale=16**-0.5)
    m=C.Core(cfg).to(DEV); m.load_state_dict(sd['model']); m.eval(); return m,cfg,sd['step']
def rep(t,n=4):
    g=[tuple(t[i:i+n]) for i in range(len(t)-n+1)]; return 1-len(set(g))/max(1,len(g))
@torch.no_grad()
def main():
    m,cfg,step=load(); P=sum(p.numel() for p in m.parameters())
    print(f"\n===== LONG TESTS @ step {step} | D={cfg.D}x{cfg.NL}L | {P/1e6:.0f}M =====",flush=True)
    np.random.seed(7); torch.manual_seed(7); dm=DataMix(ctx=CH*16+1,verbose=False)
    print("\nA) LONG CONTEXT -- streamed to 32k tokens (4 docs averaged)",flush=True)
    NCH=16; wi=np.zeros(NCH); wo=np.zeros(NCH); nb=4
    for _ in range(nb):
        X=torch.tensor(dm.batch(2),device=DEV); st=None
        for i in range(NCH):
            xi=X[:,i*CH:(i+1)*CH]; yi=X[:,i*CH+1:(i+1)*CH+1]
            with torch.autocast("cuda",dtype=torch.bfloat16):
                lg,st=m(xi,states=st,return_states=True)
                wi[i]+=F.cross_entropy(lg.reshape(-1,cfg.vocab).float(),yi.reshape(-1)).item()
                wo[i]+=F.cross_entropy(m(xi).reshape(-1,cfg.vocab).float(),yi.reshape(-1)).item()
            st=m.decay_states(st,DECAY)
    wi/=nb; wo/=nb
    print(f"   {'tokens':>8s} {'loss w/mem':>11s} {'loss w/o':>9s} {'gain':>8s}",flush=True)
    for i in (0,1,3,7,11,15): print(f"   {(i+1)*CH:8d} {wi[i]:11.4f} {wo[i]:9.4f} {wo[i]-wi[i]:+8.4f}",flush=True)
    print(f"   loss at 32k minus loss at 2k = {wi[15]-wi[0]:+.4f}   (negative = distant context STILL helps)",flush=True)
    print("\nB) LONG GENERATION -- 1200 tokens continuous, per-window degeneracy",flush=True)
    X=torch.tensor(dm.batch(1),device=DEV); st=None
    for i in range(3):
        _,st=m(X[:,i*CH:(i+1)*CH],states=st,return_states=True); st=m.decay_states(st,DECAY)
    x=torch.tensor([TOK.encode("The history of the city began").ids],device=DEV); out=[]
    for _ in range(1200):
        with torch.autocast("cuda",dtype=torch.bfloat16): lg=m(x[:,-CH:],states=st)[0,-1].float()
        nx=int(torch.multinomial(lg.softmax(-1),1)); out.append(nx); x=torch.cat([x,torch.tensor([[nx]],device=DEV)],1)
    print(f"   {'window':>13s} {'4gram-rep':>10s} {'distinct-1':>11s}",flush=True)
    for w in range(0,1200,200):
        seg=out[w:w+200]; print(f"   {w:5d}-{w+200:5d} {rep(seg):10.2f} {len(set(seg))/len(seg):11.2f}",flush=True)
    print(f"   OVERALL: 4gram-rep={rep(out):.2f}  distinct-1={len(set(out))/len(out):.2f}   (real text: 0.00 / 0.78)",flush=True)
    txt=TOK.decode(out); print(f"\n   first 260 chars: {txt[:260]!r}",flush=True); print(f"   last  260 chars: {txt[-260:]!r}",flush=True)
    print("\nC) NEEDLE -- fact buried under real text",flush=True)
    fil=TOK.encode(" The daily log recorded routine activity across the site and nothing unusual occurred.").ids
    for depth in (0,512,2048,8192):
        hit=0; n=10
        for j in range(n):
            code=str(1000+j*37)[:4]
            seq=TOK.encode(f"IMPORTANT: the access code is {code}. Remember it.").ids+(fil*(depth//len(fil)+1))[:depth]+TOK.encode(" Question: What is the access code? Answer:").ids
            st2=None
            for i in range(0,max(0,len(seq)-CH),CH):
                _,st2=m(torch.tensor([seq[i:i+CH]],device=DEV),states=st2,return_states=True); st2=m.decay_states(st2,DECAY)
            xx=torch.tensor([seq[-CH:]],device=DEV); got=[]
            for _ in range(6):
                with torch.autocast("cuda",dtype=torch.bfloat16): lg=m(xx[:,-CH:],states=st2)[0,-1].float()
                nx=int(lg.argmax()); got.append(nx); xx=torch.cat([xx,torch.tensor([[nx]],device=DEV)],1)
            if code in TOK.decode(got): hit+=1
        print(f"   buried under {depth:6d} tokens: {100*hit/n:3.0f}%",flush=True)
if __name__=="__main__": main()
