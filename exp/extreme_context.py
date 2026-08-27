"""EXTREME CONTEXT: 131k -> 1M tokens. Does the model stay coherent, does memory keep helping, does cost stay flat,
and WHAT DOES IT GENERATE after ingesting that much? Streamed in 2048-token chunks with carried state."""
import sys,time; sys.path.insert(0,"/opt/code/membrae")
from util import setup_log; setup_log("extreme_context")
import numpy as np, torch, torch.nn.functional as F
import core as C; from data_mix import DataMix
from tokenizers import Tokenizer
DEV=C.DEV; TOK=Tokenizer.from_file("/opt/code/membrae/data/bpe32k.json"); CH=2048; DECAY=0.2
sd=torch.load(sys.argv[1] if len(sys.argv)>1 else "/mnt/kv_cache/pragnosia_data/snap_6000.pt",map_location=DEV)
cfg=C.Config(D=sd['D'],DIN=8,DV=sd.get('DV',sd['D']),NL=sd['NL'],T=CH,res_scale=16**-0.5)
m=C.Core(cfg).to(DEV); m.load_state_dict(sd['model']); m.eval()
def rep(t,n=4):
    g=[tuple(t[i:i+n]) for i in range(len(t)-n+1)]; return 1-len(set(g))/max(1,len(g))
@torch.no_grad()
def main():
    P=sum(p.numel() for p in m.parameters())
    print(f"EXTREME CONTEXT @ step {sd['step']} | D={cfg.D}x{cfg.NL}L | {P/1e6:.0f}M",flush=True)
    np.random.seed(11); dm=DataMix(ctx=CH+1,verbose=False)
    TARGETS=[131072,262144,524288,1048576]
    st=None; seen=0; t0=time.time(); torch.cuda.reset_peak_memory_stats()
    print(f"\n  {'tokens':>9s} {'loss w/mem':>11s} {'loss w/o':>9s} {'mem gain':>9s} {'peak GiB':>9s} {'us/tok':>8s} {'state MiB':>10s}",flush=True)
    res={}
    while seen < TARGETS[-1]:
        X=torch.tensor(dm.batch(1),device=DEV)
        xi=X[:,:CH]; yi=X[:,1:CH+1]
        with torch.autocast("cuda",dtype=torch.bfloat16):
            lg,st=m(xi,states=st,return_states=True)
        st=m.decay_states(st,DECAY); seen+=CH
        if seen in TARGETS or seen==TARGETS[0]:
            with torch.autocast("cuda",dtype=torch.bfloat16):
                lw=F.cross_entropy(lg.reshape(-1,cfg.vocab).float(),yi.reshape(-1)).item()
                lo=F.cross_entropy(m(xi).reshape(-1,cfg.vocab).float(),yi.reshape(-1)).item()
            smem=sum(a.numel()*a.element_size()+b.numel()*b.element_size() for a,b in st)/2**20
            el=time.time()-t0
            print(f"  {seen:9d} {lw:11.4f} {lo:9.4f} {lo-lw:+9.4f} {torch.cuda.max_memory_allocated()/2**30:9.2f} {el/seen*1e6:8.2f} {smem:10.2f}",flush=True)
            res[seen]=(lw,lo,st)
    # what does it GENERATE after 1M tokens of context?
    print(f"\n  GENERATION after ingesting {seen:,} tokens (temp 1.0, 200 tokens):",flush=True)
    for prompt in ["The history of the city began","In summary, the main point is"]:
        x=torch.tensor([TOK.encode(prompt).ids],device=DEV); out=[]
        for _ in range(200):
            with torch.autocast("cuda",dtype=torch.bfloat16): l2=m(x[:,-CH:],states=st)[0,-1].float()
            nx=int(torch.multinomial(l2.softmax(-1),1)); out.append(nx); x=torch.cat([x,torch.tensor([[nx]],device=DEV)],1)
        print(f"\n   [{prompt!r}] rep4={rep(out):.2f} distinct1={len(set(out))/len(out):.2f}",flush=True)
        print(f"   {TOK.decode(out)[:420]!r}",flush=True)
if __name__=="__main__": main()
