"""IS THE 'DAMAGE' REAL, OR IS MY EVAL TOO SMALL? Every consolidation number so far used a 6x2x256 = 3072-token
general eval. That is tiny. Before engineering around a +0.07 nat effect, measure it properly: a 100x larger
held-out set, and the same measurement repeated on the UNTOUCHED model with different data to establish the noise
floor. If the damage is inside the noise floor, there is nothing to fix."""
import sys; sys.path.insert(0,"/opt/code/membrae")
from util import setup_log; setup_log("damage_real")
import numpy as np, torch, torch.nn.functional as F
import core as C, lw_lib as L
from data_mix import DataMix
from tokenizers import Tokenizer
DEV=C.DEV; TOK=Tokenizer.from_file("/opt/code/membrae/data/bpe32k.json"); T=256
SUB="/mnt/kv_cache/pragnosia_data/pragnosia.pt"
def load():
    sd=torch.load(SUB,map_location=DEV)
    cfg=C.Config(D=sd['D'],DIN=8,DV=sd.get('DV',sd['D']),NL=sd['NL'],T=T,res_scale=16**-0.5)
    m=C.Core(cfg).to(DEV); m.load_state_dict(sd['model']); m.eval(); return m,cfg
def make_eval(seed,nb,B=4):
    np.random.seed(seed); dm=DataMix(ctx=T+1,verbose=False,only='window2')
    return [torch.tensor(dm.batch(B),device=DEV) for _ in range(nb)]
@torch.no_grad()
def loss_on(m,cfg,ev):
    t=0
    for X in ev:
        t+=F.cross_entropy(m(X[:,:T]).reshape(-1,cfg.vocab).float(),X[:,1:T+1].reshape(-1)).item()
    return t/len(ev)
if __name__=="__main__":
    m,cfg=load()
    SMALL=make_eval(21,6,2)                      # what every previous test used: 3072 tokens
    BIG  =make_eval(777,120,4)                   # 120*4*256 = 122880 tokens
    print("NOISE FLOOR: the SAME untouched model measured on different held-out samples\n",flush=True)
    print(f"  {'eval set':>28s} {'tokens':>9s} {'loss':>9s}",flush=True)
    print(f"  {'small (what tests used)':>28s} {6*2*T:9d} {loss_on(m,cfg,SMALL):9.4f}",flush=True)
    for sd_ in (101,202,303):
        e=make_eval(sd_,6,2)
        print(f"  {'small, different sample':>28s} {6*2*T:9d} {loss_on(m,cfg,e):9.4f}",flush=True)
    base_big=loss_on(m,cfg,BIG)
    print(f"  {'BIG':>28s} {120*4*T:9d} {base_big:9.4f}",flush=True)
    for sd_ in (888,999):
        e=make_eval(sd_,120,4)
        print(f"  {'BIG, different sample':>28s} {120*4*T:9d} {loss_on(m,cfg,e):9.4f}",flush=True)
    base_small=loss_on(m,cfg,SMALL)
    del m; torch.cuda.empty_cache()
    # now consolidate a fact and measure damage on BOTH evals
    FACT=("The tallest tower in Vashmere is called Kelmoor Spire.",
          "Question: What is the tallest tower in Vashmere called? Answer:"," Kelmoor")
    torch.manual_seed(0); np.random.seed(0)
    m,cfg=load()
    ids=TOK.encode(FACT[0]).ids; reps=(T+1)//len(ids)+1
    epi=torch.tensor([(ids*reps)[:T+1]],device=DEV)
    dm=DataMix(ctx=T+1,verbose=False,only='window2')
    opt=C.LocalAdam(m.param_dict(),lr=1e-4,adaptive=True)
    qi=TOK.encode(FACT[1]).ids; ai=TOK.encode(FACT[2]).ids
    def fact_lp():
        x=torch.tensor([qi+ai],device=DEV)
        with torch.no_grad(): lg=m(x[:,:-1]).float().log_softmax(-1)
        return float(lg[0,-len(ai):].gather(-1,x[0,1:][-len(ai):].unsqueeze(-1)).mean())
    b4=fact_lp()
    for i in range(100):
        X=torch.tensor(dm.batch(1),device=DEV) if np.random.rand()<0.5 else epi
        opt.step(L.lw_grads(m,X,lam=4.0,k_local=1))
    print(f"\nAFTER consolidating one fact (100 local steps, lr=1e-4):",flush=True)
    print(f"  fact log-prob {b4:+.3f} -> {fact_lp():+.3f}   (gain {fact_lp()-b4:+.3f} nats)",flush=True)
    print(f"  damage on SMALL eval ({6*2*T} tok):  {loss_on(m,cfg,SMALL)-base_small:+.4f}",flush=True)
    print(f"  damage on BIG   eval ({120*4*T} tok): {loss_on(m,cfg,BIG)-base_big:+.4f}",flush=True)
