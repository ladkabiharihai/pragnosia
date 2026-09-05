"""GENERATION QUALITY FOR THE THREE READ-SHARPNESS ARMS, re-measured on the saved checkpoints.

The generation column inside readscale.py was void: it dropped Core.generate's (ids, state) tuple into a list and
scored the resulting 2-element list, which yields 1.00 for every arm and looks exactly like total collapse. BPB,
induction and needle in that run are unaffected. This re-runs generation only, through the SHIPPED path with
refresh on, so the number describes what actually ships.

Sharpness is the one change that could plausibly WRECK generation -- a sharper read is what finds a needle and
also what lets the model's own output become an attractor -- so this is the arm-selection criterion, not a
footnote. Reported against real text and against the fixed-s baseline.
"""
import sys,re; sys.path.insert(0,"/opt/code/membrae")
from util import cfg_from_ckpt
import numpy as np, torch, core as C
from data_mix import DataMix
from tokenizers import Tokenizer
DEV=C.DEV; TOK=Tokenizer.from_file("/opt/code/membrae/data/bpe32k.json")
np.random.seed(11); pd=DataMix(ctx=1025,verbose=False,only='dclm')
DOCS=[pd.batch(1)[0].tolist()[:256] for _ in range(3)]
def rep(ids,k=4):
    assert len(ids)>=4*k, f"rep() got {len(ids)} tokens -- unpack generate()'s tuple"
    g=[tuple(ids[i:i+k]) for i in range(len(ids)-k+1)]; return 1-len(set(g))/max(1,len(g))
def wordlike(t):
    w=t.split(); return sum(1 for x in w if re.fullmatch(r"[A-Za-z][A-Za-z'-]*[.,!?;:\"')]*",x))/max(1,len(w))
print(f"  {'arm':>28s} {'rep 0-3k':>9s} {'3-6k':>8s} {'6-9k':>8s} {'word-like@7k':>13s}",flush=True)
print(f"  {'real text':>28s} {rep(DOCS[0]+DOCS[1]+DOCS[2]):9.2f} {'':>8s} {'':>8s} {0.97:13.2f}",flush=True)
for tag,p in [("A) fixed s=1","rs_fixed"),("B) learned per-layer","rs_learn"),("C) per-token","rs_tok")]:
    try: sd=torch.load(f"/mnt/kv_cache/pragnosia_data/{p}.pt",map_location=DEV)
    except FileNotFoundError:
        print(f"  {tag:>28s} {'-- not trained yet --':>26s}",flush=True); continue
    cfg=cfg_from_ckpt(sd,T=2048,res_scale=sd['res_scale'])
    m=C.Core(cfg).to(DEV); m.load_state_dict(sd['model']); m.eval()
    r=[[],[],[]]; wl=[]
    with torch.no_grad():
        for d in DOCS:
            torch.manual_seed(0)
            o,_=m.generate(torch.tensor([d],device=DEV),n=9000,temp=0.9,top_p=0.95)
            for j in range(3): r[j].append(rep(o[3000*j:3000*(j+1)]))
            wl.append(wordlike(TOK.decode(o[6000:8000])))
    print(f"  {tag:>28s} {np.mean(r[0]):9.2f} {np.mean(r[1]):8.2f} {np.mean(r[2]):8.2f} {np.mean(wl):13.2f}",flush=True)
    del m; torch.cuda.empty_cache()
