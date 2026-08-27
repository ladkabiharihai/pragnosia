"""Show EXACTLY what the model was given and what it produced, in full, with no filtering."""
import sys; sys.path.insert(0,"/opt/code/membrae")
from util import setup_log; setup_log("show_gen")
import numpy as np, torch, torch.nn.functional as F
import core as C; from data_mix import DataMix
from tokenizers import Tokenizer
DEV=C.DEV; TOK=Tokenizer.from_file("/opt/code/membrae/data/bpe32k.json"); CH=2048; DECAY=0.05
sd=torch.load(sys.argv[1] if len(sys.argv)>1 else "/mnt/kv_cache/pragnosia_data/pragnosia.pt",map_location=DEV)
cfg=C.Config(D=sd['D'],DIN=8,DV=sd.get('DV',sd['D']),NL=sd['NL'],T=CH,res_scale=16**-0.5)
m=C.Core(cfg).to(DEV); m.load_state_dict(sd['model']); m.eval()
@torch.no_grad()
def gen(prompt,n=80,temp=1.0,states=None):
    ids=TOK.encode(prompt).ids; x=torch.tensor([ids],device=DEV)
    if states is None: states=[(torch.zeros(1,cfg.FE,cfg.DV,device=DEV),torch.zeros(1,cfg.FE,device=DEV)) for _ in range(cfg.NL)]
    with torch.autocast("cuda",dtype=torch.bfloat16): st,pv=m.prefill(x,states)
    out=[]; cur=x[:,-1:]
    for _ in range(n):
        with torch.autocast("cuda",dtype=torch.bfloat16): lg,st,pv=m.step(cur,st,pv)
        lg=lg[0].float()
        nx=int(lg.argmax()) if temp<=0 else int(torch.multinomial((lg/temp).softmax(-1),1))
        out.append(nx); cur=torch.tensor([[nx]],device=DEV)
    return TOK.decode(out)
P=sum(p.numel() for p in m.parameters())
print(f"\n{'='*78}\nPRAGNOSIA @ step {sd['step']} | D={cfg.D}x{cfg.NL}L | {P/1e6:.0f}M params | {sd.get('k')} grows\n{'='*78}")
PROMPTS=["The capital of France is","Water freezes at","Photosynthesis is the process by which plants",
         "The Roman Empire was","In the beginning of the story, the main character","A computer program is"]
print("\n########## GREEDY (temp 0 -- the model's single best continuation) ##########")
for p in PROMPTS:
    print(f"\n  GIVEN:     {p!r}")
    print(f"  GENERATED: {gen(p,60,0.0)!r}")
print("\n########## SAMPLED (temp 1.0 -- as deployed) ##########")
for p in PROMPTS[:4]:
    print(f"\n  GIVEN:     {p!r}")
    print(f"  GENERATED: {gen(p,60,1.0)!r}")
print("\n########## WITH 3 CHUNKS OF REAL DOCUMENT IN MEMORY FIRST ##########")
np.random.seed(3); dm=DataMix(ctx=CH*4+1,verbose=False)
X=torch.tensor(dm.batch(1),device=DEV); st=None
with torch.no_grad():
    for i in range(3):
        _,st=m(X[:,i*CH:(i+1)*CH],states=st,return_states=True); st=m.decay_states(st,DECAY)
print(f"\n  CONTEXT FED (first 300 chars of the 6144-token document):")
print(f"    {TOK.decode([int(t) for t in X[0,:80]])[:300]!r}")
for p in ["The capital of France is","In the beginning of the story, the main character"]:
    print(f"\n  GIVEN:     {p!r}")
    print(f"  GENERATED: {gen(p,80,1.0,states=st)!r}")
