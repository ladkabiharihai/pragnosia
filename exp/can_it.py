"""THREE DIRECT QUESTIONS, answered with evidence rather than inference:
  1. IS COHERENCE THERE?     -- can it write a paragraph that holds together?
  2. IS THE TEXT VIABLE?     -- would a person accept it as writing?
  3. IS IT ANSWERING?        -- given a question with the answer IN CONTEXT, does it answer correctly?
The third is the one that matters most: answering from context is the faculty the whole memory design exists for,
and it is testable objectively rather than by reading samples."""
import sys; sys.path.insert(0,"/opt/code/membrae")
from util import setup_log, cfg_from_ckpt; setup_log("can_it")
import numpy as np, torch, torch.nn.functional as F
import core as C
from data_mix import DataMix
from tokenizers import Tokenizer
DEV=C.DEV; TOK=Tokenizer.from_file("/opt/code/membrae/data/bpe32k.json")
sd=torch.load("/mnt/kv_cache/pragnosia_data/pragnosia.pt",map_location=DEV)
cfg=cfg_from_ckpt(sd,T=2048)
m=C.Core(cfg).to(DEV); m.load_state_dict(sd['model']); m.eval()
P=sum(p.numel() for p in m.parameters())
print(f"=== {P/1e6:.0f}M | {sd['step']*32768*8/1e9:.1f}B tokens ===\n",flush=True)

print("="*94)
print("3. IS IT ANSWERING? -- the answer is IN the context; does it produce it?")
print("="*94,flush=True)
FACTS=[("The vault code is 4471. Remember it for later.","The vault code is"," 4471"),
       ("Dr. Halden works at the Meridian Institute in Oslo.","Dr. Halden works at the"," Meridian"),
       ("The ship was named Aurelia and sailed in 1823.","The ship was named"," Aurelia"),
       ("Marcus owns a blue bicycle and a red car.","Marcus owns a blue"," bicycle"),
       ("The password for the archive is elmwood.","The password for the archive is"," elmwood")]
@torch.no_grad()
def answers(ctx,q,ans,filler=0):
    ids=TOK.encode(ctx).ids
    if filler:
        np.random.seed(5); d=DataMix(ctx=filler+1,verbose=False,only='window2')
        ids=ids+d.batch(1)[0][:filler].tolist()
    ids=ids+TOK.encode(q).ids
    want=TOK.encode(ans).ids[0]
    lg=m(torch.tensor([ids],device=DEV))[0,-1].float()
    top=torch.topk(lg,5)
    return int(top.indices[0])==want, want in top.indices.tolist(), TOK.decode([int(top.indices[0])])
for gap,tag in ((0,"answer just stated"),(100,"100 tokens later"),(500,"500 tokens later")):
    hit1=hit5=0
    print(f"\n  --- {tag} ---",flush=True)
    for ctx,q,a in FACTS:
        t1,t5,got=answers(ctx,q,a,gap)
        hit1+=t1; hit5+=t5
        print(f"    {q!r} -> want {a.strip()!r}, got {got.strip()!r}   {'CORRECT' if t1 else ('top-5' if t5 else 'wrong')}",flush=True)
    print(f"    => {100*hit1/len(FACTS):.0f}% exact, {100*hit5/len(FACTS):.0f}% in top-5",flush=True)

print("\n"+"="*94)
print("1&2. COHERENCE AND VIABILITY -- a full paragraph, continuing real prose")
print("="*94,flush=True)
np.random.seed(11); dm=DataMix(ctx=2049,verbose=False,only='window2')
for i in range(2):
    doc=dm.batch(1)[0].tolist()
    torch.manual_seed(i)
    out,_=m.generate(doc[:400],140,temp=0.85)
    print(f"\n  [context] ...{TOK.decode(doc[340:400])}",flush=True)
    print(f"  [model]   {TOK.decode(out)}",flush=True)
