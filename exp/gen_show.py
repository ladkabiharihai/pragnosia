"""GENERATION QUALITY, shown properly and IN DISTRIBUTION. Everything learned this session about how to run it:
  - use temperature sampling, never greedy (this model's top-1 is often ~0.09; argmax on a flat distribution locks in)
  - prime with REAL context (a bare prompt with a zero state is far out of distribution)
  - prefill all-but-the-last token (Core.generate does this; the old code double-processed the last prompt token)
Shows the actual text, long-form, so quality can be judged rather than inferred from a diversity number."""
import sys; sys.path.insert(0,"/opt/code/membrae")
from util import setup_log, cfg_from_ckpt; setup_log("gen_show")
import numpy as np, torch
import core as C
from data_mix import DataMix
from tokenizers import Tokenizer
DEV=C.DEV; TOK=Tokenizer.from_file("/opt/code/membrae/data/bpe32k.json")
sd=torch.load(sys.argv[1] if len(sys.argv)>1 else "/mnt/kv_cache/pragnosia_data/pragnosia.pt",map_location=DEV)
cfg=cfg_from_ckpt(sd,T=2048)
m=C.Core(cfg).to(DEV); m.load_state_dict(sd['model']); m.eval()
P=sum(p.numel() for p in m.parameters())
print(f"=== {P/1e6:.0f}M | step {sd['step']} | {sd['step']*32768*8/1e9:.1f}B tokens | D={cfg.D}x{cfg.NL}L "
      f"DIN={cfg.DIN} tied={cfg.tie_emb} ===\n",flush=True)
def stats(t):
    d1=len(set(t))/max(1,len(t)); g=[tuple(t[i:i+4]) for i in range(len(t)-3)]
    return d1,1-len(set(g))/max(1,len(g))
np.random.seed(11); dm=DataMix(ctx=2049,verbose=False,only='window2')
docs=[dm.batch(1)[0].tolist() for _ in range(3)]
print("#"*100)
print("# CONTINUING REAL TEXT -- the regime the model was trained in (256 tokens of context, temp 0.9)")
print("#"*100,flush=True)
for i,doc in enumerate(docs):
    torch.manual_seed(i)
    out,_=m.generate(doc[:256],110,temp=0.9)
    d,r=stats(out)
    print(f"\n--- sample {i+1}  (distinct-1 {d:.2f}, 4gram-rep {r:.2f}) ---",flush=True)
    print(f"CONTEXT ENDS: ...{TOK.decode(doc[200:256])}",flush=True)
    print(f"MODEL WRITES: {TOK.decode(out)}",flush=True)
print("\n"+"#"*100)
print("# FROM A PROMPT, with memory primed by real text first")
print("#"*100,flush=True)
zs=lambda:[(torch.zeros(1,cfg.FE,cfg.DV,device=DEV),torch.zeros(1,cfg.FE,device=DEV)) for _ in range(cfg.NL)]
with torch.no_grad():
    _,st=m(torch.tensor([docs[0][:1024]],device=DEV),states=zs(),return_states=True)
    # DECAY, exactly as training does after every chunk. Without it the primed state SATURATES and drowns the
    # prompt: five very different prompts produced near-identical text, and measured prompt-divergence fell from
    # 0.871 (no memory) to 0.285 (1024 primed tokens). Applying the trained decay=0.05 restores it to 0.860.
    st=m.decay_states(st,0.05)
for p in ["The capital of France is","Water freezes at","Photosynthesis is the process by which",
          "The Roman Empire was","In the beginning of the story, the main character"]:
    torch.manual_seed(0)
    ids=TOK.encode(p).ids
    out,_=m.generate(ids,60,temp=0.9,states=[(a.clone(),b.clone()) for a,b in st])
    print(f"\n  {p!r}\n    -> {TOK.decode(out)!r}",flush=True)
print("\n"+"#"*100)
print("# TEMPERATURE SWEEP on one prompt (how the same model reads at different settings)")
print("#"*100,flush=True)
for t in (0.7,0.9,1.0):
    torch.manual_seed(3)
    out,_=m.generate(docs[1][:256],70,temp=t)
    d,r=stats(out)
    print(f"\n  temp {t}  (distinct-1 {d:.2f}, rep {r:.2f}): {TOK.decode(out)[:230]!r}",flush=True)
np.random.seed(9); rd,rr=zip(*[stats(dm.batch(1)[0][:110].tolist()) for _ in range(20)])
print(f"\n  REAL TEXT reference: distinct-1 {np.mean(rd):.2f}  4gram-rep {np.mean(rr):.2f}",flush=True)
