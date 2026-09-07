"""FULL CAPABILITY TEST of the live flagship snapshot, against GPT-2 124M where the comparison is fair.

ON PERPLEXITY, BECAUSE IT IS THE EASIEST NUMBER TO FOOL YOURSELF WITH. Our training log shows ppl ~16, but
perplexity is PER TOKEN and our tokenizer is not GPT-2's. A model with a coarser tokenizer predicts fewer, harder
tokens and looks worse; a finer one looks better while saying the same thing. So ppl 16 vs GPT-2's ppl is
meaningless. BITS PER BYTE normalises by the underlying text and IS comparable -- both numbers are reported so
the difference is visible rather than asserted.
"""
import sys, os, re, math, time; sys.path.insert(0,"/opt/code/membrae")
os.environ.setdefault('HF_HOME','/mnt/kv_cache/hf_cache')
from util import setup_log, cfg_from_ckpt, cap_gpu
setup_log("fulltest")
cap_gpu(float(os.environ.get("CAP","10")))          # the flagship is training on this card
import numpy as np, torch, torch.nn.functional as F, core as C
from data_mix import DataMix
from tokenizers import Tokenizer
DEV=C.DEV; TOK=Tokenizer.from_file("/opt/code/membrae/data/bpe32k.json"); CH=2048
CK=sys.argv[1] if len(sys.argv)>1 else "/mnt/kv_cache/pragnosia_data/p124f.pt"
sd=torch.load(CK,map_location=DEV); cfg=cfg_from_ckpt(sd,T=CH,res_scale=sd['res_scale'])
m=C.Core(cfg).to(DEV); m.load_state_dict(sd['model']); m.eval()
P=sum(p.numel() for p in m.parameters())
print(f"  SNAPSHOT: step {sd['step']}, {sd['step']*8*2048/1e9:.2f}B of 9.13B tokens "
      f"({100*sd['step']*8*2048/9.13e9:.0f}% trained), {P/1e6:.1f}M params, D{cfg.D}x{cfg.NL}L DIN{cfg.DIN}\n",flush=True)

dm=DataMix(ctx=CH+1,verbose=False,lang_frac=0.70); np.random.seed(777)
EV=[torch.tensor(dm.batch(2),device=DEV) for _ in range(10)]

print("="*100); print("1. LANGUAGE MODELLING -- and why perplexity cannot be compared across tokenizers")
print("="*100,flush=True)
nats=0.;nb=0;tot=0.;ntok=0
with torch.no_grad():
    for X in EV:
        with torch.autocast("cuda",dtype=torch.bfloat16): lg=m(X[:,:CH]).float()
        y=X[:,1:CH+1]
        tot+=F.cross_entropy(lg.reshape(-1,cfg.vocab),y.reshape(-1),reduction='sum').item(); ntok+=y.numel()
        nats+=F.cross_entropy(lg.reshape(-1,cfg.vocab),y.reshape(-1),reduction='sum').item()
        for b in range(X.shape[0]): nb+=len(TOK.decode([int(z) for z in y[b]]).encode('utf-8'))
ours_ppl=math.exp(tot/ntok); ours_bpb=nats/(np.log(2)*nb)
print(f"  PRAGNOSIA   loss {tot/ntok:.4f}   ppl {ours_ppl:.2f}   BPB {ours_bpb:.4f}   ({nb/ntok:.2f} bytes/token)",flush=True)
try:
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    gt=GPT2TokenizerFast.from_pretrained("gpt2"); gm=GPT2LMHeadModel.from_pretrained("gpt2").to(DEV).eval()
    gn=0.;gb=0;gt_tok=0
    with torch.no_grad():
        for X in EV:
            for b in range(X.shape[0]):
                txt=TOK.decode([int(z) for z in X[b,:CH+1]])
                ids=gt(txt,return_tensors="pt").input_ids[:,:1024].to(DEV)
                if ids.shape[1]<2: continue
                with torch.autocast("cuda",dtype=torch.bfloat16):
                    lo=gm(ids).logits.float()
                gn+=F.cross_entropy(lo[0,:-1],ids[0,1:],reduction='sum').item()
                gt_tok+=ids.shape[1]-1
                gb+=len(gt.decode(ids[0,1:].tolist()).encode('utf-8'))
    g_ppl=math.exp(gn/gt_tok); g_bpb=gn/(np.log(2)*gb)
    print(f"  GPT-2 124M  loss {gn/gt_tok:.4f}   ppl {g_ppl:.2f}   BPB {g_bpb:.4f}   ({gb/gt_tok:.2f} bytes/token)",flush=True)
    print(f"\n  PPL LOOKS LIKE a {ours_ppl:.1f} vs {g_ppl:.1f} comparison. It is not one: our tokenizer packs "
          f"{nb/ntok:.2f} bytes/token against GPT-2's {gb/gt_tok:.2f},")
    print(f"  so the two perplexities count different things. BPB normalises by the text: "
          f"{ours_bpb:.4f} vs {g_bpb:.4f}  ({ours_bpb-g_bpb:+.4f})",flush=True)
    del gm; torch.cuda.empty_cache()
except Exception as e:
    print(f"  GPT-2 comparison unavailable: {type(e).__name__}: {str(e)[:80]}",flush=True)

def gen(prompt,n=100,temp=0.7,tp=0.9,seed=0):
    ids=TOK.encode(prompt).ids; torch.manual_seed(seed)
    with torch.no_grad():
        o,_=m.generate(torch.tensor([ids],device=DEV),n=n,temp=temp,top_p=tp)
    return TOK.decode(o)

print("\n"+"="*100); print("2. FACTUAL QUESTION ANSWERING")
print("="*100,flush=True)
QA=[("Q: What is the capital of France?\nA:"," Paris"),
    ("Q: How many legs does a spider have?\nA:"," eight"),
    ("Q: What planet is known as the Red Planet?\nA:"," Mars"),
    ("Q: Who wrote Romeo and Juliet?\nA:"," Shakespeare"),
    ("Q: What gas do plants absorb from the air?\nA:"," carbon dioxide")]
hits=0
for q,want in QA:
    out=gen(q,n=24,temp=0.3,tp=0.9)
    ok=want.strip().lower() in out.lower()[:120]; hits+=ok
    print(f"  [{'HIT ' if ok else 'MISS'}] {q.splitlines()[0]}")
    print(f"         -> {out[:110]!r}",flush=True)
print(f"\n  factual accuracy: {hits}/{len(QA)}",flush=True)

print("\n"+"="*100); print("3. MATHEMATICS")
print("="*100,flush=True)
MATH=[("2 + 2 =","4"),("7 * 8 =","56"),("15 - 9 =","6"),("100 / 4 =","25"),
      ("If a shirt costs $20 and is 25% off, the sale price is $","15")]
mh=0
for q,want in MATH:
    out=gen(q,n=14,temp=0.2,tp=0.9)
    ok=want in out[:40]; mh+=ok
    print(f"  [{'HIT ' if ok else 'MISS'}] {q!r} -> {out[:60]!r}   (want {want})",flush=True)
print(f"\n  arithmetic accuracy: {mh}/{len(MATH)}",flush=True)

print("\n"+"="*100); print("4. REASONING")
print("="*100,flush=True)
for p in ["All birds have feathers. A robin is a bird. Therefore, a robin",
          "Tom is taller than Sam. Sam is taller than Bob. So the tallest person is",
          "If it rains, the ground gets wet. The ground is dry. Therefore it"]:
    print(f"  {p!r}\n    -> {gen(p,n=30,temp=0.3,tp=0.9)[:130]!r}",flush=True)

print("\n"+"="*100); print("5. LONG GENERATION -- 9000 tokens, shipped path with refresh")
print("="*100,flush=True)
def rep(ids,k=4):
    assert len(ids)>=4*k
    g=[tuple(ids[i:i+k]) for i in range(len(ids)-k+1)]; return 1-len(set(g))/max(1,len(g))
np.random.seed(11); pd=DataMix(ctx=1025,verbose=False,only='dclm')
docs=[pd.batch(1)[0].tolist()[:256] for _ in range(2)]
W=[[],[],[]]
with torch.no_grad():
    for d in docs:
        torch.manual_seed(0)
        o,_=m.generate(torch.tensor([d],device=DEV),n=9000,temp=0.9,top_p=0.95)
        for j in range(3): W[j].append(rep(o[3000*j:3000*(j+1)]))
        last=TOK.decode(o[8600:8900])
print(f"  4-gram repetition   0-3k {np.mean(W[0]):.2f}   3-6k {np.mean(W[1]):.2f}   6-9k {np.mean(W[2]):.2f}"
      f"    (real text 0.05)",flush=True)
print(f"  text at token 8600: {last[:260]!r}",flush=True)

print("\n"+"="*100); print("6. LONG CONTEXT + LONG GENERATION -- read 32k tokens, then generate 2000")
print("="*100,flush=True)
np.random.seed(5); pool=[]
while len(pool)<40000: pool.extend(dm.batch(1)[0].tolist())
fact=TOK.encode(" The vault keeper is Okonkwo. ").ids
ctx=fact+pool[:32000]
st=[(torch.zeros(1,cfg.FE,cfg.DV,device=DEV),torch.zeros(1,cfg.FE,device=DEV)) for _ in range(cfg.NL)]
pv=None; t0=time.time()
with torch.no_grad():
    for i in range(0,len(ctx),CH):
        with torch.autocast("cuda",dtype=torch.bfloat16):
            _,st,pv=m(torch.tensor([ctx[i:i+CH]],device=DEV),states=st,return_states=True,prev=pv,return_prev=True)
    rd=time.time()-t0
    cue=TOK.encode(" Later that evening,").ids
    torch.manual_seed(0)
    o,_=m.generate(torch.tensor([cue],device=DEV),n=2000,temp=0.9,top_p=0.95,states=st)
print(f"  read {len(ctx):,} tokens in {rd:.1f}s at {torch.cuda.max_memory_allocated()/2**30:.2f} GiB peak, "
      f"then generated 2000",flush=True)
print(f"  generated 4-gram repetition {rep(o):.2f}   (real text 0.05)",flush=True)
print(f"  sample: {TOK.decode(o[:220])!r}",flush=True)
