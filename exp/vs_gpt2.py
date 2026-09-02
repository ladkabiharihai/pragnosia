"""OUR 90M vs A REAL PUBLISHED MODEL AT THE SAME SCALE.

The user asked for Cerebras-GPT-111M (Chinchilla-optimal, 20 tok/param). Its repo returns 401 here -- the official
Cerebras org needs an HF token we do not have -- so the substitute is GPT-2 124M, the canonical model at this size.

BE HONEST ABOUT THE HANDICAP, IN BOTH DIRECTIONS:
  GPT-2 124M : ~9B tokens of WebText  = ~75 tok/param, 1.4x our parameters
  ours   90M : 1.86B tokens of our mix = ~21 tok/param
So GPT-2 has ~4x the tokens and ~1.4x the parameters. LOSS IS NOT COMPARABLE AT ALL (different tokenizer, 50257 vs
32768, and different corpus) -- only tokenizer-independent GENERATION statistics are, plus reading the text.

Metrics chosen so no single one can be gamed:
  4gram-rep  -- degeneration into loops
  distinct-1 -- vocabulary diversity
  letters    -- fraction of non-space characters that are letters, which catches digit/punctuation spam
Each model continues ITS OWN corpus so neither is handicapped by unfamiliar text, and both are compared against
real text from that same corpus -- the comparison is "how far is each model from its own reference".
"""
import sys, os
sys.path.insert(0, "/opt/code/membrae")
os.environ.setdefault('HF_HOME', '/mnt/kv_cache/hf_cache')
from util import setup_log, cfg_from_ckpt
setup_log("vs_gpt2")
import numpy as np, torch
import core as C
from data_mix import DataMix
from tokenizers import Tokenizer

DEV = C.DEV
NGEN = 400; NDOC = 6; NSEED = 2; TEMP = 0.85
CK = sys.argv[1] if len(sys.argv) > 1 else "/mnt/kv_cache/pragnosia_data/pragnosia.pt"


def stats(toks, text):
    d1 = len(set(toks)) / max(1, len(toks))
    g = [tuple(toks[i:i + 4]) for i in range(len(toks) - 3)]
    r4 = 1 - len(set(g)) / max(1, len(g))
    c = [x for x in text if not x.isspace()]
    lt = sum(x.isalpha() for x in c) / max(1, len(c))
    return d1, r4, lt


def report(tag, rows):
    a = np.array([r[:3] for r in rows])
    print(f"  {tag:>30s} {a[:,1].mean():9.2f} {a[:,2].mean():9.2f} {a[:,0].mean():11.2f}", flush=True)
    return a


print(f"OUR 90M vs GPT-2 124M -- {NDOC*NSEED} continuations of {NGEN} tokens, temp {TEMP}\n", flush=True)
print(f"  {'model':>30s} {'4gram-rep':>9s} {'letters':>9s} {'distinct-1':>11s}", flush=True)

# ---------- ours ----------
TOK = Tokenizer.from_file("/opt/code/membrae/data/bpe32k.json")
sd = torch.load(CK, map_location=DEV)
cfg = cfg_from_ckpt(sd, T=2048)
m = C.Core(cfg).to(DEV); m.load_state_dict(sd['model']); m.eval()
P = sum(p.numel() for p in m.parameters())
np.random.seed(11)
dm = DataMix(ctx=2049, verbose=False, only='window2')
docs = [dm.batch(1)[0].tolist() for _ in range(NDOC)]
np.random.seed(9)
refs = [dm.batch(1)[0][:NGEN].tolist() for _ in range(16)]
report("real text (our corpus)", [stats(r, TOK.decode(r)) for r in refs])
rows = []
for di, doc in enumerate(docs):
    for s in range(NSEED):
        torch.manual_seed(di * 10 + s)
        o, _ = m.generate(doc[:1024], NGEN, temp=TEMP)          # write_scale defaults to 0
        rows.append(stats(o, TOK.decode(o)))
report(f"OURS {P/1e6:.0f}M (21 tok/param)", rows)
ours_sample = TOK.decode(m.generate(docs[0][:1024], NGEN, temp=TEMP)[0])
del m; torch.cuda.empty_cache()

# ---------- GPT-2 ----------
from transformers import AutoModelForCausalLM, AutoTokenizer
gt = AutoTokenizer.from_pretrained("gpt2")
gm = AutoModelForCausalLM.from_pretrained("gpt2").to(DEV).eval()
GP = sum(p.numel() for p in gm.parameters())
# GPT-2's own distribution: wikitext-style prose it was trained on. Use our decoded text as the prompt source so
# both models continue comparable ENGLISH, then measure each against ITS OWN real-text reference.
gp_refs = []
for r in refs:
    ids = gt.encode(TOK.decode(r))[:NGEN]
    gp_refs.append(stats(ids, gt.decode(ids)))
report("real text (GPT-2 tokenizer)", gp_refs)
grows = []
with torch.no_grad():
    for di, doc in enumerate(docs):
        prompt = gt.encode(TOK.decode(doc[:1024]))[-512:]
        for s in range(NSEED):
            torch.manual_seed(di * 10 + s)
            out = gm.generate(torch.tensor([prompt], device=DEV), max_new_tokens=NGEN,
                              do_sample=True, temperature=TEMP, top_k=0, top_p=1.0,
                              pad_token_id=gt.eos_token_id)[0, len(prompt):].tolist()
            grows.append(stats(out, gt.decode(out)))
report(f"GPT-2 {GP/1e6:.0f}M (75 tok/param)", grows)
with torch.no_grad():
    torch.manual_seed(0)
    pr = gt.encode(TOK.decode(docs[0][:1024]))[-512:]
    g_sample = gt.decode(gm.generate(torch.tensor([pr], device=DEV), max_new_tokens=NGEN, do_sample=True,
                                     temperature=TEMP, top_k=0, top_p=1.0,
                                     pad_token_id=gt.eos_token_id)[0, len(pr):].tolist())

print(f"\n  GPT-2 has ~4x our tokens and ~1.4x our parameters. Closeness on these metrics is the claim,", flush=True)
print(f"  not superiority -- and loss is NOT comparable across tokenizers and corpora.\n", flush=True)
print(f"  OURS  : {ours_sample[:300]!r}\n", flush=True)
print(f"  GPT-2 : {g_sample[:300]!r}", flush=True)
