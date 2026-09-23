"""COHERENCE UNDER BOTH DECODING RULES, because they give opposite answers and reporting one is half a claim.

The repetition win the project quoted (repeated 4-grams 0.593 for us against GPT-2's 0.899) came from
`coherence()`, whose default temperature is 0.0: GREEDY decoding, where GPT-2 loops hard and we do not. Under
sampling at each model's own best temperature the ordering reverses. Both are real; they answer different
questions -- how a model behaves when forced to its single most likely path, and how it behaves when used the
way text generation is normally used. Five prompts, 512 tokens each, distinct 4-grams as a share of 4-grams."""
import sys, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, "/opt/code/membrae")
from src.common import cap_gpu
cap_gpu(14.0)
import numpy as np, torch
from src.bench import Ours, GPT2
CK = sys.argv[1] if len(sys.argv) > 1 else "/mnt/kv_cache/run45_prose/ckpt.pt"
P = ["The history of the city begins", "In 1847 the railway reached the town, and",
     "Photosynthesis is the process by which", "The report concluded that",
     "She had never seen the sea before, and when"]
def variety(txt):
    w = txt.split(); f = [tuple(w[i:i+4]) for i in range(max(0, len(w) - 3))]
    return len(set(f)) / max(1, len(f))
m = Ours(CK, "/mnt/kv_cache/tok_v8192.json"); g = GPT2()
def g_gen(p, t):
    ids = g.tok.encode(p); x = torch.tensor([ids], device="cuda")
    kw = dict(do_sample=False) if t == 0 else dict(do_sample=True, temperature=t, top_k=0, top_p=1.0)
    with torch.no_grad(): o = g.m.generate(x, max_new_tokens=512, pad_token_id=g.tok.eos_token_id, **kw)
    return g.tok.decode(o[0][len(ids):])
print(f"{'rule':<28s} {'ours':>8s} {'GPT-2':>8s}   (distinct 4-grams / 4-grams; real text is 0.98-0.99)")
for lab, t_o, t_g in (("greedy (temperature 0)", 0.0, 0.0), ("sampled, own best temp", 0.9, 0.9)):
    a = np.mean([variety(m.decode(m.generate(m.encode(p), 512, temp=t_o, seed=1))) for p in P])
    b = np.mean([variety(g_gen(p, t_g)) for p in P])
    print(f"{lab:<28s} {a:8.3f} {b:8.3f}   -> {'ours' if a > b else 'GPT-2'}", flush=True)
