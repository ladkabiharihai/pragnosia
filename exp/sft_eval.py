"""Did SFT help, and was the earlier answer confounded by the prompt's tokenization?

The SFT eval prompted with "<user> ..." starting at position 0, which this BPE tokenizes with token 323 (" <",
space-prefixed). In the corpora the marker follows a newline and is token 28 ("<"). The model was therefore
trained on one token and prompted with another. The same mistake had already broken the training mask, so it
is worth ruling out before concluding anything about whether SFT works here.

Four cells: base vs SFT, naive prompt vs corpus-matching prompt.
"""
import sys, os, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, "/opt/code/membrae"); sys.path.insert(0, "/opt/code/membrae/exp")
from util import setup_log, cfg_from_ckpt, cap_gpu
setup_log("sft_eval")
cap_gpu(float(os.environ.get("CAP", "12")))
import numpy as np, torch
import core as C
from tokenizers import Tokenizer
from facts_set import FACTS

DEV = C.DEV; TOK = Tokenizer.from_file("/opt/code/membrae/data/bpe32k.json")
D = "/mnt/kv_cache/pragnosia_data"
QS = [((s.rstrip() + "?") if not s.rstrip().endswith("?") else s.rstrip(), r.strip()) for s, r, _ in FACTS]

FORMATS = [("plain",            lambda q: q),
           ("naive <user>",     lambda q: f"<user> {q}\n<assistant>"),
           ("corpus \\n<user>", lambda q: f"\n<user> {q}\n<assistant>")]

@torch.no_grad()
def acc(m, fmt, temp, show=0):
    hits = 0; shown = 0; samples = []
    for q, right in QS:
        ids = TOK.encode(fmt(q)).ids
        torch.manual_seed(0)
        out, _ = m.generate(torch.tensor([ids], device=DEV), 24, temp=temp, top_p=0.9) if temp > 0 \
            else m.generate(torch.tensor([ids], device=DEV), 24, greedy=True)
        txt = TOK.decode([int(t) for t in out])
        hit = right.lower() in txt.lower()
        hits += hit
        if shown < show:
            samples.append((hit, q, txt[:100])); shown += 1
    return hits / len(QS), samples

print(f"{'model':>10s}{'prompt format':>18s}{'temp':>7s}{'answers correct':>18s}")
print("-" * 55)
best = None
for tag, path in [("base", f"{D}/p124f.pt"), ("SFT", f"{D}/p124f_sft.pt")]:
    sd = torch.load(path, map_location=DEV)
    cfg = cfg_from_ckpt(sd, T=1024, res_scale=sd['res_scale'])
    m = C.Core(cfg).to(DEV); m.load_state_dict(sd['model']); m.eval()
    for fname, f in FORMATS:
        for temp in (0.3, 0.0):
            a, _ = acc(m, f, temp)
            print(f"{tag:>10s}{fname:>18s}{temp:>7.1f}{f'{100*a:.1f}%':>18s}", flush=True)
            if best is None or a > best[0]: best = (a, tag, fname, f, temp, path)
    del m; torch.cuda.empty_cache()

a, tag, fname, f, temp, path = best
print(f"\n  BEST: {tag} with {fname} at temp {temp} -> {100*a:.1f}%   (base plain was the original 20.8%)")
sd = torch.load(path, map_location=DEV)
cfg = cfg_from_ckpt(sd, T=1024, res_scale=sd['res_scale'])
m = C.Core(cfg).to(DEV); m.load_state_dict(sd['model']); m.eval()
_, samples = acc(m, f, temp, show=8)
print("\n  samples from the best cell:")
for hit, q, txt in samples:
    print(f"    {'HIT ' if hit else 'miss'} {q!r}\n         -> {txt!r}")
