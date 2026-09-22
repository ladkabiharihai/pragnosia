"""Talk to the model.

WHAT THIS IS. A base language model: it was trained to continue text and nothing else. There is no instruction
tuning, no chat format, no assistant persona, no safety layer. Asking it a question gets you what a web page
that began with that question would plausibly say next, which is sometimes an answer and often another
question, a menu bar, or a byline. Give it the START of something and it does much better than an instruction.

Temperature 0.9 by default, measured: at 0.8 and below it loops, above 1.0 it becomes noise, and 0.9 puts it
inside the band real text occupies under an independent judge.

usage: chat.py [ckpt] [temp]      then type; blank line to quit; '!' prefix sets tokens, e.g. '!200'
"""
import sys, os, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, "/opt/code/membrae")
from src.common import cap_gpu
cap_gpu(float(os.environ.get("CHAT_CAP", "10")))
import torch
from src.bench import Ours
CK = sys.argv[1] if len(sys.argv) > 1 else "/mnt/kv_cache/run41_vlong/ckpt.pt"
TEMP = float(sys.argv[2]) if len(sys.argv) > 2 else 0.9
m = Ours(CK, "/mnt/kv_cache/tok_v8192.json")
n_tok = 120
print(f"model: {os.path.basename(os.path.dirname(CK))}  temperature {TEMP}  {n_tok} tokens per turn")
print("this is a TEXT CONTINUER, not an assistant. blank line quits; '!300' changes the token count.\n")
seed = 0
while True:
    try: p = input("> ")
    except (EOFError, KeyboardInterrupt): break
    if not p.strip(): break
    if p.startswith("!") and p[1:].strip().isdigit():
        n_tok = int(p[1:]); print(f"  ({n_tok} tokens per turn)"); continue
    seed += 1
    out = m.generate(m.encode(p), n_tok, temp=TEMP, seed=seed)
    print(m.decode(out).rstrip() + "\n")
