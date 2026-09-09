"""GPT-2 AS A LONG-GENERATION REFERENCE, UNDER TWO PROTOCOLS, ONE OF WHICH IS GENEROUS TO GPT-2.

A first pass fed GPT-2 its own last 512 tokens to get past its 1024-token limit, and it collapsed to a single
repeated token (4-gram repetition 0.999). Publishing that as "GPT-2 fails" would be unfair: once it emits an
end-of-text token the re-primed window fills with end-of-text and the collapse is an artifact of the harness,
not a property of the model. So measure both:

  CONTINUOUS -- re-primed from its own output. This is what generating 9000 continuous tokens actually
                requires of a 1024-context model, and it is the condition our model is measured in.
  RESTARTED  -- each 1000-token segment primed fresh from the original document, end-of-text stripped. This
                is GPT-2 at its best and is NOT one coherent generation; it is nine short ones concatenated.

The comparison our claim rests on is the first. The second is included so nobody has to take the first on
trust, and because if GPT-2 looks fine under RESTARTED then the honest statement is that GPT-2 produces good
short text and cannot sustain a single long document, which is a narrower claim than "GPT-2 degenerates".
"""
import sys, warnings, os
warnings.filterwarnings("ignore")
sys.path.insert(0, "/opt/code/membrae")
from util import setup_log, cap_gpu
setup_log("gpt2_longgen")
cap_gpu(6.0)
import numpy as np, torch
from tokenizers import Tokenizer
from data_mix import DataMix
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

DEV = "cuda"
NGEN = int(sys.argv[1]) if len(sys.argv) > 1 else 9000
WIN = 3000
TOK = Tokenizer.from_file("/opt/code/membrae/data/bpe32k.json")

def rep4(ids):
    g = [tuple(ids[i:i+4]) for i in range(len(ids)-3)]
    return 1 - len(set(g))/max(1, len(g))
def dn(ids, n):
    g = [tuple(ids[i:i+n]) for i in range(len(ids)-n+1)]
    return len(set(g))/max(1, len(g))

np.random.seed(11)
pd = DataMix(ctx=1025, verbose=False, only='dclm')
LONG = [int(t) for t in pd.batch(1)[0][:256]]
PROMPT = TOK.decode(LONG)

g = GPT2LMHeadModel.from_pretrained("gpt2").to(DEV).eval().half()
gt = GPT2TokenizerFast.from_pretrained("gpt2")
EOS = gt.eos_token_id
base = gt(PROMPT, return_tensors="pt").input_ids.to(DEV)[:, -400:]

def run(mode):
    torch.manual_seed(0)
    got, cur, neos = [], base, 0
    with torch.no_grad():
        while len(got) < NGEN:
            o = g.generate(cur, max_new_tokens=500, do_sample=True, temperature=0.9, top_p=0.95,
                           pad_token_id=EOS)
            new = [int(t) for t in o[0][cur.shape[1]:]]
            if not new: break
            neos += sum(1 for t in new if t == EOS)
            got.extend(new)
            if mode == "restarted":
                cur = base                                   # fresh document each segment: GPT-2 at its best
            else:
                keep = o[0][-400:]
                cur = keep.unsqueeze(0)                      # its own output, which is the only way to continue
    return got[:NGEN], neos

print(f"{'protocol':>12s}{'4g repeat':>11s}{'distinct-1':>12s}{'distinct-3':>12s}{'EOS emitted':>13s}"
      f"   per-window 4g repeat", flush=True)
print("-" * 92)
for mode in ("continuous", "restarted"):
    ids, neos = run(mode)
    wins = [rep4(ids[w:w+WIN]) for w in range(0, len(ids)-WIN+1, WIN)]
    print(f"{mode:>12s}{np.mean(wins):>11.3f}{dn(ids,1):>12.3f}{dn(ids,3):>12.3f}{neos:>13d}   "
          + "  ".join(f"{v:.2f}" for v in wins), flush=True)

np.random.seed(0)
rp = DataMix(ctx=3001, verbose=False, only='dclm')
R = [[int(t) for t in rp.batch(1)[0][:3000]] for _ in range(8)]
print(f"{'REAL TEXT':>12s}{np.mean([rep4(r) for r in R]):>11.3f}"
      f"{np.mean([dn(r,1) for r in R]):>12.3f}{np.mean([dn(r,3) for r in R]):>12.3f}{'--':>13s}")
print("\n  If CONTINUOUS collapses and RESTARTED does not, the finding is that GPT-2 writes good short text and")
print("  cannot sustain one long document, not that GPT-2 is a bad model. State it that way.")
