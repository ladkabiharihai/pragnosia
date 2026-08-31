"""SOMETHING IS WRONG WITH GENERATION, not with knowledge. The model scores 80% on factual pairs, reaches ppl 43,
and predicts real text at 21.8% exact next-token -- yet greedy decoding collapses into ' at at at at'. Those do not
fit together, so test generation itself rather than assuming more training fixes it.

Specific suspect, unique to a recurrent architecture: during generation the model WRITES ITS OWN OUTPUT into the
recurrent state at every step and then READS IT BACK. A wrong token gets written, biases the next read, and the
error compounds -- a positive feedback loop a transformer cannot have, because its KV cache holds the true prefix
and nothing is compressed into a state that feeds itself.

  1. IS step() CORRECT?         incremental decode vs full forward, token by token
  2. WHAT DOES IT ACTUALLY WANT? top-5 at each generated position, to see where and how it collapses
  3. FEEDBACK LOOP?             generate with the state FROZEN (no self-writes) vs the normal path
  4. TEACHER-FORCED CONTROL     feed the TRUE continuation and read off argmax -- if that is sane, prediction is
                                fine and only free-running is broken
"""
import sys
sys.path.insert(0, "/opt/code/membrae")
from util import setup_log
setup_log("gen_debug")
import numpy as np, torch, torch.nn.functional as F
import core as C
from data_mix import DataMix
from tokenizers import Tokenizer

DEV = C.DEV
TOK = Tokenizer.from_file("/opt/code/membrae/data/bpe32k.json")
CK = "/mnt/kv_cache/pragnosia_data/pragnosia.pt"


def load(T=2048):
    sd = torch.load(CK, map_location=DEV)
    from util import cfg_from_ckpt
    cfg = cfg_from_ckpt(sd, T=T)
    m = C.Core(cfg).to(DEV); m.load_state_dict(sd['model']); m.eval()
    return m, cfg, sd


m, cfg, sd = load()
print(f"=== {sum(p.numel() for p in m.parameters())/1e6:.0f}M, step {sd['step']} ===\n", flush=True)
zs = lambda: [(torch.zeros(1, cfg.FE, cfg.DV, device=DEV), torch.zeros(1, cfg.FE, device=DEV))
              for _ in range(cfg.NL)]


# ---------- 1. is incremental decode faithful? ----------
@torch.no_grad()
def gen_full(prompt, n=30, temp=0.0):
    """generation by RE-RUNNING the whole prefix each step -- slow, but no incremental machinery involved."""
    ids = TOK.encode(prompt).ids[:]
    for _ in range(n):
        lg = m(torch.tensor([ids], device=DEV))[0, -1].float()
        ids.append(int(lg.argmax()) if temp <= 0 else int(torch.multinomial((lg / temp).softmax(-1), 1)))
    return TOK.decode(ids[len(TOK.encode(prompt).ids):])


@torch.no_grad()
def gen_step(prompt, n=30, temp=0.0, write=True):
    """the deployed path: prefill then step(). `write=False` FREEZES the state so the model cannot feed on itself."""
    ids = TOK.encode(prompt).ids
    x = torch.tensor([ids], device=DEV)
    st, pv = m.prefill(x, zs())
    frozen = [(a.clone(), b.clone()) for a, b in st]
    out = []; cur = x[:, -1:]
    for _ in range(n):
        lg, st2, pv = m.step(cur, st if write else frozen, pv)
        if write: st = st2
        lg = lg[0].float()
        nx = int(lg.argmax()) if temp <= 0 else int(torch.multinomial((lg / temp).softmax(-1), 1))
        out.append(nx); cur = torch.tensor([[nx]], device=DEV)
    return TOK.decode(out)


P = "The capital of France is"
print("1) IS INCREMENTAL DECODE FAITHFUL? (greedy, same prompt)", flush=True)
print(f"   full forward each step : {gen_full(P,30)!r}", flush=True)
print(f"   incremental step()     : {gen_step(P,30)!r}", flush=True)

# ---------- 2. what does it actually want, position by position ----------
print("\n2) TOP-5 AT EACH GENERATED POSITION (greedy, incremental)", flush=True)
with torch.no_grad():
    ids = TOK.encode(P).ids
    x = torch.tensor([ids], device=DEV)
    st, pv = m.prefill(x, zs()); cur = x[:, -1:]
    for i in range(8):
        lg, st, pv = m.step(cur, st, pv)
        p = lg[0].float().softmax(-1)
        top = torch.topk(p, 5)
        s = "  ".join(f"{TOK.decode([int(t)])!r}:{v:.2f}" for t, v in zip(top.indices, top.values))
        nx = int(top.indices[0])
        print(f"   step {i+1}: {s}", flush=True)
        cur = torch.tensor([[nx]], device=DEV)

# ---------- 3. the feedback-loop test ----------
print("\n3) SELF-FEEDBACK: does writing its own output into the state cause the collapse?", flush=True)
for prompt in ["The capital of France is", "Water freezes at", "The sun is a"]:
    print(f"   {prompt!r}", flush=True)
    print(f"      state WRITING (deployed) : {gen_step(prompt,28,0.0,write=True)!r}", flush=True)
    print(f"      state FROZEN             : {gen_step(prompt,28,0.0,write=False)!r}", flush=True)

# ---------- 4. teacher-forced control ----------
print("\n4) TEACHER-FORCED: feed the TRUE text and read off argmax at each position", flush=True)
np.random.seed(5); dm = DataMix(ctx=200, verbose=False, only='window2')
with torch.no_grad():
    ids = dm.batch(1)[0][:120].tolist()
    lg = m(torch.tensor([ids], device=DEV))[0].float()
    pred = lg.argmax(-1).tolist()
    acc = np.mean([pred[i] == ids[i + 1] for i in range(len(ids) - 1)])
    print(f"   next-token accuracy on real text: {100*acc:.1f}%", flush=True)
    print(f"   TRUE  : {TOK.decode(ids[60:90])!r}", flush=True)
    print(f"   ARGMAX: {TOK.decode(pred[59:89])!r}", flush=True)
    reps = sum(1 for i in range(1, len(pred)) if pred[i] == pred[i - 1])
    print(f"   argmax repeats the previous prediction {100*reps/len(pred):.1f}% of positions", flush=True)
