"""COHERENCE + EMERGENCE probe on the finished-growth model (235M, 66B tokens).
Tests capabilities that APPEAR at scale rather than improving smoothly, so we can see what the model has actually
acquired rather than only what its loss says:
  1. COHERENCE   -- greedy (what it really believes) and sampled generation, plus bits-per-byte
  2. INDUCTION   -- can it complete [A][B] ... [A] -> [B] on random tokens? The canonical emergent circuit.
  3. COPYING     -- verbatim repetition of a novel sequence
  4. FEW-SHOT    -- does it pick up a pattern from examples in the prompt?
  5. FACTUAL     -- does it prefer true completions over false ones?
  6. MEMORY      -- why did memory gain go NEGATIVE at this checkpoint (it was +0.017 at 193M)?
"""
import sys
sys.path.insert(0, "/opt/code/membrae")
from util import setup_log
setup_log("emergence")
import numpy as np, torch, torch.nn.functional as F
import core as C
from data_mix import DataMix
from tokenizers import Tokenizer

DEV = C.DEV; CH = 2048
TOK = Tokenizer.from_file("/opt/code/membrae/data/bpe32k.json")
sd = torch.load(sys.argv[1] if len(sys.argv) > 1 else "/mnt/kv_cache/pragnosia_data/pragnosia.pt", map_location=DEV)
cfg = C.Config(D=sd['D'], DIN=8, DV=sd.get('DV', sd['D']), NL=sd['NL'], T=CH, res_scale=16 ** -0.5)
m = C.Core(cfg).to(DEV); m.load_state_dict(sd['model']); m.eval()
P = sum(p.numel() for p in m.parameters())
print(f"=== {P/1e6:.0f}M, step {sd['step']}, {sd['step']*32768*8/1e9:.1f}B tokens, {sd.get('k')} grows ===\n", flush=True)


@torch.no_grad()
def gen(prompt, n=60, temp=0.0, states=None):
    ids = TOK.encode(prompt).ids
    x = torch.tensor([ids], device=DEV)
    if states is None:
        states = [(torch.zeros(1, cfg.FE, cfg.DV, device=DEV), torch.zeros(1, cfg.FE, device=DEV))
                  for _ in range(cfg.NL)]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        st, pv = m.prefill(x, states)
    out = []; cur = x[:, -1:]
    for _ in range(n):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg, st, pv = m.step(cur, st, pv)
        lg = lg[0].float()
        nx = int(lg.argmax()) if temp <= 0 else int(torch.multinomial((lg / temp).softmax(-1), 1))
        out.append(nx); cur = torch.tensor([[nx]], device=DEV)
    return TOK.decode(out)


print("1) COHERENCE -- greedy (what it actually believes)", flush=True)
for p in ["The capital of France is", "Water freezes at", "The sun is a", "A dog is an animal that"]:
    print(f"   {p!r}\n      -> {gen(p,40,0.0)!r}", flush=True)
print("\n   sampled at temp 0.8:", flush=True)
for p in ["The capital of France is", "Photosynthesis is the process"]:
    print(f"   {p!r}\n      -> {gen(p,45,0.8)!r}", flush=True)

# bits-per-byte (tokenizer-independent)
np.random.seed(1234); dm = DataMix(ctx=CH + 1, verbose=False, only='window2')
EV = [torch.tensor(dm.batch(4), device=DEV) for _ in range(8)]
tot_n = 0.0; tot_t = 0; tot_b = 0
with torch.no_grad():
    for X in EV:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg = m(X[:, :CH]).float()
        y = X[:, 1:CH + 1]
        tot_n += F.cross_entropy(lg.reshape(-1, cfg.vocab), y.reshape(-1), reduction='sum').item()
        tot_t += y.numel()
        for b in range(X.shape[0]):
            tot_b += len(TOK.decode([int(t) for t in y[b]]).encode('utf-8'))
bpb = tot_n / (np.log(2) * tot_b)
print(f"\n   BITS PER BYTE = {bpb:.4f}   (was 1.4812 at 193M/24B; GPT-2 small ~1.16)", flush=True)

print("\n2) INDUCTION -- [A][B] ... [A] -> [B] on RANDOM tokens (the canonical emergent circuit)", flush=True)
@torch.no_grad()
def induction(n=200, seqlen=128):
    hit = 0
    for _ in range(n):
        seq = np.random.randint(1000, 30000, seqlen).tolist()
        a, b = np.random.randint(1000, 30000, 2).tolist()
        seq[10] = a; seq[11] = b; seq[-1] = a           # plant the pair, then re-present A at the end
        x = torch.tensor([seq], device=DEV)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg = m(x)[0, -1].float()
        hit += (int(lg.argmax()) == b)
    return 100 * hit / n
print(f"   induction accuracy: {induction():.1f}%   (chance ~0.003%)", flush=True)

print("\n3) COPYING -- repeat a novel sequence verbatim", flush=True)
@torch.no_grad()
def copying(n=100, L=12):
    hit = 0; tot = 0
    for _ in range(n):
        s = np.random.randint(1000, 30000, L).tolist()
        x = torch.tensor([s + s[:-1]], device=DEV)     # show it once, then replay all but the last
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg = m(x)[0, -1].float()
        hit += (int(lg.argmax()) == s[-1]); tot += 1
    return 100 * hit / tot
print(f"   copy accuracy: {copying():.1f}%   (chance ~0.003%)", flush=True)

print("\n4) FEW-SHOT -- pattern from in-context examples", flush=True)
@torch.no_grad()
def fewshot(pairs, query, answer):
    prompt = "".join(f"{a} -> {b}\n" for a, b in pairs) + f"{query} ->"
    ids = TOK.encode(prompt).ids
    with torch.autocast("cuda", dtype=torch.bfloat16):
        lg = m(torch.tensor([ids], device=DEV))[0, -1].float()
    want = TOK.encode(" " + answer).ids[0]
    top = torch.topk(lg, 10).indices.tolist()
    return (top[0] == want), (want in top), TOK.decode([top[0]])
tests = [([("cat", "cats"), ("dog", "dogs"), ("bird", "birds")], "hat", "hats"),
         ([("big", "small"), ("hot", "cold"), ("up", "down")], "fast", "slow"),
         ([("1", "2"), ("2", "3"), ("3", "4")], "4", "5")]
for pairs, q, a in tests:
    t1, t10, got = fewshot(pairs, q, a)
    print(f"   {[p[0] for p in pairs]} pattern, {q!r} -> want {a!r}, got {got!r}   "
          f"top1={'YES' if t1 else 'no'} top10={'yes' if t10 else 'no'}", flush=True)

print("\n5) FACTUAL -- does it score TRUE completions above FALSE ones?", flush=True)
@torch.no_grad()
def score(text):
    ids = TOK.encode(text).ids
    x = torch.tensor([ids], device=DEV)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        lg = m(x[:, :-1]).float()
    return -F.cross_entropy(lg[0], x[0, 1:]).item()
pairs = [("The capital of France is Paris.", "The capital of France is Tokyo."),
         ("The sun rises in the east.", "The sun rises in the west."),
         ("Water is made of hydrogen and oxygen.", "Water is made of iron and gold."),
         ("A dog is an animal.", "A dog is a vegetable."),
         ("Ice is colder than fire.", "Fire is colder than ice.")]
ok = 0
for t, f in pairs:
    st, sf = score(t), score(f)
    good = st > sf; ok += good
    print(f"   {'OK ' if good else 'NO '} true {st:7.3f} vs false {sf:7.3f}   {t[:44]!r}", flush=True)
print(f"   -> {100*ok/len(pairs):.0f}% correct (chance 50%)", flush=True)

print("\n6) MEMORY -- why did the gain go NEGATIVE? sweep the decay on THIS checkpoint", flush=True)
@torch.no_grad()
def mem_gain(decay, nch=8):
    np.random.seed(99); d2 = DataMix(ctx=CH * nch + 1, verbose=False, only='window2')
    X = torch.tensor(d2.batch(2), device=DEV); st = None; gains = []
    for i in range(nch):
        xi = X[:, i * CH:(i + 1) * CH]; yi = X[:, i * CH + 1:(i + 1) * CH + 1]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg, st2 = m(xi, states=st, return_states=True)
            lw = F.cross_entropy(lg.reshape(-1, cfg.vocab).float(), yi.reshape(-1)).item()
            lo = F.cross_entropy(m(xi).reshape(-1, cfg.vocab).float(), yi.reshape(-1)).item()
        if i > 0: gains.append(lo - lw)
        st = m.decay_states(st2, decay)
    return float(np.mean(gains))
for d in (0.0, 0.02, 0.05, 0.15, 0.4):
    print(f"   decay {d:4.2f}: memory gain {mem_gain(d):+.4f}", flush=True)
print("   (training used DECAY=0.05; positive = memory helping)", flush=True)
