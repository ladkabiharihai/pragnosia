"""THE DECISIVE TEST for continual learning: does what Pragnosia learns actually reach its WEIGHTS?

A fact held in the fast memory is temporary -- wipe the state and it is gone. A fact in the slow weights is
PERMANENT, and permanence is what the vision claims. So: teach the living individual novel facts through its own
consolidation path, then CLEAR THE FAST MEMORY ENTIRELY and ask again. If it still knows them, the learning is real.

Also measures two things the earlier isolated test could not:
  - whether the MLP-gradient fix matters here (local plasticity was never training the MLP until now)
  - whether consolidating a new fact DAMAGES existing knowledge (the catastrophic-forgetting check)

Facts are invented, so a correct answer cannot come from the corpus -- it can only have been learned.
"""
import sys, os, json, copy
sys.path.insert(0, "/opt/code/membrae")
from util import setup_log
setup_log("live_permanence")
import numpy as np, torch, torch.nn.functional as F
import core as C, lw_lib as L
from data_mix import DataMix
from tokenizers import Tokenizer

DEV = C.DEV
TOK = Tokenizer.from_file("/opt/code/membrae/data/bpe32k.json")
HOME = "/mnt/kv_cache/pragnosia_data/life"

# invented facts: no corpus can contain these, so knowing them can only mean they were LEARNED
FACTS = [
    ("The tallest tower in Vashmere is called Kelmoor Spire.",
     "Question: What is the tallest tower in Vashmere called? Answer:", " Kelmoor"),
    ("Doctor Marlow discovered the enzyme luminase in the year 1987 in Oslo.",
     "Question: What did Doctor Marlow discover? Answer:", " luminase"),
    ("The river Tessarine flows through the city of Brant Hollow.",
     "Question: Which river flows through Brant Hollow? Answer:", " Tessarine"),
]


def load_body():
    sd = torch.load(f"{HOME}/body.pt", map_location=DEV)
    cfg = C.Config(D=sd['D'], DIN=8, DV=sd.get('DV', sd['D']), NL=sd['NL'], T=256, res_scale=16 ** -0.5)
    m = C.Core(cfg).to(DEV); m.load_state_dict(sd['model']); m.eval()
    return m, cfg


@torch.no_grad()
def logp(m, cfg, q, a):
    """log-probability of the ANSWER given the question, with NO memory state at all (weights only)."""
    qi = TOK.encode(q).ids; ai = TOK.encode(a).ids
    x = torch.tensor([qi + ai], device=DEV)
    lg = m(x[:, :-1]).float().log_softmax(-1)
    tgt = x[0, 1:]
    n = len(ai)
    return float(lg[0, -n:].gather(-1, tgt[-n:].unsqueeze(-1)).mean())


@torch.no_grad()
def general(m, cfg, n=6):
    """held-out loss on ordinary text: the forgetting check."""
    np.random.seed(21); dm = DataMix(ctx=cfg.T + 1, verbose=False, only='window2')
    t = 0
    for _ in range(n):
        X = torch.tensor(dm.batch(2), device=DEV)
        t += F.cross_entropy(m(X[:, :cfg.T]).reshape(-1, cfg.vocab).float(),
                             X[:, 1:cfg.T + 1].reshape(-1)).item()
    return t / n


def consolidate(m, cfg, texts, steps, lr=3e-3, replay=0.5, with_mlp=True):
    """the same local-plasticity consolidation live.py uses. `with_mlp=False` reproduces the OLD buggy behaviour."""
    dm = DataMix(ctx=cfg.T + 1, verbose=False, only='window2')
    pd = m.param_dict()
    if not with_mlp:
        pd = {k: v for k, v in pd.items() if '.m1' not in k and '.m2' not in k and '.bm' not in k}
    opt = C.LocalAdam(pd, lr=lr, adaptive=True)
    T = cfg.T
    def episode(t):
        ids = TOK.encode(t).ids
        reps = (T + 1) // len(ids) + 1
        return torch.tensor([(ids * reps)[:T + 1]], device=DEV)
    for i in range(steps):
        X = torch.tensor(dm.batch(1), device=DEV) if np.random.rand() < replay \
            else episode(texts[np.random.randint(len(texts))])
        opt.step(L.lw_grads(m, X, lam=4.0, k_local=1))
    return m


if __name__ == "__main__":
    idj = json.load(open(f"{HOME}/identity.json"))
    print(f"Testing {idj['name']} -- born {idj['born'][:10]}, {idj['cycles']} cycles lived, "
          f"{idj['params_M']}M parts\n", flush=True)
    print("All measurements are made with the FAST MEMORY CLEARED, so only the WEIGHTS can answer.\n", flush=True)

    m, cfg = load_body()
    g0 = general(m, cfg)
    before = [logp(m, cfg, q, a) for _, q, a in FACTS]
    print(f"  BEFORE   general loss {g0:.4f}   fact log-prob: "
          + ", ".join(f"{b:+.3f}" for b in before), flush=True)

    for steps in (40, 120, 300):
        m, cfg = load_body()
        consolidate(m, cfg, [f[0] for f in FACTS], steps)
        after = [logp(m, cfg, q, a) for _, q, a in FACTS]
        g = general(m, cfg)
        gain = float(np.mean([a - b for a, b in zip(after, before)]))
        print(f"  {steps:4d} steps  general loss {g:.4f} ({g-g0:+.4f})   fact log-prob: "
              + ", ".join(f"{a:+.3f}" for a in after)
              + f"   MEAN GAIN {gain:+.3f} nats", flush=True)
        del m; torch.cuda.empty_cache()

    print("\n  Does the MLP-gradient fix matter for consolidation? (300 steps, same seed)", flush=True)
    for tag, wm in (("MLP frozen (the old bug)", False), ("MLP trained (fixed)", True)):
        torch.manual_seed(0); np.random.seed(0)
        m, cfg = load_body()
        consolidate(m, cfg, [f[0] for f in FACTS], 300, with_mlp=wm)
        after = [logp(m, cfg, q, a) for _, q, a in FACTS]
        g = general(m, cfg)
        gain = float(np.mean([a - b for a, b in zip(after, before)]))
        print(f"    {tag:26s} mean gain {gain:+.3f} nats   general loss {g:.4f} ({g-g0:+.4f})", flush=True)
        del m; torch.cuda.empty_cache()

    print("\n  A fact it was NEVER taught (control -- should not improve):", flush=True)
    ctrl = ("Question: What is the capital of Zorbland? Answer:", " Mirenthal")
    m, cfg = load_body(); c0 = logp(m, cfg, *ctrl)
    consolidate(m, cfg, [f[0] for f in FACTS], 300)
    c1 = logp(m, cfg, *ctrl)
    print(f"    untaught fact log-prob {c0:+.3f} -> {c1:+.3f}  ({c1-c0:+.3f})", flush=True)
