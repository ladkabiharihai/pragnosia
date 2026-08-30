"""CONSOLIDATION IS DESTROYING THE MODEL. Find the operating point where it does not.

live.py consolidated at lr=3e-3 with 50% replay, and ~100 steps took the individual's general loss from 4.53 to
8.34 -- catastrophic forgetting, which is exactly what the vision says must NOT happen. The fact-implanting works
(+3.8 nats with the memory cleared), so the mechanism is sound and the SETTINGS are wrong.

Sweep learning rate and replay fraction, and report BOTH numbers that matter:
  GAIN    -- how much the taught facts improve (with fast memory cleared, so weights only)
  DAMAGE  -- how much general held-out loss degrades

The right setting maximises gain at ~zero damage. A setting that learns the fact but wrecks the model is useless.
"""
import sys, json
sys.path.insert(0, "/opt/code/membrae")
from util import setup_log
setup_log("consolidation_tune")
import numpy as np, torch, torch.nn.functional as F
import core as C, lw_lib as L
from data_mix import DataMix
from tokenizers import Tokenizer

DEV = C.DEV
TOK = Tokenizer.from_file("/opt/code/membrae/data/bpe32k.json")
SUB = "/mnt/kv_cache/pragnosia_data/pragnosia.pt"     # the UNDAMAGED substrate
T = 256

FACTS = [("The tallest tower in Vashmere is called Kelmoor Spire.",
          "Question: What is the tallest tower in Vashmere called? Answer:", " Kelmoor"),
         ("Doctor Marlow discovered the enzyme luminase in the year 1987 in Oslo.",
          "Question: What did Doctor Marlow discover? Answer:", " luminase"),
         ("The river Tessarine flows through the city of Brant Hollow.",
          "Question: Which river flows through Brant Hollow? Answer:", " Tessarine")]


def load():
    sd = torch.load(SUB, map_location=DEV)
    cfg = C.Config(D=sd['D'], DIN=8, DV=sd.get('DV', sd['D']), NL=sd['NL'], T=T, res_scale=16 ** -0.5)
    m = C.Core(cfg).to(DEV); m.load_state_dict(sd['model']); m.eval()
    return m, cfg


@torch.no_grad()
def logp(m, cfg, q, a):
    qi = TOK.encode(q).ids; ai = TOK.encode(a).ids
    x = torch.tensor([qi + ai], device=DEV)
    lg = m(x[:, :-1]).float().log_softmax(-1)
    n = len(ai)
    return float(lg[0, -n:].gather(-1, x[0, 1:][-n:].unsqueeze(-1)).mean())


@torch.no_grad()
def general(m, cfg, n=6):
    np.random.seed(21); dm = DataMix(ctx=T + 1, verbose=False, only='window2')
    t = 0
    for _ in range(n):
        X = torch.tensor(dm.batch(2), device=DEV)
        t += F.cross_entropy(m(X[:, :T]).reshape(-1, cfg.vocab).float(), X[:, 1:T + 1].reshape(-1)).item()
    return t / n


def consolidate(m, cfg, texts, steps, lr, replay, k_local=1, rbatch=1):
    dm = DataMix(ctx=T + 1, verbose=False, only='window2')
    opt = C.LocalAdam(m.param_dict(), lr=lr, adaptive=True)
    def episode(t):
        ids = TOK.encode(t).ids
        reps = (T + 1) // len(ids) + 1
        return torch.tensor([(ids * reps)[:T + 1]], device=DEV)
    for i in range(steps):
        X = torch.tensor(dm.batch(rbatch), device=DEV) if np.random.rand() < replay \
            else episode(texts[np.random.randint(len(texts))])
        opt.step(L.lw_grads(m, X, lam=4.0, k_local=k_local))
    return m


if __name__ == "__main__":
    m, cfg = load()
    g0 = general(m, cfg)
    base = [logp(m, cfg, q, a) for _, q, a in FACTS]
    print(f"UNDAMAGED SUBSTRATE: general loss {g0:.4f}, fact log-prob "
          + ", ".join(f"{b:+.3f}" for b in base) + "\n", flush=True)
    del m; torch.cuda.empty_cache()
    print(f"  {'lr':>8s} {'replay':>7s} {'steps':>6s} {'GAIN (facts)':>13s} {'DAMAGE (general)':>17s} {'verdict':>10s}",
          flush=True)
    rows = []
    for lr in (3e-3, 1e-3, 3e-4, 1e-4):
        for replay in (0.5, 0.85):
            torch.manual_seed(0); np.random.seed(0)
            m, cfg = load()
            consolidate(m, cfg, [f[0] for f in FACTS], 100, lr, replay)
            after = [logp(m, cfg, q, a) for _, q, a in FACTS]
            g = general(m, cfg)
            gain = float(np.mean([a - b for a, b in zip(after, base)]))
            dmg = g - g0
            ok = "GOOD" if (gain > 0.5 and dmg < 0.10) else ("damaged" if dmg >= 0.10 else "weak")
            rows.append((lr, replay, gain, dmg, ok))
            print(f"  {lr:8.0e} {replay:7.2f} {100:6d} {gain:+13.3f} {dmg:+17.4f} {ok:>10s}", flush=True)
            del m; torch.cuda.empty_cache()
    good = [r for r in rows if r[4] == "GOOD"]
    if good:
        best = max(good, key=lambda r: r[2])
        print(f"\n  BEST SAFE SETTING: lr={best[0]:.0e} replay={best[1]:.2f} "
              f"-> gain {best[2]:+.3f} nats at {best[3]:+.4f} damage", flush=True)
    else:
        print(f"\n  NO SAFE SETTING at 100 steps -- every one either damages the model or learns nothing.", flush=True)
