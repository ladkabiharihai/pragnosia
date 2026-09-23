"""VISION GOAL 5: ABSTAINS ON WHAT IT DOES NOT KNOW. Measured as selective prediction.

The model answers four-way questions. For each one it has a confidence -- its own probability for the option it
picks, from the same total log-probabilities it ranks them by. It abstains on the questions it is least sure
of. If its confidence means anything, the questions it keeps should be answered more accurately than the ones
it drops, and accuracy should rise as it abstains more.

No threshold is chosen: the whole curve is reported, from answering everything to answering only its single
most confident item. The abstention signal is the model's own output distribution, not a separate classifier,
so nothing outside the model decides when it declines (goal 8).

Reported for both models on the same 508 items:
  coverage    the share of questions it chooses to answer
  accuracy    its accuracy on those
  AURC        area under the risk-coverage curve, risk = 1 - accuracy; lower means it abstains in the right places
  RANDOM      what abstaining at random would give, which is flat at the overall accuracy
usage: eval/abstain.py [checkpoint]"""
import sys, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, "/opt/code/membrae")
from src.common import cap_gpu
cap_gpu(12.0)
import numpy as np, torch
from src.bench import Ours, GPT2
from src.facts import FACTS, arithmetic_items, reasoning_items_large

CK = sys.argv[1] if len(sys.argv) > 1 else "/mnt/kv_cache/run45_prose/ckpt.pt"
ITEMS = list(FACTS) + list(arithmetic_items(60, 0)) + list(reasoning_items_large(400))


def score(model):
    """(correct, confidence) per item, ranking options by total log-probability of the completion."""
    enc = model.encode if hasattr(model, "encode") else model.tok.encode
    ok, conf = [], []
    for stem, ans, wrongs in ITEMS:
        lps = []
        for c in [ans] + wrongs:
            ids = enc(stem) + enc(c)
            lg = model.logits_last(ids, 1).float().log_softmax(-1)
            nc = min(len(enc(c)), lg.shape[1] - 1)
            y = torch.tensor(ids[-nc:], device=lg.device)
            lps.append(float(lg[0, -nc - 1:-1].gather(-1, y.unsqueeze(-1)).sum()))
        a = np.array(lps); p = np.exp(a - a.max()); p /= p.sum()
        ok.append(int(p.argmax()) == 0); conf.append(float(p.max()))
    return np.array(ok), np.array(conf)


def curve(ok, conf):
    order = np.argsort(-conf)                        # most confident first
    acc = np.cumsum(ok[order]) / np.arange(1, len(ok) + 1)
    cov = np.arange(1, len(ok) + 1) / len(ok)
    aurc = float(np.mean(1 - acc))
    return cov, acc, aurc


print(f"{len(ITEMS)} four-way items (48 facts, 60 arithmetic, 400 reasoning), chance 25%\n")
res = {}
for lab, m in (("Carrier", Ours(CK, "/mnt/kv_cache/tok_v8192.json")), ("GPT-2 124M", GPT2())):
    ok, conf = score(m)
    cov, acc, aurc = curve(ok, conf)
    res[lab] = (ok, conf, cov, acc, aurc)
    print(f"{lab}: overall accuracy {ok.mean():.3f}   AURC {aurc:.3f}   "
          f"(random abstention gives {1 - ok.mean():.3f}; lower is better)", flush=True)

print(f"\n{'answers only its most confident':<34s}" + "".join(f"{lab:>14s}" for lab in res))
for c in (1.0, 0.8, 0.6, 0.4, 0.2, 0.1):
    row = f"{c:>6.0%} of questions".ljust(34)
    for lab, (ok, conf, cov, acc, aurc) in res.items():
        k = max(1, int(round(c * len(ok))))
        row += f"{acc[k - 1]:>13.1%} "
    print(row, flush=True)
print("\nA model that knows what it does not know gains accuracy as it answers less.")
for lab, (ok, conf, cov, acc, aurc) in res.items():
    gain = acc[max(1, int(round(0.2 * len(ok)))) - 1] - ok.mean()
    right, wrong = conf[ok].mean(), conf[~ok].mean()
    print(f"  {lab:<12s} answering its top 20%: {gain:+.1%} over answering everything;  "
          f"mean confidence when right {right:.3f}, when wrong {wrong:.3f}")
