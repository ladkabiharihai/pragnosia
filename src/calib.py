"""Does the model know when it is wrong, and if not, is the information absent or merely unexpressed?

VISION GOALS 4 AND 5. Measured on the previous lineage, confidence did not predict correctness (AUROC 0.681 at
125M and 0.402 at 405M, the latter below chance) and confidence on fabricated entities was HIGHER than on real
ones. Every model tested has failed this, so it is the gap worth attacking, and it is cheap to measure.

THE DECISIVE QUESTION IS NOT "is it calibrated". It is where the failure lives:

  If a probe on the hidden state predicts correctness much better than the output confidence does, the model
  KNOWS it is uncertain and the output distribution does not carry it. That is fixable: the signal exists and
  needs a path to the output.

  If the probe is no better than the confidence, the information is genuinely absent from the representation,
  and no amount of output-side calibration will manufacture it. That is a much deeper problem and would say
  the architecture or the objective has to change.

This distinction is the same shape as the one already measured on facts: the model could RANK the right answer
50% of the time while only SAYING it 10% of the time. Knowledge present, expression missing.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F


def auroc(scores, labels) -> float:
    """Probability a positive outranks a negative. Rank-based, needs no threshold."""
    s = np.asarray(scores, dtype=np.float64); y = np.asarray(labels, dtype=np.int64)
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    order = np.argsort(s); ranks = np.empty(len(s), dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1)
    npos = int(y.sum()); nneg = len(y) - npos
    return float((ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


@torch.no_grad()
def collect(model, bins, ks, ps, rng, n_batches=40, B=8, T=512, device="cuda"):
    """Gather, per predicted token: was it correct, how confident was the model, and the hidden state.

    Positions are subsampled rather than taken wholesale so that adjacent, highly correlated tokens do not
    dominate the sample and inflate every downstream number."""
    model.eval()
    H, C, E, Y = [], [], [], []
    for _ in range(n_batches):
        o = np.empty((B, T + 1), dtype=np.int64)
        for i in range(B):
            a = bins[ks[rng.choice(len(ks), p=ps)]]
            j = rng.integers(0, len(a) - T - 2)
            o[i] = a[j:j + T + 1]
        x = torch.from_numpy(o).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, h = model(x[:, :T], return_hidden=True)
        logits = logits.float()
        tgt = x[:, 1:T + 1]
        p = logits.softmax(-1)
        conf, pred = p.max(-1)
        ent = -(p * (p + 1e-9).log()).sum(-1)
        idx = torch.randint(T // 4, T, (B, 64), device=device)          # skip the warm-up positions
        g = lambda t: t.gather(1, idx).reshape(-1)
        H.append(h.gather(1, idx.unsqueeze(-1).expand(-1, -1, h.shape[-1])).reshape(-1, h.shape[-1]).float().cpu())
        C.append(g(conf).cpu()); E.append(g(ent).cpu())
        Y.append((g(pred) == g(tgt)).long().cpu())
    model.train()
    return (torch.cat(H).numpy(), torch.cat(C).numpy(), torch.cat(E).numpy(), torch.cat(Y).numpy())


def probe(Htr, ytr, Hte, yte, steps=600, lr=0.05, device="cuda"):
    """Logistic regression on the hidden state, predicting whether the model was right.

    Linear deliberately: a deep probe can manufacture signal that the model itself has no way to use, so a
    linear probe is the honest question, namely whether the information is linearly available where the output
    head could already have read it."""
    Xtr = torch.tensor(Htr, device=device); Ytr = torch.tensor(ytr, device=device).float()
    Xte = torch.tensor(Hte, device=device)
    mu, sd = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True) + 1e-6
    Xtr = (Xtr - mu) / sd; Xte = (Xte - mu) / sd
    w = torch.zeros(Xtr.shape[1], 1, device=device, requires_grad=True)
    b = torch.zeros(1, device=device, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits((Xtr @ w).squeeze(-1) + b, Ytr)
        loss.backward(); opt.step()
    with torch.no_grad():
        s = ((Xte @ w).squeeze(-1) + b).cpu().numpy()
    return auroc(s, yte)


def report(model, bins, ks, ps, seed=0, n_batches=40, log=print):
    rng = np.random.default_rng(seed)
    H, C, E, Y = collect(model, bins, ks, ps, rng, n_batches=n_batches)
    n = len(Y); half = n // 2
    acc = float(Y.mean())
    a_conf = auroc(C, Y)
    a_ent = auroc(-E, Y)
    a_probe = probe(H[:half], Y[:half], H[half:], Y[half:])
    log(f"  next-token accuracy        {100*acc:.1f}%   ({n} positions)")
    log(f"  AUROC from max-probability {a_conf:.3f}")
    log(f"  AUROC from -entropy        {a_ent:.3f}")
    log(f"  AUROC from a LINEAR PROBE  {a_probe:.3f}   on the hidden state")
    gap = a_probe - max(a_conf, a_ent)
    log(f"  probe minus best output signal: {gap:+.3f}")
    if gap > 0.05:
        log("  -> the representation carries uncertainty the output distribution does not express.")
    else:
        log("  -> no hidden signal beyond what the output already shows.")
    return dict(acc=acc, conf=a_conf, ent=a_ent, probe=a_probe, gap=gap, n=n)
