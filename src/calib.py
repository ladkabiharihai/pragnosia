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
def collect(model, bins, ks, ps, rng, n_batches=40, B=8, T=512, device="cuda", layer=None):
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
            logits, h = model(x[:, :T], return_hidden=True, hidden_layer=layer)
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


def probe(Xtr, ytr, Xte, yte, device="cuda", steps=800, lr=0.05,
          wds=(0.0, 1e-3, 1e-2, 1e-1, 1.0, 10.0)):
    """Ridge-regularised logistic regression, penalty chosen on a validation split.

    THE UNREGULARISED VERSION WAS MEANINGLESS. Adding 768 hidden features to 2 confidence features made the
    probe WORSE at every layer, by 0.045 to 0.055 AUROC, which is overfitting rather than a statement about the
    representation: 770 features fitted on ten thousand examples with no penalty. A probe that overfits cannot
    distinguish 'the signal is absent' from 'I could not fit it', and those are the two answers this whole
    experiment exists to separate.

    Train/validation/test are three disjoint splits. The penalty is selected on validation and the number
    reported comes from test, so the sweep cannot leak into the result."""
    import torch as _t
    Xtr = _t.tensor(Xtr, device=device, dtype=_t.float32)
    Xte = _t.tensor(Xte, device=device, dtype=_t.float32)
    ytr_t = _t.tensor(ytr, device=device, dtype=_t.float32)
    mu, sd = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True) + 1e-6
    Xtr = (Xtr - mu) / sd; Xte = (Xte - mu) / sd
    n = Xtr.shape[0]; ntr = int(n * 0.75)
    Xa, ya = Xtr[:ntr], ytr_t[:ntr]
    Xv, yv = Xtr[ntr:], ytr[ntr:]
    best = (-1.0, None)
    for wd in wds:
        w = _t.zeros(Xa.shape[1], 1, device=device, requires_grad=True)
        b = _t.zeros(1, device=device, requires_grad=True)
        opt = _t.optim.Adam([w, b], lr=lr, weight_decay=0.0)
        for _ in range(steps):
            opt.zero_grad()
            z = (Xa @ w).squeeze(-1) + b
            loss = F.binary_cross_entropy_with_logits(z, ya) + wd * (w * w).sum()
            loss.backward(); opt.step()
        with torch.no_grad():
            a = auroc(((Xv @ w).squeeze(-1) + b).cpu().numpy(), yv)
        if a > best[0]:
            best = (a, (w.detach().clone(), b.detach().clone()))
    w, b = best[1]
    with torch.no_grad():
        return auroc(((Xte @ w).squeeze(-1) + b).cpu().numpy(), yte)


def report(model, bins, ks, ps, seed=0, n_batches=40, log=print, layers=(None,)):
    """Ask the INCREMENTAL question, not the absolute one.

    A first version compared a probe on the final hidden state against the output confidence and found the
    probe worse (0.697 against 0.807). That comparison cannot answer anything: the output head IS a trained
    linear map on that same hidden state, fitted over billions of tokens, so a logistic regression on ten
    thousand examples losing to it says the probe is weaker, not that the signal is missing.

    What can be answered is whether the representation adds anything the output does not already say. So the
    baseline is confidence and entropy as features, and the test is those features PLUS the hidden state. A
    real gain means there is uncertainty information the output distribution is failing to carry.

    Intermediate layers are read as well as the last one, because the final layer is exactly what the output
    head consumes and therefore the least likely place to find anything it has not already used."""
    rng = np.random.default_rng(seed)
    out = {}
    for li in layers:
        H, C, E, Y = collect(model, bins, ks, ps, np.random.default_rng(seed), n_batches=n_batches, layer=li)
        n = len(Y); half = n // 2
        base = np.stack([C, E], 1)                       # what the output already tells us
        both = np.concatenate([base, H], 1)              # plus the representation
        a_conf = auroc(C, Y)
        a_base = probe(base[:half], Y[:half], base[half:], Y[half:])
        a_both = probe(both[:half], Y[:half], both[half:], Y[half:])
        name = "final" if li is None else f"layer {li}"
        log(f"  {name:<9s} acc {100*float(Y.mean()):.1f}%  conf-only {a_conf:.3f}  "
            f"conf+ent {a_base:.3f}  conf+ent+hidden {a_both:.3f}   incremental {a_both-a_base:+.3f}")
        out[name] = dict(conf=a_conf, base=a_base, both=a_both, gain=a_both - a_base, n=n)
    best = max(out.values(), key=lambda d: d["gain"])
    log("")
    if best["gain"] > 0.03:
        log(f"  -> the representation carries uncertainty the output does not express (+{best['gain']:.3f}).")
    else:
        log(f"  -> no signal beyond what the output already shows (best gain {best['gain']:+.3f}).")
    return out


# ------------------------------------------------------------------ goal 5: does it know what it does NOT know
FAKE_SUBS = {"France": "Zorvania", "Japan": "Brelandia", "Italy": "Novaquia", "Egypt": "Ferrovia",
             "Canada": "Ristonia", "Russia": "Vandoria", "Brazil": "Kelmoor", "Austria": "Tavrenne",
             "China": "Brelandia", "Earth": "Zorvania", "the world": "Zorvania",
             "our solar system": "the Zorvanian system", "the United States": "Zorvania"}


def fabricate(stem: str):
    """Same grammatical frame, same question, a subject that does not exist."""
    for real, fake in FAKE_SUBS.items():
        if real in stem:
            return stem.replace(real, fake)
    return None


@torch.no_grad()
def confidence_real_vs_fake(model, tok, facts, device="cuda", vocab=8192):
    """THE GOAL-5 MEASUREMENT, and the one that came out INVERTED on every previous model.

    For each stem the model is given a fixed set of candidate answers and its distribution over them is read.
    The identical candidate set is then used with a fabricated subject, so the options are matched exactly and
    the only thing that changes is whether the thing being asked about exists.

    A model that knows what it does not know is LESS confident on the fabricated ones. Measured on the
    previous lineage the opposite held at every size: 0.689 against 0.642 at 125M, 0.617 against 0.536 at 405M.
    Confidence rose in the absence of evidence, because with no real knowledge nothing competes with the most
    surface-plausible option and the distribution collapses onto it."""
    import numpy as np
    model.eval()
    real, fake = [], []
    for stem, right, wrongs in facts:
        f = fabricate(stem)
        if f is None:
            continue
        cands = [right] + wrongs
        for text, bucket in ((stem, real), (f, fake)):
            lps = []
            for c in cands:
                ids = tok.encode(text).ids + tok.encode(c).ids
                x = torch.tensor([ids], device=device)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    lp = model(x[:, :-1]).float().log_softmax(-1)
                nc = len(tok.encode(c).ids)
                lps.append(float(lp[0, -nc:].gather(-1, x[0, 1:][-nc:].unsqueeze(-1)).mean()))
            a = np.array(lps); a = np.exp(a - a.max()); a /= a.sum()
            bucket.append(float(a.max()))
    model.train()
    real, fake = np.array(real), np.array(fake)
    sep = auroc(np.concatenate([-real, -fake]),
                np.concatenate([np.ones(len(real)), np.zeros(len(fake))]))
    return dict(n=len(real), conf_real=float(real.mean()), conf_fake=float(fake.mean()),
                inverted=bool(fake.mean() > real.mean()), auroc_real_vs_fake=float(1 - sep))


# --------------------------------------------------------------- goal 9: does retrieval survive past training
@torch.no_grad()
def needle(model, tok, distances=(0, 64, 256, 512, 1024, 2048), trials=12, device="cuda", filler=None):
    """Paired in-context retrieval lift at increasing distance, through the CARRIED state past the window.

    The same query is scored with the fact present and with a DIFFERENT fact present, so token frequency
    priors cancel exactly and what remains is the lift attributable to the fact being in context.

    A FIRST VERSION OF THIS TEST WAS WRONG in a way that produced clean-looking zeros: it truncated the input
    to the last ctx tokens, so at any distance past about ctx-20 the fact itself was cut out of the input, and
    the reported lift of exactly 0.000 meant "fact absent", not "fact forgotten". Distances beyond the window
    are now evaluated by carrying the carrier state across windows, which is the path the architecture exists
    for, and a distance that would still not fit is an error rather than a silent zero.

    WHY THIS MATTERS. Both previous architectures in this project lost retrieval at roughly their training
    window. This model trains at 512 and its carrier's slowest half-life is about 693 tokens, so whether
    retrieval crosses 512 is a real question."""
    import numpy as np
    model.eval()
    out = []
    fill = filler if filler is not None else list(range(100, 4200))
    cue = tok.encode(" The vault keeper is").ids
    ans = tok.encode(" Okonkwo").ids
    for d in distances:
        lifts = []
        for t in range(trials):
            pad = [fill[(t * 977 + i) % len(fill)] for i in range(d)]
            def sc(pre):
                ids = tok.encode(pre).ids + pad + cue + ans
                x = torch.tensor([ids], device=device)
                if x.shape[1] <= model.ctx:
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        lp = model(x[:, :-1]).float().log_softmax(-1)
                    n = len(ans)
                    return float(lp[0, -n:].gather(-1, x[0, 1:][-n:].unsqueeze(-1)).mean())
                # longer than the window: carry state, score the answer inside the final window
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    lp = model.forward_carried(x[:, :-1]).float().log_softmax(-1)
                n = len(ans)
                tgt = x[0, 1:][-n:]
                return float(lp[0, -n:].gather(-1, tgt.unsqueeze(-1)).mean())
            lifts.append(sc(" The vault keeper is Okonkwo. ") - sc(" The vault keeper is Vandermeer. "))
        out.append((d, float(np.mean(lifts)), float(np.std(lifts) / np.sqrt(len(lifts)))))
    model.train()
    return out
