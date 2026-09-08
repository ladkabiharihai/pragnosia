"""PRAGNOSIA: ONE CONTINUOUS RUN, INFANT TO 405M, ALL GROWTH INTRINSIC, MUON THROUGHOUT.

WHY ONE RUN AND NOT TWO. The previous 400M was a CONTINUATION: train a 125M, finish it, then restart from it.
Measured, that restart costs real quality -- continuing the annealed 125M at 3e-4 took BPB 0.9477 -> 0.9699 in
37M tokens and 300M did not recover it. An annealed model sits in a converged basin and a fresh schedule knocks
it out. One continuous run never anneals mid-way, so that cost is never paid.

WHY MUON. From scratch at 24.5M it beat fused Adam by 0.0774 BPB, reaching Adam's 175M-token loss at 100M tokens
(~1.75x fewer tokens). That is 6.4x the measured seed noise floor (std 0.0042, 2-sigma bar 0.0120), so it is a
real result. It did NOT survive a warm start onto an Adam-trained model, which is exactly why this is from
scratch rather than a continuation of the existing flagship.

THE LADDER, and every rung fires on the model's OWN saturation signal, not a schedule:
  35.0M  D448x10L DIN16   ->  64.7M  ->  125.0M  ->  191.1M  ->  278.9M  ->  405.2M D1408x18L DIN40
The 22 tok/param floor only decides when growth is ALLOWED; the SaturationSensor decides when it happens.
Climbing the whole ladder at the floor costs 23.4B tokens.

WHY WSD AND NOT COSINE. Cosine bakes the endpoint into the schedule, so stopping early leaves a model that was
never annealed and wastes the run. WSD holds the LR flat and anneals only at the end, so the budget can be set by
the CLOCK rather than guessed now:
    touch /mnt/kv_cache/pragnosia_data/pbrain_DECAY_NOW   -> anneals cleanly, saves, stops.
A checkpoint is written before every growth, so the 125M rung survives as its own artifact.
"""

import sys, os, time, math
sys.path.insert(0, "/opt/code/membrae")
from util import setup_log, pbar
setup_log("trainbrain")
import numpy as np, torch, torch.nn.functional as F
import core as C, growth as G
from muon_opt import Muon
from data_mix import DataMix
from tokenizers import Tokenizer
import torch._dynamo as _d; _d.config.recompile_limit = 64
import torch._inductor.config as _ic; _ic.triton.prefer_nd_tiling = True; _ic.compile_threads = 8

DEV = C.DEV
TOK = Tokenizer.from_file("/opt/code/membrae/data/bpe32k.json")
V = 32768
D, NL, DIN = 448, 10, 16                      # INFANT 34.9M; grows to 832x14L DIN32 = 125.0M
TARGET_NL = 18                                # res_scale is pinned to the TARGET depth so growth needs no rescale
PLAN = [('wdi', (160, 2, 8)), ('wdi', (224, 2, 8)), ('wdi', (160, 2, 0)),
        ('wdi', (192, 1, 8)), ('wdi', (224, 1, 0))]   # 35 -> 65 -> 125 -> 191 -> 279 -> 405M
# PER-RUNG FLOOR. 22 tok/param is the ordinary earned floor, but the 125M and 405M rungs get **70** -- 125M
# because that is what makes it comparable to GPT-2 on tokens as well as parameters (a uniform 22 would have the
# model grow away from 125M after only 2.75B tokens and throw that artifact away), and 405M because it is the
# endpoint and should be properly trained rather than merely reached.
# Indexed by k = growths so far, i.e. the CURRENT size: [35M, 65M, 125M, 191M, 279M].
GATE_BY_RUNG = [22.0, 22.0, 70.0, 22.0, 22.0]
FINAL_TPP = 70.0                              # tok/param the 405M endpoint is budgeted for
CH = 2048
B = int(os.environ.get("B", "8"))
LR     = float(os.environ.get("LR", "0.02"))     # Muon's scale, from the 24.5M run -- NOT Adam's 1e-3
LR_AUX = float(os.environ.get("LR_AUX", "3e-3"))  # Adam for biases/norms/scalars
DECAY_FLAG = "/mnt/kv_cache/pragnosia_data/pbrain_DECAY_NOW"
DECAY_FRAC = 0.10
TPP = float(os.environ.get("TPP", "73"))      # of the ENDPOINT; WSD means we can stop before it
LANG_FRAC = float(os.environ.get("LANG_FRAC", "0.70"))
WARM = 2000
SAVE_EVERY = 2000
EVAL_EVERY = 4000
CKPT = "/mnt/kv_cache/pragnosia_data/pbrain.pt"

dm = DataMix(ctx=CH + 1, verbose=True, lang_frac=LANG_FRAC)
np.random.seed(777)
EV = [torch.tensor(dm.batch(2), device=DEV) for _ in range(10)]


def build():
    cfg = C.Config(D=D, DIN=DIN, DV=D, NL=NL, T=CH, res_scale=TARGET_NL ** -0.5, tie_emb=True)
    cfg.gated = True; cfg.qknorm = True
    cfg.qk_learn_scale = True          # the fix; init s=1 so the infant starts identical to the old model
    cfg.sel_write = True               # the vision's propose gate, which had never once run under gated=True.
                                       # It does NOT extend the horizon (measured: half-life 1824 vs 1848, and
                                       # d=2000 unmoved), so that hypothesis is dead -- but it is worth -0.0125
                                       # BPB and +36% needle lift at d=200, so it ships on its own merits.
    return C.Core(cfg).to(DEV), cfg


@torch.no_grad()
def bpb(m, cfg):
    """tokenizer-independent, so it compares RUNS to each other. NOT a GPT-2 comparison: this scores
    2048-token windows, whose later positions have up to 2047 tokens of context, while GPT-2 can never exceed
    1023 -- and later positions are easier. See exp/gpt2_bpb.py, which matches context and domain."""
    m.eval(); nats = 0.0; nb = 0
    for X in EV:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg = m(X[:, :CH]).float()
        y = X[:, 1:CH + 1]
        nats += F.cross_entropy(lg.reshape(-1, V), y.reshape(-1), reduction='sum').item()
        for b in range(X.shape[0]):
            nb += len(TOK.decode([int(z) for z in y[b]]).encode('utf-8'))
    m.train()
    return nats / (np.log(2) * nb)


def main():
    torch.manual_seed(0); np.random.seed(0)
    m, cfg = build()
    P = lambda: sum(p.numel() for p in m.parameters())
    P_FINAL = 405_200_000                      # the budget is set by the ENDPOINT so total tokens match GPT-2
    # total budget = every rung at its own floor + the endpoint at FINAL_TPP. WSD anneals over the last 10% of
    # THIS, and the DECAY flag ends it early if the clock demands -- which it will, since 405M at 70 is ~105h.
    _LADDER = [35.0e6, 64.7e6, 125.0e6, 191.1e6, 278.9e6]
    _TOTAL = sum(g * p for g, p in zip(GATE_BY_RUNG, _LADDER)) + FINAL_TPP * P_FINAL
    STEPS = int(_TOTAL) // (B * CH)
    def make_opt(model, lr):
        two=[p for p in model.parameters() if p.ndim==2 and p.numel()>4096]
        rest=[p for p in model.parameters() if not (p.ndim==2 and p.numel()>4096)]
        return [Muon(two, lr=lr), torch.optim.Adam(rest, lr=LR_AUX, betas=(0.9,0.95), fused=True)]
    opts = make_opt(m, LR)
    step = 0; k = 0; tokens_at_size = 0
    if os.path.exists(CKPT):
        sd = torch.load(CKPT, map_location=DEV)
        k = sd.get('k', 0)
        # RESUME IS SAFE WITHOUT READING head_scale, and deliberately so: replaying the growths below rebuilds it,
        # because grow_width divides cfg.head_scale by c on every width growth. That is why the run has been
        # correct all along even though the SAVED field was hardcoded to 1.0 (now fixed at both save sites).
        # The stale field only ever hurt EXTERNAL loaders that trust it -- an eval reading 1.0 instead of the
        # true 0.7338 reported BPB 1.1550 where the live model scores 1.0698.
        for kk in range(k):                    # rebuild the grown shape AND the accumulated head_scale
            kind, amt = PLAN[kk]; G.grow_width(m, amt[0]); G.grow_depth(m, amt[1]); G.grow_din(m, amt[2])
        assert abs(m.cfg.head_scale - (D / sd['D']) ** 0.5) < 1e-6, "head_scale did not telescope as expected"
        m.load_state_dict(sd['model'])
        opts = make_opt(m, LR)     # Muon carries only momentum; cheap to rebuild rather than restore
        step = sd['step']; tokens_at_size = sd.get('tokens_at_size', 0)
        print(f"RESUMED at step {step}/{STEPS}, {k} grows, {P()/1e6:.0f}M", flush=True)
    fwd = torch.compile(m)
    sensor = G.SaturationSensor(window=400, t_stat=1.0, min_gap=2500, floor_tokens=1, consec=3)
    sat_first = {i: None for i in range(len(PLAN) + 1)}   # when the SENSOR first said saturated, per rung
    print(f"PRAGNOSIA CONTINUOUS: infant D{D}x{NL}L DIN{DIN} = {P()/1e6:.1f}M -> 405.2M in 5 SENSOR-DRIVEN grows\n"
          f"  ladder 35 -> 65 -> 125 -> 191 -> 279 -> 405M; the 22 tok/param floor only says when growth is ALLOWED\n"
          f"  floors per rung {GATE_BY_RUNG} then {FINAL_TPP:.0f} at 405M = {_TOTAL/1e9:.1f}B tokens, {STEPS} steps\n"
          f"  125M and 405M get 70 tok/param; the rest grow at 22\n"
          f"  MUON on 2D weights (lr {LR}) + Adam on biases/norms (lr {LR_AUX}); WSD, not cosine\n"
          f"  stop any time with: touch {DECAY_FLAG}   -- anneals cleanly, saves, exits\n"
          f"  NOTE: this BPB uses 2048-token windows. It tracks progress and compares runs to each other,\n"
          f"  but it is NOT comparable to GPT-2, which is capped at 1024 context -- use exp/gpt2_bpb.py for that.", flush=True)

    t0 = time.time(); ema = None; decay_from = None
    bar = pbar(range(step, STEPS), "brain", total=STEPS - step)
    for it in bar:
        # WSD: warm up, hold flat, anneal only at the end -- of the planned budget or from the DECAY flag,
        # whichever comes first, so the run can be ended by the clock without wasting it.
        if decay_from is None and os.path.exists(DECAY_FLAG):
            decay_from = it
            bar.write(f"    >>> DECAY TRIGGERED at step {it} -- annealing over {int(DECAY_FRAC*STEPS)} steps")
        d0 = decay_from if decay_from is not None else int((1 - DECAY_FRAC) * STEPS)
        dN = int(DECAY_FRAC * STEPS)
        sc = ((it + 1) / WARM if it < WARM else 1.0 if it < d0 else max(0.02, 1.0 - (it - d0) / max(1, dN)))
        lr = LR * sc
        if decay_from is not None and it >= decay_from + dN:
            bar.write(f"    >>> decay complete at step {it}; saving and stopping"); break
        for o in opts:
            for g in o.param_groups: g['lr'] = (LR if o is opts[0] else LR_AUX) * sc
        X = torch.tensor(dm.batch(B), device=DEV)
        for o in opts: o.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg = fwd(X[:, :CH])
            l = F.cross_entropy(lg.reshape(-1, V), X[:, 1:CH + 1].reshape(-1))
        l.backward()
        ent = G.SaturationSensor.entropy(lg.detach())      # the sensor's signal is ENTROPY, not the loss
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        for o in opts: o.step()
        v = l.item(); ema = v if ema is None else 0.99 * ema + 0.01 * v
        tokens_at_size += B * CH
        # INTRINSIC GROWTH (VISION 0a): the sensor decides WHEN; the 22 tok/param floor only decides when it is
        # ALLOWED to. A hand-set schedule would be exactly the decision-scaffolding the vision forbids.
        gate = GATE_BY_RUNG[k] if k < len(GATE_BY_RUNG) else FINAL_TPP
        earned_ok = tokens_at_size >= gate * P()
        # Feed the sensor REAL tokens always, so its opinion is formed independently of the floor. Previously it
        # was fed 0 until the floor opened, so it could only ever fire the instant permission arrived -- which is
        # exactly what happened at growth #1 (step 46,947 against a floor of 46,947). Now we RECORD when the
        # sensor first says saturated, and can state honestly which of the two actually decided.
        _sat, _t = sensor.verdict()
        if _sat and sat_first[k] is None:
            sat_first[k] = (it, tokens_at_size / max(1.0, P()))
            bar.write(f"    [sensor] SATURATED at step {it} = {sat_first[k][1]:.1f} tok/param "
                      f"(t={_t:+.2f}); floor opens at {gate:.0f} tok/param")
        fired = sensor.observe(ent, it, tokens_at_size if earned_ok else 0)
        if k < len(PLAN) and fired and earned_ok:
            kind, amt = PLAN[k]; earned = tokens_at_size / P()
            torch.save({'model': m.state_dict(), 'step': it + 1, 'k': k, 'D': m.cfg.D, 'NL': m.cfg.NL,
                        'DIN': m.cfg.DIN, 'DV': m.cfg.DV, 'tie_emb': True, 'head_scale': m.cfg.head_scale, 'D0': D,
                        'gated': True, 'qknorm': True, 'qk_learn_scale': True, 'sel_write': True, 'res_scale': m.cfg.res_scale},
                       f"{CKPT[:-3]}_before_grow{k+1}.pt")
            G.grow_width(m, amt[0]); G.grow_depth(m, amt[1]); G.grow_din(m, amt[2])
            k += 1; tokens_at_size = 0
            sensor.reset()          # the window would otherwise straddle the growth and mix two models
            opts = make_opt(m, lr)     # rebuild after growth: the parameter set changed
            torch.cuda.empty_cache()
            try:
                fwd = torch.compile(m)
                _x = torch.zeros(B, CH, dtype=torch.long, device=DEV)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    F.cross_entropy(fwd(_x).reshape(-1, V), _x.reshape(-1)).backward()
                m.zero_grad(set_to_none=True)
                for o in opts: o.zero_grad(set_to_none=True)
                del _x
            except Exception as e:
                bar.write(f"    compile failed after growth ({type(e).__name__}) -- continuing EAGER"); fwd = m
            bar.write(f"    >>> SELF-GREW #{k} at step {it+1}: D{m.cfg.D}x{m.cfg.NL}L DIN{m.cfg.DIN} "
                      f"-> {P()/1e6:.0f}M (earned {earned:.1f} tok/param, floor {gate:.0f})")
            _sf = sat_first[k - 1]
            if _sf is None:
                bar.write(f"        DECIDED BY: the SENSOR (it had not flagged saturation before the floor opened)")
            elif earned - _sf[1] < 0.5:
                bar.write(f"        DECIDED BY: the FLOOR -- sensor was already saturated at {_sf[1]:.1f} tok/param "
                          f"(step {_sf[0]}), {earned - _sf[1]:.1f} tok/param before growth was permitted")
            else:
                bar.write(f"        DECIDED BY: the FLOOR -- sensor saturated at {_sf[1]:.1f} tok/param, "
                          f"growth waited until {earned:.1f}")
        if (it + 1) % 50 == 0:
            done = (it + 1 - step) * B * CH
            free = torch.cuda.mem_get_info()[0] / 2 ** 30
            _s2, _t2 = sensor.verdict()
            _stag = "sat" if _s2 else (f"t{_t2:+.1f}" if _t2 is not None else "warm")
            bar.set_postfix_str(f"loss={ema:.3f} ppl={math.exp(min(20,ema)):.0f} {_stag} lr={lr:.1e} "
                                f"{done/(time.time()-t0)/1e3:.0f}ktok/s {(it+1)*B*CH/1e9:.2f}B free={free:.0f}G")
        if (it + 1) % SAVE_EVERY == 0:
            torch.save({'model': m.state_dict(), 'step': it + 1,
                        'D': m.cfg.D, 'NL': m.cfg.NL, 'DIN': m.cfg.DIN, 'DV': m.cfg.DV, 'tie_emb': True, 'head_scale': m.cfg.head_scale, 'D0': D,
                        'gated': True, 'qknorm': True, 'qk_learn_scale': True, 'sel_write': True, 'res_scale': m.cfg.res_scale,
                        'k': k, 'tokens_at_size': tokens_at_size}, CKPT)
        if (it + 1) % EVAL_EVERY == 0:
            b = bpb(m, m.cfg)
            bar.write(f"    step {it+1} | {(it+1)*B*CH/1e9:.2f}B tokens | {(it+1)*B*CH/P_FINAL:.1f} tok/param "
                      f"| BPB {b:.4f} | (our 2048-ctx metric; see exp/gpt2_bpb.py for the fair GPT-2 comparison)")
    torch.save({'model': m.state_dict(), 'step': STEPS,
                'D': m.cfg.D, 'NL': m.cfg.NL, 'DIN': m.cfg.DIN, 'DV': m.cfg.DV, 'tie_emb': True, 'head_scale': m.cfg.head_scale, 'D0': D,
                'gated': True, 'qknorm': True, 'qk_learn_scale': True, 'sel_write': True, 'res_scale': m.cfg.res_scale,
                        'k': k, 'tokens_at_size': tokens_at_size}, CKPT)
    print(f"\nDONE: {STEPS*B*CH/1e9:.2f}B tokens, final BPB {bpb(m, m.cfg):.4f}  (2048-ctx metric, not a GPT-2 comparison)", flush=True)


if __name__ == "__main__":
    main()
