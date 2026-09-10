"""PRAGNOSIA 400M -- grown from the FINISHED 124M, not trained from scratch.

WHY THIS RUNS AFTER THE 124M FINISHES ITS FULL BUDGET, NOT AT 22 TOK/PARAM.
Growth is function-preserving, so the 400M starts at exactly the function the 124M ended at -- nothing is lost by
letting the 124M finish first. But growing EARLY would permanently cost the one comparison that is hardest to
argue with: 124.8M params and 9.1B tokens, both matched to GPT-2 124M. A 400M model beating GPT-2 124M proves
nothing an evaluator will credit -- they will say you used 3x the parameters, and they will be right. So we bank
the matched claim, then grow. Both artifacts survive: p124f.pt is the head-to-head, p400f.pt is the scaling and
growth demonstration.

D832x14L DIN32 (125.0M) -> D1408x18L DIN40 (405.2M). DIN goes 32 -> 40 in the same step because selectivity
saturates at DIN:1, so the read ceiling rises with the model instead of staying behind it.
Compute: 400M at 22 tok/param is 8.9B tokens = 2.1e19 flops, ~3x the 124M's 6.8e18.

  python train400f.py                        # resumes automatically; WSD, no fixed endpoint
  touch /mnt/kv_cache/pragnosia_data/p400f_DECAY_NOW   # finish early: anneals, saves, stops cleanly
"""
import sys, os, time, math
sys.path.insert(0, "/opt/code/membrae")
from util import setup_log, pbar, cfg_from_ckpt
setup_log("train400f")
import numpy as np, torch, torch.nn.functional as F
import core as C, growth as G
from data_mix import DataMix
from tokenizers import Tokenizer
import torch._dynamo as _d; _d.config.recompile_limit = 64
import torch._inductor.config as _ic; _ic.triton.prefer_nd_tiling = True; _ic.compile_threads = 8

DEV = C.DEV
TOK = Tokenizer.from_file("/opt/code/membrae/data/bpe32k.json")
V = 32768
# INCREMENTAL, SENSOR-DRIVEN GROWTH -- not one hand-specified jump.
# The vision (0a) says growth must be intrinsic: the model decides WHEN from its own predictive entropy, and the
# 22 tok/param floor only decides when it is ALLOWED to. A single manual grow of +576 width is a schedule, not a
# decision, so this climbs a ladder instead and lets the SaturationSensor fire each step.
# The ladder is 3 steps rather than 4 because of the clock: each rung must earn 22 tok/param before the next, and
# a 4-rung ladder costs 69 GPU-hours to climb before the 405M trains at all, which is the entire remaining budget.
# ONE RUNG, NOT THREE. Restarting the ladder from 191M with 38 hours left, a 3-rung climb spends 6.7h earning
# the floor at 191M and 14h at 279M, leaving the 405M rung about 12 tok/param, which is WORSE than the 405M we
# already have at 27. Growth is cheap but training the result is not, and an under-trained larger model is not
# an improvement. So this run climbs one rung and then trains it properly.
PLAN = [(192, 1, 8)]      # D992x16L DIN32 191M -> D1184x17L DIN40 ~279M
TOK_PER_PARAM_GATE = 22.0    # FLOOR: the earliest growth is ALLOWED. Stops growing before the size is earned.
TOK_PER_PARAM_CEIL = 40.0    # CEILING: growth is FORCED. A sensor that never fires must not be able to stall
                             # the run: the endpoint (405M) is a deliverable and cannot depend on a heuristic
                             # firing. Between floor and ceiling the SENSOR decides, and the log says which.
# Start from the last rung the run produced BEFORE it went wrong. The 191M reached BPB 0.9190 and was still
# improving; everything after it degraded. NOTE this checkpoint carries no optimizer state and no
# tokens_at_size, so Adam starts cold here and the warmup below is doing real work rather than being a formality.
SRC = "/mnt/kv_cache/pragnosia_data/p400f_before_grow2.pt"
CKPT = "/mnt/kv_cache/pragnosia_data/p279f.pt"
END_PARAMS = 279_000_000     # the endpoint the token budget is measured against
CH = 2048
B = int(os.environ.get("B", "8"))
# FUSED ADAM: the optimizer update was 13.9% of a COMPILED step, spent in multi_tensor_apply -- a separate
# kernel launch per parameter group per moment. torch's fused Adam does the whole update in one kernel and
# measured 9.0% off the step time (65.3 -> 59.4 ms) with identical maths. Free, so it is on.
# ---------------------------------------------------------------------------------------------------------
# LEARNING RATE. The previous run held 6e-4 flat from 125M through 405M and went backwards for billions of
# tokens: BPB fell to 0.9329 at 191M, then climbed smoothly and monotonically from 0.9363 to 0.9821 over 32,000
# steps at 279M and never recovered. A smooth monotone climb at constant LR is not an instability, it is the
# model equilibrating at the noise floor that LR implies, and that floor was worse than where it already was.
#
# Two things were wrong and both are fixed here.
#   1. The rate never changed as the model grew. muP for Adam scales hidden-weight LR as 1/fan_in, so the
#      matrices now get LR_REF * D_REF/D while embeddings, norms, biases and scalars stay flat. D_REF/LR_REF are
#      taken from the width and rate at which this model demonstrably trained well (D=992 at 6e-4), rather than
#      from a constant someone picked.
#   2. The schedule held flat and only annealed in the last 10%. The 125M that reached 0.9083 was annealed; this
#      run was not, and a flat rate is precisely what lets a model sit at its noise floor. Cosine decay now
#      spans the whole budget, with warmup at the start AND after every growth.
LR = float(os.environ.get("LR", "6e-4"))      # peak rate at the reference width
D_REF = 992                                   # width at which 6e-4 was measured to train well
LR_MIN_MULT = 0.05                            # cosine floor, as a fraction of peak
TPP = float(os.environ.get("TPP", "63"))      # of the 279M endpoint: ~17.6B tokens, about 38 GPU-hours
GRAD_SPIKE = float(os.environ.get("GRAD_SPIKE", "4.0"))   # skip an update whose grad norm exceeds this x median
# WSD (warmup-stable-decay) instead of cosine, so the run is NOT committed to a token budget in advance.
# Cosine bakes the endpoint into the schedule: stop early and you stop mid-schedule at a high LR, with a model
# that was never annealed, which wastes the run. WSD holds LR flat and anneals only at the end, so the budget can
# be decided by the clock rather than guessed now. To finish early, write the file below and it starts the decay
# from that step; without it, decay begins at 90% of TPP.
DECAY_FLAG = "/mnt/kv_cache/pragnosia_data/p279f_DECAY_NOW"
DECAY_FRAC = 0.10
WARM = 2000        # also re-applied after every growth, where the model is most fragile
SAVE_EVERY = 2000
EVAL_EVERY = 4000
LANG_FRAC = float(os.environ.get("LANG_FRAC", "0.70"))

dm = DataMix(ctx=CH + 1, verbose=True, lang_frac=LANG_FRAC)
np.random.seed(777)
EV = [torch.tensor(dm.batch(2), device=DEV) for _ in range(10)]


@torch.no_grad()
def bpb(m):
    m.eval(); nats = 0.0; nb = 0
    for X in EV:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg = m(X[:, :CH]).float()
        y = X[:, 1:CH + 1]
        nats += F.cross_entropy(lg.reshape(-1, V), y.reshape(-1), reduction='sum').item()
        for b in range(X.shape[0]):
            nb += len(TOK.decode([int(z) for z in y[b]]).encode('utf-8'))
    m.train(); return nats / (np.log(2) * nb)


def make_opt(m, lr):
    """Two parameter groups with muP-style width scaling.

    Adam's update size is set by the ratio of the gradient to sqrt of its own second moment, which is roughly
    scale free per parameter, so the same nominal LR moves a wide matrix much further in function space than a
    narrow one. muP's rule for Adam is that hidden weight LR scales as 1/fan_in while embeddings, biases,
    norms and scalars stay flat. The previous run ignored this and kept 6e-4 from D=832 to D=1408.

    The reference (D_REF, LR) is a width and rate this model was MEASURED to train well at, not a guess: at
    D=992 on 6e-4 the BPB fell from 0.9603 to 0.9329 and was still falling when the model grew."""
    mats = [p for n, p in m.named_parameters() if p.dim() == 2 and 'emb' not in n]
    rest = [p for n, p in m.named_parameters() if not (p.dim() == 2 and 'emb' not in n)]
    # Capped at 1.0: scaling DOWN as the model widens is the point, scaling UP a narrower-than-reference model
    # above a rate we never validated is not.
    mult = min(1.0, D_REF / m.cfg.D)
    return torch.optim.Adam([{'params': mats, 'lr': lr * mult, 'lr_mult': mult},
                             {'params': rest, 'lr': lr,        'lr_mult': 1.0}],
                            lr=lr, betas=(0.9, 0.95), fused=True), mult


def main():
    torch.manual_seed(0); np.random.seed(0)
    step = 0
    if os.path.exists(CKPT):
        sd = torch.load(CKPT, map_location=DEV)
        cfg = cfg_from_ckpt(sd, T=CH, res_scale=sd['res_scale'])
        m = C.Core(cfg).to(DEV); m.load_state_dict(sd['model'])
        step = sd['step']; k = sd.get('k', 0); tokens_at_size = sd.get('tokens_at_size', 0)
        print(f"RESUMED at step {step}, {k} growths, {sum(p.numel() for p in m.parameters())/1e6:.0f}M", flush=True)
        opt, _mult = make_opt(m, LR)
        if 'opt' in sd:
            try: opt.load_state_dict(sd['opt'])
            except Exception as e: print(f"  optimizer state not reusable ({type(e).__name__}), starting cold", flush=True)
    else:
        sd = torch.load(SRC, map_location=DEV)
        cfg = cfg_from_ckpt(sd, T=CH, res_scale=sd['res_scale'])
        m = C.Core(cfg).to(DEV); m.load_state_dict(sd['model'])
        P0m = sum(p.numel() for p in m.parameters())
        print(f"STARTING from the 191M (last rung before the run degraded) at {bpb(m):.4f} BPB", flush=True)
        opt, _mult = make_opt(m, LR)
        k = 0
        # CARRY FORWARD WHAT THE 125M ALREADY EARNED. It finished with 6.93B tokens at this size (55 tok/param,
        # 2.5x the floor), so resetting this to zero would force it to re-earn 22 tok/param -- ~6.5 hours of
        # retraining a model that is already fully trained -- before the sensor is even allowed to fire.
        # The floor exists to stop a model growing before it has earned its current size. This one has.
        tokens_at_size = sd.get('tokens_at_size', 0)
        print(f"  carrying forward {tokens_at_size/1e9:.2f}B tokens already earned at this size "
              f"({tokens_at_size/P0m:.0f} tok/param) -- the sensor may fire as soon as it sees saturation",
              flush=True)
    P = sum(p.numel() for p in m.parameters())
    STEPS = int(TPP * END_PARAMS) // (B * CH)       # budget set by the ENDPOINT
    fwd = torch.compile(m)
    # window is now in BLOCKS of `block` steps: 40 x 50 = 2000 steps per window, 3 consecutive disjoint
    # windows = 6000 steps of sustained flat entropy before the model is judged saturated.
    sensor = G.SaturationSensor(window=40, t_stat=1.0, min_gap=2500, floor_tokens=1, block=50)
    print(f"PRAGNOSIA-279  |  now {P/1e6:.1f}M, one rung: 191 -> ~279M (floor {TOK_PER_PARAM_GATE:.0f} tok/param)\n"
          f"  LR {LR:.1e} at reference width {D_REF}; this model gets x{_mult:.3f} on matrices "
          f"= {LR*_mult:.2e}, cosine to {LR_MIN_MULT:.0%} of peak over the run, warmup {WARM} at start "
          f"and after every growth\n"
          f"  budget {TPP:.0f} tok/param OF THE {END_PARAMS/1e6:.0f}M ENDPOINT = {TPP*END_PARAMS/1e9:.1f}B "
          f"tokens = {STEPS} steps\n"
          f"  NOTE: this BPB uses 2048-token windows. It tracks progress and compares runs to each other,\n"
          f"  but it is NOT comparable to GPT-2, which is capped at 1024 context -- use exp/gpt2_bpb.py for that.", flush=True)

    t0 = time.time(); ema = None; sat_before_floor = False
    bar = pbar(range(step, STEPS), "p279f", total=STEPS - step)
    decay_from = None
    stage_start = step        # warmup restarts from here and again after every growth
    gnorms = []               # running grad-norm history for the spike guard
    n_skipped = 0
    for it in bar:
        # WSD: warm up, hold flat, then anneal over the last DECAY_FRAC -- of the planned budget, or from
        # wherever the DECAY flag was dropped, whichever comes first.
        if decay_from is None and os.path.exists(DECAY_FLAG):
            decay_from = it
            bar.write(f"    >>> DECAY TRIGGERED at step {it} -- annealing over the next "
                      f"{int(DECAY_FRAC*STEPS)} steps, then stopping")
        dN = int(DECAY_FRAC * STEPS)
        if decay_from is not None and it >= decay_from + dN:
            bar.write(f"    >>> decay complete at step {it}; saving and stopping"); break
        # WARMUP THEN COSINE ACROSS THE WHOLE BUDGET. The previous run held LR flat until the last 10% and
        # equilibrated at that rate's noise floor for hundreds of thousands of steps. Warmup is measured from
        # `stage_start`, which is reset at every growth, because a just-grown model with a just-carried
        # optimizer is exactly where a full-rate step does the most damage.
        # The schedule value at this point in the RUN, independent of any growth.
        if decay_from is not None:
            sched = LR * max(LR_MIN_MULT, 1.0 - (it - decay_from) / max(1, dN))
        else:
            f = min(1.0, max(0.0, (it - WARM) / max(1, STEPS - WARM)))
            sched = LR * (LR_MIN_MULT + (1 - LR_MIN_MULT) * 0.5 * (1 + math.cos(math.pi * f)))
        # Warmup SCALES the schedule, it does not replace it. Ramping to LR would send a model that had already
        # decayed to 4.1e-4 back up to the 6e-4 peak right after a growth, which is hotter than it was before
        # and precisely the condition that wrecked the previous run. This ramps from ~0 to whatever the
        # schedule says now, and never above it.
        s_in_stage = it - stage_start
        lr = sched * min(1.0, (s_in_stage + 1) / WARM)
        for g in opt.param_groups: g['lr'] = lr * g.get('lr_mult', 1.0)
        X = torch.tensor(dm.batch(B), device=DEV)
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg = fwd(X[:, :CH])
            l = F.cross_entropy(lg.reshape(-1, V), X[:, 1:CH + 1].reshape(-1))
        l.backward()
        gn = float(torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0))
        # SKIP AN ANOMALOUS BATCH RATHER THAN LET IT INTO THE WEIGHTS. Clipping bounds the step but still
        # applies a direction derived from a pathological batch, and Adam then carries that direction in its
        # moments for thousands of steps. Comparing against the running median rather than a fixed threshold
        # means this needs no retuning as the model and loss scale change.
        if len(gnorms) >= 200 and np.isfinite(gn) and gn > GRAD_SPIKE * float(np.median(gnorms)):
            opt.zero_grad(set_to_none=True); n_skipped += 1
            if n_skipped <= 20 or n_skipped % 100 == 0:
                bar.write(f"    grad spike at step {it}: |g|={gn:.2f} vs median "
                          f"{float(np.median(gnorms)):.2f} -- update SKIPPED (total {n_skipped})")
        else:
            opt.step()
        if np.isfinite(gn):
            gnorms.append(gn)
            if len(gnorms) > 1000: gnorms.pop(0)
        ent = G.SaturationSensor.entropy(lg.detach()) if 'lg' in dir() else None
        # FEED THE SENSOR EVERY STEP. This used to sit on the right of an `and` after the floor test, so Python
        # short-circuited it and observe() was never called until the floor had already opened -- after which
        # the sensor still needed its full 400-sample window before it could hold any opinion at all. That is a
        # third way the floor was deciding while the log said the sensor did. The window is now continuous, so
        # by the time the floor opens the sensor already has a formed opinion to act on.
        sat_now = sensor.observe(ent, it, tokens_at_size) if ent is not None else False
        v = l.item(); ema = v if ema is None else 0.99 * ema + 0.01 * v
        tokens_at_size += B * CH
        # THE SENSOR DECIDES WHEN. The floor only decides when it is allowed to.
        Pn = sum(p.numel() for p in m.parameters())
        earned = tokens_at_size / Pn
        floor_open = earned >= TOK_PER_PARAM_GATE
        ceil_hit = earned >= TOK_PER_PARAM_CEIL
        sat_op, t_op = sensor.verdict()          # the sensor's opinion, independent of any floor
        # WHICH CONSTRAINT IS BINDING. If the sensor was already asking to grow before the floor opened, then
        # the FLOOR is what held it back and the floor decided, however loudly the sensor was signalling. Only
        # a sensor that starts asking AFTER the floor is already open has actually made the decision.
        if sat_now and not floor_open: sat_before_floor = True
        if k < len(PLAN) and ent is not None and ((floor_open and sat_now) or ceil_hit):
            decided_by = ("CEILING" if not (floor_open and sat_now)
                          else ("FLOOR (sensor was saturated and waiting)" if sat_before_floor else "SENSOR"))
            amt = PLAN[k]
            # Snapshot Adam's moments BEFORE the tensors are replaced. Growth is function preserving for the
            # weights and has never been for the optimizer: rebuilding Adam zeroes exp_avg_sq for every
            # parameter, so its first post-growth step is near-maximal size on a just-enlarged model. Measured
            # on a small model, the reset takes a 1.5x larger first step than carrying the state does.
            opt_snap = G.snapshot_opt_state(m, opt)
            b_before = bpb(m)
            torch.save({'model': m.state_dict(), 'step': it + 1, 'k': k, 'D': m.cfg.D, 'NL': m.cfg.NL,
                        'DIN': m.cfg.DIN, 'DV': m.cfg.DV, 'tie_emb': True, 'head_scale': m.cfg.head_scale,
                        'D0': 448, 'gated': True, 'qknorm': True, 'qk_learn_scale': True, 'sel_write': True,
                        'res_scale': m.cfg.res_scale}, f"{CKPT[:-3]}_before_grow{k+1}.pt")
            G.grow_width(m, amt[0]); G.grow_depth(m, amt[1]); G.grow_din(m, amt[2])
            b_after = bpb(m)
            k += 1; tokens_at_size = 0
            # ORDER MATTERS: rebuild the groups for the NEW width (so muP rescales the matrices), then carry
            # the moments in, then re-warm. Doing this once, here, rather than creating an unscaled optimizer
            # first and fixing it afterwards.
            stage_start = it + 1
            opt, _mult = make_opt(m, LR)
            n_carried = G.restore_opt_state(m, opt, opt_snap)
            torch.cuda.empty_cache()
            try:
                fwd = torch.compile(m)
                _x = torch.zeros(B, CH, dtype=torch.long, device=DEV)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    F.cross_entropy(fwd(_x).reshape(-1, V), _x.reshape(-1)).backward()
                m.zero_grad(set_to_none=True); opt.zero_grad(set_to_none=True); del _x
            except Exception as e:
                bar.write(f"    compile failed after growth ({type(e).__name__}) -- continuing EAGER"); fwd = m
            bar.write(f"    >>> GREW #{k} at step {it+1}: D{m.cfg.D}x{m.cfg.NL}L DIN{m.cfg.DIN} -> "
                      f"{sum(p.numel() for p in m.parameters())/1e6:.0f}M (earned {earned:.1f} tok/param), "
                      f"BPB {b_before:.4f} -> {b_after:.4f} across the grow")
            # WHO DECIDED. If this says FLOOR or CEILING then the model did not choose and the vision's claim
            # that it grows on its own signal is not demonstrated by this event. Stated at the event so it
            # cannot be inferred wrongly later.
            bar.write(f"        DECIDED BY: {decided_by}   (floor {TOK_PER_PARAM_GATE:.0f}, ceiling "
                      f"{TOK_PER_PARAM_CEIL:.0f}, earned {earned:.1f} tok/param; sensor opinion "
                      f"saturated={sat_op} t={t_op if t_op is None else round(t_op,2)})")
            sensor.reset(); sat_before_floor = False
            bar.write(f"        optimizer state carried: {n_carried} tensors; LR rescaled for width "
                      f"D{m.cfg.D}: matrices x{_mult:.3f} = {LR*_mult:.2e}; warmup {WARM} steps restarts now")
        if (it + 1) % 50 == 0:
            done = (it + 1 - step) * B * CH
            bar.set_postfix_str(f"loss={ema:.3f} ppl={math.exp(min(20,ema)):.0f} lr={lr:.1e} "
                                f"{done/(time.time()-t0)/1e3:.0f}ktok/s {(it+1)*B*CH/1e9:.2f}B "
                                f"free={torch.cuda.mem_get_info()[0]/2**30:.0f}G")
        if (it + 1) % SAVE_EVERY == 0:
            torch.save({'model': m.state_dict(), 'opt': opt.state_dict(), 'step': it + 1,
                        'D': m.cfg.D, 'NL': m.cfg.NL, 'DIN': m.cfg.DIN, 'DV': m.cfg.DV, 'tie_emb': True,
                        'head_scale': m.cfg.head_scale, 'D0': sd['D0'] if 'D0' in sd else 448,
                        'gated': True, 'qknorm': True, 'qk_learn_scale': True, 'sel_write': True,
                        'res_scale': m.cfg.res_scale, 'k': k, 'tokens_at_size': tokens_at_size}, CKPT)
        if (it + 1) % 2000 == 0 and k < len(PLAN):
            bar.write(f"    [growth] earned {earned:.1f} tok/param (floor {TOK_PER_PARAM_GATE:.0f}, ceiling "
                      f"{TOK_PER_PARAM_CEIL:.0f})  sensor: saturated={sat_op} "
                      f"t={t_op if t_op is None else round(t_op,2)}  "
                      f"{'FLOOR OPEN' if floor_open else 'floor closed'}")
        if (it + 1) % EVAL_EVERY == 0:
            b = bpb(m)
            bar.write(f"    step {it+1} | {(it+1)*B*CH/1e9:.2f}B tokens | {(it+1)*B*CH/P:.1f} tok/param "
                      f"| BPB {b:.4f} | (our 2048-ctx metric; see exp/gpt2_bpb.py for the fair GPT-2 comparison)")
    torch.save({'model': m.state_dict(), 'opt': opt.state_dict(), 'step': STEPS,
                'D': m.cfg.D, 'NL': m.cfg.NL, 'DIN': m.cfg.DIN, 'DV': m.cfg.DV, 'tie_emb': True,
                'head_scale': m.cfg.head_scale, 'D0': sd['D0'] if 'D0' in sd else 448,
                'gated': True, 'qknorm': True, 'qk_learn_scale': True, 'sel_write': True,
                'res_scale': m.cfg.res_scale, 'k': k, 'tokens_at_size': tokens_at_size}, CKPT)
    print(f"\nDONE: {STEPS*B*CH/1e9:.2f}B tokens, final BPB {bpb(m):.4f}  (2048-ctx metric, not a GPT-2 comparison)", flush=True)


if __name__ == "__main__":
    main()
