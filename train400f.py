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
PLAN = [(160, 2, 0),      # D832x14L DIN32 125.0M -> D992x16L DIN32 ~190M
        (192, 1, 8),      # -> D1184x17L DIN40 ~285M
        (224, 1, 0)]      # -> D1408x18L DIN40  405.2M
TOK_PER_PARAM_GATE = 22.0    # FLOOR: the earliest growth is ALLOWED. Stops growing before the size is earned.
TOK_PER_PARAM_CEIL = 40.0    # CEILING: growth is FORCED. A sensor that never fires must not be able to stall
                             # the run: the endpoint (405M) is a deliverable and cannot depend on a heuristic
                             # firing. Between floor and ceiling the SENSOR decides, and the log says which.
SRC = "/mnt/kv_cache/pragnosia_data/p124f.pt"
CKPT = "/mnt/kv_cache/pragnosia_data/p400f.pt"
CH = 2048
B = int(os.environ.get("B", "8"))
# FUSED ADAM: the optimizer update was 13.9% of a COMPILED step, spent in multi_tensor_apply -- a separate
# kernel launch per parameter group per moment. torch's fused Adam does the whole update in one kernel and
# measured 9.0% off the step time (65.3 -> 59.4 ms) with identical maths. Free, so it is on.
LR = float(os.environ.get("LR", "6e-4"))      # lower than the 124M's 1e-3: this resumes a converged model
TPP = float(os.environ.get("TPP", "73"))      # matched to the 125M's budget; WSD lets us stop anywhere
# WSD (warmup-stable-decay) instead of cosine, so the run is NOT committed to a token budget in advance.
# Cosine bakes the endpoint into the schedule: stop early and you stop mid-schedule at a high LR, with a model
# that was never annealed, which wastes the run. WSD holds LR flat and anneals only at the end, so the budget can
# be decided by the clock rather than guessed now. To finish early, write the file below and it starts the decay
# from that step; without it, decay begins at 90% of TPP.
DECAY_FLAG = "/mnt/kv_cache/pragnosia_data/p400f_DECAY_NOW"
DECAY_FRAC = 0.10
WARM = 1000
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


def main():
    torch.manual_seed(0); np.random.seed(0)
    step = 0
    if os.path.exists(CKPT):
        sd = torch.load(CKPT, map_location=DEV)
        cfg = cfg_from_ckpt(sd, T=CH, res_scale=sd['res_scale'])
        m = C.Core(cfg).to(DEV); m.load_state_dict(sd['model'])
        step = sd['step']; k = sd.get('k', 0); tokens_at_size = sd.get('tokens_at_size', 0)
        print(f"RESUMED at step {step}, {k} growths, {sum(p.numel() for p in m.parameters())/1e6:.0f}M", flush=True)
        opt = torch.optim.Adam(m.parameters(), lr=LR, betas=(0.9, 0.95), fused=True); opt.load_state_dict(sd['opt'])
    else:
        sd = torch.load(SRC, map_location=DEV)
        cfg = cfg_from_ckpt(sd, T=CH, res_scale=sd['res_scale'])
        m = C.Core(cfg).to(DEV); m.load_state_dict(sd['model'])
        P0m = sum(p.numel() for p in m.parameters())
        print(f"STARTING from the FINISHED 125M at {bpb(m):.4f} BPB -- growth is sensor-driven from here",
              flush=True)
        opt = torch.optim.Adam(m.parameters(), lr=LR, betas=(0.9, 0.95), fused=True)
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
    STEPS = int(TPP * 405_200_000) // (B * CH)      # budget set by the ENDPOINT, as the 124M run did
    fwd = torch.compile(m)
    # window is now in BLOCKS of `block` steps: 40 x 50 = 2000 steps per window, 3 consecutive disjoint
    # windows = 6000 steps of sustained flat entropy before the model is judged saturated.
    sensor = G.SaturationSensor(window=40, t_stat=1.0, min_gap=2500, floor_tokens=1, block=50)
    print(f"PRAGNOSIA-400  |  now {P/1e6:.1f}M, ladder "
          f"125 -> ~190 -> ~285 -> 405M on the SENSOR's signal (floor {TOK_PER_PARAM_GATE:.0f} tok/param)\n"
          f"  budget {TPP:.0f} tok/param OF THE ENDPOINT = {TPP*405.2e6/1e9:.1f}B tokens = {STEPS} steps; "
          f"WSD, so it can be stopped early and annealed cleanly\n"
          f"  grown from the FINISHED 124M, which keeps its GPT-2-matched claim intact\n"
          f"  NOTE: this BPB uses 2048-token windows. It tracks progress and compares runs to each other,\n"
          f"  but it is NOT comparable to GPT-2, which is capped at 1024 context -- use exp/gpt2_bpb.py for that.", flush=True)

    t0 = time.time(); ema = None; sat_before_floor = False
    bar = pbar(range(step, STEPS), "p400f", total=STEPS - step)
    decay_from = None
    for it in bar:
        # WSD: warm up, hold flat, then anneal over the last DECAY_FRAC -- of the planned budget, or from
        # wherever the DECAY flag was dropped, whichever comes first.
        if decay_from is None and os.path.exists(DECAY_FLAG):
            decay_from = it
            bar.write(f"    >>> DECAY TRIGGERED at step {it} -- annealing over the next "
                      f"{int(DECAY_FRAC*STEPS)} steps, then stopping")
        d0 = decay_from if decay_from is not None else int((1 - DECAY_FRAC) * STEPS)
        dN = int(DECAY_FRAC * STEPS)
        if it < WARM:            lr = LR * (it + 1) / WARM
        elif it < d0:            lr = LR
        else:                    lr = LR * max(0.02, 1.0 - (it - d0) / max(1, dN))
        if decay_from is not None and it >= decay_from + dN:
            bar.write(f"    >>> decay complete at step {it}; saving and stopping"); break
        for g in opt.param_groups: g['lr'] = lr
        X = torch.tensor(dm.batch(B), device=DEV)
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg = fwd(X[:, :CH])
            l = F.cross_entropy(lg.reshape(-1, V), X[:, 1:CH + 1].reshape(-1))
        l.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
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
            b_before = bpb(m)
            torch.save({'model': m.state_dict(), 'step': it + 1, 'k': k, 'D': m.cfg.D, 'NL': m.cfg.NL,
                        'DIN': m.cfg.DIN, 'DV': m.cfg.DV, 'tie_emb': True, 'head_scale': m.cfg.head_scale,
                        'D0': 448, 'gated': True, 'qknorm': True, 'qk_learn_scale': True, 'sel_write': True,
                        'res_scale': m.cfg.res_scale}, f"{CKPT[:-3]}_before_grow{k+1}.pt")
            G.grow_width(m, amt[0]); G.grow_depth(m, amt[1]); G.grow_din(m, amt[2])
            b_after = bpb(m)
            k += 1; tokens_at_size = 0
            opt = torch.optim.Adam(m.parameters(), lr=lr, betas=(0.9, 0.95), fused=True)
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
