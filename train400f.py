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

  python train400f.py            # resumes automatically
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
GROW = (576, 4, 8)                            # D832x14L DIN32 -> D1408x18L DIN40 = 405.2M
SRC = "/mnt/kv_cache/pragnosia_data/p124f.pt"
CKPT = "/mnt/kv_cache/pragnosia_data/p400f.pt"
CH = 2048
B = int(os.environ.get("B", "8"))
LR = float(os.environ.get("LR", "6e-4"))      # lower than the 124M's 1e-3: this resumes a converged model
TPP = float(os.environ.get("TPP", "22"))      # the earned floor; raise if time allows
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
        step = sd['step']
        print(f"RESUMED at step {step}", flush=True)
        opt = torch.optim.Adam(m.parameters(), lr=LR, betas=(0.9, 0.95)); opt.load_state_dict(sd['opt'])
    else:
        sd = torch.load(SRC, map_location=DEV)
        cfg = cfg_from_ckpt(sd, T=CH, res_scale=sd['res_scale'])
        m = C.Core(cfg).to(DEV); m.load_state_dict(sd['model'])
        P0 = sum(p.numel() for p in m.parameters())
        b0 = bpb(m)
        # FUNCTION-PRESERVING: verify on real data that growth did not change the model, rather than trusting it.
        G.grow_width(m, GROW[0]); G.grow_depth(m, GROW[1]); G.grow_din(m, GROW[2])
        b1 = bpb(m)
        print(f"GROWN {P0/1e6:.1f}M -> {sum(p.numel() for p in m.parameters())/1e6:.1f}M   "
              f"D{m.cfg.D}x{m.cfg.NL}L DIN{m.cfg.DIN}", flush=True)
        print(f"  BPB across the growth: {b0:.4f} -> {b1:.4f}  ({b1-b0:+.4f})   "
              f"{'function-preserving' if abs(b1-b0) < 0.01 else 'WARNING: growth changed the function'}", flush=True)
        opt = torch.optim.Adam(m.parameters(), lr=LR, betas=(0.9, 0.95))
    P = sum(p.numel() for p in m.parameters())
    STEPS = int(TPP * P) // (B * CH)
    fwd = torch.compile(m)
    print(f"PRAGNOSIA-400  |  {P/1e6:.1f}M, budget {TPP:.0f} tok/param = {TPP*P/1e9:.2f}B tokens = {STEPS} steps\n"
          f"  grown from the FINISHED 124M, which keeps its GPT-2-matched claim intact\n"
          f"  target: GPT-2 small scores BPB 0.9335 on this eval set", flush=True)

    t0 = time.time(); ema = None
    bar = pbar(range(step, STEPS), "p400f", total=STEPS - step)
    for it in bar:
        lr = LR * ((it + 1) / WARM if it < WARM else
                   0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (it - WARM) / max(1, STEPS - WARM))))
        for g in opt.param_groups: g['lr'] = lr
        X = torch.tensor(dm.batch(B), device=DEV)
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            l = F.cross_entropy(fwd(X[:, :CH]).reshape(-1, V), X[:, 1:CH + 1].reshape(-1))
        l.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        v = l.item(); ema = v if ema is None else 0.99 * ema + 0.01 * v
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
                        'res_scale': m.cfg.res_scale, 'k': 3}, CKPT)
        if (it + 1) % EVAL_EVERY == 0:
            b = bpb(m)
            bar.write(f"    step {it+1} | {(it+1)*B*CH/1e9:.2f}B tokens | {(it+1)*B*CH/P:.1f} tok/param "
                      f"| BPB {b:.4f} | GPT-2 0.9335 | {'AHEAD' if b < 0.9335 else 'behind'}")
    torch.save({'model': m.state_dict(), 'opt': opt.state_dict(), 'step': STEPS,
                'D': m.cfg.D, 'NL': m.cfg.NL, 'DIN': m.cfg.DIN, 'DV': m.cfg.DV, 'tie_emb': True,
                'head_scale': m.cfg.head_scale, 'D0': sd['D0'] if 'D0' in sd else 448,
                'gated': True, 'qknorm': True, 'qk_learn_scale': True, 'sel_write': True,
                'res_scale': m.cfg.res_scale, 'k': 3}, CKPT)
    print(f"\nDONE: {STEPS*B*CH/1e9:.2f}B tokens, final BPB {bpb(m):.4f}  (GPT-2 0.9335)", flush=True)


if __name__ == "__main__":
    main()
