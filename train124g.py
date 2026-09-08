"""PRAGNOSIA 124M -- matched to GPT-2 124M in parameters AND token budget, so the comparison is clean on both axes.

WHY THIS EXISTS INSTEAD OF train_pragnosia.py. That trainer costs 0.50 BPB versus a plain loop at identical shape,
data and eval (1.7175 vs 1.2179 at 24.5M). Ablation found +0.126 from carried-state training and +0.014 from
recall batches; learning rate and res_scale were ruled out, and ~0.30 remains unexplained. Rather than keep
hunting it, this uses the loop that is MEASURED GOOD and adds nothing that has not been paid for:

  - plain 2048-token windows, no state carried during training   (carried state costs +0.126 BPB)
  - no recall batches                                            (+0.014 BPB, and it is a narrow synthetic task)
  - single cosine over the whole run, LR 1e-3, no per-stage restarts
  - no growth: growth is function-preserving and works (5 clean events measured), but it is a separate claim and
    mixing it in would confound the one number this run exists to produce

The memory faculty is NOT abandoned -- carried-state training is a short fine-tune AFTER this run (finetune_mem.py),
so the +0.126 is paid once at the end rather than across all 9B tokens.

  python train124.py            # train, resuming automatically
  BPB is logged every EVAL_EVERY steps. It uses 2048-token windows, so it is a RUN-TO-RUN metric,
  not a GPT-2 comparison -- GPT-2 cannot exceed 1024 context. exp/gpt2_bpb.py does that comparison fairly.
"""
import sys, os, time, math
sys.path.insert(0, "/opt/code/membrae")
from util import setup_log, pbar
setup_log("train124g")
import numpy as np, torch, torch.nn.functional as F
import core as C, growth as G
from data_mix import DataMix
from tokenizers import Tokenizer
import torch._dynamo as _d; _d.config.recompile_limit = 64
import torch._inductor.config as _ic; _ic.triton.prefer_nd_tiling = True; _ic.compile_threads = 8

DEV = C.DEV
TOK = Tokenizer.from_file("/opt/code/membrae/data/bpe32k.json")
V = 32768
D, NL, DIN = 448, 10, 16                      # INFANT 34.9M; grows to 832x14L DIN24 = 124.8M
TARGET_NL = 14                                # res_scale is pinned to the TARGET depth so growth needs no rescale
PLAN = [('wdi', (160, 2, 4)), ('wdi', (224, 2, 4))]   # 34.9M -> 64.6M -> 124.8M
TOK_PER_PARAM_GATE = 22.0                     # earned floor; the SENSOR decides the actual moment
CH = 2048
B = int(os.environ.get("B", "8"))
LR = float(os.environ.get("LR", "1e-3"))
TPP = float(os.environ.get("TPP", "73"))      # GPT-2 saw ~9B tokens on 124M params
LANG_FRAC = float(os.environ.get("LANG_FRAC", "0.70"))
WARM = 2000
SAVE_EVERY = 2000
EVAL_EVERY = 4000
CKPT = "/mnt/kv_cache/pragnosia_data/p124g.pt"

dm = DataMix(ctx=CH + 1, verbose=True, lang_frac=LANG_FRAC)
np.random.seed(777)
EV = [torch.tensor(dm.batch(2), device=DEV) for _ in range(10)]


def build():
    cfg = C.Config(D=D, DIN=DIN, DV=D, NL=NL, T=CH, res_scale=TARGET_NL ** -0.5, tie_emb=True)
    cfg.gated = True; cfg.qknorm = True
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
    P_FINAL = 124_800_000                      # the budget is set by the ENDPOINT so total tokens match GPT-2
    STEPS = int(TPP * P_FINAL) // (B * CH)
    opt = torch.optim.Adam(m.parameters(), lr=LR, betas=(0.9, 0.95), fused=True)
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
        opt = torch.optim.Adam(m.parameters(), lr=LR, betas=(0.9, 0.95), fused=True); opt.load_state_dict(sd['opt'])
        step = sd['step']; tokens_at_size = sd.get('tokens_at_size', 0)
        print(f"RESUMED at step {step}/{STEPS}, {k} grows, {P()/1e6:.0f}M", flush=True)
    fwd = torch.compile(m)
    sensor = G.SaturationSensor(window=400, t_stat=1.0, min_gap=2500, floor_tokens=1)
    print(f"PRAGNOSIA-124 GROWN: infant D{D}x{NL}L DIN{DIN} = {P()/1e6:.1f}M -> 124.8M  (GPT-2: 124M)\n"
          f"  budget {TPP:.0f} tok/param of the ENDPOINT = {TPP*P_FINAL/1e9:.2f}B tokens = {STEPS} steps\n"
          f"  plain loop: no carried state, no recall batches, single cosine at LR {LR}\n"
          f"  NOTE: this BPB uses 2048-token windows. It tracks progress and compares runs to each other,\n"
          f"  but it is NOT comparable to GPT-2, which is capped at 1024 context -- use exp/gpt2_bpb.py for that.", flush=True)

    t0 = time.time(); ema = None
    bar = pbar(range(step, STEPS), "p124g", total=STEPS - step)
    for it in bar:
        lr = LR * ((it + 1) / WARM if it < WARM else
                   0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (it - WARM) / max(1, STEPS - WARM))))
        for g in opt.param_groups: g['lr'] = lr
        X = torch.tensor(dm.batch(B), device=DEV)
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg = fwd(X[:, :CH])
            l = F.cross_entropy(lg.reshape(-1, V), X[:, 1:CH + 1].reshape(-1))
        l.backward()
        ent = G.SaturationSensor.entropy(lg.detach())      # the sensor's signal is ENTROPY, not the loss
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        v = l.item(); ema = v if ema is None else 0.99 * ema + 0.01 * v
        tokens_at_size += B * CH
        # INTRINSIC GROWTH (VISION 0a): the sensor decides WHEN; the 22 tok/param floor only decides when it is
        # ALLOWED to. A hand-set schedule would be exactly the decision-scaffolding the vision forbids.
        earned_ok = tokens_at_size >= TOK_PER_PARAM_GATE * P()
        if k < len(PLAN) and sensor.observe(ent, it, tokens_at_size if earned_ok else 0) and earned_ok:
            kind, amt = PLAN[k]; earned = tokens_at_size / P()
            torch.save({'model': m.state_dict(), 'step': it + 1, 'k': k, 'D': m.cfg.D, 'NL': m.cfg.NL,
                        'DIN': m.cfg.DIN, 'DV': m.cfg.DV, 'tie_emb': True, 'head_scale': m.cfg.head_scale, 'D0': D,
                        'gated': True, 'qknorm': True, 'res_scale': m.cfg.res_scale},
                       f"{CKPT[:-3]}_before_grow{k+1}.pt")
            G.grow_width(m, amt[0]); G.grow_depth(m, amt[1]); G.grow_din(m, amt[2])
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
            bar.write(f"    >>> SELF-GREW #{k} at step {it+1}: D{m.cfg.D}x{m.cfg.NL}L DIN{m.cfg.DIN} "
                      f"-> {P()/1e6:.0f}M (earned {earned:.0f} tok/param)")
        if (it + 1) % 50 == 0:
            done = (it + 1 - step) * B * CH
            free = torch.cuda.mem_get_info()[0] / 2 ** 30
            bar.set_postfix_str(f"loss={ema:.3f} ppl={math.exp(min(20,ema)):.0f} lr={lr:.1e} "
                                f"{done/(time.time()-t0)/1e3:.0f}ktok/s {(it+1)*B*CH/1e9:.2f}B free={free:.0f}G")
        if (it + 1) % SAVE_EVERY == 0:
            torch.save({'model': m.state_dict(), 'opt': opt.state_dict(), 'step': it + 1,
                        'D': m.cfg.D, 'NL': m.cfg.NL, 'DIN': m.cfg.DIN, 'DV': m.cfg.DV, 'tie_emb': True, 'head_scale': m.cfg.head_scale, 'D0': D,
                        'gated': True, 'qknorm': True, 'res_scale': m.cfg.res_scale,
                        'k': k, 'tokens_at_size': tokens_at_size}, CKPT)
        if (it + 1) % EVAL_EVERY == 0:
            b = bpb(m, m.cfg)
            bar.write(f"    step {it+1} | {(it+1)*B*CH/1e9:.2f}B tokens | {(it+1)*B*CH/P_FINAL:.1f} tok/param "
                      f"| BPB {b:.4f} | (our 2048-ctx metric; see exp/gpt2_bpb.py for the fair GPT-2 comparison)")
    torch.save({'model': m.state_dict(), 'opt': opt.state_dict(), 'step': STEPS,
                'D': m.cfg.D, 'NL': m.cfg.NL, 'DIN': m.cfg.DIN, 'DV': m.cfg.DV, 'tie_emb': True, 'head_scale': m.cfg.head_scale, 'D0': D,
                'gated': True, 'qknorm': True, 'res_scale': m.cfg.res_scale,
                        'k': k, 'tokens_at_size': tokens_at_size}, CKPT)
    print(f"\nDONE: {STEPS*B*CH/1e9:.2f}B tokens, final BPB {bpb(m, m.cfg):.4f}  (2048-ctx metric, not a GPT-2 comparison)", flush=True)


if __name__ == "__main__":
    main()
