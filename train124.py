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
  BPB is logged every EVAL_EVERY steps so progress is visible against GPT-2's 0.9335.
"""
import sys, os, time, math
sys.path.insert(0, "/opt/code/membrae")
from util import setup_log, pbar
setup_log("train124")
import numpy as np, torch, torch.nn.functional as F
import core as C
from data_mix import DataMix
from tokenizers import Tokenizer
import torch._dynamo as _d; _d.config.recompile_limit = 64
import torch._inductor.config as _ic; _ic.triton.prefer_nd_tiling = True; _ic.compile_threads = 8

DEV = C.DEV
TOK = Tokenizer.from_file("/opt/code/membrae/data/bpe32k.json")
V = 32768
D, NL, DIN = 832, 14, 24                      # 124.8M, matching GPT-2's 124M
CH = 2048
B = int(os.environ.get("B", "8"))
LR = float(os.environ.get("LR", "1e-3"))
TPP = float(os.environ.get("TPP", "73"))      # GPT-2 saw ~9B tokens on 124M params
LANG_FRAC = float(os.environ.get("LANG_FRAC", "0.70"))
WARM = 2000
SAVE_EVERY = 2000
EVAL_EVERY = 4000
CKPT = "/mnt/kv_cache/pragnosia_data/p124.pt"

dm = DataMix(ctx=CH + 1, verbose=True, lang_frac=LANG_FRAC)
np.random.seed(777)
EV = [torch.tensor(dm.batch(2), device=DEV) for _ in range(10)]


def build():
    cfg = C.Config(D=D, DIN=DIN, DV=D, NL=NL, T=CH, res_scale=NL ** -0.5, tie_emb=True)
    cfg.gated = True; cfg.qknorm = True
    return C.Core(cfg).to(DEV), cfg


@torch.no_grad()
def bpb(m, cfg):
    """tokenizer-independent, so it is directly comparable to GPT-2's 0.9335 on this same text"""
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
    P = sum(p.numel() for p in m.parameters())
    STEPS = int(TPP * P) // (B * CH)
    opt = torch.optim.Adam(m.parameters(), lr=LR, betas=(0.9, 0.95))
    step = 0
    if os.path.exists(CKPT):
        sd = torch.load(CKPT, map_location=DEV)
        m.load_state_dict(sd['model']); opt.load_state_dict(sd['opt']); step = sd['step']
        print(f"RESUMED at step {step}/{STEPS} ({step*B*CH/1e9:.2f}B tokens)", flush=True)
    fwd = torch.compile(m)
    print(f"PRAGNOSIA-124: D{D}x{NL}L DIN{DIN} = {P/1e6:.1f}M params  (GPT-2: 124M)\n"
          f"  budget {TPP:.0f} tok/param = {TPP*P/1e9:.2f}B tokens = {STEPS} steps at B{B} x CH{CH}\n"
          f"  plain loop: no carried state, no recall batches, single cosine at LR {LR}\n"
          f"  target: GPT-2 small scores BPB 0.9335 on this eval set", flush=True)

    t0 = time.time(); ema = None
    bar = pbar(range(step, STEPS), "p124", total=STEPS - step)
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
            free = torch.cuda.mem_get_info()[0] / 2 ** 30
            bar.set_postfix_str(f"loss={ema:.3f} ppl={math.exp(min(20,ema)):.0f} lr={lr:.1e} "
                                f"{done/(time.time()-t0)/1e3:.0f}ktok/s {(it+1)*B*CH/1e9:.2f}B free={free:.0f}G")
        if (it + 1) % SAVE_EVERY == 0:
            torch.save({'model': m.state_dict(), 'opt': opt.state_dict(), 'step': it + 1,
                        'D': D, 'NL': NL, 'DIN': DIN, 'DV': D, 'tie_emb': True, 'head_scale': 1.0,
                        'gated': True, 'qknorm': True, 'res_scale': cfg.res_scale}, CKPT)
        if (it + 1) % EVAL_EVERY == 0:
            b = bpb(m, cfg)
            bar.write(f"    step {it+1} | {(it+1)*B*CH/1e9:.2f}B tokens | {(it+1)*B*CH/P:.1f} tok/param "
                      f"| BPB {b:.4f} | GPT-2 0.9335 | {'AHEAD' if b < 0.9335 else 'behind'}")
    torch.save({'model': m.state_dict(), 'opt': opt.state_dict(), 'step': STEPS,
                'D': D, 'NL': NL, 'DIN': DIN, 'DV': D, 'tie_emb': True, 'head_scale': 1.0,
                'gated': True, 'qknorm': True, 'res_scale': cfg.res_scale}, CKPT)
    print(f"\nDONE: {STEPS*B*CH/1e9:.2f}B tokens, final BPB {bpb(m, cfg):.4f}  (GPT-2 0.9335)", flush=True)


if __name__ == "__main__":
    main()
