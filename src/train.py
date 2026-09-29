"""Train the LM on the tokenized slice. One file, one loop, checkpoints so intermediate states can be probed.

Run:  .venv/bin/python -m src.train --steps 120000 --out /mnt/kv_cache/run1
"""
from __future__ import annotations
import argparse, gc, json, math, os, time
import numpy as np
import torch
import torch.nn.functional as F

from .common import Run, cap_gpu, seed_all, require_prod, eval_bpb
from .model import LM
from .optim import Fabric

MIX = {"dclm": .413, "fineweb-edu": .322, "starcoderdata": .130, "finemath": .036,
       "open-web-math": .030, "arxiv": .030, "wikipedia": .023, "algebraic-stack": .015}


def loaders(data, only=None):
    """only: restrict to these sources, renormalising the mix. Used for a long-context phase, where the point
    is to train on windows that lie inside ONE document: 21% of documents in the full mix exceed 2048 tokens,
    but arxiv's median is 12,710 and algebraic-stack's 10,375."""
    bins = {k: np.memmap(f"{data}/{k}.bin", dtype=np.uint16, mode="r")
            for k in MIX if os.path.exists(f"{data}/{k}.bin") and (only is None or k in only)}
    # a corpus outside the designed mix (e.g. eval/build_recall.py's recall.bin) is loaded too, at weight 1 before
    # normalising; it used to be skipped silently, leaving an empty corpus that failed at the first sample
    if only is None:
        for f in sorted(os.listdir(data)):
            k = f[:-4]
            if f.endswith(".bin") and k not in bins and k not in MIX:
                bins[k] = np.memmap(f"{data}/{f}", dtype=np.uint16, mode="r")
    ks = list(bins)
    ps = np.array([MIX.get(k, 1.0) for k in ks], dtype=np.float64); ps /= ps.sum()
    return bins, ks, ps


def growth_signal(m, op, x, fused_scan=True, chunks=4):
    """SATURATION, MEASURED BY THE MODEL ON ITSELF (goal 7). Grow a throwaway copy by `op` ("wN" or "d"), which is
    exact, so the copy computes the same function; take ONE gradient on the batch x; compare the mean squared
    gradient per parameter on the NEW weights with the same on the EXISTING ones. Above 1, a new parameter would be
    pushed harder than an average existing one: the current capacity is saturated relative to what growth offers.
    The break-even 1 is the only threshold, and it is not tuned. Idea after GradMax (Evci et al. 2022), which uses
    the new weights' gradient to decide HOW to grow; here it decides WHEN."""
    import copy
    c = copy.deepcopy(m)
    if op == "d":
        g_ = c.deepen(); newset = {id(p) for i in g_.new_layers for p in list(g_.mix[i].parameters()) + list(g_.mlp[i].parameters())}
        regions = None
    else:
        grown = c.grow(int(op[1:])); g_ = c; newset = set()
        regions = {id(p): sh for p, (_, sh) in grown.items()}
    if fused_scan:
        for mx in g_.mix:
            if hasattr(mx, "need_sent"): mx.need_sent = False
    g_.train(); g_.zero_grad(set_to_none=True)
    T = x.shape[1] - 1
    with torch.autocast("cuda", dtype=torch.bfloat16):
        h = g_.trunk(x[:, :T])
    g_.loss(h, x[:, 1:T + 1], chunks=chunks).backward()
    # LIKE FOR LIKE. At an exact growth only the zero-initialised OUTPUT projections of the new capacity receive
    # any gradient (everything behind them is multiplied by zero), so averaging over all new weights diluted the new
    # side with exact zeros: measured, that ratio plateaued at 0.10-0.14 for 12,000 steps and could never reach 1.
    # So the comparison is between output projections only (each layer's mixer .o and MLP .b), new against existing.
    def role(n): return n.endswith(".o.weight") or n.endswith(".b.weight")
    ns = nn_ = os_ = on = 0.0
    for n, p in g_.named_parameters():
        if p.grad is None or not role(n): continue
        g2 = p.grad.float().square()
        if id(p) in newset:
            ns += float(g2.sum()); nn_ += g2.numel()
        elif regions is not None and id(p) in regions:
            old = g2[tuple(slice(0, k) for k in regions[id(p)])]
            os_ += float(old.sum()); on += old.numel()
            ns += float(g2.sum()) - float(old.sum()); nn_ += g2.numel() - old.numel()
        else:
            os_ += float(g2.sum()); on += g2.numel()
    del c, g_; torch.cuda.empty_cache()
    return (ns / max(1, nn_)) / max(1e-30, os_ / max(1, on))


def batch(bins, ks, ps, rng, B, T, extra=None, extra_p=0.0):
    """extra: a SECOND corpus, drawn for extra_p of the sequences, with its own mix inside it. Used to raise
    the weight of a form the main mix is poor in without rebuilding the main mix: measured 2026-09-21, labels
    and lists occur 0.22-0.55 times per million bytes in ours, and a model that could condition on a label in
    the prompt at 0.66B tokens could no longer do it at 10B."""
    o = np.empty((B, T + 1), dtype=np.int64)
    for i in range(B):
        bn, kn, pn = (extra if (extra is not None and rng.random() < extra_p) else (bins, ks, ps))
        a = bn[kn[rng.choice(len(kn), p=pn)]]
        j = rng.integers(0, len(a) - T - 2)
        o[i] = a[j:j + T + 1]
    # pinned + non-blocking: a pageable .cuda() waits for every kernel already queued before it returns, so
    # the host could never run ahead of the card and the card idled through each step's sampling.
    return torch.from_numpy(o).pin_memory().cuda(non_blocking=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=120000)
    ap.add_argument("--d", type=int, default=768)
    ap.add_argument("--layers", type=int, default=12)
    ap.add_argument("--mlp", type=int, default=4)
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=6e-3)
    ap.add_argument("--warm", type=int, default=500)
    ap.add_argument("--data", default="/mnt/kv_cache/tok1b_v8192")
    ap.add_argument("--sources", default="", help="comma-separated subset of the mix (long-context phase)")
    ap.add_argument("--data2", default="", help="a second corpus directory, sampled for --data2-frac of the sequences")
    ap.add_argument("--data2-frac", type=float, default=0.0)
    ap.add_argument("--ul", type=float, default=0.0, help="unlikelihood weight on recently emitted tokens (0 = off)")
    ap.add_argument("--optim", default="fabric", choices=["fabric", "psf", "sfp"],
                    help="psf: schedule-free Prodigy (Defazio 2024; Mishchenko & Defazio 2023), every setting at its "
                         "published default: no rate, no schedule, no warm-up; evaluation and checkpoints use the averaged weights")
    ap.add_argument("--intrinsic-lr", type=int, default=0,
                    help="the optimizer sets its own step size (goal 8), no warm-up, no schedule: 1 = hypergradient "
                         "(--lr/--new-lr are only where it starts), 2 = layer-wise DoG (no rate at all)")
    ap.add_argument("--hyper-beta", type=float, default=0.02, help="how fast an intrinsic rate moves per step, per unit cosine")
    ap.add_argument("--grow-auto", default="",
                    help="growth events in ORDER, fired when the model saturates rather than at set steps (goal 7, "
                         "needs --intrinsic-lr 1): an event fires when the rate has annealed one e-fold below where "
                         "the stage started; the rates then restart there. Events comma-separated, ops in one event "
                         "joined by '+', e.g. w768,d,a2+w1024")
    ap.add_argument("--trial-steps", type=int, default=1000,
                    help="with --grow-auto: length of each branch of a growth TRIAL (grow vs stay, same batches); the "
                         "model grows only if growing reduced held-out loss more per unit of compute. A measurement "
                         "window, not a threshold")
    ap.add_argument("--deepen-init", default="zero", choices=["zero", "copy", "gate"],
                    help="new layers at a deepen: zero = exact (outputs start at zero); copy = stacking, each new layer "
                         "a copy of the nearest trained layer of its type, with its optimizer memory")
    ap.add_argument("--growth-probe", default="", help="log growth_signal() for these ops every 1,000 steps, e.g. w768,d")
    ap.add_argument("--new-warm", type=int, default=1000, help="steps over which layers added mid-run ramp from 0 to --new-lr")
    ap.add_argument("--new-lr", type=float, default=3e-4, help="absolute step for layers added by LM.deepen(), which start at zero output")
    ap.add_argument("--norm", default="ln", choices=["ln", "rms"], help="residual-stream norm; rms makes widening exact")
    ap.add_argument("--grow-schedule", default="", help="growth events as STEP:OP, OP = wN (widen to N), d (double depth) or aN (split each micro-step N ways, N x the accumulation: same tokens per update, less activation memory), e.g. 45750:w768,91500:d,137250:w1024")
    ap.add_argument("--deepen-at", type=int, default=0, help="step at which to double the depth in place with LM.deepen() (0: never)")
    ap.add_argument("--fused-proj", type=int, default=0, help="one matmul for the carrier's five input projections and attention's q,k,v (same function)")
    ap.add_argument("--fused-scan", type=int, default=0, help="carrier scan through flash-linear-attention's fused kernel "
                    "(identical maths, 1.5-1.7x faster); switches off the attention layers' state read, which needs "
                    "per-chunk states that kernel does not produce")
    ap.add_argument("--ul-window", type=int, default=128, help="how far back the unlikelihood term looks")
    ap.add_argument("--vocab", type=int, default=8192, help="must match the tokenizer the data was made with")
    ap.add_argument("--chunks", type=int, default=1, help="head chunks for the loss; >1 never holds all logits at once")
    ap.add_argument("--accum", type=int, default=1, help="gradient-accumulation steps: --batch is the MICRO-batch; tokens per update = accum*batch*ctx. At ctx 2048 the 78M carrier peaks at 20.2 GiB with batch 16 and 5.9 GiB with batch 4 x accum 4, at 88% of the speed")
    ap.add_argument("--scan-chunk", type=int, default=0, help="associative scan chunk (0: the carrier's default)")
    ap.add_argument("--recompute", type=int, default=0, help="recompute each scan chunk in backward: at ctx 8192 this halves memory (25.4 -> 14.9 GiB) for 2% of speed")
    ap.add_argument("--out", default="/mnt/kv_cache/run1")
    ap.add_argument("--cap", type=float, default=20.0)
    ap.add_argument("--save-every", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--compile", type=int, default=1)
    ap.add_argument("--grow-at", type=int, default=0, help="step at which to widen the model in place (0: never)")
    ap.add_argument("--grow-to", type=int, default=0, help="width to grow to at --grow-at")
    ap.add_argument("--fused", type=int, default=1, help="one compiled kernel group per parameter update")
    ap.add_argument("--graph", type=int, default=1, help="replay the optimizer step as a CUDA graph")
    ap.add_argument("--ctx-mix", default="", help="train on a MIX of context lengths, e.g. 1024,2048,4096,8192; "
                    "the batch scales so every length carries the same tokens per micro-step")
    ap.add_argument("--graph-step", type=int, default=1,
                    help="capture forward+loss+backward as ONE CUDA graph and replay it per micro-batch")
    ap.add_argument("--resume", default="")
    ap.add_argument("--mixer", default="spin", choices=["spin", "gspin", "fspin", "aspin", "dspin", "wspin", "xspin", "w2spin", "hspin", "hdspin", "sdspin", "altspin", "alt4spin", "rspin", "tspin", "trspin", "aspin1", "hspin1", "r6spin", "grspin", "attn"])
    ap.add_argument("--spike", type=float, default=4.0)
    ap.add_argument("--eval-every", type=int, default=2000)
    a = ap.parse_args()
    # the positional table is sized to the model's context, so a mix must build the model at its LONGEST:
    # capturing a graph at 4,096 on a model with 2,048 rows indexes past the table and asserts on the device
    # ...but never SHRINK it below what was asked for. Resuming a 32,768-window checkpoint with a mix topping out
    # at 8,192 used to rebuild the model at 8,192, truncate the positional table, leave the optimizer's statistics
    # for it at 32,768 rows, fail to load them, start cold, and then crash compiling the fused update against the
    # mismatched shapes (2026-09-23). A smaller mix simply uses the first rows of a larger table.
    if a.ctx_mix:
        a.ctx = max(a.ctx, max(int(v) for v in a.ctx_mix.split(",") if v))
    if a.fused_proj:
        import src.assoc as _A0
        _A0.FUSED_PROJ = True
    # THE GROWTH SCHEDULE: step -> widen to N, and steps at which to double the depth. The older single-event
    # flags fold into it.
    grow_w, grow_d, grow_a = {}, set(), {}
    for ev in [e for e in a.grow_schedule.split(",") if e]:
        st_, op_ = ev.split(":")
        if op_ == "d": grow_d.add(int(st_))
        elif op_.startswith("a"): grow_a[int(st_)] = int(op_[1:])
        elif op_.startswith("w"): grow_w[int(st_)] = int(op_[1:])
        else: raise ValueError(f"unknown growth op {op_!r} in --grow-schedule")
    # sfp: the same schedule-free Prodigy as psf, fused into Fabric's compiled, CUDA-graphed step (goal: training
    # efficiency; psf's per-tensor Python update cost 15% of step time at 46M). SF marks behaviour shared by both.
    SF = a.optim in ("psf", "sfp")
    auto_events = [e.split("+") for e in a.grow_auto.split(",") if e]
    if auto_events and a.intrinsic_lr != 1 and not SF:
        raise ValueError("--grow-auto needs a model-set step size: --intrinsic-lr 1 or --optim psf")
    if a.grow_at: grow_w[a.grow_at] = a.grow_to
    if a.deepen_at: grow_d.add(a.deepen_at)
    os.makedirs(a.out, exist_ok=True)

    require_prod()
    with Run("train", note=f"d{a.d} L{a.layers} mlp{a.mlp} ctx{a.ctx} B{a.batch} "
                           f"{a.steps} steps -> {a.out}", seed=a.seed) as r:
        cap_gpu(a.cap)
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(f"/mnt/kv_cache/tok_v{a.vocab}.json")
        assert tok.get_vocab_size() == a.vocab, (tok.get_vocab_size(), a.vocab)
        bins, ks, ps = loaders(a.data, only=[x for x in a.sources.split(",") if x] or None)
        r.log(f"corpora {ks}")
        extra = None
        if a.data2:
            extra = loaders(a.data2)
            r.log(f"second corpus {a.data2} at {a.data2_frac:.0%}: {extra[1]}")
        m = LM(vocab=a.vocab, d=a.d, layers=a.layers, mlp_mult=a.mlp, ctx=a.ctx, mixer=a.mixer, norm=a.norm).cuda()
        if a.fused_scan:
            import src.assoc as _A
            _A.FUSED_SCAN = True
            for _mx in m.mix:
                if hasattr(_mx, "need_sent"): _mx.need_sent = False
            r.log("fused scan on: carrier through flash-linear-attention, attention state read off")
        # The carrier is 94% of an uncompiled step (195 ms of 206), because its cross-chunk pass is a python
        # loop issuing many tiny kernels. Compiling fuses them: 40 -> 154 ktok/s measured, a 3.8x speedup.
        for mx in m.mix:
            if a.scan_chunk and hasattr(mx, "chunk"): mx.chunk = a.scan_chunk
            if a.recompute and hasattr(mx, "recompute"): mx.recompute = True
        fwd = torch.compile(m) if a.compile else m
        trunk = torch.compile(m.trunk) if a.compile else m.trunk
        P = sum(p.numel() for p in m.parameters())
        car = sum(p.numel() for n, p in m.named_parameters() if "mix." in n)
        base = None                                  # set from the optimizer's own groups, just below
        # The eager optimizer was CPU-bound: 19.5 ms issuing ~4,500 kernels per step. Fused and graphed it is
        # a few hundred kernels replayed without python; grads must then be zeroed in place, not set to None.
        def make_opt(model):
            if a.optim == "psf":
                from prodigyplus import ProdigyPlusScheduleFree
                o_ = ProdigyPlusScheduleFree([{"params": g["params"]} for g in model.param_groups(lr=a.lr, new_lr=a.new_lr)])
                o_.train(); o_.cuda_graph = False; o_.recaptures = 0
                o_.state_bytes = lambda: sum(v.numel() * v.element_size() for st in o_.state.values()
                                             for v in st.values() if torch.is_tensor(v))
                return o_
            if a.optim == "sfp":
                o_ = Fabric([{"params": g["params"]} for g in model.param_groups(lr=a.lr, new_lr=a.new_lr)], lr=1.0,
                            fused=bool(a.fused), sfp=True, cuda_graph=bool(a.graph and a.fused))
                o_.recaptures = getattr(o_, "recaptures", 0)
                return o_
            return Fabric(model.param_groups(lr=a.lr, new_lr=a.new_lr), lr=a.lr, fused=bool(a.fused),
                          hyper=a.intrinsic_lr == 1, hyper_beta=a.hyper_beta, dog=a.intrinsic_lr == 2,
                          cuda_graph=bool(a.graph and a.fused))
        D_KEYS = ("d", "d_prev", "d_numerator", "d_denom", "running_d_numerator", "running_d_denom",
                  "max_d_numerator", "prev_d_numerator")
        def carry_d(old, new):
            # A GROWN MODEL KEEPS THE STEP SIZE IT HAD LEARNED. Rebuilt from scratch, Prodigy restarts at d0 = 1e-6
            # and the grown copy spent a whole 300-step growth trial warming up (grown 5.3377 -> 5.3090 while
            # unchanged went to 5.1364, 2026-09-27): a trial biased against growth by the optimizer, not the model.
            # Only the step-size estimate is carried; the schedule-free average restarts at the grown weights, which
            # compute the same function. A group new to the model (added layers) takes the main group's estimate.
            for gi, g in enumerate(new.param_groups):
                src = old.param_groups[gi] if gi < len(old.param_groups) else old.param_groups[0]
                for k in D_KEYS:
                    if k in src: g[k] = src[k]
        def psf_carry(old, new, grown=None, pairs=None):
            """CARRY SCHEDULE-FREE PRODIGY'S MEMORY ACROSS A GROWTH, as Fabric.grew does for Fabric. Rebuilding it fresh
            restarted every weight's statistics, which hurt the grown branch of a trial more than the unchanged one
            (more weights to disturb): the grown copy never beat the unchanged one in five trials (2026-09-27).
              same-shape weights (all of a deepen's old layers; weights a widen leaves alone): state moved as is
              widened weights: z (the schedule-free base) takes the old values in the old slots and the grown weights'
              values in the new ones, so new slots start consistent (x = y = z); the second moment is padded with its
              median; the distance baseline for Prodigy (p0, s) restarts for that tensor -- the group's step size d,
              carried below, is a running maximum and does not fall with it.
            Group state (d and its accumulators, the schedule-free weight sum, the step count) is copied by position;
            a group new to the model (added layers) takes the main group's step size and starts its own average."""
            for gi, g in enumerate(new.param_groups):
                if gi < len(old.param_groups):
                    for k, v in old.param_groups[gi].items():
                        if k != "params": g[k] = v
                else:
                    for k in D_KEYS:
                        if k in old.param_groups[0]: g[k] = old.param_groups[0][k]
                    g["train_mode"] = old.param_groups[0].get("train_mode", False)
            src = {id(pn): po for po, pn in (pairs or [])}
            for g in new.param_groups:
                for pn in g["params"]:
                    if grown and pn in grown:
                        po, sh = grown[pn]; st = old.state.get(po)
                        if not st: continue
                        ns = {}
                        z = pn.detach().clone(); z[tuple(slice(0, k) for k in sh)] = st["z"]; ns["z"] = z
                        if "exp_avg_sq_row" in st:
                            dr, dc = st["exp_avg_sq_metadata"]
                            for key, dim_one in (("exp_avg_sq_row", dr), ("exp_avg_sq_col", dc)):
                                ov = st[key]; tgt = list(pn.shape); tgt[dim_one] = 1
                                nv = torch.full(tgt, float(ov.median()), dtype=ov.dtype, device=ov.device)
                                nv[tuple(slice(0, k) for k in ov.shape)] = ov; ns[key] = nv
                            ns["exp_avg_sq_metadata"] = (dr, dc)
                        elif "exp_avg_sq" in st:
                            ov = st["exp_avg_sq"]; nv = torch.full(pn.shape, float(ov.median()), dtype=ov.dtype, device=ov.device)
                            nv[tuple(slice(0, k) for k in sh)] = ov; ns["exp_avg_sq"] = nv
                        sl = new.get_sliced_tensor(pn)
                        ns["p0"] = sl.to(torch.bfloat16).clone() if pn.any() else torch.tensor(0.0, dtype=torch.bfloat16, device=pn.device)
                        ns["s"] = torch.zeros_like(sl, dtype=torch.bfloat16)
                        new.state[pn] = ns
                    else:
                        po = src.get(id(pn), pn)
                        if po in old.state: new.state[pn] = old.state[po]
        if SF:
            # GOAL 8, SECOND DESIGN (2026-09-27). The hypergradient rate collapsed over a long run (3e-3 -> 1.5e-6 in
            # 120k steps, learning stopped): any rule that tunes the rate by its short-term effect lowers it, because a
            # lower rate always buys an immediate drop in noise. Schedule-free averaging gives that noise reduction
            # through the averaged weights instead, so the step size never has to fall; Prodigy estimates it from
            # the distance travelled, a lower bound that can only rise. Both at their published defaults.
            opt = make_opt(m)
            set_none = False                         # the captured micro-step writes into fixed grad buffers
        else:
            opt = Fabric(m.param_groups(lr=a.lr), lr=a.lr, fused=bool(a.fused), hyper=a.intrinsic_lr == 1, hyper_beta=a.hyper_beta, dog=a.intrinsic_lr == 2, cuda_graph=bool(a.graph and a.fused))
            set_none = not opt.cuda_graph
        base = [g["lr"] for g in opt.param_groups]
        rng = np.random.default_rng(a.seed)

        # THE MICRO-STEP AS ONE CUDA GRAPH. With the scan in Triton the step was launch-bound: 1.3 ms of
        # kernels inside a 23.9 ms micro-step at micro-batch 4, the rest python and launch gaps. The same
        # recipe as the optimizer and the decode step: a static input buffer, grads zeroed in place, the whole
        # forward + chunked loss + backward captured once and replayed per accumulation step. Measured at 78M
        # ctx 2048: delta 143 -> 264 ktok/s, plain 282 -> 295, attention 312 -> 325. torch.compile's own
        # "reduce-overhead" mode could not do this (it refuses grads that outlive the replay); manual capture
        # can. Copying the batch into the static buffer is the only per-step host work left.
        # A MIX OF LENGTHS, not a sequence of two. Measured 2026-09-21: after 305,000 steps at 2,048 and
        # 13,000 at 8,192 the model was the best ever measured at exactly 8,192 and the worst of three at
        # every other length, and its long-range recall had gone backwards. Training at two lengths teaches
        # two lengths. Each length gets its own captured graph, and the batch scales inversely so a micro-step
        # always carries the same number of tokens whichever length it draws.
        mix = [int(v) for v in a.ctx_mix.split(",") if v] or [a.ctx]
        tok_per_micro = a.batch * max(mix)
        shapes = [(t, max(1, tok_per_micro // t)) for t in mix]
        # A MICRO-BATCH GREATER THAN ONE IS NOT SAFE ON THE STATE-READING MIXER UNDER torch.compile.
        # Measured 2026-09-22: rspin at ctx 8,192 with batch 4 dies with an illegal memory access inside a
        # compiled kernel; batch 2 is fine, the all-carrier mixer at batch 4 is fine, and --compile 0 at batch
        # 4 is fine. The trunk hands the carrier's per-chunk states to the attention layers through a module
        # attribute, and inductor is free to reuse that buffer. Rather than give up compile, which is worth
        # roughly twice the throughput, every length runs at batch 1 and the lengths are drawn in proportion
        # to 1/length so each still contributes the same tokens over a run. Tokens per micro-step then vary,
        # which the accumulation absorbs.
        mix_p = None
        # With --fused-scan the state read is off and the per-chunk states are never materialised, so the buffer
        # the fault reused does not exist; the all-carrier mixer, which has no state read, never hit it.
        if len(mix) > 1 and any(B_ > 1 for _, B_ in shapes) and not a.fused_scan and getattr(m.mix[3] if len(m.mix) > 3 else m.mix[0], "read_state", 0):
            shapes = [(t, 1) for t in mix]
            w_ = np.array([1.0 / t for t in mix]); mix_p = w_ / w_.sum()
            r.log(f"state-reading mixer: every length at batch 1, drawn with p = "
                  f"{{{', '.join(f'{t}: {q:.3f}' for t, q in zip(mix, mix_p))}}}")

        def capture(T=None, B=None):
            if not (a.graph_step and a.compile): return None
            T = T or a.ctx; B = B or a.batch
            xs = torch.empty((B, T + 1), dtype=torch.long, device="cuda")
            xs.copy_(batch(bins, ks, ps, np.random.default_rng(11), B, T, extra, a.data2_frac))
            for p_ in m.parameters():
                if p_.grad is None: p_.grad = torch.zeros_like(p_)
            out = {}
            def micro():
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    h = trunk(xs[:, :T])
                l_ = m.loss(h, xs[:, 1:T + 1], chunks=max(1, a.chunks), ul_alpha=a.ul, ul_window=a.ul_window)
                (l_ / a.accum).backward()
                out["loss"] = l_.detach()
            s_ = torch.cuda.Stream(); s_.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s_):
                for _ in range(3): micro()                       # compile and warm on a side stream
            torch.cuda.current_stream().wait_stream(s_)
            for p_ in m.parameters(): p_.grad.zero_()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                micro()
            for p_ in m.parameters(): p_.grad.zero_()
            return g, xs, out["loss"]
        tok_total = a.steps * a.batch * a.ctx * a.accum
        r.log(f"{P/1e6:.1f}M params, carrier {100*car/P:.1f}%, "
              f"{tok_total/1e9:.2f}B tokens = {tok_total/P:.1f} tok/param")

        # RESUME. The last run diverged at step 14,400 and the trainer then wrote NaN weights over a healthy
        # checkpoint, so the only survivor was a copy taken by accident for a probe.
        start = 0; grew_ctx = None
        if a.resume and os.path.exists(a.resume):
            sd = torch.load(a.resume, map_location="cuda")
            if sd["cfg"].get("ctx", a.ctx) != a.ctx:
                # A LONG-CONTEXT PHASE. The positional table is sized to the old window; tile the trained
                # pattern to fill the new one. That is exactly what inference already does when it processes a
                # long input as consecutive windows with the state carried, so training this way teaches the
                # model to use the state ACROSS a window boundary rather than only within one.
                old_ctx = sd["cfg"]["ctx"]
                reps = -(-a.ctx // old_ctx)
                w = sd["model"]["p.weight"]
                sd["model"]["p.weight"] = w.repeat(reps, 1)[: a.ctx]
                r.log(f"extended the positional table {old_ctx} -> {a.ctx} by tiling the trained pattern {reps}x")
                sd["cfg"]["ctx"] = a.ctx
                grew_ctx = (old_ctx, sd["cfg"]["d"])
            if sd["cfg"]["d"] != m.d or sd["cfg"].get("layers", m.layers) != m.layers:
                # a grown run: rebuild at the checkpoint's own width AND depth
                m = LM(vocab=a.vocab, d=sd["cfg"]["d"], layers=sd["cfg"].get("layers", a.layers), mlp_mult=a.mlp,
                       ctx=a.ctx, mixer=a.mixer, norm=a.norm).cuda()
                if a.fused_scan:
                    import src.assoc as _A
                    _A.FUSED_SCAN = True
                    for _mx in m.mix:
                        if hasattr(_mx, "need_sent"): _mx.need_sent = False
                fwd = torch.compile(m) if a.compile else m
                trunk = torch.compile(m.trunk) if a.compile else m.trunk
                opt = make_opt(m)
            m.load_state_dict(sd["model"]); start = sd["step"]
            if sd.get("new_layers"):
                # a checkpoint written by eval/deepen.py: its added layers train with absolute steps (see
                # LM.param_groups), and it carries no optimizer state, so this is a cold start by design
                m.new_layers = list(sd["new_layers"])
                opt = make_opt(m)
                base = [g["lr"] for g in opt.param_groups]
                r.log(f"deepened checkpoint: {len(m.new_layers)} new layers {m.new_layers} train at absolute lr {a.new_lr}")
            # RESTORE THE OPTIMIZER TOO. Loading only the weights leaves the optimizer cold, with its second
            # moment at zero, so its first steps after a resume are near-maximal on a model that was already
            # converged to that point. This project measured the same failure at growth events: a rebuilt
            # optimizer takes a 1.5x larger first step and costs about a nat of shock. A resume is the same
            # event. Recorded because the first version of this resume shipped without it.
            if "opt" in sd:
                try:
                    opt.load_state_dict(sd["opt"])
                    if SF: opt.train()   # checkpoints hold the averaged weights; back to training ones
                    if grew_ctx is not None:
                        # the positional table grew with the window, so its second moment is still sized to
                        # the old one. Fabric.grew extends it with the median of the existing rows; without
                        # this the fused update tried to add a 2048-vector to an 8192-vector and the compile
                        # failed rather than silently doing something wrong.
                        opt.grew(m.p.weight, grew_ctx)
                        r.log(f"extended the optimizer's statistics for the positional table to {a.ctx} rows")
                    r.log(f"resumed optimizer state ({len(sd['opt'].get('state', {}))} tensors)")
                except Exception as e:
                    r.log(f"optimizer state present but not loadable ({type(e).__name__}: {str(e)[:300]}); starting cold")
            else:
                r.log("WARNING: checkpoint carries no optimizer state; the first steps will be a shock")
            r.log(f"resumed from {a.resume} at step {start}, loss {sd['loss']:.4f}")
        # TOKENS ACTUALLY PROCESSED per optimizer step, averaged over the lengths the mix draws. Counting every step
        # at batch x ctx overstated a mixed-length run 2.5x (2026-09-22) and 4x once the window stopped shrinking
        # to the mix (2026-09-23): run47 printed 1,138 ktok/s while processing ~285.
        # a resumed run past a split point starts at the split micro-batch
        for st_, f_ in grow_a.items():
            if st_ < start: shapes = [(t, max(1, B_ // f_)) for t, B_ in shapes]; a.accum *= f_
        _pw = mix_p if mix_p is not None else [1.0 / len(shapes)] * len(shapes)
        tok_per_step = a.accum * sum(q * B_ * T_ for q, (T_, B_) in zip(_pw, shapes))
        graphs = {(B_, T_): capture(T_, B_) for T_, B_ in shapes}          # one captured graph per length
        gs = graphs[(shapes[0][1], shapes[0][0])]
        if gs is not None:
            set_none = False
            r.log(f"micro-step captured as a CUDA graph; {torch.cuda.memory_reserved()/2**30:.1f} GiB reserved after capture")

        t0 = time.time(); ema = None; t_prev = 0.0

        stage_t0, stage_next = [start], [4 * a.trial_steps]
        if a.resume and os.path.exists(a.resume):
            ck_ = torch.load(a.resume, map_location="cpu", weights_only=False)
            st_ = ck_.get("stage")
            if st_: stage_t0[0], stage_next[0] = st_          # a resumed run remembers how long its stage has run
            elif ck_["cfg"]["d"] == a.d and ck_["cfg"].get("layers", a.layers) == a.layers:
                # an older checkpoint without the clock, still at its starting shape: it has never grown, so its stage
                # began at step 0; the next reconsider is the next doubling of that stage
                stage_t0[0] = 0; nxt_ = 4 * a.trial_steps
                while nxt_ <= start: nxt_ *= 2
                stage_next[0] = nxt_
            del ck_
        mark, rate_hist = [None], []         # (step, held-out loss, params) where the current price was last measured
        stage_rate = [a.lr]                  # the main rate when the current stage began: saturation is judged
                                             # against it, so each stage anneals on from where the last one ended
        def restart_rates():
            # AFTER A SATURATION-TRIGGERED GROWTH THE RATES ARE NOT RESET. Resetting the trained layers from their
            # learned ~1e-3 back to 3e-3 knocked a deepened model from loss 3.30 to 5.74 (measured 2026-09-24).
            # They keep their rate; layers new to this stage (the absolute-step group) start small and the
            # hypergradient raises them as they prove useful, so even their ramp-in is the model's own.
            if not a.grow_auto or SF: return      # psf has no rate to restart or record
            # NO RATE IS RESET AT A WIDEN. Restarting it at the stage start (tried 2026-09-24) caused a transient that
            # made a 500-step growth trial read the grown branch as WORSE (3.9234 -> 3.9655 while unchanged went to
            # 3.8695), and a reset is a hand-made intervention besides. The grown model keeps its own rate; the
            # hypergradient raises it if the new width can use larger steps.
            if len(opt.param_groups) > 2 and grown_at == i:
                g_ = opt.param_groups[2]; g_["lr"] = a.new_lr
                if g_["lr_t"] is not None: g_["lr_t"].fill_(a.new_lr)
            stage_rate[0] = opt.rates()[0]
        # THE GUARD LIVES ON THE CARD. Reading the gradient norm and the loss every step were the only two
        # syncs left in the loop, and each one drained the queue and left the card idle through the host's
        # bookkeeping: 2 ms of a 128 ms update measured piece by piece. The norm history, its median, the
        # keep/skip decision and the loss average are device tensors now; the host reads them once per 200
        # steps. A skipped step zeroes its gradients through one foreach multiply instead of a branch, so the
        # optimizer sees a zero gradient (its moments decay once) rather than being skipped outright; on
        # steps that do not spike, which is all but a few per ten thousand, nothing changes.
        # the history starts EMPTY (NaN), not at zero: a zero slot drags the median down, the guard then rejects
        # more steps, each rejection leaves another zero slot, and the run locks itself out (2026-09-23)
        hist = torch.full((1000,), float("nan"), device="cuda"); g0 = start; grown_at = -10**9
        skips_t = torch.zeros((), device="cuda"); skips_t = torch.zeros((), device="cuda"); ema_t = None
        nonfin = torch.zeros((), device="cuda"); nf_prev = 0
        marks = []; bpbs = []; skipped = 0
        T_n, B_n = shapes[0]
        nxt = [batch(bins, ks, ps, rng, B_n, T_n, extra, a.data2_frac) for _ in range(a.accum)]

        # ---- GROWTH TRIALS (goals 7 + 8): the model decides to grow by trying it ----------------------------------
        # When its rate has annealed an e-fold (time to RECONSIDER, nothing more), the model snapshots itself, trains a
        # GROWN copy for --trial-steps, then goes back to the snapshot and trains UNCHANGED for the same steps on the
        # SAME batches, and keeps whichever branch bought more held-out loss per unit of compute (params x steps).
        # Replaces two signals that grew it long before it was saturated: the rate e-fold alone (5 and 2 tokens per
        # parameter in its first stages, 2026-09-24) and a one-step gradient comparison (break-even at 1/3 token per
        # parameter: a gradient says what new capacity COULD do now, not whether the old has run out).
        import copy as _copy
        xs_eval = batch(bins, ks, ps, np.random.default_rng(424242), 4, 2048)
        trial = None
        def heldout():
            if SF: opt.eval()
            m.eval()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                l_ = F.cross_entropy(m(xs_eval[:, :2048]).float().reshape(-1, a.vocab), xs_eval[:, 1:2049].reshape(-1)).item()
            m.train()
            if SF: opt.train()
            return l_
        def to_dev(o, dev):
            if torch.is_tensor(o): return o.to(dev)
            if isinstance(o, dict): return {k: to_dev(v, dev) for k, v in o.items()}
            if isinstance(o, (list, tuple)): return type(o)(to_dev(v, dev) for v in o)
            return o
        def snapshot():
            mc = _copy.deepcopy(m).to("cpu")
            for p_ in mc.parameters(): p_.grad = None
            return dict(m=mc, opt=to_dev(opt.state_dict(), "cpu"), rng=_copy.deepcopy(rng.bit_generator.state),
                        nxt=[x.clone() for x in nxt], shapes=list(shapes), accum=a.accum, batch=a.batch,
                        auto=[list(e) for e in auto_events])
        def install(S):
            """make snapshot S the live model: optimizer rebuilt from its state, graphs recaptured"""
            nonlocal m, opt, fwd, trunk, gs, g_, base, set_none, shapes, nxt
            graphs.clear(); gs = None; g_ = None
            fwd = trunk = None; torch._dynamo.reset(); gc.collect(); torch.cuda.empty_cache()
            shapes = S["shapes"]; a.accum = S["accum"]; a.batch = S["batch"]
            rng.bit_generator.state = S["rng"]; nxt = S["nxt"]
            m = S["m"].to("cuda")
            if a.fused_scan:
                for mx_ in m.mix:
                    if hasattr(mx_, "need_sent"): mx_.need_sent = False
            opt = make_opt(m)
            opt.load_state_dict(to_dev(S["opt"], "cuda"))
            if SF: opt.train()
            base = [g["lr"] for g in opt.param_groups]; set_none = False if SF else not opt.cuda_graph
            fwd = torch.compile(m) if a.compile else m
            trunk = torch.compile(m.trunk) if a.compile else m.trunk
            if len(shapes) > 1: graphs.update({(B_, T_): capture(T_, B_) for T_, B_ in shapes})
            else: gs = capture()
        def nparams(): return sum(p_.numel() for p_ in m.parameters())

        for i in range(start, a.steps):
            if trial is not None and i == trial["end"] - trial["K"] // 2:
                trial["Lg_mid" if trial["phase"] == "grow" else "Ls_mid"] = heldout()
            if trial is not None and i == trial["end"]:
                if trial["phase"] == "grow":
                    trial["Lg"], trial["Pg"] = heldout(), nparams()
                    trial["G"] = snapshot()
                    install(trial.pop("S0"))
                    if False:
                        # (SUPERSEDED by psf_carry: neither branch restarts its average now.) THE SAME START FOR BOTH BRANCHES. The grown branch starts from the averaged weights with a
                        # fresh schedule-free average; left alone, the unchanged branch would keep its long average,
                        # which lowers held-out loss by itself, and growth could never win (measured 2026-09-27:
                        # unchanged 5.1266 -> 5.0147 vs grown -> 5.1207 at the second trial). So it too restarts
                        # from its averaged weights, same carried step size: the only difference left is the size.
                        opt.eval(); fresh = make_opt(m); carry_d(opt, fresh); opt = fresh
                    hist.fill_(float("nan")); g0 = i
                    trial["phase"], trial["end"] = "stay", i + trial["K"]
                    r.log(f"    TRIAL {'+'.join(trial['ev'])}: grown branch {trial['L0']:.4f} -> {trial['Lg']:.4f} over "
                          f"{trial['K']} steps at {trial['Pg']/1e6:.1f}M; now the same steps unchanged")
                else:
                    Ls, Ps = heldout(), nparams()
                    gg = (trial["L0"] - trial["Lg"]) / trial["Pg"]; gs_ = (trial["L0"] - Ls) / Ps
                    # THE DECISION, PENALTY-FREE (2026-09-27). Restarting the schedule-free average costs both branches
                    # the same, and at small scale it outweighed 600 steps of learning ("both got worse"), so any rule
                    # on each branch's own gain would call almost every trial inconclusive and the model would never
                    # grow. Growth's ADVANTAGE (Ls - Lg: same start, same batches, same restart, so the penalty cancels)
                    # is weighed against what the EXTRA compute of the bigger model would have bought if spent on
                    # more steps at the current size, priced at the current size's own recent rate of held-out
                    # improvement per unit compute (measured between trials at the averaged weights, no restart).
                    if SF and "Lg_mid" in trial and "Ls_mid" in trial:
                        # MARGINAL RETURNS (2026-09-29). Grow when the grown model turns compute into held-out loss
                        # FASTER at the margin -- over the second half of its trial, per unit of compute -- than the
                        # unchanged one over the same steps, and is not behind: the compute-optimal condition (equal
                        # marginal returns), measured on the model itself. Replaces advantage-vs-price below, whose
                        # price came from the steeper window BEFORE the trial and overstated what staying buys. The
                        # long test (eval/growth_long.txt) showed growth's lead growing steadily (0.0006 at +4k steps,
                        # 0.0098 at +20k), so the question is when its marginal rate overtakes, not whether it leads early.
                        half = trial["K"] // 2
                        slope_g = (trial["Lg_mid"] - trial["Lg"]) / (trial["Pg"] * half)
                        slope_s = (trial["Ls_mid"] - Ls) / (Ps * half)
                        grow_now = slope_g > slope_s and trial["Lg"] <= Ls
                        r.log(f"    TRIAL {'+'.join(trial['ev'])}: marginal held-out gain per 1e12 param-steps over the last "
                              f"{half} steps: grown {slope_g*1e12:.4f} vs unchanged {slope_s*1e12:.4f}; grown "
                              f"{'ahead' if trial['Lg'] <= Ls else 'behind'} ({trial['Lg']:.4f} vs {Ls:.4f})")
                        gg, gs_ = slope_g, slope_s
                    elif SF:
                        # (named extra_c: "extra" is the second corpus, and reusing the name crashed run60 at its
                        # first decision, 2026-09-27)
                        adv = Ls - trial["Lg"]
                        extra_c = (trial["Pg"] - Ps) * trial["K"]
                        price = max(0.0, rate_hist[-1]) if rate_hist else 0.0
                        grow_now = adv > price * extra_c
                        r.log(f"    TRIAL {'+'.join(trial['ev'])}: growth's advantage {adv:+.4f} vs what its extra compute buys "
                              f"at the current size {price * extra_c:.4f} (recent held-out gain {price*1e12:.3f} per 1e12 param-steps)")
                        gg = adv; gs_ = price * extra_c
                    # GROW ONLY ON A REAL GAIN. Dividing a loss INCREASE by compute makes the larger model look
                    # better (a bigger denominator shrinks a negative number): a trial where both branches got worse
                    # and the unchanged one got less worse (4.1020 vs 4.1206) was decided GROW (2026-09-27). Now the
                    # grown branch must improve held-out loss, by more per unit compute than staying did (a stay
                    # that did not improve counts as zero); otherwise the trial is inconclusive and the model stays.
                    if (grow_now if SF else (trial["L0"] - trial["Lg"] > 0 and gg > max(gs_, 0.0))):
                        G = trial["G"]; install(G); auto_events[:] = G["auto"][1:]
                        stage_t0[0], stage_next[0] = i, 4 * a.trial_steps
                        hist.fill_(float("nan")); g0 = i
                        if not SF: stage_rate[0] = opt.rates()[0]
                        r.log(f"    DECISION at step {i}: GROW ({'+'.join(trial['ev'])}). Held-out gain per unit compute "
                              f"{gg*1e9:.3f} grown vs {gs_*1e9:.3f} unchanged ({trial['L0']:.4f} -> {trial['Lg']:.4f} vs "
                              f"-> {Ls:.4f}); continuing from the grown branch at {nparams()/1e6:.1f}M")
                    else:
                        if not SF: stage_rate[0] = opt.rates()[0]
                        r.log(f"    DECISION at step {i}: STAY. Held-out gain per unit compute {gs_*1e9:.3f} unchanged vs "
                              f"{gg*1e9:.3f} grown ({trial['L0']:.4f} -> {Ls:.4f} vs -> {trial['Lg']:.4f}); "
                              + ("will reconsider when this stage's length doubles again" if SF else "will reconsider after the rate anneals another e-fold"))
                    trial = None
                    mark[0] = (i, heldout(), nparams())
            w = min(1.0, (i + 1) / a.warm)
            # cosine over the whole budget: a flat rate lets a model equilibrate at that rate's noise floor,
            # which is what drove the previous project's BPB upward for hundreds of thousands of steps.
            f = min(1.0, max(0.0, (i - a.warm) / max(1, a.steps - a.warm)))
            sched = 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * f))
            if i in grow_a:
                # SPLIT THE MICRO-STEP before a stage whose activations would not fit under the cap: half the rows
                # per micro-step, twice the micro-steps, so the update sees the same tokens. The graphs are freed
                # first and recaptured at the new shape (by the growth event below if one shares this step).
                graphs.clear(); gs = None; g_ = None; gc.collect(); torch.cuda.empty_cache()
                shapes = [(t, max(1, B_ // grow_a[i])) for t, B_ in shapes]; a.accum *= grow_a[i]
                a.batch = shapes[-1][1] if len(shapes) > 1 else max(1, a.batch // grow_a[i])
                if i not in grow_w and i not in grow_d:
                    if len(shapes) > 1: graphs.update({(B_, T_): capture(T_, B_) for T_, B_ in shapes})
                    else: gs = capture()
                T_n, B_n = shapes[0]
                nxt = [batch(bins, ks, ps, rng, B_n, T_n, extra, a.data2_frac) for _ in range(a.accum)]
                r.log(f"    SPLIT micro-steps {grow_a[i]} ways at step {i}: shapes {shapes}, accumulation {a.accum}")
            if i in grow_w and m.d < grow_w[i]:
                # GROWTH (vision goal 7): widen in place, function preserved, optimizer state carried. The
                # loss right before and right after is logged so the cost of the event is a number.
                if SF: opt.eval()          # measured and grown at the AVERAGED weights, the model
                xg = batch(bins, ks, ps, np.random.default_rng(7), a.batch, a.ctx)
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    lb = F.cross_entropy(m(xg[:, :a.ctx]).reshape(-1, a.vocab), xg[:, 1:a.ctx + 1].reshape(-1)).item()
                # free the old CUDA graphs BEFORE growing: old and new together exceed the cap at the late stages
                graphs.clear(); gs = None; g_ = None; gc.collect()
                torch._dynamo.reset(); torch.cuda.empty_cache()
                grown_ = m.grow(grow_w[i])
                if a.optim == "psf":
                    # rebuilt for the new shapes WITH its memory carried (psf_carry); returned to training weights
                    # after the exactness check below
                    o_old = opt; opt = make_opt(m); psf_carry(o_old, opt, grown=grown_); del o_old; set_none = False
                else:
                    for p_, (old_p, sh) in grown_.items():
                        opt.grew(p_, sh, replaces=old_p)
                opt.zero_grad(set_to_none=True)             # grad buffers are the old shape
                fwd = torch.compile(m) if a.compile else m
                trunk = torch.compile(m.trunk) if a.compile else m.trunk
                # every window's graph, not only the first: a mixed-length run held stale graphs after widening
                if len(shapes) > 1:
                    graphs.update({(B_, T_): capture(T_, B_) for T_, B_ in shapes})
                else:
                    gs = capture()
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    la = F.cross_entropy(m(xg[:, :a.ctx]).reshape(-1, a.vocab), xg[:, 1:a.ctx + 1].reshape(-1)).item()
                P = sum(p.numel() for p in m.parameters())
                r.log(f"    GREW to d={m.d} at step {i}: {P/1e6:.1f}M params, same batch loss {lb:.4f} -> {la:.4f}, "
                      f"optimizer state {opt.state_bytes()/2**20:.0f} MiB carried")
                if SF: opt.train()
                hist.fill_(float("nan")); g0 = i      # the grown model's gradient norm is its own; re-learn it
                restart_rates()
            if i in grow_d:
                # DEPTH GROWTH MID-RUN (vision goal 7): the model doubles its layers in place and computes exactly
                # what it did before (new layers start at zero output; measured 1.949016 -> 1.949016 on run47).
                # Every existing layer keeps its optimizer state; the new ones get their own absolute-step group.
                # Growing a shallow model late in the run costs half the compute per token until then.
                if SF: opt.eval()          # measured and deepened at the AVERAGED weights, the model
                xg = batch(bins, ks, ps, np.random.default_rng(7), a.batch, a.ctx)
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    lb = F.cross_entropy(m(xg[:, :a.ctx]).reshape(-1, a.vocab), xg[:, 1:a.ctx + 1].reshape(-1)).item()
                old_m = m
                big = old_m.deepen(init=a.deepen_init)
                if a.fused_scan:
                    for _mx in big.mix:
                        if hasattr(_mx, "need_sent"): _mx.need_sent = False
                keep = [q for q in range(big.layers) if q not in set(big.new_layers)]
                pairs = [(po, pn) for mod in ("e", "p", "n")
                         for po, pn in zip(getattr(old_m, mod).parameters(), getattr(big, mod).parameters())]
                for j, q in enumerate(keep):
                    pairs += list(zip(old_m.mix[j].parameters(), big.mix[q].parameters()))
                    pairs += list(zip(old_m.mlp[j].parameters(), big.mlp[q].parameters()))
                opt2 = make_opt(big)
                if a.optim == "psf":
                    psf_carry(opt, opt2, pairs=pairs)
                    if a.deepen_init in ("copy", "gate"):
                        # a stacked layer takes a COPY of its source layer's optimizer memory as well as its weights
                        import copy as _cp
                        for i_new, q_src in big.copied_from.items():
                            # paired BY NAME: a gated copy has "o.lin.weight" where its source has "o.weight", plus a
                            # gate of its own that starts with fresh memory
                            for sub in ("mix", "mlp"):
                                src_ = dict(getattr(big, sub)[q_src].named_parameters())
                                for nm, pn in getattr(big, sub)[i_new].named_parameters():
                                    ps_ = src_.get(nm.replace(".lin.", "."))
                                    if ps_ is None or ps_.shape != pn.shape or ps_ not in opt2.state: continue
                                    opt2.state[pn] = {k: (v.clone() if torch.is_tensor(v) else _cp.copy(v))
                                                      for k, v in opt2.state[ps_].items()}
                if a.intrinsic_lr == 1:            # the rates the optimizer found are kept; new layers start at --new-lr
                    for g_old, g_new in zip(opt.rates()[:2], opt2.param_groups[:2]):
                        g_new["lr"] = g_old
                carried = 0
                for po, pn in (pairs if a.optim != "psf" else ()):
                    if po in opt.state: opt2.state[pn] = opt.state[po]; carried += 1
                if a.optim == "sfp":
                    # the fused optimizer keeps its learned step size and averaging weight across the deepen; a group
                    # new to the model (added layers) takes the main group's step size and starts its own average
                    for gi_, g_ in enumerate(opt2.param_groups):
                        src_ = opt.param_groups[gi_] if gi_ < len(opt.param_groups) else opt.param_groups[0]
                        if src_.get("sf") is not None:
                            f_ = src_["sf"]
                            g_["sf_saved"] = {"d": float(f_["d"]), "dnum": float(f_["dnum"]) if gi_ < len(opt.param_groups) else 0.0,
                                              "ws": float(f_["ws"]) if gi_ < len(opt.param_groups) else 0.0}
                        g_["train_mode"] = src_.get("train_mode", True)
                    if a.deepen_init in ("copy", "gate"):
                        import copy as _cp
                        for i_new, q_src in big.copied_from.items():
                            for sub in ("mix", "mlp"):
                                src_p = dict(getattr(big, sub)[q_src].named_parameters())
                                for nm, pn in getattr(big, sub)[i_new].named_parameters():
                                    ps_ = src_p.get(nm.replace(".lin.", "."))
                                    if ps_ is None or ps_.shape != pn.shape or ps_ not in opt2.state: continue
                                    opt2.state[pn] = {k: (v.clone() if torch.is_tensor(v) else _cp.copy(v))
                                                      for k, v in opt2.state[ps_].items()}
                # free the shallow model's CUDA graphs BEFORE capturing the deep one's: both at once would be
                # ~15 + ~28 GiB, over the cap
                graphs.clear(); gs = None; g_ = None; gc.collect()
                m, opt = big, opt2
                del old_m, opt2, big, pairs      # pairs held every shallow parameter (and its grad) alive
                base = [g["lr"] for g in opt.param_groups]
                set_none = False if SF else not opt.cuda_graph   # psf: graphs write into fixed grad buffers
                fwd = trunk = None; torch._dynamo.reset(); gc.collect(); torch.cuda.empty_cache()
                fwd = torch.compile(m) if a.compile else m
                trunk = torch.compile(m.trunk) if a.compile else m.trunk
                if len(shapes) > 1:
                    graphs.update({(B_, T_): capture(T_, B_) for T_, B_ in shapes})
                else:
                    gs = capture()
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    la = F.cross_entropy(m(xg[:, :a.ctx]).reshape(-1, a.vocab), xg[:, 1:a.ctx + 1].reshape(-1)).item()
                P = sum(p.numel() for p in m.parameters())
                r.log(f"    DEEPENED to {m.layers} layers at step {i}: {P/1e6:.1f}M params, same batch loss "
                      f"{lb:.4f} -> {la:.4f}, optimizer state carried for {carried} tensors, "
                      f"{len(m.new_layers)} new layers at absolute lr {a.new_lr}, ramped over {a.new_warm} steps")
                if SF: opt.train()
                grown_at = i; hist.fill_(float("nan")); g0 = i
                restart_rates()
            # rates are set AFTER any growth event: a deepen rebuilds the optimizer at its base rates
            for gi, g in (() if (a.intrinsic_lr or SF) else enumerate(opt.param_groups)):
                g["lr"] = base[gi] * sched * w
            # (never under psf: its rates are not reset each step, so this multiply COMPOUNDED and drove the added
            # layers' rate to ~0 within a few steps -- every psf deepen until 2026-09-29 trained no new layer at all)
            if len(opt.param_groups) > 2 and not a.intrinsic_lr and not SF:
                # layers added mid-run start from empty optimizer state; their first absolute steps are sign-like
                # (+-lr on every element, zero-initialised output projections included), so they ramp in
                opt.param_groups[2]["lr"] *= min(1.0, (i - grown_at + 1) / max(1, a.new_warm))
            opt.zero_grad(set_to_none=set_none)
            loss = None
            for _acc in range(a.accum):
                x = nxt[_acc]
                g_ = graphs.get((x.shape[0], x.shape[1] - 1)) if len(shapes) > 1 else gs
                if g_ is not None:
                    g_[1].copy_(x); g_[0].replay(); l_ = g_[2].clone()
                elif a.chunks > 1:
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        h = trunk(x[:, :a.ctx])
                    l_ = m.loss(h, x[:, 1:a.ctx + 1], chunks=a.chunks, ul_alpha=a.ul, ul_window=a.ul_window)
                else:
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        l_ = F.cross_entropy(fwd(x[:, :a.ctx]).reshape(-1, a.vocab), x[:, 1:a.ctx + 1].reshape(-1))
                if (g_ if len(shapes) > 1 else gs) is None: (l_ / a.accum).backward()   # else the replay did
                loss = l_.detach() if loss is None else loss + l_.detach()
            loss = loss / a.accum
            # the next step's batches are sampled NOW, while the card is still busy with this one; the
            # gradient-norm read below is a sync and everything after it is a gap the card sits through
            T_n, B_n = shapes[rng.choice(len(shapes), p=mix_p)] if len(shapes) > 1 else shapes[0]
            nxt = [batch(bins, ks, ps, rng, B_n, T_n, extra, a.data2_frac) for _ in range(a.accum)]
            # (schedule-free Prodigy must not be clipped: its step-size estimate reads the raw gradient)
            gn = torch.nn.utils.clip_grad_norm_(m.parameters(), float("inf") if SF else 1.0)
            # SPIKE GUARD. The previous run went from loss 4.0998 to NaN between step 14,200 and 14,400, a
            # sudden spike rather than a drift. Clipping bounds the step but still applies a direction taken
            # from a pathological batch, and the optimizer then carries that direction in its state for
            # thousands of steps. Compared against a running median so it needs no retuning as loss scale
            # changes. This guard existed in the previous project's trainer and I failed to carry it here.
            j = i - g0; fin = torch.isfinite(gn)
            med = hist.nanmedian() if j else gn
            keep = fin & ((gn <= a.spike * med) if j >= 200 else torch.ones((), dtype=torch.bool, device="cuda"))
            hist[j % 1000] = torch.where(keep, gn, hist[j % 1000])            # only kept steps enter the history
            skips_t.add_(~keep)
            # zero the gradients of a skipped step; a non-finite one becomes NaN * 0 here and is zeroed for
            # good inside the fused update (nan_to_num), so the optimizer state never sees it
            torch._foreach_mul_([p_.grad for p_ in m.parameters() if p_.grad is not None], keep.float())
            opt.step()
            # A NON-FINITE LOSS MUST NOT POISON THE AVERAGE. The first version of this device-side guard folded
            # every loss into the EMA, so one NaN batch made the average NaN for ever and the run stopped at the
            # next log line even though the guard had already zeroed that step's gradients and the weights were
            # still finite: run20 died that way at step 32,600 on 2026-09-17 with a healthy checkpoint on disk.
            # The average now skips non-finite losses and they are counted instead; what decides a real
            # divergence is the weights, checked below.
            fin_l = torch.isfinite(loss)
            nonfin.add_(~fin_l)
            ema_t = loss.clone() if ema_t is None else torch.where(fin_l, ema_t * 0.99 + loss * 0.01, ema_t)
            if (i + 1) % 200 == 0:
                ema, nsk, medv = ema_t.item(), int(skips_t.item()), float(med.item())   # the one sync
                nnf = int(nonfin.item())
                el = time.time() - t0
                # Rate and ETA must count from the RESUME point, not from step zero, or a resumed run
                # reports a throughput it never achieved and an ETA that is wildly optimistic.
                done = i + 1 - start
                r.log(f"{i+1}/{a.steps} loss {ema:.4f} ppl {math.exp(min(20,ema)):.1f} "
                      f"lr {(opt.rates()[0] if a.intrinsic_lr else opt.param_groups[0]['lr']):.2e} {done*tok_per_step/el/1e3:.0f} ktok/s "
                      f"(last 200: {200*tok_per_step/max(1e-9, el-t_prev)/1e3:.0f}) "
                      f"eta {(a.steps-i-1)*el/max(1,done)/60:.0f}m"
                      f" {torch.cuda.memory_reserved()/2**30:.1f} GiB"
                      + (" rates " + " ".join(f"{q:.2e}" for q in opt.rates()) if a.intrinsic_lr else "")
                      + (" d " + " ".join(f"{float(g['sf']['d'] if g.get('sf') else g.get('d', 0)):.2e}" for g in opt.param_groups) if SF else ""))
                t_prev = el
                if opt.recaptures:
                    r.log(f"    optimizer step recaptured {opt.recaptures} times")
                if a.growth_probe and (i + 1) % 1000 == 0:
                    xp = batch(bins, ks, ps, np.random.default_rng(99), 2, 2048)
                    sig = {op_: growth_signal(m, op_, xp, bool(a.fused_scan), a.chunks) for op_ in a.growth_probe.split(",")}
                    r.log(f"    PROBE step {i+1}: new/existing gradient per parameter " + "  ".join(f"{k} {v:.3f}" for k, v in sig.items()))
                if auto_events and mark[0] is None and (i + 1 - start) >= a.trial_steps:
                    mark[0] = (i + 1, heldout(), nparams())
                if auto_events and trial is None and i + 1 + 2 * a.trial_steps < a.steps:
                    # WHEN TO RECONSIDER: under the hypergradient, at each e-fold of its rate; under schedule-free
                    # Prodigy, which never anneals, each time the current stage's length has doubled (first at 4
                    # trial lengths), so the checks space out geometrically with the stage. The trial decides.
                    if SF:
                        due = (i + 1 - stage_t0[0]) >= stage_next[0]
                        rt = stage_rate[0] = 1.0
                    else:
                        rt = opt.rates()[0]; due = rt < stage_rate[0] / math.e
                    if due:
                        stage_next[0] = 2 * (i + 1 - stage_t0[0])
                        ev = auto_events[0]
                        # THE TRIAL GROWS WITH THE STAGE: a quarter of its length so far, at least --trial-steps. At a
                        # fixed 1,000 steps the widened copy learned exactly as much as the unchanged one (advantage
                        # -0.0002 at step 26,000): new capacity starts as a zero-output no-op and pays off over longer
                        # than that, so a short trial under-credits growth for good. Scale-free: the cost of a trial is
                        # a fixed fraction of the stage it decides about.
                        K_ = max(a.trial_steps, (i + 1 - stage_t0[0]) // 4)
                        if i + 1 + 2 * K_ >= a.steps: K_ = 0
                        L0_ = heldout()
                        if mark[0] is not None:
                            s0_, l0_, p0_ = mark[0]
                            rate_hist.append((l0_ - L0_) / max(1.0, p0_ * (i + 1 - s0_)))
                        trial = dict(phase="grow", ev=ev, L0=L0_, S0=snapshot(), K=K_, end=i + 1 + K_) if K_ else None
                        for op_ in ev:
                            if op_ == "d": grow_d.add(i + 1)
                            elif op_.startswith("w"): grow_w[i + 1] = int(op_[1:])
                            elif op_.startswith("a"): grow_a[i + 1] = int(op_[1:])
                        why = (f"this stage's length has doubled to {i+1-stage_t0[0]} steps" if SF else
                               f"rate {rt:.2e} is one e-fold below the stage start {stage_rate[0]:.2e}")
                        r.log(f"    RECONSIDER at step {i+1}: {why}; TRIAL of {'+'.join(ev)} vs unchanged, {max(a.trial_steps, (i + 1 - stage_t0[0]) // 4)} steps each")
                if nnf > nf_prev:
                    # the guard zeroed those steps; the run continues unless the weights themselves went bad
                    alive = bool(torch.stack([p_.isfinite().all() for p_ in m.parameters()]).all().item())
                    recent_nf = nnf - nf_prev; nf_prev = nnf
                    r.log(f"    {recent_nf} non-finite loss(es) in the last 200 steps (total {nnf}); "
                          f"weights {'still finite' if alive else 'NON-FINITE'}")
                    if not alive or recent_nf > 100:
                        r.log(f"    DIVERGED: the weights are not finite; stopping rather than burning the card")
                        break
                if not math.isfinite(ema):
                    r.log(f"    NON-FINITE RUNNING LOSS with finite weights; stopping, this should not happen")
                    break
                if nsk > skipped:
                    r.log(f"    {nsk - skipped} spike(s) skipped in the last 200 steps (median |g| {medv:.2f}, total {nsk})")
                marks.append((i + 1, nsk)); skipped = nsk
                # DIVERGED, not spiking: the attention control blew up at step ~11,000 and the guard then
                # skipped every one of the next 27,000 steps while the log showed a healthy-looking rate.
                # A guard that skips everything is a frozen model burning the card; stop and say so.
                if len(marks) >= 4 and nsk - marks[-4][1] >= 250:
                    r.log(f"    DIVERGED: {nsk} skips, {nsk - marks[-4][1]} of them within the last 600 steps; stopping")
                    break
            if (i + 1) % a.eval_every == 0:
                # BITS PER BYTE, not loss per token. Per-token loss depends on the vocabulary and is
                # comparable to nothing; bits per byte removes the tokenizer and is the only number that can
                # be set against another model.
                # BOUNDED, and the same length whatever the run trains at. Evaluating at a.ctx put a
                # 32,768-token forward beside a 32.6 GiB graph pool and the run died at its first eval with
                # the batch already halved to one (2026-09-22). A fixed 8,192 also makes the number
                # comparable across runs, which evaluating at each run's own window never was.
                torch.cuda.empty_cache()
                if SF: opt.eval()          # measured at the averaged weights, which are the model
                b, nb = eval_bpb(m, tok, bins, ks, ps, np.random.default_rng(1234),
                                 n_batches=8, B=4, T=min(a.ctx, 8192), vocab=a.vocab)
                if SF: opt.train()
                torch.cuda.empty_cache()
                r.log(f"  eval step {i+1}: BPB {b:.4f}  ({nb/1e3:.0f}k bytes)")
                # DEGRADING, not frozen: fspin at 6e-3 went 1.7075 -> 1.7534 -> 2.1130 over three evals while
                # the skip guard saw only 200 skips in 15k steps. Three consecutive rises, each above the
                # eval's own noise (measured spread ~0.005 on this fixed sample), is a run going backwards.
                bpbs.append(b)
                if len(bpbs) >= 4 and all(bpbs[-k] > bpbs[-k-1] + 0.01 for k in (1, 2, 3)):
                    r.log(f"    DEGRADING: BPB rose over the last three evals {bpbs[-4:]}; stopping")
                    break
            if (i + 1) % a.save_every == 0 or i + 1 == a.steps:
                # NEVER OVERWRITE A GOOD CHECKPOINT WITH A BROKEN ONE. The previous run saved NaN weights over
                # a healthy checkpoint and only a stray copy survived. Verify before writing, and keep the
                # previous file as last-known-good.
                if SF: opt.eval()          # the checkpoint holds the averaged weights
                bad = sum(1 for v in m.state_dict().values() if not torch.isfinite(v).all())
                if bad:
                    r.log(f"    refusing to checkpoint: {bad} non-finite tensors")
                else:
                    if os.path.exists(f"{a.out}/ckpt.pt"):
                        os.replace(f"{a.out}/ckpt.pt", f"{a.out}/ckpt_prev.pt")
                    torch.save({"model": m.state_dict(), "opt": opt.state_dict(),
                                "step": i + 1, "loss": ema_t.item(),   # the current average, not the last log's
                                "cfg": dict(vocab=a.vocab, d=m.d, layers=m.layers,
                                            mlp_mult=a.mlp, ctx=a.ctx, mixer=a.mixer, norm=a.norm),
                                "new_layers": list(getattr(m, "new_layers", [])),
                                "stage": (stage_t0[0], stage_next[0]) if a.grow_auto else None},
                               f"{a.out}/ckpt.pt")
                if SF: opt.train()
        if ema_t is None:
            r.log(f"NOTHING TO DO: resumed at step {start} with --steps {a.steps}. A resumed run counts from the "
                  f"checkpoint's step, so --steps must exceed it.")
        else:
            ema = ema_t.item(); skipped = int(skips_t.item())
            r.log(f"final loss {ema:.4f} ppl {math.exp(min(20,ema)):.1f}, {skipped} spike-skips")


if __name__ == "__main__":
    main()
