"""Train the LM on the tokenized slice. One file, one loop, checkpoints so intermediate states can be probed.

Run:  .venv/bin/python -m src.train --steps 120000 --out /mnt/kv_cache/run1
"""
from __future__ import annotations
import argparse, gc, json, math, os, time, faulthandler
faulthandler.enable()          # a segfault prints the Python stack into the log (two silent SIGSEGV exits, 2026-10-05)
import numpy as np
import torch
import torch.nn.functional as F

from .common import Run, cap_gpu, seed_all, require_prod, eval_bpb
from .model import LM, Attn
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
    ps = np.array([WEIGHTS.get(k, MIX.get(k, 1.0)) for k in ks], dtype=np.float64); ps /= ps.sum()
    return bins, ks, ps


# --weights (2026-10-08, the knowledge + arithmetic phase the user approved): per-source sampling weights by name, for the main
# corpora and the second corpus alike (each normalised within its own corpus). Empty: the designed MIX, extras at 1.
WEIGHTS = {}


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


class Streams:
    """CARRIED-STATE TRAINING DATA (--carry N). `groups` groups of B rows; each row reads one corpus onward from a random
    start, one window (T inputs + 1 target) per advance, and a group restarts its rows every N advances (groups staggered),
    so a model trained on it sees what inference sees: window after window of one stream with its state carried."""
    def __init__(self, bins, ks, ps, extra, extra_p, B, T, groups, N, rng):
        self.src, self.extra, self.extra_p = (bins, ks, ps), extra, extra_p
        self.B, self.T, self.N, self.rng = B, T, N, rng
        self.k = [(g * N) // groups for g in range(groups)]; self.rows = [None] * groups
        self.reg, self.reg_id = [], {}             # rows hold corpus INDICES, so a snapshot (deep-copied) never copies a corpus
    def _start(self, g):
        rows = []
        for _ in range(self.B):
            bn, kn, pn = self.extra if (self.extra is not None and self.rng.random() < self.extra_p) else self.src
            arr = bn[kn[self.rng.choice(len(kn), p=pn)]]
            if id(arr) not in self.reg_id: self.reg_id[id(arr)] = len(self.reg); self.reg.append(arr)
            rows.append([self.reg_id[id(arr)], int(self.rng.integers(0, len(arr) - self.N * self.T - 2))])
        self.rows[g] = rows
    def next(self, g):
        fresh = self.rows[g] is None or self.k[g] % self.N == 0
        if fresh: self._start(g)
        o = np.empty((self.B, self.T + 1), dtype=np.int64)
        for b, row in enumerate(self.rows[g]):
            o[b] = self.reg[row[0]][row[1]:row[1] + self.T + 1]; row[1] += self.T
        self.k[g] += 1
        return torch.from_numpy(o).pin_memory().cuda(non_blocking=True), fresh


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
    ap.add_argument("--remat", type=int, default=0,
                    help="recompute each layer in backward instead of storing its activations (goal 6: training on a 4-8 GB card)")
    ap.add_argument("--weights", default="", help="per-source sampling weights, e.g. wikipedia:20,fineweb-edu:34,arith:6 "
                                                   "(main corpora and --data2 alike, each normalised within its corpus)")
    ap.add_argument("--recall-span", type=int, default=0,
                    help="keys recall keeps in its carried store in training (0: one context window). Longer (2026-10-08): "
                         "the model sees a store of that size, and text that far back, while it learns")
    ap.add_argument("--mix-auto", type=int, default=0,
                    help="THE MODEL CHOOSES ITS DATA MIX (every N steps; 0 off): each source's weight follows how fast the "
                         "model's loss on a held-out slice of that source is falling (learning progress)")
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
    ap.add_argument("--deepen-init", default="zero", choices=["zero", "copy", "gate", "memory"],
                    help="new layers at a deepen: zero = exact (outputs start at zero); copy = stacking, each new layer "
                         "a copy of the nearest trained layer of its type, with its optimizer memory")
    ap.add_argument("--sf-restart", type=int, default=0,
                    help="at a growth, restart the schedule-free AVERAGE (weight sum 0, base z = the averaged weights), "
                         "keeping every other part of the optimizer's memory: the average then covers the new stage only")
    ap.add_argument("--sf-clock", type=int, default=1,
                    help="sfp: one schedule-free averaging clock per weight (1) or one per group, as the package (0)")
    ap.add_argument("--trial-frac", type=int, default=8,
                    help="a growth trial's branches each run 1/N of the stage so far (at least --trial-steps): long enough for "
                         "new capacity to show (~5-6k steps after the crossover, eval/post_growth.txt), a bounded share of compute")
    ap.add_argument("--resume-stage", default="",
                    help="T0,NEXT: set the growth stage clock on resume (after re-applying a decision the model took before a crash)")
    ap.add_argument("--trial-now", type=int, default=0,
                    help="validation: run one growth trial right after the start/resume (trial length --trial-steps)")
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
    ap.add_argument("--pos-table", type=int, default=0,
                    help="keep the learned position table at N entries while windows run longer; positions wrap modulo N "
                         "inside a window, the condition inference meets at every window boundary (0: table = window)")
    ap.add_argument("--attn-window", type=int, default=512, help="the windowed attention layers' window")
    ap.add_argument("--recall-gated", type=int, default=1,
                    help="recall switched on (--recall or growth op M) with a learnable per-head gate born at -4 (a smooth start)")
    ap.add_argument("--recall", type=int, default=-1,
                    help="RECALL (model.py Attn._recall): each sliding attention query also attends to its top-k keys older than "
                         "the window (k; 0 off). -1: as the checkpoint has it. Growth op M turns it on (16) or doubles it")
    ap.add_argument("--set-window", default="",
                    help="layer:window[,layer:window] -- set individual attention layers' windows after load (an experiment "
                         "on one long-range layer); saved in the checkpoint as attn_windows")
    ap.add_argument("--attn-sliding", type=int, default=0,
                    help="1: a TRUE sliding window in training and inference alike (model.py _swa); 0: the block-local "
                         "training of every checkpoint before 2026-10-02")
    ap.add_argument("--short-conv", type=int, default=0,
                    help="short causal convolution over K tokens on every carrier's q, k, v (identity at birth; added "
                         "exactly to a resumed checkpoint that has none, with a cold optimizer)")
    ap.add_argument("--mixer", default="spin", choices=["spin", "gspin", "fspin", "aspin", "dspin", "wspin", "xspin", "w2spin", "hspin", "hdspin", "sdspin", "altspin", "alt4spin", "rspin", "tspin", "trspin", "aspin1", "hspin1", "r6spin", "grspin", "attn"])
    ap.add_argument("--spike", type=float, default=4.0)
    ap.add_argument("--carry-groups", type=int, default=0,
                    help="carried stream groups used in rotation (default: one per micro-step). With more groups than "
                         "carried micro-steps, consecutive steps continue DIFFERENT streams")
    ap.add_argument("--carry-half", type=int, default=0,
                    help="1: every other micro-step is an ordinary fresh window (as without --carry), the rest continue "
                         "streams -- training on short contexts as before and long ones besides")
    ap.add_argument("--share-d", type=int, default=0,
                    help="1: ONE schedule-free Prodigy step size for the whole model (Prodigy as published) instead of one per "
                         "group. The small group of carrier timescales and frequencies estimated its own, which only rises: "
                         "it ratcheted 6.1e-3 -> 1.7e-2 in run83 and its long memory drifted (2026-10-04)")
    ap.add_argument("--freeze-d", type=int, default=0,
                    help="DIAGNOSTIC: hold schedule-free Prodigy's step size where the resumed run left it (no growth)")
    ap.add_argument("--carry", type=int, default=0,
                    help="N > 0: CARRIED-STATE training. Each micro-step continues its own streams from the previous step's "
                         "window, with every layer's state (carrier memory, attention cache) carried in, detached; streams "
                         "restart every N steps. Trains exactly as inference runs past the window (2026-10-03)")
    ap.add_argument("--heldout-T", type=int, default=2048, help="token length of the growth trials' held-out rows")
    ap.add_argument("--heldout-data2", type=int, default=0,
                    help="1: the trials' held-out sample mirrors the TRAINING mix, second corpus included at its share, "
                         "so a growth that buys what that corpus teaches (long-range retrieval) can show it")
    ap.add_argument("--heldout-long", type=int, default=0,
                    help="N > 0: the trials' held-out is N-token streams (training mix, second corpus at its share) read "
                         "window by window with every state carried -- how the model is used -- so a growth that helps "
                         "inside a window but hurts past it (a longer attention reach) is judged on both")
    ap.add_argument("--spike-adapt", type=int, default=0,
                    help="1: every finite gradient norm enters the guard's history, skipped or not, so a LASTING shift "
                         "in scale (an architecture change) moves the median instead of locking the guard out")
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
    grow_w, grow_d, grow_a, grow_r, grow_m = {}, set(), {}, set(), set()
    for ev in [e for e in a.grow_schedule.split(",") if e]:
        st_, op_ = ev.split(":")
        if op_ == "d": grow_d.add(int(st_))
        elif op_.startswith("a"): grow_a[int(st_)] = int(op_[1:])
        elif op_.startswith("w"): grow_w[int(st_)] = int(op_[1:])
        elif op_ == "R": grow_r.add(int(st_))                    # reach: double the attention window
        elif op_ == "M": grow_m.add(int(st_))                    # recall: far keys on (16) or doubled
        else: raise ValueError(f"unknown growth op {op_!r} in --grow-schedule")
    # sfp: the same schedule-free Prodigy as psf, fused into Fabric's compiled, CUDA-graphed step (goal: training
    # efficiency; psf's per-tensor Python update cost 15% of step time at 46M). SF marks behaviour shared by both.
    SF = a.optim in ("psf", "sfp")
    # growth plan: entries in order; an entry may offer ALTERNATIVES separated by '|' (e.g. "w768|d"), each a '+'-joined
    # list of ops. The model trials every alternative of the next entry against staying, and picks (goal 7: it decides
    # how to grow as well as when)
    auto_events = [[alt.split("+") for alt in e.split("|")] for e in a.grow_auto.split(",") if e]
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
        if a.remat:
            import src.model as _M; _M.REMAT = True
            r.log("activations recomputed layer by layer in backward (--remat)")
        for kv_ in [x for x in a.weights.split(",") if x]:
            k_, v_ = kv_.split(":"); WEIGHTS[k_.strip()] = float(v_)
        bins, ks, ps = loaders(a.data, only=[x for x in a.sources.split(",") if x] or None)
        r.log(f"corpora {ks}" + (f", weights " + " ".join(f"{k_} {100 * p_:.1f}%" for k_, p_ in zip(ks, ps)) if WEIGHTS else ""))
        mix_dev = {}
        if a.mix_auto:
            # THE MIX BY LEARNING PROGRESS (2026-10-06, goal 8: the data-mix fractions were the last hand-set numbers). Each
            # source's held-out slice is cut from its END and never trained on (a small source repeated many times would
            # otherwise look like fast progress by being memorised); 4 rows of the training window from it are the probe.
            for k_ in ks:
                n_ = len(bins[k_]); dl_ = min(n_ // 10, max(4 * (a.ctx + 1) * 2, n_ // 100))
                full_ = bins[k_]; bins[k_] = full_[:n_ - dl_]; dv_ = np.asarray(full_[n_ - dl_:])
                st_ = np.linspace(0, len(dv_) - a.ctx - 2, 4).astype(int)
                mix_dev[k_] = torch.from_numpy(np.stack([dv_[j_:j_ + a.ctx + 1].astype(np.int64) for j_ in st_])).cuda()
            # the trainer's bits-per-byte stays on the ORIGINAL main-corpora weights, comparable with every run before
            ps_eval = np.array([MIX.get(k_, 0.0) for k_ in ks]); ps_eval = ps_eval / ps_eval.sum() if ps_eval.sum() > 0 else ps.copy()
            ps[:] = 1.0 / len(ks)              # no prior: every source equal until the model's first look (a checkpoint's mix wins)
            r.log(f"mix chosen by the model every {a.mix_auto} steps, from {len(ks)} sources (held-out slices cut off), "
                  f"starting equal")
        extra = None
        if a.data2:
            extra = loaders(a.data2)
            r.log(f"second corpus {a.data2} at {a.data2_frac:.0%}: " + " ".join(f"{k_} {100 * p_:.1f}%" for k_, p_ in zip(extra[1], extra[2])))
        PT = a.pos_table or a.ctx                     # position table size; a.ctx stays the longest window
        m = LM(vocab=a.vocab, d=a.d, layers=a.layers, mlp_mult=a.mlp, ctx=PT, mixer=a.mixer, norm=a.norm,
               short_conv=a.short_conv, attn_window=a.attn_window, attn_sliding=bool(a.attn_sliding),
               recall_k=max(0, a.recall), recall_gated=bool(a.recall_gated and a.recall > 0 and not a.resume)).cuda()
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
                            fused=bool(a.fused), sfp=True, sfp_clock=bool(a.sf_clock), cuda_graph=bool(a.graph and a.fused))
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
        def sf_restart(o_):
            """RESTART THE SCHEDULE-FREE AVERAGE at a growth (called while the parameters hold the averaged weights x).
            Late in training the average moves at ~1/t per step, so capacity added at step 30,000 enters the evaluated
            weights at ~1/30,000 per step: new capacity is invisible for thousands of steps whatever it learns. With the
            weight sum at 0 and z = x the model is unchanged, and the average restarts at the growth, covering the new
            stage only. Everything else the optimizer knows (second moments, step size d) is kept."""
            for g in o_.param_groups:
                if a.optim == "psf":
                    g["weight_sum"] = 0.0; g["running_weight_sum"] = 0.0
                elif g.get("sf") is not None:
                    g["sf"]["ws"].zero_()
                for p_ in g["params"]:
                    st_ = o_.state.get(p_)
                    if st_ and "z" in st_: st_["z"] = p_.detach().clone().to(st_["z"].dtype)
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
        def train_span():
            """recall's carried store in training: the previous context window (a fixed shape for the graphs); held-out and
            every loader keep the whole past (span 0)"""
            for mx_ in m.mix:
                if getattr(mx_, "recall_k", 0): mx_.recall_span = a.recall_span or a.ctx
        def capture_carry():
            """two graphs over one buffer of layer states: FRESH (a stream's first window: no state in) and CONTINUED (the
            state in); both write the window's final states back into the buffer after the backward pass, which may
            still read the incoming state (the scan recomputes chunk states from it)"""
            train_span()
            T, B = a.ctx, a.batch
            xs = torch.empty((B, T + 1), dtype=torch.long, device="cuda")
            xs.copy_(batch(bins, ks, ps, np.random.default_rng(11), B, T, extra, a.data2_frac))
            for p_ in m.parameters():
                if p_.grad is None: p_.grad = torch.zeros_like(p_)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                _, st0 = trunk(xs[:, :T], None, True)
            buf = [tuple((t.detach().clone() if torch.is_tensor(t) else torch.tensor(t, device="cuda")) for t in st_)
                   for st_ in st0]
            del st0
            _mdbg = os.environ.get("MEMBRAE_MEMDBG")
            if _mdbg: r.log(f"    [mem] buf {sum(t.numel() * t.element_size() for b_ in buf for t in b_) / 2**30:.2f} GiB "
                            f"({len(buf)} layers); allocated {torch.cuda.memory_allocated() / 2**30:.2f} GiB")
            out = {}
            def micro(fresh):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    h, new = trunk(xs[:, :T], None if fresh else buf, True)
                l_ = m.loss(h, xs[:, 1:T + 1], chunks=max(1, a.chunks), ul_alpha=a.ul, ul_window=a.ul_window)
                (l_ / a.accum).backward()
                with torch.no_grad():
                    for b_, n_ in zip(buf, new):
                        for bt, nt in zip(b_, n_):
                            if torch.is_tensor(nt): bt.copy_(nt.detach())
                            else: bt.fill_(nt)
                out["f" if fresh else "c"] = l_.detach()
            s_ = torch.cuda.Stream(); s_.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s_):
                for fr in (True, True, False, False, False): micro(fr)
            torch.cuda.current_stream().wait_stream(s_)
            if _mdbg: r.log(f"    [mem] after 5 eager micro-steps: allocated {torch.cuda.memory_allocated() / 2**30:.2f} GiB, "
                            f"peak {torch.cuda.max_memory_allocated() / 2**30:.2f}")
            for p_ in m.parameters(): p_.grad.zero_()
            gf = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gf): micro(True)
            gc_ = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gc_, pool=gf.pool()): micro(False)
            for p_ in m.parameters(): p_.grad.zero_()
            if _mdbg: r.log(f"    [mem] after the two graphs: allocated {torch.cuda.memory_allocated() / 2**30:.2f} GiB; "
                            f"one group's saved state {sum(t.numel() * t.element_size() for b_ in buf for t in b_) / 2**30:.2f} GiB x {NG}")
            return dict(gf=gf, gc=gc_, xs=xs, buf=buf, lf=out["f"], lc=out["c"],
                        saved=[[tuple(t.clone() for t in b_) for b_ in buf] for _ in range(NG)])
        tok_total = a.steps * a.batch * a.ctx * a.accum
        r.log(f"{P/1e6:.1f}M params, carrier {100*car/P:.1f}%, "
              f"{tok_total/1e9:.2f}B tokens = {tok_total/P:.1f} tok/param")

        # RESUME. The last run diverged at step 14,400 and the trainer then wrote NaN weights over a healthy
        # checkpoint, so the only survivor was a copy taken by accident for a probe.
        start = 0; grew_ctx = None
        if a.resume and os.path.exists(a.resume):
            sd = torch.load(a.resume, map_location="cuda")
            if sd["cfg"].get("ctx", PT) != PT:
                # A LONG-CONTEXT PHASE. The positional table is sized to the old window; tile the trained
                # pattern to fill the new one. That is exactly what inference already does when it processes a
                # long input as consecutive windows with the state carried, so training this way teaches the
                # model to use the state ACROSS a window boundary rather than only within one.
                old_ctx = sd["cfg"]["ctx"]
                reps = -(-PT // old_ctx)
                w = sd["model"]["p.weight"]
                sd["model"]["p.weight"] = w.repeat(reps, 1)[: PT]
                r.log(f"extended the positional table {old_ctx} -> {PT} by tiling the trained pattern {reps}x")
                sd["cfg"]["ctx"] = PT
                grew_ctx = (old_ctx, sd["cfg"]["d"])
            if sd["cfg"]["d"] != m.d or sd["cfg"].get("layers", m.layers) != m.layers:
                # a grown run: rebuild at the checkpoint's own width AND depth
                m = LM(vocab=a.vocab, d=sd["cfg"]["d"], layers=sd["cfg"].get("layers", a.layers), mlp_mult=a.mlp,
                       ctx=PT, mixer=a.mixer, norm=a.norm, short_conv=a.short_conv,
                       attn_window=a.attn_window, attn_sliding=bool(a.attn_sliding)).cuda()
                if a.fused_scan:
                    import src.assoc as _A
                    _A.FUSED_SCAN = True
                    for _mx in m.mix:
                        if hasattr(_mx, "need_sent"): _mx.need_sent = False
                fwd = torch.compile(m) if a.compile else m
                trunk = torch.compile(m.trunk) if a.compile else m.trunk
                opt = make_opt(m)
            # the attention window and its kind are the checkpoint's (a model that grew its own REACH keeps it); they
            # carry no parameters, so they are set on the modules rather than rebuilt
            if "attn_window" in sd["cfg"] or "attn_sliding" in sd["cfg"]:
                for mx_ in m.mix:
                    if isinstance(mx_, Attn) and mx_.window:
                        mx_.window = sd["cfg"].get("attn_window", mx_.window); mx_.sliding = sd["cfg"].get("attn_sliding", mx_.sliding)
                m.attn_window = sd["cfg"].get("attn_window", m.attn_window); m.attn_sliding = sd["cfg"].get("attn_sliding", m.attn_sliding)
                r.log(f"attention from the checkpoint: window {m.attn_window}, {'sliding' if m.attn_sliding else 'block-local'}")
            if sd["cfg"].get("attn_windows"):
                for k_, v_ in sd["cfg"]["attn_windows"].items(): m.mix[int(k_)].window = int(v_)
                m.attn_windows = {int(k_): int(v_) for k_, v_ in sd["cfg"]["attn_windows"].items()}
            # the checkpoint's recall wins when it has one (the model may have grown it by trial; an automatic resume repeats
            # the launch's --recall and must not undo that); --recall applies to a checkpoint without recall
            rk_ = sd["cfg"].get("recall_k", 0) or max(0, a.recall)
            gt_ = sd["cfg"].get("recall_gated", False) if sd["cfg"].get("recall_k", 0) else bool(a.recall_gated)
            # a checkpoint WITHOUT gates that this run will gate: weights and optimizer load exactly as saved first, the
            # gates are added after (gates_after_ below), so nothing is missing from the state and the optimizer stays warm
            gates_after_ = bool(gt_ and not sd["cfg"].get("recall_gated", False))
            if gates_after_: gt_ = False
            if rk_ or getattr(m, "recall_k", 0):
                had_ = {id(p_) for p_ in m.parameters()}
                m.set_recall(rk_, gated=gt_)
                if any(id(p_) not in had_ for p_ in m.parameters()):
                    # the gates are parameters: the optimizer must list them (in the model's order) before its saved
                    # state is loaded, or the state does not fit and the resumed run starts cold (smoke test, 2026-10-06)
                    opt = make_opt(m); base = [g["lr"] for g in opt.param_groups]
                r.log(f"recall: each sliding attention query also attends to its {rk_} best keys older than its window"
                      + (" (from the checkpoint)" if sd["cfg"].get("recall_k", 0) else " (--recall)"))
            conv_added = bool(a.short_conv) and not any(k_.endswith(".sconv") for k_ in sd["model"])
            if conv_added:
                # the checkpoint has no short convolutions: they are added here as the identity, so the function is
                # unchanged; every other key must load exactly
                miss, unexp = m.load_state_dict(sd["model"], strict=False)
                assert not unexp and miss and all(k_.endswith(".sconv") for k_ in miss), (miss, unexp)
                r.log(f"ADDED short causal convolutions (K={a.short_conv}, identity) to {len(miss)} carrier layers")
            else:
                m.load_state_dict(sd["model"])
            start = sd["step"]
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
            if conv_added:
                r.log("optimizer starts COLD: the added convolutions have no state, and the saved state's parameter order "
                      "no longer matches")
            elif "opt" in sd:
                try:
                    opt.load_state_dict(sd["opt"])
                    if SF: opt.train()   # checkpoints hold the averaged weights; back to training ones
                    if grew_ctx is not None:
                        # the positional table grew with the window, so its second moment is still sized to
                        # the old one. Fabric.grew extends it with the median of the existing rows; without
                        # this the fused update tried to add a 2048-vector to an 8192-vector and the compile
                        # failed rather than silently doing something wrong.
                        opt.grew(m.p.weight, grew_ctx)
                        r.log(f"extended the optimizer's statistics for the positional table to {PT} rows")
                    r.log(f"resumed optimizer state ({len(sd['opt'].get('state', {}))} tensors)")
                except Exception as e:
                    r.log(f"optimizer state present but not loadable ({type(e).__name__}: {str(e)[:300]}); starting cold")
            else:
                r.log("WARNING: checkpoint carries no optimizer state; the first steps will be a shock")
            if a.mix_auto and sd.get("mix"):
                for j_, k_ in enumerate(ks):
                    if k_ in sd["mix"]: ps[j_] = sd["mix"][k_]
                ps[:] = ps / ps.sum()
                r.log("the model's data mix from the checkpoint: " + " ".join(f"{k_} {100 * ps[j_]:.1f}%" for j_, k_ in enumerate(ks)))
            if gates_after_:
                m.set_recall(rk_, gated=True)
                for g_o, g_n in zip(opt.param_groups, m.param_groups(lr=a.lr, new_lr=a.new_lr)):
                    g_o["params"] = list(g_n["params"])
                opt._graph = None; opt._warm = False
                r.log(f"recall gates added after the resume (born at -4; the optimizer state of every other weight kept)")
            r.log(f"resumed from {a.resume} at step {start}, loss {sd['loss']:.4f}")
        # TOKENS ACTUALLY PROCESSED per optimizer step, averaged over the lengths the mix draws. Counting every step
        # at batch x ctx overstated a mixed-length run 2.5x (2026-09-22) and 4x once the window stopped shrinking
        # to the mix (2026-09-23): run47 printed 1,138 ktok/s while processing ~285.
        # a resumed run past a split point starts at the split micro-batch
        for st_, f_ in grow_a.items():
            if st_ < start: shapes = [(t, max(1, B_ // f_)) for t, B_ in shapes]; a.accum *= f_
        _pw = mix_p if mix_p is not None else [1.0 / len(shapes)] * len(shapes)
        tok_per_step = a.accum * sum(q * B_ * T_ for q, (T_, B_) in zip(_pw, shapes))
        # one captured graph per length. The FIRST capture splits under the cap too: only captures after a growth
        # went through capture_all()'s split, and a 32,768-token window died here at startup (2026-10-01).
        if a.set_window:
            for ent in a.set_window.split(","):
                l_, w_ = (int(v) for v in ent.split(":"))
                assert isinstance(m.mix[l_], Attn), f"layer {l_} is not attention"
                m.mix[l_].window = w_; m.attn_windows[l_] = w_
            r.log(f"attention windows set per layer: {m.attn_windows}")
        if a.share_d:
            assert a.optim == "sfp", "--share-d is implemented for the fused optimizer (--optim sfp)"
            opt.share_d = True
            r.log("ONE shared step size for every parameter group (Prodigy as published)")
        if a.freeze_d:
            for g_f in opt.param_groups: g_f["freeze_d"] = True
            r.log("DIAGNOSTIC: the optimizer's step size is frozen at its resumed value")
        carry_g = None
        # (2026-10-03) carrying in EVERY micro-step with streams restarted every 16 steps left 1 window in 16 starting
        # empty: retrieval inside a fresh window fell (needles at 4,096 tokens 100% -> 46%) and 8k-context prediction
        # worsened, and every step continued the same 8 streams. --carry-half keeps ordinary fresh windows in every
        # step; --carry-groups rotates many stream groups so consecutive steps continue different streams.
        NG = a.carry_groups or a.accum
        cursor = [0]
        def carry_batches():
            out_ = []
            for j_ in range(a.accum):
                if a.carry_half and j_ % 2 == 0:
                    out_.append((batch(bins, ks, ps, rng, a.batch, a.ctx, extra, a.data2_frac), True, None))
                else:
                    g_c = cursor[0] % NG; cursor[0] += 1
                    x_c, f_c = streams.next(g_c); out_.append((x_c, f_c, g_c))
            return out_
        if a.carry:
            assert len(shapes) == 1 and a.graph_step and a.compile, "--carry needs one window length and captured graphs"
            carry_g = capture_carry(); graphs = {(a.batch, a.ctx): None}; set_none = False
            r.log(f"CARRIED-STATE training: {NG} stream groups x {a.batch} streams in rotation, each restarted every "
                  f"{a.carry} windows (up to {a.carry * a.ctx:,} tokens of history)"
                  f"{'; every other micro-step a fresh window' if a.carry_half else ''}; "
                  f"{torch.cuda.memory_reserved()/2**30:.1f} GiB reserved")
        while not a.carry:
            try:
                graphs = {(B_, T_): capture(T_, B_) for T_, B_ in shapes}
                break
            except torch.OutOfMemoryError:
                graphs = {}; gc.collect(); torch.cuda.empty_cache()
                if all(B_ == 1 for _, B_ in shapes): raise
                shapes = [(t, max(1, B_ // 2)) for t, B_ in shapes]; a.accum *= 2
                a.batch = shapes[-1][1] if len(shapes) > 1 else max(1, a.batch // 2)
                r.log(f"did not fit under the cap at startup: micro-steps split to {shapes}, accumulation {a.accum}")
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
        if a.resume_stage:
            stage_t0[0], stage_next[0] = (int(v) for v in a.resume_stage.split(","))
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
        C_ = None; grown_ = None                      # (names the step loop and a widening bind; install/roomy clear them)
        mix_prev, mix_prog, mix_share = [None], {}, [None]   # the auto-mix's last losses, smoothed progress, shares since
        skips_t = torch.zeros((), device="cuda"); skips_t = torch.zeros((), device="cuda"); ema_t = None
        nonfin = torch.zeros((), device="cuda"); nf_prev = 0
        marks = []; bpbs = []; skipped = 0; stopped_early = None
        T_n, B_n = shapes[0]
        nxt = [batch(bins, ks, ps, rng, B_n, T_n, extra, a.data2_frac) for _ in range(a.accum)]
        if a.carry:
            streams = Streams(bins, ks, ps, extra, a.data2_frac, a.batch, a.ctx, NG, a.carry, rng)
            nxt = carry_batches()

        # ---- GROWTH TRIALS (goals 7 + 8): the model decides to grow by trying it ----------------------------------
        # When its rate has annealed an e-fold (time to RECONSIDER, nothing more), the model snapshots itself, trains a
        # GROWN copy for --trial-steps, then goes back to the snapshot and trains UNCHANGED for the same steps on the
        # SAME batches, and keeps whichever branch bought more held-out loss per unit of compute (params x steps).
        # Replaces two signals that grew it long before it was saturated: the rate e-fold alone (5 and 2 tokens per
        # parameter in its first stages, 2026-09-24) and a one-step gradient comparison (break-even at 1/3 token per
        # parameter: a gradient says what new capacity COULD do now, not whether the old has run out).
        import copy as _copy
        # THE TRIALS' HELD-OUT SAMPLE. 4 x 2,048 tokens of the main mix by default; with --heldout-data2 it mirrors the
        # training mix exactly (8 rows, the second corpus's share of them drawn from it, deterministic), and
        # --heldout-T 8192 lets a growth of REACH show what it buys: at 2,048 tokens a window past 2,048 can buy nothing
        HT = a.heldout_T
        if a.heldout_long:
            # (2026-10-03) the 8k-window held-out could not see what a longer reach costs past the window: the model chose
            # 8,192 and paid +0.033 bits/byte at 32k of context, while a 512 reach is flat (-0.0003 +- 0.0005)
            HT = a.heldout_long
            r_h = np.random.default_rng(424242); n2 = int(round(8 * a.data2_frac)) if extra is not None else 0
            xs_eval = torch.cat([batch(bins, ks, ps, r_h, 8 - n2, HT)] + ([batch(bins, ks, ps, r_h, n2, HT, extra, 1.0)] if n2 else []))
        elif a.heldout_data2 and extra is not None:
            r_h = np.random.default_rng(424242); n2 = int(round(8 * a.data2_frac))
            xs_eval = torch.cat([batch(bins, ks, ps, r_h, 8 - n2, HT), batch(bins, ks, ps, r_h, n2, HT, extra, 1.0)])
        else:
            xs_eval = batch(bins, ks, ps, np.random.default_rng(424242), 4, HT)
        trial = None
        def heldout():
            def run_():
                if SF: opt.eval()
                m.eval()
                # recall reads the WHOLE past on the held-out streams, as at inference (training keeps one window of it);
                # put back in `finally`: an out-of-memory retry (roomy) once saw the zeroed span as the one to restore
                for mx_ in m.mix:
                    if getattr(mx_, "recall_k", 0): mx_.recall_span = 0
                try:
                  with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    # through the trunk and the chunked loss: the full logits of a large model were what ran a trial out
                    # of memory at the cap (run70, 2026-09-30)
                    if a.heldout_long:
                        # every token of the streams, window by window, the states carried: the loss as the model is used.
                        # WITH RECALL, ONE STREAM AT A TIME: the whole past is kept, ~1.25 GiB per 32k stream, and all
                        # eight at once beside a recall branch's graphs (27 GiB) passed the 34 GiB cap; the failure inside a
                        # kernel killed the trainer twice with SIGSEGV (2026-10-05). The mean over rows is unchanged.
                        nr_ = xs_eval.shape[0]
                        sub_ = 1 if any(getattr(mx_, "recall_k", 0) for mx_ in m.mix) else nr_
                        tot_ = 0.0
                        for r0_ in range(0, nr_, sub_):
                            st_ = None; xr_ = xs_eval[r0_:r0_ + sub_]
                            for w0 in range(0, HT, a.ctx):
                                h_, st_ = m.trunk(xr_[:, w0:w0 + a.ctx], st_, True)
                                tot_ += (m.loss(h_, xr_[:, w0 + 1:w0 + a.ctx + 1], chunks=4).item() * min(a.ctx, HT - w0)
                                         * xr_.shape[0] / nr_)
                            del st_, h_
                        l_ = tot_ / HT
                    else:
                        l_ = m.loss(m.trunk(xs_eval[:, :HT]), xs_eval[:, 1:HT + 1], chunks=4).item()
                finally:
                    train_span()
                    m.train()
                    if SF: opt.train()
                return l_
            return roomy(run_)
        def to_dev(o, dev):
            if torch.is_tensor(o): return o.to(dev)
            if isinstance(o, dict): return {k: to_dev(v, dev) for k, v in o.items()}
            if isinstance(o, (list, tuple)): return type(o)(to_dev(v, dev) for v in o)
            return o
        def snapshot(): return roomy(snapshot_)
        def snapshot_():
            # STRAIGHT TO THE CPU (2026-10-06): deepcopy made the whole model -- and the grads it held -- again on the card,
            # and the widened branch's snapshot (168M) passed the 34 GiB cap at the end of its trial, twice. Every
            # parameter and buffer is copied to the CPU through deepcopy's memo, so the card holds no second copy
            memo_ = {}
            for p_ in m.parameters():
                memo_[id(p_)] = torch.nn.Parameter(p_.detach().to("cpu", copy=True), requires_grad=p_.requires_grad)
            for b_ in m.buffers():
                memo_[id(b_)] = b_.detach().to("cpu", copy=True)
            mc = _copy.deepcopy(m, memo_).to("cpu")
            for p_ in mc.parameters(): p_.grad = None
            S_ = dict(m=mc, opt=to_dev(opt.state_dict(), "cpu"), rng=_copy.deepcopy(rng.bit_generator.state),
                      nxt=[(t[0].clone(),) + tuple(t[1:]) if isinstance(t, tuple) else t.clone() for t in nxt],
                      shapes=list(shapes), accum=a.accum, batch=a.batch, auto=_copy.deepcopy(auto_events))
            if carry_g is not None:
                # CARRIED TRAINING: both branches of a trial must continue the same streams from the same states (2026-10-05):
                # each stream's corpus and position, each group's step count, and every group's carried layer states
                S_["streams"] = (list(streams.k), [[[row[0], row[1]] for row in g_rows] if g_rows else None
                                                   for g_rows in streams.rows], cursor[0])
                S_["saved"] = [[tuple(t.detach().to("cpu") for t in b_) for b_ in grp] for grp in carry_g["saved"]]
            return S_
        def capture_all():
            """(Re)capture every window's graph. If the model does not fit under the cap -- a grown model can outgrow it
            -- split the micro-step (half the rows, twice the accumulation: the same tokens per update) and try again,
            so no split step is written into the growth plan by hand."""
            nonlocal shapes, gs, nxt, carry_g, C_
            if a.carry:
                saved_ = carry_g["saved"] if carry_g else None
                # C_ is the training loop's alias of carry_g: left set, it kept the OLD graphs' pool (15.7 GiB) alive while
                # the new ones were captured, and the first REACH trial branch ran out of memory at its cap six times in a
                # row (2026-10-08, knowledge run, step 389,000)
                carry_g = None; C_ = None; gc.collect(); torch.cuda.empty_cache()
                carry_g = capture_carry()
                if saved_ and len(saved_[0]) == len(carry_g["buf"]) and all(
                        tuple(t.shape for t in a_) == tuple(t.shape for t in b_) for a_, b_ in zip(saved_[0], carry_g["buf"])):
                    carry_g["saved"] = saved_
                else:
                    # the model's layers changed (a growth): the carried states no longer fit; every stream restarts
                    streams.rows = [None] * len(streams.rows)
                    if trial is not None and trial.get("phase") == "grow": trial["restarted"] = True
                    r.log("    carried streams restarted: the layer states changed shape")
                return
            while True:
                try:
                    if len(shapes) > 1: graphs.update({(B_, T_): capture(T_, B_) for T_, B_ in shapes})
                    else: gs = capture()
                    return
                except torch.OutOfMemoryError:
                    graphs.clear(); gs = None; gc.collect(); torch.cuda.empty_cache()
                    if all(B_ == 1 for _, B_ in shapes): raise
                    shapes = [(t, max(1, B_ // 2)) for t, B_ in shapes]; a.accum *= 2
                    a.batch = shapes[-1][1] if len(shapes) > 1 else max(1, a.batch // 2)
                    T_s, B_s = shapes[0]
                    nxt = [batch(bins, ks, ps, rng, B_s, T_s, extra, a.data2_frac) for _ in range(a.accum)]
                    r.log(f"    did not fit under the cap at {sum(p_.numel() for p_ in m.parameters())/1e6:.1f}M: micro-steps "
                          f"split to {shapes}, accumulation {a.accum}")
        roomy_calls = [0]
        def roomy(fn):
            """WORK OUTSIDE THE CAPTURED GRAPHS (held-out loss, bits-per-byte eval, growth exactness checks, snapshots)
            runs beside the graphs' memory pool, and capture_all()'s automatic split only proves that the TRAINING
            STEP fits under the cap. run70 died at the cap in a growth check right after a capture that fit with
            ~0.3 GiB to spare (2026-09-30), as a run had at its first eval on 2026-09-22: each was patched at the call
            site, never as a class. Now if such work does not fit, the graphs are released for it and recaptured
            after (seconds, and only when needed): room is measured by trying, not set by hand."""
            nonlocal gs, g_, carry_g, C_
            try:
                roomy_calls[0] += 1                 # test hook (eval/oom_test.sh): MEMBRAE_TEST_OOM=N fails the Nth call once
                if roomy_calls[0] == int(os.environ.get("MEMBRAE_TEST_OOM", "0")):
                    raise torch.OutOfMemoryError("MEMBRAE_TEST_OOM")
                return fn()
            except torch.OutOfMemoryError:
                pass                               # leaving the except block frees the failed attempt's tensors
            graphs.clear(); gs = None; g_ = None
            # CARRIED TRAINING's graphs are released too (2026-10-06: they were not, so the retry ran beside the same pool and
            # failed again); the streams' carried states are kept and handed back after the recapture
            sv_ = None
            if carry_g is not None:
                sv_ = [[tuple(t.detach().to("cpu") for t in b_) for b_ in grp] for grp in carry_g["saved"]]
                rows_kept_ = [[[r_[0], r_[1]] for r_ in g_r] if g_r else None for g_r in streams.rows]
                carry_g = None; C_ = None
            gc.collect(); torch.cuda.empty_cache()
            r.log(f"    out-of-graph work did not fit beside the captured graphs under the cap: released them for it")
            out = fn()
            with torch.enable_grad():              # a capture inside a caller's no_grad would record no backward
                capture_all()
            if sv_ is not None and carry_g is not None and len(sv_[0]) == len(carry_g["buf"]) and all(
                    tuple(t.shape for t in a_) == tuple(t.shape for t in b_) for a_, b_ in zip(sv_[0], carry_g["buf"])):
                carry_g["saved"] = [[tuple(t.to("cuda") for t in b_) for b_ in grp] for grp in sv_]
                streams.rows = rows_kept_          # the streams continue where they were (capture_all restarted them)
            return out
        def install(S):
            """make snapshot S the live model: optimizer rebuilt from its state, graphs recaptured"""
            nonlocal m, opt, fwd, trunk, gs, g_, base, set_none, shapes, nxt, carry_g, C_, grown_
            graphs.clear(); gs = None; g_ = None
            # RELEASE THE OUTGOING BRANCH FIRST (2026-10-06): its carried-training graphs (their memory pool), its model and
            # its optimizer were still on the card while the snapshot was loaded, and after the widened branch (168M, 1.3 GiB
            # of optimizer state) that passed the 34 GiB cap three times. The snapshot restores the streams and their states
            # ...and every other name that held them: the step loop's own handle on the carried graphs (C_ kept the whole
            # ~20 GiB pool alive after carry_g was dropped: the fourth failure, 06:10 UTC) and a widening's map of old to new
            # parameters
            carry_g = None; C_ = None; grown_ = None; m = None; opt = None
            fwd = trunk = None; torch._dynamo.reset(); gc.collect(); torch.cuda.empty_cache()
            r.log(f"    install: {torch.cuda.memory_allocated() / 2**30:.2f} GiB still allocated after releasing the outgoing "
                  f"branch ({torch.cuda.memory_reserved() / 2**30:.2f} reserved)")
            shapes = S["shapes"]; a.accum = S["accum"]; a.batch = S["batch"]
            rng.bit_generator.state = S["rng"]; nxt = S["nxt"]
            m = S["m"].to("cuda")
            if a.fused_scan:
                for mx_ in m.mix:
                    if hasattr(mx_, "need_sent"): mx_.need_sent = False
            opt = make_opt(m)
            opt.load_state_dict(to_dev(S["opt"], "cuda"))
            if SF: opt.train()
            if a.share_d: opt.share_d = True          # the rebuilt optimizer keeps one step size, as the run does
            if a.freeze_d:
                for g_f in opt.param_groups: g_f["freeze_d"] = True
            base = [g["lr"] for g in opt.param_groups]; set_none = False if SF else not opt.cuda_graph
            fwd = torch.compile(m) if a.compile else m
            trunk = torch.compile(m.trunk) if a.compile else m.trunk
            capture_all()
            if "streams" in S:
                # the snapshot's streams and carried states, AFTER the recapture (which restarts streams when shapes change)
                k_, rows_, cur_ = S["streams"]
                streams.k = list(k_); streams.rows = [[[r_[0], r_[1]] for r_ in g_r] if g_r else None for g_r in rows_]
                cursor[0] = cur_
                carry_g["saved"] = [[tuple(t.to("cuda") for t in b_) for b_ in grp] for grp in S["saved"]]
        def nparams(): return sum(p_.numel() for p_ in m.parameters())
        def check_loss():
            """the growth exactness check: the same light row before and after an event"""
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                return m.loss(m.trunk(xg[:, :2048]), xg[:, 1:2049], chunks=4).item()
        xg = None

        for i in range(start, a.steps):
            if trial is not None and i == trial["end"] - trial["K"] // 2:
                trial["Lg_mid" if trial["phase"] == "grow" else "Ls_mid"] = heldout()
                torch.cuda.synchronize(); trial["t_mid"] = time.time()
            if trial is not None and i == trial["end"]:
                # COMPUTE IS TIME (2026-10-05). The trials priced compute as parameters x steps, so an option that adds
                # time per step without parameters (a longer window R, recall M) looked free. The second half of each
                # branch is now also timed (wall clock between the mid-point held-out and the end, synchronised), and the
                # decision compares held-out gain per SECOND; parameters x steps is still logged beside it
                torch.cuda.synchronize(); tsec_ = time.time() - trial.pop("t_mid", float("nan"))
                if trial["phase"] == "grow":
                    res_ = dict(ev=trial["ev"], Lg_mid=trial.pop("Lg_mid", None), Lg=heldout(), Pg=nparams(), G=snapshot(),
                                Tg=tsec_)
                    trial["res"].append(res_)
                    r.log(f"    TRIAL {'+'.join(res_['ev'])}: grown branch {trial['L0']:.4f} -> {res_['Lg']:.4f} over "
                          f"{trial['K']} steps at {res_['Pg']/1e6:.1f}M")
                    if trial["ai"] + 1 < len(trial["alts"]):
                        # the next alternative, from the same snapshot, on the same batches
                        install(_copy.deepcopy(trial["S0"]))
                        hist.fill_(float("nan")); g0 = i
                        trial["ai"] += 1; trial["ev"] = trial["alts"][trial["ai"]]
                        for op_ in trial["ev"]:
                            if op_ == "d": grow_d.add(i)
                            elif op_ == "W": grow_w[i] = m.d + 256
                            elif op_ == "R": grow_r.add(i)
                            elif op_ == "M": grow_m.add(i)
                            elif op_.startswith("w"): grow_w[i] = int(op_[1:])
                            elif op_.startswith("a"): grow_a[i] = int(op_[1:])
                        trial["end"] = i + trial["K"]
                        continue_trial = True
                    else:
                        continue_trial = False
                    install(trial.pop("S0")) if not continue_trial else None
                    if False:
                        # (SUPERSEDED by psf_carry: neither branch restarts its average now.) THE SAME START FOR BOTH BRANCHES. The grown branch starts from the averaged weights with a
                        # fresh schedule-free average; left alone, the unchanged branch would keep its long average,
                        # which lowers held-out loss by itself, and growth could never win (measured 2026-09-27:
                        # unchanged 5.1266 -> 5.0147 vs grown -> 5.1207 at the second trial). So it too restarts
                        # from its averaged weights, same carried step size: the only difference left is the size.
                        opt.eval(); fresh = make_opt(m); carry_d(opt, fresh); opt = fresh
                    if not continue_trial:
                        hist.fill_(float("nan")); g0 = i
                        trial["phase"], trial["end"] = "stay", i + trial["K"]
                        if a.carry and trial.get("restarted"):
                            # THE SAME BATCHES FOR EVERY BRANCH: a growth that changes the layer states' shapes restarts
                            # every carried stream (capture_all), so the unchanged branch restarts them too; the restored
                            # generator then draws the same new streams (review, 2026-10-05: the stay branch had kept the
                            # snapshot's streams while every grown branch restarted)
                            streams.rows = [None] * len(streams.rows)
                        r.log(f"    TRIAL: now the same {trial['K']} steps unchanged"
                              + (" (carried streams restarted, as in the grown branches)" if a.carry and trial.get("restarted") else ""))
                else:
                    Ls, Ps = heldout(), nparams(); Ts = tsec_
                    # THE BEST ALTERNATIVE: highest marginal held-out gain per unit compute over the second half of its
                    # trial among those not behind the unchanged branch (by total gain per compute if no mid-points)
                    half_ = trial["K"] // 2
                    unit_, mul_ = "unit compute (x1e9)", 1e9
                    timed_ = math.isfinite(Ts) and Ts > 0 and all(math.isfinite(rr.get("Tg", float("nan"))) and rr["Tg"] > 0
                                                                for rr in trial["res"])
                    def score_(rr):
                        if SF and rr["Lg_mid"] is not None and "Ls_mid" in trial:
                            if timed_: return ((rr["Lg"] <= Ls), (rr["Lg_mid"] - rr["Lg"]) / rr["Tg"])
                            return ((rr["Lg"] <= Ls), (rr["Lg_mid"] - rr["Lg"]) / (rr["Pg"] * half_))
                        return (True, (trial["L0"] - rr["Lg"]) / rr["Pg"])
                    for rr in trial["res"]:
                        ok_, sc_ = score_(rr)
                        ps_ = (rr["Lg_mid"] - rr["Lg"]) / (rr["Pg"] * half_) if rr["Lg_mid"] is not None else float("nan")
                        r.log(f"    TRIAL {'+'.join(rr['ev'])}: {'ahead of' if rr['Lg'] <= Ls else 'behind'} unchanged "
                              f"({rr['Lg']:.4f} vs {Ls:.4f}), marginal gain per 1e12 param-steps {ps_*1e12:.4f}"
                              + (f", per hour {sc_*3600:.4f} ({rr['Tg']/half_:.3f} s/step vs unchanged {Ts/half_:.3f})" if timed_ else ""))
                    if not trial["res"]:
                        r.log(f"    DECISION at step {i}: STAY. No growth option fits under the {a.cap:g} GiB cap "
                              f"({', '.join('+'.join(e_) for e_ in trial.get('unfit', []))}); will reconsider when this stage's length doubles again")
                        trial = None
                        continue
                    best_ = max(trial["res"], key=lambda rr: score_(rr))
                    trial.update(ev=best_["ev"], Lg=best_["Lg"], Pg=best_["Pg"], G=best_["G"], Tg=best_.get("Tg", float("nan")))
                    if best_["Lg_mid"] is not None: trial["Lg_mid"] = best_["Lg_mid"]
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
                        if timed_:
                            slope_g = (trial["Lg_mid"] - trial["Lg"]) / trial["Tg"]
                            slope_s = (trial["Ls_mid"] - Ls) / Ts
                            unit_, mul_ = "hour", 3600
                        else:
                            slope_g = (trial["Lg_mid"] - trial["Lg"]) / (trial["Pg"] * half)
                            slope_s = (trial["Ls_mid"] - Ls) / (Ps * half)
                            unit_, mul_ = "1e12 param-steps", 1e12
                        grow_now = slope_g > slope_s and trial["Lg"] <= Ls
                        r.log(f"    TRIAL {'+'.join(trial['ev'])}: marginal held-out gain per {unit_} over the last "
                              f"{half} steps: grown {slope_g*mul_:.4f} vs unchanged {slope_s*mul_:.4f}; grown "
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
                        r.log(f"    DECISION at step {i}: GROW ({'+'.join(trial['ev'])}). Held-out gain per {unit_} "
                              f"{gg*mul_:.4f} grown vs {gs_*mul_:.4f} unchanged ({trial['L0']:.4f} -> {trial['Lg']:.4f} vs "
                              f"-> {Ls:.4f}); continuing from the grown branch at {nparams()/1e6:.1f}M")
                    else:
                        if not SF: stage_rate[0] = opt.rates()[0]
                        r.log(f"    DECISION at step {i}: STAY. Held-out gain per {unit_} {gs_*mul_:.4f} unchanged vs "
                              f"{gg*mul_:.4f} grown ({trial['L0']:.4f} -> {Ls:.4f} vs -> {trial['Lg']:.4f}); "
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
            unfit_ = False
            try:
                if i in grow_r:
                    # REACH (goals 7 + 9): double how far back the windowed attention sees exactly. Exact retrieval lives
                    # only inside this window (needles: 71% at 256 tokens, 0% from 1,024; 2026-10-02), so this is the growth
                    # that buys it, and the model decides by trial whether it pays. NOT exact: the same weights now attend
                    # over more keys, so the cost of the event is logged rather than assumed away.
                    ats = [mx for mx in m.mix if isinstance(mx, Attn) and mx.window]
                    if not ats or not all(mx.sliding for mx in ats) or ats[0].window >= a.ctx:
                        r.log(f"    REACH not applied at step {i}: it needs sliding windowed attention shorter than the "
                              f"training window ({ats[0].window if ats else 0} vs {a.ctx})")
                    else:
                        if SF: opt.eval()
                        xg = batch(bins, ks, ps, np.random.default_rng(7), 1, 2048)
                        lb = roomy(check_loss)
                        for mx in ats: mx.window *= 2
                        m.attn_window = ats[0].window
                        graphs.clear(); gs = None; g_ = None; gc.collect()
                        fwd = trunk = None; torch._dynamo.reset(); torch.cuda.empty_cache()
                        fwd = torch.compile(m) if a.compile else m
                        trunk = torch.compile(m.trunk) if a.compile else m.trunk
                        capture_all()
                        la = roomy(check_loss)
                        r.log(f"    REACH doubled to a {m.attn_window}-token window at step {i}: same-row loss {lb:.4f} -> {la:.4f}")
                        if SF: opt.train()
                        hist.fill_(float("nan")); g0 = i
                if i in grow_m:
                    # RECALL (goal 9: retrieval past the window; model.py Attn._recall): each sliding attention query also
                    # attends to its top-k keys older than the window, by its own q.k. Untrained it took needles at 1-32k from
                    # 0% to 100-31% but cost prediction past the first context window (+0.11 at 16-32k); the trial says whether,
                    # trained, it pays on the long held-out streams. No parameters: NOT exact, the event's cost is logged.
                    if SF: opt.eval()
                    xg = batch(bins, ks, ps, np.random.default_rng(7), 1, 2048)
                    lb = roomy(check_loss)
                    had_ = {id(p_) for p_ in m.parameters()}
                    m.set_recall(16 if not getattr(m, "recall_k", 0) else 2 * m.recall_k, gated=bool(a.recall_gated))
                    if any(id(p_) not in had_ for p_ in m.parameters()):
                        # THE GATES ARE NEW PARAMETERS: the optimizer's groups are re-listed in the model's own order (as a
                        # rebuilt model lists them on resume, so the saved state maps back), the new ones initialised lazily
                        # with their averaging clock where their group's stands; the captured step is rebuilt
                        for g_o, g_n in zip(opt.param_groups, m.param_groups(lr=a.lr, new_lr=a.new_lr)):
                            g_o["params"] = list(g_n["params"])
                        opt._graph = None; opt._warm = False
                        for p_ in m.parameters():
                            if p_.grad is None: p_.grad = torch.zeros_like(p_)
                    graphs.clear(); gs = None; g_ = None; gc.collect()
                    fwd = trunk = None; torch._dynamo.reset(); torch.cuda.empty_cache()
                    fwd = torch.compile(m) if a.compile else m
                    trunk = torch.compile(m.trunk) if a.compile else m.trunk
                    capture_all()
                    la = roomy(check_loss)
                    r.log(f"    RECALL on at step {i}: each query also attends to its {m.recall_k} best keys older than the "
                          f"{m.attn_window}-token window; same-row loss {lb:.4f} -> {la:.4f}")
                    if SF: opt.train()
                    hist.fill_(float("nan")); g0 = i
                if i in grow_w and m.d < grow_w[i]:
                    # GROWTH (vision goal 7): widen in place, function preserved, optimizer state carried. The
                    # loss right before and right after is logged so the cost of the event is a number.
                    if SF: opt.eval()          # measured and grown at the AVERAGED weights, the model
                    xg = batch(bins, ks, ps, np.random.default_rng(7), 1, 2048)   # the exactness check: one row, light
                    lb = roomy(check_loss)
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
                    if SF and a.sf_restart: sf_restart(opt)
                    opt.zero_grad(set_to_none=True)             # grad buffers are the old shape
                    fwd = torch.compile(m) if a.compile else m
                    trunk = torch.compile(m.trunk) if a.compile else m.trunk
                    # every window's graph, not only the first: a mixed-length run held stale graphs after widening
                    capture_all()
                    la = roomy(check_loss)
                    P = sum(p.numel() for p in m.parameters())
                    r.log(f"    GREW to d={m.d} at step {i}: {P/1e6:.1f}M params, same batch loss {lb:.4f} -> {la:.4f}, "
                          f"optimizer state {opt.state_bytes()/2**20:.0f} MiB carried")
                    if SF: opt.train()
                    grown_ = None                         # it held every old AND new parameter alive (2026-10-06)
                    hist.fill_(float("nan")); g0 = i      # the grown model's gradient norm is its own; re-learn it
                    restart_rates()
                if i in grow_d:
                    # DEPTH GROWTH MID-RUN (vision goal 7): the model doubles its layers in place and computes exactly
                    # what it did before (new layers start at zero output; measured 1.949016 -> 1.949016 on run47).
                    # Every existing layer keeps its optimizer state; the new ones get their own absolute-step group.
                    # Growing a shallow model late in the run costs half the compute per token until then.
                    if SF: opt.eval()          # measured and deepened at the AVERAGED weights, the model
                    xg = batch(bins, ks, ps, np.random.default_rng(7), 1, 2048)   # the exactness check: one row, light
                    lb = roomy(check_loss)
                    old_m = m
                    if a.deepen_init == "memory":
                        # ONE CARRIER LAYER ON TOP built to hold for long and to write only what it learns to (2026-10-05):
                        # every head at the long end of the decay range, writes CLOSED by default, output zero (exact). The
                        # same as eval/deepen.py memory, here with every existing weight's optimizer state carried
                        big = old_m.add_layers(1)
                        nm_ = big.mix[big.new_layers[0]]
                        assert hasattr(nm_, "theta"), "the added layer must be a carrier"
                        with torch.no_grad():
                            nm_.A.fill_(math.log(1e-4 / 0.69)); nm_.wb.bias.fill_(-6.0)
                    else:
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
                                        if "wsp" in opt2.state[pn]: opt2.state[pn]["wsp"].zero_()   # a copy is new capacity: own clock from 0
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
                            # the live device scalars if this optimizer has stepped, else the saved ones: a deepen on the
                            # FIRST step after a resume or a trial restore found no live values and restarted d at 1e-6,
                            # which froze every post-crossover deepen test of 2026-09-30 (fixed that day)
                            f_ = src_.get("sf") or {k: torch.tensor(v) for k, v in src_.get("sf_saved", {}).items()}
                            if f_:
                                g_["sf_saved"] = {"d": float(f_["d"]), "dnum": float(f_["dnum"]) if gi_ < len(opt.param_groups) else 0.0,
                                                  "ws": float(f_["ws"]) if gi_ < len(opt.param_groups) else 0.0}
                                if gi_ >= len(opt.param_groups):
                                    # THE ADDED LAYERS' ABSOLUTE STEP (2026-10-09). sfp gives every group the shared d; the
                                    # documented --new-lr was never applied to it, so zero-output layers with empty optimizer
                                    # memory took the full shared step from their first update: at d 3.3e-3 a deepened trial
                                    # branch went 2.40 -> 7.71 in 40 steps (it had been stable at d 9.7e-4). The group's
                                    # multiplier makes its step d x lr = --new-lr (the step size is held: --freeze-d).
                                    g_["lr"] = a.new_lr / max(float(f_["d"]), 1e-12)
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
                                        if "wsp" in opt2.state[pn]: opt2.state[pn]["wsp"].zero_()   # a copy is new capacity: own clock from 0
                    # free the shallow model's CUDA graphs BEFORE capturing the deep one's: both at once would be
                    # ~15 + ~28 GiB, over the cap
                    graphs.clear(); gs = None; g_ = None; gc.collect()
                    if SF and a.sf_restart: sf_restart(opt2)
                    m, opt = big, opt2
                    if a.share_d: opt.share_d = True          # the rebuilt optimizer keeps ONE step size, as before the growth
                    if a.freeze_d:
                        for g_f in opt.param_groups: g_f["freeze_d"] = True
                    del old_m, opt2, big, pairs      # pairs held every shallow parameter (and its grad) alive
                    base = [g["lr"] for g in opt.param_groups]
                    set_none = False if SF else not opt.cuda_graph   # psf: graphs write into fixed grad buffers
                    fwd = trunk = None; torch._dynamo.reset(); gc.collect(); torch.cuda.empty_cache()
                    fwd = torch.compile(m) if a.compile else m
                    trunk = torch.compile(m.trunk) if a.compile else m.trunk
                    capture_all()
                    la = roomy(check_loss)
                    P = sum(p.numel() for p in m.parameters())
                    r.log(f"    DEEPENED to {m.layers} layers at step {i}: {P/1e6:.1f}M params, same batch loss "
                          f"{lb:.4f} -> {la:.4f}, optimizer state carried for {carried} tensors, "
                          f"{len(m.new_layers)} new layers at absolute lr {a.new_lr}, ramped over {a.new_warm} steps")
                    if SF: opt.train()
                    grown_at = i; hist.fill_(float("nan")); g0 = i
                    restart_rates()
            except torch.OutOfMemoryError:
                # A GROWTH THAT DOES NOT FIT IS NOT A CANDIDATE (2026-10-09, goal 6: training on the card it has). Building
                # the deepened 168M model (24 layers) ran out of memory at the 32 GiB cap and crashed the run; the trial
                # now logs the option as not fitting, restores its snapshot, and goes on to the next option or to the
                # unchanged branch. An option that does not fit is never chosen; if none fits, the model stays.
                if trial is None or trial["phase"] != "grow": raise
                unfit_ = True
            if unfit_:
                # (outside the except clause: inside it the traceback kept the half-built model and its graphs alive,
                # and restoring the snapshot found 25.75 GiB still allocated -- tested 2026-10-09 at a 26 GiB cap)
                r.log(f"    TRIAL {'+'.join(trial['ev'])}: does not fit under the {a.cap:g} GiB cap (out of memory building "
                      f"it at step {i}); not a candidate")
                trial.setdefault("unfit", []).append(trial["ev"])
                grow_d.discard(i); grow_w.pop(i, None); grow_r.discard(i); grow_m.discard(i)
                big = opt2 = old_m = pairs = None; xg = None; gc.collect(); torch.cuda.empty_cache()
                if trial["ai"] + 1 < len(trial["alts"]):
                    install(_copy.deepcopy(trial["S0"]))
                    hist.fill_(float("nan")); g0 = i
                    trial["ai"] += 1; trial["ev"] = trial["alts"][trial["ai"]]
                    for op_ in trial["ev"]:
                        if op_ == "d": grow_d.add(i + 1)
                        elif op_ == "W": grow_w[i + 1] = m.d + 256
                        elif op_ == "R": grow_r.add(i + 1)
                        elif op_ == "M": grow_m.add(i + 1)
                        elif op_.startswith("w"): grow_w[i + 1] = int(op_[1:])
                        elif op_.startswith("a"): grow_a[i + 1] = int(op_[1:])
                    trial["end"] = i + 1 + trial["K"]
                else:
                    install(trial.pop("S0"))
                    hist.fill_(float("nan")); g0 = i
                    trial["phase"], trial["end"] = "stay", i + 1 + trial["K"]
                    if a.carry and trial.get("restarted"): streams.rows = [None] * len(streams.rows)
                    r.log(f"    TRIAL: now the same {trial['K']} steps unchanged"
                          + (" (carried streams restarted, as in the grown branches)" if a.carry and trial.get("restarted") else ""))
                continue
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
                if carry_g is not None:
                    # this group's streams continue: its saved states into the graphs' buffer, the window in, replay,
                    # the new states back out
                    x, fresh_, grp = nxt[_acc]; C_ = carry_g
                    C_["xs"].copy_(x)
                    if not fresh_:
                        for b_, s_ in zip(C_["buf"], C_["saved"][grp]):
                            for bt, st in zip(b_, s_): bt.copy_(st)
                    (C_["gf"] if fresh_ else C_["gc"]).replay()
                    if grp is not None:
                        for b_, s_ in zip(C_["buf"], C_["saved"][grp]):
                            for bt, st in zip(b_, s_): st.copy_(bt)
                    l_ = (C_["lf"] if fresh_ else C_["lc"]).clone()
                    loss = l_ if loss is None else loss + l_
                    continue
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
            nxt = (carry_batches() if carry_g is not None else
                   [batch(bins, ks, ps, rng, B_n, T_n, extra, a.data2_frac) for _ in range(a.accum)])
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
            # only kept steps enter the history -- unless --spike-adapt: widening the attention window to 8,192 raised
            # the gradient norm for good, every step after the first 200 read as a spike against the old median, none
            # entered the history, and the run was stopped as DIVERGED at a healthy loss of 2.60 (2026-10-02). A real
            # divergence is still caught by the weights' finiteness and the held-out eval.
            hist[j % 1000] = torch.where(keep | (fin & bool(a.spike_adapt)), gn, hist[j % 1000])
            skips_t.add_(~keep)
            # zero the gradients of a skipped step; a non-finite one becomes NaN * 0 here and is zeroed for
            # good inside the fused update (nan_to_num), so the optimizer state never sees it
            torch._foreach_mul_([p_.grad for p_ in m.parameters() if p_.grad is not None], keep.float())
            # and under schedule-free the skipped step must not move anything (optim._sfp_update, live)
            if hasattr(opt, "set_live"): opt.set_live(keep)
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
                    if a.trial_now and (i + 1 - start) >= a.trial_steps // 4 and not getattr(main, "_trial_done", False):
                        due = True; main._trial_done = True
                        stage_t0[0] = i + 1 - 4 * a.trial_steps          # so the trial length is exactly --trial-steps
                        rt = stage_rate[0] = 1.0
                    elif SF:
                        due = (i + 1 - stage_t0[0]) >= stage_next[0]
                        rt = stage_rate[0] = 1.0
                    else:
                        rt = opt.rates()[0]; due = rt < stage_rate[0] / math.e
                    if due:
                        stage_next[0] = 2 * (i + 1 - stage_t0[0])
                        alts = auto_events[0]; ev = alts[0]
                        # THE TRIAL GROWS WITH THE STAGE: a quarter of its length so far, at least --trial-steps. At a
                        # fixed 1,000 steps the widened copy learned exactly as much as the unchanged one (advantage
                        # -0.0002 at step 26,000): new capacity starts as a zero-output no-op and pays off over longer
                        # than that, so a short trial under-credits growth for good. Scale-free: the cost of a trial is
                        # a fixed fraction of the stage it decides about.
                        K_ = max(a.trial_steps, (i + 1 - stage_t0[0]) // a.trial_frac)
                        if i + 1 + 2 * K_ >= a.steps: K_ = 0
                        L0_ = heldout()
                        if mark[0] is not None:
                            s0_, l0_, p0_ = mark[0]
                            rate_hist.append((l0_ - L0_) / max(1.0, p0_ * (i + 1 - s0_)))
                        trial = dict(phase="grow", ev=ev, alts=alts, ai=0, res=[], L0=L0_, S0=snapshot(), K=K_,
                                     end=i + 1 + K_) if K_ else None
                        for op_ in ev:
                            if op_ == "d": grow_d.add(i + 1)
                            elif op_ == "W": grow_w[i + 1] = m.d + 256        # widen one step, no size written down
                            elif op_ == "R": grow_r.add(i + 1)                # double the attention window's reach
                            elif op_ == "M": grow_m.add(i + 1)                # recall past the window: on, or doubled
                            elif op_.startswith("w"): grow_w[i + 1] = int(op_[1:])
                            elif op_.startswith("a"): grow_a[i + 1] = int(op_[1:])
                        why = (f"this stage's length has doubled to {i+1-stage_t0[0]} steps" if SF else
                               f"rate {rt:.2e} is one e-fold below the stage start {stage_rate[0]:.2e}")
                        r.log(f"    RECONSIDER at step {i+1}: {why}; TRIAL of {' or '.join('+'.join(x) for x in alts)} vs unchanged, "
                              f"{max(a.trial_steps, (i + 1 - stage_t0[0]) // a.trial_frac)} steps each")
                if nnf > nf_prev:
                    # the guard zeroed those steps; the run continues unless the weights themselves went bad
                    alive = bool(torch.stack([p_.isfinite().all() for p_ in m.parameters()]).all().item())
                    recent_nf = nnf - nf_prev; nf_prev = nnf
                    r.log(f"    {recent_nf} non-finite loss(es) in the last 200 steps (total {nnf}); "
                          f"weights {'still finite' if alive else 'NON-FINITE'}")
                    if not alive or recent_nf > 100:
                        r.log(f"    DIVERGED: the weights are not finite; stopping rather than burning the card")
                        stopped_early = "diverged"; break
                if not math.isfinite(ema):
                    r.log(f"    NON-FINITE RUNNING LOSS with finite weights; stopping, this should not happen")
                    stopped_early = "non-finite loss"; break
                if nsk > skipped:
                    r.log(f"    {nsk - skipped} spike(s) skipped in the last 200 steps (median |g| {medv:.2f}, total {nsk})")
                marks.append((i + 1, nsk, ema)); skipped = nsk
                # DIVERGED, not spiking: the attention control blew up at step ~11,000 and the guard then
                # skipped every one of the next 27,000 steps while the log showed a healthy-looking rate.
                # A guard that skips everything is a frozen model burning the card; stop and say so.
                # ...and NOT LEARNING: the 8,192-token attention window raised its gradient norm while the loss fell
                # 2.6026 -> 2.5733, and this stop killed it twice as "diverged" (2026-10-02). Skipping and frozen is
                # the failure; skipping while the running loss keeps falling is not.
                if len(marks) >= 4 and nsk - marks[-4][1] >= 250 and not (ema < marks[-4][2] - 0.01):
                    r.log(f"    DIVERGED: {nsk} skips, {nsk - marks[-4][1]} of them within the last 600 steps, and the loss "
                          f"is not falling ({marks[-4][2]:.4f} -> {ema:.4f}); stopping")
                    stopped_early = "diverged"; break
            if a.mix_auto and (i + 1) % a.mix_auto == 0 and trial is None:
                def mix_losses_():
                    if SF: opt.eval()
                    m.eval(); out_ = {}
                    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                        for k_, x_ in mix_dev.items():
                            out_[k_] = m.loss(m.trunk(x_[:, :a.ctx]), x_[:, 1:a.ctx + 1], chunks=4).item()
                    m.train()
                    if SF: opt.train()
                    return out_
                Lm_ = roomy(mix_losses_)
                def mix_align_():
                    # THE MIX BY GRADIENT ALIGNMENT (2026-10-06; DoGE, Fan et al. 2023). Learning progress, even absolute and
                    # per share, let the main corpora and the needles get WORSE (dclm 2.719 -> 2.726, needle 2.338 -> 2.348 in
                    # 4,000 steps) while the new instruction sets improved fast. The objective is the model's held-out loss
                    # AVERAGED OVER EVERY SOURCE: each source's weight is how well a gradient step on its training data
                    # (fresh rows) agrees with the gradient that lowers that average (the held-out slices). A source whose
                    # training would hurt the others gets only the floor. torch.autograd.grad leaves the graphs' .grad alone.
                    params_ = [p_ for p_ in m.parameters() if p_.requires_grad]; Tg = min(2048, a.ctx)
                    gT = [torch.zeros_like(p_, dtype=torch.float32) for p_ in params_]
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        for k_, x_ in mix_dev.items():
                            l_ = m.loss(m.trunk(x_[:, :Tg]), x_[:, 1:Tg + 1], chunks=1)      # all 4 held-out rows
                            for t_, g_ in zip(gT, torch.autograd.grad(l_, params_, allow_unused=True)):
                                if g_ is not None: t_.add_(g_.float(), alpha=1.0 / len(mix_dev))
                        dots_ = {}; rr_ = np.random.default_rng(i)
                        for k_ in ks:
                            a_ = bins[k_]                                               # 4 fresh training rows
                            x_ = torch.from_numpy(np.stack([np.asarray(a_[j_:j_ + Tg + 1]).astype(np.int64)
                                                            for j_ in rr_.integers(0, len(a_) - Tg - 2, 4)])).cuda()
                            l_ = m.loss(m.trunk(x_[:, :Tg]), x_[:, 1:Tg + 1], chunks=1)
                            gs_ = torch.autograd.grad(l_, params_, allow_unused=True)
                            dots_[k_] = float(sum((g_.float() * t_).sum() for g_, t_ in zip(gs_, gT) if g_ is not None))
                    del gT
                    return dots_
                D_ = roomy(mix_align_)
                pos_ = np.array([max(0.0, D_[k_]) for k_ in ks])
                if pos_.sum() > 0:
                    w_ = pos_ + pos_.mean() / len(ks); w_ /= w_.sum()
                    ps[:] = 0.8 * ps + 0.2 * w_; ps[:] = ps / ps.sum()   # smoothed: 4 rows of 2,048 per source are still noisy
                r.log("    MIX " + " ".join(f"{k_} {100 * ps[j_]:.1f}% ({Lm_[k_]:.3f}, {D_[k_]:+.2e})" for j_, k_ in enumerate(ks)))
                mix_prev[0] = Lm_; mix_share[0] = ps.copy()
            if (i + 1) % a.eval_every == 0:
                # BITS PER BYTE, not loss per token. Per-token loss depends on the vocabulary and is
                # comparable to nothing; bits per byte removes the tokenizer and is the only number that can
                # be set against another model.
                # BOUNDED, and the same length whatever the run trains at. Evaluating at a.ctx put a
                # 32,768-token forward beside a 32.6 GiB graph pool and the run died at its first eval with
                # the batch already halved to one (2026-09-22). A fixed 8,192 also makes the number
                # comparable across runs, which evaluating at each run's own window never was.
                torch.cuda.empty_cache()
                def bpb_():
                    if SF: opt.eval()      # measured at the averaged weights, which are the model
                    b_ = eval_bpb(m, tok, bins, ks, ps_eval if a.mix_auto else ps, np.random.default_rng(1234),
                                  n_batches=8, B=4, T=min(a.ctx, 8192), vocab=a.vocab)
                    if SF: opt.train()
                    return b_
                b, nb = roomy(bpb_)
                torch.cuda.empty_cache()
                r.log(f"  eval step {i+1}: BPB {b:.4f}  ({nb/1e3:.0f}k bytes)")
                # DEGRADING, not frozen: fspin at 6e-3 went 1.7075 -> 1.7534 -> 2.1130 over three evals while
                # the skip guard saw only 200 skips in 15k steps. Three consecutive rises, each above the
                # eval's own noise (measured spread ~0.005 on this fixed sample), is a run going backwards.
                bpbs.append(b)
                if len(bpbs) >= 4 and all(bpbs[-k] > bpbs[-k-1] + 0.01 for k in (1, 2, 3)):
                    r.log(f"    DEGRADING: BPB rose over the last three evals {bpbs[-4:]}; stopping")
                    stopped_early = "degrading"; break
            # never checkpoint a trial's GROWN branch: it is a temporary model (run70 saved one at step 136,000)
            if ((i + 1) % a.save_every == 0 or i + 1 == a.steps) and not (trial is not None and trial["phase"] == "grow"):
                # NEVER OVERWRITE A GOOD CHECKPOINT WITH A BROKEN ONE. The previous run saved NaN weights over
                # a healthy checkpoint and only a stray copy survived. Verify before writing, and keep the
                # previous file as last-known-good.
                if SF: opt.eval()          # the checkpoint holds the averaged weights
                bad = roomy(lambda: sum(1 for v in m.state_dict().values() if not torch.isfinite(v).all()))
                if bad:
                    r.log(f"    refusing to checkpoint: {bad} non-finite tensors")
                else:
                    if os.path.exists(f"{a.out}/ckpt.pt"):
                        os.replace(f"{a.out}/ckpt.pt", f"{a.out}/ckpt_prev.pt")
                    torch.save({"model": m.state_dict(), "opt": opt.state_dict(),
                                "step": i + 1, "loss": ema_t.item(),   # the current average, not the last log's
                                "cfg": dict(vocab=a.vocab, d=m.d, layers=m.layers,
                                            mlp_mult=a.mlp, ctx=m.ctx, mixer=a.mixer, norm=a.norm,
                                            **({"short_conv": m.short_conv} if getattr(m, "short_conv", 0) else {}),
                                            **({"attn_window": m.attn_window, "attn_sliding": m.attn_sliding}
                                               if (m.attn_window, m.attn_sliding) != (512, False) else {}),
                                            **({"attn_windows": dict(m.attn_windows)} if getattr(m, "attn_windows", None) else {}),
                                            **({"recall_k": m.recall_k} if getattr(m, "recall_k", 0) else {}),
                                            **({"recall_gated": True} if getattr(m, "recall_gated", False) else {})),
                                "new_layers": list(getattr(m, "new_layers", [])),
                                "stage": (stage_t0[0], stage_next[0]) if a.grow_auto else None,
                                "mix": {k_: float(ps[j_]) for j_, k_ in enumerate(ks)} if a.mix_auto else None},
                               f"{a.out}/ckpt.pt")
                if SF: opt.train()
        if ema_t is None:
            r.log(f"NOTHING TO DO: resumed at step {start} with --steps {a.steps}. A resumed run counts from the "
                  f"checkpoint's step, so --steps must exceed it.")
        else:
            ema = ema_t.item(); skipped = int(skips_t.item())
            r.log(f"final loss {ema:.4f} ppl {math.exp(min(20,ema)):.1f}, {skipped} spike-skips")
        if stopped_early:
            # NOT a success: run80's "DIVERGED" stop exited 0, so its wrapper wrote the done marker and the pipeline
            # began fine-tuning a 2,000-step checkpoint (2026-10-02). Exit 3 lets keep_training.sh resume or give up.
            raise SystemExit(f"stopped early: {stopped_early}")


if __name__ == "__main__":
    main()
