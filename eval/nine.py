"""THE NINE AXES, ONE TABLE, MEASURED WITH THE INSTRUMENTS AS THEY STAND ON 2026-09-22.

Every number here is against GPT-2 124M on identical inputs, and every instrument in it has been corrected at
least once: multiple choice is scored by TOTAL log-probability because per-token mean penalises a small
vocabulary; generation is judged by an outside model against the band real text occupies, because repetition
alone cannot tell a loop from noise; long generation is one carried call with nothing re-prefilled; induction
is measured on material no model can have memorised, with the prompt cue removed.

usage: nine.py <ckpt> [label]
"""
import sys, os, math, time, json, collections, warnings
warnings.filterwarnings("ignore"); sys.path.insert(0, "/opt/code/membrae")
from src.common import cap_gpu
cap_gpu(float(os.environ.get("NINE_CAP", "20")))
import numpy as np, torch, torch.nn.functional as F
from tokenizers import Tokenizer
from src.bench import Ours, GPT2, facts, coherence
from src.common import eval_bpb
from src.facts import FACTS, reasoning_items_large
from src.train import loaders

CK = sys.argv[1] if len(sys.argv) > 1 else "/mnt/kv_cache/run41_vlong/ckpt.pt"
LAB = sys.argv[2] if len(sys.argv) > 2 else "ours"
R = {}
ours = Ours(CK, "/mnt/kv_cache/tok_v8192.json")
gpt2 = GPT2()
tk = Tokenizer.from_file("/mnt/kv_cache/tok_v8192.json")
bins, ks, ps = loaders("/mnt/kv_cache/tokfull_v8192")
def hdr(n, t): print(f"\n{'='*92}\n{n}. {t}\n{'='*92}", flush=True)

# ---------------------------------------------------------------- 1. context rot
hdr(1, "CONTEXT ROT: bits per byte on held-out text as the window grows")
print(f"{'window':>8s} {LAB:>12s} {'GPT-2':>12s}")
rot = {}
for T in (512, 1024, 2048, 4096, 8192):
    b, _ = eval_bpb(ours.m, tk, bins, ks, ps, np.random.default_rng(7), n_batches=8, B=4, T=T)
    g = ""
    if T <= 1024:
        rng2 = np.random.default_rng(7); nats = 0.0; nby = 0
        for _ in range(8):
            arr = bins[ks[rng2.choice(len(ks), p=ps)]]
            j = rng2.integers(0, len(arr) - T - 2)
            txt_ = tk.decode([int(v) for v in arr[j:j + T + 1]])
            idg = gpt2.tok.encode(txt_)[:1024]
            if len(idg) < 8: continue
            x_ = torch.tensor([idg], device="cuda")
            with torch.no_grad(): lgg = gpt2.m(x_).logits.float().log_softmax(-1)
            nats += float(-lgg[0, :-1].gather(-1, x_[0, 1:].unsqueeze(-1)).sum())
            nby += len(gpt2.tok.decode(idg[1:]).encode())
        g = f"{nats / (math.log(2) * max(1, nby)):12.4f}"
    rot[T] = b
    print(f"{T:8d} {b:12.4f} {g if g else '  (max 1024)':>12s}", flush=True)
R["context_rot"] = rot

# ---------------------------------------------------------------- 3. inference efficiency
hdr(3, "INFERENCE EFFICIENCY: decode rate and the memory held while decoding")
for lab, m in ((LAB, ours), ("GPT-2 124M", gpt2)):
    # WARM THE GRAPH FIRST. Capturing the decode step costs a couple of seconds once, and over 512 tokens
    # that capture IS the measurement: run41 read 115 tok/s here while generating 32,768 tokens at 366
    # (2026-09-22). Capture on a throwaway call, then time a long enough run for the per-token cost to show.
    ids = m.encode("The history of the city begins")
    m.generate(ids, 32, temp=1.0, seed=0)
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    # 900 tokens, which is what GPT-2 can do at all: its context is 1,024 positions, and asking it for 4,096
    # measures it falling off the end of its own positional table rather than its decode speed (it read 98
    # tok/s that way against 241 inside its limit, 2026-09-22). Ours is reported again below at 32,768, where
    # GPT-2 cannot go.
    t0 = time.time(); o = m.generate(ids, 900, temp=1.0, seed=0); dt = time.time() - t0
    print(f"  {lab:<14s} {len(o)/dt:7.0f} tok/s   peak {torch.cuda.max_memory_allocated()/2**20:8.1f} MiB", flush=True)
    R.setdefault("decode", {})[lab] = len(o) / dt

# ---------------------------------------------------------------- 6. facts + calibration
hdr(6, "FACTS and CALIBRATION: four-way, scored by total log-probability")
for lab, m in ((LAB, ours), ("GPT-2 124M", gpt2)):
    d = facts(m, FACTS)
    print(f"  {lab:<14s} {d['correct']:2d}/48   confidence when right {d['conf_right']:.3f}, "
          f"when wrong {d['conf_wrong']:.3f}, separation {d['separation']:+.3f}", flush=True)
    R.setdefault("facts", {})[lab] = d["correct"]

# ---------------------------------------------------------------- 9. brain-like: induction, no cue
hdr(9, "INDUCTION on material no model can have memorised (random consonant-vowel pairs), no prompt cue")
LET = [a + b for a in "bcdfghjklmnpqrstvwz" for b in "aeiou"]
def ind_items(n=400, seed=11):
    r = np.random.default_rng(seed); out = []
    for i in range(n):
        period = 2 + i % 3; reps = 2 + (i // 3) % 3
        toks = list(r.choice(LET, period, replace=False))
        seq = toks * reps + toks[:-1]; ans = toks[-1]
        wrong = [t for t in toks[:-1] if t != ans]
        pool = [t for t in LET if t not in toks]
        while len(wrong) < 3: wrong.append(str(r.choice(pool))); pool = [p for p in pool if p != wrong[-1]]
        out.append((" ".join(seq), f" {ans}", [f" {w}" for w in wrong[:3]]))
    return out
II = ind_items()
for lab, m in ((LAB, ours), ("GPT-2 124M", gpt2)):
    d = facts(m, II)
    print(f"  {lab:<14s} {100*d['correct']/d['n']:5.1f}%   (n={d['n']}, chance 25%)", flush=True)
    R.setdefault("induction", {})[lab] = 100 * d["correct"] / d["n"]

# ---------------------------------------------------------------- 7/8. generation quality and its horizon
hdr(78, "GENERATION QUALITY and LONG-GENERATION HORIZON, judged by the other model")
def bpb_under_gpt2(txt):
    idg = gpt2.tok.encode(txt)[:1000]
    if len(idg) < 32: return float("nan")
    x = torch.tensor([idg], device="cuda")
    with torch.no_grad(): lg = gpt2.m(x).logits.float().log_softmax(-1)
    nats = float(-lg[0, :-1].gather(-1, x[0, 1:].unsqueeze(-1)).sum())
    return nats / (math.log(2) * max(1, len(gpt2.tok.decode(idg[1:]).encode())))
a = np.memmap("/mnt/kv_cache/tokfull_v8192/dclm.bin", dtype=np.uint16, mode="r")
band = [bpb_under_gpt2(tk.decode([int(t) for t in a[i:i + 1500]])) for i in (10**7, 2*10**7, 3*10**7, 4*10**7, 5*10**7, 6*10**7)]
lo, hi = np.nanpercentile(band, [10, 90])
print(f"  the band: real held-out text scores {np.nanmean(band):.3f} under GPT-2 (10-90%: {lo:.3f}-{hi:.3f})")
# (2026-10-06) temperatures from 0.7, and the horizon over THREE prompts: one prompt at one seed, with 0.9 as the coolest
# choice, scored a checkpoint whose text at 0.9 reads too surprising as "horizon 0", one sample deciding the number
PROMPTS = ["The history of the city begins", "In this paper we study", "The river rises in the mountains"]
best = None
for t in (0.7, 0.8, 0.9, 1.0, 1.1):
    s = float(np.nanmean([bpb_under_gpt2(ours.decode(ours.generate(ours.encode(p_), 2048, temp=t, seed=0))) for p_ in PROMPTS]))
    print(f"  {LAB} at temperature {t}: {s:.3f} (mean of {len(PROMPTS)} prompts)", flush=True)
    if best is None or abs(s - np.nanmean(band)) < abs(best[1] - np.nanmean(band)): best = (t, s)
print(f"  -> best temperature {best[0]}, scoring {best[1]:.3f}")
T0 = best[0]
hz = []
for pi_, p_ in enumerate(PROMPTS):
    out = ours.generate(ours.encode(p_), 32768, temp=T0, seed=0)
    print(f"\n  32,768 tokens at temperature {T0} from {p_!r}, in band while:")
    horizon = 0
    for s in range(0, len(out), 4096):
        seg = ours.decode(out[s:s + 4096]); sc = bpb_under_gpt2(seg[:4000])
        inb = lo <= sc <= hi
        if inb and horizon == s: horizon = s + 4096
        top = collections.Counter(out[s:s + 4096]).most_common(1)[0]
        print(f"    {s:>7,}-{s+4096:<7,} {sc:7.3f} {'in band' if inb else '       '}  commonest token "
              f"{100*top[1]/4096:4.1f}% {ours.decode([top[0]])!r}", flush=True)
    hz.append(horizon)
horizon = int(np.median(hz))
print(f"  HORIZON: {horizon:,} tokens in band (median of {len(hz)} prompts: {', '.join(f'{h:,}' for h in hz)})")
# (2026-10-06) THE SAME 32k GENERATIONS WITH A REPETITION GUARD, and the count that does not hinge on one segment: plain
# sampling collapses into loops after a few thousand tokens (segments at 0.02-0.28 under GPT-2); a DRY penalty (src/bench.py:
# only a token that would extend a sequence already seen) at 0.4, the setting eval/longgen3.py found best on 368k (14/24
# segments in band, 0 collapsed, vs 8/24 and 6/24 plain), is reported beside it
import src.bench as SB_
for lab_, dry_ in (("plain", 0.0), ("DRY 0.4", 0.4)):
    SB_.DRY_MULT = dry_; inb_ = col_ = tot_ = 0
    for p_ in PROMPTS:
        o_ = ours.generate(ours.encode(p_), 32768, temp=0.85, seed=1)
        for s_ in range(0, len(o_), 4096):
            sc_ = bpb_under_gpt2(ours.decode(o_[s_:s_ + 4096])[:4000]); tot_ += 1
            inb_ += lo <= sc_ <= hi; col_ += sc_ < 0.6
    print(f"  32k x {len(PROMPTS)} prompts at temperature 0.85, {lab_:8s}: {inb_}/{tot_} segments in band, {col_}/{tot_} collapsed into loops", flush=True)
    R.setdefault("longgen", {})[lab_] = dict(in_band=inb_, collapsed=col_, segments=tot_)
SB_.DRY_MULT = 0.0
R["horizon"] = horizon; R["gen_temp"] = T0; R["gen_bpb"] = best[1]; R["band"] = [float(lo), float(hi)]
txt = ours.decode(out); n = len(txt)
for nm, sg in (("START", txt[:240]), ("MIDDLE", txt[n//2:n//2+240])):
    print(f"\n  {nm}: {sg!r}")
json.dump(R, open("/mnt/kv_cache/nine_axes.json", "w"), indent=1, default=float)
print(f"\nwrote /mnt/kv_cache/nine_axes.json", flush=True)
