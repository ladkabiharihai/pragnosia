"""DOES RECALL (src/recall.py) RETRIEVE PAST THE ATTENTION WINDOW, AND WHAT DOES IT COST PREDICTION? (2026-10-05)
  exact   k = 0 must be the trained model: logits with the patch at k = 0 against the unpatched model, a 20k-token prompt
  needle  eval/incontext.py's in-distribution needle (real prose filler, one needle D tokens before the question, greedy
          exact), the same needles and filler for every k (paired); RECALL=0,16,64  NEEDLE_DIST=...  NEEDLE_N=24
  passkey the standard passkey (repeated filler, five digits, 5 depths), RECALL=..., PASSKEY_LENS=...
  stream  loss on long streams of the main mix read window by window (states carried, as the model is used), by position
          in the stream, recall k against k = 0, paired per stream: does recall make ordinary prediction worse?
usage: eval/recall_test.py <ckpt> exact|needle|passkey|stream"""
import os, sys, math, random, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, "/opt/code/membrae")
from src.common import cap_gpu
cap_gpu(float(os.environ.get("CAP", "10")))
import numpy as np, torch, torch.nn.functional as F
from src.bench import Ours
from src.recall import enable, store_bytes
CK, MODE = sys.argv[1], sys.argv[2]
M = Ours(CK, "/mnt/kv_cache/tok_v8192.json"); m = M.m
KS = os.environ.get("RECALL", "0,16,64").split(",")     # "16g": gated (only far keys beating every key in the window)
def on(k): return enable(m, int(k.rstrip("g")), k.endswith("g"))
wiki = np.memmap("/mnt/kv_cache/tokfull_v8192/wikipedia.bin", dtype=np.uint16, mode="r")
win = max(getattr(b, "window", 0) for b in m.mix)
print(f"{CK} (step {M.step})  attention window {win}  context window {M.ctx}  mode {MODE}", flush=True)

if MODE == "exact":
    ids = np.asarray(wiki[10_000_000:10_030_000]); ids = ids[ids != 8191][:20000].tolist()
    a = M.logits_last(ids, 512)
    os.environ['RECALL_REF'] = '1'; enable(m, 0); b = M.logits_last(ids, 512); os.environ.pop('RECALL_REF')
    enable(m, 64); c = M.logits_last(ids, 512)
    print(f"  k=0 vs unpatched: max |logit difference| {float((a - b).abs().max()):.2e}  (fp32 numerics ~1e-4)")
    os.environ["RECALL_REF"] = "1"; enable(m, 64); d_ = M.logits_last(ids, 512); os.environ.pop("RECALL_REF")
    print(f"  k=64 vs unpatched: max {float((a - c).abs().max()):.2e}, mean {float((a - c).abs().mean()):.2e}")
    print(f"  k=64, the model's recall vs the explicit reference: max {float((c - d_).abs().max()):.2e}, mean {float((c - d_).abs().mean()):.2e}")
    y = torch.tensor(ids[-8191:], device="cuda")
    for nm, l_ in (("unpatched", a), ("k=64 model", c), ("k=64 reference", d_)):
        print(f"  last-window loss {nm}: {float(-l_.float().log_softmax(-1)[0, :-1].gather(-1, y.unsqueeze(-1)).mean()):.5f}")
    sys.exit(0)

if MODE == "needle":
    NEED_ = [("The secret code is {V}.", "What is the secret code?"),
             ("The combination for the {N} lock is {V}.", "What is the combination for the {N} lock?"),
             ("{N}'s favourite number is {V}.", "What is {N}'s favourite number?"),
             ("The {N} team meets in room {V}.", "Which room does the {N} team meet in?")]
    NM = ["Alice", "Omar", "Priya", "Lukas", "Mei", "Tomas", "Aisha", "Diego"]
    DIST = [int(v) for v in os.environ.get("NEEDLE_DIST", "256,1024,2048,4096,8192,16384,32768").split(",")]
    N_ = int(os.environ.get("NEEDLE_N", "24"))
    print(f"IN-DISTRIBUTION NEEDLE, one needle D tokens before the question, real prose filler, greedy exact, {N_} per cell")
    print(f"{'distance':>9s}" + "".join(f"{'k=' + str(k):>10s}" for k in KS) + "   (k = far keys each query may recall; 0 = trained)")
    for D in DIST:
        row = f"{D:>9,d}"
        for k in KS:
            on(k); r_ = random.Random(4242 + D); rn_ = np.random.default_rng(4242 + D); ok = 0; shown = False
            for t in range(N_):
                j = r_.randrange(len(NEED_)); n_ = r_.choice(NM); v_ = str(r_.randint(100, 999999))
                st = M.encode(" " + NEED_[j][0].format(N=n_, V=v_) + " ")
                q = M.encode("\n\nQuestion: " + NEED_[j][1].format(N=n_) + " Answer:")
                pre = 1024; o = int(rn_.integers(0, len(wiki) - 3 * (D + pre)))
                f = np.asarray(wiki[o:o + 3 * (D + pre)]); f = f[f != 8191].tolist()
                ids = f[:pre] + st + f[pre:pre + D] + q
                out = M.decode(M.generate(ids, 8, temp=0.0))
                hit = v_ in out.replace(" ", "")[:12]; ok += hit
                if not shown and not hit and k == KS[-1] and D > win:
                    print(f"   (D={D} k={k} miss: wanted {v_}, wrote {out!r})"); shown = True
            row += f"{100 * ok / N_:>9.1f}%"
        print(row + f"   (peak GPU allocated {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB"
              + (", the store in host memory)" if os.environ.get("RECALL_OFFLOAD") else ")"), flush=True)
    sys.exit(0)

if MODE == "passkey":
    FILL = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. "
    LENS = [int(v) for v in os.environ.get("PASSKEY_LENS", "2048,8192,32768").split(",")]
    DEPTHS = (0.0, 0.25, 0.5, 0.75, 1.0); REPS = int(os.environ.get("PASSKEY_REPS", "4"))
    print(f"PASSKEY: five digits in filler, greedy exact, {len(DEPTHS)} depths x {REPS} keys per length")
    print(f"{'tokens':>9s}" + "".join(f"{'k=' + str(k):>10s}" for k in KS))
    fill = M.encode(FILL)
    for L in LENS:
        row = f"{L:>9,d}"
        for k in KS:
            on(k); r_ = random.Random(77 + L); ok = 0
            for dp in DEPTHS:
                for _ in range(REPS):
                    key = r_.randint(10000, 99999)
                    needle = M.encode(f" The pass key is {key}. Remember it. {key} is the pass key. ")
                    q = M.encode(" What is the pass key? The pass key is")
                    body = (fill * (L // len(fill) + 2))[:max(0, L - len(needle) - len(q))]
                    at = int(dp * len(body)); ids = body[:at] + needle + body[at:] + q
                    ok += str(key) in M.decode(M.generate(ids, 8, temp=0.0)).replace(" ", "")[:10]
            row += f"{100 * ok / (len(DEPTHS) * REPS):>9.1f}%"
        print(row, flush=True)
    sys.exit(0)

if MODE == "stream":
    # long streams of the main mix, the way the trials' held-out reads them (window by window, every state carried)
    SRC = os.environ.get("STREAM_SRC", "arxiv,wikipedia,dclm,fineweb-edu").split(",")
    TL = int(os.environ.get("STREAM_T", "32768")); NS = int(os.environ.get("STREAM_N", "8"))
    rng = np.random.default_rng(31337)
    streams = []
    for i in range(NS):
        b = np.memmap(f"/mnt/kv_cache/tokfull_v8192/{SRC[i % len(SRC)]}.bin", dtype=np.uint16, mode="r")
        o = int(rng.integers(0, len(b) - TL - 2)); streams.append(np.asarray(b[o:o + TL + 1]).astype(np.int64))
    edges = [0, 1024, 4096, 8192, 16384, TL]
    def nll(s):
        x = torch.tensor(s[None], device="cuda"); st = None; out = []
        with torch.no_grad():
            for w0 in range(0, TL, M.ctx):
                h, st = m.trunk(x[:, w0:w0 + M.ctx], st, True)
                lp = F.linear(h, m.e.weight).float().log_softmax(-1)
                out.append(-lp[0].gather(-1, x[0, w0 + 1:w0 + M.ctx + 1].unsqueeze(-1)).squeeze(-1))
        return torch.cat(out).cpu().numpy()
    res = {}
    for k in KS:
        on(k); res[k] = np.stack([nll(s) for s in streams])
        print(f"  k={k}: mean loss {res[k].mean():.4f} nats/token over {NS} x {TL:,} tokens", flush=True)
    print(f"loss by position in the stream (nats/token), and k minus k=0 per stream: mean +- 95% interval over {NS} streams")
    for k in KS[1:]:
        cells = []
        for a_, b_ in zip(edges[:-1], edges[1:]):
            dlt = res[k][:, a_:b_].mean(1) - res[KS[0]][:, a_:b_].mean(1)
            cells.append(f"{a_:>6,}-{b_:<6,} {res[KS[0]][:, a_:b_].mean():.4f} {dlt.mean():+.4f}+-{1.96 * dlt.std(ddof=1) / math.sqrt(NS):.4f}")
        print(f"  k={k}: " + " | ".join(cells), flush=True)
    sys.exit(0)
