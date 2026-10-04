"""ONE MILLION TOKENS of context, on a model trained at 8,192 (vision goal 9).

The state is fixed size, so the only question is whether quality survives 122 window boundaries and whether
memory and time stay flat. Loss is measured on the SAME final 256 tokens at each total context, so the
comparison is like for like, and the model then generates from the one-million-token state.
"""
import sys, time, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, "/opt/code/membrae")
import numpy as np, torch, torch.nn.functional as F
from tokenizers import Tokenizer
from src.common import cap_gpu
from src.train import loaders
from src.model import LM
cap_gpu(20.0)
tok = Tokenizer.from_file("/mnt/kv_cache/tok_v8192.json")
bins, ks, ps = loaders("/mnt/kv_cache/toklong_v8192", only=None)
a = bins["arxiv"]
N = 1_048_576 + 512
j = 1_000_000
stream = np.asarray(a[j:j + N], dtype=np.int64)          # one continuous slice, so context is real context

# The checkpoint comes from the command line. It used to be hard-coded to run35_mixlen, so every "one
# million tokens on the final model" measurement taken through this file before 2026-09-23 08:00 UTC was in
# fact run35 again; the architecture is the same, so the flat memory carries over, but the numbers did not.
CK = sys.argv[1] if len(sys.argv) > 1 else "/mnt/kv_cache/run45_prose/ckpt.pt"
print(f"checkpoint {CK}", flush=True)
sd = torch.load(CK, map_location="cuda")
m = LM(**sd["cfg"]).cuda(); m.load_state_dict(sd["model"]); m.eval()
W = sd["cfg"]["ctx"]
print(f"model ctx {W}, state per layer {m.mix[0].h if hasattr(m.mix[0],'h') else '?'} heads", flush=True)

TARGETS = (8192, 32768, 131072, 524288, 1048576)
# POSITIONS PAST THE TRAINING WINDOW (2026-10-01). The learned position table has ctx entries and restarts at every
# window, so from the second window on, position 0 arrives with a FULL state -- a pairing training never shows (there
# position 0 always means an empty state). The same 256 tokens scored 1.1195 bits/byte after 8,192 tokens of context
# and 1.1430 after 32,768: more context, worse prediction. MILLION_POS chooses what windows after the first are given:
#   wrap  (default, as trained and as everything else here does): 0 .. W-1 again
#   clamp: W-1 for every token (always "late in a window", which a full state is)
#   half : the second half's positions W/2 .. W-1, twice (late, and local order kept within each half)
import os
POSMODE = os.environ.get("MILLION_POS", "wrap")
if len(sys.argv) > 2: TARGETS = tuple(int(t) for t in sys.argv[2].split(","))
def positions(wi, n):
    if wi == 0 or POSMODE == "wrap": return torch.arange(n, device="cuda")
    if POSMODE == "clamp": return torch.full((n,), W - 1, device="cuda")
    return W // 2 + torch.arange(n, device="cuda") % (W // 2)
print(f"positions after the first window: {POSMODE}", flush=True)
if os.environ.get("MILLION_CARRIER_WRAP"):                 # carrier rotation positions modulo this (assoc.py pos_wrap)
    for mx in m.mix:
        if hasattr(mx, "theta"): mx.pos_wrap = int(os.environ["MILLION_CARRIER_WRAP"])
    print(f"carrier rotation positions wrapped at {os.environ['MILLION_CARRIER_WRAP']}", flush=True)
score = stream[-256:]
torch.cuda.reset_peak_memory_stats()
def measure(total):
    ids = stream[-(total + 256):]
    st = [None] * m.layers; t0 = time.time()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        nwin = (len(ids) + W - 1) // W
        for wi in range(nwin):
            xw = torch.tensor(ids[wi * W:(wi + 1) * W], device="cuda").unsqueeze(0)
            if xw.shape[1] < 2: break
            h = m.e(xw) + m.p(positions(wi, xw.shape[1]))
            for i, (mx, mp) in enumerate(zip(m.mix, m.mlp)):
                h, st[i] = mx(h, st[i]); h = mp(h)
            last = h
        lg = F.linear(m.n(last)[0, -256:], m.e.weight).float()[:-1]   # h[k] predicts token k+1
        y = torch.tensor(ids[-255:], device="cuda")
        nats = float(F.cross_entropy(lg, y, reduction="sum"))
    nbytes = len(tok.decode([int(t) for t in ids[-255:]]).encode("utf-8"))
    print(f"  {total:>9,} tokens of context: {nats/nbytes/np.log(2):.4f} bits/byte   "
          f"{time.time()-t0:6.1f} s to ingest   peak {torch.cuda.max_memory_allocated()/2**30:.2f} GiB", flush=True)

# For a full-attention model the cache grows with every token it reads, so it runs out of memory somewhere
# on the way to a million. WHERE it runs out is the result, and is printed rather than hidden.
# MANY PASSAGES (2026-10-02): the 8k-vs-32k comparison rested on ONE 256-token passage. MILLION_SAMPLES=K scores K
# passages ending at evenly spaced points of the stream, each after every context length, paired, with a standard error.
if int(os.environ.get("MILLION_SAMPLES", "0")):
    K = int(os.environ["MILLION_SAMPLES"]); res = {t: [] for t in TARGETS}
    for kk in range(K):
        end = N - kk * (N - max(TARGETS) - 256) // max(1, K)
        for total in TARGETS:
            ids = stream[end - (total + 256):end]; st = [None] * m.layers
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                for wi in range((len(ids) + W - 1) // W):
                    xw = torch.tensor(ids[wi * W:(wi + 1) * W], device="cuda").unsqueeze(0)
                    if xw.shape[1] < 2: break
                    h = m.e(xw) + m.p(positions(wi, xw.shape[1]))
                    for i, (mx, mp) in enumerate(zip(m.mix, m.mlp)): h, st[i] = mx(h, st[i]); h = mp(h)
                    last = h
                lg = F.linear(m.n(last)[0, -256:], m.e.weight).float()[:-1]
                nats = float(F.cross_entropy(lg, torch.tensor(ids[-255:], device="cuda"), reduction="sum"))
            nb = len(tok.decode([int(t) for t in ids[-255:]]).encode("utf-8"))
            res[total].append(nats / nb / np.log(2))
    base = np.array(res[TARGETS[0]])
    for total in TARGETS:
        v = np.array(res[total]); dlt = v - base
        print(f"  {total:>9,} tokens of context: {v.mean():.4f} bits/byte over {K} passages; vs {TARGETS[0]:,}: "
              f"{dlt.mean():+.4f} +- {1.96 * dlt.std(ddof=1) / np.sqrt(K) if K > 1 else 0:.4f}", flush=True)
    sys.exit(0)
ran_out = False
for total in TARGETS:
    try:
        measure(total)
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        print(f"  {total:>9,} tokens of context: OUT OF MEMORY at a 20 GiB cap -- this model's cache grows with every token", flush=True)
        ran_out = True
        break
if ran_out or len(sys.argv) > 2:
    sys.exit(0)

ids = stream[-(1_048_576 + 256):]
st = [None] * m.layers
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    for wi in range((len(ids) + W - 1) // W):
        xw = torch.tensor(ids[wi * W:(wi + 1) * W], device="cuda").unsqueeze(0)
        if xw.shape[1] < 2: break
        h = m.e(xw) + m.p(torch.arange(xw.shape[1], device="cuda"))
        for i, (mx, mp) in enumerate(zip(m.mix, m.mlp)):
            h, st[i] = mx(h, st[i]); h = mp(h)
    nxt = int(torch.argmax(F.linear(m.n(h), m.e.weight)[0, -1]))
    out = [nxt]
    for _ in range(80):
        x1 = torch.tensor([[nxt]], device="cuda")
        h1 = m.e(x1) + m.p(torch.tensor([0], device="cuda"))
        for i, (mx, mp) in enumerate(zip(m.mix, m.mlp)):
            h1, st[i] = mx(h1, st[i]); h1 = mp(h1)
        nxt = int(torch.argmax(F.linear(m.n(h1), m.e.weight)[0, -1])); out.append(nxt)
print(f"\n  generated after 1,048,576 tokens of context:\n  {tok.decode(out)[:400]!r}", flush=True)
