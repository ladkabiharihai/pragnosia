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
score = stream[-256:]
torch.cuda.reset_peak_memory_stats()
for total in TARGETS:
    ids = stream[-(total + 256):]
    st = [None] * m.layers; t0 = time.time()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        nwin = (len(ids) + W - 1) // W
        for wi in range(nwin):
            xw = torch.tensor(ids[wi * W:(wi + 1) * W], device="cuda").unsqueeze(0)
            if xw.shape[1] < 2: break
            h = m.e(xw) + m.p(torch.arange(xw.shape[1], device="cuda"))
            for i, (mx, mp) in enumerate(zip(m.mix, m.mlp)):
                h, st[i] = mx(h, st[i]); h = mp(h)
            last = h
        lg = F.linear(m.n(last)[0, -256:], m.e.weight).float()[:-1]   # h[k] predicts token k+1
        y = torch.tensor(ids[-255:], device="cuda")
        nats = float(F.cross_entropy(lg, y, reduction="sum"))
    nbytes = len(tok.decode([int(t) for t in ids[-255:]]).encode("utf-8"))
    dt = time.time() - t0
    print(f"  {total:>9,} tokens of context: {nats/nbytes/np.log(2):.4f} bits/byte   "
          f"{dt:6.1f} s to ingest   peak {torch.cuda.max_memory_allocated()/2**30:.2f} GiB", flush=True)

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
