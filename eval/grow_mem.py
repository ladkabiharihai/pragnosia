"""GOAL 6: INFERENCE AT ANY PARAMETER COUNT WITHIN 4 GB (2026-10-06). The carrier's state per layer does not grow with context,
so a model's inference memory is its weights plus one window's work, at any length. Shown here at growing parameter counts:
the trained checkpoint is grown by the model's own function-preserving operators (widen with zero writers, deepen with
zero-output layers), so every copy computes the same function as the trained model -- this measures MEMORY at size, not the
quality a model of that size would reach if trained. Weights in bf16; bits per byte on the same 256 tokens after 8,192 and
after 1,048,576 tokens of context (an arxiv stream), and the peak GPU memory PyTorch allocated (+~0.5 GiB CUDA context).
usage: eval/grow_mem.py <ckpt>"""
import sys, time, copy, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, "/opt/code/membrae")
from src.common import cap_gpu
cap_gpu(6.0)
import numpy as np, torch, torch.nn.functional as F
from tokenizers import Tokenizer
from src.model import LM
from src.train import loaders
import src.assoc as A; A.FUSED_SCAN = True
tok = Tokenizer.from_file("/mnt/kv_cache/tok_v8192.json")
bins, ks, ps = loaders("/mnt/kv_cache/toklong_v8192", only=None)
N = 1_048_576 + 512; stream = np.asarray(bins["arxiv"][1_000_000:1_000_000 + N], dtype=np.int64)
sd = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
base = LM(**sd["cfg"]); base.load_state_dict(sd["model"]); base.eval(); W = sd["cfg"]["ctx"]

def measure(m, total):
    ids = stream[-(total + 256):]; st = [None] * m.layers
    torch.cuda.reset_peak_memory_stats(); t0 = time.time()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for wi in range((len(ids) + W - 1) // W):
            xw = torch.tensor(ids[wi * W:(wi + 1) * W], device="cuda").unsqueeze(0)
            if xw.shape[1] < 2: break
            h = m.e(xw) + m.p(torch.arange(xw.shape[1], device="cuda"))
            for i, (mx, mp) in enumerate(zip(m.mix, m.mlp)): h, st[i] = mx(h, st[i]); h = mp(h)
            last = h
        lg = F.linear(m.n(last)[0, -256:], m.e.weight).float()[:-1]
        nats = float(F.cross_entropy(lg, torch.tensor(ids[-255:], device="cuda"), reduction="sum"))
    nb = len(tok.decode([int(t) for t in ids[-255:]]).encode("utf-8"))
    return nats / nb / np.log(2), torch.cuda.max_memory_allocated() / 2**30, time.time() - t0

print(f"{'model':<34s} {'params':>8s}  {'bpb @8k':>8s} {'bpb @1M':>8s}  {'peak GiB @1M':>12s}  {'ingest 1M':>9s}")
for lab, d_new, deepen in (("trained (d768, 12 layers)", None, 0), ("widened to d1024", 1024, 0), ("d1024, 24 layers", 1024, 1),
                           ("d1536, 24 layers", 1536, 1), ("d2048, 24 layers", 2048, 1), ("d2048, 48 layers", 2048, 2)):
    m = copy.deepcopy(base)
    if d_new: m.grow(d_new)
    for _ in range(deepen): m = m.deepen()
    m = m.to(torch.bfloat16).cuda().eval()
    for mx in m.mix:
        if hasattr(mx, "need_sent"): mx.need_sent = False
    P = sum(p.numel() for p in m.parameters())
    try:
        b8, _, _ = measure(m, 8192); b1, pk, tt = measure(m, 1_048_576)
        print(f"{lab:<34s} {P / 1e6:7.0f}M  {b8:8.4f} {b1:8.4f}  {pk:12.2f}  {tt:8.0f}s", flush=True)
    except torch.OutOfMemoryError:
        print(f"{lab:<34s} {P / 1e6:7.0f}M  does not fit under the 6 GiB cap", flush=True)
    del m; torch.cuda.empty_cache()
