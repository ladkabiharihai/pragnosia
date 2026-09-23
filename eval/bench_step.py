"""THE throughput harness. usage: bench_step.py <repo root> [mixers...] [--ctx N] [--batch N]
One CUDA graph of forward + chunked loss + backward, replayed per micro-batch, 32k tokens per update, the
fused graphed optimizer. Reports ktok/s and the process's RESERVED memory (the graph pool hides itself from
max_memory_allocated). Run it under the GPU lock: flock <scratchpad>/gpu.lock .venv/bin/python bench_step.py ..."""
import sys, time, warnings; warnings.filterwarnings("ignore")
root = sys.argv.pop(1); sys.path.insert(0, root)
args = sys.argv[1:]
def opt_(name, default):
    if name in args:
        i = args.index(name); v = int(args[i + 1]); del args[i:i + 2]; return v
    return default
T = opt_("--ctx", 2048); B = opt_("--batch", 4); CH = opt_("--chunks", 4); DD = opt_("--d", 704); TOK = 32768; acc = max(1, TOK // (B * T))
import torch
from src.model import LM
from src.optim import Fabric
from src.common import cap_gpu
cap_gpu(30.0)
for mx in args or ("dspin", "aspin", "attn"):
    torch.manual_seed(0); m = LM(d=DD, layers=12, ctx=T, mixer=mx).cuda()
    f = torch.compile(m.trunk); opt = Fabric(m.param_groups(lr=2e-3), lr=2e-3, fused=True, cuda_graph=True)
    xs = torch.randint(0, 8192, (B, T + 1), device="cuda")
    for p in m.parameters(): p.grad = torch.zeros_like(p)
    def micro():
        with torch.autocast("cuda", dtype=torch.bfloat16): h = f(xs[:, :T])
        (m.loss(h, xs[:, 1:], chunks=CH) / acc).backward()
    try:
        s_ = torch.cuda.Stream(); s_.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s_):
            for _ in range(3): micro()
        torch.cuda.current_stream().wait_stream(s_)
        for p in m.parameters(): p.grad.zero_()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g): micro()
        def step():
            for p in m.parameters(): p.grad.zero_()
            for _ in range(acc):
                xs.copy_(torch.randint(0, 8192, (B, T + 1), device="cuda")); g.replay()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        for _ in range(3): step()
        torch.cuda.synchronize(); t0 = time.time()
        for _ in range(8): step()
        torch.cuda.synchronize(); ms = (time.time() - t0) / 8 * 1e3
        print(f"{mx:<7s} d{DD} ctx{T} B{B}: {TOK/ms:5.0f} ktok/s  {ms:6.1f} ms/update  reserved {torch.cuda.memory_reserved()/2**30:.2f} GiB", flush=True)
    except Exception as e:
        print(f"{mx}: {type(e).__name__}: {str(e)[:300]}", flush=True)
    del m, f, opt, g; torch.cuda.empty_cache()
