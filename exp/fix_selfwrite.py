"""FIX THE SELF-WRITE COLLAPSE, AND VERIFY IT BEFORE SPENDING COMPUTE ON A FULL RUN.

THE DEFECT (measured): long generation degenerates after ~600 tokens -- 4-gram repetition 0.07 -> 0.48 -- and
freezing the recurrent state removes it entirely (0.01 for the full 1200). So the model is not incoherent; it is
poisoned by its own output accumulating in the state. The learned forget half-life is ~2000 tokens, so self-writes
barely decay before they dominate the read.

ROOT CAUSE: during training the state only ever receives REAL text. The model has never been given a reason to
discount its own writes, because it has never seen them.

THE FIX -- SCHEDULED SAMPLING ON THE STATE. For a fraction of positions, feed the model its OWN sampled token
instead of the true one, while still supervising against the TRUE continuation. The wrong tokens get written into
the state exactly as they would at generation time, so the model gets gradient pressure to recover from -- and
discount -- unreliable state content.

CONTROL: an identical run with scheduled sampling OFF, same steps, same data, same seed. Without it, any
improvement could simply be "more training". The control is what makes this a test rather than a hope.
"""
import sys, time, copy
sys.path.insert(0, "/opt/code/membrae")
from util import setup_log, cfg_from_ckpt
setup_log("fix_selfwrite")
import numpy as np, torch, torch.nn.functional as F
import core as C
from data_mix import DataMix
from tokenizers import Tokenizer
import torch._dynamo as _dynamo; _dynamo.config.recompile_limit = 64
import torch._inductor.config as _ic; _ic.triton.prefer_nd_tiling = True

DEV = C.DEV
TOK = Tokenizer.from_file("/opt/code/membrae/data/bpe32k.json")
V = 32768; CH = 4096; DOC = 32768; NCH = DOC // CH; B = 4
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 400
CK = "/tmp/claude-1000/-opt-code-membrae/ab0dd831-724a-4bd0-be9e-dd7e1ea6f491/scratchpad/pinned.pt"


def load():
    sd = torch.load(CK, map_location=DEV)
    cfg = cfg_from_ckpt(sd, T=CH)
    m = C.Core(cfg).to(DEV); m.load_state_dict(sd['model'])
    return m, cfg, sd['step']


@torch.no_grad()
def sample_from_hidden(m, h, nsp=8, temp=1.0):
    """Sample the model's OWN next-token guess at every position, projecting to vocab in slices so the
    [B,T,vocab] tensor is never built (the same trick ce_from_hidden uses)."""
    sc = getattr(m.cfg, 'head_scale', 1.0)
    hf = h.reshape(-1, h.shape[-1]); n = hf.shape[0]
    out = torch.empty(n, dtype=torch.long, device=h.device)
    for i in range(nsp):
        sl = slice(i * n // nsp, (i + 1) * n // nsp)
        p = ((m.head(hf[sl]) * sc).float() / temp).softmax(-1)
        out[sl] = torch.multinomial(p, 1).squeeze(-1)
    return out.view(h.shape[0], h.shape[1])


def finetune(m, steps, sched_p, seed=1, lr=1e-4):
    """sched_p = fraction of positions replaced by the model's own sample. 0.0 is the control."""
    torch.manual_seed(seed); np.random.seed(seed)
    dm = DataMix(ctx=DOC + 1, verbose=False, only='window2')
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    t0 = time.time()
    for it in range(steps):
        Xd = torch.tensor(dm.batch(B), device=DEV)
        st = None; pv = None; opt.zero_grad()
        for i in range(NCH):
            xi = Xd[:, i * CH:(i + 1) * CH]; yi = Xd[:, i * CH + 1:(i + 1) * CH + 1]
            xin = xi
            if sched_p > 0:
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    h0, _, _ = m(xi, states=st, return_states=True, prev=pv, return_prev=True, return_hidden=True)
                    samp = sample_from_hidden(m, h0)
                keep = torch.rand(xi.shape, device=DEV) >= sched_p
                xin = xi.clone()
                # position t receives the model's own guess made AT t-1, exactly as at generation time
                xin[:, 1:] = torch.where(keep[:, 1:], xi[:, 1:], samp[:, :-1])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                h, st, pv = m(xin, states=st, return_states=True, prev=pv, return_prev=True, return_hidden=True)
                l = C.ce_from_hidden(m, h, yi) / NCH        # supervised against the TRUE continuation
            st = [(a.detach(), b.detach()) for a, b in st]; pv = [q.detach() for q in pv]
            l.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    return time.time() - t0


def stats(t):
    d1 = len(set(t)) / max(1, len(t))
    g = [tuple(t[i:i + 4]) for i in range(len(t) - 3)]
    return d1, 1 - len(set(g)) / max(1, len(g))


def letters(txt):
    c = [x for x in txt if not x.isspace()]
    return sum(x.isalpha() for x in c) / max(1, len(c))


def stability(m, docs, n=1200):
    """windowed 4-gram repetition over a long generation, WITH self-writes on (the deployed setting)"""
    m.eval(); allw = []; d1s = []; lts = []
    for di, doc in enumerate(docs):
        torch.manual_seed(di)
        o, _ = m.generate(doc[:1024], n, temp=0.85, write_scale=1.0)
        allw.append([stats(o[a:a + 200])[1] for a in range(0, n, 200)])
        d1s.append(stats(o)[0]); lts.append(letters(TOK.decode(o)))
    m.train()
    return np.mean(allw, axis=0), np.mean(d1s), np.mean(lts)


if __name__ == "__main__":
    np.random.seed(11); dmv = DataMix(ctx=CH + 1, verbose=False, only='window2')
    docs = [dmv.batch(1)[0].tolist() for _ in range(3)]
    np.random.seed(9)
    refs = [dmv.batch(1)[0][:1200].tolist() for _ in range(12)]
    rw = [np.mean([stats(r[a:a + 200])[1] for r in refs]) for a in range(0, 1200, 200)]
    rd1 = np.mean([stats(r)[0] for r in refs]); rlt = np.mean([letters(TOK.decode(r)) for r in refs])

    m, cfg, step = load()
    print(f"SUBJECT: the pinned {sum(p.numel() for p in m.parameters())/1e6:.0f}M checkpoint at step {step}\n"
          f"Fine-tuning {STEPS} steps each way. The CONTROL isolates the fix from simply training longer.\n", flush=True)
    hdr = " ".join(f"{f'{a}-{a+200}':>9s}" for a in range(0, 1200, 200))
    print(f"  {'arm':>26s} {hdr} {'d1':>6s} {'letters':>8s}", flush=True)
    print(f"  {'REAL TEXT':>26s} " + " ".join(f"{r:9.2f}" for r in rw) + f" {rd1:6.2f} {rlt:8.2f}", flush=True)

    w, d1, lt = stability(m, docs)
    print(f"  {'before (the defect)':>26s} " + " ".join(f"{r:9.2f}" for r in w) + f" {d1:6.2f} {lt:8.2f}", flush=True)
    del m; torch.cuda.empty_cache()

    for tag, p in (("CONTROL: same steps, p=0", 0.0), ("FIX: scheduled sampling p=.25", 0.25)):
        m, cfg, _ = load()
        dt = finetune(m, STEPS, p)
        w, d1, lt = stability(m, docs)
        print(f"  {tag:>26s} " + " ".join(f"{r:9.2f}" for r in w) + f" {d1:6.2f} {lt:8.2f}"
              f"   ({dt/60:.1f} min)", flush=True)
        if p > 0:
            torch.manual_seed(0)
            o, _ = m.generate(docs[0][:1024], 1200, temp=0.85, write_scale=1.0)
            print(f"\n  FIX, last 260 chars: {TOK.decode(o)[-260:]!r}", flush=True)
            with torch.no_grad():
                print(f"  gate half-lives now: "
                      f"{[int(np.log(0.5)/np.log(min(0.9999999,torch.sigmoid(L.Wg.bias).item()))) for L in m.layers]}",
                      flush=True)
        del m; torch.cuda.empty_cache()
