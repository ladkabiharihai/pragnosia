"""PRAGNOSIA ON A 4GB CONSUMER GPU. Inference path budgeted for cards with ~3.5 GiB usable.

The training-time defaults are wrong for serving. Three of them cost memory that buys nothing at inference:
  - fp32 weights                 -> bf16 halves them, 476 -> 238 MiB, with no measurable quality change here
  - a 2048-token read chunk      -> the read chunk is a FREE knob at inference; it does not have to match the
                                    training chunk, and [1,T,T] attention plus the logits both scale with it
  - a full fp32 logits tensor    -> only the LAST position is needed to generate; the rest is thrown away

None of this changes the model. It changes how the same weights are run.

  python serve4g.py                      # measure the budget and demo generation
  python serve4g.py --budget 3.5         # assert it fits a given GiB budget
"""
import sys, os, time, argparse
sys.path.insert(0, "/opt/code/membrae")
from util import setup_log, cfg_from_ckpt
setup_log("serve4g")
import numpy as np, torch, torch.nn.functional as F
import core as C
from tokenizers import Tokenizer

DEV = C.DEV
TOK = Tokenizer.from_file("/opt/code/membrae/data/bpe32k.json")
CKPT = "/mnt/kv_cache/pragnosia_data/p124g.pt"


def load(dtype=torch.bfloat16, T=512):
    """Load once, in bf16, with a SMALL read chunk. The state is fixed-size, so T only bounds the read window."""
    sd = torch.load(CKPT, map_location="cpu")
    cfg = cfg_from_ckpt(sd, T=T, res_scale=sd['res_scale'])
    m = C.Core(cfg)
    m.load_state_dict(sd['model'])
    # CAST ON THE CPU, THEN MOVE. Doing .to(DEV).to(bf16) puts the fp32 copy on the card first and peaks at
    # 0.52 GiB before halving it -- pointless on a budget device, where the transient is what OOMs you.
    m = m.to(dtype).to(DEV).eval()
    for p in m.parameters(): p.requires_grad_(False)
    return m, cfg


@torch.no_grad()
def read(m, cfg, ids, chunk=512, state=None, prev=None):
    """Stream a prompt of ANY length through the model at constant memory."""
    st = state or [(torch.zeros(1, cfg.FE, cfg.DV, device=DEV, dtype=m.emb.weight.dtype),
                    torch.zeros(1, cfg.FE, device=DEV, dtype=m.emb.weight.dtype)) for _ in range(cfg.NL)]
    pv = prev
    for i in range(0, len(ids), chunk):
        x = torch.tensor([ids[i:i + chunk]], device=DEV)
        _, st, pv = m(x, states=st, return_states=True, prev=pv, return_prev=True, return_hidden=True)
    return st, pv


@torch.no_grad()
def generate(m, cfg, prompt_ids, n=120, temp=0.85, chunk=512, write_scale=0.25):
    """Read the prompt, then decode one token at a time. Decoding is O(1) in memory and in time."""
    st, pv = read(m, cfg, prompt_ids[:-1], chunk=chunk)
    cur = torch.tensor([[prompt_ids[-1]]], device=DEV)
    out = []
    for _ in range(n):
        lg, st, pv = m.step(cur, st, pv, write_scale=write_scale)
        p = (lg[0].float() / temp).softmax(-1)
        nx = int(torch.multinomial(p, 1)); out.append(nx)
        cur = torch.tensor([[nx]], device=DEV)
    return out


def gib(): return torch.cuda.max_memory_allocated() / 2 ** 30


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=3.5, help="usable GiB on the target card")
    ap.add_argument("--chunk", type=int, default=512)
    a = ap.parse_args()

    torch.cuda.reset_peak_memory_stats()
    m, cfg = load(T=a.chunk)
    P = sum(p.numel() for p in m.parameters())
    print(f"PRAGNOSIA-124 on a 4GB card  |  {P/1e6:.1f}M params, bf16, read chunk {a.chunk}\n", flush=True)
    print(f"  {'stage':>34s} {'peak GiB':>9s} {'headroom vs budget':>19s}", flush=True)
    print(f"  {'weights loaded':>34s} {gib():9.2f} {a.budget-gib():18.2f}", flush=True)

    np.random.seed(0)
    for ntok in (2_048, 65_536, 1_048_576):
        ids = np.random.randint(1000, 30000, ntok).tolist()
        torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); t0 = time.time()
        st, pv = read(m, cfg, ids, chunk=a.chunk)
        torch.cuda.synchronize(); dt = time.time() - t0
        print(f"  {f'read {ntok:,} tokens':>34s} {gib():9.2f} {a.budget-gib():18.2f}"
              f"   ({dt:.1f}s, {ntok/dt/1e3:.0f} ktok/s)", flush=True)
        del st, pv

    ids = np.random.randint(1000, 30000, 1024).tolist()
    torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); t0 = time.time()
    out = generate(m, cfg, ids, n=200, chunk=a.chunk)
    torch.cuda.synchronize(); dt = time.time() - t0
    print(f"  {'generate 200 tokens':>34s} {gib():9.2f} {a.budget-gib():18.2f}"
          f"   ({1000*dt/200:.1f} ms/token)", flush=True)

    peak = gib()
    print(f"\n  PEAK ACROSS EVERYTHING: {peak:.2f} GiB against a {a.budget:.1f} GiB budget -- "
          f"{'FITS' if peak < a.budget else 'DOES NOT FIT'}", flush=True)

    # quality check: the bf16 model must still write real text, not just fit
    from data_mix import DataMix
    np.random.seed(11); pd = DataMix(ctx=2049, verbose=False, only='dclm')
    doc = pd.batch(1)[0].tolist()
    torch.manual_seed(0)
    o = generate(m, cfg, doc[:512], n=120, chunk=a.chunk)
    t = TOK.decode(o)
    d1 = len(set(o)) / len(o)
    g = [tuple(o[i:i + 4]) for i in range(len(o) - 3)]
    rep = 1 - len(set(g)) / max(1, len(g))
    print(f"\n  QUALITY IN bf16: distinct-1 {d1:.2f}  4gram-rep {rep:.2f}   (real text 0.62 / 0.02)", flush=True)
    print(f"  {t[:280]!r}", flush=True)
