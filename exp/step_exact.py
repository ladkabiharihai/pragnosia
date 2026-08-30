"""IS step() ACTUALLY EXACT? In exp/gen_debug.py, GREEDY decoding gave DIFFERENT text via the full forward and via
the incremental step():

  full forward : ' the father of of of of of of the of the father of Ruth.'
  step()       : ' the mother of of of of of of the of the father of of the father.'

Greedy is deterministic, so if step() reproduced the full forward exactly these would be identical. They diverge at
position 2, where ' mother' (0.16) and ' father' (0.15) are nearly tied -- consistent with a small numerical
difference flipping the argmax, but it could equally be a real defect in the incremental path.

Earlier I verified "cosine 1.000000", which is far too coarse a check: cosine stays 1.000000 while logits differ by
enough to flip a near-tie. Measure the ACTUAL difference, in fp32 and under bf16 autocast, over many positions.
"""
import sys
sys.path.insert(0, "/opt/code/membrae")
from util import setup_log
setup_log("step_exact")
import numpy as np, torch
import core as C
from data_mix import DataMix
from tokenizers import Tokenizer

DEV = C.DEV
TOK = Tokenizer.from_file("/opt/code/membrae/data/bpe32k.json")
CH = 2048
sd = torch.load("/mnt/kv_cache/pragnosia_data/pragnosia.pt", map_location=DEV)
cfg = C.Config(D=sd['D'], DIN=8, DV=sd.get('DV', sd['D']), NL=sd['NL'], T=CH, res_scale=16 ** -0.5)
m = C.Core(cfg).to(DEV); m.load_state_dict(sd['model']); m.eval()
zs = lambda: [(torch.zeros(1, cfg.FE, cfg.DV, device=DEV), torch.zeros(1, cfg.FE, device=DEV))
              for _ in range(cfg.NL)]

np.random.seed(17); dm = DataMix(ctx=300, verbose=False, only='window2')
ids = dm.batch(1)[0][:200].tolist()


@torch.no_grad()
def compare(use_amp):
    """Run the SAME token sequence through the full forward and through prefill+step, and compare logits."""
    ctx = torch.dtype
    def fwd_full(x):
        if use_amp:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                return m(x).float()
        return m(x).float()
    full = fwd_full(torch.tensor([ids], device=DEV))[0]        # [T, vocab]
    # incremental: prefill the first 64, then step through the rest feeding the TRUE tokens
    npre = 64
    x = torch.tensor([ids[:npre]], device=DEV)
    if use_amp:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            st, pv = m.prefill(x, zs())
    else:
        st, pv = m.prefill(x, zs())
    diffs = []; flips = 0; n = 0
    cur = x[:, -1:]
    for i in range(npre, len(ids)):
        if use_amp:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg, st, pv = m.step(cur, st, pv)
        else:
            lg, st, pv = m.step(cur, st, pv)
        a = lg[0].float(); b = full[i - 1]
        diffs.append((a - b).abs().max().item())
        if int(a.argmax()) != int(b.argmax()): flips += 1
        n += 1
        cur = torch.tensor([[ids[i]]], device=DEV)
    return np.mean(diffs), np.max(diffs), flips, n


for tag, amp in (("fp32 (no autocast)", False), ("bf16 autocast (as deployed)", True)):
    md, xd, fl, n = compare(amp)
    print(f"  {tag:30s} mean |diff| {md:.3e}   max |diff| {xd:.3e}   "
          f"argmax flips {fl}/{n} ({100*fl/n:.1f}%)", flush=True)

print("\n  A correct incremental decode should give ~0 difference in fp32. If fp32 also differs, step() has a real\n"
      "  defect. If only bf16 differs, it is precision and the fix is to compute the head in fp32.", flush=True)

# where exactly does the difference come from? compare the STATE after prefill vs after stepping the same tokens
print("\n  Cross-check: run the SAME tokens two ways and compare the resulting recurrent state.", flush=True)
with torch.no_grad():
    n1 = 128
    st_a, _ = m.prefill(torch.tensor([ids[:n1]], device=DEV), zs())        # all at once
    st_b, pv = m.prefill(torch.tensor([ids[:64]], device=DEV), zs())       # first half, then step the rest
    cur = torch.tensor([[ids[63]]], device=DEV)
    for i in range(64, n1):
        _, st_b, pv = m.step(torch.tensor([[ids[i]]], device=DEV), st_b, pv)
    for li in (0, cfg.NL // 2, cfg.NL - 1):
        dM = (st_a[li][0] - st_b[li][0]).abs().max().item()
        rel = dM / max(1e-9, st_a[li][0].abs().max().item())
        print(f"    layer {li:2d}: max |dM| {dM:.3e}   relative {rel:.3e}", flush=True)
