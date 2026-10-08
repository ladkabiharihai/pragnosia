"""Write a DEPTH-DOUBLED checkpoint from a trained one (vision goal 7), and prove it computes the same thing.
The new layers start at exactly zero output (see LM.deepen); the checkpoint records which they are so the
trainer gives them their own absolute-step optimizer group, and carries no optimizer state, so the step count
restarts and the schedule re-warms. usage: eval/deepen.py <in ckpt> <out ckpt> [memory]

memory (2026-10-05): instead of doubling the depth, ADD ONE CARRIER LAYER ON TOP built to hold things for long and to WRITE
ONLY WHAT IT LEARNS TO: every head at the long end of the decay range (half-life ~7,000 tokens) and its writes CLOSED by default
(write strength sigmoid(-6) = 0.25%), output zero (exact). Measured 2026-10-04: the existing long heads are overwritten by
filler within ~1-2k tokens; forced to write only on a needle they retrieve it at 4-8k with no retraining. Closed by default
makes selective writing the starting point instead of something training must discover against the language-model loss.
The step count is kept (the trainer counts on from the source step); no optimizer state (a cold start, as for deepen)."""
import sys, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, "/opt/code/membrae")
from src.common import cap_gpu
cap_gpu(12.0)
import numpy as np, torch, torch.nn.functional as F
import src.assoc as A
from src.model import LM
from src.train import loaders, batch
src_ck, out = sys.argv[1], sys.argv[2]
sd = torch.load(src_ck, map_location="cuda", weights_only=False)
m = LM(**sd["cfg"]).cuda(); m.load_state_dict(sd["model"]); m.eval()
MEM = len(sys.argv) > 3 and sys.argv[3] == "memory"
if MEM:
    import math
    big = m.add_layers(1).eval()
    nm = big.mix[big.new_layers[0]]
    assert hasattr(nm, "theta"), "the added layer must be a carrier"
    with torch.no_grad():
        nm.A.fill_(math.log(1e-4 / 0.69))           # every head at the long end of the default decay range
        nm.wb.bias.fill_(-6.0)                        # writes closed by default: beta = sigmoid(-6 + small) ~ 0.25%
else:
    big = m.deepen().eval()
A.FUSED_SCAN = True                                          # state read off, as it will train
for net in (m, big):
    for mx in net.mix:
        if hasattr(mx, "need_sent"): mx.need_sent = False
bins, ks, ps = loaders("/mnt/kv_cache/tokfull_v8192")
x = batch(bins, ks, ps, np.random.default_rng(5), 2, 2048)
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    la = F.cross_entropy(m(x[:, :2048]).float().reshape(-1, 8192), x[:, 1:2049].reshape(-1)).item()
    lb = F.cross_entropy(big(x[:, :2048]).float().reshape(-1, 8192), x[:, 1:2049].reshape(-1)).item()
cfg = dict(sd["cfg"]); cfg["layers"] = big.layers
torch.save({"cfg": cfg, "model": big.state_dict(), "step": sd["step"] if MEM else 0, "loss": sd.get("loss", 0.0),
            "new_layers": big.new_layers, "deepened_from": src_ck, "memory_layer": MEM}, out)
print(f"{m.layers} -> {big.layers} layers, {sum(p.numel() for p in m.parameters())/1e6:.1f}M -> "
      f"{sum(p.numel() for p in big.parameters())/1e6:.1f}M params; loss on real text {la:.6f} -> {lb:.6f} "
      f"({lb-la:+.2e}) with the state read off; wrote {out}")
