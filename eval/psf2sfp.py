"""Convert a checkpoint trained with the prodigy-plus-schedule-free package (--optim psf) into one the fused optimizer
(--optim sfp) resumes exactly: the same z, second moments (the package's row/col naming mapped to Fabric's vr/vc, tall
matrices included), Prodigy p0/s, group step size d, numerator and schedule-free weight sum, plus a per-weight clock
equal to the group's sum. The parameters in a checkpoint are the averaged weights x, as both optimizers save them.
usage: eval/psf2sfp.py <psf ckpt> <out ckpt>"""
import sys, torch
sys.path.insert(0, "/opt/code/membrae")
from src.model import LM
from src.optim import Fabric
src, dst = sys.argv[1], sys.argv[2]
sd = torch.load(src, map_location="cpu", weights_only=False)
m = LM(**sd["cfg"]); m.load_state_dict(sd["model"])
if sd.get("new_layers"): m.new_layers = list(sd["new_layers"])
groups = [{"params": g["params"]} for g in m.param_groups(lr=1.0)]
o = Fabric(groups, lr=1.0, sfp=True)
po = sd["opt"]; pstate = po["state"]; pgroups = po["param_groups"]
idx = 0
for gi, g in enumerate(o.param_groups):
    pg = pgroups[gi]
    g["sf_saved"] = {"d": float(pg["d"]), "dnum": float(pg["d_numerator"]), "ws": float(pg.get("weight_sum", 0.0))}
    g["train_mode"] = False                                   # the saved parameters are x
    assert len(pg["params"]) == len(g["params"]), (gi, len(pg["params"]), len(g["params"]))
    for p, key in zip(g["params"], pg["params"]):          # the package's state is keyed by parameter index
        st = pstate.get(key)
        if not st: continue
        ns = {"step": 1, "z": st["z"].float().clone(), "p0": st["p0"].clone(), "s": st["s"].clone(),
              "wsp": torch.full(p.shape, float(pg.get("weight_sum", 0.0)))}
        if "exp_avg_sq_row" in st:
            dr, dc = st["exp_avg_sq_metadata"]
            r, c = st["exp_avg_sq_row"].float(), st["exp_avg_sq_col"].float()   # row: reduced over dim dr; col: over dc
            per_row = c if dr == 0 else r                          # the statistic that varies along dim 0
            per_col = r if dr == 0 else c
            ns["vr"] = per_row.reshape(-1).clone(); ns["vc"] = per_col.reshape(-1).clone()
        else:
            ns["v"] = st["exp_avg_sq"].float().clone()
        o.state[p] = ns
out = dict(sd); out["opt"] = o.state_dict()
torch.save(out, dst)
print(f"converted {sum(1 for _ in o.state)} tensors; groups d = {[round(g['sf_saved']['d'], 8) for g in o.param_groups]}, ws = {[round(g['sf_saved']['ws'], 6) for g in o.param_groups]}")
