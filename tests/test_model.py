"""Verify in-place growth (vision goal 7) before a run depends on it. Run: .venv/bin/python tests/test_model.py

The claims in LM.grow's docstring, each measured: the function is preserved (new stream dimensions exactly zero,
loss unchanged to a few thousandths), the new capacity receives gradient on the first step (no zero-gradient
fixed point), the optimizer's state follows the parameters, and training simply continues.
"""
import sys, warnings; warnings.filterwarnings("ignore")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch, torch.nn.functional as F
from src.model import LM
from src.optim import Fabric

ok = fail = 0
def check(name, cond, detail=""):
    global ok, fail
    if cond: ok += 1; print(f"  PASS  {name}" + (f"   {detail}" if detail else ""))
    else:    fail += 1; print(f"  FAIL  {name}" + (f"   {detail}" if detail else ""))

dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
V, D0, D1 = 512, 64, 96
m = LM(vocab=V, d=D0, layers=3, ctx=32).to(dev)
x = torch.randint(0, V, (4, 33), device=dev)
opt = Fabric(m.param_groups(lr=3e-3), lr=3e-3)
def loss(): return F.cross_entropy(m(x[:, :-1]).reshape(-1, V), x[:, 1:].reshape(-1))
for _ in range(20):
    opt.zero_grad(); loss().backward(); opt.step()

print("growth preserves the function")
with torch.no_grad(): lb = loss().item(); n_before = sum(p.numel() for p in m.parameters())
grown = m.grow(D1)
for p, (old_p, sh) in grown.items(): opt.grew(p, sh, replaces=old_p)
with torch.no_grad():
    la = loss().item()
    h = m.e(x[:, :-1]) + m.p(torch.arange(32, device=dev))
    for mx, mp in zip(m.mix, m.mlp): h = mp(mx(h)[0])
n_after = sum(p.numel() for p in m.parameters())
check(f"widened {D0} -> {D1}: {n_before} -> {n_after} params", n_after > n_before and m.d == D1)
check("new residual dimensions are exactly zero through every layer", h[:, :, D0:].abs().max().item() == 0.0)
check("loss unchanged to 1e-2 (LayerNorm's mean shift is the residual)", abs(la - lb) < 1e-2, f"{lb:.4f} -> {la:.4f}")
check("no parameter object of the old width remains", all(p.shape[-1] != D0 or p.dim() == 0 for p in m.parameters()
      if p.dim() and p.shape[-1] in (D0, D1)))

print("\nnew capacity learns from the first step")
opt.zero_grad(); loss().backward()
writers = {n: p for n, p in m.named_parameters() if n in ("e.weight", "p.weight") or n.endswith("C.weight") or n.endswith("b.weight")}
zero = []
for n, p in writers.items():
    g = p.grad[:, D0:] if n in ("e.weight", "p.weight") else p.grad[D0:]
    if g.abs().max().item() == 0.0: zero.append(n)
check("every writer's new rows/columns has non-zero gradient", not zero, f"zero: {zero}" if zero else "")
readers_new_cols = [p.grad[:, D0:].abs().max().item() for n, p in m.named_parameters() if n.endswith("B.weight") or n.endswith("a.weight")]
check("readers' new columns have zero gradient on step one (their input is exactly zero), as designed",
      max(readers_new_cols) == 0.0)
opt.step(); opt.zero_grad(); loss().backward()
readers_new_cols = [p.grad[:, D0:].abs().max().item() for n, p in m.named_parameters() if n.endswith("B.weight") or n.endswith("a.weight")]
check("and non-zero on step two (the writers moved)", min(readers_new_cols) > 0.0)

print("\noptimizer state follows the parameters")
shapes_ok = True
for gp in opt.param_groups:
    for p in gp["params"]:
        st = opt.state[p]
        if "vr" in st: shapes_ok &= st["vr"].numel() == p.shape[0] and st["vc"].numel() == p.shape[1]
        if "v" in st: shapes_ok &= st["v"].shape == p.shape
        if "m" in st: shapes_ok &= st["m"].shape == p.shape and st["m"].dtype == torch.bfloat16
check("every state tensor matches its parameter's new shape", shapes_ok)
check("no state is keyed by a discarded parameter", all(any(p is q for gp in opt.param_groups for q in gp["params"]) for p in opt.state))
ids = [id(p) for gp in opt.param_groups for p in gp["params"]]
check("param groups hold exactly the model's parameters, once each", sorted(ids) == sorted(id(p) for p in m.parameters()))

print("\ntraining continues")
l0 = loss().item()
for _ in range(40):
    opt.zero_grad(); l = loss(); l.backward(); opt.step()
check("loss falls over 40 steps after growth", l.item() < l0 - 0.1 and torch.isfinite(l).item(), f"{l0:.3f} -> {l.item():.3f}")
sd = m.state_dict(); m2 = LM(vocab=V, d=D1, layers=3, ctx=32).to(dev); m2.load_state_dict(sd)
with torch.no_grad(): same = torch.allclose(m2(x[:, :-1]), m(x[:, :-1]), atol=1e-5)
check("state_dict loads into a fresh model of the new width and computes the same thing", same)
check("forward_carried works at the new width", tuple(m.forward_carried(torch.randint(0, V, (2, 64), device=dev)).shape) == (2, 32, V))

print(f"\n{ok} passed, {fail} failed")
sys.exit(1 if fail else 0)
