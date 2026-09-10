"""Verify Fabric before any training depends on it. Run: .venv/bin/python tests/test_optim.py

CONSTRAINTS.md rule 2: prove it before building on it. An optimizer that trains something is not evidence it
is correct; these check the specific claims made in its docstring, each against a case where the claim should
hold and, where possible, a case where it should not.
"""
import sys, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch
from src.optim import Fabric

ok = fail = 0
def check(name, cond, detail=""):
    global ok, fail
    if cond: ok += 1; print(f"  PASS  {name}" + (f"   {detail}" if detail else ""))
    else:    fail += 1; print(f"  FAIL  {name}" + (f"   {detail}" if detail else ""))

dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)

print("it actually optimises")
# least squares with a known solution, so 'it went down' is not the bar; it has to reach the answer
A = torch.randn(256, 64, device=dev); xstar = torch.randn(64, 16, device=dev); B = A @ xstar
x = torch.zeros(64, 16, device=dev, requires_grad=True)
opt = Fabric([x], lr=2e-2)
for _ in range(2000):
    opt.zero_grad(); ((A @ x - B) ** 2).mean().backward(); opt.step()
err = ((x - xstar).norm() / xstar.norm()).item()
check("converges on least squares", err < 0.02, f"relative error {err:.4f}")
# From a zero start the trust ratio moves lr*|w| and |w| is 0, so early progress is governed by min_norm.
# That is a real property of scale-free stepping, not a bug, and it is why this needs more steps than Adam.

print("\nfactored second moment costs O(R+C), not O(R*C)")
w = torch.randn(4096, 4096, device=dev, requires_grad=True); w.grad = torch.randn_like(w)
f = Fabric([w], factored=True, momentum=False); f.step()
d = Fabric([w], factored=False, momentum=False); d.step()
fb, db = f.state_bytes(), d.state_bytes()
check("factored state is far smaller", fb < db / 100, f"{fb/1e6:.3f} MB vs dense {db/1e6:.1f} MB")
check("factored state is O(R+C)", abs(fb - (4096+4096)*4) < 4096, f"{fb} bytes vs expected {(4096+4096)*4}")

print("\nbytes per parameter against the 4GB budget")
for name, kw in [("Fabric factored+mom", dict(factored=True, momentum=True)),
                 ("Fabric factored only", dict(factored=True, momentum=False)),
                 ("Fabric dense (Adam-like)", dict(factored=False, momentum=True))]:
    p = torch.randn(2048, 2048, device=dev, requires_grad=True); p.grad = torch.randn_like(p)
    o = Fabric([p], **kw); o.step()
    per = o.state_bytes() / p.numel()
    print(f"    {name:<26s} {per:>5.2f} B/param state -> {4e9/(4+per)/1e6:>6.0f}M params in 4GB")
p = torch.randn(2048,2048,device=dev,requires_grad=True); p.grad=torch.randn_like(p)
o = Fabric([p], factored=True, momentum=True); o.step()
check("factored+momentum beats Adam's 8 B/param", o.state_bytes()/p.numel() < 8.0,
      f"{o.state_bytes()/p.numel():.2f} B/param")

print("\ntrust ratio makes the step scale-free")
sizes = {}
for scale in (0.01, 1.0, 100.0):
    p = (torch.randn(512, 512, device=dev) * scale).requires_grad_(True)
    p.grad = torch.randn_like(p) * scale
    before = p.detach().clone()
    Fabric([p], lr=1e-2, trust=True).step()
    sizes[scale] = ((p - before).norm() / before.norm()).item()      # RELATIVE step
rel = list(sizes.values())
check("relative step is invariant to weight scale", max(rel)/min(rel) < 1.2,
      "  ".join(f"scale {k}: {v:.4f}" for k, v in sizes.items()))
p = (torch.randn(512,512,device=dev)*100).requires_grad_(True); p.grad=torch.randn_like(p)*100
before=p.detach().clone(); Fabric([p], lr=1e-2, trust=False).step()
notrust = ((p-before).norm()/before.norm()).item()
check("without trust it is NOT invariant", abs(notrust - rel[1]) > 1e-4,
      f"no-trust relative step {notrust:.6f} vs trust {rel[1]:.4f}")

print("\ngrowth: new capacity inherits statistics instead of zero")
p = torch.randn(64, 64, device=dev, requires_grad=True)
o = Fabric([p], factored=True)
for _ in range(30):
    o.zero_grad(); p.grad = torch.randn_like(p); o.step()
vr_before = o.state[p]["vr"].clone()
big = torch.zeros(96, 96, device=dev); big[:64,:64] = p.detach()
newp = big.requires_grad_(True)
o.state[newp] = o.state.pop(p); o.param_groups[0]["params"] = [newp]
o.grew(newp, (64, 64))
vr = o.state[newp]["vr"]
check("state resized to the new shape", vr.numel() == 96, f"vr {vr.numel()}")
check("old slots preserved exactly", torch.allclose(vr[:64], vr_before))
check("new slots are NOT zero", (vr[64:] > 0).all().item(), f"new slots = {vr[64:].mean():.3e}")
check("new slots equal the median of the old", abs(vr[64:].mean().item() - vr_before.median().item()) < 1e-6)

print(f"\n{ok} passed, {fail} failed")
sys.exit(1 if fail else 0)
