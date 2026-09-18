"""Verify the associative carrier before a run depends on it. Run: .venv/bin/python tests/test_assoc.py"""
import sys, math, warnings; warnings.filterwarnings("ignore")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch, torch.nn.functional as F
from src.assoc import assoc_scan, AssocCarrier, rotate
from src.model import LM

ok = fail = 0
def check(name, cond, detail=""):
    global ok, fail
    if cond: ok += 1; print(f"  PASS  {name}" + (f"   {detail}" if detail else ""))
    else:    fail += 1; print(f"  FAIL  {name}" + (f"   {detail}" if detail else ""))
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)

print("the chunked scan against the literal recurrence")
B, H, T, D = 2, 3, 45, 8
q = torch.randn(B, H, T, D, device=dev, dtype=torch.float64); k = torch.randn_like(q); v = torch.randn_like(q)
la = -torch.rand(B, H, T, device=dev, dtype=torch.float64) * 3.0
S0 = torch.randn(B, H, D, D, device=dev, dtype=torch.float64)
def ref(q, k, v, la, S):
    out = []
    for t in range(q.shape[2]):
        S = torch.exp(la[:, :, t])[..., None, None] * S + k[:, :, t].unsqueeze(-1) @ v[:, :, t].unsqueeze(-2)
        out.append((q[:, :, t].unsqueeze(-2) @ S).squeeze(-2))
    return torch.stack(out, 2), S
yr, Sr = ref(q, k, v, la, S0.clone())
yc, Sc = assoc_scan(q, k, v, la, S0, chunk=8)
check("outputs match (fp64, T=45, chunk 8, carried state)", torch.allclose(yc, yr, atol=1e-11), f"max diff {(yc-yr).abs().max().item():.1e}")
check("final state matches", torch.allclose(Sc, Sr, atol=1e-11), f"max diff {(Sc-Sr).abs().max().item():.1e}")
yc2, _ = assoc_scan(q, k, v, la, S0, chunk=45); yc3, _ = assoc_scan(q, k, v, la, S0, chunk=1)
check("chunk size does not change the answer (45, 8, 1)", torch.allclose(yc2, yr, atol=1e-11) and torch.allclose(yc3, yr, atol=1e-11))
y0, _ = assoc_scan(q, k, v, la, None, chunk=8); yr0, _ = ref(q, k, v, la, torch.zeros_like(S0))
check("without a carried state", torch.allclose(y0, yr0, atol=1e-11))
a, st = assoc_scan(q[:, :, :20], k[:, :, :20], v[:, :, :20], la[:, :, :20], S0, chunk=8)
b, _ = assoc_scan(q[:, :, 20:], k[:, :, 20:], v[:, :, 20:], la[:, :, 20:], st, chunk=8)
check("two windows carried == one", torch.allclose(torch.cat([a, b], 2), yr, atol=1e-11))
for t_ in (q, k, v, la): t_.requires_grad_(True)
(assoc_scan(q, k, v, la, None, chunk=8)[0] ** 2).sum().backward(); g1 = [t_.grad.clone() for t_ in (q, k, v, la)]
for t_ in (q, k, v, la): t_.grad = None
(ref(q, k, v, la, torch.zeros_like(S0))[0] ** 2).sum().backward(); g2 = [t_.grad.clone() for t_ in (q, k, v, la)]
worst = max((x - y).abs().max().item() / (y.abs().max().item() + 1e-30) for x, y in zip(g1, g2))
check("gradients w.r.t. q, k, v, log-decay match the definition", worst < 1e-9, f"worst rel diff {worst:.1e}")
qf = torch.randn(2, 4, 512, 64, device=dev); yf, _ = assoc_scan(qf, torch.randn_like(qf), torch.randn_like(qf), -torch.rand(2, 4, 512, device=dev) * 40, chunk=64)
check("fp32, T=512, decays down to exp(-40): finite", torch.isfinite(yf).all().item())

print("\nthe carrier")
m = AssocCarrier(64).to(dev); x = torch.randn(2, 70, 64, device=dev)
y, (S, p) = m(x)
check("output shape and state shape", tuple(y.shape) == (2, 70, 64) and tuple(S.shape) == (2, 1, 64, 64) and p == 70)
ya, sa = m(x[:, :30]); yb, _ = m(x[:, 30:], h0=sa)
check("carried across windows == one pass, kernel path (bf16 matmuls, so 1e-2)", torch.allclose(torch.cat([ya, yb], 1), y, atol=1e-2),
      f"max diff {(torch.cat([ya, yb], 1) - y).abs().max().item():.1e}")
m.kernel = False
y, _ = m(x); ya, sa = m(x[:, :30]); yb, _ = m(x[:, 30:], h0=sa)
check("carried across windows == one pass, torch path (fp32, so 1e-4)", torch.allclose(torch.cat([ya, yb], 1), y, atol=1e-4),
      f"max diff {(torch.cat([ya, yb], 1) - y).abs().max().item():.1e}")
m.kernel = True
y, _ = m(x)                                      # the kernel path again: the causal check must compare like with like
x2 = x.clone(); x2[:, 50] += 3.0
y2, _ = m(x2)
check("causal: editing token 50 changes nothing before it", torch.equal(y2[:, :50], y[:, :50]) or (y2[:, :50] - y[:, :50]).abs().max().item() == 0.0,
      f"max change before 50: {(y2[:, :50]-y[:, :50]).abs().max().item():.1e}; at/after: {(y2[:, 50:]-y[:, 50:]).abs().max().item():.1e}")
xs = torch.randn(2, 40, 64, device=dev, requires_grad=True)
(m(xs)[0] ** 2).sum().backward()
check("all gradients finite", all(torch.isfinite(p.grad).all().item() for p in m.parameters() if p.grad is not None))
# rotation is a relative-position code: q_t . k_s depends on t - s only
th = torch.rand(1, 4, device=dev) * 0.3; qq = torch.randn(1, 1, 1, 8, device=dev); kk = torch.randn(1, 1, 1, 8, device=dev)
s1 = (rotate(qq, th, torch.tensor([10], device=dev)) * rotate(kk, th, torch.tensor([3], device=dev))).sum()
s2 = (rotate(qq, th, torch.tensor([107], device=dev)) * rotate(kk, th, torch.tensor([100], device=dev))).sum()
check("rotation makes q.k depend on t - s only", abs(s1.item() - s2.item()) < 1e-4, f"{s1.item():.5f} vs {s2.item():.5f}")

print("\nin the model")
for mx, d in (("aspin", 704), ("attn", 704), ("fspin", 704)):
    print(f"  {mx} d{d}: {sum(p.numel() for p in LM(d=d, layers=12, mixer=mx).parameters())/1e6:.2f}M")
m = LM(vocab=512, d=64, layers=2, ctx=32, mixer="aspin").to(dev); xi = torch.randint(0, 512, (2, 33), device=dev)
l = F.cross_entropy(m(xi[:, :-1]).reshape(-1, 512), xi[:, 1:].reshape(-1)); l.backward()
check("LM forward/backward with the associative mixer", torch.isfinite(l).item() and all(torch.isfinite(p.grad).all().item() for p in m.parameters() if p.grad is not None), f"loss {l.item():.3f}")
lg = m.forward_carried(torch.randint(0, 512, (2, 96), device=dev))
check("forward_carried threads the memory across windows", tuple(lg.shape) == (2, 32, 512))
g = m.param_groups(); spec = sum(p.numel() for p in g[1]["params"])
check("timescales, frequencies and gates are in the spectral (absolute-step) group", spec == sum(p.numel() for mx_ in m.mix for p in mx_.spectral_parameters()))

if dev == "cuda":
    print("\nbf16 matmuls, exact decay")
    # The scan's four matmuls run in the autocast dtype; the cumulative decay, the exponentials and the state S
    # stay fp32. Checked against the all-fp32 path: agreement to bf16's own precision, and NOT to less.
    torch.manual_seed(7)
    qb = F.normalize(torch.randn(2, 4, 512, 64, device=dev), dim=-1)
    kb = F.normalize(torch.randn_like(qb), dim=-1); vb = F.normalize(torch.randn_like(qb), dim=-1)
    lb = -F.softplus(torch.randn(2, 4, 512, device=dev)) * 0.01
    o32, S32 = assoc_scan(qb, kb, vb, lb, None, chunk=128)
    o16, S16 = assoc_scan(qb, kb, vb, lb, None, chunk=128, mm=torch.bfloat16)
    ro = ((o16 - o32).abs().max() / o32.abs().max()).item(); rs = ((S16 - S32).abs().max() / S32.abs().max()).item()
    check("outputs match fp32 to bf16 precision", ro < 0.03, f"max rel diff {ro:.1e}")
    check("the accumulated state stays close (it is never demoted)", rs < 0.03, f"max rel diff {rs:.1e}")
    # the decay itself must be exact: a 512-token chunk at a slow rate, where a bf16 cumsum would drift
    ls = torch.full((1, 1, 512), -1e-3, device=dev)
    o_a, _ = assoc_scan(qb[:1, :1], kb[:1, :1], vb[:1, :1], ls, None, chunk=512, mm=torch.bfloat16)
    o_b, _ = assoc_scan(qb[:1, :1].double(), kb[:1, :1].double(), vb[:1, :1].double(), ls.double(), None, chunk=512)
    check("a 512-token chunk at half-life 693 still matches fp64 within bf16 error",
          ((o_a - o_b.float()).abs().max() / o_b.abs().max()).item() < 0.03,
          f"max rel diff {((o_a - o_b.float()).abs().max() / o_b.abs().max()).item():.1e}")
    # and the model under autocast uses it without changing the loss materially
    torch.manual_seed(0)
    mm_ = LM(vocab=512, d=128, layers=2, ctx=64, mixer="aspin").cuda(); xi_ = torch.randint(0, 512, (4, 65), device=dev)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        la_ = F.cross_entropy(mm_(xi_[:, :-1]).float().reshape(-1, 512), xi_[:, 1:].reshape(-1))
    lb_ = F.cross_entropy(mm_(xi_[:, :-1]).float().reshape(-1, 512), xi_[:, 1:].reshape(-1))
    check("loss under autocast matches the fp32 loss", abs(la_.item() - lb_.item()) < 0.02, f"{la_.item():.4f} vs {lb_.item():.4f}")

print("\nthe delta write rule")
# The plain rule ADDS k v^T per token, so two values under similar keys leave their sum and neither reads back.
# The delta rule writes v minus what the memory already returns for that key. Each token's correction must see
# the writes before it in its chunk; a per-chunk (Jacobi) correction is stable only for beta < 2/chunk and went
# non-finite at step 128 of a real run. The triangular solve is exact and chunk-independent, which is the test.
kk = F.normalize(torch.randn(1, 1, 1, 4, device=dev, dtype=torch.float64), dim=-1)
va = torch.tensor([[[[1., 0, 0, 0]]]], device=dev, dtype=torch.float64)
vb = torch.tensor([[[[0., 1., 0, 0]]]], device=dev, dtype=torch.float64)
Kd = torch.cat([kk, kk, kk], 2); Vd = torch.cat([va, vb, va], 2)
lad = torch.zeros(1, 1, 3, device=dev, dtype=torch.float64); bed = torch.ones(1, 1, 3, device=dev, dtype=torch.float64)
r1 = assoc_scan(Kd, Kd, Vd, lad, None, chunk=1, beta=bed)[0][0, 0, 1]
r3 = assoc_scan(Kd, Kd, Vd, lad, None, chunk=3, beta=bed)[0][0, 0, 1]
rp = assoc_scan(Kd, Kd, Vd, lad, None, chunk=3)[0][0, 0, 1]
check("a key written twice reads back the SECOND value", torch.allclose(r1, vb[0, 0, 0], atol=1e-9),
      f"{[round(x, 4) for x in r1.tolist()]}")
check("and the plain rule reads back their sum, which is the interference", torch.allclose(rp, (va + vb)[0, 0, 0], atol=1e-9),
      f"{[round(x, 4) for x in rp.tolist()]}")
check("chunk size does not change the delta result", torch.allclose(r1, r3, atol=1e-10))
torch.manual_seed(1)
Kr = F.normalize(torch.randn(2, 4, 256, 64, device=dev), dim=-1); Vr = F.normalize(torch.randn_like(Kr), dim=-1)
lr_ = torch.zeros(2, 4, 256, device=dev); br = torch.ones(2, 4, 256, device=dev)
y16, S16 = assoc_scan(Kr, Kr, Vr, lr_, None, chunk=16, beta=br)
y256, S256 = assoc_scan(Kr, Kr, Vr, lr_, None, chunk=256, beta=br)
check("256 correlated writes stay bounded (the Jacobi form diverged here)", S16.abs().max().item() < 2.0,
      f"|S| max {S16.abs().max().item():.3f}")
check("chunk 16 and chunk 256 agree to fp32 rounding", (y16 - y256).abs().max().item() < 1e-5,
      f"max diff {(y16 - y256).abs().max().item():.1e}")
md = LM(vocab=512, d=128, layers=2, ctx=64, mixer="dspin").to(dev)
xd = torch.randint(0, 512, (2, 65), device=dev)
ld = F.cross_entropy(md(xd[:, :-1]).reshape(-1, 512), xd[:, 1:].reshape(-1)); ld.backward()
check("dspin trains: finite loss and gradients", torch.isfinite(ld).item()
      and all(torch.isfinite(p.grad).all().item() for p in md.parameters() if p.grad is not None), f"loss {ld.item():.3f}")

if dev == "cuda":
    print("\nthe Triton scan kernel against the two-pass form")
    from src.assoc_kernel import assoc_scan_kernel
    torch.manual_seed(0)
    for Dk in (64, 128):
        qk = F.normalize(torch.randn(2, 3, 333, Dk, device=dev), dim=-1); kk_ = F.normalize(torch.randn_like(qk), dim=-1)
        vk = F.normalize(torch.randn_like(qk), dim=-1); lak = -F.softplus(torch.randn(2, 3, 333, device=dev)) * 0.05
        S0k = torch.randn(2, 3, Dk, Dk, device=dev) * 0.1
        yr, Sr = assoc_scan(qk, kk_, vk, lak, S0k, chunk=64, mm=torch.bfloat16); yk, Sk, _ = assoc_scan_kernel(qk, kk_, vk, lak, S0k)
        check(f"D={Dk}, T=333 (partial last chunk), carried S0: forward within bf16 precision",
              ((yk - yr).abs().max() / yr.abs().max()).item() < 1e-2 and ((Sk - Sr).abs().max() / Sr.abs().max()).item() < 1e-2,
              f"y {((yk-yr).abs().max()/yr.abs().max()).item():.1e} S {((Sk-Sr).abs().max()/Sr.abs().max()).item():.1e}")
        args = [t.clone().requires_grad_(True) for t in (qk, kk_, vk, lak)]; wk = torch.randn_like(qk)
        (assoc_scan(*args, None, chunk=64)[0] * wk).sum().backward(); gr = [t.grad.clone() for t in args]
        for t in args: t.grad = None
        (assoc_scan_kernel(*args, None)[0] * wk).sum().backward()
        worst = max(((a.grad - b).abs().max() / (b.abs().max() + 1e-12)).item() for a, b in zip(args, gr))
        check(f"D={Dk}: gradients of q, k, v and the decay within bf16 precision of the fp32 reference", worst < 1e-2, f"worst {worst:.1e}")

    print("\nthe delta rule's kernels against the two-pass delta form")
    # The kernel path was measured against the reference by hand when it was built and then only for finiteness
    # in this file; every dtype and structure change to it since is checked here. bf16 dots, so 1e-2 relative.
    from src.assoc_kernel import delta_scan_kernel
    torch.manual_seed(2)
    for Dk in (64, 128):
        qk = F.normalize(torch.randn(2, 3, 333, Dk, device=dev), dim=-1); kk_ = F.normalize(torch.randn_like(qk), dim=-1)
        vk = F.normalize(torch.randn_like(qk), dim=-1); lak = -F.softplus(torch.randn(2, 3, 333, device=dev)) * 0.05
        bek = torch.sigmoid(torch.randn(2, 3, 333, device=dev)); S0k = torch.randn(2, 3, Dk, Dk, device=dev) * 0.1
        ar = [t.clone().requires_grad_(True) for t in (qk, kk_, vk, lak, bek, S0k)]
        ak = [t.clone().requires_grad_(True) for t in (qk, kk_, vk, lak, bek, S0k)]
        wk = torch.randn_like(qk); wS = torch.randn_like(S0k)
        yr, Sr = assoc_scan(*ar[:4], ar[5], chunk=64, beta=ar[4], kernel=False)
        yk, Sk, _ = delta_scan_kernel(*ak[:4], ak[4], ak[5])
        rel = lambda a, b: ((a - b).abs().max() / (b.abs().max() + 1e-12)).item()
        check(f"delta D={Dk}, T=333, carried S0: forward and final state within bf16 precision",
              rel(yk, yr) < 1e-2 and rel(Sk, Sr) < 1e-2, f"y {rel(yk, yr):.1e} S {rel(Sk, Sr):.1e}")
        ((yr * wk).sum() + (Sr * wS).sum()).backward(); ((yk * wk).sum() + (Sk * wS).sum()).backward()
        worst = {n: rel(a.grad, b.grad) for n, a, b in zip(("q", "k", "v", "decay", "beta", "S0"), ak, ar)}
        check(f"delta D={Dk}: all six gradients within bf16 precision of the reference", max(worst.values()) < 1e-2,
              " ".join(f"{n} {x:.1e}" for n, x in worst.items()))

    print("\nthe fused plumbing kernel (normalise, gate, rotate, cast) against the torch form")
    from src.assoc_kernel import prep_kernel
    def prep_ref(qp, kp, vp, gl, theta, pos):
        q_, k_, v_ = [F.normalize(t.transpose(1, 2).float(), dim=-1) for t in (qp, kp, vp)]
        if gl is not None:
            k_ = k_ * torch.sigmoid(gl.float()).transpose(1, 2).unsqueeze(-1)
        return rotate(q_, theta, pos), rotate(k_, theta, pos), v_
    rel = lambda a, b: ((a.float() - b.float()).abs().max() / (b.float().abs().max() + 1e-12)).item()
    torch.manual_seed(0)
    for Dk, Tp, gate, p0 in ((64, 2048, True, 0), (64, 333, False, 1000), (128, 512, True, 4096)):
        Hp = 3
        mk = lambda: (torch.randn(2, Tp, Hp, Dk, device=dev) * 3).to(torch.bfloat16).requires_grad_()
        qp, kp, vp = mk(), mk(), mk()
        gl = (torch.randn(2, Tp, Hp, device=dev) * 2).to(torch.bfloat16).requires_grad_() if gate else None
        th = (10000.0 ** (-torch.arange(0, Dk, 2, device=dev).float() / Dk)).unsqueeze(0).repeat(Hp, 1).requires_grad_()
        pos = p0 + torch.arange(Tp, device=dev)
        outs = prep_kernel(qp, kp, vp, gl, th, pos); refs = prep_ref(qp, kp, vp, gl, th, pos)
        e_out = max(rel(a, b) for a, b in zip(outs, refs))
        ws = [torch.randn_like(r) for r in refs]
        ins = [qp, kp, vp, th] + ([gl] if gate else [])
        gk = torch.autograd.grad(sum((o.float() * w).sum() for o, w in zip(outs, ws)), ins)
        gr = torch.autograd.grad(sum((r * w).sum() for r, w in zip(refs, ws)), ins)
        e_gr = max(rel(a, b) for a, b in zip(gk, gr))
        check(f"D={Dk}, T={Tp}, gate={gate}, positions from {p0}: outputs and every gradient (projections, theta"
              + (", gate" if gate else "") + ") within bf16 precision", e_out < 1e-2 and e_gr < 1e-2,
              f"out {e_out:.1e} grads {e_gr:.1e}")

print(f"\n{ok} passed, {fail} failed")
sys.exit(1 if fail else 0)
