"""Does the carrier work? Run: .venv/bin/python tests/test_spin.py

Four things, each of which would invalidate every later measurement if wrong:
  1. the fast chunked scan equals the sequential definition
  2. it is causal, so a token cannot see the future
  3. it is stable over sequences far longer than it was chunked for
  4. it actually carries information ACROSS tokens, rather than acting per-token
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch
from src.spin import SpinCarrier

ok = fail = 0
def check(n, c, d=""):
    global ok, fail
    if c: ok += 1; print(f"  PASS  {n}" + (f"   {d}" if d else ""))
    else: fail += 1; print(f"  FAIL  {n}" + (f"   {d}" if d else ""))

dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)

print("fast path equals the definition")
for T, C in [(64, 64), (128, 64), (200, 64), (512, 128), (1000, 64)]:
    m = SpinCarrier(32, chunk=C).to(dev).double()
    u = torch.randn(3, T, 32, device=dev, dtype=torch.float64)
    ref, _ = m.scan_reference(u)
    fast, _ = m.scan(u)
    err = (ref - fast).abs().max().item() / ref.abs().max().item()
    check(f"T={T} chunk={C}", err < 1e-8, f"max relative error {err:.2e}")

print("\nfast path in FLOAT32, the training dtype")
# An FFT-based scan has roundoff of order 1e-7 in float32. That is the regime training runs in, and it sits
# five orders of magnitude below bf16 precision, so this is the tolerance that matters in practice.
m32 = SpinCarrier(64).to(dev)
u32 = torch.randn(3, 512, 64, device=dev)
ref32, _ = m32.double().scan_reference(u32.double())
m32 = m32.float()
fast32, _ = m32.scan(u32)
e32 = ((ref32.float() - fast32).abs().max() / ref32.abs().max()).item()
check("float32 scan within 1e-5 of the definition", e32 < 1e-5, f"max relative error {e32:.2e}")

print("\ncausality")
m = SpinCarrier(32, chunk=64).to(dev).double()
u = torch.randn(2, 96, 32, device=dev, dtype=torch.float64)
y1, _ = m.scan(u)
u2 = u.clone(); u2[:, 50:] = torch.randn_like(u2[:, 50:])       # change the future only
y2, _ = m.scan(u2)
check("changing t>=50 leaves t<50 untouched", torch.allclose(y1[:, :50], y2[:, :50], atol=1e-9),
      f"max diff before the edit {(y1[:,:50]-y2[:,:50]).abs().max().item():.2e}")
check("changing t>=50 DOES change t>=50", not torch.allclose(y1[:, 50:], y2[:, 50:], atol=1e-6))

print("\nstability over long sequences")
m = SpinCarrier(64, chunk=64).to(dev)
u = torch.randn(1, 8192, 64, device=dev)
y, (hr, hi) = m.scan(u)
check("no NaN or inf at T=8192", torch.isfinite(y).all().item())
check("output magnitude stays bounded", y.abs().max().item() < 1e4, f"max |y| = {y.abs().max().item():.1f}")
mag, _ = m.lam()
check("|lambda| < 1 for every channel", (mag < 1.0).all().item(), f"max |lambda| = {mag.max().item():.6f}")

print("\nit actually carries information across tokens")
m = SpinCarrier(48, chunk=64).to(dev).double()
u = torch.zeros(1, 128, 48, device=dev, dtype=torch.float64)
u[0, 10] = 1.0                                                   # a single impulse at t=10
y, _ = m.scan(u)
e_before = y[0, :10].abs().sum().item()
e_at = y[0, 10].abs().sum().item()
e_far = y[0, 100].abs().sum().item()
check("nothing before the impulse", e_before < 1e-9, f"{e_before:.1e}")
check("response at the impulse", e_at > 1e-3, f"{e_at:.3f}")
check("still responding 90 tokens later", e_far > 1e-6, f"{e_far:.2e} at t=100 (decayed from {e_at:.3f})")

print("\ngradients flow")
m = SpinCarrier(32, chunk=64).to(dev)
x = torch.randn(2, 256, 32, device=dev, requires_grad=True)
out, _ = m(x)
out.sum().backward()
check("input gradient is finite and nonzero", torch.isfinite(x.grad).all().item() and x.grad.abs().sum() > 0)
check("nu and theta receive gradient",
      m.nu_raw.grad is not None and m.theta.grad is not None
      and m.nu_raw.grad.abs().sum() > 0 and m.theta.grad.abs().sum() > 0)

print("\noutput scale is independent of the channel timescale")
# The failure this catches: without gamma, a channel with |lambda| near 1 accumulates the whole sequence and
# its gain grows with T. Measured at init before the fix: loss 301 against a random baseline of 9.0.
m = SpinCarrier(256, chunk=64).to(dev)
x = torch.randn(4, 512, 256, device=dev)
with torch.no_grad():
    u = m.B(m.ln(x)) * m.gamma()
    y, _ = m.scan(u)
mag, _ = m.lam()
slow = mag > mag.median()          # long-memory channels
fast = ~slow
rs = y[:, :, slow].std().item(); rf = y[:, :, fast].std().item()
check("slow and fast channels have comparable scale", 0.25 < rs/rf < 4.0,
      f"slow std {rs:.3f} vs fast std {rf:.3f}  ratio {rs/rf:.2f}")
with torch.no_grad():
    y2, _ = m.scan(m.B(m.ln(torch.randn(4, 2048, 256, device=dev))) * m.gamma())
check("scale does not grow with sequence length", abs(y2.std().item()/y.std().item() - 1) < 0.5,
      f"T=512 std {y.std().item():.3f} vs T=2048 std {y2.std().item():.3f}")

print("\ncarried state is exact: two windows equal one long one")
# The real-FFT scan only emits Re(h); the imaginary state is recovered separately for carrying. If that
# recovery is wrong, every distance beyond the window in the goal-9 test is measured against a corrupted
# state and reads as a clean architectural limit. So: split a sequence, carry, and demand equality.
m = SpinCarrier(48, chunk=64).to(dev).double()
u = torch.randn(2, 300, 48, device=dev, dtype=torch.float64)
full, (fr, fi) = m.scan(u)
ref_full, (rr, ri) = m.scan_reference(u)
check("final real state matches reference", torch.allclose(fr, rr, atol=1e-10), f"{(fr-rr).abs().max().item():.1e}")
check("final IMAGINARY state matches reference", torch.allclose(fi, ri, atol=1e-10), f"{(fi-ri).abs().max().item():.1e}")
a, st = m.scan(u[:, :128])
b, _ = m.scan(u[:, 128:], h0=st)
joined = torch.cat([a, b], 1)
check("scan(first) then scan(rest, carried) == scan(all)", torch.allclose(joined, full, atol=1e-10),
      f"max diff {(joined-full).abs().max().item():.1e}")
a2, st2 = m.scan(u[:, :100]); b2, st3 = m.scan(u[:, 100:200], h0=st2); c2, _ = m.scan(u[:, 200:], h0=st3)
check("three windows carried == one", torch.allclose(torch.cat([a2, b2, c2], 1), full, atol=1e-10))

print("\nsurvives the forgetting rates that killed the chunked scan")
# The chunked scan computed lambda^(-128), which is exp(128*nu). Training pushed nu_max to 0.38 in a thousand
# steps (lambda^-128 = 1.7e21) and the run went NaN at about step 14,500, twice. Here nu is forced far past
# that. The FFT kernel is lambda^(+t), magnitude <= 1, so it cannot overflow at any rate.
m = SpinCarrier(64).to(dev)
with torch.no_grad():
    m.nu_raw.fill_(3.0)          # softplus(3) + 1e-3 = 3.05 -> lambda^-128 would be exp(390) = overflow
x = torch.randn(2, 512, 64, device=dev)
with torch.no_grad():
    y, _ = m.scan(m.B(m.ln(x)) * m.gamma())
mag, _ = m.lam()
check("nu=3.05 (lambda^-128 would be 1e169): output finite", torch.isfinite(y).all().item(),
      f"|lambda| = {mag.max().item():.4f}, max |y| = {y.abs().max().item():.2f}")
with torch.no_grad():
    u_in = (m.B(m.ln(x)) * m.gamma()).detach()
ref, _ = m.double().scan_reference(u_in.double())
m = m.float()
e = ((ref.float() - y).abs().max() / ref.abs().max()).item()
check("and still matches the definition", e < 1e-4, f"max relative error {e:.2e}")
with torch.no_grad():
    m.nu_raw.fill_(-3.0)         # very slow channels, |lambda| -> 0.998
    y2, _ = m.scan(m.B(m.ln(x)) * m.gamma())
check("and the slow extreme is finite too", torch.isfinite(y2).all().item())

print("\nhand-written backward of the causal convolution")
# The Function's backward is two correlations derived by hand. Autograd through the same forward (with the
# Function bypassed) is the reference; in fp64 they must agree to round-off, on u AND on the kernel.
from src.spin import _CausalConv
torch.manual_seed(3)
u64 = torch.randn(2, 37, 5, device=dev, dtype=torch.float64, requires_grad=True)
k64 = torch.randn(37, 5, device=dev, dtype=torch.float64, requires_grad=True)
w = torch.randn(2, 37, 5, device=dev, dtype=torch.float64)
(_CausalConv.apply(u64, k64) * w).sum().backward()
gu_f, gk_f = u64.grad.clone(), k64.grad.clone(); u64.grad = None; k64.grad = None
# reference: the same convolution by autograd through plain FFT ops
n = 74
yr = torch.fft.irfft(torch.fft.rfft(u64, n=n, dim=1) * torch.fft.rfft(k64, n=n, dim=0).unsqueeze(0), n=n, dim=1)[:, :37]
(yr * w).sum().backward()
check("dL/du matches autograd (fp64)", torch.allclose(gu_f, u64.grad, atol=1e-12, rtol=1e-10),
      f"max diff {(gu_f-u64.grad).abs().max().item():.1e}")
check("dL/dk matches autograd (fp64)", torch.allclose(gk_f, k64.grad, atol=1e-12, rtol=1e-10),
      f"max diff {(gk_f-k64.grad).abs().max().item():.1e}")
check("torch.autograd.gradcheck passes", torch.autograd.gradcheck(_CausalConv.apply, (u64, k64), eps=1e-6, atol=1e-6))
# odd T, T=1, and B=1 edge shapes
for (B_, T) in [(1, 1), (3, 2), (1, 129)]:
    a = torch.randn(B_, T, 4, device=dev, dtype=torch.float64, requires_grad=True)
    b = torch.randn(T, 4, device=dev, dtype=torch.float64, requires_grad=True)
    check(f"gradcheck at B={B_} T={T}", torch.autograd.gradcheck(_CausalConv.apply, (a, b), eps=1e-6, atol=1e-6))
# and end to end: the whole carrier's gradient on its spectral parameters through the Function equals the
# gradient through the sequential definition
m = SpinCarrier(16).double().to(dev)
x = torch.randn(2, 64, 16, device=dev, dtype=torch.float64)
m.zero_grad(); (m(x)[0] ** 2).sum().backward()
g1 = [p.grad.clone() for p in m.parameters()]
m.zero_grad()
u_ = m.B(m.ln(x)) * m.gamma(); y_, _ = m.scan_reference(u_)
((x + torch.sigmoid(m.gate) * m.C(y_)) ** 2).sum().backward()
g2 = [p.grad.clone() for p in m.parameters()]
worst = max((a - b).abs().max().item() / (b.abs().max().item() + 1e-30) for a, b in zip(g1, g2))
check("carrier gradients (nu, theta, B, C, gate) match the sequential definition", worst < 1e-8,
      f"worst relative diff {worst:.1e}")

print("\nper-token decay: the gated scan against the definition")
# lambda_t differs at every token; the reference is the literal recurrence. Includes a channel whose decay is
# extreme (logmag -8 on some tokens, |lambda|=3e-4: a near-reset) and one near 1, and a carried state.
torch.manual_seed(5)
B2, T2, D2 = 2, 45, 6
u2 = torch.randn(B2, T2, D2, device=dev, dtype=torch.float64)
lm2 = -torch.rand(B2, T2, D2, device=dev, dtype=torch.float64) * 8.0
lm2[:, :, 0] = -1e-3                                                      # a slow channel
an2 = torch.randn(B2, T2, D2, device=dev, dtype=torch.float64)
h0r = torch.randn(B2, D2, device=dev, dtype=torch.float64); h0i = torch.randn(B2, D2, device=dev, dtype=torch.float64)
def ref_gated(u, lm, an, hr, hi):
    out = []
    for t in range(u.shape[1]):
        lr, li = torch.exp(lm[:, t]) * torch.cos(an[:, t]), torch.exp(lm[:, t]) * torch.sin(an[:, t])
        hr, hi = hr * lr - hi * li + u[:, t], hr * li + hi * lr
        out.append(hr)
    return torch.stack(out, 1), (hr, hi)
yr, (rr, ri) = ref_gated(u2, lm2, an2, h0r.clone(), h0i.clone())
yg, (gr, gi) = SpinCarrier.scan_gated(u2, lm2, an2, (h0r, h0i), chunk=8)
check("gated scan matches the definition (fp64, T=45, chunk 8, carried state)", torch.allclose(yg, yr, atol=1e-12),
      f"max diff {(yg-yr).abs().max().item():.1e}")
check("final complex state matches", torch.allclose(gr, rr, atol=1e-12) and torch.allclose(gi, ri, atol=1e-12))
yg2, _ = SpinCarrier.scan_gated(u2, lm2, an2, (h0r, h0i), chunk=45)
check("chunk size does not change the answer", torch.allclose(yg2, yr, atol=1e-12))
a, st = SpinCarrier.scan_gated(u2[:, :20], lm2[:, :20], an2[:, :20], (h0r, h0i), chunk=8)
b, _ = SpinCarrier.scan_gated(u2[:, 20:], lm2[:, 20:], an2[:, 20:], st, chunk=8)
check("two windows carried == one", torch.allclose(torch.cat([a, b], 1), yr, atol=1e-12))
# gradients through it, against the reference
for t_ in (u2, lm2, an2): t_.requires_grad_(True)
(SpinCarrier.scan_gated(u2, lm2, an2, None, chunk=8)[0] ** 2).sum().backward()
g1 = [t_.grad.clone() for t_ in (u2, lm2, an2)]
for t_ in (u2, lm2, an2): t_.grad = None
(ref_gated(u2, lm2, an2, torch.zeros_like(h0r), torch.zeros_like(h0i))[0] ** 2).sum().backward()
g2 = [t_.grad.clone() for t_ in (u2, lm2, an2)]
worst = max((x - y).abs().max().item() / (y.abs().max().item() + 1e-30) for x, y in zip(g1, g2))
check("gradients w.r.t. u, logmag, angle match the definition", worst < 1e-9, f"worst rel diff {worst:.1e}")
# nothing can overflow: extreme decays, long sequence, fp32
uf = torch.randn(2, 512, 8, device=dev); lmf = -torch.rand(2, 512, 8, device=dev) * 30.0; anf = torch.randn(2, 512, 8, device=dev)
yf, _ = SpinCarrier.scan_gated(uf, lmf, anf, None, chunk=16)
check("fp32, T=512, decays down to exp(-30): finite", torch.isfinite(yf).all().item())

if dev == "cuda":
    print("\nthe Triton scan kernel against the definition")
    from src.scan_kernel import scan_triton
    torch.manual_seed(5)
    Bk, Tk, Dk = 2, 45, 300                              # D not a multiple of the block, T odd
    uk = torch.randn(Bk, Tk, Dk, device=dev); lmk = -torch.rand(Bk, Tk, Dk, device=dev) * 8; ank = torch.randn(Bk, Tk, Dk, device=dev)
    h0k = (torch.randn(Bk, Dk, device=dev), torch.randn(Bk, Dk, device=dev))
    yr, (rr, ri) = SpinCarrier.scan_gated(uk.double(), lmk.double(), ank.double(), (h0k[0].double(), h0k[1].double()), chunk=8)
    yt, (tr, ti) = scan_triton(uk, lmk, ank, h0k)
    check("forward matches the fp64 definition to fp32 precision", (yt - yr.float()).abs().max().item() < 1e-5
          and (tr - rr.float()).abs().max().item() < 1e-5 and (ti - ri.float()).abs().max().item() < 1e-5,
          f"max diff {(yt-yr.float()).abs().max().item():.1e}")
    w = torch.randn(Bk, Tk, Dk, device=dev); wl = torch.randn(Bk, Dk, device=dev); wi = torch.randn(Bk, Dk, device=dev)
    def lossf(fn, *a):
        y, (r, i) = fn(*a); return (y * w).sum() + (r * wl).sum() + (i * wi).sum()
    a64 = [t.double().requires_grad_(True) for t in (uk, lmk, ank)] + [t.double().requires_grad_(True) for t in h0k]
    lossf(lambda a, b, c, d, e: SpinCarrier.scan_gated(a, b, c, (d, e), chunk=8), *a64).backward()
    a32 = [t.clone().requires_grad_(True) for t in (uk, lmk, ank)] + [t.clone().requires_grad_(True) for t in h0k]
    lossf(lambda a, b, c, d, e: scan_triton(a, b, c, (d, e)), *a32).backward()
    worst = max(((a.grad - b.grad.float()).abs().max() / (b.grad.abs().max() + 1e-30)).item() for a, b in zip(a32, a64))
    check("backward (u, logmag, angle, h0 real and imaginary; gradient through the final Im state too) matches", worst < 1e-5,
          f"worst rel diff {worst:.1e}")
    yt0, _ = scan_triton(uk, lmk, ank, None); yr0, _ = SpinCarrier.scan_gated(uk.double(), lmk.double(), ank.double(), None, chunk=8)
    check("without a carried state", (yt0 - yr0.float()).abs().max().item() < 1e-5)
    m = SpinCarrier(48, gated="rf").to(dev); xg = torch.randn(2, 70, 48, device=dev)
    yA, _ = m(xg)
    import src.spin as _sp
    z = m.ln(xg); mag, ang = m.lam(); lg = -m.gate_c * torch.sigmoid(m.Wr(z)) * (-torch.log(mag))
    ug = m.B(z) * torch.sqrt(1 - torch.exp(2 * lg)).clamp_min(1e-4) * torch.sigmoid(m.Wi(z))
    yB = xg + torch.sigmoid(m.gate) * m.C(_sp.SpinCarrier.scan_gated(ug.double(), lg.double(), ang.double().expand_as(lg), None)[0].float())
    check("the forget-gated carrier through the kernel equals it through the chunked fp64 scan", (yA - yB).abs().max().item() < 1e-5,
          f"max diff {(yA-yB).abs().max().item():.1e}")
    lmx = torch.full((2, 512, 8), -40.0, device=dev); yx, _ = scan_triton(torch.randn(2, 512, 8, device=dev), lmx, torch.randn(2, 512, 8, device=dev))
    check("decay exp(-40) at every token, T=512: finite", torch.isfinite(yx).all().item())

print(f"\n{ok} passed, {fail} failed")
sys.exit(1 if fail else 0)
