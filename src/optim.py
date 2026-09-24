"""A memory-lean, scale-free, growth-aware optimizer.

Each piece is individually known. The combination is what is being tested, and each can be switched off so an
ablation can say which part earned its place.

WHY NOT ADAM. Three reasons, all measured on the previous effort rather than assumed:

  1. MEMORY. Adam keeps m and v per parameter, 8 bytes each in fp32, so weights + grads + state is 12 bytes
     per parameter and a 4 GB budget stops at 333M parameters. Factoring the second moment into row and column
     vectors, as Adafactor does, removes almost all of that: state becomes O(R+C) instead of O(R*C).

  2. SCALE. A single learning rate held constant while a model grew from 125M to 405M drove BPB from 0.9329 up
     to 1.0285 over hundreds of thousands of steps, a smooth monotone climb, which is a model equilibrating at
     the noise floor its rate implies. A trust ratio makes each parameter move a fixed FRACTION OF ITS OWN
     NORM, so the step size follows the weights instead of a hand-set constant. That removes a piece of
     scaffolding rather than tuning it (vision goal 8).

  3. GROWTH. When new parameters appear mid-training, Adam starts their second moment at zero and divides by
     its square root, so the first update is near maximal on exactly the capacity least able to absorb it.
     Measured: a reset optimizer takes a 1.5x larger first step than one whose state is carried. Here, new rows
     and columns inherit the median statistics of the tensor they were grown into, so a grown parameter starts
     with the optimizer's existing opinion instead of no opinion.
"""
from __future__ import annotations
import torch
import math
from torch.optim import Optimizer


def _update(p, g, vr, vc, v, m, lr, b1: float, b2: float, eps: float, wd: float,
            min_norm: float, clip: float, trust: bool, acc=None, dsp=None, dog=None):
    """One parameter's update, in place on p (pass p.detach()). THE math; eager and fused paths both run it.

    vr/vc are the factored second moment (2D), else v is the dense one. m is the momentum or None. lr is a
    0-d tensor so the same compiled kernel serves every step of a schedule."""
    # a non-finite gradient contributes nothing: the trainer's guard zeroes a skipped step's gradients with a
    # multiply, and NaN * 0 is NaN, so the zeroing has to finish here, fused into the update
    gf = torch.nan_to_num(g.float(), nan=0.0, posinf=0.0, neginf=0.0)
    if acc is not None:
        # THE HYPERGRADIENT, over a HORIZON (goal 8). dLoss/dlr = -g . (the steps that brought us here). Taking
        # only the last step (Baydin et al. 2018) is greedy and known to settle on too-small rates ("short-horizon
        # bias", Wu et al. 2018): measured here, 1,000 steps at d512 reached loss 6.76 against 5.66 for the tuned
        # schedule, with the rate stuck near 2.6e-4. So the last step is replaced by `dsp`, a decaying sum of the
        # steps actually taken, with the same decay as the second moment (b2): the horizon over which the
        # optimizer already remembers gradient scale. If this gradient still points along where the parameters
        # have been heading, the steps were too short; if back against it, too long. Three sums leave here.
        dp = dsp.float()
        acc[0] += (gf * dp).sum()
        acc[1] += gf.square().sum()
        acc[2] += dp.square().sum()
    if vr is not None:
        sq = gf.square().add(eps)
        vr.mul_(b2).add_(sq.mean(dim=1), alpha=1 - b2)
        vc.mul_(b2).add_(sq.mean(dim=0), alpha=1 - b2)
        # reconstruct: outer product renormalised by the row mean, which is the Adafactor estimator
        r = vr / vr.mean().clamp_min(eps)
        vv = (r.unsqueeze(1) * vc.unsqueeze(0)).clamp_min(eps)
    else:
        v.mul_(b2).addcmul_(gf, gf, value=1 - b2)
        vv = v.clamp_min(eps)
    u = gf / vv.sqrt()
    if m is not None:
        m.mul_(b1).add_(u.bfloat16(), alpha=1 - b1)
        u = m.float()
    if wd:
        u = u.add(p.float(), alpha=wd)
    if dog is not None:
        # DISTANCE OVER GRADIENTS, layer-wise (Ivgi, Hinder & Carmon 2023; goal 8): the step size is how far this
        # tensor has already moved from where it started, over the root of the summed squared update sizes.
        #   eta_t = rbar_t / sqrt(sum_k |u_k|^2),   rbar_t = max(r_eps, max_k |w_k - w_0|)
        # Moving coherently grows rbar, so the rate rises on its own (a warm-up nobody set); once the steps are
        # noise, rbar stalls while the sum keeps growing, so the rate falls on its own (a decay nobody set).
        # A zero-initialised tensor starts at r_eps and grows like any other: no special rate for new layers.
        # dog = (w0 bf16, [rbar, G] fp32). Replaces the trust ratio and `lr` entirely.
        w0, rg = dog
        rg[0] = torch.maximum(rg[0], (p.float() - w0.float()).norm())
        rg[1] += u.square().sum()
        u = u * (rg[0] / rg[1].sqrt().clamp_min(1e-30))
        lr = torch.ones((), device=p.device)
    elif trust:
        # Move a fixed fraction of the parameter's OWN norm, so |dp| = lr * |w| exactly and the step is
        # invariant to how the weights happen to be scaled.
        #
        # trust_clip defaults high enough not to bind. A tight clip (10.0 was tried) destroys the very
        # invariance this exists for: measured, a parameter at norm 100 took a relative step of 0.0010 where
        # one at norm 1 took 0.0100, because the ratio hit the clip. The runaway case a clip is meant to
        # catch, |u| near zero, is already handled by clamp_min(eps): the resulting step has norm lr*|w| no
        # matter how small |u| is.
        #
        # min_norm lets zero-initialised capacity move at all. Without it a freshly grown tensor has |w| = 0,
        # so the step is 0, and new capacity would be frozen forever.
        wn = p.float().norm().clamp_min(min_norm)
        un = u.norm().clamp_min(eps)
        u = u * (wn / un).clamp(max=clip)
    # A ZERO GRADIENT MEANS NO STEP. The trainer's spike guard skips a step by zeroing its gradients, but the
    # momentum still carried the previous direction and the trust ratio rescaled that decaying momentum back
    # to a full lr * |w| step, so a "skipped" step moved every weight as far as a real one. Measured 2026-09-23:
    # 299 consecutive skipped steps after a deepen drove the loss from 6.40 to 9.4, above uniform.
    live = (gf.abs().amax() > 0).to(p.dtype)
    p.sub_(u.to(p.dtype) * lr * live)
    if dsp is not None:
        dsp.mul_(b2).add_((u * lr * live).to(dsp.dtype))     # the displacement the next hypergradient reads


_fused = None      # torch.compile(_update), built on first use


class Fabric(Optimizer):
    """Factored second moment + trust-ratio step.

    Args:
        lr: the trust ratio when `trust=True`, i.e. the fraction of its own norm a parameter moves per step.
            With `trust=False` it is an ordinary learning rate.
        betas: (momentum, second-moment) decay.
        trust: scale-free stepping. Off, this is close to Adafactor with momentum.
        factored: factor the second moment for 2D parameters. Off, this stores v densely like Adam.
        momentum: keep a first moment. Off, this halves state again at some cost in stability.
        fused: run each parameter's update as one torch.compile'd kernel group instead of ~28 eager kernels.
            Same math, one rounding fewer on the bf16 momentum (measured 3e-3 relative on the step).
        cuda_graph: with fused, capture the whole step once and replay it. The eager step is CPU-bound:
            measured 19.5 ms of CPU issuing 4,500 kernels for 160 tensors, against a 35 ms model step, so
            the optimizer was ~40% of every training step. Replay needs grads and state at fixed addresses;
            the trainer must zero grads in place (set_to_none=False). Addresses are checked before every
            replay and the graph is recaptured if any moved, so a silent stale read cannot happen.
    """

    def __init__(self, params, lr=2e-3, betas=(0.9, 0.99), eps=1e-30, weight_decay=0.0,
                 trust=True, factored=True, momentum=True, trust_clip=1e4, min_norm=1e-3,
                 intrinsic_lr=False, adapt=0.02, lr_bounds=(1e-4, 3e-2), fused=False, cuda_graph=False,
                 hyper=False, hyper_beta=0.02, dog=False):
        # intrinsic_lr defaults OFF. Three designs were tried on 2026-09-10 and all three produced NaN. They
        # are kept in the code because the failure modes are the useful part; see the comment in step().
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                                      trust=trust, factored=factored, momentum=momentum,
                                      trust_clip=trust_clip, min_norm=min_norm,
                                      intrinsic_lr=intrinsic_lr, adapt=adapt, lr_bounds=lr_bounds,
                                      hyper=hyper, dog=dog))
        # hyper: the rate is the optimizer's own (goal 8). `lr` is only where it STARTS; every group then moves
        # its rate multiplicatively by hyper_beta x cos(g_t, dw_{t-1}), a bounded signal with a restoring force
        # (overshoot makes it negative), unlike the three designs recorded in _step_intrinsic, which ratcheted.
        self.hyper_beta = hyper_beta
        if cuda_graph and not fused:
            raise ValueError("cuda_graph needs fused=True")
        if fused and intrinsic_lr:
            raise ValueError("the intrinsic-rate path is python per step and cannot be fused")
        self.fused, self.cuda_graph = fused, cuda_graph
        self._graph = None; self._ptrs = None; self._warm = False; self.recaptures = 0
        for gp in self.param_groups:
            gp["lr_t"] = None; gp["acc"] = None

    def state_dict(self):
        for gp in self.param_groups:             # an intrinsic rate lives on the device; bring it home to save
            if gp.get("hyper") and gp["lr_t"] is not None:
                gp["lr"] = float(gp["lr_t"])
        sd = super().state_dict()
        for g in sd["param_groups"]:
            g.pop("lr_t", None); g.pop("acc", None)   # device scalars rebuilt on the next step; not state
        return sd

    def rates(self):
        """The rate each group is using now (one sync; for logging). Under DoG there is no group rate, so this
        is the median over tensors of the relative step it implies, rbar / (|w| sqrt(steps))."""
        out = []
        for gp in self.param_groups:
            if gp.get("dog"):
                r = [float(self.state[p]["rg"][0]) / (float(p.float().norm().clamp_min(1e-3)) * max(1, self.state[p]["step"]) ** 0.5)
                     for p in gp["params"] if self.state.get(p) and "rg" in self.state[p]]
                out.append(float(sorted(r)[len(r) // 2]) if r else 0.0)
            else:
                out.append(float(gp["lr_t"]) if gp["lr_t"] is not None else gp["lr"])
        return out

    def load_state_dict(self, sd):
        super().load_state_dict(sd)
        self._graph = None; self._warm = False   # state tensors were replaced; a captured graph is stale
        for gp in self.param_groups:
            gp["lr_t"] = None; gp["acc"] = None
            for p in gp["params"]:
                st = self.state.get(p)
                # torch casts every floating state tensor to the PARAMETER's dtype on load, which silently
                # turned the bf16 momentum into fp32 on every --resume: 2x the state memory and a different
                # rounding from the run being resumed. bf16 -> fp32 -> bf16 is exact, so restoring it is free.
                for k in ("m", "dsp", "w0"):
                    if st and k in st and st[k].dtype != torch.bfloat16:
                        st[k] = st[k].to(torch.bfloat16)

    # ---------------------------------------------------------------------------------- the step
    def _init_state(self, p, gp):
        st = self.state[p]
        st["step"] = 0
        if gp["factored"] and p.dim() == 2:
            st["vr"] = torch.zeros(p.shape[0], device=p.device, dtype=torch.float32)
            st["vc"] = torch.zeros(p.shape[1], device=p.device, dtype=torch.float32)
        else:
            st["v"] = torch.zeros_like(p, dtype=torch.float32)
        if gp["momentum"]:
            st["m"] = torch.zeros_like(p, dtype=torch.bfloat16)
        if gp.get("hyper"):
            st["dsp"] = torch.zeros_like(p, dtype=torch.bfloat16)
        if gp.get("dog"):
            # r_eps = 1e-4 (1 + |w0|), the DoG paper's default, reported insensitive over orders of magnitude
            st["w0"] = p.detach().to(torch.bfloat16).clone()
            st["rg"] = torch.tensor([1e-4 * (1 + float(p.float().norm())), 0.0], device=p.device)
        st["lr"] = gp["lr"]
        return st

    def _lr_t(self, gp, device):
        if gp["lr_t"] is None or gp["lr_t"].device != device:
            gp["lr_t"] = torch.full((), float(gp["lr"]), device=device)
            gp["acc"] = torch.zeros(3, device=device)
        if not gp.get("hyper"):                  # an intrinsic rate is never overwritten from the host
            gp["lr_t"].fill_(gp["lr"])
        return gp["lr_t"]

    def _apply(self, p, gp, st, lr_t):
        global _fused
        fn = _update
        if self.fused:
            if _fused is None:
                import torch._dynamo
                torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 64)
                _fused = torch.compile(_update, dynamic=True, fullgraph=True)
            fn = _fused
        fn(p.detach(), p.grad, st.get("vr"), st.get("vc"), st.get("v"), st.get("m"), lr_t,
           gp["betas"][0], gp["betas"][1], gp["eps"], gp["weight_decay"], gp["min_norm"],
           gp["trust_clip"], gp["trust"], gp["acc"] if gp.get("hyper") else None,
           st.get("dsp") if gp.get("hyper") else None,
           (st["w0"], st["rg"]) if gp.get("dog") else None)

    def _run(self, fill_lr=True):
        """One step over all groups, through _update (eager or fused).

        fill_lr=False while capturing: a fill_(float) recorded inside the graph would bake the capture-time
        rate into every replay. The graph reads lr_t; the caller fills it before each replay."""
        for gp in self.param_groups:
            for p in gp["params"]:
                if p.grad is None:
                    continue
                st = self.state[p] or self._init_state(p, gp)
                st["step"] += 1
                if fill_lr or gp["lr_t"] is None:
                    self._lr_t(gp, p.device)
                self._apply(p, gp, st, gp["lr_t"])
            if gp.get("hyper") and gp["acc"] is not None:
                # one multiplicative move per group per step, entirely on the device (so it lives inside the
                # captured graph). cos is in [-1, 1]; a skipped step has zero gradient, cos 0, no move. The
                # clamp is a NaN guard far outside any rate a run reaches, not a bound the rate is meant to hit.
                a = gp["acc"]
                cos = a[0] / (a[1] * a[2]).sqrt().clamp_min(1e-30)
                gp["lr_t"].mul_(torch.exp(self.hyper_beta * cos)).clamp_(1e-7, 1.0)
                a.zero_()

    def _addresses(self):
        out = []
        for gp in self.param_groups:
            for p in gp["params"]:
                if p.grad is None:
                    out.append(None); continue
                st = self.state[p]
                out.append((p.data_ptr(), p.grad.data_ptr()) + tuple(st[k].data_ptr() for k in ("vr", "vc", "v", "m", "dsp", "w0", "rg") if k in st))
        return out

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        if any(gp["intrinsic_lr"] for gp in self.param_groups):
            self._step_intrinsic(); return loss
        if not self.cuda_graph:
            self._run(); return loss
        # CUDA-graph path. First call: run for real on a side stream (compiles and warms the allocator).
        # Second: capture, then replay to perform the step. After: replay, unless an address moved.
        if not self._warm:
            s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                self._run()
            torch.cuda.current_stream().wait_stream(s)
            self._warm = True
            return loss
        if self._graph is None or self._ptrs != self._addresses():
            if self._graph is not None:
                self.recaptures += 1
            for gp in self.param_groups:                       # lr tensors filled OUTSIDE the capture
                for p in gp["params"]:
                    if p.grad is not None:
                        self._lr_t(gp, p.device); break
            self._graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self._graph):
                self._run(fill_lr=False)
            self._ptrs = self._addresses()
            # capture executes nothing; the step counters advanced inside _run, which is what one step needs
        else:
            for gp in self.param_groups:
                if gp["lr_t"] is not None and not gp.get("hyper"):
                    gp["lr_t"].fill_(gp["lr"])
                for p in gp["params"]:
                    if p.grad is not None:
                        self.state[p]["step"] += 1
        self._graph.replay()
        return loss

    @torch.no_grad()
    def _step_intrinsic(self):
        """The eager loop with the intrinsic-rate experiment. Kept verbatim because its failures are documented
        in place and are the useful part; it is not fused and not graphed."""
        for gp in self.param_groups:
            b1, b2 = gp["betas"]; eps = gp["eps"]
            for p in gp["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                st = self.state[p] or self._init_state(p, gp)
                st["step"] += 1
                if "vr" in st:
                    sq = g.float().square().add_(eps)
                    st["vr"].mul_(b2).add_(sq.mean(dim=1), alpha=1 - b2)
                    st["vc"].mul_(b2).add_(sq.mean(dim=0), alpha=1 - b2)
                    r = st["vr"] / st["vr"].mean().clamp_min(eps)
                    v = (r.unsqueeze(1) * st["vc"].unsqueeze(0)).clamp_min(eps)
                else:
                    st["v"].mul_(b2).addcmul_(g.float(), g.float(), value=1 - b2)
                    v = st["v"].clamp_min(eps)
                u = g.float() / v.sqrt()

                if gp["momentum"]:
                    st["m"].mul_(b1).add_(u.bfloat16(), alpha=1 - b1)
                    u = st["m"].float()

                if gp["weight_decay"]:
                    u = u.add(p.float(), alpha=gp["weight_decay"])

                # INTRINSIC RATE (vision goal 8), from gradient signal-to-noise.
                #
                # A FIRST DESIGN FAILED AND IS RECORDED HERE. It multiplied the rate by (1 + 0.02*cos(g,m))
                # every step. That compounds: over 1200 steps any persistent bias becomes 1.02^1200, so the
                # rate saturated at a bound within a few hundred steps and had no way back. Measured, it pinned
                # every tensor at the 1e-4 floor and the run went to NaN. A multiplicative update with no
                # restoring force is not an adaptation, it is a ratchet.
                #
                # This instead reads a BOUNDED quantity: the fraction of the gradient that is consistent
                # signal rather than noise. |EMA(u)|^2 / EMA(|u|^2) lies in [0,1] by construction, is 1 when
                # successive updates agree perfectly and near 0 when they are noise, and needs no schedule, no
                # global step counter and no hand-set rate. It also responds to a distribution shift on its
                # own, which is what continuous learning requires and what a cosine decay cannot do.
                if gp["intrinsic_lr"] and gp["momentum"]:
                    # THIRD DESIGN. The first two are recorded above and below because both failed and the
                    # failures are the useful part.
                    #
                    # Adadelta's insight: a step should have the UNITS of the parameter. The ratio of the
                    # accumulated size of past updates to the accumulated size of the current update supplies
                    # exactly that, so no rate needs to be chosen at all. Here it scales the trust fraction
                    # rather than the raw update, so scale-freedom is kept.
                    #
                    # Design 2 failed by having no dynamic range: |EMA(u)|^2 / EMA(|u|^2) sits near 1 for a
                    # momentum of an already normalised update, so every tensor pinned to the ceiling and the
                    # run went to NaN. This ratio is not bounded to [0,1] and does move.
                    # DESIGN 3 ALSO FAILED, and its failure is a self-referential fixed point: last_rel
                    # starts at 0, so the ratio starts at the floor, so last_rel becomes the floor, so the
                    # ratio stays at the floor. Adadelta escapes this because it uses the RMS of the actual
                    # PARAMETER UPDATE, which carries the gradient magnitude with it; collapsing it onto the
                    # scalar rate removes the only thing that could move it. Measured: pinned at 1e-4, NaN.
                    #
                    # STATUS: an intrinsic rate is NOT SOLVED. What works today is the trust ratio with a
                    # fixed fraction, which already removes the per-SCALE constant that wrecked the previous
                    # project, but the fraction itself is still hand-set. Do not enable this flag expecting it
                    # to work.
                    st["dx2"] = b1 * st.get("dx2", 0.0) + (1 - b1) * st.get("last_rel", 0.0) ** 2
                    st["u2"] = b1 * st.get("u2", 0.0) + (1 - b1) * float(u.float().square().mean())
                    lo, hi = gp["lr_bounds"]
                    raw = math.sqrt(st["dx2"] + 1e-12) / math.sqrt(st["u2"] + 1e-12)
                    st["lr"] = float(min(hi, max(lo, raw)))
                    step_lr = st["lr"]
                else:
                    step_lr = gp["lr"]

                if gp["trust"]:
                    wn = p.float().norm().clamp_min(gp["min_norm"])
                    un = u.norm().clamp_min(eps)
                    u = u * (wn / un).clamp(max=gp["trust_clip"])

                p.add_(u.to(p.dtype), alpha=-step_lr)
                if gp["intrinsic_lr"]:
                    # record the relative step actually taken; it is the numerator of the next ratio
                    st["last_rel"] = step_lr

    # ---------------------------------------------------------------------------------- growth support
    @torch.no_grad()
    def grew(self, p: torch.Tensor, old_shape: tuple[int, ...], fill: str = "median", replaces=None) -> None:
        """Tell the optimizer a parameter was enlarged, so new slots inherit statistics.

        `replaces` is the old Parameter object when the model made a new one (LM.grow does, because autograd
        caches a per-parameter accumulator with the creation-time shape): its state moves to `p` and it is
        swapped out of the param groups, so the groups keep their order and their settings.

        Zeroing them is the default elsewhere and it is wrong: the second moment appears in a denominator, so
        a zero means an unboundedly large first step on exactly the new capacity. New rows and columns take the
        MEDIAN of the existing ones, which is the optimizer's current opinion about this tensor rather than no
        opinion at all.

        `fill` exists so the alternative is testable rather than asserted. Note there is no do-nothing option:
        carrying state across a growth WITHOUT resizing it does not degrade gracefully, it raises, because the
        factored state is still the old width. Zeroing new slots is a deliberate act, not a default."""
        self._graph = None; self._warm = False   # state tensors change shape and address
        if replaces is not None:
            for gp in self.param_groups:
                gp["params"] = [p if q is replaces else q for q in gp["params"]]
            if replaces in self.state:
                self.state[p] = self.state.pop(replaces)
        st = self.state.get(p)
        if not st:
            return
        if "vr" in st:
            for k, n_new, dim in (("vr", p.shape[0], 0), ("vc", p.shape[1], 1)):
                old = st[k]
                if n_new > old.numel():
                    fv = 0.0 if fill == "zero" else float(
                        old.median() if old.numel() else torch.ones((), device=p.device))
                    new = torch.full((n_new,), fv, device=p.device, dtype=torch.float32)
                    new[: old.numel()] = old
                    st[k] = new
        elif "v" in st and st["v"].shape != p.shape:
            old = st["v"]
            fv = 0.0 if fill == "zero" else float(
                old.median() if old.numel() else torch.ones((), device=p.device))
            new = torch.full_like(p, fv, dtype=torch.float32)
            sl = tuple(slice(0, s) for s in old.shape)
            new[sl] = old
            st["v"] = new
        if "w0" in st and st["w0"].shape != p.shape:
            # DoG measures distance from the start; a grown tensor restarts that origin where it now stands (rbar,
            # the distance already covered, is kept, so its rate does not collapse)
            st["w0"] = p.detach().to(torch.bfloat16).clone()
        for k in ("m", "dsp"):                  # momentum and displacement: zero IS right, they are directions
            if k in st and st[k].shape != p.shape:
                old = st[k]
                new = torch.zeros_like(p, dtype=torch.bfloat16)
                sl = tuple(slice(0, s) for s in old.shape)
                new[sl] = old
                st[k] = new

    def state_bytes(self) -> int:
        n = 0
        for gp in self.param_groups:
            for p in gp["params"]:
                for k, v in self.state.get(p, {}).items():
                    if torch.is_tensor(v):
                        n += v.numel() * v.element_size()
        return n
