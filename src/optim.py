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
from torch.optim import Optimizer


class Fabric(Optimizer):
    """Factored second moment + trust-ratio step.

    Args:
        lr: the trust ratio when `trust=True`, i.e. the fraction of its own norm a parameter moves per step.
            With `trust=False` it is an ordinary learning rate.
        betas: (momentum, second-moment) decay.
        trust: scale-free stepping. Off, this is close to Adafactor with momentum.
        factored: factor the second moment for 2D parameters. Off, this stores v densely like Adam.
        momentum: keep a first moment. Off, this halves state again at some cost in stability.
    """

    def __init__(self, params, lr=2e-3, betas=(0.9, 0.99), eps=1e-30, weight_decay=0.0,
                 trust=True, factored=True, momentum=True, trust_clip=1e4, min_norm=1e-3):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                                      trust=trust, factored=factored, momentum=momentum,
                                      trust_clip=trust_clip, min_norm=min_norm))

    # ---------------------------------------------------------------------------------- second moment
    @staticmethod
    def _factored(p, g, st, b2, eps):
        """v ~ outer(row, col). Store O(R+C) instead of O(R*C)."""
        sq = g.float().square().add_(eps)
        st["vr"].mul_(b2).add_(sq.mean(dim=1), alpha=1 - b2)
        st["vc"].mul_(b2).add_(sq.mean(dim=0), alpha=1 - b2)
        # reconstruct: outer product renormalised by the row mean, which is the Adafactor estimator
        r = st["vr"] / st["vr"].mean().clamp_min(eps)
        return (r.unsqueeze(1) * st["vc"].unsqueeze(0)).clamp_min(eps)

    @staticmethod
    def _dense(p, g, st, b2, eps):
        st["v"].mul_(b2).addcmul_(g.float(), g.float(), value=1 - b2)
        return st["v"].clamp_min(eps)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for gp in self.param_groups:
            b1, b2 = gp["betas"]; eps = gp["eps"]
            for p in gp["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                st = self.state[p]
                if not st:
                    st["step"] = 0
                    fac = gp["factored"] and p.dim() == 2
                    if fac:
                        st["vr"] = torch.zeros(p.shape[0], device=p.device, dtype=torch.float32)
                        st["vc"] = torch.zeros(p.shape[1], device=p.device, dtype=torch.float32)
                    else:
                        st["v"] = torch.zeros_like(p, dtype=torch.float32)
                    if gp["momentum"]:
                        st["m"] = torch.zeros_like(p, dtype=torch.bfloat16)
                st["step"] += 1

                fac = "vr" in st
                v = self._factored(p, g, st, b2, eps) if fac else self._dense(p, g, st, b2, eps)
                u = g.float() / v.sqrt()

                if gp["momentum"]:
                    st["m"].mul_(b1).add_(u.bfloat16(), alpha=1 - b1)
                    u = st["m"].float()

                if gp["weight_decay"]:
                    u = u.add(p.float(), alpha=gp["weight_decay"])

                if gp["trust"]:
                    # Move a fixed fraction of the parameter's OWN norm, so |dp| = lr * |w| exactly and the
                    # step is invariant to how the weights happen to be scaled.
                    #
                    # trust_clip defaults high enough not to bind. A tight clip (10.0 was tried) destroys the
                    # very invariance this exists for: measured, a parameter at norm 100 took a relative step
                    # of 0.0010 where one at norm 1 took 0.0100, because the ratio hit the clip. The runaway
                    # case a clip is meant to catch, |u| near zero, is already handled by clamp_min(eps): the
                    # resulting step has norm lr*|w| no matter how small |u| is.
                    #
                    # min_norm lets zero-initialised capacity move at all. Without it a freshly grown tensor
                    # has |w| = 0, so the step is 0, and new capacity would be frozen forever.
                    wn = p.float().norm().clamp_min(gp["min_norm"])
                    un = u.norm().clamp_min(eps)
                    u = u * (wn / un).clamp(max=gp["trust_clip"])

                p.add_(u.to(p.dtype), alpha=-gp["lr"])
        return loss

    # ---------------------------------------------------------------------------------- growth support
    @torch.no_grad()
    def grew(self, p: torch.Tensor, old_shape: tuple[int, ...]) -> None:
        """Tell the optimizer a parameter was enlarged in place, so new slots inherit statistics.

        Zeroing them is the default elsewhere and it is wrong: the second moment appears in a denominator, so
        a zero means an unboundedly large first step on exactly the new capacity. New rows and columns take the
        MEDIAN of the existing ones, which is the optimizer's current opinion about this tensor rather than no
        opinion at all."""
        st = self.state.get(p)
        if not st:
            return
        if "vr" in st:
            for k, n_new, dim in (("vr", p.shape[0], 0), ("vc", p.shape[1], 1)):
                old = st[k]
                if n_new > old.numel():
                    fill = old.median() if old.numel() else torch.ones((), device=p.device)
                    new = torch.full((n_new,), float(fill), device=p.device, dtype=torch.float32)
                    new[: old.numel()] = old
                    st[k] = new
        elif "v" in st and st["v"].shape != p.shape:
            old = st["v"]
            fill = old.median() if old.numel() else torch.ones((), device=p.device)
            new = torch.full_like(p, float(fill), dtype=torch.float32)
            sl = tuple(slice(0, s) for s in old.shape)
            new[sl] = old
            st["v"] = new
        if "m" in st and st["m"].shape != p.shape:
            old = st["m"]
            new = torch.zeros_like(p, dtype=torch.bfloat16)   # momentum: zero IS right, it is a direction
            sl = tuple(slice(0, s) for s in old.shape)
            new[sl] = old
            st["m"] = new

    def state_bytes(self) -> int:
        n = 0
        for gp in self.param_groups:
            for p in gp["params"]:
                for k, v in self.state.get(p, {}).items():
                    if torch.is_tensor(v):
                        n += v.numel() * v.element_size()
        return n
