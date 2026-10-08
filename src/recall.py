"""RECALL: EXACT RETRIEVAL PAST THE ATTENTION WINDOW, NO RETRAINING (2026-10-05; user: "we need retrieval beyond 1k").

The carrier's long heads write non-selectively, so filler overwrites a needle within ~1-2k tokens (eval/carrier_probe.py):
past the 1,024-token window exact retrieval is ~0%. The oracle showed the readout itself works when writes are selective.
This gives the sliding attention layers what the carrier cannot keep: every key and value that has slid out of the
window stays in a store, and each query also attends to its own top-k far keys, chosen by the same q.k score the layer
already computes (Memorizing Transformers, Wu et al. 2022; InfLLM, Xiao et al. 2024, without the block approximation).
The attention has no rotary positions (position enters once, at the input, and restarts every window), so a far key needs
no re-indexing: it is scored exactly as the layer scores a key inside its window.

  softmax over  { the keys inside the window }  U  { the query's top-k keys older than the window }

k = 0 is the trained layer exactly (checked by eval/recall_test.py exact). The store is the cost: 2 x d x 2 bytes per token
per attention layer (9 KB/token for run85's three layers at d768; 300 MB at 32k tokens), and the far scoring is linear in
the store per query. Inference only: the weights never see it.
usage: from src.recall import enable; enable(model, k)       k = 0 turns it off"""
import torch, torch.nn.functional as F
import src.assoc as _A

CH = 256                    # queries per scoring chunk: B x heads x CH x store floats live at once


def _qkv(at, x):
    B_, T_, D = x.shape; z = at.n(x)
    if _A.FUSED_PROJ:
        y = F.linear(z, torch.cat([at.q.weight, at.k.weight, at.v.weight]), torch.cat([at.q.bias, at.k.bias, at.v.bias]))
        q, k, v = [t.reshape(B_, T_, at.h, D // at.h).transpose(1, 2) for t in y.split(D, dim=-1)]
    else:
        q, k, v = [t(z).view(B_, T_, at.h, D // at.h).transpose(1, 2) for t in (at.q, at.k, at.v)]
    return z, at.qn(q), at.kn(k), v


def _forward(at, x, h0=None, mem=None):
    B_, T_, D = x.shape
    z, q, k, v = _qkv(at, x)
    if h0 is None: at._fk = at._fv = None                 # a new sequence: nothing stored
    ks = [t for t in (at._fk, None if h0 is None else h0[0], k) if t is not None]
    vs = [t for t in (at._fv, None if h0 is None else h0[1], v) if t is not None]
    K = torch.cat(ks, 2); V = torch.cat(vs, 2); L = K.shape[2]; w = at.window; top = at.ref_k
    out = torch.empty_like(q); sc = q.shape[-1] ** -0.5
    j = torch.arange(L, device=x.device)
    Kf, Vf = K.float(), V.float()
    for c0 in range(0, T_, CH):
        qc = q[:, :, c0:c0 + CH].float(); n = qc.shape[2]
        d = (L - T_ + c0 + torch.arange(n, device=x.device)).unsqueeze(1) - j.unsqueeze(0)     # distance back, (n, L)
        allow = (d >= 0) & (d < w)
        S = (qc @ Kf.transpose(-1, -2)) * sc                                                  # (B, H, n, L)
        far = d >= w
        if top and bool(far.any()):
            Sf = S.masked_fill(~far, float("-inf"))
            thr = Sf.topk(min(top, L), dim=-1).values[..., -1:]       # the k-th best far score of each query and head
            ok = far & (Sf >= thr) & torch.isfinite(Sf)
            if at.recall_gate:
                # ONLY A FAR KEY THAT BEATS EVERYTHING IN THE WINDOW: no parameter. Ungated, the far keys took softmax mass
                # on ordinary text and the loss rose +0.05-0.07 nats (the layer never saw them in training); a needle's key
                # is the best match there is for its question, so it still gets in
                ok = ok & (Sf > S.masked_fill(~allow, float("-inf")).amax(-1, keepdim=True))
            allow = allow | ok
        out[:, :, c0:c0 + n] = (S.masked_fill(~allow, float("-inf")).softmax(-1) @ Vf).to(q.dtype)
    keep = max(0, L - w)
    at._fk, at._fv = K[:, :, :keep], V[:, :, :keep]            # everything older than the window
    return x + at.o(at._state_read(out, z, None, B_, T_, D)), (K[:, :, keep:], V[:, :, keep:])


def enable(m, k, gate=False):
    """THE MODEL'S OWN RECALL (src/model.py Attn._recall: the same function training uses), the whole past kept (span 0).
    gate=True, or RECALL_REF=1, falls back to the explicit reference below (the first measurements, 2026-10-05)."""
    import os
    if not gate and not os.environ.get("RECALL_REF"):
        for at in m.mix: at.__dict__.pop("forward", None)      # drop the reference's patch, if any
        m._no_graph = False
        m.set_recall(k, span=0)
        # RECALL_OFFLOAD=1: the store in host memory (model.py FarStore), GPU memory independent of the length read
        for at in m.mix: at.recall_offload = bool(os.environ.get("RECALL_OFFLOAD"))
        return sum(1 for at in m.mix if getattr(at, "recall_k", 0))
    return enable_ref(m, k, gate)


def enable_ref(m, k, gate=False):
    m.recall_k = k; m._no_graph = True                         # the reference's store grows and it does not mask the graphed
                                                               # decode's zero-padded ring: never the graphed decode
    """Patch every sliding attention layer of LM m to recall its top-k far keys (k = 0: the trained layer, no store);
    gate: only far keys that score above every key inside the window"""
    n = 0
    for at in m.mix:
        if type(at).__name__ == "Attn" and getattr(at, "sliding", False) and at.window:
            assert not at.read_state
            at.recall_k = 0; at.ref_k = k; at.recall_gate = gate; at._fk = at._fv = None   # recall_k 0: the model's own path off
            at.forward = (lambda x, h0=None, mem=None, at=at: _forward(at, x, h0, mem))
            n += 1
    return n


def store_bytes(m):
    return sum(t.numel() * t.element_size() for at in m.mix for t in (getattr(at, "_fk", None), getattr(at, "_fv", None))
               if t is not None)
