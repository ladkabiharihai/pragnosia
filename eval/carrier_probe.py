"""WHY DOES NO RETRIEVAL FLOW THROUGH THE CARRIER'S LONG TIMESCALES? (2026-10-04; PROGRESS.md) Two measurements before any
more training. Exact retrieval works only inside the attention window; past it, passkey and needles are 0%, and three
interventions failed (one long-range layer, a far-needle curriculum, a late short convolution).

  halflife  each carrier head's ACTUAL forgetting on real text: the per-token decay exp(la_t), la = -softplus(wa z) exp(A),
            as a half-life in tokens (median over tokens), and the write strength beta. If (almost) every head forgets
            within ~1k tokens, nothing can be retrieved past that, whatever the training.
  retain    two contexts identical except the needle's value (code A vs code B), then N tokens of real prose:
              STORED  how different the carrier states are just before the question, relative to their size, per layer
                      (zero = the value is gone: overwritten or decayed)
              READ    log p(code A | context A) - log p(code A | context B) at the answer, in nats per code (zero = the
                      model's prediction does not depend on the value it was shown)
            Stored but not read = a readout problem; not stored = a storage problem.
usage: eval/carrier_probe.py <ckpt> halflife|retain"""
import os, sys, math, random, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, "/opt/code/membrae")
from src.common import cap_gpu
cap_gpu(12.0)
import numpy as np, torch, torch.nn.functional as F
from tokenizers import Tokenizer
import src.assoc as A
from src.model import LM
A.FUSED_SCAN = True
CK, MODE = sys.argv[1], sys.argv[2]
tok = Tokenizer.from_file("/mnt/kv_cache/tok_v8192.json")
sd = torch.load(CK, map_location="cuda", weights_only=False)
m = LM(**sd["cfg"]).cuda(); m.load_state_dict(sd["model"]); m.eval()
for mx in m.mix:
    if hasattr(mx, "need_sent"): mx.need_sent = False
W = sd["cfg"]["ctx"]
car = [i for i, mx in enumerate(m.mix) if hasattr(mx, "theta")]
wiki = np.memmap("/mnt/kv_cache/tokfull_v8192/wikipedia.bin", dtype=np.uint16, mode="r")
rng = np.random.default_rng(5)
print(f"{CK}  window {getattr(m.mix[3], 'window', '?')}  carrier layers {car}", flush=True)

def prose(n):
    o = int(rng.integers(0, len(wiki) - 3 * n - 10)); f = np.asarray(wiki[o:o + 3 * n]); return f[f != 8191][:n].tolist()

if MODE == "halflife":
    grab = {}
    hooks = [m.mix[i].register_forward_pre_hook(lambda mod, inp, i=i: grab.__setitem__(i, inp[0].detach())) for i in car]
    las, bts = {i: [] for i in car}, {i: [] for i in car}
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for _ in range(4):
            x = torch.tensor([prose(W)], device="cuda")
            m.trunk(x)
            for i in car:
                mx = m.mix[i]; z = mx.ln(grab[i]).float()
                la = -F.softplus(F.linear(z, mx.wa.weight.float(), mx.wa.bias.float())) * torch.exp(mx.A.float())   # (1,T,H)
                las[i].append(la[0]); bts[i].append(torch.sigmoid(F.linear(z, mx.wb.weight.float(), mx.wb.bias.float()))[0])
    for h in hooks: h.remove()
    print("per carrier layer: each head's median half-life in tokens (ln 2 / -la), sorted; and its median write strength")
    allhl = []
    for i in car:
        la = torch.cat(las[i]); bt = torch.cat(bts[i])
        hl = (math.log(2) / (-la.float()).clamp_min(1e-9)).median(0).values.cpu().numpy()
        allhl += list(hl)
        bm = bt.float().median(0).values.cpu().numpy(); order = np.argsort(hl)
        print(f"  layer {i:2d}: (half-life, write strength) per head, by half-life: " +
              "  ".join(f"({hl[h]:.0f}, {bm[h]:.2f})" for h in order))
        pairs_all = globals().setdefault("pairs_all", []); pairs_all += [(hl[h], bm[h]) for h in range(len(hl))]
    allhl = np.array(allhl)
    for lim in (512, 1024, 4096, 16384):
        sel = [b for h_, b in pairs_all if h_ > lim]
        print(f"  heads with half-life > {lim:>6,} tokens: {len(sel)} of {len(allhl)}; their median write strength "
              f"{np.median(sel) if sel else float('nan'):.3f} (heads <= 512: {np.median([b for h_, b in pairs_all if h_ <= 512]):.3f})")
    sys.exit(0)

# retain / oracle. ORACLE: the long heads (median half-life > 1,024 tokens on prose) write ONLY on the needle's value
# tokens (write strength x 0 everywhere else, x 1 there); every other head unchanged. If READ past the attention window
# rises, overwriting by filler is what erases the needle (causal test of the overwrite explanation, no training).
LONG = {}
if MODE in ("oracle", "surprise"):
    grab = {}
    hooks = [m.mix[i].register_forward_pre_hook(lambda mod, inp, i=i: grab.__setitem__(i, inp[0].detach())) for i in car]
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        m.trunk(torch.tensor([prose(W)], device="cuda"))
    for h_ in hooks: h_.remove()
    for i in car:
        mx = m.mix[i]; z = mx.ln(grab[i]).float()
        la = -F.softplus(F.linear(z, mx.wa.weight.float(), mx.wa.bias.float())) * torch.exp(mx.A.float())
        LONG[i] = ((math.log(2) / (-la[0]).clamp_min(1e-9)).median(0).values > 1024)
    print("oracle long heads: " + " ".join(f"L{i}:{LONG[i].nonzero().flatten().tolist()}" for i in car))
GATE = None          # surprise mode: (B, L) 1.0 where the token is among the sequence's most surprising, else 0.0
def set_oracle(w0, Tw, vpos, B=2):
    for i in car:
        if MODE not in ("oracle", "surprise") or (MODE == "surprise" and GATE is None): m.mix[i].beta_scale = None; continue
        sc = torch.ones(B, LONG[i].numel(), Tw, device="cuda")
        if MODE == "oracle":
            sc[:, LONG[i], :] = 0.0
            for p_ in vpos:
                if w0 <= p_ < w0 + Tw: sc[:, LONG[i], p_ - w0] = 1.0
        else:
            sc[:, LONG[i], :] = GATE[:, None, w0:w0 + Tw]
        m.mix[i].beta_scale = sc
NEED = [("The secret code is {V}.", "Question: What is the secret code? Answer:"),
        ("The combination for the Omar lock is {V}.", "Question: What is the combination for the Omar lock? Answer:"),
        ("Remember this number: {V}.", "Question: What number were you asked to remember? Answer:")]
def code_pair():
    while True:
        a, b = str(rng.integers(100000, 999999)), str(rng.integers(100000, 999999))
        ta, tb = tok.encode(" " + a).ids, tok.encode(" " + b).ids
        if len(ta) == len(tb) and a != b: return ta, tb

print("distance = filler tokens between the needle and the question; 12 pairs (3 phrasings x 4) per distance")
print(f"{'distance':>9s}  {'READ (nats)':>12s}   STORED (relative state difference, per carrier layer)")
for N in ((1024, 2048, 4096, 8192, 16384) if MODE in ("oracle", "surprise") else (128, 512, 1024, 2048, 4096, 8192, 16384)):
    reads, stored = [], []
    GATE = None
    for k in range(12):
        st_, q_ = NEED[k % 3]; ta, tb = code_pair()
        if os.environ.get("RETAIN_SAME"): tb = ta             # control: identical rows -> the difference floor of the numerics
        pre = prose(512); fill = prose(N)
        na = tok.encode(" " + st_.split("{V}")[0].rstrip()).ids
        tail = tok.encode(st_.split("{V}")[1]).ids
        qi = tok.encode("\n\n" + q_).ids
        seqs = [pre + na + t + tail + fill + qi + ta for t in (ta, tb)]   # both rows answer with code A
        x = torch.tensor(seqs, device="cuda"); L = x.shape[1]
        qend = L - len(ta)                                    # answer tokens start here
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            st = None; lps = []
            vpos = list(range(len(pre) + len(na), len(pre) + len(na) + len(ta)))   # the needle's value tokens
            if MODE == "surprise":
                # pass 0: the model's own surprisal of every token, ordinary writes; then gate the long heads' writes on it
                GATE = None; st0 = None; sp = [torch.zeros(2, 1, device="cuda")]
                for w0 in range(0, L - 1, W):
                    set_oracle(w0, 0, vpos)
                    h0, st0 = m.trunk(x[:, w0:min(L - 1, w0 + W)], st0, True)
                    lp0 = F.linear(h0, m.e.weight).float().log_softmax(-1)
                    sp.append(-lp0.gather(-1, x[:, w0 + 1:min(L, w0 + W + 1)].unsqueeze(-1)).squeeze(-1))
                spr = torch.cat(sp, 1)[:, :L]                   # surprisal of token t (token 0: 0)
                q = torch.quantile(spr.float(), 1 - float(os.environ.get("SURPRISE_TOP", "0.05")), dim=1, keepdim=True)
                GATE = (spr >= q).float()
                if k == 0: print(f"   (N={N}) the code's tokens in the top {os.environ.get('SURPRISE_TOP', '0.05')}: "
                                 f"{int(GATE[0, vpos].sum())}/{len(vpos)}", flush=True)
            for w0 in range(0, L - 1, W):                      # pass 1: the whole sequence, as inference runs it
                set_oracle(w0, min(L - 1, w0 + W) - w0, vpos)
                h, st = m.trunk(x[:, w0:min(L - 1, w0 + W)], st, True)
                lps.append(F.linear(h, m.e.weight).float().log_softmax(-1))
            st = None; qs = qend - len(qi)                    # pass 2: the same windows up to the question's start
            for w0 in range(0, qs, W):
                set_oracle(w0, min(qs, w0 + W) - w0, vpos)
                _, st = m.trunk(x[:, w0:min(qs, w0 + W)], st, True)
            S_before = [st[i][0].float() for i in car]
            lp = torch.cat(lps, 1)                            # predicts x[:, 1:]
            y = x[:, qend:]                                    # answer tokens (A in row 0, B in row 1)
            lpA_ctxA = lp[0, qend - 1:L - 1].gather(-1, y[0].unsqueeze(-1)).sum()
            lpA_ctxB = lp[1, qend - 1:L - 1].gather(-1, y[0].unsqueeze(-1)).sum()
            reads.append(float(lpA_ctxA - lpA_ctxB))
            stored.append([float((s[0] - s[1]).norm() / s[0].norm().clamp_min(1e-9)) for s in S_before])
    r = np.array(reads); s = np.array(stored).mean(0)
    print(f"{N:>9,d}  {r.mean():+8.3f} +-{1.96 * r.std(ddof=1) / math.sqrt(len(r)):5.3f}   " +
          " ".join(f"L{i}:{v:.3f}" for i, v in zip(car, s)), flush=True)
