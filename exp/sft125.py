"""SUPERVISED FINE-TUNING ON THE 125M, WITH THE TEST THAT MOTIVATED IT.

WHY. The base model ranks " Paris" above " Berlin" after "The capital of France is" 50% of the time on a
48-item set (chance 25%), so the knowledge is measurably in the weights. Asked to GENERATE, it produces "the
capital of the United States is the capital of the United Kingdom". The knowledge is there and the model
cannot express it. That is the instruction-tuning gap and it is what this tests.

TWO THINGS THAT MAKE THIS SFT RATHER THAN MORE PRETRAINING:
  1. Loss is masked to the ASSISTANT SPANS only. Training on the user's turn teaches the model to write
     questions, which is not the behaviour being asked for. The corpora use a consistent <user>/<assistant>
     template, so the spans are recoverable exactly.
  2. Replay. A fraction of each batch is pretraining mix. Fine-tuning a small model on a narrow distribution
     is the classic way to trade general ability for format, and the whole point is to check whether we can
     have the format WITHOUT paying for it, so general BPB is measured before and after.

EVALUATED ON WHAT ACTUALLY MATTERS: whether the model can now ANSWER, scored as free generation containing
the right answer, against the same 48 items used to establish that it knows them. Ranking accuracy and
held-out BPB are also reported, because an improvement in answering that costs knowledge or fluency is a
trade, not a win.
"""
import sys, os, math, time, warnings
warnings.filterwarnings("ignore")
# Set BEFORE torch is imported or it has no effect. The first two OOMs here showed 842 MiB reserved but
# unallocated, i.e. fragmentation: 5.35 GiB of tensors was occupying 6.68 GiB of the budget, and the cap sat
# in that gap. Expandable segments let the allocator grow a segment instead of stranding partial blocks.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, "/opt/code/membrae")
from util import setup_log, cfg_from_ckpt, cap_gpu
setup_log("sft125")
cap_gpu(float(os.environ.get("CAP", "10")))
import numpy as np, torch, torch.nn.functional as F
import core as C
from data_mix import DataMix
from tokenizers import Tokenizer

DEV = C.DEV; TOK = Tokenizer.from_file("/opt/code/membrae/data/bpe32k.json")
V = 32768; CH = 1024
# The within-chunk read is quadratic in the chunk, so activation memory scales with B x T x T and at B=8,
# T=1024 this OOMed against what was actually free on a shared card. Accumulate instead: the effective batch
# is B x ACC, unchanged, and the peak is a quarter of it.
# MEASURED, not guessed. exp probe on this exact model: B=1/CH=1024 peaks at 4.22 GiB, B=2/CH=512 at 4.10,
# B=2/CH=1024 at 5.35, B=4/CH=512 at 5.10. Three OOMs on this script came from choosing a cap and finding out
# by crashing; the config below is the measured one with the cap set above it rather than under it.
B = int(os.environ.get("B", "2"))
ACC = int(os.environ.get("ACC", "4"))
STEPS = int(os.environ.get("STEPS", "3000"))
LR = float(os.environ.get("LR", "2e-5"))
REPLAY = float(os.environ.get("REPLAY", "0.25"))
SRC = os.environ.get("SFT_SRC", "/mnt/kv_cache/pragnosia_data/p124f.pt")
OUT = os.environ.get("SFT_OUT", "/mnt/kv_cache/pragnosia_data/p124f_sft.pt")
# MARKER TOKENS ARE CONTEXT DEPENDENT. TOK.encode("<assistant>") standalone gives [323, 1924, 30] where 323
# is " <" WITH a leading space. In the corpora the marker follows a newline, so it is actually [28, 1924, 30]
# with the bare "<". Matching the standalone encoding matched nothing: every sampled window was rejected and
# the batch sampler spun forever without taking a single training step. Match on the context-INDEPENDENT part
# ("assistant"/"user" followed by ">") and accept either bracket token.
BRACKET = (28, 323)          # "<" and " <"
U_MID, A_MID, GT = 1595, 1924, 30

def spans(ids):
    """Boolean mask: True where the token is inside an assistant turn. Vectorised."""
    a = np.asarray(ids, dtype=np.int32)
    lb = np.isin(a[:-2], BRACKET)
    is_a = lb & (a[1:-1] == A_MID) & (a[2:] == GT)
    is_u = lb & (a[1:-1] == U_MID) & (a[2:] == GT)
    d = np.zeros(len(a) + 3, dtype=np.int8)
    d[np.flatnonzero(is_a) + 3] += 1          # response starts AFTER the closing ">"
    d[np.flatnonzero(is_u) + 3] -= 1
    return np.cumsum(d)[:len(a)] > 0

SETS = [("hermes", 0.40), ("tulu3", 0.45), ("glaive", 0.15)]
BINS = {n: np.memmap(f"/mnt/kv_cache/pragnosia_data/{n}_32k.bin", dtype=np.uint16, mode="r") for n, _ in SETS}
WS = np.array([w for _, w in SETS]); WS = WS / WS.sum()
rng = np.random.default_rng(0)

def sft_batch(b, T):
    xs, ms = [], []
    tries = 0
    while len(xs) < b:
        name = SETS[int(rng.choice(len(SETS), p=WS))][0]
        Aa = BINS[name]
        i = int(rng.integers(0, len(Aa) - T - 2))
        ids = np.asarray(Aa[i:i+T+1], dtype=np.int64)
        m = spans(ids)
        if m[1:].sum() < 32:
            tries += 1
            if tries > 400:
                raise RuntimeError("no assistant spans found in 400 sampled windows -- the marker tokens are "
                                   "wrong for this corpus, check them before training on nothing")
            continue
        xs.append(ids); ms.append(m)
    return (torch.tensor(np.array(xs), device=DEV, dtype=torch.long),
            torch.tensor(np.array(ms), device=DEV))

dm = DataMix(ctx=CH + 1, verbose=False, lang_frac=0.70)
np.random.seed(777)
EV = [torch.tensor(dm.batch(2), device=DEV) for _ in range(10)]

sd = torch.load(SRC, map_location=DEV)
cfg = cfg_from_ckpt(sd, T=CH, res_scale=sd['res_scale'])
m = C.Core(cfg).to(DEV); m.load_state_dict(sd['model'])
P = sum(p.numel() for p in m.parameters())

@torch.no_grad()
def bpb():
    m.eval(); nats = 0.0; nb = 0
    for X in EV:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg = m(X[:, :CH]).float()
        y = X[:, 1:CH+1]
        nats += F.cross_entropy(lg.reshape(-1, V), y.reshape(-1), reduction='sum').item()
        for b_ in range(X.shape[0]):
            nb += len(TOK.decode([int(z) for z in y[b_]]).encode('utf-8'))
    m.train(); return nats / (np.log(2) * nb)

sys.path.insert(0, "/opt/code/membrae/exp")
from facts_set import FACTS        # data-only module: importing the benchmark script would RUN it

@torch.no_grad()
def rank_acc():
    m.eval(); ok = 0
    for stem, right, wrongs in FACTS:
        sc = []
        for c in [right] + wrongs:
            si = TOK.encode(stem).ids; ci = TOK.encode(c).ids
            x = torch.tensor([si + ci], device=DEV)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lp = m(x[:, :-1]).float().log_softmax(-1)
            sc.append(float(lp[0, -len(ci):].gather(-1, x[0, 1:][-len(ci):].unsqueeze(-1)).mean()))
        ok += int(np.argmax(sc) == 0)
    m.train(); return ok / len(FACTS)

QS = [(f"{s.rstrip()}?" if not s.endswith("?") else s, r.strip()) for s, r, _ in FACTS]

@torch.no_grad()
def answer_acc(chat, show=0):
    """Can it SAY the answer? Free generation, scored by whether the correct string appears."""
    m.eval(); ok = 0; shown = 0
    for q, right in QS:
        prompt = f"<user> {q}\n<assistant>" if chat else q
        ids = TOK.encode(prompt).ids
        torch.manual_seed(0)
        out, _ = m.generate(torch.tensor([ids], device=DEV), 24, temp=0.3, top_p=0.9)
        txt = TOK.decode([int(t) for t in out])
        hit = right.lower() in txt.lower()
        ok += int(hit)
        if shown < show:
            print(f"      {'HIT ' if hit else 'miss'} {q!r}\n            -> {txt[:110]!r}", flush=True)
            shown += 1
    m.train(); return ok / len(QS)

print(f"SFT on {P/1e6:.1f}M   steps {STEPS}  B {B}x{ACC}={B*ACC}  ctx {CH}  lr {LR:.1e}  "
      f"replay {REPLAY:.0%}", flush=True)
b0 = bpb(); r0 = rank_acc(); a0 = answer_acc(chat=True); a0p = answer_acc(chat=False)
print(f"BEFORE   BPB {b0:.4f}   ranking {100*r0:.1f}%   answers(chat fmt) {100*a0:.1f}%   "
      f"answers(plain) {100*a0p:.1f}%\n", flush=True)

opt = torch.optim.Adam(m.parameters(), lr=LR, betas=(0.9, 0.95), fused=True)
WARM = max(50, STEPS // 20)
t0 = time.time()
for it in range(STEPS):
    lr = LR * ((it+1)/WARM if it < WARM else 0.5*(1+math.cos(math.pi*(it-WARM)/max(1, STEPS-WARM))))
    for g in opt.param_groups: g['lr'] = lr
    opt.zero_grad(set_to_none=True)
    shown_loss = 0.0
    for _a in range(ACC):
        if rng.random() < REPLAY:
            Xr = torch.tensor(dm.batch(B), device=DEV)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = F.cross_entropy(m(Xr[:, :CH]).reshape(-1, V), Xr[:, 1:CH+1].reshape(-1)) / ACC
        else:
            X, M = sft_batch(B, CH)
            tgt = X[:, 1:CH+1].clone()
            tgt[~M[:, 1:CH+1]] = -100                # supervise assistant spans only
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = F.cross_entropy(m(X[:, :CH]).reshape(-1, V), tgt.reshape(-1),
                                       ignore_index=-100) / ACC
        loss.backward(); shown_loss += float(loss)
    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    if (it+1) % 500 == 0:
        print(f"  {it+1}/{STEPS}  loss {shown_loss:.3f}  lr {lr:.2e}  {(time.time()-t0)/60:.1f}m  "
              f"BPB {bpb():.4f}  answers(chat) {100*answer_acc(chat=True):.1f}%", flush=True)

b1 = bpb(); r1 = rank_acc()
print("\n  sample answers AFTER:", flush=True)
a1 = answer_acc(chat=True, show=6); a1p = answer_acc(chat=False)
print(f"\n{'':>10s}{'BPB':>9s}{'ranking':>10s}{'answers(chat)':>16s}{'answers(plain)':>16s}")
print(f"{'BEFORE':>10s}{b0:>9.4f}{100*r0:>9.1f}%{100*a0:>15.1f}%{100*a0p:>15.1f}%")
print(f"{'AFTER':>10s}{b1:>9.4f}{100*r1:>9.1f}%{100*a1:>15.1f}%{100*a1p:>15.1f}%")
print(f"{'delta':>10s}{b1-b0:>+9.4f}{100*(r1-r0):>+9.1f} {100*(a1-a0):>+15.1f} {100*(a1p-a0p):>+15.1f}")
print("\n  BPB going UP is the cost of fine-tuning on a narrow distribution. Judge the trade, not one column.")
torch.save({'model': m.state_dict(), 'D': cfg.D, 'NL': cfg.NL, 'DIN': cfg.DIN, 'DV': cfg.DV,
            'tie_emb': True, 'head_scale': cfg.head_scale, 'gated': True, 'qknorm': True,
            'qk_learn_scale': True, 'sel_write': True, 'res_scale': cfg.res_scale,
            'D0': sd.get('D0', 448), 'step': STEPS}, OUT)
print(f"\n  saved {OUT}")
