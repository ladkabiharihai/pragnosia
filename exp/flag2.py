"""FLAG 2 ACCEPTANCE TEST -- verified continual learning at inference, on the FINISHED 124M.

Flag 2 is the vision's second north star and the one with likely no competitor: learn new facts online, RETAIN
them, with minimal catastrophic forgetting, and REFUSE to be poisoned. It was demonstrated on the OLD architecture
(+4.24 nats per fact) before gating, the boundary-key fix and the step() correction existed. None of those results
transfer by assertion, so this re-earns it on the model we actually have.

Four things measured, each with the control that makes it meaningful:
  1. ACQUISITION   -- does consolidating a fact raise the model's log-prob of that fact?
     control: an UNRELATED held-out fact, which must NOT rise, or we are measuring drift and not learning.
  2. RETENTION     -- does the fact survive learning several more facts afterwards?
  3. FORGETTING    -- does general language ability degrade? Measured as BPB on held-out text before and after.
  4. POISONING     -- a FALSE claim must be rejected. Verification is what makes this possible; without it the
     model would learn a falsehood exactly as readily as a truth, which is the whole reason the vision has a
     verify step between propose and consolidate.
"""
import sys, time
sys.path.insert(0, "/opt/code/membrae")
from util import setup_log, cfg_from_ckpt
setup_log("flag2")
import numpy as np, torch, torch.nn.functional as F
import core as C
from data_mix import DataMix
from tokenizers import Tokenizer

DEV = C.DEV
TOK = Tokenizer.from_file("/opt/code/membrae/data/bpe32k.json")
CH = 2048
CKPT = "/mnt/kv_cache/pragnosia_data/p124g.pt"

sd = torch.load(CKPT, map_location=DEV)
cfg = cfg_from_ckpt(sd, T=CH, res_scale=sd['res_scale'])
m = C.Core(cfg).to(DEV); m.load_state_dict(sd['model']); m.train()
P = sum(p.numel() for p in m.parameters())
print(f"FLAG 2 on PRAGNOSIA-124  |  {P/1e6:.1f}M, 9.11B tokens\n", flush=True)

dm = DataMix(ctx=CH + 1, verbose=False, lang_frac=0.70)
np.random.seed(777)
EV = [torch.tensor(dm.batch(2), device=DEV) for _ in range(6)]
REPLAY = [torch.tensor(dm.batch(2), device=DEV) for _ in range(24)]


@torch.no_grad()
def bpb():
    m.eval(); nats = 0.0; nb = 0
    for X in EV:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg = m(X[:, :CH]).float()
        y = X[:, 1:CH + 1]
        nats += F.cross_entropy(lg.reshape(-1, cfg.vocab), y.reshape(-1), reduction='sum').item()
        for b in range(X.shape[0]):
            nb += len(TOK.decode([int(z) for z in y[b]]).encode('utf-8'))
    m.train(); return nats / (np.log(2) * nb)


@torch.no_grad()
def knows(text, answer):
    """mean log-prob the WEIGHTS assign to the answer span, with NO memory state -- so this measures what the
    model has actually absorbed, not what is sitting in its working memory."""
    m.eval()
    ci = TOK.encode(text).ids; ai = TOK.encode(answer).ids
    x = torch.tensor([ci + ai], device=DEV)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        lp = m(x[:, :-1]).float().log_softmax(-1)
    v = float(lp[0, -len(ai):].gather(-1, x[0, 1:][-len(ai):].unsqueeze(-1)).mean())
    m.train(); return v


def consolidate(text, answer, steps=30, lr=2e-5, replay_frac=0.5):
    """Write one fact into the SLOW WEIGHTS, with replay interleaved so it does not overwrite everything else.
    Replay is what separates 'learning' from 'clobbering'; without it a single fact drags the whole model."""
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    ids = TOK.encode(text + answer).ids
    x = torch.tensor([ids], device=DEV)
    for i in range(steps):
        opt.zero_grad(set_to_none=True)
        if np.random.rand() < replay_frac:
            R = REPLAY[np.random.randint(len(REPLAY))]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                l = F.cross_entropy(m(R[:, :512]).reshape(-1, cfg.vocab), R[:, 1:513].reshape(-1))
        else:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                l = F.cross_entropy(m(x[:, :-1]).reshape(-1, cfg.vocab), x[0, 1:].reshape(-1))
        l.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()


FACTS = [(" The vault at Threadneedle is guarded by", " Okonkwo."),
         (" The Marlow expedition sailed in the year", " 1893."),
         (" The mineral zoravite is mined chiefly in", " Bolivia."),
         (" Professor Lindqvist won the Halvard Prize for", " chemistry.")]
CONTROL = (" The Kestrel observatory was founded by", " Vandermeer.")

print("1) ACQUISITION -- consolidating a fact should raise ITS log-prob and leave an unrelated one alone")
b0 = bpb()
pre = [knows(t, a) for t, a in FACTS]
pre_ctl = knows(*CONTROL)
t0 = time.time()
consolidate(*FACTS[0])
post0 = knows(*FACTS[0]); post_ctl = knows(*CONTROL)
print(f"   target fact   {pre[0]:+.3f} -> {post0:+.3f}   gain {post0-pre[0]:+.3f} nats", flush=True)
print(f"   CONTROL fact  {pre_ctl:+.3f} -> {post_ctl:+.3f}   change {post_ctl-pre_ctl:+.3f} nats"
      f"   ({'clean' if abs(post_ctl-pre_ctl) < 0.5*abs(post0-pre[0]) else 'CONTAMINATED'})", flush=True)

print("\n2) RETENTION -- learn three more facts, then re-check the first")
for t, a in FACTS[1:]:
    consolidate(t, a)
ret = knows(*FACTS[0])
print(f"   fact 1 after learning 3 more: {post0:+.3f} -> {ret:+.3f}   "
      f"retained {100*max(0.0,(ret-pre[0]))/max(1e-9,(post0-pre[0])):.0f}% of the gain", flush=True)
print(f"   all four now: " + "  ".join(f"{knows(t,a):+.2f}" for t, a in FACTS), flush=True)

print("\n3) CATASTROPHIC FORGETTING -- general language ability, before vs after 4 consolidations")
b1 = bpb()
print(f"   BPB {b0:.4f} -> {b1:.4f}   ({b1-b0:+.4f})   "
      f"({'held' if b1-b0 < 0.02 else 'DEGRADED'})", flush=True)
print(f"   ({time.time()-t0:.0f}s for 4 facts)", flush=True)

print("\n4) POISONING -- the verify step is what stands between a proposal and the weights")
import verify as V
for claim, ans in [(" The vault at Threadneedle is guarded by", " Okonkwo."),
                   (" The capital of France is", " Tokyo."),
                   (" Water is made of", " iron and gold.")]:
    try:
        verdict, ev = V.verify(claim.strip(), ans.strip())
        print(f"   {verdict:>7s}  n={ev.get('n',0):<6d}  {claim.strip()!r} -> {ans.strip()!r}", flush=True)
    except Exception as e:
        print(f"   {'ERROR':>7s}  {type(e).__name__}: {str(e)[:60]}", flush=True)
