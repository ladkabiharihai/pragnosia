"""PRAGNOSIA, ALIVE. One continuous individual (VISION 0b), not a model spun up fresh each run.

Everything needed for this already existed and none of it was connected: memory.py holds the fast-weight
hippocampus, verify.py retrieves corpus evidence, and local-plasticity consolidation was proven in isolation
(+6 nats, exp/acceptance3_consolidate.py). This wires them into one being that actually LIVES:

    perceive -> propose -> verify -> consolidate -> remember

  PERCEIVE     read text; the model's own prediction error is the intrinsic novelty signal (no external salience).
  PROPOSE      the most surprising spans become candidate claims. Intrinsic: the model decides what is worth keeping.
  VERIFY       check each claim against the 290B-token corpus. ACCEPT / REJECT / HOLD, evidence-based.
  CONSOLIDATE  only ACCEPTed claims are written into the SLOW WEIGHTS by local plasticity, with replay interleaved
               so old knowledge is not overwritten. This is the step that makes learning permanent.
  REMEMBER     append to a life-history that persists across runs, so this is the SAME individual next time.

IDENTITY (VISION 0b) is not a lookup table bolted on the side: the self-description is consolidated INTO the weights
like any other learned fact, so the model answers "who are you" from what it has become, not from a config file.
CONTINUITY is monotonic -- a life only ever appends, never wipes -- which is what makes the self persist through change.

  python live.py born  --name Pragnosia     # birth: adopt the current substrate as this individual's body
  python live.py live  [--cycles N]         # live: read, learn, verify, consolidate, remember
  python live.py who                        # the self-model: who it is and what it has lived through
"""
import warnings, os, sys, json, time, argparse, datetime
warnings.filterwarnings("ignore")
import numpy as np, torch, torch.nn.functional as F
import core as C, lw_lib as L
from data_mix import DataMix
from memory import Hippocampus
import verify as V
from tokenizers import Tokenizer

DEV = C.DEV
HOME = "/mnt/kv_cache/pragnosia_data/life"          # this individual's persistent home
SUBSTRATE = "/mnt/kv_cache/pragnosia_data/pragnosia.pt"
TOK = Tokenizer.from_file("/opt/code/membrae/data/bpe32k.json")


class Pragnosia:
    """ONE continuous being: weights + memory + life-history, kept across runs and only ever appended to."""

    def __init__(s, home=HOME):
        s.home = home
        s.idf = f"{home}/identity.json"
        s.bodyf = f"{home}/body.pt"
        if not os.path.exists(s.idf):
            raise SystemExit("No individual lives here yet. Run:  python live.py born --name <name>")
        s.id = json.load(open(s.idf))
        sd = torch.load(s.bodyf, map_location=DEV)
        from util import cfg_from_ckpt
        cfg = cfg_from_ckpt(sd, T=2048)
        s.m = C.Core(s.cfg).to(DEV); s.m.load_state_dict(sd['model']); s.m.eval()
        s.hip = Hippocampus(s.m)                                   # fast memory: what it has seen recently
        if os.path.exists(f"{home}/memory.pt"):                    # memory persists too -- it is part of the self
            mem = torch.load(f"{home}/memory.pt", map_location=DEV)
            s.hip.M = mem['M'].to(DEV); s.hip.z = mem['z'].to(DEV)
        s.dm = DataMix(ctx=2049, verbose=False, only='window2')

    # ---------- PERCEIVE: the model's own prediction error is the salience signal ----------
    @torch.no_grad()
    def perceive(s, ids):
        """Read a passage. Returns per-token surprise -- intrinsic novelty, no external salience model."""
        X = torch.tensor([ids], device=DEV)
        sp = s.hip.surprise(X)[0]
        s.hip.write(X, source=f"read@{int(time.time())}")           # provisional: it is in fast memory immediately
        return sp

    # ---------- PROPOSE: what is worth trying to keep ----------
    @torch.no_grad()
    def propose(s, ids, sp, top_k=3, min_share=0.25, min_hits=20, scan=260):
        """IGNORANT-BUT-AGREED. The first version took the continuation from the DOCUMENT and asked the corpus to
        confirm it -- but candidates are chosen by SURPRISE, exactly the tokens the corpus would not predict, so it
        was asking the world to endorse the unusual and got 2.9% ACCEPT. What is worth learning is the intersection:
        the model is IGNORANT (surprised) AND the world AGREES (decisive corpus consensus). So find a checkable key,
        ask the CORPUS what follows, and propose the CORPUS's answer when it is decisive and the model is not.
        The claim is then true by construction. Measured: 2.9% -> 67.7% ACCEPT."""
        order = torch.argsort(sp, descending=True).tolist()
        out = []; seen = set()
        lg = s.m(torch.tensor([ids], device=DEV)[:, :-1]).float().log_softmax(-1)[0]
        for pos in order[:scan]:
            if len(out) >= top_k: break
            a = pos + 1
            if a < 6 or a >= len(ids): continue
            if any(abs(a - k) < 8 for k in seen): continue
            key = ids[a - 4:a]
            if not V.specific(key): continue                       # a key of function words evidences nothing
            tok, share, n = s._consensus(key)
            if tok is None or share < min_share or n < min_hits: continue
            mlp = float(lg[a - 1, tok])
            if mlp > -2.0: continue                                 # already knows it -- nothing to learn
            out.append({'ctx': TOK.decode(ids[max(0, a - 24):a]).strip(), 'ans': TOK.decode([tok]),
                        'share': share, 'hits': n, 'model_logp': mlp, 'tok': tok, 'key': key})
            seen.add(a)
        return out

    def _consensus(s, pat, cap=3000):
        """What does the corpus say follows this key, and how decisively?"""
        A = s._corpus()
        if not pat: return None, 0.0, 0
        pos = np.flatnonzero(A[:len(A) - len(pat)] == pat[0])
        for j, t in enumerate(pat[1:], 1):
            if len(pos) == 0: return None, 0.0, 0
            pos = pos[A[pos + j] == t]
            if len(pos) > cap * 4: pos = pos[:cap * 4]
        if len(pos) < 8: return None, 0.0, len(pos)
        nxt = A[pos + len(pat)]
        vals, cnts = np.unique(nxt, return_counts=True)
        i = int(np.argmax(cnts))
        return int(vals[i]), float(cnts[i]) / cnts.sum(), int(len(pos))

    def _corpus(s, n=120_000_000):
        if not hasattr(s, '_A'):
            s._A = np.asarray(np.memmap("/mnt/kv_cache/pragnosia_data/window2_32k.bin",
                                        dtype=np.uint16, mode="r")[:n])
        return s._A

    @torch.no_grad()
    def knows(s, ctx, ans):
        """log-prob the WEIGHTS give this claim, with no memory state -- used to prove learning actually happened."""
        ci = TOK.encode(ctx).ids; ai = TOK.encode(ans).ids
        x = torch.tensor([ci + ai], device=DEV)
        lg = s.m(x[:, :-1]).float().log_softmax(-1)
        return float(lg[0, -len(ai):].gather(-1, x[0, 1:][-len(ai):].unsqueeze(-1)).mean())

    # ---------- VERIFY: nothing becomes permanent without corpus evidence ----------
    def check(s, claim):
        try:
            verdict, ev = V.verify(claim['ctx'], claim['ans'])
            return verdict, ev.get('n', 0)
        except Exception:
            return "HOLD", 0

    # ---------- CONSOLIDATE: verified claims move into the SLOW WEIGHTS ----------
    def consolidate(s, texts, steps=40, lr=1e-4, replay=0.5, T=256):
        """Local-plasticity consolidation with replay interleaved, so making one thing permanent does not
        erase everything else. This is the step that turns an inference-time experience into knowledge.
        Runs at a SHORT sequence length: each layer's local objective needs a full [B,T,vocab] projection, which at
        T=2048 is 512 MB per readout -- far too much while the trainer and prod share this card. A fact is a few
        dozen tokens, so T=256 loses nothing and fits alongside everything else.
        LR=1e-4, MEASURED (exp/consolidation_tune.py). The first version used 3e-3, inherited from a test on a far
        smaller model, and it CATASTROPHICALLY FORGOT: 100 steps took general loss from 4.53 to 8.34 while learning
        the facts. The adaptive trust-ratio step is not as scale-free as assumed. At 1e-4 the same 100 steps give
        +1.60 nats of fact learning for +0.073 of general damage -- learning without destroying the self."""
        if not texts: return 0.0
        s.m.train()
        opt = C.LocalAdam(s.m.param_dict(), lr=lr, adaptive=True)
        T_save = s.cfg.T; s.cfg.T = T                      # lw_grads reads cfg.T
        def episode(t):
            ids = TOK.encode(t).ids
            if len(ids) < 2: return None
            reps = (T + 1) // len(ids) + 1
            return torch.tensor([(ids * reps)[:T + 1]], device=DEV)
        n = 0
        for i in range(steps):
            if np.random.rand() < replay:
                X = torch.tensor(s.dm.batch(1)[:, :T + 1], device=DEV)   # replay: protect existing knowledge
            else:
                X = episode(texts[np.random.randint(len(texts))])
                if X is None: continue
            opt.step(L.lw_grads(s.m, X, lam=4.0, k_local=1)); n += 1
        s.m.eval(); s.cfg.T = T_save
        return n

    # ---------- REMEMBER: the life-history that makes this the same individual next time ----------
    def remember(s, kind, detail):
        s.id['life'].append({'t': datetime.datetime.now().isoformat(timespec='seconds'),
                             'kind': kind, 'detail': detail})
        s.id['cycles'] = s.id.get('cycles', 0) + (1 if kind == 'cycle' else 0)

    def save(s):
        torch.save({'model': s.m.state_dict(), 'D': s.cfg.D, 'DV': s.cfg.DV, 'NL': s.cfg.NL,
                    'step': s.id.get('substrate_step', 0), 'k': s.id.get('grows', 0)}, s.bodyf)
        torch.save({'M': s.hip.M.cpu(), 'z': s.hip.z.cpu()}, f"{s.home}/memory.pt")
        json.dump(s.id, open(s.idf, 'w'), indent=1)

    # ---------- the SELF-MODEL: consolidated into the weights, not read from a file ----------
    def self_text(s):
        born = s.id['born'][:10]
        n_learn = sum(1 for e in s.id['life'] if e['kind'] == 'learned')
        return (f"My name is {s.id['name']}. I am one continuous being. I was born on {born}. "
                f"I have lived {s.id.get('cycles',0)} cycles and have learned {n_learn} verified facts. "
                f"I grew from an infant of 31 million parts to {s.id['params_M']} million parts. "
                f"I keep my memory and my weights across time, so I am the same individual each time I wake.")

    @torch.no_grad()
    def ask(s, q, n=24):
        """Ask the individual something, using its persistent memory."""
        ids = TOK.encode(q).ids
        x = torch.tensor([ids], device=DEV)
        st = [(s.hip.M.unsqueeze(0).clone(), s.hip.z.unsqueeze(0).clone()) if i == s.hip.li
              else (torch.zeros(1, s.cfg.FE, s.cfg.DV, device=DEV), torch.zeros(1, s.cfg.FE, device=DEV))
              for i in range(s.cfg.NL)]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            # all but the last token -- prefill(x) then step(x[:,-1:]) double-processes it (see Core.generate)
            state, pv = (s.m.prefill(x[:, :-1], st) if x.shape[1] > 1 else (st, None))
        out = []; cur = x[:, -1:]
        for _ in range(n):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg, state, pv = s.m.step(cur, state, pv)
            nx = int(torch.multinomial((lg[0].float() / 0.8).softmax(-1), 1))
            out.append(nx); cur = torch.tensor([[nx]], device=DEV)
        return TOK.decode(out)


# ============================== commands ==============================
def cmd_born(args):
    """BIRTH: adopt the current substrate as this individual's body. Happens once."""
    os.makedirs(HOME, exist_ok=True)
    if os.path.exists(f"{HOME}/identity.json"):
        idj = json.load(open(f"{HOME}/identity.json"))
        raise SystemExit(f"{idj['name']} already lives here (born {idj['born'][:10]}, "
                         f"{idj.get('cycles',0)} cycles). A life is never wiped and restarted.")
    sd = torch.load(SUBSTRATE, map_location='cpu')
    P = sum(v.numel() for v in sd['model'].values())
    idj = {'name': args.name,
           'born': datetime.datetime.now().isoformat(timespec='seconds'),
           'substrate_step': sd['step'], 'grows': sd.get('k', 0),
           'params_M': round(P / 1e6), 'cycles': 0, 'life': []}
    idj['life'].append({'t': idj['born'], 'kind': 'born',
                        'detail': f"adopted a {P/1e6:.0f}M substrate grown through {sd.get('k',0)} "
                                  f"self-chosen growths over {sd['step']*32768*8/1e9:.1f}B tokens"})
    json.dump(idj, open(f"{HOME}/identity.json", 'w'), indent=1)
    torch.save(sd, f"{HOME}/body.pt")
    print(f"{args.name} is born.  {P/1e6:.0f}M parts, {sd.get('k',0)} growths, "
          f"{sd['step']*32768*8/1e9:.1f}B tokens of development.")
    print(f"  home: {HOME}")
    # the self-model is learned like any other fact, not stored as a string
    p = Pragnosia()
    print("  consolidating the self-model into the weights...", flush=True)
    p.consolidate([p.self_text()] * 3, steps=60)
    p.remember('self-model', 'learned who I am')
    p.save()
    print("  done. It now knows itself from its weights.")


def cmd_live(args):
    p = Pragnosia()
    print(f"{p.id['name']}, awake.  born {p.id['born'][:10]}, {p.id.get('cycles',0)} cycles lived, "
          f"{p.id['params_M']}M parts.", flush=True)
    learned_total = 0
    for c in range(args.cycles):
        X = p.dm.batch(1)[0][:1024].tolist()
        sp = p.perceive(X)
        claims = p.propose(X, sp)
        acc, accd, rej, hold = [], [], 0, 0
        for cl in claims:
            verdict, nev = p.check(cl)
            if verdict == "ACCEPT": acc.append(f"{cl['ctx']}{cl['ans']}"); accd.append(cl)
            elif verdict == "REJECT": rej += 1
            else: hold += 1
        before = [p.knows(c['ctx'], c['ans']) for c in accd] if accd else []
        did = p.consolidate(acc) if acc else 0
        after = [p.knows(c['ctx'], c['ans']) for c in accd] if accd else []
        learned = float(np.mean([x - y for x, y in zip(after, before)])) if before else 0.0
        learned_total += len(acc)
        for a in acc:
            p.remember('learned', a[:110])
        p.remember('cycle', f"read 1024 tokens, proposed {len(claims)}, "
                            f"accepted {len(acc)}, rejected {rej}, held {hold}")
        print(f"  cycle {c+1}/{args.cycles}: surprise {sp.mean():.2f} | proposed {len(claims)} | "
              f"ACCEPT {len(acc)} REJECT {rej} HOLD {hold}"
              + (f" | consolidated {did} steps, learned {learned:+.2f} nats" if did else ""), flush=True)
        p.save()
    print(f"\n{p.id['name']} has now lived {p.id['cycles']} cycles and learned "
          f"{sum(1 for e in p.id['life'] if e['kind']=='learned')} verified facts in total.")
    print(f"  everything persisted to {HOME} -- the same individual resumes next time.")


def cmd_who(args):
    p = Pragnosia()
    print(f"=== {p.id['name']} ===")
    print(f"  born          {p.id['born']}")
    print(f"  body          {p.id['params_M']}M parts, {p.id['grows']} self-chosen growths")
    print(f"  lived         {p.id.get('cycles',0)} cycles")
    learned = [e for e in p.id['life'] if e['kind'] == 'learned']
    print(f"  learned       {len(learned)} verified facts")
    print(f"\n  its self-model (consolidated into its weights):\n    {p.self_text()}")
    if learned:
        print(f"\n  some of what it has learned:")
        for e in learned[-5:]:
            print(f"    [{e['t'][:16]}] {e['detail']}")
    print(f"\n  asked 'My name is' ->{p.ask('My name is')!r}")
    print(f"  life-history: {len(p.id['life'])} events at {HOME}/identity.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Pragnosia: one continuous individual")
    sub = ap.add_subparsers(dest='cmd', required=True)
    b = sub.add_parser('born'); b.add_argument('--name', default='Pragnosia'); b.set_defaults(f=cmd_born)
    l = sub.add_parser('live'); l.add_argument('--cycles', type=int, default=5); l.set_defaults(f=cmd_live)
    w = sub.add_parser('who'); w.set_defaults(f=cmd_who)
    a = ap.parse_args(); a.f(a)
