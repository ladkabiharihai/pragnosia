"""THE SCORECARD. Every axis the vision is accountable on, measured in one place, against a real LLM.

The rule this exists to serve, in the user's words: until we beat the LLMs on context rot, training efficiency,
inference efficiency, large-context efficiency, coherence, facts, generation quality, long-generation efficiency
and brain-like qualities, scaling only buys a more expensive loss. So nothing scales until this table says it is
worth scaling, and every number here is measured on the same text, at the same context, with the same decoding.

The axes, and what each number means:

  COMPRESSION      bits per byte on identical UTF-8. The only cross-tokenizer comparable quality number.
  CONTEXT ROT      bits per byte on the SAME final 256 tokens, given 512, 1k, 2k, ... 32k tokens of preceding
                   context. A model that uses context improves as the context grows; one that rots gets worse.
                   This is goal 9 stated as a measurement rather than an aspiration.
  RETRIEVAL        the needle lift, in nats, at increasing distance: does a fact stated d tokens ago still
                   raise the probability of its answer.
  FACTS            48 four-way factual questions, and whether confidence separates right answers from wrong
                   ones (goals 4 and 5: knowing what it knows).
  COHERENCE        repetition and diversity of 512-token generations. Every model in this project has looped
                   under greedy decoding; the rate is the number.
  TRAIN EFFICIENCY tokens per second at a matched step, and the peak memory it took.
  INFER EFFICIENCY tokens per second of generation, the state or cache it holds, and how both grow with the
                   context already consumed. This is where a constant-size state is supposed to pay.

Models are loaded by name so the same table covers ours and a baseline: "run7" style paths for ours, "gpt2"
for the anchor. GPT-2 124M is the baseline because it is the smallest real LLM with a public checkpoint and it
was trained on roughly nine times our tokens, so beating it is a real claim and losing to it is a real loss.
"""
from __future__ import annotations
import math
import time
import numpy as np
import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------------- model wrappers
class Ours:
    """One of our checkpoints. Long contexts go through the carried state, window by window."""
    def __init__(self, path, tok_path, device="cuda"):
        from tokenizers import Tokenizer
        from src.model import LM
        sd = torch.load(path, map_location=device)
        self.m = LM(**sd["cfg"]).to(device).eval()
        self.m.load_state_dict(sd["model"])
        self.tok = Tokenizer.from_file(tok_path)
        self.ctx = sd["cfg"]["ctx"]
        self.name = f"{sd['cfg']['mixer']} {sum(p.numel() for p in self.m.parameters())/1e6:.0f}M"
        self.step = sd["step"]
        self.device = device
        self.max_ctx = 1 << 20                      # no architectural limit: the state carries
        self.mixer = sd["cfg"]["mixer"]
        # A CUDA-graphed decode needs every layer's state to keep its shape. A carrier's does; an attention
        # layer's key and value cache grows by one token per step, so a model with any attention layer in it
        # decodes eagerly. Gating on the mixer NAME was wrong the moment hybrids existed: the graphed path
        # tried to copy a 7-token cache into a 6-token buffer (2026-09-18).
        # A windowed attention layer's cache is a RING once it is full, so its shape stops changing and the
        # graph can hold it; an unwindowed one grows for ever and cannot be graphed.
        self.graphable = all(type(b).__name__ != "Attn" or getattr(b, "window", 0) for b in self.m.mix)

    def encode(self, text): return self.tok.encode(text).ids
    def decode(self, ids): return self.tok.decode(ids)

    @torch.no_grad()
    def logits_last(self, ids, n_score):
        """Log-probabilities for the LAST window of ids, with everything before it carried in the state.

        Windows are tiled from the END so the final one is always full: scoring the last n_score positions
        would otherwise fall off a short remainder window (it did, and the slice raised rather than lying)."""
        w = self.ctx
        assert n_score + 1 <= w, (n_score, w)
        x = torch.tensor([ids], device=self.device)
        L = x.shape[1]
        bounds = list(range(L, 0, -w))[::-1]          # ends of each window, last one at L
        starts = [max(0, b - w) for b in bounds]
        states = [None] * self.m.layers
        pos = torch.arange(w, device=self.device)
        out = None
        for s0, e0 in zip(starts, bounds):
            if self.mixer == "attn":
                states = [None] * self.m.layers       # a 512-token window keeps nothing across its boundary
            xw = x[:, s0:e0]
            h = self.m.e(xw) + self.m.p(pos[: xw.shape[1]])
            for i, (mx, mp) in enumerate(zip(self.m.mix, self.m.mlp)):
                h, states[i] = mx(h, states[i])
                h = mp(h)
            out = F.linear(self.m.n(h), self.m.e.weight)
        return out                                   # (1, last window, V)

    @torch.no_grad()
    def state_bytes(self, B=1, T=None):
        """Bytes of state for one sequence. For a carrier this is constant; for the attention control it is a
        KV cache that grows with the window, so the number is reported at a full window."""
        x = torch.randint(0, self.m.vocab, (B, T or self.ctx), device=self.device)
        h = self.m.e(x) + self.m.p(torch.arange(x.shape[1], device=self.device))
        tot = 0
        for mx, mp in zip(self.m.mix, self.m.mlp):
            h, st = mx(h)
            h = mp(h)
            if st is not None:
                tot += sum(t.numel() * t.element_size() for t in st if torch.is_tensor(t))
        return tot

    @torch.no_grad()
    def generate(self, prompt_ids, n, temp=0.0, seed=0, carry=True, stop=None):
        """INCREMENTAL generation: the prompt is consumed once into the state, then each new token is one step.

        The first version of this re-ran the whole context for every token and reported 2.5 seconds per token at
        32k context, which measured the harness rather than the model. A recurrent state exists so that is
        unnecessary: the cost per token here is independent of how much context came before.

        Positions restart at each window boundary, as in training; the state carries across the boundary."""
        g = torch.Generator(device=self.device).manual_seed(seed)
        w = self.ctx
        states = [None] * self.m.layers
        lg = None; pw = 0
        for s in range(0, len(prompt_ids), w):
            x = torch.tensor([prompt_ids[s:s + w]], device=self.device)
            nn_ = x.shape[1]
            h = self.m.e(x) + self.m.p(torch.arange(nn_, device=self.device))
            for i, (mx, mp) in enumerate(zip(self.m.mix, self.m.mlp)):
                h, states[i] = mx(h, states[i]); h = mp(h)
            lg = F.linear(self.m.n(h[:, -1:]), self.m.e.weight)[0, -1].float()
            pw = nn_ % w
        out = []
        if self.graphable and n > 8:
            return out + self._generate_graphed(states, pw, lg, n, temp, g, stop)
        for _ in range(n):
            nxt = int(lg.argmax()) if temp == 0 else int(torch.multinomial(torch.softmax(lg / temp, -1), 1, generator=g))
            out.append(nxt)
            if nxt == stop: break
            if pw == 0 and self.mixer == "attn":
                states = [None] * self.m.layers          # the window ended; attention keeps nothing across it
            x = torch.tensor([[nxt]], device=self.device)
            h = self.m.e(x) + self.m.p(torch.tensor([pw], device=self.device))
            for i, (mx, mp) in enumerate(zip(self.m.mix, self.m.mlp)):
                h, states[i] = mx(h, states[i]); h = mp(h)
            lg = F.linear(self.m.n(h), self.m.e.weight)[0, -1].float()
            pw = (pw + 1) % w
        return out

    @torch.no_grad()
    def _generate_graphed(self, states, pw, lg, n, temp, g, stop=None):
        """The decode step as ONE CUDA graph, captured once per model and reused across calls.

        Measured eagerly: 965 kernel launches and 12.65 ms per token for 1.7 ms of GPU work, so the step was
        seven times CPU-bound. Every state tensor lives in a static buffer the graph updates in place; the token
        and the position are static device scalars filled before each replay; the positional index wraps on
        the device. Only the argmax or the sample runs outside. Capture costs ~120 ms once; amortised over a
        64-token measurement it read as 4.2 ms/token, so it is done once and kept."""
        w = self.ctx; dev = self.device
        def flat(states):
            out = []
            for s_ in states:
                if isinstance(s_, tuple) and len(s_) == 2 and torch.is_tensor(s_[0]) and not torch.is_tensor(s_[1]):
                    out.append((s_[0], torch.tensor(float(s_[1]), device=dev)))    # assoc: (S, p0) with p0 an int
                else:
                    out.append(tuple(s_))
            return out
        fresh = flat(states)
        # fill every windowed attention cache to its full width before capture, so the ring is already at the
        # size it will keep for the rest of the decode
        for i, blk in enumerate(self.m.mix):
            win = getattr(blk, "window", 0)            # NOT w: w is the context width the position wraps on,
            if win and isinstance(fresh[i], tuple) and torch.is_tensor(fresh[i][0]) and fresh[i][0].shape[2] < win:
                pad = win - fresh[i][0].shape[2]       # and clobbering it made the wrap a modulo by zero
                fresh[i] = tuple(torch.cat([t[:, :, :1].expand(-1, -1, pad, -1), t], dim=2) for t in fresh[i])
        if getattr(self, "_g", None) is None:
            st = [tuple(t.clone().contiguous() for t in s_) for s_ in fresh]
            tok = torch.zeros(1, 1, dtype=torch.long, device=dev)
            pos = torch.tensor([pw], device=dev)
            def step():
                h = self.m.e(tok) + self.m.p(pos)
                for i, (mx, mp) in enumerate(zip(self.m.mix, self.m.mlp)):
                    h, new = mx(h, st[i]); h = mp(h)
                    for a, b in zip(st[i], new):
                        a.copy_(b)
                pos.copy_((pos + 1) % w)
                return F.linear(self.m.n(h), self.m.e.weight)[0, -1].float()
            s_ = torch.cuda.Stream(); s_.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s_):
                for _ in range(2): step()
            torch.cuda.current_stream().wait_stream(s_)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out_lg = step()
            self._g = (graph, st, tok, pos, out_lg)
        graph, st, tok, pos, out_lg = self._g
        for s_, f_ in zip(st, fresh):                                            # load this call's state
            for a, b in zip(s_, f_): a.copy_(b)
        pos.fill_(pw)
        out = []; cur = lg
        for _ in range(n):
            nxt = int(cur.argmax()) if temp == 0 else int(torch.multinomial(torch.softmax(cur / temp, -1), 1, generator=g))
            out.append(nxt)
            if nxt == stop: break              # the answer ended: HF's generate stops at EOS, and so must this
            tok.fill_(nxt)
            graph.replay()
            cur = out_lg
        return out


class GPT2:
    def __init__(self, device="cuda"):
        from transformers import GPT2LMHeadModel, GPT2TokenizerFast
        self.m = GPT2LMHeadModel.from_pretrained("gpt2").to(device).eval()
        self.tok = GPT2TokenizerFast.from_pretrained("gpt2")
        self.ctx = 1024
        self.max_ctx = 1024                          # hard architectural limit, and part of the comparison
        self.name = "GPT-2 124M"
        self.step = 0
        self.device = device
        self.mixer = "attn"

    def encode(self, text): return self.tok.encode(text)
    def decode(self, ids): return self.tok.decode(ids)

    @torch.no_grad()
    def logits_last(self, ids, n_score):
        ids = ids[-self.max_ctx:]                    # it cannot see further; that is the point
        x = torch.tensor([ids], device=self.device)
        return self.m(x).logits

    @torch.no_grad()
    def state_bytes(self, B=1):
        return 0                                     # no recurrent state; see kv_bytes

    def kv_bytes(self, T):
        c = self.m.config
        return 2 * c.n_layer * T * c.n_embd * 2      # bf16 K and V

    @torch.no_grad()
    def generate(self, prompt_ids, n, temp=0.0, seed=0, carry=True):
        """With its KV cache, which is the fair comparison: attention decodes incrementally too, at the cost of
        a cache that grows with every token and a context it cannot exceed."""
        g = torch.Generator(device=self.device).manual_seed(seed)
        ids = list(prompt_ids)[-self.max_ctx:]
        x = torch.tensor([ids], device=self.device)
        o = self.m(x, use_cache=True)
        past = o.past_key_values; lg = o.logits[0, -1].float()
        out = []
        for _ in range(n):
            nxt = int(lg.argmax()) if temp == 0 else int(torch.multinomial(torch.softmax(lg / temp, -1), 1, generator=g))
            out.append(nxt)
            ids.append(nxt)
            if len(ids) > self.max_ctx:               # the cache is full: it must re-read a truncated window
                ids = ids[-self.max_ctx:]
                o = self.m(torch.tensor([ids], device=self.device), use_cache=True)
                past = o.past_key_values; lg = o.logits[0, -1].float(); continue
            o = self.m(torch.tensor([[nxt]], device=self.device), past_key_values=past, use_cache=True)
            past = o.past_key_values; lg = o.logits[0, -1].float()
        return out


def kv_bytes(model, T):
    if isinstance(model, GPT2):
        return model.kv_bytes(T)
    return None


# --------------------------------------------------------------------------------- the axes
EOD = 8191                                            # our tokenizer's end-of-document marker


def _docs(bins, ks, ps, rng, n, length, tok, single_doc=False, sources=None):
    """n passages of `length` OUR tokens, decoded to text so every model tokenizes them itself.

    single_doc restricts the passage to lie INSIDE one document, which is the only way the context-rot
    measurement means anything: 61.8% of our tokens live in documents longer than 2048, but a random span of
    2048 usually straddles several unrelated ones, and a model that ignores the earlier ones is then right."""
    keys = sources or ks
    pr = np.array([ps[ks.index(k)] for k in keys]); pr = pr / pr.sum()
    out = []
    tries = 0
    while len(out) < n and tries < 400 * n:
        tries += 1
        a = bins[keys[rng.choice(len(keys), p=pr)]]
        j = int(rng.integers(0, len(a) - length - 2))
        span = np.asarray(a[j:j + length])
        if single_doc and (span == EOD).any():
            continue
        out.append(tok.decode([int(t) for t in span]))
    return out


def compression(model, texts, n_score=256):
    """Bits per byte of the last n_score tokens of each text, with the rest as context."""
    nats = 0.0; nbytes = 0
    for t in texts:
        ids = model.encode(t)
        if len(ids) < n_score + 2:
            continue
        lg = model.logits_last(ids, n_score).float().log_softmax(-1)
        if lg.shape[1] < n_score + 1:
            continue
        y = torch.tensor(ids[-n_score:], device=lg.device)
        nats += float(-lg[0, -n_score - 1:-1].gather(-1, y.unsqueeze(-1)).sum())
        nbytes += len(model.decode(ids[-n_score:]).encode("utf-8"))
    return nats / max(1, nbytes) / math.log(2), nbytes


def context_rot(model, bins, ks, ps, tok, lengths=(512, 1024, 2048, 4096, 8192, 16384, 32768), n=8, n_score=256,
                seed=11, single_doc=False, sources=None):
    """The SAME final 256 tokens, scored with more and more preceding context. Falling bits per byte means the
    model is using the context; rising means it is being hurt by it."""
    rng = np.random.default_rng(seed)
    out = {}
    base = _docs(bins, ks, ps, rng, n, max(lengths) + n_score, tok, single_doc=single_doc, sources=sources)
    for L in lengths:
        texts = []
        for b in base:
            ids = tok.encode(b).ids
            texts.append(tok.decode(ids[-(L + n_score):]))
        if L + n_score > model.max_ctx and isinstance(model, GPT2):
            out[L] = None                            # cannot be given this context at all
            continue
        try:
            out[L] = compression(model, texts, n_score)[0]
        except torch.OutOfMemoryError:
            out[L] = float("nan"); torch.cuda.empty_cache()
    return out


def coherence(model, prompts, n=512, temp=0.0, seed=0):
    """Repetition of long greedy generations. rep4 is the fraction of 4-grams that have appeared before; a
    looping model approaches 1. distinct2 is unique bigrams over bigrams. Measured on the tokens each model
    emits, then on the decoded text so tokenizer differences cannot flatter anyone."""
    r4 = []; d2 = []; secs = []
    for p in prompts:
        ids = model.encode(p)
        t0 = time.time(); gen = model.generate(ids, n, temp=temp, seed=seed); secs.append(time.time() - t0)
        w = model.decode(gen).split()
        if len(w) < 8:
            continue
        four = [tuple(w[i:i + 4]) for i in range(len(w) - 3)]
        seen = set(); rep = 0
        for g in four:
            if g in seen: rep += 1
            seen.add(g)
        r4.append(rep / len(four))
        two = [tuple(w[i:i + 2]) for i in range(len(w) - 1)]
        d2.append(len(set(two)) / len(two))
    return dict(rep4=float(np.mean(r4)), distinct2=float(np.mean(d2)), tok_per_s=float(n / np.mean(secs)))


def gen_cost_vs_context(model, bins, ks, ps, tok, contexts=(512, 2048, 8192, 32768), n=32, seed=3):
    """Two separate costs, because conflating them measures nothing.

      INGEST: reading C tokens of context once. Linear in C for everyone; the question is the constant.
      DECODE: milliseconds per token AFTER that. Attention's grows with C because every new token attends to
              all of them; a constant-size state's does not. This is the long-generation axis.

    Also returned: the memory held while decoding, which for attention is the KV cache and grows with C."""
    rng = np.random.default_rng(seed)
    out = {}
    for C in contexts:
        if C > model.max_ctx:
            out[C] = None
            continue
        text = _docs(bins, ks, ps, rng, 1, C, tok)[0]
        ids = model.encode(text)[:C]
        model.generate(ids[:64], 2, temp=0.0)                 # warm
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        t0 = time.time(); model.generate(ids, 1, temp=0.0); torch.cuda.synchronize()
        ingest = time.time() - t0
        t0 = time.time(); model.generate(ids, n + 1, temp=0.0); torch.cuda.synchronize()
        per = (time.time() - t0 - ingest) / n
        held = kv_bytes(model, C + n)
        if held is None:
            held = model.state_bytes()
        out[C] = (ingest * 1e3, per * 1e3, held / 2**20)
    return out


def facts(model, FACTS, rule="sum"):
    """Top-1 among four same-kind candidates by completion log-probability, and whether confidence separates
    right from wrong: a model that knows what it knows is more confident when it is right.
    rule "sum" (default) is the TOTAL log-probability of the option, which is the plain question "which of
    these four continuations does the model find most likely". "mean" divides by the number of TOKENS, and so
    depends on how finely each vocabulary cuts the option: 1.00 pieces for GPT-2 on the induction options
    against 1.91 for our 8,192-vocabulary, and switching rules moves our induction score by 11 points while
    leaving GPT-2's unchanged. "byte" divides by the option's length in bytes, which is vocabulary-neutral but
    puts every model BELOW CHANCE on the letter-counting items (5.3%), so it is not a safe default either.
    Every multiple-choice number in this project before 2026-09-21 13:00 UTC used "mean" and is restated in
    PROGRESS.md under this rule."""
    right = 0; rc = []; wc = []
    for stem, ans, wrongs in FACTS:
        lps = []
        for c in [ans] + wrongs:
            ids = model.encode(stem) + model.encode(c)
            lg = model.logits_last(ids, 1).float().log_softmax(-1)
            nc = min(len(model.encode(c)), lg.shape[1] - 1)
            y = torch.tensor(ids[-nc:], device=lg.device)
            lp = lg[0, -nc - 1:-1].gather(-1, y.unsqueeze(-1))
            lps.append(float(lp.mean() if rule == "mean" else lp.sum() / (len(c.encode()) if rule == "byte" else 1)))
        a = np.array(lps); p = np.exp(a - a.max()); p /= p.sum()
        if int(p.argmax()) == 0: right += 1; rc.append(p.max())
        else: wc.append(p.max())
    return dict(correct=right, n=len(FACTS), conf_right=float(np.mean(rc)) if rc else 0.0,
                conf_wrong=float(np.mean(wc)) if wc else 0.0,
                separation=float(np.mean(rc) - np.mean(wc)) if rc and wc else 0.0)

def arithmetic_exact(model, items, max_new=6):
    """The harder half of the arithmetic probe: GENERATE the answer instead of ranking four candidates.

    Multiple choice can be passed by elimination; exact match cannot. The first integer in the greedy
    continuation is the answer, so a model that writes "= 122 and then" still counts as right. Chance is
    effectively zero here, which is what makes it worth running on a model too small to do arithmetic: the
    first non-zero score is a real event, and until then the multiple-choice number carries the signal."""
    right = 0; blank = 0
    for stem, ans, _ in items:
        ids = model.encode(stem)
        txt = model.decode(model.generate(ids, max_new, temp=0.0))     # generate returns the NEW tokens only
        digits = ""
        for ch in txt:
            if ch.isdigit(): digits += ch
            elif digits: break
        if not digits: blank += 1
        elif digits == ans.strip(): right += 1
    return dict(correct=right, n=len(items), no_number=blank)
