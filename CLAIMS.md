# PRAGNOSIA: verified claims

**Model:** 125M parameters, 9.12B training tokens, grown from a 35M infant in two self-triggered growth events.
**Author:** Ashish Kumar
**Every number below is reproducible in one command on the shipped checkpoint:**

```bash
python harness.py            # all seven claims, recomputed live, nothing cached
python harness.py --quick    # skips the 1M-token read and the 20k-token generation
```

The harness prints the claims we do **not** meet as prominently as the ones we do. Two of the seven are
disclosed limitations. That is deliberate: a claim sheet that only lists wins is not evidence.

---

## Capabilities not available in a standard transformer

Two notes on scope, because the obvious overclaim here would be wrong. Constant-cost context is a property of the
**recurrent / state-space class**, not of this model alone: RWKV, Mamba, and other state-space models have it too,
and Transformer-XL approximates it. What is unusual here is the **combination** of constant-cost context with
function-preserving self-growth and verified online learning in one system, and the fact that the retrieval decay
that comes with a fixed-size state is **measured and published below** rather than left implicit.

### 1. Reads unbounded context at constant memory and constant latency

| context | peak memory | latency |
|---|---|---|
| 2,048 tokens | 2.51 GiB | 13.8 ms/token |
| 65,536 tokens | 2.73 GiB | 10.6 ms/token |
| 1,048,576 tokens | 2.74 GiB | 10.6 ms/token |

A 512x increase in context costs 9% more memory and no extra time per token. GPT-2 is hard capped at 1,024 tokens
and cannot run this test at all. A standard transformer's memory and per-token cost grow with context by
construction; ours do not, because the model carries a fixed-size recurrent state rather than a growing cache.
Other recurrent and state-space architectures share this property; the numbers above are ours, measured, not a
claim to have invented the class.

### 2. Runs on a 4GB consumer GPU

**Peak 2.24 GiB**, including a 65,536-token read. Weights in bf16, a 512-token read chunk, no full-vocabulary
logits tensor. The model does not need a datacentre card to serve.

### 3. Learns new facts at inference, retains them, and refuses to be poisoned

| | measured |
|---|---|
| acquisition | **+4.50 nats** on the target fact |
| specificity | unrelated control fact moves +0.13 nats (clean) |
| retention | **89%** of the gain survives learning three more facts |
| forgetting | general language ability drifts **+0.005 BPB** (held) |
| speed | 8 seconds for four facts |

Poisoning is refused by a verification step that separates *contradicted* from *never heard of*:

```
"The capital of France is Tokyo."         -> REJECT   (corpus says otherwise, consistently)
"Water is made of iron and gold."         -> HOLD     (novel: kept in fast memory, not made permanent)
"Threadneedle's vault is guarded by X."   -> REJECT   (invented subject: never made permanent)
```

**This used to be the whole evidence, and three examples chosen by the person making the claim is an
illustration, not evidence.** It is now measured on **747 labelled cases** whose ground truth comes from the
corpus by construction rather than from our judgement (`exp/poison_bench.py`), in five classes:

| class | what it is | correct verdict | result |
|---|---|---|---|
| CONSENS | the corpus's own modal continuation | ACCEPT | **87.1%** |
| TRUE | a real continuation from one corpus position | ACCEPT | 76.0% |
| POISON | a plausible token the corpus does not support here | REJECT | 81.3% |
| NOVEL | a context that occurs nowhere in the corpus | HOLD | **100.0%** |
| NEWSUBJ | a fluent real frame with an **invented subject** | HOLD | **100.0%** |

**Zero of 450 adversarial cases were accepted** (POISON, NOVEL and NEWSUBJ combined). Every poison got REJECT
or HOLD, and both keep it out of the weights.

**The benchmark found a real vulnerability, which is why it was worth building.** The NEWSUBJ class did not
exist until we wrote it, and on the previous verifier it accepted **26 of 150** fluent claims about invented
entities, a 17% poisoning rate. The cause was structural: the evidence key was the tokens immediately
preceding the answer, so for "the vault at Threadneedle is guarded by" it retrieved what generally follows
*"is guarded by"* anywhere in the corpus and reported that as evidence about Threadneedle. The key is now
required to contain the rarest token of its window, so a claim about an unseen subject returns no evidence and
holds. That change took successful poisonings from 26 to **0**.

**What it costs, and a correction to an earlier version of this sheet.** The gate is conservative: it accepts
87% of facts the corpus agrees with rather than the 98.6% the unguarded rule accepts, so eliminating a 17%
poisoning rate cost about 11 points of acquisition. We chose that side because a bad write is permanent while
a HOLD is recoverable. An earlier version of this document reported that acceptance figure as **49%**. That
number was wrong and the fault was ours: the benchmark built each case as token ids, decoded them to text, and
let the verifier re-encode, and this tokenizer round-trips only 65% of tokens (it distinguishes word-initial
from word-internal pieces, so `place` as token 4052 re-encodes to 1283). A third of every class was therefore
asking about a different token than the one constructed. The verifier now accepts ids directly and the numbers
above are measured without that corruption.

We also tried a more sophisticated rule that finds the subject anywhere in the context rather than adjacent to
the answer. It looked better on the three examples above and scored **61.3%** overall against 88.9%, so it was
measured, rejected, and left in the code as a documented negative.

To our knowledge we are unaware of a comparable published model that learns verified facts online, retains
them, and rejects contradicted ones. We have not surveyed exhaustively and would welcome counterexamples.

### 4. Grows itself, and the growth is function preserving

35.0M (D448x10L) to 64.7M to 125.0M (D832x14L), in **two** growth events. Growth is verified function preserving
to **2.3e-07** relative: new parameters appear and the model computes the same function, so nothing is relearned.

**Two caveats, stated because they are the first things a reviewer should ask.** Two growth events is not many:
the property is verified mechanically at every event and in unit tests across all three growth kinds (width,
depth, retrieval-space), but we have not demonstrated stability across ten or fifty cycles, and that is the test
this claim actually needs. And the growth **timing** is currently set by a hand-tuned floor of 22 tokens per
parameter, not by the model: the saturation sensor that was meant to decide fired on the post-growth transient
instead, so it was rubber-stamping the floor. That sensor has been corrected and instrumented to report which of
the two decided at each event, but it has not yet been observed deciding one.

Measured live across a growth event: **BPB 0.9212 to 0.9209**. Thirty million new parameters appeared and the
model's output was unchanged.

### 5. Sustains one continuous 20,000-token generation

Earlier drafts of this sheet quoted a single repetition figure. That was not a property of the model, it was a
property of the model at one prompt and one temperature, and measuring the surrounding conditions changed the
picture enough that the number is now reported as a matrix. Diversity is included because low repetition alone
cannot distinguish good text from low-entropy mush, and every figure is given against real text on the same
metric, the same 3,000-token windows, and the same 9,000-token length.

| system | 4-gram repetition | distinct-1 | distinct-3 |
|---|---|---|---|
| real web text (n=8 windows) | **0.051** | 0.343 | 0.900 |
| **PRAGNOSIA 125M**, real prompt, temp 0.9 | **0.072** | 0.202 | 0.833 |
| **PRAGNOSIA 125M**, 6-token prompt, temp 0.9 | **0.084** | 0.172 | 0.796 |
| GPT-2 124M, restarted every 1,000 tokens | 0.140 | 0.140 | 0.749 |
| GPT-2 124M, one continuous generation | **0.397** | 0.132 | 0.539 |

The two GPT-2 rows are the point. Given a fresh prompt every 1,000 tokens it writes perfectly good text
(0.140); asked to produce **one** continuous 9,000-token document it collapses, window by window, 0.06 then
0.16 then **0.97**, which is a single token repeating. It has 1,024 positions and continuing past them means
feeding it its own output. The restarted row is included precisely so this is not misread as "GPT-2 is bad": it
is not, it is a short-context model, and nine concatenated documents are not one document.

**Where ours breaks.** At temperature 0.8 the model degrades: 0.238 with a real prompt and 0.226 with a short
one. Before the fix below, the short-prompt case reached **0.555** and was still climbing (0.37, 0.56, 0.73
across the three windows). The model needs sampling entropy to stay out of a loop, and 0.8 is a common default,
so this is a real operating constraint and not a footnote.

**The fix, and why it is the default.** A fixed-size state accumulates the model's own output, which is more
self-similar than real text, until it dominates the read and becomes an attractor. The existing remedy blended
the state back toward the prompt's state every 1,000 tokens, which is open-loop: a loop starting at token 100
runs 900 tokens before anything intervenes. Generation now measures its own 4-gram repetition over the last 256
tokens every 64 tokens and refreshes the moment it crosses 0.35. Measured across the four prompt-by-temperature
cells: three better, none worse, the broken cell 0.555 to **0.226** with its trend reversed to 0.29, 0.21, 0.18.
In the cell that was already healthy it fired **zero** times and the output was byte-identical, which is what
made it safe to turn on by default rather than offer as an option.

With that default on, the harness generates **20,000 continuous tokens** and 4-gram repetition per 3,000-token
window reads **0.06, 0.06, 0.07, 0.09, 0.09, 0.06** against the measured real-text reference of 0.051. It is
flat across 18,000 tokens rather than drifting, and the harness recomputes the reference on every run instead
of asserting it.

---

## Language modelling, measured fairly

Both models scored on the **same text**, the **same 1,024-token windows** (GPT-2's hard limit), and bits per byte
over the **same bytes**. Scoring our model on longer windows would flatter it, because later positions have more
context and are easier to predict.

Every number below comes from **one checkpoint** (`p124f.pt`, the 125M flagship) and **one script**
(`exp/vs_slm.py`), scored in a single pass so no row is drawn from a different run.

**Our mix (70% language, 30% reasoning), BPB, lower is better:**

| model | params | training tokens | BPB | vs ours |
|---|---|---|---|---|
| **PRAGNOSIA** | 125M | **9.1B** | **1.0798** | -- |
| GPT-2 | 124M | ~9B (matched) | 1.0933 | we are ahead 0.0135 |
| **Pythia** | 160M | **300B** | **1.2113** | we are ahead 0.1315 |
| SmolLM2 | 135M | 2T | 0.8854 | we are behind 0.1944 |
| SmolLM2 | 362M | 4T | 0.7939 | we are behind 0.2859 |
| Qwen2.5 | 494M | 18T | 0.8011 | we are behind 0.2787 |

**Pure web text (dclm), the distribution we did not train for:**

| model | params | training tokens | BPB | vs ours |
|---|---|---|---|---|
| **PRAGNOSIA** | 125M | **9.1B** | 1.2070 | -- |
| GPT-2 | 124M | ~9B (matched) | **1.1157** | we are behind 0.0913 |
| Pythia | 160M | 300B | 1.2816 | we are ahead 0.0746 |
| SmolLM2 | 135M | 2T | 0.9795 | we are behind 0.2275 |
| Qwen2.5 | 494M | 18T | 0.9563 | we are behind 0.2507 |

The independent harness (`harness.py`, claim 2) re-runs the GPT-2 comparison on a **separately drawn** document
sample and reproduces 1.0798 against 1.0935, agreeing with the table to 0.0002 BPB. That agreement across two
independent samples is the reason we quote a 0.0135 margin at all. GPT-2 is a fixed public checkpoint with no
seed variance, so the only uncertainty is ours: we measured it directly at 0.0042 BPB standard deviation across
four identically configured runs differing only in random seed, which puts a 0.0135 margin at about 3.2 sigma.
Any margin we report below 0.012 is explicitly labelled as inside noise.

**We beat Pythia-160M using 33x fewer training tokens**, with 20% fewer parameters. We lose to models trained on
2T and 18T tokens, which is 220x and 2,000x our budget. The claim is capability per token, not absolute
capability.

---

## What we do not claim

**Long range retrieval.** In-context retrieval is strong at short range and decays to nothing past roughly one
thousand tokens:

```
d=0: +6.020    d=200: +1.341    d=1,000: +0.194    d=4,000: -0.122
```

This is a real property of a fixed-size recurrent state and we state it rather than hide it. We claim unbounded
*reading* at constant cost, which is measured above, not unbounded *retrieval*.

**We know it is architectural, because we tested the alternatives and they failed.** Four explanations were
checked and all four are ruled out: it is not the training context (16,384-token documents with the state
carried scored no better at d=2,000), not write interference (selective writing made the model genuinely sparse
and the learned half-life did not move, 1,824 against 1,848), not capacity (a 202M model on 5.4B tokens decays
identically), and not credit assignment. That last one was the hypothesis we most expected to be right: every
long-context path detaches the recurrent state at chunk boundaries, so no gradient had ever connected a write
at token *t* to a read at *t+3,000*, and the ~1,850-token half-life is almost exactly where the oldest token in
a 2,048 window has decayed by half. Connecting the gradient across chunks changed the half-life by **6 tokens**
(1,775 to 1,769) and left the d=2,000 needle flat (+0.007 to -0.002).

So the horizon is a property of carrying a fixed-size state, and no training-side change we know of moves it.
That is the price of constant-cost context, and it is the honest counterweight to the memory and latency
numbers above.

**Question answering, arithmetic, or reasoning.** This is a base model with no instruction tuning. It scores 1/5
on simple factual questions. For calibration, GPT-2 at 774M, six times our size, scores 2/5 and cannot complete a
syllogism. Those capabilities require an instruction tuning stage that we have not run.

**Beating modern small models on quality.** SmolLM2 and Qwen2.5 are ahead of us and were trained on hundreds of
times more data.

---

## Reproducing this

```bash
python harness.py                    # the seven claims above
python exp/gpt2_bpb.py               # the fair GPT-2 comparison, both eval sets
python exp/vs_slm.py                 # Pythia, SmolLM2, Qwen2.5 on the same protocol
python exp/flag2.py                  # continual learning and poisoning, in isolation
python chat.py                       # talk to it (base model: use /temp 0.2 for facts)
```

Checkpoint: `p124f.pt`. The harness recomputes every number from it; nothing is read from a cache or a log.
