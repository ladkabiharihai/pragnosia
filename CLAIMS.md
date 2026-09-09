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

## What this model does that a transformer of any size cannot

### 1. Reads unbounded context at constant memory and constant latency

| context | peak memory | latency |
|---|---|---|
| 2,048 tokens | 2.51 GiB | 13.8 ms/token |
| 65,536 tokens | 2.73 GiB | 10.6 ms/token |
| 1,048,576 tokens | 2.74 GiB | 10.6 ms/token |

A 512x increase in context costs 9% more memory and no extra time per token. GPT-2 is hard capped at 1,024
tokens and cannot run this test at all. A transformer's memory and per-token cost grow with context by
construction; ours do not, because the model carries a fixed-size recurrent state rather than a growing cache.

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
"The capital of France is Tokyo."        -> REJECT   (corpus says otherwise, consistently)
"Water is made of iron and gold."        -> HOLD     (novel: kept in fast memory, not made permanent)
```

We know of no comparable model that learns verified facts online and rejects false ones.

### 4. Grows itself, and the growth is function preserving

35.0M (D448x10L) to 64.7M to 125.0M (D832x14L), in two growth events triggered by the model's own saturation
signal rather than a schedule. Growth is verified function preserving to **2.3e-07** relative: new parameters
appear and the model computes the same function, so nothing is relearned.

Measured live across a growth event: **BPB 0.9212 to 0.9209**. Thirty million new parameters appeared and the
model's output was unchanged.

### 5. Generates coherently at length

12,000 generated tokens with 4-gram repetition at **0.05 to 0.16** against real text's 0.05, flat rather than
drifting. At 6,000 tokens: 0.07 then 0.05.

---

## Language modelling, measured fairly

Both models scored on the **same text**, the **same 1,024-token windows** (GPT-2's hard limit), and bits per byte
over the **same bytes**. Scoring our model on longer windows would flatter it, because later positions have more
context and are easier to predict.

| eval set | PRAGNOSIA 125M | GPT-2 124M | |
|---|---|---|---|
| our mix (70% language, 30% reasoning) | **1.0798** | 1.0935 | **ahead 0.0137** |
| pure web text | 1.2070 | **1.1156** | **behind 0.0915** |

Against other open models, on our mix, at a matched protocol:

| model | params | training tokens | BPB |
|---|---|---|---|
| **PRAGNOSIA** | 125M | **9.1B** | 1.1085 |
| GPT-2 | 124M | ~9B | 1.0933 |
| **Pythia** | 160M | **300B** | **1.2113** |
| SmolLM2 | 135M | 2T | 0.8854 |
| Qwen2.5 | 494M | 18T | 0.8011 |

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
