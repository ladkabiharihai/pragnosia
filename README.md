# Pragnosia

A **growing recurrent language model** with a fixed-size state: it reads unbounded context at constant memory
and constant per-token latency, grows itself while training without relearning what it knows, and can learn
verified facts at inference time.

We do not compete on benchmark leaderboards. The axes are **efficiency, continual learning, long context, and
long generation**, at a size that runs on a consumer GPU.

Author: Ashish Kumar.

## Start here

- **[CLAIMS.md](CLAIMS.md)** is the claim sheet. Every number in it is recomputed live by `harness.py` from
  the shipped checkpoint. It lists the claims we do **not** meet as prominently as the ones we do.
- **[VISION.md](VISION.md)** is the full design, including the negative results and the things we tried that
  did not work. Sections 5b through 5i are the measured findings, several of which are failures.

```bash
python harness.py            # all seven claims, recomputed from the checkpoint, nothing cached
python harness.py --quick    # skips the 1M-token read and the 20k-token generation
python chat.py               # talk to it (base model, no instruction tuning: use /temp 0.2 for facts)
```

## What it is, stated precisely

A recurrent linear-attention model. Each layer carries a fixed-size state that is updated as tokens arrive, so
context costs memory and time **once**, not quadratically and not cumulatively. Reading 1,048,576 tokens uses
2.74 GiB against 2.51 GiB for 2,048 tokens, a 512x context increase for 9% more memory.

Constant-cost context is a property of the **recurrent and state-space class**, shared with RWKV, Mamba and
others. It is not something we invented. What is unusual here is the combination of that with function
preserving self-growth and a verified online-learning path, and that the retrieval decay which comes with a
fixed-size state is measured and published rather than left implicit.

## What it is NOT

**It is not trained by local plasticity, and it does not avoid backpropagation.** Training is chunked
backpropagation with Adam. Local plasticity is a research direction recorded in VISION.md, not a property of
the shipped model, and an earlier version of this file claimed otherwise. If you are evaluating this work,
evaluate the model that exists.

It is also not instruction tuned, so it does not follow instructions or answer questions reliably. It scores
1/5 on simple factual questions; GPT-2 at 774M, six times the size, scores 2/5.

## The honest limitation

In-context retrieval is strong at short range and decays to nothing past roughly a thousand tokens:

```
d=0: +6.020    d=200: +1.341    d=1,000: +0.194    d=4,000: -0.122
```

Four candidate causes were tested and all four are ruled out: training context, write interference, capacity,
and credit assignment. Connecting the gradient across chunk boundaries, the hypothesis we most expected to be
right, moved the retention half-life by 6 tokens. The horizon is a property of carrying a fixed-size state.
We claim unbounded *reading* at constant cost, which is measured. We do not claim unbounded *retrieval*.

## Layout

| path | what it is |
|---|---|
| `core.py` | the model: recurrent layer, state, growth-compatible parameterisation, generation |
| `growth.py` | function-preserving width/depth/retrieval-space growth, and the saturation sensor |
| `verify.py` | the gate deciding what may become permanent knowledge (anti-poisoning) |
| `harness.py` | independent verification of every claim in CLAIMS.md |
| `train400f.py` | the current flagship run, 125M growing to 405M |
| `exp/` | every experiment, including the ones that failed |
