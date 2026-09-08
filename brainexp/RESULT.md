# BRAIN-REGION CONDITIONAL COMPUTATION — experiment result (2026-09-08)

Run in an isolated copy (`brainexp/`); nothing in the main training path was touched. Main line locked at tag
`pre-brain-exp`.

## The structure
| region | role | specialisation | routed? |
|---|---|---|---|
| pituitary | global disposition | tiny MLP, biases routing for every token, zero-init (starts neutral) | — |
| amygdala | salience | the existing selective-write gate | always on |
| cerebellum | fast procedural | narrow, cheap | always on |
| occipital | local pattern | DIN 16, 2x MLP | routed |
| temporal | language + memory | DIN 40 (big state) | routed |
| parietal | integration | 6x MLP | routed |
| frontal | reasoning | DIN 32, 4x MLP | routed |

Layers here are RECURRENT, so tokens cannot be gathered arbitrarily as an MoE gathers them for a feedforward
expert — the state must see its tokens in order. Each region takes its top-`cap` tokens **in sequence order** and
runs its recurrence over that subsequence. This is not MoE (VISION 6): no expert duplication, no added parameters
per expert, no top-1 collapse mode. It skips computation on parameters that already exist.

## What is TRUE
**1. The growth property is real, and it is arithmetic.**
| cortical regions | top-k | params | MFLOPs/token |
|---|---|---|---|
| 4 | 2 | 41.2M | 27.6 |
| 6 | 2 | 51.9M | 27.6 |
| 8 | 2 | **62.5M** | **27.6** |

**+52% parameters, identical per-token cost.** The current architecture does not have this: 125M -> 405M costs
2.5x more per token.

**2. Real throughput and memory win, once the batch fills the idle GPU.**
| arm | best batch | ktok/s | peak GiB |
|---|---|---|---|
| DENSE | 24 | 257 | 18.13 |
| **ROUTED k=2** | **16** | **354** | **9.43** |

**1.38x faster in wall-clock, 1.3x less memory.** At batch 8 routed was SLOWER (188 vs 224) — the user spotted
that routed sat at 64-74% GPU utilisation against dense's 100%, i.e. it was launch-bound, not compute-bound. The
FLOP saving was real but unconverted; a bigger batch converts it. The gap between 1.38x measured and 1.77x
analytic is gather/scatter overhead that a fused kernel would recover.

**3. The router never collapsed.** Entropy 1.323-1.382 against a 1.386 maximum in every arm. That is the failure
mode that kills these schemes and it did not happen.

## What is FALSE
**Routing does not buy capability per FLOP.**
| comparison | result |
|---|---|
| matched PARAMETERS: dense-4 vs routed top-2 | routing costs **+0.0502 BPB** for 1.77x fewer FLOPs |
| matched PARAMETERS: dense-4 vs routed top-1 | **+0.0828 BPB** for 2.88x fewer FLOPs (sub-linear: 63% more cost for 63% more saving) |
| **matched COMPUTE: dense-4 (41.2M) vs routed-8 (62.5M)** | routed is **+0.0082 BPB WORSE** with 1.52x the capacity |

52% more parameters at the same bill did **not** beat using fewer parameters fully. That is close to what VISION 6
concluded for MoE, reached from a different direction.

**CAVEAT that must be stated: +0.0082 is inside plausible seed noise, and we have never measured the run-to-run
seed floor on BPB.** So the honest reading is "no measurable gain", not "measurably worse".

## The conclusion
**Constant-load routing buys SPEED and MEMORY, not capability-per-FLOP.**
- Want to serve the same model 1.38x faster in 1.3x less memory? Routing works.
- Want a better model at the same cost? It does not.
- Want growth to stop costing anything at serving time? The property holds — but the capacity it adds is not free
  quality, it is quality you still have to pay for in tokens.

Not adopted into the main line. Kept here as a measured negative on the capability question and a measured
positive on the serving question.
