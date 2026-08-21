# Pragnosia, Progress & Results Log

Durable record of experiments and results. Newest at top of each section. See VISION.md for the design.

## De-risk phase: does PURE LOCAL PLASTICITY (no backprop) train our core? (VISION 0c, the load-bearing bet)

| # | scope | result | verdict |
|---|-------|--------|---------|
| 0 | feedforward Predictive Coding (spatial credit), 2-moons | PC 97.0% vs backprop 98.7% (gap 1.7) | PASS |
| 1 | 1-layer recurrent recall (temporal credit), MQAR | local eligibility 100% == BPTT 100%, grad cosine 1.0000 | PASS (exact) |
| 2 | real-language causal LM gradient, window2_32k, all params | local grad == true grad, cosine 1.0000 (emb,Wk,Wq,Wv,Wo,bo) | PASS (exact) |
| 3 | end-to-end training, FULL 290B 32K mix | local loss tracks BPTT step-for-step, final gap -0.029 | PASS |
| 4 | MULTI-LAYER (PC depth + eligibility time), 2-layer, real mix | local TRAINS (8.3->6.9) but LAGS BPTT 6.10 by +0.83; more settling (K=20) did not close it | PARTIAL / OPEN |

### Key facts established
- Eligibility traces give EXACT temporal credit for our linear Hebbian memory (cosine 1.0): read is
  r = sum_t beta_t v_t / sum_t beta_t, beta_t = qf . kf_t; each write leaves trace kf_t, credited locally at the query.
  Temporal credit is local BY CONSTRUCTION -> VChunkRecall is the right substrate for a no-backprop brain.
- Causal LM temporal credit = a REVERSE-CUMULATIVE learning signal (e-prop), still local, no time-unrolling.
- ARCHITECTURE REQUIREMENT: recall needs a key->NEXT-value SHIFT (short-conv). Single-layer same-token linear
  attention cannot bind: BPTT 40.8% -> 100.0% with the shift. Bake the shift into the core.
- PC (cross-layer/DEPTH credit) is APPROXIMATE: single-layer/feedforward it is ~exact-enough (97 vs 98.7), but
  stacked on a recurrent core the approximation gap is ~0.83 loss and did NOT close with more settling. THIS IS THE
  CURRENT OPEN PROBLEM.

### #4 open problem: multi-layer cross-layer credit without backprop
Naive PC settling lags backprop by ~0.8 on the 2-layer recurrent core. To investigate next (do NOT assume, test):
incremental/precision-weighted PC, longer/annealed settling schedules, equilibrium propagation, target propagation,
or accept a small principled gap. Blocks the deep-narrow local-training run until resolved (a deep core needs good
cross-layer credit). Single-layer local learning is already exact and could carry an initial shallow-but-wide core.

## Files (kept minimal; workspace clean)
- VISION.md, PROGRESS.md (this), data/ (tokenizers), .gitignore
- pc_derisk.py (#0), pc_derisk1_v3.py (#1), pc_derisk2_seqlm.py (#2), pc_derisk3_train.py (#3),
  pc_derisk4_multilayer.py (#4), data_mix.py (290B all-corpora mixer).
- Data (gitignored, on /mnt/kv_cache/pragnosia_data): ~290B tokens @32K across 11 corpora.

## Prod safety
Prod (Qwen3.8-27B vLLM :8000 + Whisper/Kokoro) shares the H100, must not go down. Kill only exact PIDs.
