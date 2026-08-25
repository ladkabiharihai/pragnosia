# Pragnosia, Progress & Results Log

Durable record of experiments and results. Newest at top of each section. See VISION.md for the design.

## De-risk phase: does PURE LOCAL PLASTICITY (no backprop) train our core? (VISION 0c, the load-bearing bet)

| # | scope | result | verdict |
|---|-------|--------|---------|
| 0 | feedforward Predictive Coding (spatial credit), 2-moons | PC 97.0% vs backprop 98.7% (gap 1.7) | PASS |
| 1 | 1-layer recurrent recall (temporal credit), MQAR | local eligibility 100% == BPTT 100%, grad cosine 1.0000 | PASS (exact) |
| 2 | real-language causal LM gradient, window2_32k, all params | local grad == true grad, cosine 1.0000 (emb,Wk,Wq,Wv,Wo,bo) | PASS (exact) |
| 3 | end-to-end training, FULL 290B 32K mix | local loss tracks BPTT step-for-step, final gap -0.029 | PASS |
| 4 | MULTI-LAYER (PC depth + eligibility time), 2-layer, real mix | RESOLVED (approximate): gap 0.83->0.41 after fixing settling; PC-grad 0.994 cosine to backprop | PASS (PC-approximate) |

### #4 RESOLVED (was OPEN) -- it was a settling-config bug, not structural
Diagnostic (pc_derisk4_diag.py, cosine of PC-grad vs backprop-grad vs settling K):
- The lag was under-settled + UNSTABLE settling (old K=8, eta=0.3). Pattern is textbook Z-IL: at K=1 the top layer's
  grad is EXACT (cosine 1.000), and the error propagates one layer deeper per settling step.
- With stable settling (K=12, eta=0.05): ALL layers reach cosine >=0.994 (mean 0.997) to backprop. Training gap
  0.83 -> 0.41. More settling (K=16..50) does NOT improve cosine (plateaus ~0.994) and costs more -> K=12 is the
  efficiency operating point (§0b2). So PC converges to a ~0.994-cosine APPROXIMATION of backprop (its inherent nature,
  same category as #0's 97 vs 98.7), trains real language locally, tracks BPTT within ~0.4. Exactness never required.
- EFFICIENCY note (§0b2): settling costs ~K x forward per step. For the real deep-narrow model, warm-START settling
  from the previous step's equilibrium (weights change slowly) to cut K sharply; and/or Z-IL-style per-layer-timed
  updates (error front reaches one layer per step) for O(depth) exact-ish credit. To implement in the real trainer.
- CONCLUSION: all de-risks pass. Pure local plasticity (eligibility = EXACT temporal credit; PC = ~0.994 spatial/depth
  credit) trains our recurrent multi-layer core on real language with NO backprop. §0c fully validated at small scale.

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

## Build phase, step 1: DEPTH-SCALING of PC-depth credit (core.py) -- KEY FINDING

Tested cosine(local-grad, backprop-grad) as layers grow (deep-narrow was the planned architecture).
- Pure PC settling DEGRADES with depth: bottom-layer credit at K=3*NL, eta=0.05: 0.80 (2L) -> 0.05 (4L) ->
  0.002 (6L) -> 0.000 (8L). The error front does not reach the bottom cheaply.
- It is EXPENSIVE, not vanishing: with enough propagation budget (eta*K, max stable eta~0.3, K~3*NL) the bottom
  layer reaches ~0.97 (6L, eta=0.3, K=18). But cost = ~3*NL x forward/step (6L~18x, 20L~60x) -> fights §0b2.
- A persistent weak layer stays ~0.85 (min-over-layers) REGARDLESS of K -> an approximation floor from coupling
  eligibility(time)+PC(depth) in the same layers. eta>=0.5 is unstable (NaN).
- CONCLUSION: pure PC is cheap+near-exact SHALLOW (2 layers: 0.994), expensive+approximate DEEP. The learning rule
  constrains the architecture. NOTE: the neocortex is ~6 layers deep and enormously WIDE; deep-narrow (100+ serial
  layers) is a backprop/transformer artifact. So PC's shallow preference is arguably MORE brain-true, pointing to a
  SHALLOW-AND-WIDE core (grow width/state/features, which PC handles well; few layers). ARCHITECTURE DECISION PENDING
  (user): shallow-wide (recommended) vs expensive-deep-PC vs Z-IL/explicit-error-front (exact+O(depth) but ~backprop).

### De-risk phase (learning-rule validity) COMPLETE
All five de-risks pass. Pure local plasticity (no backprop, no autograd) trains our recurrent multi-layer core on real
language: eligibility traces give EXACT temporal credit (cosine 1.0); PC gives a ~0.994-cosine spatial/depth credit
(approximate by nature). Ready to build the real persistent core. Efficiency levers for the real trainer: warm-started
settling, Z-IL-style timing.

## Files (kept minimal; workspace clean)
- VISION.md, PROGRESS.md (this), data/ (tokenizers), .gitignore
- pc_derisk.py (#0), pc_derisk1_v3.py (#1), pc_derisk2_seqlm.py (#2), pc_derisk3_train.py (#3),
  pc_derisk4_multilayer.py (#4), data_mix.py (290B all-corpora mixer).
- Data (gitignored, on /mnt/kv_cache/pragnosia_data): ~290B tokens @32K across 11 corpora.

## Prod safety
Prod (Qwen3.8-27B vLLM :8000 + Whisper/Kokoro) shares the H100, must not go down. Kill only exact PIDs.

## Build step 1b: can we make DEEP local credit cheap AND accurate? (user: need BOTH depth+width)
User is right: width-only = too shallow for knowledge/reasoning; depth-only (naive PC) = expensive + accuracy floor;
efficiency is non-negotiable; the 0.85 floor must be fixed. So "efficient+accurate DEEP local credit" is THE problem.
- Function-preserving IDENTITY deep layers do NOT propagate credit cheaper (identity 6L ~= random 6L at low K, both
  ~0 at bottom). Settling moves the error front ~eta*K layers regardless of weights -> growth-by-identity does not
  dodge the depth cost.
- FUNDAMENTAL TENSION: exact deep credit in O(depth) work needs an explicit downward error sweep = mathematically
  backprop (weight transport). Pure parallel settling is O(depth^2) work and approximate. This is why backprop is
  efficient and biologically-plausible deep credit is hard.
- SYNTHESIS forming (grow BOTH, but bounded): grow depth to a MODEST cortex-like range (~6-12 layers, NOT 100), so PC
  settling cost is a BOUNDED constant (~depth x), affordable; grow WIDTH/state/features/vocab large (PC-cheap, most
  capacity). Recurrence adds effective depth in time. Still must fix the ~0.85 accuracy floor.
- Research verdict (1 agent, 2023-26 lit): iPC fixes SPEED not depth-accuracy. **PC-ALM (Augmented Lagrangian PC,
  2026)** is the lead: per-layer DUAL variables accumulate error -> BALLISTIC (depth-independent) credit, closes BP gap
  at T=2L up to 128 layers on feedforward MLPs, fully local, no weight transport. Also: recurrence = cheap effective
  depth (route compositional compute into TIME where our credit is EXACT); residual identity growth = "error highways";
  the 0.85 floor is eligibility/PC linearization coupling (fixable). Nobody has done PC-ALM+eligibility on recurrent
  LMs -> novel territory.

## Build step 1c: PC-ALM implemented (pcalm.py) -- deep-credit floor IMPROVED, not yet closed
Implemented PC-ALM (dual variables) for our recurrent core. Results (cosine of PC-ALM grad vs backprop):
- Beats naive PC: mean-cosine 0.90-0.98 vs naive degrading-to-0. Shallow (NL=2/4): min ~0.92-0.96.
- Depth 6: a SWEET SPOT at T=4L (24 settling steps): worst-layer ~0.89, mean ~0.93, ROBUST across 4 seeds. Below 4L
  variable; ABOVE 4L the dual ascent OVERSHOOTS (min collapses to ~0.1-0.5) -> needs a dual-ascent STOPPING criterion.
- Damped dual (rho_dual<rho) FAILS (duals must fully accumulate to reach the costate); rho_dual=1.0 required.
- HONEST STATUS: deep local credit is now GOOD (0.89-0.93 at depth 6, vs naive 0.85 floor) and BOUNDED-cost (T~4L), a
  real improvement, but NOT yet ~0.99 in our recurrent+eligibility setting. Getting to 0.99 is open frontier work:
  a principled dual-stopping rule (fix overshoot), mu-PC depth-scaling, precision-weight the worst layer, decoupled
  eligibility linearization (research §5). This combo (PC-ALM + eligibility on a recurrent LM) is genuinely novel.

## ARCHITECTURE SYNTHESIS (answers "grow BOTH depth+width" under our learning rule)
- Grow BOTH, but DEPTH to a BOUNDED cortex-like range (~4-6, up to ~12) where PC-ALM credit is adequate (0.9+) at
  bounded settling cost (T~4L). NOT transformer-deep (100+ layers) -- that is where local credit and efficiency break.
- Put most capacity in WIDTH / state / features / vocab (PC-cheap, near-exact) + RECURRENCE (exact temporal credit =
  cheap effective depth for sequential/compositional computation).
- Residual identity growth = error highways (keeps deep credit tractable through each grow).
- Deep-credit accuracy (->0.99) is an ONGOING refinement track (dual-stopping, mu-PC, precision), not a blocker to
  building at bounded depth now.
