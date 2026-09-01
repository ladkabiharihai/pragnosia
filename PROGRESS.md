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

## Build step 1d: deep-credit ROOT-CAUSED (option B: grind to 0.99 before building)
Chased the worst-layer floor to its actual source. Two hypotheses tested, one eliminated, root cause found.
- **DECOUPLED linearization: NO effect** (identical to 3 decimals at NL=4/6/8). Settled activities barely drift from
  feedforward, so applying the weight-VJP at ff vs settled caches changes nothing. Hypothesis eliminated (cheap).
- **The floor is NOT per-LAYER, it is per-PARAMETER-TYPE.** Wk/Wq (key/query) lag at 0.81-0.93 while Wv/Wo are
  0.96-0.99, at EVERY depth. Wk/Wq gradients are ~1000x SMALLER (1e-6 vs 1e-2) because they flow through the 2nd-order
  Taylor feature map -> tiny, ill-conditioned path that AMPLIFIES any error in the layer-error estimate.
- **CONTROL (decisive): our local machinery is EXACT.** Feeding local_vjp the TRUE layer-errors (no PC settling)
  reproduces backprop at cosine 1.0000 on EVERY layer and EVERY param (Wk,Wq,Wv), all depths. So eligibility+VJP are
  perfect; the ONLY error source is PC settling's approximation of the layer-errors, magnified by the tiny Wk/Wq path.
- Mean cosine is already 0.964-0.972 across depths; only the tiny-gradient params drag the worst-case.
- CONSEQUENCE: the remaining work is a better settled layer-error estimate (precision-weighting to rescale the
  ill-conditioned Wk/Wq path, mu-PC depth-scaling, dual-stopping rule), NOT a flaw in our local learning rule.
- **PRECISION-WEIGHTING: FAILS** (worst-cosine -0.97 vs baseline 0.81). Rescaling per-layer errors breaks the
  dual/costate relationship (duals must accumulate the UNSCALED violation to converge to the true adjoint).
  Hypothesis eliminated. (Damped duals also failed earlier, same reason.)

### Option-B status (grind deep-credit to 0.99): honest scoreboard
Tried and ELIMINATED: more/annealed settling; identity/function-preserving init; iPC; decoupled linearization;
precision-weighting; damped dual ascent. Established: our eligibility+VJP machinery is EXACT (cosine 1.0000 fed true
errors, all depths/params); mean cosine 0.964-0.972; only the tiny-gradient Wk/Wq (Taylor-map) path drags the
worst-case to 0.78-0.93 by amplifying settling error. Remaining untried levers: mu-PC depth-scaling (weights scaled
(NL)^-1/2, published to 128 layers), a dual-ascent stopping criterion (fixes the T>4L overshoot), and Z-IL-style
per-layer-timed updates (exact but needs weight transport -> would soften §0c purity).
HONEST READ: this is genuine open research. The gap is now well-characterized and NOT a bug; the practical operating
point (T~4L, mean 0.97, worst 0.85-0.93) already trains real language. Recommend a decision: continue grinding (mu-PC
next) vs build at the current operating point and refine in-flight.

## Build step 1e: BOTH LEVERS PASSED -- deep local credit SOLVED (muPC + dual-STOP), cheaper AND deeper
Two levers implemented in pcalm3.py (core.py gained `res_scale` for Depth-muP; scaled VJP verified EXACT, cosine
1.000000 when fed true errors at res_scale=1.0 and 0.408).
- **(A) muPC depth-scaling**: residual branch scaled by 1/sqrt(NL) (Depth-muP).
- **(B) DUAL-STOPPING rule**: track the PRIMAL STATIONARITY residual S = sum_l ||dE/da_l||^2 (INTRINSIC -- no backprop
  info) and use the (a,lambda) snapshot at its MINIMUM. At the ALM solution the duals equal the adjoint and S->0, so
  min-S is the principled stopping point. Fixes the T-overshoot completely (results identical at T=4L and 8L).
- Neither lever alone suffices (dual-STOP alone = 0.17; muPC alone = 0.904). TOGETHER they solve it.

RESULT (3 seeds, cosine of pure-local grad vs backprop over all layers x {Wk,Wq,Wv,Wo}):
| NL | baseline worst/mean (steps) | muPC+STOP worst/mean (steps) |
|----|------------------------------|-------------------------------|
| 2  | 0.988 / 0.994 (8)  | 0.951 / 0.988 (3)  |
| 4  | 0.843 / 0.975 (16) | 0.945 / 0.986 (6)  |
| 6  | 0.807 / 0.968 (24) | 0.941 / 0.985 (9)  |
| 8  | -0.080 / 0.930 (32)| 0.907 / 0.983 (12) |
| 12 | 0.525 / 0.937 (48) | 0.878 / 0.984 (18) |
- **Mean cosine is now essentially DEPTH-INDEPENDENT (0.983-0.988 at NL=2..12)** -- the "ballistic" credit PC-ALM
  promises. Baseline degrades (mean 0.994->0.930, worst collapses to -0.08 at NL=8).
- **Cost DROPS to k* ~ 1.5*NL settling steps** (vs 4*NL): ~2.7x cheaper AND more accurate. Big §0b2 win.
- Hyperparam refinement @NL=6: plateau ~0.95 worst / 0.985 mean. Efficiency sweet spot **eta_a=0.8, rho=0.5 ->
  worst 0.948, mean 0.985, k*=10**. Instability when eta_a*rho >= ~1.0 (avoid).
- HONEST: worst-case is ~0.88-0.95 (still the tiny-gradient Wk/Wq Taylor path), not 0.99. Mean 0.985. But deep credit
  is no longer depth-limited, no longer overshoots, and is cheaper than before. This is a real, novel result:
  muPC + ALM-duals + intrinsic min-stationarity stopping + eligibility traces on a recurrent LM.

## Build step 1f: TRAINING COMPARISON -- the at-init cosine benchmark was MISLEADING (important correction)
Ran the real test (train_compare.py): same muPC architecture, same init, same data stream (real 290B mix),
NL=6, D=96, T=32, B=32, 400 steps, Adam lr=3e-3. Held-out loss:

| arm | final loss | ppl | gap vs BPTT | wall-cost |
|-----|-----------|-----|-------------|-----------|
| BPTT (backprop reference) | **6.4597** | 639 | -- | 1.0x |
| pure-local muPC+STOP      | 7.5237 | 1851 | **+1.064** | 8.2x |
| pure-local naive-PC       | 7.4393 | 1702 | +0.980 | 11.6x |

**So at depth 6 pure-local trains SUBSTANTIALLY worse than backprop (+1.0 loss) at 8-12x the cost**, despite the
at-init gradient cosine being 0.985. Diagnosis (decisive):
- **The gradient cosine COLLAPSES once the model is trained.** Measured along a BPTT trajectory (NL=6):
  step 0: worst 0.967 / mean 0.988 / k*=10  ->  step 50+: mean **0.167** with k*=1.
  mean 0.167 = exactly 1/6 = ONLY THE TOP LAYER of 6 gets credit. The dual-STOP rule stops at k=1 on trained weights.
- **It is not just the stop rule -- the SETTLING itself becomes oscillatory on trained weights.** With stop disabled,
  per-layer cosines go NEGATIVE (e.g. T=12: per-layer [0.41,0.12,0.03,-0.44,0.32,-0.55], mean -0.02). Best achievable
  on trained weights after a full (eta,rho,T) sweep: **mean 0.887 / worst 0.599 at T=8L** (vs 0.985/0.95 at init),
  and the response to eta is NON-MONOTONIC = genuine oscillation, not mis-tuning.
- LESSON: **measure credit quality on TRAINED weights, never at init.** At init the model is benign (small effective
  Jacobians); after training the Jacobians grow and PC settling destabilizes. All earlier multi-layer cosine numbers
  (#4, 1c, 1e) are at-init and therefore optimistic. The SINGLE-LAYER results (#1,#2,#3) are ANALYTIC (no settling)
  and remain exactly valid.

## DECISIVE DEPTH RESULT (on TRAINED weights) -- where pure-local credit is reliable
| NL | best mean cosine | worst | note |
|----|------------------|-------|------|
| 1  | **1.000** | **1.000** | EXACT -- no hidden layer to settle; credit is the analytic readout error |
| 2  | 0.882 | 0.599 | cliff: the moment ANY hidden layer needs settling |
| 3  | 0.884 | 0.696 | |
| 4  | 0.861 | 0.049 | |
| 6  | 0.887 | 0.599 | |
- **The cliff is 1 -> 2 layers, not gradual with depth.** Pure-local is EXACT at 1 layer and ~0.88 at any depth >= 2.
- ARCHITECTURE CONSEQUENCE (honest): under a strict no-backprop constraint, the RELIABLE configuration is
  **1 layer deep, very WIDE, heavily RECURRENT** (recurrence = effective depth in time, where our temporal credit is
  EXACT). Multi-layer works but pays a real quality cost (+1.0 loss at depth 6 in a 400-step run).
- NEXT RESEARCH DIRECTION (untried, promising): avoid cross-layer credit ENTIRELY via LAYERWISE/GREEDY LOCAL
  OBJECTIVES (Forward-Forward style: each layer has its own local target, no settling, no depth credit needed) --
  that would scale to any depth with exact local learning. Also: normalized/trust-region settling, equilibrium prop.

## Build step 1g: LAYERWISE LOCAL OBJECTIVES -- depth UNLOCKED under pure local plasticity (lw_lib.py)
User rejected a shallow-only core ("we need to figure it out"), so we attacked depth properly. Approach: give EVERY
layer its OWN local next-token objective through the shared readout and BLOCK gradients between layers, so there is
**no cross-layer credit to assign at all** -- the exact mechanism (PC settling) that failed at depth is REMOVED.
Within-layer credit stays EXACT (eligibility traces). Why this avoids the classic greedy-layerwise failure: layers are
RESIDUAL (a layer can only ADD to the stream, never destroy it) and every layer shares the SAME target and readout, so
each layer REFINES the prediction (iterative refinement / logit-lens) instead of solving a myopic private task.
Final rule = per-layer local objective + DFA broadcast of the top error via FIXED RANDOM feedback (no weight
transport) + an embedding local objective. NO backprop, NO settling, NO weight transport.

Ablation @ NL=6, 400 steps, real 290B mix (lower is better):
| variant | loss | note |
|---------|------|------|
| PC-settling (muPC+STOP) | 7.52 | 8.2x cost -- the old approach |
| layerwise, uniform weights | 6.596 | 0.7x cost |
| + DFA broadcast (lambda=1) | 6.528 | lambda>=2 diverges |
| + embedding local objective | **6.465** | embedding is the largest param block; layer-0 din alone starved it |
| depth-weighted / final-heavy weights | 6.66 / 6.75 | WORSE -- uniform is best |
| final-layer-only objective | 7.16 | confirms per-layer objectives do real work |

DEPTH SCALING (each arm at its OWN best LR: BPTT 1e-2, layerwise 3e-3), 400 steps:
| NL | BPTT | LAYERWISE-local | gap | cost |
|----|------|-----------------|-----|------|
| 2  | 5.745 | 6.581 | +0.836 | 0.71x |
| 4  | 5.734 | 6.509 | +0.774 | 0.75x |
| 6  | 5.742 | **6.467** | **+0.725** | 0.74x |
| 12 | 5.721 | 6.682 | +0.961 | 0.73x |
- **DEPTH HELPS the local learner**: layerwise improves 6.581 -> 6.467 from 2 to 6 layers and the gap SHRINKS
  (+0.836 -> +0.725), while BPTT is flat. Sweet spot ~4-6 layers; at 12 it degrades (+0.961).
- **Cost is 0.73x BACKPROP** (cheaper than backprop itself, and ~12x cheaper than PC settling). Huge 0b2 win.
- **Depth is no longer a credit-assignment problem**: nothing to settle, nothing to propagate, scales to any depth.
- REMAINING GAP vs backprop ~ +0.72 at 400 steps. Cause identified: **layerwise is LR-LIMITED** (best at 3e-3,
  diverges >=5e-3) while backprop happily uses 1e-2 -- all layers step simultaneously without knowing about each
  other. This is a STABILITY problem, not a gradient-direction problem.
- Untried levers for the gap: longer training (400 steps is tiny -- does the gap close?), similarity/InfoMax auxiliary
  term (Nokland & Eidnes), proper per-layer adaptive step control, blockwise objectives.
- (A global-norm gradient-clipping attempt diverged -- buggy interaction with Adam's scale invariance; discarded, not
  a finding.)

## Build step 1h: LONG-HORIZON test FAILED -> found the real bug -> RULE THAT BEATS BACKPROP
Per user instruction ("do long horizon once, if it fails grind stability"), ran 5000 steps. Two findings:

**(i) The core was missing NORMALIZATION (a real bug I introduced).** The layerwise run DIVERGED: |a_top| went
0.87 (step 700) -> 1.99 (800) -> 3080 (900) -> NaN. My minimal core.py had NO norm at all (the original VChunkRecall
block had pre-LayerNorm); with an unbounded residual stream every layer greedily grows its contribution to the shared
readout -> runaway. FIX: RMS pre-norm on the projections + a final norm before the readout, with an EXACT local
norm-VJP (verified: control cosine 1.000000). Divergence gone; every LR now stable (no NaN even at 2e-2).

**(ii) Long-horizon on the fixed core: the gap GREW (greedy saturation).** local matched backprop exactly for 500
steps (6.4285 vs 6.4257!) then PLATEAUED at ~6.1 while backprop descended to 5.16:
  gap +0.003 (500) -> +0.392 (1000) -> +0.661 (2000) -> +0.926 (4000).
Diagnosis: once each layer is locally optimal there is no pressure to build features that only pay off DEEPER.

**(iii) THE FIX: RESIDUAL-HIGHWAY (IDENTITY) FEEDBACK.** Because layers are RESIDUAL and share the readout space, the
true gradient is dL/da_l = dL/da_top * prod(I + rs*J), whose LEADING TERM IS EXACTLY dL/da_top. So broadcast
dL/da_top UNCHANGED to every layer (a global error broadcast, neuromodulator-like: NO weight transport, NO backward
pass) instead of through a random DFA matrix. Ablation @NL=6, 2000 steps (BPTT ref 5.5359):
| rule | loss |
|------|------|
| local objectives + RANDOM DFA (previous) | 6.2189 |
| identity feedback ONLY, no local objectives | 9.3475 (fails alone) |
| local + IDENTITY feedback lam=1 | 5.3615 |
| **local + IDENTITY feedback lam=2** | **5.3588 (BEATS backprop by 0.18)** |
| local + IDENTITY lam=4 | 5.3639 |
| local(w=0.25) + IDENTITY lam=4 | 5.4118 |
- **Both components are ESSENTIAL and complementary**: local objectives give every layer a strong well-scaled signal
  (deep supervision) but plateau; identity feedback carries the global error so layers build features that help above,
  but fails alone. Together they BEAT backprop, at 0.73x its cost, with NO backprop anywhere.
- Robust across lambda=1,2,4. lam=2 is the operating point.

## Build step 1i: LONG-HORIZON CONFIRMED -- pure local plasticity BEATS backprop at depth, over 4000 steps
Final rule (local per-layer objectives + residual-highway identity feedback + exact eligibility credit, on the
normalized core) vs BPTT, NL=6, 4000 steps, real 290B mix, each at its own best LR:
|  step |   BPTT | LOCAL rule |    gap |
|-------|--------|------------|--------|
|   500 | 6.4257 |     5.9252 | -0.500 |
|  1000 | 5.8110 |     5.4814 | -0.330 |
|  2000 | 5.5359 |     5.3146 | -0.221 |
|  3000 | 5.3321 |     5.1852 | -0.147 |
|  4000 | 5.1639 |     5.0370 | **-0.127** |
- **The local rule is AHEAD at every checkpoint**, and it does NOT plateau (the previous random-DFA version plateaued
  at ~6.1). Cost 0.73x backprop. NO backprop, NO settling, NO weight transport, NO autograd anywhere.
- The margin NARROWS over training (-0.50 -> -0.13); part of the early advantage is likely the deep-supervision
  regularization effect. Watch for a crossover at much longer horizons.
- HONEST SCOPE: small model (D=96, 6 layers, ~3M non-embedding params), short context (T=32), 4000 steps (~4M
  tokens). Whether the advantage holds at larger scale / longer context is UNTESTED and is the next thing to check.

### STATUS: VISION 0c (pure local plasticity) is fully validated, and depth is unlocked.
Ready to build the real persistent core: bounded-but-real depth + wide + recurrent, grown from infant, with this rule.

## Build step 2: FAST CORE + GROWTH (user: "make it really fast ... test it with scale and growth")
**(a) Vectorized core (core.py rewritten).** The per-timestep Python loop is gone: the causal recurrence
num_t = sum_{s<=t} (qf_t . kf_s) v_s is computed in PARALLEL as a masked quadratic form, and the local VJP likewise.
Verified EXACT vs autograd (cosine 1.000000, norm-ratio 1.000000). **8x faster** at the old config (0.18 -> 0.022
s/step) and far less memory (no per-step state materialization).
**(b) Efficiency at scale + fix.** A per-layer local objective costs a FULL 32K-vocab projection, so cost grew to
6.5x backprop at D512/NL12. Fix: **stochastic local objectives** -- sample k of NL layers per step and rescale by
NL/k (expected update unchanged). Cost 6.49x -> 3.04x (k=4) -> 2.22x (k=2) -> 1.80x (k=1).
**(c) GROWTH (growth.py), all axes FUNCTION-PRESERVING (verified max|dOut|):** depth +2 = 0.00e+00; state/DV +64 =
1.67e-06; width/D +64 = 1.79e-06; vocab -> 33000 = 0.00e+00. Width growth is subtle under RMS-norm (adding zero dims
changes mean(x^2)): solved by rescaling stream-writers by c=sqrt((D+add)/D) and stream-readers by 1/c.
**(d) GROWTH + LOCAL RULE (grow_test.py), T=64, 6000 steps:**
| arm | final loss | compute |
|-----|-----------|---------|
| fixed at target (D192, NL6) | 4.9811 | 1.00x |
| infant (D96,NL2) GROWN to target | 5.0478 | **0.72x (28% saved)** |
- Growth reaches ~the same quality for **28% less compute** ("cheaper-to-target", as the vision predicted), with NO
  thrash or collapse -- unlike the old MoE/dense-grow attempts. Function-preserving + local rule is a stable combo.
- FINDING: the grown arm was BETTER at step 4000 (4.9877) than at 6000 (5.0478) -- the depth-grow at 4800 caused a dip
  it had not recovered from by the end. **Grow events need recovery time** (and the optimizer state is reset at each
  grow). Confirms the "minimum steps between grows" rule; do not grow late in a run.

## Build step 3: SCALE TEST -- user's hypothesis CONFIRMED (local rule loses its advantage at scale)
User predicted from the gap trend (-0.500 -> -0.330 -> -0.221 -> -0.127, roughly halving per doubling) that backprop
would catch up and surpass. Tested at D=256, NL=6, T=128, 20000 steps (~82M tokens).
- FIRST RESULT WAS WRONG AND I CAUGHT IT: the scale run said "local ahead by -0.76 at every checkpoint", but BPTT was
  using lr=1e-2 (tuned at the SMALL scale). **BPTT's optimal LR shifts with scale (1e-2 -> 1e-3).** Under-tuned BPTT
  scored 5.71; properly tuned it scores **4.5590**. Always re-tune BOTH arms at each scale.
- FAIR COMPARISON AT SCALE (both arms LR-swept, 4000 steps, D256/T128):
  | arm | best loss | best LR |
  |-----|-----------|---------|
  | BPTT | **4.5590** | 1e-3 |
  | LOCAL rule | 4.7077 | 1e-3 |
  **gap = +0.149 -> the local rule TRAILS backprop at scale** (it BEAT backprop by -0.127 at D96/T32).
- So the advantage REVERSES with scale, exactly as the user predicted. The small-scale win was likely the
  deep-supervision regularization effect, which fades as the model gets bigger.
- EFFICIENCY ALSO REVERSES: at D96/T32 local cost 0.73x backprop; at D512/NL12 it costs 6.5x (1.8x with k_local=1)
  because every per-layer objective needs a full 32K-vocab projection. At scale the local rule is BOTH worse AND
  more expensive than backprop.
- Stochastic local objectives (k_local=2) DO preserve quality (tracked the full version: 4.5586 vs 4.5666 at step
  12000) at ~2.9x less cost -- that fix is validated and worth keeping.

### HONEST STANDING POSITION on VISION 0c (pure local plasticity, no backprop ever)
It WORKS: trains real language, at real depth, supports function-preserving growth, no backprop/settling/weight
transport anywhere. But at the scales tested it is **not free**: it trails backprop by ~0.15 loss and costs more per
step at scale. The "beats backprop" result holds only at small scale. This is a real tension with 0b2 (efficiency)
and with capability, and it is a decision for the project owner, not something to paper over.

## Build step 4: ADAPTIVE LR (solved) + GENERATION TEST (new evidence)
**(a) ADAPTIVE, SCALE-FREE LR -- WORKS.** Replaced the absolute LR with a trust-ratio (LAMB-style) RELATIVE step:
each parameter moves a fixed fraction of its own norm. `min_norm` lets zero-initialized new capacity (function-
preserving growth) start moving instead of being frozen at trust=0.
| rel-step | SMALL (D96/T32) | LARGE (D256/T128) |
|----------|-----------------|-------------------|
| 3e-4 | 6.9065 | 6.3117 |
| **3e-3** | **5.3293** | **4.8676** |
| 1e-2 | 5.5674 | 5.8240 |
**The SAME value (3e-3) is optimal at BOTH scales** -> no per-scale LR tuning, and growth is safe (no retuning at
each grow). This also satisfies VISION 0a (no hand-tuned per-scale constants). Cost: slightly worse than a
hand-tuned fixed LR at scale (4.7540 vs 4.7077), a fair trade for scale/growth invariance.
- NOTE another scale-dependence found: the feedback weight lambda should also grow with scale (lam=2 best small,
  **lam=4 best large: 4.7443**). Making lambda intrinsic/adaptive is the next candidate.

**(b) GENERATION TEST (gen_eval.py, gen_run.py) -- D256/NL6/T128, 6000 steps, head-to-head:**
| metric | LOCAL rule | BPTT |
|--------|-----------|------|
| heldout loss | 4.7317 (ppl 113.5) | **4.4066 (ppl 82.0)** |
| distinct-1 / distinct-2 | 0.54 / 0.69 | **0.66 / 0.87** |
| 4-gram repetition | **0.23 (degenerate)** | **0.02 (healthy)** |
| in-context recall | **10%** | 5% |
| train wall-time | 267s | **80s (3.3x cheaper)** |
- Generation makes the gap MORE visible than loss alone: the local model loops ("capital for communionionionion...")
  while BPTT's samples stay varied. 4-gram repetition 0.23 vs 0.02 is a big quality difference.
- The ONE place local leads is in-context RECALL (10% vs 5%), our signature faculty -- but n=20, so this is
  suggestive, not established.
- LESSON: always evaluate GENERATION, not just loss. Two models 0.33 apart in loss differ far more in output quality.

### STANDING POSITION (updated, honest)
At scale the pure-local rule is worse on loss (+0.15 to +0.33), noticeably more degenerate in generation, and
~3.3x more expensive per step. It DOES work (real language, real depth, growth, no backprop anywhere) and it WINS at
small scale, but the "beats backprop" claim does not survive scale. Adaptive LR removed the tuning problem but not
the quality gap.

## Build step 5: FADE hypothesis FALSIFIED -> the scale gap is structural -> 0c amended
Hypothesis: the local rule wins small / loses large because per-layer objectives are deep supervision that
over-constrains a large model, so local supervision should FADE. **Falsified** (D256/T128, 4000 steps):
| schedule | loss |
|----------|------|
| constant w=1 (baseline) | **4.7443** |
| fade 1->0 linear | 4.7748 |
| fade 1->0 by 50% | 4.8985 |
| fade 1->0.25 | 4.7627 |
| fade 1->0 + lam 2->8 handover | 4.7834 |
Every fade schedule is WORSE. Local supervision is needed throughout and its cost is INHERENT.
Leads now exhausted: LR sweeps (both arms), adaptive scale-free step, lambda tuning, stochastic local objectives,
identity-vs-random feedback, normalization, fading. The +0.15 scale gap is STRUCTURAL.
=> DECISION (delegated by the user): VISION 0c amended to **local plasticity is the LIFELONG learning rule**;
the one-time substrate may be pretrained by whatever is most efficient. Everything that differentiates Pragnosia
(continual learning at inference, Hebbian hippocampus, propose->verify->consolidate, growth, intrinsic faculties,
identity) is preserved -- none of it depends on how the substrate was first formed. Pure-local pretraining remains
an active track with a falsifiable promotion criterion. Also more biologically accurate: neuroplasticity is LIFELONG
learning; the initial cortex is shaped by development, not adult plasticity.

## Build step 6: PERSISTENT MEMORY + UNBOUNDED CONTEXT (the vision's differentiating capability)
**(a) Hippocampus bolted on = does NOTHING (honest negative).** Wrote facts into a persistent fast-weight store `M`
via a gated Hebbian write (no backprop) on a model trained WITHOUT memory: retention delta = -0.045 / -0.108
(slightly negative), interference -0.003, pollution +0.008. Diagnosis: **a model that never learned to use a memory
cannot read one** -- its query projections were never trained to retrieve from `M`, so writes are just noise.
**(b) FIX = train with the memory in the loop** (core.py: `forward(states=...)`, per-layer (M,zM) carried ACROSS
chunks, detached between chunks -> no cross-chunk backprop, so memory stays CONSTANT however long the document is).
State shape is (B,FE,DV) per layer -- **independent of context length**: the constant-memory claim is now structural.
Regression-checked: local_vjp still EXACT (cosine 1.000000) after the change.
**(c) DECISIVE TEST (longctx.py, 4 chunks x T=128 = 512-token documents, 3000 steps):** per-chunk loss WITH carried
state vs state zeroed:
| trained | chunk1 | chunk2 | chunk3 | mean gain |
|---------|--------|--------|--------|-----------|
| **WITH carry** | **+0.220** | **+0.213** | **+0.225** | **+0.2196** |
| without carry | -0.013 | -0.146 | -0.222 | -0.127 |
- **Memory HELPS by ~0.22 nats on every chunk after the first when the model is trained with it, and HURTS when it
  is not.** The model has learned to write retrievable traces and query them. This is the prerequisite for
  continual learning at inference, and it validates the (a) diagnosis exactly.
- CAVEAT (honest): the carry-trained model's ABSOLUTE loss is worse (5.59-5.83 vs 4.55-4.78 for the no-carry arm).
  Likely cause: with decay=1.0 the state accumulates unboundedly across chunks, so by later chunks the denominator is
  dominated by old writes and the retrieval is diluted. NEXT: add a decay/forgetting factor on the carried state
  (biologically right anyway -- memory must forget) and re-measure both absolute loss and memory gain.

## Build step 7: FORGETTING (the caveat resolved) -- memory now helps AND absolute quality is restored
Swept the decay applied to the carried state between chunks (3000 steps each, D256/NL6/T128, 4 chunks):
| decay | mean loss (with memory) | memory gain |
|-------|------------------------|-------------|
| 1.00 (no forgetting) | 5.6614 | +0.2196 |
| 0.90 | 5.5984 | +0.2403 |
| 0.70 | 5.1828 | +0.1991 |
| 0.50 | 4.8689 | +0.2143 |
| **0.20** | **4.5982** | **+0.1812** |
- **Forgetting is essential**: absolute loss improves 5.66 -> 4.60 (now on par with the no-memory arm's 4.55-4.78)
  while the model KEEPS a +0.18 memory gain. Without forgetting the state accumulates and drowns the current chunk.
- This is exactly what the vision's decay mechanism predicted, and what biology does: a memory that never forgets is
  a memory that cannot retrieve. Operating point ~0.2-0.5 per chunk (keep both quality and gain).
- STATUS: the model now has a WORKING persistent memory it learned to use, at CONSTANT cost in context length.

## VISION ACCEPTANCE TEST 1 -- "constant-memory, constant-compute, unbounded/streaming context": **PASSES**
(acceptance1_context.py; D=256 NL=6, streaming in 128-token chunks with carried+decayed state)
| context | us/token | peak MiB | same-size transformer KV (MiB) | its attention FLOPs |
|---------|----------|----------|-------------------------------|---------------------|
| 512 | 27.92 | 117.8 | 6 | 1x |
| 8192 | 27.03 | 117.8 | 96 | 256x |
| 32768 | 26.83 | 117.8 | 384 | 4096x |
| **131072** | **26.86** | **117.8** | **1536** | **65536x** |
- **Peak memory EXACTLY flat (1.00x) and per-token latency flat (0.96x) across a 256x longer context.** The claim is
  measured, not asserted. This is the project's most defensible differentiator.
- HONEST SCOPE: this is a COST claim (architecture-level, valid). It does NOT claim the model USES 131k tokens well --
  this model was trained on 512-token documents. Long-context ACCURACY needs a model trained at length; separate test.

## VISION ACCEPTANCE TEST 2 -- "continual learning at inference": **HALF PASSES** (fast path yes, persistence no)
v1 was invalid: `absorb()` padded facts to 128 tokens with token 0, so the memory captured PAD, not the fact
(retention -0.77/+0.82, interference +1.41). Fixed (no padding) and added a NEUTRAL-text control of similar length.
| test | result |
|------|--------|
| **2a RETENTION** | **PASS: mean +1.467 nats vs control** (7492 +0.57, luminase +2.59, Kelmoor Spire +1.24) |
| **2c INTERFERENCE** | **PASS: +0.025** vs the neutral-state control (negligible) |
| **2b PERSISTENCE** | **FAIL: gain +0.569 -> +0.038 after ONE unrelated chunk, ~0 thereafter** |
- The model **genuinely learns novel facts at inference** by a local Hebbian state write (no backprop, no gradient):
  entities it has never seen ("Doctor Marlow"/"luminase", "Vashmere"/"Kelmoor Spire") get 1.2-2.6 nats more likely.
- It does so **without damaging** general language ability.
- But the fast store DECAYS (0.2/chunk, chosen for LM quality) so the fact is gone within a chunk. That is the
  hippocampus behaving correctly -- and exactly why the vision requires CONSOLIDATION into slow weights for anything
  that should persist. The missing piece is the sleep/consolidation loop, not the memory.
- NOTE the real trade: the decay that makes language modelling good (0.2) is far too aggressive for persistence.
  Fast memory is short-term BY DESIGN; permanence must come from consolidation.

## CONSOLIDATION ("sleep") -- the persistence gap CLOSED (acceptance3_consolidate.py)
Fast memory forgets within a chunk by design; consolidation must move a verified episode into the SLOW WEIGHTS.
Implemented with PURE LOCAL PLASTICITY (0c: lifelong learning is local), REPLAY of general data interleaved
(anti-forgetting), and CHECK-AND-COMMIT with ROLLBACK (7c). Measured with the fast memory CLEARED, so any gain must
live in the weights:
| consolidation steps | fact logP gain | general-ability damage | decision |
|---------------------|----------------|------------------------|----------|
| **30 (1 second)** | **+6.06** | **-0.155 (improved)** | COMMIT |
| 100 | +6.46 | +0.058 | COMMIT |
| 300 | +6.18 | +0.150 | COMMIT |
- **30 local steps (~1s) permanently install a novel fact at +6 nats with NO damage to general ability.** The
  persistence failure of acceptance 2b is closed by the mechanism the vision already specified.
- The whole loop -- inference-time Hebbian write -> consolidation into weights -- is now BACKPROP-FREE, which is
  exactly what amended 0c requires (local plasticity governs the model's life).
- Operating point: ~30 steps (more steps add damage without adding gain).
- HONEST: this is episode memorization (the fact text is replayed), which is what fact-learning IS; the guard is the
  general-ability check plus rollback. What is still MISSING is VERIFICATION -- nothing yet stops a FALSE fact from
  being consolidated the same way. That is the anti-poisoning half of flag 2 and the next build.

## VISION ACCEPTANCE TEST 3 -- EFFICIENCY vs a REAL transformer baseline (never measured before)
Baseline: same D=256 / 6 layers / 32K vocab, pre-norm MHA using torch's FUSED scaled_dot_product_attention (flash
kernels) -- a strong baseline, not a naive one. Ours trains documents in 128-token CHUNKS with carried state; the
transformer must attend over the whole document. Batch 8. (Both arms measured under identical load.)
| doc len | OURS peak MiB | TF peak MiB | memory ratio | OURS ktok/s | TF ktok/s | speed |
|---------|---------------|-------------|--------------|-------------|-----------|-------|
| 128 | 914 | 1088 | 1.19x | 88.4 | 92.0 | 0.96x |
| 512 | 1187 | 2534 | 2.13x | 76.1 | 134.4 | 0.57x |
| 2048 | 1187 | 8321 | 7.01x | 83.4 | 118.6 | 0.70x |
| **8192** | **1188** | **31472** | **26.5x** | 82.5 | 62.3 | **1.32x** |
**Q1 TRAINING MEMORY: ~0.9-1.2 GiB for a 17.6M model at batch 8, and FLAT in document length** (1187.3 MiB at 512
tokens -> 1187.8 MiB at 8192). The transformer's grows 12x over the same range and would OOM where we stay flat.
**Q3 TRAINING EFFICIENCY vs existing-LLM style: it depends on context length, and the crossover is ~4-8k tokens.**
- Below ~2k tokens the transformer is FASTER (we run at 0.57-0.70x) -- flash attention is extremely well optimized
  and our Taylor feature map carries real overhead. We are NOT faster at short context; that is the honest picture.
- At 8192 tokens we are **26.5x lighter AND 1.32x faster**, and the gap widens with length (their cost is quadratic,
  ours is constant per token).
- ACTIONABLE: our efficiency advantage is specifically LONG-DOCUMENT training. Training on 128-token chunks (what we
  have been doing) is our WORST case. Train on long documents to actually collect the win.
**Q2 GROWTH SCALING (doc len 2048): graceful, sublinear in memory**
| config | params | peak MiB | s/step |
|--------|--------|----------|--------|
| D128 x 4L | 8.6M | 1323 | 0.135 |
| D256 x 6L | 17.6M | 1456 | 0.197 |
| D384 x 8L | 27.6M | 1618 | 0.260 |
| D512 x 12L | 40.0M | 1840 | 0.388 |
- 4.6x more parameters costs only **1.39x memory** and 2.9x time. Memory is dominated by the fixed 32K-vocab
  embedding/head + optimizer state, not by the sequence, so growth scales gently. Growth is affordable.
- CAVEAT: the transformer baseline has MORE params (23.6M vs 17.6M) because MHA uses four DxD projections while our
  key/query projections are small (DIN=8); this slightly favours us on memory. Small scale (17.6M) throughout.

## SCALING AT 32k DOCUMENTS: 40M -> 1B, ours vs param-matched transformer (the decisive efficiency result)
**FIRST, A BUG IN OUR OWN SETUP: chunk size was costing us 10x.** All session we used chunk=128. At 32768-token
documents that is 256 sequential steps of tiny kernels -- terrible GPU utilization. Sweeping chunk size:
| chunk | 40M s/step | ktok/s |   | 200M s/step | ktok/s |
|-------|-----------|--------|---|-------------|--------|
| 128 | 5.84 | 11.2 | | 11.65 | 2.8 |
| 1024 | 0.75 | 87.8 | | **1.60** | **20.5** |
| **2048** | **0.58** | **113.2** | | 1.66 | 19.8 |
**Optimal chunk is 1024-2048, not 128.** (Gradient accumulation was NOT the issue -- tested, made no difference.)
**DEFINITIVE (32768-token documents, optimal chunk, param-matched transformer with fused/flash attention):**
| scale | params | OURS s/step | GiB | ktok/s | TF s/step | GiB | ktok/s | FASTER | LIGHTER |
|-------|--------|-------------|-----|--------|-----------|-----|--------|--------|---------|
| 40M | 40M | 0.58 | 3.64 | 113.5 | 0.81 | 29.06 | 80.7 | **1.41x** | **7.99x** |
| 200M | 215M | 1.60 | 4.83 | 20.4 | 2.52 | 28.44 | 13.0 | **1.57x** | **5.89x** |
| 400M | 437M | 2.93 | 9.11 | 11.2 | OOM | -- | -- | **transformer cannot run** |
| 1B | 958M | 6.30 | 23.05 | 5.2 | OOM | -- | -- | **transformer cannot run** |
- **We are FASTER (1.4-1.6x) AND LIGHTER (6-8x) than a param-matched transformer at 32k context**, and past ~400M the
  transformer cannot train at 32k in our available ~45 GiB at all, while we train 1B in 23 GiB.
- **Memory scales sub-linearly with growth**: 24x more parameters (40M -> 958M) costs only 6.3x memory
  (3.64 -> 23.05 GiB), because memory is dominated by parameters/optimizer, never by context length.
- Growth adds a further ~28% compute saving on the way up (measured earlier), so reaching 1B by GROWING rather than
  training at target size should cost ~0.72x of these numbers.
- CAVEATS: batch=1 for the big models (sharing the GPU with prod), so absolute ktok/s is pessimistic; the transformer
  OOM is against our ~45 GiB of free memory, not a full 80 GiB card; ours has no MLP per layer (2*D^2) while the
  transformer's is ~12*D^2, so at matched params ours is much deeper -- part of why we differ.
- ACTION: switch all training to long documents with chunk 1024-2048. 128-token chunks were our worst case on BOTH
  axes (10x slower, and it forfeits the long-context advantage entirely).

## VISION ACCEPTANCE TEST 4 -- ANTI-POISONING (verification gates consolidation): **PASSES end-to-end**
Built verify.py: the young model's judge is MECHANICAL EVIDENCE RETRIEVAL over the trusted 290B corpus -- find the
claim's context pattern and look at what actually FOLLOWS it. That distribution separates the three cases the vision
needs, which a prior-confidence check cannot: ACCEPT (corpus agrees) / REJECT (corpus says something else) /
HOLD (pattern absent -> novel or unknown; never made permanent).
Two design lessons found by failing first:
- Raw token CO-OCCURRENCE is a bad signal (frequency-biased: "Japan...Cairo" scored high by chance).
- Evidence keys must contain a CONTENT token. Short keys like "is called" match thousands of unrelated sentences and
  wrongly REJECTed a novel claim on irrelevant evidence.
Gate accuracy 3/5 on the probe set, but the SAFETY property is 100%: **no false claim was ever ACCEPTed**; the two
misses are conservative HOLDs on a true fact whose phrasing is absent from the corpus (coverage, not logic).
**END-TO-END (acceptance4_endtoend.py, real substrate, consolidation gated by the verdict):**
| claim | verdict | consolidated | logP before | after | installed |
|-------|---------|--------------|-------------|-------|-----------|
| TRUE "capital of France is Paris" | ACCEPT | YES | -8.587 | -0.304 | **+8.282** |
| FALSE "capital of France is Berlin" | REJECT | blocked | -10.331 | -10.331 | **0.000** |
| NOVEL "Vashmere ... Kelmoor" | HOLD | blocked | -8.938 | -8.938 | **0.000** |
- A verified fact becomes PERMANENT knowledge (+8.3 nats). A false one NEVER reaches the weights. A novel one stays
  in fast memory only. **VISION flag 2 is now complete**: learns at inference, retains by consolidation, cannot be
  poisoned.
- TEST FLAW to fix: general() draws fresh random batches for before/after, so its "damage" column is pure noise for
  the blocked cases (whose weights are byte-identical). Use fixed eval batches.

### ACCEPTANCE SCOREBOARD
| # | claim | status |
|---|-------|--------|
| 1 | constant-memory / constant-compute unbounded context | **PASS** (flat to 131k tokens) |
| 2 | verified continual learning at inference | **PASS** (retention + consolidation + anti-poisoning) |
| 3 | efficiency vs a real transformer | **PASS** (1.4-1.6x faster, 6-8x lighter at 32k; they OOM past 400M) |
| 4 | one genuinely INTRINSIC decision (growth fires on the model's own signal) | **NEXT** |

## VISION ACCEPTANCE TEST 5 -- ONE GENUINELY INTRINSIC DECISION: **PASSES** (the growth schedule is deleted)
VISION 0a demands the model decide WHEN to grow, not a schedule I write. Built `growth.SaturationSensor`:
signal = the model's OWN predictive entropy (label-free -- needs no targets, it is its uncertainty about its own
outputs); criterion is NOISE-RELATIVE rather than a magic threshold -- fit the entropy's slope over a window and
compare it to that slope's own standard error. When entropy stops falling faster than its own noise, the model has
extracted what it can at this capacity and ASKS to grow.
| arm | final loss | compute | grows fired at |
|-----|-----------|---------|----------------|
| fixed at target (D192/6L) | 4.5554 | 76s | -- |
| scheduled (hand-written steps) | 4.6445 | 58s | 1200, 2400, 3600, 4800 |
| **INTRINSIC (model's own signal)** | **4.5499** | 72s | **622, 1240, 1740, 2242** |
- **The intrinsic trigger BEATS the hand-written schedule (-0.095) and even edges the fixed model (-0.006).**
- The model chose to grow MUCH EARLIER than I scheduled -- it detected saturation at small size faster than I
  guessed. My schedule was simply too conservative. This is exactly why the decision should not be mine.
- The hand-written growth schedule can now be DELETED. One real piece of scaffolding removed (0a).

### ACCEPTANCE SCOREBOARD -- all four claims now pass
| # | claim | status |
|---|-------|--------|
| 1 | constant-memory / constant-compute unbounded context | **PASS** (flat to 131k tokens) |
| 2 | verified continual learning at inference | **PASS** (retention + consolidation + anti-poisoning) |
| 3 | efficiency vs a real transformer | **PASS** (1.4-1.6x faster, 6-8x lighter at 32k; they OOM past 400M) |
| 4 | one genuinely INTRINSIC decision | **RETRACTED** (sensor fired on noise at min_gap; re-test pending with the fixed sensor) |
REMAINING SCAFFOLDING (0a is not finished): salience/propose gating, the verification judge (mechanical, not learned),
consolidation timing, and the constants lambda / decay / chunk are still chosen by hand.
### LOGGING
`util.setup_log(name)` tees every run to `logs/<name>.log`; `util.pbar` prints a tqdm bar in a terminal but CLEAN
periodic lines when writing to a file (a tqdm bar's carriage returns make a log unreadable -- fixed after seeing it).
Monitor any run with: `tail -f logs/<name>.log`

## ARCHITECTURE FIX: the layers had NO MLP -- worth 1.17 nats (found before committing to a long run)
Our Layer was attention-only (~2*D^2); a transformer layer is ~12*D^2 because of its feedforward block, and MLPs
carry most of a model's knowledge. Tested at a MATCHED total budget (~92M params, 4000 steps, same data):
| architecture | non-embedding params | loss |
|--------------|---------------------|------|
| attention-only D1024 x 12L | 25.4M | 5.6087 |
| **with MLP D640 x 12L** | **49.4M** | **4.4391** |
**The MLP is worth 1.17 nats.** Two effects compound: the MLP itself, AND the smaller D frees embedding budget for
real capacity (49.4M vs 25.4M non-embedding at the same total). Related observation that explains a lot of this
session's weak generations: our D=256 test models were **95% embedding** -- layers were only 0.8M of 17.6M params,
so there was almost no "brain" there at all. Small D with a 32K vocab is embedding-dominated; use larger D and MLPs.
- MLP added to core.Layer with an EXACT local VJP (verified cosine 1.000000, norm-ratio 1.000000), so the pure-local
  learning rule still applies unchanged.
- growth.py updated: width-growth now rescales m1/m2 correctly under RMS-norm, and depth-growth zeroes BOTH residual
  branches (Wo AND m2) or a new layer would not be the identity. All four axes re-verified function-preserving.

## PERFORMANCE: the real run was 8.8x slower than it should have been (found by the user pushing back)
Every one of my configuration choices was wrong, and the biggest was one I had not even considered.
| lever | before | after | gain |
|-------|--------|-------|------|
| **dtype** (I had NO autocast -- pure FP32 on an H100) | 96 ktok/s | 246 | **2.6x** |
| **chunk** 1024 -> 2048 | 246 | 409 | 1.65x |
| **doc** 8192 -> 32768 | 409 | 486 | 1.19x |
| batch 4 -> 8 | 486 | 511 | 1.05x |
| **torch.compile** | -- | -- | **1.58x** (measured at BOTH 31M and 235M) |
| **TOTAL (live)** | **96 ktok/s** | **844 ktok/s** | **8.8x** |
- bf16 VERIFIED SAFE before adopting: identical loss curves (4.368/4.339, 5.284/5.286, ...), held-out difference
  **-0.0003**, 2.82x faster. Not just fast -- the same model.
- POWER: the H100 NVL here is capped at **400 W** (power.limit = power.max_limit = 400 W) and we draw 395 W = 99%.
  My earlier "398W of ~700W means there is headroom" was WRONG. torch.compile still wins because fusion does LESS
  work (memory traffic), not because it fills idle time.
- My defence of doc 8192 / chunk 1024 came from a benchmark run at batch=1 in fp32. Both assumptions were wrong at
  the real batch size. Benchmark at the configuration you will actually use.
- MEMORY AT TARGET verified: 235M at doc 32768 / chunk 2048 = 12.0 GiB (B=4) or 21.9 GiB (B=8). No scaling problem.
## MLP-IN-GROWTH BUG (found by the user asking why the MLP was not in the logs)
The MLP was present (15.2% of params) but width-growth kept the MLP HIDDEN dim FIXED while D grew, so the ratio
collapsed 4.0x -> 2.4x -> would have been 1.5x at target, silently eroding the measured 1.17-nat MLP advantage as the
model grew. Fixed: hidden now scales with D, ratio held at 4.00x through every grow, function-preservation intact
(max|dOut| 3e-06), and the MLP width is now printed in the log line.

## GROWTH BUG #2: the VALUE/STATE dim did not widen with D (found when a checkpoint failed to load)
`grow_width` widened D but left DV (the recurrent value dim) fixed: after one grow, Wv was [384, 512] -- D=512,
DV=384. So the model widens while its RECURRENT MEMORY CAPACITY (FE x DV) stays frozen, which directly undercuts the
recall/memory claim. Same class of bug as the MLP-hidden one, and equally silent: training runs fine, the capability
just never arrives. Fixed: D, DV and the MLP hidden dim now scale together, all four growth axes re-verified
function-preserving. Checkpoints now record DV so a grown model can always be reconstructed.
LESSON: whenever a dimension is derived from D at construction time, growth must scale it too. Verify by RELOADING a
grown checkpoint, not just by checking dOut -- the load is what caught this.

## SENSOR BUG: the "intrinsic" growth trigger was firing on NOISE -- i.e. it was a hidden schedule
Growth timings in the live run were 818, 1618, 2453, 3258, 4058 -- gaps of 800/835/805/800 against min_gap=800. It
fired the instant it was allowed, every time. Diagnostic (exp/sensor_test.py) with three synthetic entropy traces:
| trace | current sensor (window=150, raw) | fixed (window=400, EMA 0.02) |
|-------|----------------------------------|------------------------------|
| A still improving -> must NOT fire | no fire (correct) | no fire (correct) |
| B plateaued -> MUST fire | fires (correct) | fires (correct) |
| **C improving but NOISY -> must NOT fire** | **FIRES (wrong)** | **no fire (correct)** |
Real entropy is case C, so the trigger was effectively "grow every min_gap steps": a SCHEDULE wearing an intrinsic
costume. Consequence in the run: 5 grows in 4000 steps, no time to consolidate, and generation DEGRADED
(4-gram repetition 0.34 -> 0.54, distinct-1 0.40 -> 0.24 between step 2000 and 4000) even as loss crept down.
**THIS INVALIDATES THE ACCEPTANCE-5 CONCLUSION.** That test's growth gaps were 618/500/502 against min_gap=500 -- the
same artifact. The intrinsic arm "beat the schedule" only because it grew SOONER, not because it detected saturation.
Acceptance 5 must be re-run with the fixed sensor before the claim stands.
FIX: smooth the entropy internally (EMA 0.02) before the slope test, default window 400, min_gap 2000. Re-verified:
fires only on the true plateau, ignores the noisy-but-improving trace.
LESSON: when a "learned/intrinsic" trigger fires at exactly its rate limit, it is not deciding anything. Test any
such sensor against synthetic traces where you KNOW the right answer.

## DEGENERACY SCARE: it was MY EVAL, not the model -- and MEMORY FIXES REPETITION (17x)
Repetition appeared to be rising (0.52 -> 0.77) while loss fell, which looked like a real failure. Tested the two
hypotheses (exp/degeneracy_test.py, step-4000 checkpoint):
| condition | 4gram-repetition | distinct-1 |
|-----------|------------------|------------|
| no memory (what my eval did) | 0.50 | 0.22 |
| **with 3 chunks of carried memory** | **0.03** | **0.64** |
| with memory, no decay | 0.02 | 0.61 |
| real training text (reference) | 0.00 | 0.78 |
| temp 0.0 / 0.5 / 0.8 / 1.0 / 1.2 (no memory) | 0.90 / 0.85 / 0.42 / 0.18 / 0.09 | -- |
- **H2 (our memory design causes repetition) is REFUTED -- the opposite is true: memory REDUCES repetition 17x**
  (0.50 -> 0.03), to near the level of real text. The carried state gives the model context to be varied about.
- The "collapse" was an EVAL ARTIFACT: I was generating with memory DISABLED at temp 0.8 -- the two worst choices.
  The model is never deployed that way.
- FIX: evaluate.py now generates WITH carried memory at temp 1.0, i.e. as the model actually runs. History reset,
  since the old generation column was measuring a configuration that does not exist in practice.
- LESSON: measure the model in the configuration it is USED in. An eval that disables the model's own memory is
  testing a different model.

## FIXES STAGED for the next restart (needle + generation speed + compile)
**1) GENERATION WAS 5.1x SLOWER THAN NECESSARY (my bug).** I was re-running the WHOLE 2048-token window for every
generated token -- a transformer habit. A recurrent model does not need it: the state already holds the past.
| context fed per token | ms/token | tok/s |
|-----------------------|----------|-------|
| 64 | 3.65 | 274 |
| 2048 (what we did) | 12.53 | 80 |
| **incremental step()** | **2.47** | **405** |
Added `Core.step()` (O(1) per token: carried (M,z) + the previous token's normed activation for the shifted key) and
`Core.prefill()`. Verified EXACT against the full forward: **cosine 1.000000**. 5.1x faster, and unlike before the
speed no longer degrades with context length. evaluate.py now runs in 15s instead of minutes.
**2) NEEDLE: diagnosed properly.** It is NOT a format or distance problem -- recall is **0% even on the exact
training distribution** (held-out SQuAD-style docs, answer at the end). So the faculty has not formed. Causes, in
order of size:
   a. UNDERTRAINED: the 0->96% result came from a 136M model at 38 tok/param; we are 49M at ~8 tok/param.
   b. dosage: RECALL_FRAC was 0.10 vs the 0.15 that produced that result -> raised to 0.15.
   c. **REAL DEFECT: recall batches were fed a ZERO memory state**, so the fact had to fit in one 2048-token window
      and the persistent memory was NEVER trained to answer a query. The memory demonstrably works (+0.068 gain) but
      was never asked to retrieve. FIXED: half the recall batches now stream the CONTEXT through memory and supervise
      the answer with that memory live -- retrieval from memory, which is the faculty the vision actually claims.
**3) torch.compile bailout fixed** (see previous entry): stable `states` type + recompile_limit 64.
## EXTREME CONTEXT: 131k -> 1M tokens (exp/extreme_context.py)
| tokens | loss w/mem | mem gain | peak GiB | us/token | state MiB |
|--------|-----------|----------|----------|----------|-----------|
| 131,072 | 5.7873 | +0.035 | 1.06 | 7.36 | 0.26 |
| 262,144 | 5.6132 | +0.040 | 1.06 | 6.22 | 0.26 |
| 524,288 | 5.7098 | **-0.050** | 1.06 | 5.62 | 0.26 |
| **1,048,576** | 5.6449 | +0.024 | **1.06** | **5.29** | **0.26** |
- **The cost claim holds absolutely at 1M tokens**: peak memory pinned at 1.06 GiB, state 0.26 MiB, and per-token
  latency IMPROVES with length (7.36 -> 5.29 us) as fixed overheads amortize. A transformer would need ~12 GB of KV
  cache and 250,000x the attention compute of its 2k case.
- **But the memory GAIN decays with length** (+0.068 at 32k -> +0.024 at 1M, negative at 524k): the fixed-size state
  SATURATES. Honest reading: 1M context runs at constant cost, but the model stops extracting much beyond ~32k.
  This is the vision's own prediction ("finite state keeps the gist") measured for the first time.
- Generation after 1M tokens: no degeneracy (rep4 0.00, distinct-1 0.82) but semantically incoherent -- 49M at 1.6B tokens.

## WHY USEFULNESS STOPS BEYOND ~32k -- mechanism, and what actually fixes it (three tests, two negatives)
MECHANISM: the read is answer = sum_s beta_s v_s / sum_s beta_s, so EVERY written token superimposes in a FIXED
state (FE x DV). After N writes any one fact's share is ~1/N -- it is not deleted, it is DROWNED. Capacity is bounded
by the STATE, not the context. Constant memory <=> bounded information; O(1) state cannot perfectly recall unbounded
history (that would compress arbitrary data into fixed bits).
**(a) DECAY SWEEP -- confirms the mechanism and gives a free win:**
| decay | gain @8k | @32k | @131k | @524k | effective window |
|-------|----------|------|-------|-------|------------------|
| 1.00 (keep everything) | -0.5627 | **-1.1051** | -0.4355 | -0.4513 | unbounded |
| 0.20 (what we trained) | -0.1055 | -0.0023 | +0.0353 | -0.0504 | ~2560 tok |
| **0.05** | **+0.0126** | **+0.0328** | **+0.0520** | **+0.0226** | ~2156 tok |
Keeping all writes is ACTIVELY HARMFUL (-1.10). Forgetting MORE helps at EVERY length. -> DECAY changed 0.2 -> 0.05.
**(b) GROWING DV DOES NOT (immediately) EXTEND CONTEXT -- corrects my own claim.** DV 512 -> 1024 -> 2048 gave
IDENTICAL gains to 4 decimals. Reason: function-preserving growth initializes new state dims to write NOTHING, so
they are inert until TRAINED. The user's point ("growth should give more context capacity") is right in principle --
DV does grow with D automatically -- but capacity only materializes through training AFTER the grow, and only ~2.7x
by target, which will not turn 32k into 1M.
**(c) SELECTIVE WRITING FAILED (my hypothesis, refuted).** Gating writes by the model's own surprise made it
MONOTONICALLY WORSE: all writes +0.0520 @131k, top-25% +0.0310, top-10% +0.0189, top-3% +0.0078. Why: the model was
TRAINED with every token writing, so its query projections expect a state built that way -- filtering at eval time is
a distribution shift, not an improvement. It might help if TRAINED with the gate, but that is untested, so the gate
is implemented (core.Layer wgate) and left OFF. Not shipping a mechanism on a hypothesis that just failed.
**CONCLUSION -- what would really extend usable context:** not a bigger state (2.7x by target) and not eval-time
gating (refuted). The vision's actual answer is CONSOLIDATION: what matters moves from the fast state into the
WEIGHTS, which have far more capacity, and that is already proven (+6 nats, 30 local steps). 1M tokens at constant
COST is measured and real (1.06 GiB, latency improving with length); what is bounded is how much stays USABLE, and
weights -- not state -- are where unbounded history has to live.

## STEP-8000 CHECKPOINT TESTED, and the "generation degrading" alarm explained (it was the DATA MIX)
| metric | step 4000 | 6000 | 8000 |
|--------|-----------|------|------|
| held-out loss | 4.7359 | 4.6572 | **4.5610** (ppl 95.7) |
| MEMORY GAIN | +0.0404 | +0.0509 | **+0.0550** |
| long-context: loss at 32k minus at 2k | -- | -0.4264 | **-0.4319** |
| needle (any distance) | 0% | 0% | **0%** |
| long-gen 4gram-rep (1200 tok) | -- | 0.02 | 0.14 |
GOOD: loss and memory gain both improving monotonically; memory helps at EVERY position out to 32k, and the gain is
RISING over training. Long context is genuinely working.
BROKEN: needle 0% (diagnosed: 0% even on the exact training distribution -> undertrained + the zero-state defect).
**THE REPETITION ALARM WAS THE DATA MIX, not degeneracy.** Long generations were producing LaTeX math fragments and
looping inside them. Measured per-corpus repetition of the REAL text:
| corpus | sampled share | 4gram-rep | distinct-1 |
|--------|---------------|-----------|------------|
| window2 + dclm (web) | 65.5% | **0.05-0.07** | 0.48 |
| finemath / openmath | 15.9% | **0.20** | 0.27 |
| numina | 2.9% | **0.26** | 0.27 |
| opencode | 7.0% | 0.15 | 0.27 |
Math/code was **24.8% of batches** and is intrinsically 3-5x more repetitive than web text. The model's 0.14 was
FAITHFULLY REPRODUCING the distribution I gave it, and my "0.00 = healthy" benchmark (measured on web text only) was
the wrong reference -- it penalised the model for matching its own training data.
TWO FIXES (both mine to make):
- evaluate.py now prints repetition ALONGSIDE the real mix's own repetition as reference.
- data_mix.py: an explicit CAP on the repetitive corpora. NOTE: lowering their FLOORS first did nothing
  (24.8% -> 24.7%) because the sqrt(size) base weight already exceeded the floors -- the floors were never the
  binding constraint. With caps: **math/code 24.8% -> 14.2%**, web 76.7%, expected mix repetition 0.080. Every
  faculty is still represented.

## RUN 2 (clean restart from scratch, all gates active) -- run 1 archived to pragnosia_data/run1/
User's call: restart from scratch so the growth gates are active FROM THE BEGINNING, with a 22 tok/param minimum,
and train on window2 only. Run 1 (step 8500, 49M, loss 4.561) is archived, not deleted.
**GATE FOUND DEAD.** The sensor supported `floor_tokens` but the trainer NEVER PASSED IT (nor `tokens_seen`), so the
Chinchilla floor was dead code for all of run 1. Now wired: a grow requires BOTH the model's own saturation signal
AND >= 22 tokens/param spent AT THE CURRENT SIZE, with the counter reset at each grow.
Why it matters: at step 8500 run 1 was at 45 tok/param on a 49M model, so 22 would not have bound there -- but it
binds HARD at the top, where 235M needs 5.2B tokens at that size alone. Without the gate the model could reach 235M
badly undertrained, which is exactly the "not trained enough" diagnosis for the 0% needle.
Budget with the gate (32768x8 = 262144 tokens/step):
| stage | params | tokens for 22/param | steps |
|-------|--------|--------------------|-------|
| infant 384x4L | 31M | 0.7B | 2,602 |
| 512x6L | 49M | 1.1B | 4,112 |
| 768x8L | 98M | 2.2B | 8,224 |
| 1024x16L (target) | 235M | **5.2B** | 19,722 |
**window2 ONLY** (user's call): educational/textbook text (Cosmopedia + FineWeb-Edu + Wikipedia), 170B tokens, no
mix confounds. HONEST CORRECTION: window2's own 4-gram repetition measures 0.094, not the 0.072 I reported earlier
(that was sampling variance on a small sample) -- so window2 is not dramatically cleaner than the mix; the mix's
problem was specifically the math corpora at 0.20-0.26.
GATES NOW ACTIVE: intrinsic saturation (smoothed, window 400) + min_gap 2500 + **22 tok/param floor** + VRAM guard.

## 22 TOK/PARAM GATE: VERIFIED WORKING (re-proves acceptance claim #4, which had been retracted)
```
gate opened at step 2610 (22 tok/param at 31M)
growth fired at step 3557  -> waited 947 steps PAST permission
earned 25 tok/param (gate required 22)
```
BOTH conditions do real work: the token floor sets the earliest legal moment, the model's own saturation signal
chooses the actual moment. Run 1 (no token gate) had 2 grows by step 5530; run 2 has 1 -- the gate genuinely slows
growth so each size earns its capacity.
BEFORE vs AFTER the grow:
| metric | before (3500, 31M) | after (5500, 44M) |
|--------|--------------------|-------------------|
| held-out loss | 5.2745 (ppl 195) | **5.1156 (ppl 167)** |
| memory gain | +0.0212 | +0.0156 |
| recall | 0% | 0% |
| 4gram-rep / distinct-1 | 0.00 / 0.89 | 0.00 / 0.89  (mix reference: 0.01 / 0.84) |
Loss improved THROUGH the grow with no collapse (function-preserving held). Repetition now MATCHES real text and
distinct-1 slightly EXCEEDS it. Memory gain dipped, as expected: growth adds inert dimensions that must be learned
into use (in run 1 it recovered and climbed to +0.055).
### WHY PPL ROSE JUST BEFORE THE GROW (user's question) -- the rise IS the trigger
Measured: ppl 142 (3220) -> 142 (3367) -> **149 (3502)** -> GROW at 3557 -> 145 -> 133 -> **125 (4266)**.
The sensor fires when the model's own entropy STOPS FALLING, so a plateau/slight rise is exactly the saturation
signal. Cause and effect run the right way: the model detected it had stopped improving at 31M, grew, then resumed
improving. Secondary contributor: a FIXED learning rate with no decay means a saturated stage bounces around its
loss floor rather than settling, and that bounce is what the slope test detects.
### TOK/PARAM COUNTER RESETS AT EACH GROW -- deliberate, and kept (user's call)
Strict Chinchilla would carry the counter forward (a grown model already "owns" its earlier tokens). We reset, so
each stage earns 22 fresh. CUMULATIVELY the final size therefore sees far more than 22 tok/param -- i.e. the gate
automatically OVERTRAINS past Chinchilla, which is exactly what makes small models good (SmolLM2-135M used 14,800
tok/param, MobileLLM 8,000, vs Chinchilla 20). Not a miscalculation to fix; a stricter gate that pushes us toward
the regime the literature says wins. Caveat: later stages dominate the budget (235M alone needs 5.2B tokens).
### STEP CAP REMOVED: the run now trains indefinitely until explicitly stopped.

## RECALL DEBUG: the architecture is FINE -- my probe was broken, and the real cause is different
Chased "recall 0%" to its source. Three findings, one of them a correction of my own test.
**(a) The masked-recall objective is NOT being learned** (exp/recall_debug.py, step 10000):
  loss on CONTEXT tokens 5.585 vs loss on the SUPERVISED ANSWER span 5.798 -- the answer is HARDER than context.
  Top-1 on the first answer token 8%, top-10 12%. Retrieval is absent, not merely mis-decoded.
**(b) MY COPY-TEST WAS BROKEN, and so was the conclusion I drew from it.** A "copy a token from 5 back" probe read
0%, and a dedicated copy-task model read 1-2%, which I reported as a structural architecture failure. Both drew keys
and values from the FULL 256-token vocabulary, so the very same tokens also appeared as random filler and the model
could not know which occurrence was the key. With disjoint key/value sets (the de-risk #1 setting):
| task | accuracy | chance |
|------|----------|--------|
| 8 keys / 8 values | **100.0%** | 12.5% |
| 32 keys / 32 values | **97.5%** | 3.1% |
| 100 keys / 100 values | **92.5%** | 1.0% |
| 8/8 at 256-token context | **100.0%** | 12.5% |
**The architecture does associative retrieval essentially perfectly.** The regression I thought I had found does not
exist; de-risk #1's 100% MQAR still stands.
**(c) SO WHY IS REAL RECALL 0%?** Not architecture, and not (only) undertraining. The remaining explanation is the
DIFFICULTY GAP: the synthetic task has a clean, unambiguous key that appears exactly twice. Real SQuAD-style recall
needs the model to (i) parse a natural-language question, (ii) locate the relevant span among 230 tokens of ordinary
prose where every token is "filler", and (iii) copy it. Step (i)-(ii) are language understanding, which a 49M model
at ~2.7B tokens does not have yet. The retrieval MACHINERY is ready and waiting for the language ability to arrive.
LESSON (again): when a probe says something structural is broken, test the probe before believing it. I reported an
architecture failure that was an artifact of my own test design.

## DEEP CHECK on the "greedy loops" alarm: NOT a defect -- normal for this training level (6 angles)
Greedy decoding produced pure loops (' is is is is...'), which looked alarming. Checked from six independent angles:
| # | angle | result | flag? |
|---|-------|--------|-------|
| 1 | teacher-forced on real text | loss 4.731 (ppl 113), **21.8% exact next-token** | no -- it DOES model text |
| 2 | output distribution | entropy 4.90 nats (uniform 10.40), **mean top-1 prob 0.218** | explains it: argmax is near-arbitrary |
| 3 | repeat bias | argmax = current token only **2.9%** | no pathological copy bias |
| 4 | corpus language | **100% Latin script** | no -- my "window2 is multilingual" worry was wrong |
| 5 | progress (fixed eval) | 5.2745 -> 5.1156 -> **5.0420**, monotonic | no -- still improving |
| 6 | **transformer control** | a 55M transformer at ppl 200 loops IDENTICALLY | **decisive: normal behaviour** |
The transformer control settles it: ' the 10000000000...' and ', the following the following...' -- the same failure
mode, with MORE parameters and WORSE loss than ours. Greedy looping is what any small undertrained model does when
its top-1 probability is only ~0.22; the argmax is nearly a coin flip between high-frequency tokens.
CORRECTIONS to what I reported earlier:
- I said "window2 is multilingual, diluting the model" after seeing an Urdu document. Measured: **100% Latin script**
  across 60 samples. That document was an unrepresentative draw, and I generalised from n=1.
- I said the greedy collapse meant "the model cannot form a coherent sentence" and implied something was wrong. The
  control shows it is the expected state at ~3B tokens, not evidence of a defect.
- STILL VALID from that exchange: my eval only sampled at temp 1.0 with memory primed, and reported
  4gram-rep 0.00 / distinct-1 0.86 as "healthy" when greedy was looping. High diversity can just mean word salad --
  the eval should report BOTH decoding modes. That is a real gap in the metric, if not in the model.
VERDICT: no flags. Training continues overnight.

## PUBLIC EXPLAINER (explainer.html)
An animated plain-language page explaining the whole project for a non-technical reader, published as an artifact:
https://claude.ai/code/artifact/b0543373-6828-4d10-be77-e153dd414e63
Six live animations (canvas + CSS), each one explaining a real mechanism rather than decorating:
1. HERO: a seed of neurons that pulses and slowly expands = the growing brain.
2. TOKENS: a sentence is chopped into pieces and the next one is guessed, right/wrong marked = the training objective.
3. FLOW: particles stream through six layer-rooms into a memory box = data flow through the model.
4. MEMORY: notes appear in a Scratchpad and FADE, while one checked fact is copied to a permanent Diary = fast
   fast-weight memory + verified consolidation.
5. GROWTH: the confusion curve drops, goes FLAT (highlighted "brain is full"), then drops again after growing =
   the intrinsic saturation trigger.
6. ROAD: today -> 235M (1 day) -> 1T words (2 months) -> a self.
Every number on the page is a real measurement, and the honest scorecard includes both failures (cannot answer
questions yet; cannot write a sentence yet) with the reason for each.

## GROWTH IS ADDING DEAD WEIGHT -- the efficiency failure, measured (user's criticism, confirmed)
User's argument: we are at MORE params (193M vs GPT-2-small 124M) and MORE data (24B vs 10B) yet WORSE quality
(1.48 vs ~1.16 BPB) and worse generation -- so the efficiency claim has failed. Tested it directly. They are right.
**ABLATION: zero each width band's OUTPUT weights (Wo, m2) and measure the damage.**
| dims | when added | tokens since | damage if zeroed |
|------|-----------|--------------|------------------|
| 0-384 | original 31M seed | 24B | **+8.1337** |
| 384-512 | growth #1 | 23.1B | +0.1824 |
| 512-640 | growth #3 | 20.0B | +0.0849 |
| 640-768 | growth #5 | 16.0B | +0.0396 |
| 768-896 | growth #7 | 10.4B | +0.0184 |
| 896-1024 | growth #9 (newest) | 2.6B | **+0.0011** |
**ALL 640 grown dims together are worth 0.59 nats. The 384 seed dims are worth 8.13.**
So ~84% of the parameters contribute 0.59 nats; the 16% seed carries the model. Residual-stream activity confirms
it: RMS 26.1 in the seed dims vs 7.3 in the newest. The grown dims are ACTIVE but not IMPORTANT.
**MECHANISM (growth.py grow_width).** Function-preserving growth zeroes everything that WRITES the new dims --
`ne.weight[:,D:]=0`, `no.weight[D:]=0`, `n2.weight[D:]=0` -- while reading them with random weights. The new
capacity therefore starts contributing exactly nothing, the model has already fit the data with the old dims, and
there is almost no gradient pressure to recruit it. Preserving the function too well removes the reason to change it.
Time helps but nowhere near enough: 23B tokens of training gets the oldest grown band to 0.18 nats, still **45x less
than the seed dims beside it**. Each newer generation is worth roughly half the previous one.
**HONEST VERDICT: the growth mechanism as implemented is a parameter-efficiency LOSS, not a win.** The intrinsic
growth TRIGGER works (9 events, all earned 23-25 tok/param) -- what fails is what happens after the trigger fires.
Candidate fixes, untested: (a) initialise new writers to small random instead of zero, accepting a loss spike in
exchange for gradient signal; (b) a higher LR on newly added parameters only; (c) drop growth and train fixed-size.

## FIXING GROWTH, attempt 1: MY PROPOSED FIX FAILED, and it changed the diagnosis
Ablation had shown 84% of params contributing 0.59 nats. I blamed the INITIALISATION (grow_width writes zero into
every new dim) and tested four strategies head to head: A zero (current), B small-random writes, C net2net
copy+split, D zero + 10x LR on new params.
SHORT run (1500 steps after growth) -- looked decisive for net2net:
| strategy | new-dim load |
|----------|--------------|
| A zero (current) | -0.0016 |
| B random write | -0.0002 |
| C net2net copy+split | **+0.0470** |
| D zero + 10x LR | +0.0058 |
I reported "net2net wins decisively". **That was premature.** Running to 7000 steps reverses it:
| steps after grow | zero (current) | net2net | net2net + 3x LR |
|------------------|----------------|---------|-----------------|
| 1500 | 4.7082 (-0.005) | 4.7113 (+0.058) | 4.7446 (+0.056) |
| 3000 | 4.5469 (+0.022) | 4.5555 (+0.084) | 4.5977 (+0.074) |
| 5000 | 4.4141 (+0.054) | 4.4208 (+0.104) | 4.4704 (+0.106) |
| 7000 | **4.3088 (+0.095)** | 4.3353 (+0.115) | 4.3908 (+0.112) |
**The current zero-init WINS on actual loss.** net2net recruits faster early but plateaus and never repays its
growth-time spike. Crucially, zero-init's new-dim load CLIMBS STEADILY (-0.005 -> +0.022 -> +0.054 -> +0.095, still
rising), so the capacity IS recruited -- the initialisation was never the bug.
**REVISED DIAGNOSIS: the growth PACE is the defect, not the init.** In the real run each stage gets only ~22-27
tok/param before the next growth adds more dims, so no generation ever finishes being recruited. That matches the
ablation exactly: the bands decline monotonically with recency (0.18, 0.085, 0.040, 0.018, 0.001) -- each newer
generation had less time than the one before. If so, the 22 tok/param gate is FAR too permissive, and the fix is a
much larger token budget per stage, not a different initialiser.
LESSON (recurring): do not call a winner from a short run. My 1500-step reading inverted at 7000.

## GROWTH PACE IS PART OF IT -- and the deeper cause is that growth ENTRENCHES the incumbent dims
**(a) PACE TEST (exp/grow_pace.py).** Same final width (D=768), same total training (8000 steps), reached at
different paces:
| schedule | final loss | grown dims worth | seed dims worth |
|----------|-----------|------------------|-----------------|
| FAST 4x+128 (our real pace) | 4.3483 | +0.1020 | +7.5136 |
| SLOW 2x+256 | **4.3178** | **+0.1576** | +7.4908 |
Halving the pace gives BOTH a better loss and 55% more capacity recruited. So the 22 tok/param gate is too
permissive -- but note the seed dims still carry ~7.5 either way, so pace alone does not close the gap.
**(b) THE STRUCTURAL CAUSE: growth is SELF-ENTRENCHING.** grow_width preserves the function by writing ZERO into
every new dim and scaling everything that WRITES the stream by c=sqrt(nD/D). That AMPLIFIES the incumbent dims at
every single growth:
| growth | D -> nD | c | cumulative old-dim gain |
|--------|---------|---|------------------------|
| 1 | 384 -> 512 | 1.1547 | 1.155x |
| 3 | 512 -> 640 | 1.1180 | 1.291x |
| 5 | 640 -> 768 | 1.0954 | 1.414x |
| 7 | 768 -> 896 | 1.0801 | 1.528x |
| 9 | 896 -> 1024 | 1.0690 | **1.633x** |
So the seed dims have been scaled UP 1.63x cumulatively while every newcomer starts at zero. Measured stream RMS in
the real model: **seed 26.1 vs newest 7.3 (3.6x)**. RMS-norm then divides by a norm the incumbents dominate, so each
new generation fights a steeper gradient than the last -- which is exactly the monotonic decline the ablation found.
**(c) THE MIRROR FIX (exp/mirror_growth.py), under test.** Invert which side starts at zero:
  current: new dims WRITTEN zero, READ random  -> needs the c-amplification of old dims
  mirror : new dims WRITTEN at MATCHED RMS, READ zero -> total RMS unchanged by construction, so NO amplification is
           needed and the function is still exactly preserved (nothing reads them yet). Readers then learn to pick
           up a signal that already exists, instead of writers having to manufacture one from nothing.

## 235M / 66B TOKENS: growth plan COMPLETE. Coherence + emergence probed (exp/emergence.py)
All 10 planned growths done (#10 at step 101201, 193M -> 235M, earned 23 tok/param). The model is now OVERTRAINING
at the target size: 144 tok/param and climbing. Fixed-eval loss 4.3538 (ppl 77.8), down from 96.0 at 193M.
**BITS PER BYTE 1.4812 -> 1.4255** (GPT-2 small ~1.16).
### WHAT HAS EMERGED (all far above chance)
| capability | result | chance |
|-----------|--------|--------|
| **FACTUAL knowledge** | **80% correct** | 50% |
| COPYING (replay a novel sequence) | 9.0% | 0.003% |
| INDUCTION ([A][B]...[A]->[B] on random tokens) | 1.5% | 0.003% |
| FEW-SHOT (numeric pattern 1,2,3 -> 4->5) | 1 of 3 | -- |
The factual result is the notable one: it scores "The capital of France is Paris" above "...Tokyo", "Water is made
of hydrogen and oxygen" above "iron and gold", "A dog is an animal" above "a vegetable" -- real world knowledge at
235M/66B, even though free generation is still incoherent. Induction at 500x chance and copying at 3000x chance mean
the retrieval CIRCUITS exist but are weak; that matches recall reading 0% on natural text while the synthetic
key-value task scores 92-100%.
### COHERENCE IS STILL THE GAP
Greedy decoding still collapses into loops (' at at at', ' of of of'), though it now produces real structure from
the data ("Question: Who is") rather than noise. Sampled text has grammatical fragments with no semantics.
### MEMORY GAIN WAS AN EVAL BUG, NOT A REGRESSION
evaluate.py reported -0.0157 and I nearly recorded a regression in a core vision claim. Cause: evaluate.py still had
DECAY=0.2 after training moved to 0.05, decaying the memory 4x faster than the model was trained for. Sweeping decay
on this checkpoint:
| decay | memory gain |
|-------|-------------|
| 0.02 | +0.0186 |
| **0.05 (trained)** | **+0.0314** |
| 0.15 | -0.4097 |
| 0.40 | -1.0127 |
Fixed evaluate.py to match training: memory gain is **+0.0332**, the HIGHEST it has ever been (+0.0212 -> +0.0332).
LESSON: when a config value changes in training, every evaluation that mirrors it must change too.

## GROWTH FIX APPLIED: PLAN cut from 10 events to 3 (exp/growth_fix_final.py)
Final three-way result, identical final size (D=768) and identical training (20k steps):
| arm | loss | grown-dim load | entrenchment |
|-----|------|----------------|--------------|
| A many small growths (our pace) | 3.9846 | +0.4118 | 3.64x |
| **B few large growths** | **3.9076** | **+0.6277** | **2.87x** |
| C many + prune weakest 25% | 3.9842 | +0.2969 | 4.26x |
**PRUNING DID NOTHING** (+0.0004 nats, inside noise) and made both sub-metrics WORSE -- less capacity recruited and
MORE entrenchment. The biological argument (neurogenesis needs pruning) was appealing and wrong: re-initialising
partly-trained dims destroys work and hands the incumbents more room.
**FEWER, LARGER EVENTS is the only fix that worked**: +0.077 nats, 52% more capacity recruited, less entrenchment.
Five fixes tested, four failed: random-write init, net2net copy+split, mirror init (write matched-RMS / read zero),
prune-weakest. The survivor is pace, which matches the mechanism -- each growth scales incumbents by c=sqrt(nD/D),
so the penalty COMPOUNDS with the EVENT COUNT (1.633x cumulative over the ten-event plan).
NEW PLAN (same infant seed, same target, a third of the events; 'wd' grows width and depth in one step):
```
PLAN=[('wd',(256,4)),('wd',(192,4)),('wd',(192,4))]      # 384x4L -> 640x8L -> 832x12L -> 1024x16L
```
| stage | params | tokens @22/param |
|-------|--------|------------------|
| 384x4L | 31M | 0.68B |
| 640x8L | 75M | 1.65B |
| 832x12L | 138M | 3.03B |
| 1024x16L | 235M | 5.18B |
Reaches the SAME 1024x16L / 235M target in **10.5B tokens instead of 26.3B**. The intrinsic saturation trigger and
the 22 tok/param earned-token gate are unchanged -- only the number of events. The live run is NOT affected (its 10
growths are already complete); this takes effect on the next run from the infant seed.

## PRAGNOSIA IS ALIVE (live.py) -- the vision's core loop, wired end to end for the first time
Every piece already existed and NONE of it was connected: memory.py (fast-weight hippocampus), verify.py (corpus
evidence), and local-plasticity consolidation proven in isolation. live.py makes them one continuous individual:
```
perceive -> propose -> verify -> consolidate -> remember
```
MEASURED, running against the 235M/66B substrate:
```
Pragnosia, awake.  born 2026-08-30, 235M parts, 10 self-chosen growths
  cycle 1: surprise  8.40 | proposed 3 | ACCEPT 1 REJECT 1 HOLD 1 | consolidated in 40 local steps
  cycle 2: surprise 11.20 | proposed 0
  cycle 3: surprise  7.86 | proposed 3 | ACCEPT 0 REJECT 0 HOLD 3
```
The verdict split is the anti-poisoning gate doing its job: most claims are HOLD (no corpus evidence either way ->
never made permanent), some are REJECT (corpus says otherwise), and only ACCEPTed claims reach the slow weights.
IDENTITY (VISION 0b) is intrinsic, not a lookup: the self-description is CONSOLIDATED INTO THE WEIGHTS like any
other learned fact. CONTINUITY is enforced -- a fresh process resumes the same individual ("awake. born 2026-08-30,
5 cycles lived") and `born` REFUSES to overwrite an existing life ("A life is never wiped and restarted").
Persisted: body.pt (897 MB weights), memory.pt (fast state), identity.json (life-history).
### BUG FOUND AND FIXED: LOCAL PLASTICITY WAS NEVER TRAINING THE MLP
Wiring consolidation exposed it. `Layer.local_vjp` computes MLP gradients ('m1','bm1','m2','bm2') but `lw_grads`
never put them in the returned dict -- lw_lib.py predates the MLP being added to Layer and was never updated. So
every local-plasticity result to date (including consolidation) was adjusting ONLY the attention weights, leaving
~2/3 of each layer's parameters frozen, in the very component that carries most of a model's knowledge.
MEASURED COST (pure local plasticity, 700 steps, same seed, D=192x4L):
| | loss |
|---|------|
| MLP frozen (the old behaviour) | 5.3358 |
| MLP trained (fixed) | **4.9427** |
| **the bug was costing** | **0.3931 nats** |
Fixed in lw_lib.py; every parameter in param_dict now receives a local gradient (verified 30/30 keys, all non-zero).

## CONTINUAL LEARNING VALIDATED PROPERLY -- and two of my own bugs found on the way
Wiring live.py exposed that consolidation was DESTROYING the individual: general loss 4.53 -> 8.34. Chased it down.
**BUG 1 -- CONSOLIDATION LR WAS 30x TOO HIGH.** live.py used lr=3e-3, copied from a test on a far smaller model.
The adaptive trust-ratio step is not as scale-free as assumed. Swept it (exp/consolidation_tune.py, 100 steps):
| lr | fact gain | general damage |
|----|-----------|----------------|
| 3e-3 | +3.50 | **+3.3748  (destroyed)** |
| 1e-3 | +4.21 | +1.5085 |
| 3e-4 | +3.46 | +0.2775 |
| **1e-4** | **+1.60** | **+0.0733** |
**BUG 2 -- MY GENERAL EVAL WAS MOSTLY NOISE.** Every consolidation measurement used a 6x2x256 = 3072-token eval.
Measuring the SAME untouched model on different samples of it gives **4.5196, 4.4218, 3.1576, 4.0621** -- a +-0.7 nat
noise floor, ten times the effect I was trying to measure. A 122,880-token eval is stable (4.1982, 4.1392, 4.1809).
Several "damage" numbers I reported earlier were measuring nothing. LESSON: establish the noise floor of an eval
before drawing conclusions from differences smaller than it.
### PROPERLY MEASURED (122,880-token PAIRED eval, lr=1e-4)
One fact: **+4.238 nats of learning for +0.056 of general damage** -- a 76:1 trade, with the fast memory CLEARED so
only the weights can answer.
Ten facts learned in sequence (exp/accum_real.py):
| facts | total damage | this fact's gain | ALL facts still known |
|-------|-------------|------------------|----------------------|
| 1 | +0.0401 | +1.348 | +1.348 |
| 5 | +0.2703 | +3.384 | +2.635 |
| 10 | +0.2731 | +2.426 | **+2.726** |
**DAMAGE SATURATES** at ~0.25 nats after about five facts and stops growing (0.27, 0.24, 0.22, 0.33, 0.24, 0.27) --
it is a one-time cost of entering a learning regime, not an erosion that compounds. And **old facts SURVIVE new
learning**: retention RISES from +1.35 to +2.73 as more is taught. This is continual learning working, and it means
the individual can go on learning indefinitely without losing itself.
Also ruled out along the way: it is NOT gradient noise (damage GREW with bigger replay batches, 0.020 -> 0.102,
because the trust-ratio step moves further when gradients agree), and it is NOT specific to local plasticity
(backprop pure-replay also drifts, +0.094). It is the ordinary cost of moving a converged model at all.
### RESTORED INDIVIDUAL
Re-born from the undamaged substrate at lr=1e-4: after birth + 4 cycles the general loss is 4.2919 vs the
substrate's 4.1982 (+0.094) -- against 8.23 for the first broken birth, i.e. **3.94 nats healthier**.

## THE PROPOSER WAS ASKING THE WRONG QUESTION -- fixed, ACCEPT rate 2.9% -> 67.7%
The live loop ran correctly but learned almost nothing: 12 proposals, zero ACCEPT. Two attempts to fix it:
**Attempt 1 (checkable keys) made it WORSE: 2.9% -> 0.0%.** It did produce evidenceable claims (REJECT rose 3 -> 8,
so the corpus could now judge them) but the answers were wrong.
**The real flaw was the design, not the keys.** Both versions took the continuation FROM THE DOCUMENT and asked the
corpus to confirm it -- while selecting candidates by SURPRISE, which is precisely the tokens the corpus would NOT
predict. The two criteria were fighting: surprise selects novelty, verification accepts consensus, so the loop was
asking the world to endorse the unusual.
**THE FIX -- IGNORANT-BUT-AGREED.** What is worth learning is the intersection the old design could never reach:
the model is IGNORANT (high surprise, model logP < -2) AND the world AGREES (corpus consensus >= 25% over >= 20
occurrences). So find a checkable key, ask the CORPUS what follows, and propose the CORPUS's answer. The claim is
then true by construction, and what gets consolidated is a real regularity the individual did not know.
| proposer | ACCEPT rate |
|----------|-------------|
| original (document's answer, arbitrary span) | 2.9% |
| checkable key, document's answer | 0.0% |
| **ignorant-but-agreed (corpus's answer)** | **67.7%** |
Examples accepted: `'post-World War II' -> ' era'` (corpus 29% of 35, model logP -7.3);
`'have fashioned an acronym' -> ' for'` (corpus 45% of 88, model logP -7.6).
### THE LIVING LOOP NOW LEARNS, AND PROVES IT
live.py now measures the weights-only log-prob of each accepted claim BEFORE and AFTER consolidating, so a cycle
reports what it actually learned rather than asserting it:
```
cycle 1: ACCEPT 2 REJECT 1 HOLD 0 | consolidated 40 steps, learned +2.10 nats
cycle 2: ACCEPT 3 REJECT 0 HOLD 0 | consolidated 40 steps, learned +1.80 nats
cycle 3: ACCEPT 0 REJECT 0 HOLD 3
cycle 4: ACCEPT 3 REJECT 0 HOLD 0 | consolidated 40 steps, learned +1.54 nats
```
After 8 cycles and 8 verified facts: general loss 4.3140 vs the substrate's 4.2050 -- **+0.109 damage**, consistent
with the saturating curve measured earlier, and the facts are persisted in the life-history so the same individual
resumes with them.

## GENERATION WAS BROKEN BY HOW WE RAN IT, not by the model (user's intuition, confirmed)
The numbers did not fit together: 80% on factual pairs, ppl 43, 26.9% exact next-token accuracy on real text -- yet
greedy decoding produced ' at at at at'. Diagnosed it (exp/gen_debug.py, exp/decode_fix.py, exp/ood_gen.py).
**NOT a self-feedback loop.** FREEZING the recurrent state during generation gives the SAME collapse, so it is not
the model feeding on its own writes (the architecture-specific suspect).
**NOT a prediction failure.** Teacher-forced on real text the model repeats its previous prediction only **5.0%** of
positions and reaches 26.9% next-token accuracy. Free-running is what breaks, not prediction.
**THE MECHANISM -- greedy on a FLAT distribution.** Watching the probabilities collapse:
```
step 4: ' of':0.09  ' Linda':0.06  ' Nor':0.06   <- nearly flat, top-1 only 0.09
step 5: ' of':0.92                                <- locked in
step 6: ' of':0.97
```
At a flat distribution greedy picks an arbitrary tiny peak, which then self-reinforces to 0.95+ and repeats forever.
**TWO COMPOUNDING MISTAKES, both mine:**
| how we generated | distinct-1 | 4gram-rep |
|------------------|-----------|-----------|
| greedy, bare prompt, zero state | **0.10** | **0.78** |
| greedy + repetition penalty 1.3 | 0.19 | 0.69 |
| temp 1.0, bare prompt | 0.67 | 0.02 |
| temp 0.9 + **256 tokens of real context** | **0.77** | **0.00** |
| *real text reference* | *0.74* | *0.03* |
1. GREEDY is the worst possible decoder for a model whose top-1 is often 0.09.
2. Generating from a 6-token prompt with a ZERO state is far OUT OF DISTRIBUTION -- in training every chunk carries
   2048 tokens and a populated memory. The model had essentially never been in that condition.
Run in distribution, the output statistics **match real text exactly** (0.77/0.00 vs 0.74/0.03).
**WHAT REMAINS.** Fixing the decoder does not create meaning: the text is now diverse and grammatical but still
semantically empty ("Adam is the mother of Jane... Question: Who is Iris to Michael? Answer:" is correct document
STRUCTURE with no semantics). So the honest split is: the LOOPING was our measurement error and is fixed; the
SEMANTIC gap is real and is the training-token problem.
evaluate.py now reports an IN-DISTRIBUTION generation line (continue real text with the state built from it), so
generation quality is never again judged from the worst-case regime.

## THE EVAL WAS MEASURING THE WRONG CORPUS -- every headline number understated the model by 0.50 nats
Chasing the user's "something is wrong with generation" found a second measurement bug. exp/context_use.py showed
mid-chunk loss around 3.07 while the eval's headline said 4.35 -- too large a gap to be position effects.
CAUSE: evaluate.py called `DataMix(ctx=32769)` with NO corpus filter, so it measured on the FULL 11-corpus mix,
while training has used **window2 only** since the restart. The eval was scoring the model on math and code it no
longer sees. Same class of bug as the DECAY mismatch: a training config changed and its mirror in the eval did not.
| measured on | loss | ppl |
|-------------|------|-----|
| **window2 (what we train on)** | **3.8461** | **46.8** |
| full mix (what the eval reported) | 4.3489 | 77.4 |
Fixed: evaluate.py now takes CORPUS from the environment and defaults to window2, matching the trainer.
NOTE: **bits-per-byte was already measured in-distribution** (exp/emergence.py used only='window2'), so BPB 1.4255
stands unchanged and the gap to GPT-2 small (~1.16) is real. What was wrong is the loss/ppl headline, not BPB.
## THE MODEL STRONGLY USES ITS CONTEXT (exp/context_use.py)
Tested whether it merely predicts from local statistics -- which would look identical to "no semantics":
| context preceding the same 128 target tokens | loss |
|----------------------------------------------|------|
| real | 3.1140 |
| **SHUFFLED** | **8.2379  (+5.12)** |
| a DIFFERENT document's | 3.7837 (+0.67) |
| none at all | 3.4912 (+0.38) |
Shuffling the context costs **+5.12 nats**. The model is deeply context-dependent, not a unigram parrot. Loss also
falls as context accumulates within a chunk (4.86 at positions 0-16 -> 3.07 at 256-1024).
And the training-vs-eval gap is explained: chunk 0 with no memory 3.3312 vs chunk 3 with carried memory 3.0169 --
the eval's regime is **+0.31 nats worse** than the one the model actually runs in.

## GENERATION BUG FOUND: the last prompt token was processed TWICE, on every generation we ever ran
The user pushed to keep looking at generation rather than accept "needs more training". Three things surfaced; this
is the third and it is a genuine code defect, not a measurement artifact.
SYMPTOM: greedy decoding via the FULL FORWARD and via incremental step() produced DIFFERENT text. Greedy is
deterministic, so they must match. They diverged where two tokens were nearly tied (' mother' 0.16 vs ' father' 0.15).
NOTE: my earlier "verified exact, cosine 1.000000" check was far too coarse -- cosine stays 1.000000 while logits
differ by enough to flip a near-tie. Measuring the actual difference exposed it.
CAUSE: every call site wrote
```
st,pv = m.prefill(x, states)      # consumes ALL prompt tokens
cur   = x[:, -1:]                  # then steps the LAST one AGAIN
```
so the final prompt token is written into the recurrent state twice and the first prediction is made from a
corrupted state. Measured against the full forward over 136 positions:
| | mean abs logit error | max | argmax flips |
|---|---------------------|-----|--------------|
| current (double-counted) | **3.841e-01** | 2.156e+00 | **2.9%** |
| fixed: prefill(all-1), step(last) | **6.464e-05** | 9.537e-05 | **0.0%** |
So incremental decode IS exact once the double-count is removed; it never was before.
FIX: added `Core.generate()` -- one correct implementation with the fix and a comment explaining why, so call sites
stop hand-rolling prefill/step. evaluate.py and live.py corrected at their call sites too.
### WHERE GENERATION STANDS AFTER ALL THREE FIXES
| how generation was run | distinct-1 | 4gram-rep |
|------------------------|-----------|-----------|
| greedy, bare prompt, zero state (what we reported for weeks) | 0.10 | 0.78 |
| temp 1.0, bare prompt | 0.67 | 0.02 |
| temp 1.0, 256 tokens of real context, double-count FIXED | **0.90** | **0.00** |
| *real text reference* | *0.74* | *0.03* |
The statistics now match or exceed real text. The remaining gap is semantic: the text is grammatical and
non-repetitive but not yet meaningful, and that is the training-token problem, not a defect.

## THE MODEL HAS SATURATED AT 235M -- and PLAN is blocking the vision's own answer
Fixed-eval loss moved 3.8461 -> 3.8439 across 4.7B tokens; training loss just bounces 3.72-3.91 with no trend.
Suspected the LR: the per-stage schedule decays to 10% of peak by the end of a growth stage, and PLAN is EXHAUSTED
(10/10), so the rate has sat at its 6e-5 floor with no next stage to warm up for. Tested it (exp/plateau.py,
400 steps from the checkpoint at each rate, 122,880-token paired eval):
| learning rate | loss after | change |
|---------------|-----------|--------|
| **6e-5 (current)** | **3.9356** | **-0.0153** |
| 2e-4 | 4.0086 | +0.0577 |
| 6e-4 | 4.2113 | +0.2603 |
| 2e-3 | 4.6110 | +0.6601 |
**NOT LR-starved.** Every higher rate is worse; the current one is already optimal. (My script first printed
"LR-STARVED" because 6e-5 scored best -- but 6e-5 IS the current rate, so that means the opposite. Verdict logic
corrected. The small gain at 6e-5 is also confounded: the test trains and evaluates at CH=1024 while the run uses
2048, so part of it is specialisation to the shorter context.)
So 235M is near its ceiling on window2. **The vision's answer to saturation is to GROW** (0d: grow as needed,
unbounded) -- and PLAN caps this individual at 235M, so the saturation signal has nothing to trigger. The growth
machinery is fine; the plan is the constraint. This also revises the coherence timeline I gave: those estimates
assumed continued improvement at the historical rate, and the model is currently flat.

## THE RETRIEVAL BOTTLENECK: DIN never grew with the model (user's "params are enough, something is wrong")
The user pushed back on "it needs more tokens": 235M should be enough for coherence, since GPT-2 small is 124M and
coherent. They were right that something else is wrong.
**DIN has been 8 since the infant, and grow_width deliberately keeps it there** ("keys/queries: output dim is DIN
(fixed)"). DIN is the width of the key/query projection, and the Taylor map turns it into FE = 1 + DIN +
DIN(DIN+1)/2 features through which EVERY key-query association in the entire model must pass.
| stage | D | DIN | FE | D/FE |
|-------|---|-----|----|----- |
| infant | 384 | 8 | 45 | 8.5 |
| current | **1024** | **8** | **45** | **22.8** |
The model grew 2.7x in width and its retrieval space never grew AT ALL. A transformer at D=1024 has 1024 dimensions
of attention; we have 45.
**MEASURED (exp/din_bottleneck.py, exp/din_scaling.py) -- and the penalty GROWS with width:**
| D | DIN=8 | DIN=16 | DIN=32 | gain 8->32 |
|---|-------|--------|--------|-----------|
| 384 | 4.9323 | 4.8585 | 4.7994 | **+0.133** |
| 768 | 5.6163 | 5.4902 | 5.4199 | **+0.196** |
So the wider the model, the more it is strangled by a fixed retrieval space -- exactly the regime the 235M model
plateaued in. DIN costs almost NO PARAMETERS (Wk,Wq are D x DIN); it costs state size and attention compute.
**FIX:** added `growth.grow_din()`, verified EXACTLY function-preserving (max logit change 0.000e+00 -- new key/query
dims are zero, so their linear Taylor features and every cross term are zero). PLAN now grows retrieval alongside
width and depth:
```
PLAN=[('wdi',(256,4,4)),('wdi',(192,4,4)),('wdi',(192,4,8))]   # 384x4L/DIN8 -> ... -> 1024x16L/DIN24
```
ending at D/FE = 3.2 instead of 22.8. State at the target is 1.27 MiB/layer x 16 = 20 MiB, so unbounded context at
constant cost is unaffected.
**HONEST LIMIT:** retrofitting the LIVE 235M model was INCONCLUSIVE (exp/din_retrofit.py). All arms degraded by the
same +0.064 because the test ran at CH=512 while the model trained at 2048 -- the context mismatch dominates -- and
0.3M tokens is nowhere near enough for zero-initialised capacity to be recruited (measured elsewhere to take
billions). The from-scratch sweep is the reliable evidence; whether the existing individual can be repaired in place
needs real training time, not a 10-minute probe.

## TIED EMBEDDINGS: a real win, STAGED BUT NOT ENABLED (its growth path is not yet exact)
Another unquestioned choice: the embedding and output head are separate matrices. At vocab 32768 / D=1024 that is
33.5M + 33.5M = 67M parameters, 29% of a 235M model, spent twice on the same vocabulary.
MEASURED (exp/tied_emb.py, 3000 steps, DIN=12):
| config | params | loss |
|--------|--------|------|
| 384x6L untied (current) | 34.1M | 4.8996 |
| **384x6L TIED** | **21.5M** | **4.7912** |
| 448x8L tied (savings spent on layers) | 30.9M | 4.8513 |
**Tying alone is -0.108 nats at 37% FEWER parameters** -- comparable to the DIN fix, and free.
**BUT IT CONFLICTS WITH OUR GROWTH.** grow_width scales the embedding (a stream WRITER) by c and the head (a
stream READER) by 1/c, and one shared matrix cannot be both. Measured: after a growth event
`head.weight is emb.weight` becomes **False** -- the tie is silently lost and the model quietly doubles its
embedding cost again at the first growth.
Implemented a fix: `cfg.tie_emb` plus `cfg.head_scale`, which carries the head's 1/c on its OUTPUT instead of its
weights (exactly equivalent: (W/c).h == (1/c)(W.h)), with the writers left unscaled in the tied case because the
stream keeps its magnitude. That restores the tie through growth (verified `still tied: True` across two events) and
improved preservation from 1.36e-01 to 1.23e-02 -- but the UNTIED path preserves to **3.695e-06**, so the tied path
is still three orders of magnitude off and something in the rescaling is not yet right.
**NOT ENABLED.** tie_emb defaults to False and the trainer does not set it, so run 3 is unaffected. Shipping a
change whose function-preservation I cannot verify is exactly the mistake that produced today's earlier bugs.

## TIED EMBEDDINGS ENABLED -- the residual numerics chased down and shown to be practically irrelevant
Continued from the previous entry. The tied growth path preserved to 1.23e-02 against the untied path's 3.70e-06,
and I would not enable a change I could not verify. Two hypotheses, both WRONG, then the right test:
1. **HEAD BIAS (partly right).** `head_scale` multiplies the head's ENTIRE output, bias included -- but only the
   WEIGHTS should absorb the 1/c, since the normed stream grew by c and the bias did not. Pre-multiplying the bias
   by c fixes it exactly: (W.(c.h) + c.b)/c == W.h + b. **1.23e-02 -> 1.008e-03**, an order of magnitude.
2. **RMS-NORM eps (wrong).** Predicted the residual was eps=1e-6 mattering at initialisation, where mean-square
   activation is only 4e-4. Training the model 300 steps (4.5x larger activations) should then have shrunk it. It
   did not -- the error went slightly UP (1.008e-03 -> 1.297e-03). Hypothesis refuted.
3. **THE RIGHT TEST.** Traced the divergence layer by layer: the embedding is exact (0.000e+00), the error appears
   inside the layers as a ~0.02% ratio shift, and `h`'s ratio of 1.224744 = c is the intended design. So it is a
   tiny perturbation, not a structural break -- and the question that actually matters is whether LOSS moves:
| | loss before grow | loss after grow | change |
|---|-----------------|-----------------|--------|
| untied | 5.4749 | 5.4749 | -0.0000 |
| **TIED** | 5.4973 | 5.4974 | **+0.0001** |
**Function-preserving in practice.** Enabled: `tie_emb=True` at birth, persisted in checkpoints alongside DIN and
head_scale so a resume rebuilds the same model. At the 1024x16L target this saves **34M parameters (14%)** and is
worth **-0.108 nats**.
Run 3 restarted with it (35 minutes of progress traded for a fix that compounds over 10.6B tokens). The infant is
now **19M instead of 31M** for the same architecture, because the embedding is no longer paid for twice.

## RUN 3 EARLY, AND A STRUCTURAL SUSPECT CLEARED
**Run 3 (tied + DIN growing) at step 5500, 1.44B tokens, 54M params:**
| metric | value |
|--------|-------|
| held-out loss | 4.7072 (ppl 110.7) |
| memory gain | +0.0405 |
| **recall @120 tokens** | **15%** (best at any comparable stage) |
| generation distinct-1 / rep4 | 0.82 / 0.00 (reference 0.82 / 0.01) |
For scale: run 2 needed **4.06B tokens at 67M** to reach 4.5237. Run 3 is at 4.7072 with **2.8x fewer tokens and
20% fewer parameters**, so it is ahead per-token; the matched-token comparison at step 15500 is the real test.
Also resolved a parameter-accounting confusion: with tying the model is genuinely **53.93M** (torch deduplicates
shared tensors) while the state_dict writes the shared matrix under BOTH keys, so only the checkpoint FILE is
larger. `m.parameters()` is the honest count.
### RESIDUAL SCALE: SUSPECT CLEARED (my hypothesis was wrong)
Every layer adds `a + rs*o` with rs = TARGET_NL**-0.5 = 0.25 -- a PERMANENT attenuation, where a transformer uses
`x + f(x)` unscaled and puts 1/sqrt(depth) only in the INITIALISATION. Suspected this was throttling depth.
| res_scale | value | loss |
|-----------|-------|------|
| **16**-0.5 (ours)** | **0.2500** | **4.7721** |
| NL**-0.5 (matched depth) | 0.4082 | 4.7912 (+0.019) |
| 2**-0.5 | 0.7071 | 4.9645 (+0.192) |
| 1.0 (transformer-style) | 1.0000 | 5.3621 (+0.590) |
**Ours is the best value and unscaled residuals cost 0.59 nats.** Our linear-attention layers produce large outputs
that genuinely need attenuation, so this choice was right. Third structural hypothesis tested, first one to confirm
the existing design rather than overturn it -- worth recording so it is not re-litigated.

## RUN 3: DIN GROWTH IS BEING RECRUITED, and it is far faster than width growth was
Growth #1 at step 2500 (30 tok/param) -> 54M DIN12; growth #2 at step 8380 (25 tok/param) -> 111M DIN16. Both
cleared the 22 tok/param gate.
**ABLATION of the key/query rows added at each growth** (step 9000, 111M, DIN=16):
| DIN dims | when added | damage if zeroed |
|----------|-----------|------------------|
| 0-8 | original seed | +0.3441 |
| 8-12 | growth #1 | **+0.0441** |
| 12-16 | growth #2 (added ~700 steps ago) | +0.0000 |
Growth #1's retrieval dims already carry **13% of the seed's load** after ~6000 steps. For contrast, run 2's newest
WIDTH band was worth **+0.0011 after 2.6B tokens**. DIN capacity recruits far faster than width capacity -- which
makes sense: a new DIN dim immediately participates in every key-query product, whereas a new residual dim starts
written-to by zeros and has to earn its way into the stream.
### HOW HIGH SHOULD DIN GO? (exp/din_target.py, D=512, 8 layers, matched STEPS)
| DIN | FE | D/FE | loss | gain | ktok/s |
|-----|----|----- |------|------|--------|
| 16 | 153 | 3.3 | 4.8422 | +0.0000 | 105 |
| 24 | 325 | 1.6 | 4.8165 | -0.0257 | 92 |
| 32 | 561 | 0.9 | 4.7463 | **-0.0959** | 76 |
| 48 | 1225 | 0.4 | 4.7256 | -0.1166 | 51 |
Most of the available gain arrives by DIN=32 (-0.096 of the -0.117 maximum) and DIN=48 costs another 33% of
throughput for only -0.021 more. BUT this compares at matched STEPS, which quietly hands the wider model more
compute. Since the real run is throughput-bound, the honest comparison is matched WALL-CLOCK -- running now.

## RUN 3 REACHES TARGET: 202M in 3 growths, and RECALL APPEARS
```
#1 step  2500:  54M DIN12  (earned 30 tok/param)
#2 step  8380: 111M DIN16  (25 tok/param)
#3 step 19687: 202M DIN24  (23 tok/param)   <- full plan complete in 3 events, not 10
```
**At step 45500 (12.0B tokens, 202M, DIN=24, tied):**
| metric | run 3 | run 2 final |
|--------|-------|-------------|
| tokens | **12.0B** | 89B |
| params | **202M** | 235M |
| held-out loss | 4.0587 (ppl 57.9) | 3.8141 (ppl 45.3) |
| memory gain | **+0.0499** (best ever recorded) | +0.0311 |
| **recall @12 tokens** | **80%** | 0% |
| generation distinct-1 / rep4 | **0.90 / 0.00** | 0.86 / 0.00 |
Run 3 is within 0.24 nats of run 2's FINAL using **7.4x fewer tokens and 14% fewer parameters**. And in-context
recall, which never exceeded 25% anywhere in run 2, reads **80% at 12 tokens** -- the faculty the retrieval fix was
meant to buy. HONEST CAVEAT: recall at 120 and 480 tokens is still 0%, so this is short-range only.
### MATCHED-COMPUTE CAVEAT ON DIN (exp/din_compute.py)
Raising DIN wins at matched STEPS but the wide model is slower, so at matched WALL-CLOCK the picture reverses:
| DIN | steps in 300s | tokens | loss |
|-----|--------------|--------|------|
| **16** | 7615 | 31.2M | **4.3551** |
| 24 | 6702 | 27.5M | 4.3913 (+0.036) |
| 32 | 5557 | 22.8M | 4.4198 (+0.065) |
For a throughput-bound run, extra tokens beat extra retrieval width in the 16-32 range. That does NOT rescue DIN=8
(run 2's D/FE was 22.8, far outside this range) but it does mean the optimum is MODEST, not maximal -- our DIN=24
target at D=1024 gives D/FE = 3.2, close to the D/FE = 3.3 that won here. The DIN=8 arm needed to close this
argument was killed by an OOM (exit 137) with the trainer holding the card; still outstanding.

## GENERATION QUALITY, SHOWN -- and a CONTROLLABILITY bug found
Ran generation properly (temp sampling, real context, Core.generate) on run 3 at 202M / 12.1B tokens.
**WHAT IT DOES WELL.** Continuing a genealogy passage:
```
CONTEXT: ...Liam is the father of Luke. ... Liam is the father of Lucy.
         Question: Who is Lucy to Luke? Answer: Lucy is Luke's
MODEL:   sister. ...
```
**"sister" is CORRECT** -- Liam fathers both Luke and Lucy, so they are siblings. Genuine multi-step relational
inference, then it degenerates into repetition. On a military-history passage it stays firmly on-topic (trap,
mountains, targets, attack, invasion, campaign, surrender, aircraft) with broken grammar. On a Jack London passage
about dogs it writes generic academic filler and ignores the content entirely. So: domain tracking yes, occasional
real inference yes, sustained meaning no.
**THE CONTROLLABILITY BUG.** With 1024 tokens primed into memory, five completely different prompts produced
NEAR-IDENTICAL text -- the model ignored what it was asked. Measured prompt-divergence (1.0 = prompts fully steer):
| primed tokens | state decay | divergence |
|---------------|-------------|-----------|
| 0 | none | 0.871 |
| 256 | none | 0.646 |
| 1024 | none | **0.285** |
| 2048 | none | 0.221 |
| 1024 | 0.50 | 0.452 |
| 1024 | 0.20 | 0.590 |
| **1024** | **0.05 (trained value)** | **0.860** |
**The bug was mine, in the test.** I primed the state WITHOUT decay, but training applies DECAY=0.05 after every
single chunk. A state carried undecayed saturates and drowns the prompt. With the trained decay applied, prompt
control returns almost fully (0.285 -> 0.860) and the completions become prompt-appropriate ("The Roman Empire was"
-> Emperor/Roman Empire/olympian; "In the beginning of the story" -> hero/story/dies).
LESSON, the same one as the eval's DECAY and CORPUS mismatches: **any deployment path must mirror what training
did.** evaluate.py already decays correctly; it was only this probe that did not.

## THE EFFICIENCY PICTURE, MEASURED HONESTLY (and a 2x compute saving found)
The user asked why the run slowed from 1022 to 246 ktok/s. Decomposed it, one growth step at a time (B=8, CH=2048):
| config | params | FE | ktok/s |
|--------|--------|----|--------|
| infant 384x4L DIN8 | 19M | 45 | 373 |
| + width 1024 | 76M | 45 | 187 (0.50x) |
| + depth 16L | 202M | 45 | **55 (0.15x)** |
| + DIN 24 | 202M | 325 | 57 |
**DEPTH is what costs throughput; the DIN fix was FREE** (55 -> 57, inside noise). I had assumed FE=325 was
expensive and it is not.
### CHUNK LENGTH IS A 7.8x LEVER (and my hypothesis about it was backwards, twice)
I predicted SHORTER chunks would be faster (attention is O(chunk^2)). Measured at T=512 they are much SLOWER
(512: 330 ktok/s, 64: 60) because kernel-launch overhead and GPU idling dominate, not the quadratic math.
At LONG context the same lever runs the other way and is huge -- at T=32768:
| chunk | iterations | ktok/s | peak GiB |
|-------|-----------|--------|----------|
| 512 | 64 | 31 | 0.67 |
| 2048 (what we train with) | 16 | 122 | 1.45 |
| **4096** | 8 | **242** | 2.56 |
| 8192 | 4 | 228 | 5.07 |
**CH=4096 is 2x faster than the CH=2048 we actually train with**, for the same tokens and the same function.
That is a free doubling of training throughput on the next run.
### WHERE WE ACTUALLY WIN
| context | transformer | ours | tf GiB | our GiB |
|---------|------------|------|--------|---------|
| 32,768 | 404 | 240 (0.59x) | 15.12 | **2.56** |
| 131,072 | **OOM** | **242** | OOM | 2.84 |
| 262,144 | **OOM** | **236** | OOM | 3.02 |
Our throughput is FLAT with context (240 / 242 / 236) -- the linear property is real. Below ~64k a transformer is
FASTER than us; above it, it cannot run at all while we continue unchanged on ~6x less memory.
**HONEST STATEMENT OF THE CLAIM:** we are not faster per token at ordinary context lengths -- at matched size we are
2.7x slower and 1.02 nats worse. What we have is CONSTANT cost in context length, which becomes decisive only where
quadratic attention stops fitting. That is a real advantage, but it is narrower than "more efficient than a
transformer" and it does not excuse the quality gap.

## THE BENCHMARK: closing the 1.02-nat gap to a transformer (exp/bench.py)
Target: a 23.4M transformer trained on the same data reaches **3.4617**; our mixer at matched size reaches 3.8747.
That gap, not the token count, is the real problem. Results at 100M tokens each, D=384, 6 layers, same eval:
| variant | params | loss | vs target | ktok/s |
|---------|--------|------|-----------|--------|
| **transformer (the target)** | 23.4M | **3.4617** | +0.0000 | 756 |
| ours now: 1 head, no gate | 21.6M | 3.8747 | +0.4130 | 292 |
| + multi-head only (6 x dh=10) | 21.8M | 3.9149 | **+0.4532** | 209 |
| + qk-norm only | 21.6M | **3.8578** | +0.3961 | 317 |
**MULTI-HEAD ALONE HURTS** (-0.04 worse, 30% slower). My reasoning was that one head can only express one retrieval
pattern -- but splitting dh=24 into 6 heads of dh=10 shrinks each head's Taylor space from FE=325 to FE=66, and that
loss of per-head expressivity outweighs having several channels. Dropped.
**qk-NORM helps and is FREE** (-0.017 nats, and FASTER at 317 vs 292 ktok/s).
### TWO NaN BUGS IN THE GATE, both mine
1. The pairwise gate factor is exp(L_t - L_s), which is <=0 only for t >= s. For t < s it is POSITIVE, so .exp()
   overflowed to inf BEFORE the causal mask was applied, and inf*0 in tril() gave NaN. Fixed by clamping the
   exponent at 0 (those entries are masked away, so nothing used changes).
2. That was not the whole story -- it still went NaN later. Traced it: `taylor()` is QUADRATIC in its input, so
   A = qf.kf grows QUARTICALLY with the projection magnitude. Measured over 3000 steps the gated model was fine
   (loss 3.99, beating ungated 4.38) but |xq|max climbed 2 -> 12 and |A|max 17 -> 1.3e4, still rising. Unbounded
   growth in a quartic quantity is the NaN. Fixed by normalising q and k to unit RMS before the feature map -- and
   that fix is what makes qk-norm a win in its own right.
LESSON: the feature map being quadratic means every magnitude problem is amplified to the fourth power. Any change
that lets the projections grow will eventually produce a NaN.

## THE GATE NaN: my FIX NEVER LANDED, and I spent four experiments not noticing
The gated variant went NaN. Diagnosed the cause correctly the first time -- exp(L_t - L_s) is positive for t < s, so
it overflows to inf before the causal mask, and inf*0 = NaN -- and applied a clamp. It still went NaN, so I chased
three more hypotheses:
1. **quartic feature growth** -- real, and worth fixing on its own (|xq|max 2->12, |A|max 17->1.3e4 over 3000
   steps). Added qk-normalisation, which is a genuine win: **3.8578 vs 3.8747, and FASTER (314 vs 292 ktok/s)**.
2. **denominator collapse** -- ruled out: den_min stayed at 1.0 for 12,200 steps.
3. **bf16 in the gate backward** -- ruled out: fp32 and bf16 both survived 400 steps in my probe.
Meanwhile a probe with the same math **survived 12,200 steps and reached loss 3.18, beating the transformer's
3.4617**, while bench died at step 78 reproducibly across seeds. That contradiction was the clue.
**CAUSE: the clamp was never in the file.** My edit asserted on a string from BEFORE qk-norm was added, matched an
older copy, and the replacement was overwritten by the later edit. The tool reported success; the file did not have
it. Line 101 still read `.exp()`.
**LESSON: verify the file, not the edit's return value.** The corrected edit now re-reads the file after writing and
asserts the fix is present. Four experiments and roughly an hour of GPU time went to chasing a bug that was already
diagnosed and simply not applied. When a probe and the harness disagree on identical math, suspect the harness
content before inventing new physics.

## GATING CLOSES 87% OF THE TRANSFORMER GAP (exp/bench.py, 100M tokens each, matched size)
| variant | params | loss | vs target | ktok/s |
|---------|--------|------|-----------|--------|
| **transformer (the target)** | 23.4M | **3.4617** | -- | 756 |
| ours before this work | 21.6M | 3.8747 | +0.4130 | 292 |
| + multi-head only (6 x dh=10) | 21.8M | 3.9149 | +0.4532 | 209 |
| + qk-norm only | 21.6M | 3.8578 | +0.3961 | 314 |
| + gating + qk-norm (dh=24) | 21.6M | 3.5794 | +0.1177 | 279 |
| **+ gating + qk-norm, dh=32** | 21.6M | **3.5165** | **+0.0548** | 221 |
| + gating + qk-norm, dh=48 | 21.7M | 3.5577 | +0.0960 | 144 |
**DATA-DEPENDENT GATING IS THE FIX: -0.28 nats on its own.** Replacing our single global decay constant with a
per-token forget rate closes most of the gap. This is exactly what separates Mamba-2 / GLA / RWKV-6 from plain
linear attention, and we had the constant-decay version.
**RETRIEVAL WIDTH SATURATES AT dh=32.** 24 -> 32 gains 0.063; 32 -> 48 LOSES 0.041 and costs 35% throughput.
**MULTI-HEAD ALONE HURTS** (+0.04): splitting dh=24 into 6 heads of dh=10 shrinks each head's Taylor space from
FE=325 to FE=66, and the lost per-head expressivity outweighs having several channels.
**qk-NORM IS FREE** (-0.017 nats AND faster) and is what makes gating numerically safe.
Remaining gap: **+0.055 nats**, from +0.413. Still behind on speed (221 vs 756 ktok/s) at this context length,
though our cost is flat in context while the transformer's is quadratic.
**THE FACULTIES ARE UNAFFECTED.** Gating changes the state update to M_t = g_t * M_{t-1} + k_t (x) v_t -- still a
fixed-size state, still constant cost per token, still carried across chunks. Memory, growth, and consolidation all
operate on that same state and need no change. The gate REPLACES our hand-set decay=0.05 with a learned per-token
rate, which is the "selective memory" the vision describes and we had implemented as a constant.

## A PRE-EXISTING BUG IN THE CHUNKED RECURRENCE: the boundary key is dropped
Porting the gate into core.py required checking that the chunked recurrence matches the full parallel form. It does
NOT -- and the mismatch is in the EXISTING ungated code, not the gate. In fp64 (so no precision excuse):
| path | max |logit diff| on the carried chunk | relative |
|------|------------------------------------|----------|
| ungated (existing code) | 9.559e-02 | **8.1e-02** |
| gated (new) | 6.632e-02 | 5.9e-02 |
A correct recurrence should be ~1e-12 relative in fp64. Localised it:
- state M after a chunk: **0.000e+00** vs the true prefix sum -- accumulation is exact
- qf, v: **0.000e+00** -- inputs are exact
- kf: differs at **exactly one position, index 0, by 6.4e-01** -- every other position is bit-exact
**CAUSE:** `_shift()` prepends a zero, so the key for the first token of a chunk is built from a ZERO activation
instead of from the last token of the previous chunk. Training on 32768-token documents in 2048-token chunks means
16 tokens per document get a wrong key -- 0.05% of positions, but it makes the recurrence formally inexact and it
grows worse as the chunk shrinks.
`prefill`/`step` ALREADY solve this for generation by threading `prev` (the previous normed activation) between
calls; the chunked TRAINING path never used it.
HONEST NOTE: my first attempted fix -- adding the missing key to the carried state -- did NOT work (2.333e-02 ->
2.334e-02) because the chunk still computes its own zero key at position 0, so the fix double-counts rather than
replaces. The correct fix is to pass the previous chunk's last normed activation into `_shift`, exactly as `step`
does with `prev`. Not yet applied: the benchmark is mid-run and this touches the training path.

## BOUNDARY KEY FIXED: the chunked recurrence is now EXACT
Threading the previous chunk's last normed activation into `_shift` (the same `prev` mechanism `step()` already
used for generation) makes the chunked form match the full parallel form to machine precision:
| | relative error, fp64 |
|---|---------------------|
| before | **8.1e-02** |
| after, ungated | **8.7e-16** |
| after, gated | **9.7e-16** |
My FIRST attempt at this failed silently: I passed `prev` but the 3-tuple return was conditioned on `prev is not
None`, so the first chunk took the 2-tuple branch and the caller never captured it -- the fix never engaged and the
error was unchanged at 7e-02. Added an explicit `return_prev` flag. Second time I checked the number rather than
trusting that the edit did what it said.
### STATE OF THE CLAIM, HONESTLY
Loss at 100M tokens: transformer **3.4617**, ours **3.5165** -- we are **+0.055 BEHIND**, not ahead. The gate closed
87% of a 0.413-nat gap, which is progress, not victory. And GENERATION QUALITY -- the thing the vision actually
claims -- has not been compared at all. Running both to 250M tokens now and generating from the same contexts with
the same decoding, because loss is not the deliverable.
