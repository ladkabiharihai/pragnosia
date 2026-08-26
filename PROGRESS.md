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
