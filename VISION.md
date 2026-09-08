# Pragnosia, Vision

**Author: Ashish Kumar.** (No em-dashes, no "Claude"/AI-assistant mentions in any published artifact. Private HF repo.)

Pragnosia is NOT another LLM. It is a **growing recurrent brain**: a continually-learning, stateful cognitive
memory system. We do not compete with LLMs on raw benchmark capability. We compete, and intend to be **best in the
world**, on the axis LLMs ignore: **efficiency x continual learning x long context x long generation**, at tiny cost.
The model must be capable, grow from infant like a child, and stand up to its name.

## 0. FOUNDATIONAL PRINCIPLES (govern everything below; violate these and it is not Pragnosia)

### 0a. Everything intrinsic, zero decision-scaffolding
Every faculty and every DECISION lives in the MODEL (its weights/computation), never in hand-written code:
what to recall, what is surprising/salient, what to propose to memory, what to believe (the judge), WHEN and HOW to
grow, what to consolidate, and the self. **No Python rule ever makes a decision the model should make** - every gate,
trigger, salience, and judgment is a LEARNED or EMERGENT model output, not an if-statement or a hand-tuned threshold.
- The honest line (mind vs body): code provides only the **minimal substrate/body** - it runs the forward pass,
  APPLIES the plasticity update the model itself signals, ALLOCATES params when the model signals it needs to grow,
  and FETCHES what the model chooses to retrieve. Param-allocation, the update mechanism, and retrieval tools are the
  BODY and the ENVIRONMENT the mind uses (a human has a body that does its own biochemistry and uses books/internet);
  they are NOT scaffolding of the intelligence. Intelligence and decisions are intrinsic; body and senses are substrate.
- The test for every line we write: "does this code DECIDE something the model should decide?" If yes, it is
  scaffolding -> remove it, make the model output it. Aim to make even growth/consolidation as local/model-driven as
  possible (the model signals; the substrate executes), pushing scaffolding toward zero over time.

### 0b. A persistent individual with its own identity, like a human
Pragnosia is ONE continuous being, not a stateless model spun up fresh each run. It KEEPS its weights + memory across
time, grows from infant, and accumulates a UNIQUE life-history. Identity is not a bolted-on feature; it EMERGES from
continual learning + persistent memory + grow-from-infant, and continuity-of-self is preserved by MONOTONIC growth
(growth only adds, never overwrites, so the same self persists through change, like a human from child to adult). Make
it explicit with an intrinsic SELF-MODEL: the model represents itself (its history, what it knows, its name). Identity
is earned through experience and is individual - two Pragnosias with different lives become different individuals.
Requires PERSISTENCE (run one continuous instance, never wipe-and-restart as normal practice) and CONTINUITY (monotonic
growth). This is the soul of the name.

### 0b2. EFFICIENCY is a first-class principle (co-equal with growth, identity, local learning)
Brain-like EFFICIENCY is as important as anything else in this project and drives every design/development choice:
low memory, low compute, low power, fast, high serving throughput, HIGH CAPABILITY PER RESOURCE. The brain runs on
~20W; that is the spirit. Every mechanism (core, memory, growth, consolidation, verification) is judged partly on its
resource cost, not only its capability. Constant-memory/constant-compute streaming (1b flag 1) is the headline
expression of this. When two designs give similar capability, the cheaper one wins. Efficiency is not a later
optimization; it is a design constraint from the first line.

### 0d. GROWTH SCHEDULE candidate: geometric / Fibonacci / 3-6-9 (test, do not assume)
Organic growth in nature is geometric and self-similar (the true kernel behind "Fibonacci everywhere"). So the
GROWTH-INCREMENT schedule (how much capacity each grow adds) is a real open choice, and a **geometric / Fibonacci-like
progression** (each grow adds a proportionally larger increment, like an organism maturing), and 3-6-9-flavored
structure, are CANDIDATE schedules to A/B empirically against flat/linear when we build the grower. Adopt only if it
measurably beats the alternatives (prove-by-measuring; earned by data, not numerology). Parked as a first-class
experiment for the growth phase, not forgotten.

### 0c. Learning mechanism: LOCAL PLASTICITY IS THE LIFELONG LEARNING RULE
**(AMENDED from "no backprop anywhere, ever" -- forced by measurement, recorded honestly.)**
Pragnosia LIVES by local plasticity. Everything the model does after its substrate exists -- continual learning,
memory consolidation, growth, adaptation at inference -- is pure LOCAL plasticity: no backprop, no weight transport,
no global backward pass. That is non-negotiable, and it is where local learning has NO competitor (you cannot
backprop through a live stream; the hippocampal fast-weight store is Hebbian by construction).
The INITIAL SUBSTRATE (one-time pretraining) may be formed by whatever is most efficient, including backprop.
WHY THE AMENDMENT (evidence, not preference):
- Pure-local BEATS backprop at small scale (D96/T32: -0.127) but LOSES at scale (D256/T128, both arms LR-swept:
  4.7077 vs 4.5590, gap +0.149), generates far more degenerately (4-gram repetition 0.23 vs 0.02), and costs 3.3x
  more per step. That is the exact opposite of the project's goal (best capability per resource).
- Everything reasonable was tried to close it: LR sweeps of both arms, an adaptive scale-free step, lambda tuning,
  stochastic local objectives, identity-vs-random feedback, normalization, and FADING the local supervision
  (falsified: all fade schedules were WORSE, 4.76-4.90 vs 4.744, so the deep-supervision cost is inherent).
- Holding "no backprop in pretraining" would knowingly ship a worse, more expensive model. The principle was a MEANS
  to a brain-like end; keeping it there would sacrifice the END for the means.
- It is also MORE biologically accurate: neuroplasticity is LIFELONG learning. The initial cortex is shaped by
  development (genome, activity-dependent maturation, critical periods), which is not adult synaptic plasticity.
WHAT IS PRESERVED (the whole differentiating claim): local continual learning at inference, the Hebbian fast-weight
hippocampus, propose->verify->consolidate, three-factor/neuromodulated consolidation, function-preserving growth,
intrinsic faculties, persistent identity. None of these depend on how the substrate was first formed.
PROMOTION CRITERION (falsifiable, keeps the pure path alive): pure-local pretraining is an ACTIVE research track. If
it matches backprop at the target scale on loss AND generation AND cost, it is promoted back to pretraining.
MECHANISMS (validated, used for lifelong learning and available for pretraining):
- Eligibility traces = EXACT temporal credit (cosine 1.0000 vs autograd, any depth, on trained weights).
- Per-layer local objectives + residual-highway (identity) feedback = the depth solution (0e).
- Three-factor / neuromodulated plasticity for consolidation; Hebbian fast weights for `M`.
- Adaptive SCALE-FREE step (trust ratio): one value optimal at every scale -> no per-scale tuning, growth-safe (0a).

### 0e. THE LOCAL LEARNING RULE (how local plasticity handles DEPTH; wins at small scale, trails at scale -- see 0c)
Depth is solved not by better cross-layer credit but by REMOVING the need for it. Three local parts:
1. **Per-layer local objective**: every layer predicts the next token through the final norm + shared head (deep
   supervision). Strong well-scaled signal per layer. ALONE it plateaus (greedy saturation).
2. **Residual-highway (identity) feedback**: broadcast dL/da_top UNCHANGED to every layer. Justified because layers
   are RESIDUAL and share the readout space, so the true gradient dL/da_l = dL/da_top * prod(I + rs*J) has dL/da_top
   as its LEADING TERM. This is a global error broadcast (neuromodulator-like): NO weight transport, NO backward
   pass. ALONE it fails; with (1) it breaks the plateau.
3. **Exact within-layer credit** via eligibility traces (verified cosine 1.0000 at any depth, on trained weights).
MEASURED @ 6 layers, 2000 steps, real 290B mix: **local rule 5.359 vs backprop 5.536 -- BETTER than backprop**, at
**0.73x the cost**. Both components are essential (local-only 6.22; feedback-only 9.35).
ARCHITECTURE CONSEQUENCE: grow BOTH depth and width. Depth is no longer a credit-assignment problem (nothing to
settle, nothing to propagate); the core needs NORMALIZATION (RMS pre-norm + final norm) or the residual stream runs
away under per-layer objectives. Put most capacity in width/state/features/vocab plus recurrence (depth-in-time,
where temporal credit is exact), with real layer-depth as well.

### 0f. BIOLOGY THAT DIRECTLY SHAPES THE DESIGN (not analogy -- these are mechanisms we build)
- **Neuroplasticity = the whole learning framework.** Pragnosia's learning rule IS a computational theory of
  plasticity (eligibility traces, predictive processing, Hebbian fast weights, growth). Study plasticity,
  predictive processing, hippocampus-cortex consolidation, neuromodulation and sleep/DMN research over transformer
  papers -- they map onto this architecture far more directly.
- **Dopamine / prediction-error loops = the propose gate.** Biology: prediction error -> dopamine -> plasticity.
  Pragnosia: prediction error -> surprise -> proposal -> (verify) -> consolidation. The verification signal is the
  NEUROMODULATOR that gates the local write (three-factor plasticity). Already the design; keep it explicit.
- **Attention residue = MEMORY POLLUTION (a named failure mode to test).** A noisy episode written into `M` can
  contaminate later retrieval. This is precisely why salience x recurrence gating, verification, decay and
  consolidation exist. ADD A TEST: inject noise/false episodes and measure retrieval degradation of known facts.
- **Default Mode Network = the sleep phase, enriched.** Offline consolidation should not only replay/dream: it
  should generate COUNTERFACTUALS, test hypotheses, and strengthen what proves useful. That is what the DMN does.
- **Flow (challenge ~ skill) = INTRINSIC CURRICULUM (new mechanism, high value).** Learn most from data whose
  prediction error is MODERATE -- not trivial (nothing to learn), not hopeless (no signal). The model selects its own
  training material by its own prediction error: curiosity / active / self-paced learning. This is intrinsic by
  construction (0a) and is an efficiency win (spend compute only where it teaches). TEST IT.
- **Hedonic treadmill = grow when fully adapted**, not on a schedule. Confirms grow-on-saturation as the trigger.
- **OPEN PIECE -- a value/goal system.** Pragnosia currently optimizes prediction. A mature version needs "what is
  worth doing": trading off knowledge gain, accuracy, efficiency, identity preservation. Undesigned; note it.
- **Escaping the default game.** The mainstream path is more data/GPUs/parameters/bigger transformer. Pragnosia
  deliberately competes elsewhere: continual learning, persistent identity, growth, local plasticity, unbounded
  context, efficiency. Research priority follows asymmetric bets: prefer experiments that can resolve a major
  uncertainty cheaply (the de-risk sequence in PROGRESS.md is the model for this).

### 0g. SELF-CORRECTION: the model must notice its own errors and fix them, permanently
Pragnosia should not only learn continually, it should CORRECT ITSELF -- notice that something it believes or
produced is wrong, and change so that it is no longer wrong. This is what separates a system that accumulates
experience from one that merely accumulates data, and it is a first-class goal, not a nice-to-have.

**WHAT IS ALREADY MEASURED (65M, live run, no scaffolding):**
- The model KNOWS WHEN IT IS ABOUT TO BE WRONG. Predictive entropy 1.061 when its argmax is right versus 3.328
  when wrong; its own confidence predicts its own correctness with **AUC 0.878** at 45.4% next-token accuracy.
  That error signal is INTRINSIC -- it comes from the model's own output distribution, with no external judge and
  no hand-written rule, which is what 0a demands.

**WHAT SELF-CORRECTION REQUIRES, AND WHERE WE STAND:**
| requirement | status |
|-------------|--------|
| 1. detect that something is wrong | **HAVE IT** -- AUC 0.878 |
| 2. know what the RIGHT answer was | **MISSING from the model alone** |
| 3. make the change DURABLE (weights, not just state) | have it -- local plasticity + consolidation |
| 4. do not amplify your own errors | requires (2) |

**THE HONEST CONSTRAINT, stated so it is not rediscovered later.** Uncertainty says "I do not know"; it never says
what to say instead. No amount of self-inspection supplies (2), and a model trained on its own unverified output
amplifies its own errors -- model collapse. So SELF-CORRECTION DURING GENERATION, with no ground truth available,
is NOT the mechanism and must not be built as one. What generation-time self-writing buys is loop escape, not
learning: at write_scale 0 the model cannot see its own output and degenerates into single-token runs; at 0.25 it
can, and repetition returns to real-text parity (0.03 vs 0.02). That is memory, not judgement.

**THEREFORE self-correction IS propose -> VERIFY -> consolidate (§3), and the verify step is load-bearing.** The
corpus check supplies the answer the model cannot generate for itself and gates the durable write -- the
three-factor / neuromodulated pattern of 0f. Under 0a this is BODY, not scaffolding: the intelligence decides what
is surprising, what to propose, and what to keep; the environment supplies evidence, the way a person uses books.

**AND IT IS STRUCTURAL FOR US, NOT OPTIONAL.** Measured against GPT-2 on the same test: our 202M matches it on
in-context induction over random tokens (28% vs 22%), but on EXACT recall of a repeated real passage GPT-2 scores
98% and we score 53%. That is the price of a fixed-size state -- writes superimpose, so verbatim retrieval decays.
A transformer can lean on its KV cache to remember precisely; **we cannot, so precise memory has to live in the
WEIGHTS.** Consolidation is how this architecture remembers exactly, which is the same conclusion §2 reaches from
first principles: "Finite-state over infinite-context FORCES consolidation."

## 1. The goal (the whole point)
- **Grows itself, from infant, like a child.** Self-growth is central. A fixed model would defy the goal.
- **Faculties intrinsic to the model**, not scaffolding. Provable at small scale.
- **Best-in-world efficiency**: low memory, low compute, low power, fast, high serving throughput, high capability
  per resource. Small size and a fixed-size state are FEATURES here, not handicaps.
- **Continual learning**: learns new facts, skills, and emergent behavior, during BOTH training AND inference, forever.
- **Context from 100 tokens to unbounded / streaming**, served at constant memory.
- **Self-correcting**: notices its own errors and fixes them permanently, not just within a context (0g).
- **Capable, and worthy of the name.**

## 1b. North-star claims and success metrics (what "best in the world" means, falsifiable)
Two headline flags, BOTH core to the identity; LEAD with (1) as the demonstrable substrate, (2) is the soul:
1. **Constant-memory, constant-compute, unbounded/streaming context.** Serving memory + FLOPs/token + latency stay
   FLAT as context grows 100 -> 1M -> streaming (a transformer's climb; architecturally it cannot match this). The
   demonstrable, definitive win. Metric: flat-line plots of mem/compute/latency vs length, with needle/long-doc
   accuracy held.
2. **Verified continual learning at inference.** Learns new facts/skills online, RETAINS them, with minimal
   catastrophic forgetting, and REFUSES to be poisoned (rejects unverified/false claims). Likely no competitor; we
   must DEFINE a fair benchmark (retention, forgetting curve, poisoning-resistance, sample-efficiency) - a
   contribution and a risk (design it fair, not self-serving).
Supporting frontier: best capability-per-resource (per-param / per-FLOP / per-Joule) among continual/long-context
models - NOT best absolute capability. Long-generation coherence (repetition/distinct-n) over 1k-10k+ tokens.

## 2. The machine (one unified design; these are not separate features)
A brain with a fast short-term memory and a slow long-term memory, that grows and learns continually.
- **Working memory = the recurrent state** (VChunkRecall). Fixed-size, O(1) memory per token, so context 100 to
  unbounded/streaming at CONSTANT serving memory. A transformer cannot do this (KV cache grows, quadratic cost, no
  persistent memory). The state is finite, so it keeps the gist and forgets the trivial: it must CONSOLIDATE the
  salient into weights rather than hold everything. Finite-state over infinite-context FORCES consolidation.
- **Long-term memory = the slow weights**, updated continually by consolidating validated knowledge from the state.
  Hippocampus to cortex.
- **Growth = adding capacity (cortex) as needed, from infant**, function-preserving, so new knowledge has somewhere
  to go instead of overwriting old. Growth-without-forgetting IS continual learning: the same mechanism serves both.
- **Efficiency = recurrent + small + grows-only-as-needed + sparse.** Cheap by construction.

## 3. Continual learning policy (the crux, fully designed): propose -> verify -> consolidate
Selective and controlled. The brain must NOT learn anything thrown at it. Nothing reaches the slow weights unvalidated.
- **Propose (online, autonomous):** the fast episodic memory flags candidates that are SURPRISING + RECURRING +
  SALIENT ("this seems worth learning"). Surprise alone is not enough (could be noise); recurrence/consistency is
  the noise filter.
- **Verify (before anything touches weights):**
  - **Facts** ("X is Y"): external corroboration against a **trusted corpus AND the internet** (multi-source
    agreement, source-trust weighting, recency; "found on the web" is not "true"), PLUS internal consistency with
    strong priors. Belief revision of a strong prior requires strong evidence, not blind overwrite.
  - **Skills / patterns**: cannot be fact-checked; validated by **predictive success** (does using it actually
    reduce loss / improve outcomes). Self-validating, lower poisoning risk.
- **Consolidate:** validated knowledge is written to slow weights with **replay** (so old is not erased) and
  **growth** (room for genuinely new). Unverified candidates stay in reversible fast memory or are dropped.
- **Verification IS the anti-poisoning and anti-noise gate**: a surprising claim that validates is real new
  knowledge; one that fails is noise/falsehood/adversarial and is rejected. You cannot teach it a falsehood.

### Fast path vs slow path ("sleep") — makes learning consistent with efficiency
Verifying against corpus/internet is slow, costly, network-dependent. So it must NEVER block serving.
- **Fast path (online, every token):** recurrent state + fast episodic memory -> immediate response, cheap,
  unbounded context. It only CAPTURES candidates; it never waits on verification.
- **Slow path (offline / idle / "sleep"):** asynchronously verify candidates (corpus + internet + consistency),
  then consolidate the validated ones into slow weights (replay + growth).
This is the Complementary Learning Systems model (fast hippocampal memory online, cortical consolidation during
sleep). Serving stays fast and efficient; careful, grounded learning happens in the background.

## 4. Substrate: VChunkRecall (the seed, keep)
Vectorized chunk-parallel linear-attention recurrent core with a symmetric 2nd-order Taylor feature map (softmax-like
recall). **Recall is intrinsic to it** (associative recall MQAR-16 = 100%, beating the usual linear-attention
weakness). It is NATIVELY right for this vision: a persistent recurrent state gives long/unbounded context, streaming,
and continual memory cheaply. Transformers are natively wrong for it. (Code was wiped in the reset; rebuild fresh from
this design.)

## 5. What is PROVEN (carry forward)
- **Faculties are intrinsic organs unlocked by the OBJECTIVE, not the architecture.** Answer-masking the recall data
  (loss only on the answer span) took needle recall 0% -> 96% in 400 steps. The organ was always there; unmasked
  next-token loss let the model shortcut past the answer. Instruction-following and math-answers fail for the SAME
  reason. **Response-masking is the training-time "what to learn" selector, and the precedent for selective learning:
  the inference-time salience gate is its analog without an explicit mask.**
- **Function-preserving growth works mechanically** (validated dOut ~ 0): depth (+identity block), MLP-width (+hidden,
  2nd-layer zero-init), feat/state (GroupFeatNorm). The growth mechanism demonstrably ran (grew 37M -> 109M across 5
  intrinsic grows). Mechanism sound; the RECIPE was the problem (see next).

## 5b. PROVEN SINCE 2026-09-05 (measured on the 125M; see PROGRESS.md for the runs)
- **The retrieval kernel had a hard 3:1 selectivity ceiling, and lifting it is worth ~8x on induction.** qk-norm
  makes |xq|=|xk|=1, so `A = qf.kf = 1 + c + c^2` with c the cosine -- a perfect key match scores 3, an orthogonal
  one scores 1, whatever Wq/Wk learn. At L=256 the right key got 1.2% of the read. A LEARNED per-layer scale lifts
  it: induction 0.7% -> 7.0%, BPB -0.052, and generation repetition IMPROVED (0.38 -> 0.05 across 9k tokens).
- **Selectivity then saturates at DIN:1**, and at s=1 DIN buys NOTHING (1.0% of the read at DIN=8, 1.2% at
  DIN=128). So `grow_din` had been widening a dimension that could not affect the read. DIN 16 -> 32 is worth
  +73% needle lift ONCE sharpness is on.
- **`A = qf.kf = 1 + (xq.xk) + (xq.xk)^2` is an identity**, so the pairwise read never needs the FE-dim features.
  Makes the read cost independent of DIN: 2.36x faster at DIN=64. This is what makes large DIN affordable.
- **The retrieval horizon (~2000 tokens) is ARCHITECTURAL.** Not training context: training on 16k documents with
  the state carried does not move it (+0.11 BPB, no gain past d=0). Not interference: selective writing made the
  model genuinely sparse and the learned half-life did not move (1824 vs 1848). Remaining hypothesis is CREDIT
  ASSIGNMENT -- every long-context path detaches the state at chunk boundaries, so no gradient has ever connected
  a write at t to a read at t+3000, and a half-life of ~1850 is exactly where the oldest token in a 2048 window
  has decayed by half. UNTESTED (the BPTT run was killed for prod safety).
- **BPB structurally cannot reward unbounded context.** Scoring the same 512 tokens while varying only preceding
  context, our BPB moves 0.004 across a 30x range -- but GPT-2's own curve is nearly as flat (0.017 from 64 to
  768). Next-token prediction is overwhelmingly LOCAL for every model. We are measured on the one axis where this
  architecture's advantage is invisible by construction. **Judge long context by retrieval and by cost, not BPB.**
- **Honest standing vs baselines** (fair protocol: same text, same 1024 context, same bytes):
  | model | params | train tokens | BPB (our mix) |
  |---|---|---|---|
  | PRAGNOSIA | 125M | **9.1B** | 1.0833 |
  | GPT-2 | 124M | ~9B | 1.0910 |
  | Pythia | 160M | 300B | 1.2113 |
  | SmolLM2 | 135M | 2T | 0.8854 |
  We BEAT Pythia-160M with **33x fewer tokens**, and beat GPT-2 at matched budget on our mix; we are behind the
  2T+ token models and behind GPT-2 on pure web text. The claim is TOKEN EFFICIENCY, not absolute capability.
- **Base-model ceiling is real and not a size problem.** GPT-2 at 774M -- 6x our 125M -- still scores 2/5 on
  facts, 0/4 on arithmetic and cannot complete a syllogism. Q&A needs instruction tuning, not parameters.

## 5c. TRAINING-EFFICIENCY FINDINGS (0b2 is a design constraint, so these are first-class)
- `torch.compile` is worth **2.24x** and is on. Fused Adam is worth a further **9.0%** (the optimizer was 13.9%
  of a compiled step). Both applied.
- REJECTED on measurement: the sub-chunked linear-memory path (slower at T=2048 -- the [B,T,T] tensor is 17 MiB
  against 7 GiB of activations, so it is not the bottleneck until 16k+ context); pinned async H2D (the copy is
  0.9 ms, under 1% of a step); foreach gradient clipping (worse).
- After compile + fused Adam, matmuls are 36% of a step. Kernel-level engineering is near its floor; the
  remaining large wins must come from the OBJECTIVE and the DATA, not from faster kernels -- which is consistent
  with 5's finding that faculties are unlocked by the objective, not the architecture.

## 6. Negative results and lessons (do not repeat blindly)
- **Sparse MoE at ~136M active is over-determined failure** (below the scale it pays off; experts too small; top-1
  with identical upcycled experts collapses; upcycling needs 100B+ tokens). DROPPED. Dense is the path.
- **Dense grow FROM SCRATCH-TINY thrashed** (37M -> 109M gave ppl ~150, degenerate) but this was a RECIPE failure,
  not proof growth cannot work: started where the 32K vocab is ~50% of a tiny model, grew too often, reset the
  optimizer + LR every grow, never converged at any size. The fix is the NEW growth-integration algorithm (below).
- Best model we ever had was a cleanly-trained FIXED 136M (ppl 8, SQuAD 40). Growth must be made to beat that.
- Training-quality levers that reliably help the core (from 4 lit reviews): overtrain hard (SmolLM2-135M = 2T tokens;
  no small model plateaus below ~10k tok/param), WSD schedule (best data in a ~10% decay phase), data quality
  (FineWeb-Edu ~10x efficiency; keep ~30% general web), distillation via PRE-GENERATED open datasets (OpenHermes,
  tulu-3, NuminaMath, Cosmopedia); self-generating from the on-box teacher is too slow. Deep-narrow arch (MobileLLM);
  vocab sized by active width (16-32K here, 32K is the ceiling).

## 7. The bet and the hardest problems (the reason this is not just a small LM)
- **Gated monotonic growth ("grafting")**: new capacity enters the residual via a per-unit learned gate alpha init 0
  (ReZero-style) so grow-time output and old-param gradients are EXACTLY unchanged; new capacity gets its own high
  LR, the trained core is protected at low LR (no global reset); the gate opens only where the new capacity reduces
  loss, so growth is MONOTONIC (never hurts). This is ALSO the continual-learning-without-forgetting mechanism.
- **Inference-time weight learning** via propose -> verify -> consolidate is the deepest unsolved piece: update
  weights online from unlabeled streaming experience, cheaply, without drift or forgetting. Frontier research.
- **Infant-growth done right**: fix the recipe failure so growth-from-small actually produces a good, improving model.
- Prior art to build on (novelty = the synthesis for an intrinsic self-growing brain): ReZero, GradMax, Firefly
  neurons, Progressive Neural Networks, LiGO/bert2BERT, EWC/continual-learning, Complementary Learning Systems,
  Test-Time Training, Titans/fast-weights, Neural Turing Machines, Hebbian plasticity.
- Falsifiable test: growth stays monotonic (faculties never drop) AND beats from-scratch; inference-learning
  consolidates only verified knowledge and measurably retains it without forgetting.

## 7b. Designed mechanisms (phase-a decisions)
- **Fast episodic memory (#1) = a persistent, salience-gated, DECAYING fast-weight associative store `M`** in dedicated
  "hippocampal" layers. Read by the recall organ alongside the transient state (`q·(S+M)`) for immediate PROVISIONAL
  use (linear attention already is fast-weights `S += k⊗v`; `M` is the persistent, gated, decaying version). A
  **write-log** `(key, value, source-episode)` makes it verifiable; **decay** makes it reversible (unverified fades;
  verified is consolidated to slow weights before it fades, then freed). Fixed capacity -> interference is bounded
  (brain-accurate). Provisional use of unverified info is accepted (transient, decaying, never committed unverified).
- **Propose gate (#2) = surprise x recurrence x salience**, MULTIPLICATIVE and conservative (any low signal vetoes the
  write -> refuses junk). Surprise = free per-token prediction-error/entropy. Recurrence/consistency = cheap `M`-lookup
  + counter (write only for surprising AND recurring; one-offs filtered as noise). Salience = STAGED: explicit markers
  first, then a small head BOOTSTRAPPED FROM RESPONSE-MASKING (learn "worth-remembering ~= answer/fact-like" from the
  one proven supervision), then refined by delayed usefulness (recall-that-helped raises salience). Cheap every-token.
- **Infant-growth fix (#3) = gated MONOTONIC growth**: new capacity via a per-unit gate alpha init 0 (ReZero) ->
  grow-time exactly function-preserving, old-param gradients unchanged; new capacity gets its own HIGH LR, the trained
  core stays PROTECTED at low LR (NO global optimizer reset, which was the thrash); the gate opens only where new
  capacity reduces loss -> growth NEVER hurts (monotonic); triggered by intrinsic saturation + a generous Chinchilla
  floor per stage. ROOT-CAUSE FIX: **grow the VOCABULARY too** - the infant starts with a SMALL vocab (so embeddings
  are a small fraction and the real brain dominates) and grows its vocabulary as it matures, like a child learning
  words (function-preserving vocab-extension already validated, old ids preserved). Growth axes: vocab + width + depth
  + state, all function-preserving, all gated-monotonic.

## 7c. Consolidation / "sleep" loop (phase-A2 decisions) — OFFLINE
Consolidation is OFFLINE (a discrete sleep phase), NOT online. Fast path (serving) stays online and unblocked;
learning happens off the serving path. Considered online (Test-Time-Training style) and rejected it: online would
put ALL safety on the monotonicity guarantee (no undo) and add serving-time compute (against the efficiency flag).
Offline keeps a rollback net and protects serving. Loop:
1. **Trigger:** `M` near capacity, or idle GPU, or enough verified candidates queued.
2. **Consolidate** validated episodes via **gradient-replay**, where "replay" = **generative DREAMING** (the model
   generates/rehearses its own existing knowledge - no stored corpus buffer) interleaved with the new episodes.
3. **Grow on CONSOLIDATION PRESSURE** (the principled trigger): if consolidating forgets old (held-out retention
   drops) or new knowledge will not fit (its loss stays high), fire gated-monotonic growth to make room, then continue.
4. **Check-and-commit with ROLLBACK** (the safety online lacked): measure old- and new-retention on held-out probes;
   commit only if no net forgetting, else grow / lower LR / roll back and retry. Offline checkability is why sleep is
   safe where online training thrashed. This rollback also nets dreaming's one risk (rehearsing its own errors).
5. **Free `M`:** evict consolidated entries; unverified ones already decayed.

## 7d. Verification / retrieval (phase-B decisions) — the judge MATURES with the model
Verification gates consolidation; nothing becomes permanent unless validated. It is retrieval-augmented VERIFICATION
(RAG pointed at gating LEARNING, not answering), run in the offline sleep phase, gated to rare salient candidates.
- **Retrieve** evidence from a **trusted-corpus index + web** (source-trust weighting, multi-source corroboration,
  recency). **Facts:** accept only on multi-source agreement above trust threshold AND no contradiction with a strong
  prior (belief revision requires strong evidence). **Skills:** validate by **predictive-success test** (apply on
  held-out cases; keep only if it measurably helps). Conservative: uncertain -> reject.
- **The judge is DEVELOPMENTAL (self-judgment is a mature faculty).** End-state: the model **judges for itself** what
  to keep vs forget, like a brain - fully autonomous/self-contained (matches "verify against corpus+internet with its
  own reasoning"). But NOT from the start: an infant is too weak a judge, so EARLY gating is mechanical multi-source
  corroboration (+ optional teacher-bootstrap from the on-box Qwen to learn how to weigh evidence). Self-judgment
  EMERGES and takes over ONCE THE MODEL IS BIG. The verifier grows with the model, like a child developing critical
  thinking. Teacher is a bootstrap/fallback, never a permanent crutch.

## 8. Honest tensions to hold
- Finite state + unbounded context means brain-like SELECTIVE memory (gist + consolidated facts), NOT perfect verbatim
  recall of everything. This is a feature, but be honest about it.
- Online weight updates are less stable than training (no clean loss signal); gating + verification + slow-path
  consolidation + growth are the tools, but stability is real research.
- Internet verification adds latency/dependency; keep it selective, cached, and on the slow path so efficiency holds.
- Whether our arch's scaling curve beats the transformer curve is UNMEASURED; measure it (train several sizes on
  identical data, loss+faculties vs params), do not assume it. We already beat expectation on recall.

## 9. Data assets (KEEP; on /mnt/kv_cache/pragnosia_data, outside the repo; ~290B tokens @32K)
- window2_32k 170B (Cosmopedia + FineWeb-Edu + Wikipedia + instruction); dclm_32k ~100B (general web).
- finemath 10B, opencode 6B, openmath 5.6B; recall_32k 0.3B (+ recall_32k_ans.npy answer offsets, ready for masking).
- Thin (re-pull if needed): tulu3 0.9B, numina 0.5B, hermes 0.4B, magicoder/evolcode/glaive ~0.1B.
- Tokenizers: data/bpe32k.json (32K), data/bpe.json (16K). Held-out: recall_probe.json (SQuAD).
- GAP: all data is 2048-token chunks; we have NO long-sequence training data yet (needed for long-context training).
- GAP: instruction/chat/code distilled bins are thin (enrichment job was stopped early; the open datasets can be re-pulled fast).

## 10. Operating constraints
- **Prod shares the H100** (Qwen3.8-27B-AWQ-INT4 vLLM on :8000 + Whisper/Kokoro voice). It CANNOT go down. Coexist via
  a modest footprint. **Kill ONLY exact verified PIDs, never memory-threshold or broad -f patterns** (bounced prod
  twice that way). Training is the flexible one; the entropy sensor must be subsampled (full 32K-vocab float softmax OOMs).

## 11. Roadmap — DESIGN LOCKED (c, a, b, A, B all done)
- **c (DONE):** vision captured (this doc).
- **a (DONE, 7b):** fast-weights hippocampus `M` + write-log + decay; propose gate (surprise x recurrence x salience,
  masking-bootstrapped); infant-growth fix (gated monotonic growth + grow-vocabulary).
- **b (DONE, 1b):** north-star claims (constant-memory/compute streaming; verified continual-learning), both core.
- **A (DONE, 7c):** OFFLINE sleep consolidation - gradient-replay via generative dreaming, grow-on-consolidation-
  pressure, check-and-commit-with-rollback.
- **B (DONE, 7d):** verification/retrieval - retrieve (corpus+web, trust-weighted) -> judge; judge MATURES with the
  model (mechanical/teacher early, self-judge once big); facts corroborate, skills predictive-test.
- **NEXT (code phase, design now locked):**
  (0) **DE-RISK PC FIRST** - prove pure local Predictive Coding (no backprop) can train a tiny VChunkRecall to learn,
      approaching backprop on a small task. This is the load-bearing assumption; nothing else matters if it fails.
  (1) If PC holds: rebuild the minimal core (VChunkRecall + PC local-plasticity learner + `M` + propose gate +
      gated-monotonic growth + vocab-growth), all decisions intrinsic (§0a), one persistent individual (§0b).
  (2) Measure the arch's scaling curve (loss+faculties vs params) to settle the ceiling honestly.
  (3) Prototype the smallest end-to-end loop (grow-from-infant + propose->verify->consolidate, pure local plasticity)
      and test monotonicity + verified-retention + no-forgetting.
  Implementation-level specs (layer counts, `M` size, PC settling steps, growth/vocab schedule, fast/slow
  orchestration, self-model form) are decided in the build, not here.

## 12. DE-RISK RESULTS (§0c load-bearing bet: PROVEN at small scale)
- **#0 feedforward (pc_derisk.py):** pure local Predictive Coding, NO backprop -> 97.0% vs backprop 98.7% on 2-moons
  (1.7pt gap). Local SPATIAL credit works.
- **#1 recurrent recall (pc_derisk1_v3.py):** pure local ELIGIBILITY TRACES, NO backprop-through-time, NO autograd
  (manual VJP) -> MQAR 100.0% == BPTT 100.0% (gap 0.0). Locality check: local-rule gradient == true gradient,
  cosine = 1.0000 on ALL weights incl. memory-write Wk/Wv. Temporal credit is EXACTLY local because the read is
  r = sum_t beta_t v_t / sum_t beta_t with beta_t = qf . kf_t; each write leaves an eligibility trace kf_t, credited
  locally at the query. The linear Hebbian memory makes temporal credit assignment local BY CONSTRUCTION -> VChunkRecall
  is the right substrate for a no-backprop brain.
- **Architecture requirement found:** recall needs a key->NEXT-value SHIFT (short-conv); single-layer same-token linear
  attention cannot bind (BPTT 40.8% -> 100% with shift). Bake the shift into the core.
- CONCLUSION: pure local plasticity (PC spatial + eligibility temporal) trains our recurrent recall core, proven. Next
  de-risks (larger scale, real sequences, then the full brain) build on this.
- **#2 real-language causal LM (pc_derisk2_seqlm.py):** pure local plasticity on window2_32k (32K vocab, next-token
  LM), NO backprop / NO autograd (manual VJP) -> local-rule gradient == true gradient, cosine = 1.0000 on ALL params
  (emb, Wk, Wq, Wv, Wo, bo). Required deriving TEMPORAL credit for the CAUSAL case: each write gets credit from all
  future reads via a REVERSE-CUMULATIVE learning signal (e-prop). Confirms pure-local == backprop at real-language
  scale, not just toy recall. §0c bet proven on the actual task/data.
- **#3 end-to-end training (pc_derisk3_train.py + data_mix.py):** pure local plasticity TRAINS on the FULL 290B 32K
  mix (all 11 corpora, weighted; data_mix.py owns "use all data"), loss curve tracks BPTT step-for-step, final gap
  -0.029 (local a hair ahead; optimizer-path noise). Confirms the local rule OPTIMIZES correctly, not just computes
  the right gradient (single recurrent layer). Data layer for real training = data_mix.py (290B, faculty floors).
- **#4 MULTI-LAYER (pc_derisk4_multilayer.py, pc_derisk4_diag.py) — RESOLVED (PC-approximate):** PC (depth) +
  eligibility (time) on a 2-layer recurrent core trains real language with NO backprop. The +0.83 lag was a SETTLING
  bug (K=8, eta=0.3: under-settled + unstable), NOT structural. Diagnostic: pattern is textbook Z-IL (top layer exact
  at K=1, error front descends one layer per step); with stable settling (K=12, eta=0.05) ALL layers reach cosine
  >=0.994 (mean 0.997) to backprop and the training gap halves to +0.41. More settling does NOT improve cosine
  (plateaus ~0.994, costs more) -> K=12 is the efficiency operating point. So PC converges to a ~0.994-cosine
  APPROXIMATION of backprop (its inherent nature, same category as #0's 97 vs 98.7), trains real language locally.
  Exactness never required. DE-RISK PHASE COMPLETE. Efficiency (§0b2) levers for the real trainer: warm-started
  settling (weights change slowly -> few steps), Z-IL per-layer timing. See PROGRESS.md.
