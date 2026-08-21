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

### 0c. Learning mechanism: PURE LOCAL PLASTICITY, no backprop anywhere (decided: B-ii)
Pragnosia learns entirely by LOCAL plasticity - no backpropagation, no external optimizer, ever (not even to
meta-learn a rule; not in "evolution", not in life). Two complementary local mechanisms:
- **Predictive Coding (PC) = SPATIAL credit** (across layers): every layer predicts its neighbor, forms a LOCAL
  prediction error, updates to reduce it (provably does credit assignment with local-only computation, no global
  backward pass). DE-RISK #0 confirmed PC matches backprop (97.0 vs 98.7) on a static task.
- **Eligibility traces (e-prop, Bellec 2020) = TEMPORAL credit** (across time): each synapse keeps a LOCAL trace of
  recent pre-synaptic activity, so a later error/learning-signal credits earlier events (e.g. the memory WRITE that
  a future READ used) - temporal credit with NO backprop-through-time. DE-RISK #1 v1 showed PC/per-position local
  WITHOUT this fails recall (12% = chance): the writes get no credit for future reads. Eligibility traces are the fix.
Plus **three-factor/neuromodulated plasticity** for consolidation (the verification signal gates the local write) and
**Hebbian fast-weights** for `M`/state (already local). PC (space) + eligibility traces (time) = full local credit. The unifying quantity is PREDICTION ERROR / SURPRISE: it is the learning signal (PC),
the salience signal (propose gate), and the model's uncertainty, all one intrinsic thing. PC nets are recurrent
(settle by minimizing error), fitting our core; this is Friston free-energy, maximally brain-true.
- HONEST RISK (the load-bearing bet): PC at our scale, with growth + continual learning, is UNPROVEN; it may be slower,
  weaker than backprop, or unstable, and costs more compute per step (settling). The whole project rests on it working.
- THEREFORE the first code step DE-RISKS PC: prove local PC (no backprop) can train a tiny VChunkRecall to learn,
  approaching backprop on a small task, BEFORE building the full brain on it.
- Alternatives if PC stalls: Equilibrium Propagation, Forward-Forward (both local, no backprop). PC is the lead.

## 1. The goal (the whole point)
- **Grows itself, from infant, like a child.** Self-growth is central. A fixed model would defy the goal.
- **Faculties intrinsic to the model**, not scaffolding. Provable at small scale.
- **Best-in-world efficiency**: low memory, low compute, low power, fast, high serving throughput, high capability
  per resource. Small size and a fixed-size state are FEATURES here, not handicaps.
- **Continual learning**: learns new facts, skills, and emergent behavior, during BOTH training AND inference, forever.
- **Context from 100 tokens to unbounded / streaming**, served at constant memory.
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
