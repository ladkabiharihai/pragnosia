# Pragnosia — Vision & Hard-Won Findings (clean-slate reset)

**Author:** Ashish Kumar. (No em-dashes, no "Claude"/AI-assistant mentions in any published artifact. Private HF repo.)

## The goal (unchanged, and it is the whole point)
A **brain-like model that grows itself** — compute-efficient, faculties INTRINSIC to the model (not scaffolding),
**provable at small scale**. The differentiating research claim is *self-growth like a brain*, NOT "another good small LM."
A fixed dense model would defy the goal. Growth is central.

## Substrate (keep)
- **VChunkRecall** — vectorized chunk-parallel linear-attention core with a symmetric 2nd-order Taylor feature map
  (softmax-like recall). Faster than attention on long context, memory-parity when checkpointed. **Recall is INTRINSIC
  to this core** (associative recall MQAR-16 = 100%, beating the usual linear-attention weakness). This is the seed.

## What is PROVEN
- **Recall is an intrinsic organ, unlocked by objective not architecture**: answer-masking the recall data (loss only on
  the answer after `Answer:`/`<assistant>`) took needle recall 0% -> 96% in 400 steps. The organ was always there;
  unmasked next-token loss just let the model shortcut past the answer. Generalizes: instruction-following and
  math-answers fail for the SAME reason -> **response-masking is the lever** (fold it INTO training, not a bolt-on).
- **Function-preserving growth works mechanically** (validated dOut~0): depth (+identity block), MLP-width (+hidden,
  2nd-layer zero-init), feat/state (GroupFeatNorm), MoE expert-add. Growth is CHEAPER-to-target (~31-36% less compute),
  NOT free capability.
- Speed: symmetric-Taylor + torch.compile + bf16 + no-checkpoint-when-VRAM-ample = big throughput wins. E=1 dense
  fast-path. Grad-checkpointing was the real bottleneck at small scale, not the model.

## What FAILED (negative results — do not repeat blindly)
- **MoE at 136M active is over-determined failure**: below the scale MoE pays off (dense wins at equal TOTAL params
  sub-1B); 6M experts too small to specialize; top-1 with 2 identical upcycled experts = worst-case router collapse;
  upcycling needs ~100B+ tokens to recover. Recall 100%->0% after the grow. Shared-expert MoE fixes only fragmentation
  (1 of 4 causes). => DROP sparse MoE at this scale.
- **Dense self-grow FROM SCRATCH-TINY thrashes**: 37M->109M intrinsic grow gave ppl ~150, degenerate generation,
  sub-chance faculties — MUCH worse than a cleanly-trained fixed 136M (ppl 8, SQuAD 40). Causes: started where 32K vocab
  is ~50% of params (real "brain" ~15M), grew ~every 94k steps, optimizer reset + LR re-warm per grow, never converged
  at any size. Nested grow_feat also makes snapshots hard to reconstruct (order-dependent).
- Scaling/capability from growth NOT demonstrable at this scale across 5+ studies. Best model ever = FIXED, cleanly trained.

## Research verdict (4 lit reviews, unanimous) — the levers that actually work at 100M-1B
1. **Overtrain hard**: we were at 38 tok/param; SmolLM2-135M (our size, SOTA) = 2T tokens (~14,800 tok/param),
   MobileLLM-125M = 8,000. No small model plateaus below ~10k tok/param. Use **WSD schedule** (long stable + ~10% decay
   phase loaded with the BEST data — MiniCPM). Repeat corpus ~4 epochs free, ~16 diminishing (datablations).
2. **Data quality**: FineWeb-Edu ~10x token efficiency (MMLU 33->37, ARC 46->57) but edu-filtering COSTS HellaSwag/PIQA
   -> keep ~30% general web (SmolLM 70/30). Phi/Cosmopedia: synthetic textbook data is a big lever at sub-1B.
3. **Distillation = use PRE-GENERATED open datasets** (OpenHermes, tulu-3, NuminaMath, OpenMathInstruct, Magicoder,
   Cosmopedia = large-teacher outputs). Self-generating from the on-box Qwen-27B is too slow (prod-safe ~550 tok/s =
   ~6 days for 300M) — only use the teacher for the small unique jobs (on-policy correction). Mismatched-tokenizer +
   API => only sequence-level/synthetic-data KD is feasible anyway.
4. **Architecture for small**: deep-narrow (MobileLLM), tied embeddings. Vocab sized by ACTIVE width: keep ~16-32K at
   136M (128K/256K would make embeddings dominate). 32K is the ceiling at this size.
- **Honest ceiling**: a well-built 136M reaches SmolLM2-135M territory (useful small brain) — NOT a general assistant;
  knowledge storage is ~param-bound (~2 bits/param). BUT this is TRANSFORMER-measured. **We must MEASURE our own
  arch's scaling curve** (train 40/80/136/300M identical data, loss+faculties vs params) rather than assume it — we
  already beat the transformer-linear-attention expectation on recall.

## THE NEW DIRECTION (the actual bet, the reason not to just build a fixed LM)
Invent a **growth-integration algorithm** so growth stops thrashing — the specific unsolved problem all our failures
share: add capacity to a trained model and keep improving, non-destructively. Biology does this via LOCAL plasticity +
gating (neurogenesis), NOT global optimizer resets. Candidate mechanism = **GATED MONOTONIC GROWTH ("grafting")**:
- New capacity enters the residual via a per-unit learned gate **alpha init 0** (ReZero-style) -> grow-time output AND
  old-param gradients EXACTLY unchanged (truly non-disruptive).
- New capacity gets its OWN high LR; trained core kept at low LR (protected) — NO global reset.
- Gate opens only where new capacity reduces loss -> growth is **MONOTONIC: never hurts, only helps or stays neutral.**
- Trigger: the model's own intrinsic saturation signal (entropy/loss plateau) + Chinchilla floor.
- **Grow from STRENGTH** (a matured core), not scratch-tiny — bert2BERT/LiGO show grow-from-a-good-base is the
  demonstrable efficiency win.
Prior art to build on (novelty = the synthesis for intrinsic self-growth): ReZero, GradMax, Firefly neurons,
Progressive Neural Networks, LiGO/bert2BERT, EWC/continual-learning. Falsifiable test: does growth stay monotonic
(faculties never drop) AND beat from-scratch at the target size?

## DATA assets (KEEP — on /mnt/kv_cache/pragnosia_data, outside the repo)
- window2_32k.bin (~170B, 32K vocab: Cosmopedia + FineWeb-Edu + Wikipedia + instruction) — DONE
- dclm_32k.bin (general web, re-tokenizing to ~100B), recall_32k.bin (+ _ans.npy answer offsets)
- 9 faculty enrichment bins @32K (finemath, opencode, numina, openmath, hermes, magicoder, evolcode, tulu3, glaive) —
  these ARE pre-distilled teacher data
- data/bpe32k.json (32K tokenizer, extends the 16K preserving old ids), data/bpe.json (16K)
- distill.jsonl (small Qwen-generated bonus), recall_probe.json (held-out SQuAD)
- 16K legacy bins (window2_train 176B, recall_train, etc.)

## Operating constraints
- **Prod shares the H100** (Qwen3.8-27B-AWQ-INT4 vLLM on :8000 + Whisper/Kokoro voice). CANNOT go down. Coexist via
  modest footprint. **KILL ONLY EXACT VERIFIED PIDs — never memory-threshold or broad -f patterns** (bounced prod
  twice that way). Training is the flexible one.
- Eager + uncapped only on the spin-era model; for the new work, torch.compile is OK on VChunkRecall (symmetric-Taylor).
- Scaling test / any real run: coexist with prod, entropy sensor must be subsampled (full 32K-vocab float softmax OOMs).

## Immediate next steps (post-reset)
1. Rebuild minimal: VChunkRecall core + a clean dense trainer + the gated-monotonic-growth mechanism.
2. Mature a small core cleanly (32K enriched, response-masked recall/instruction), overtrain.
3. Grow it with gated-monotonic growth from strength; measure monotonicity + vs-from-scratch.
4. Measure our own scaling curve to settle the ceiling question honestly.
