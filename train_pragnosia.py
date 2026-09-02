"""THE REAL RUN: grow Pragnosia from infant to ~200M on the full 290B corpus, using everything validated:
  - LONG DOCUMENTS (8192 tokens) in chunks of 1024 with the fast-weight state CARRIED across chunks
    (constant memory; this is where we are 1.4-1.6x faster and 6-8x lighter than a transformer)
  - INTRINSIC GROWTH: the model's own predictive-entropy saturation decides WHEN to grow (no schedule)
  - function-preserving growth on width and depth, MLP-aware
  - MASKED RECALL mix: loss only on the answer span, the proven lever that makes recall an intrinsic faculty
  - checkpoints + logs/train_pragnosia.log   (tail -f to watch)
Substrate pretraining uses backprop per amended VISION 0c; the model's LIFE (memory, consolidation) is local."""
import sys,os,time,math; sys.path.insert(0,"/opt/code/membrae")
from util import setup_log,pbar; setup_log("train_pragnosia")
import numpy as np, torch, torch.nn.functional as F
import torch._dynamo as _dynamo
_dynamo.config.recompile_limit=64          # 10 planned growth events legitimately change shapes; default 8 is too low
# INDUCTOR TILING BUG. Compiling the BACKWARD of the post-growth shape (D=640 NL=8 DIN=12, FE=157) raises
# `CantSplit` inside _split_iteration_ranges, and that killed run 4 TWICE -- each time immediately after a
# SUCCESSFUL growth. Swept the tiling options on the exact failing shape: defaults, max_tiles=1 and max_tiles=3
# all fail; prefer_nd_tiling=True compiles. Without this the run falls back to eager and loses ~2x throughput.
import torch._inductor.config as _ic; _ic.triton.prefer_nd_tiling=True
# Cap inductor's compile workers. The default spawns one per core; at 11+ layers that exhausted HOST RAM and the
# OOM killer took the process (exit 137) during a memory probe. 8 workers compile fine and leave prod alone.
_ic.compile_threads=8
import core as C, growth as G
from data_mix import DataMix
DEV=C.DEV
# CHUNK 4096, re-measured on the CURRENT architecture (exp/ctx_scale.py): 671 ktok/s vs 355 at CH=2048 --
# 1.9x for the same function, because 8 large kernel launches beat 16 small ones. It also DOUBLES the backprop
# horizon, since state is detached between chunks. Beyond 4096 it declines (8192: 602, 16384: 448) as the
# O(chunk^2) attention cost overtakes the launch saving.
DOC=32768; CH=2048; NCH=DOC//CH; B=int(os.environ.get("B","6")); DECAY=0.05
# CHUNK 2048, RE-MEASURED AT THIS CONFIG. 4096 was chosen when FE was 73; FE is now 273 (DIN 8 -> 16) and the
# intra-chunk term is O(T^2 x FE), so the memory tradeoff moved while the speed one did not:
#     chunk 4096 B6: 283 ktok/s, 19.31 GiB     chunk 2048 B6: 280 ktok/s,  9.48 GiB     1024 B6: 272, 5.38
# Identical speed for HALF the memory -- the GPU is no longer launch-bound at this size, so the larger chunk
# bought nothing. Costs half the backprop horizon (state is detached between chunks), which is the real tradeoff.
LR=float(os.environ.get("LR","6e-4")); STEPS=int(os.environ.get("STEPS","100000000"))   # effectively indefinite: run until told to stop (user)
WARM=int(os.environ.get("WARM","300"))            # re-warm after each grow: new capacity is inert and must be eased in
LR_MIN_MULT=float(os.environ.get("LR_MIN_MULT","0.1"))   # decay to 10% of peak by the end of a stage's token budget
D0,NL0,DIN0=768,11,16          # infant = 90.4M. Was 384x4L/DIN8 = 19M; the vision's claims are all either
                               # size-independent mechanisms or were already proven at 21.6M, so starting below
                               # 60M only spent time in a regime that proves nothing.
TARGET_D,TARGET_NL=1024,16                              # ~235M at the end
RECALL_FRAC=0.15; VRAM_MIN=8.0                          # leave headroom for prod
CKPT="/mnt/kv_cache/pragnosia_data/pragnosia.pt"
# GROWTH PLAN: THREE large events, not ten small ones. MEASURED (exp/growth_fix_final.py, identical final size
# D=768 and identical training): many small growths 3.9846 | few large growths 3.9076 | many+prune 3.9842.
# Fewer, larger events win by 0.077 nats AND recruit 52% more of the new capacity (+0.63 vs +0.41 load) with less
# incumbent entrenchment (2.87x vs 3.64x). Four other fixes were tried and FAILED: random-write init, net2net
# copy+split, a mirror init (write at matched RMS / read zero), and pruning the weakest dims at each growth.
# Cause: each growth scales the incumbent dims by c=sqrt(nD/D), so the penalty COMPOUNDS with the event count
# (1.633x cumulative over the old ten-event plan). Same infant seed, same 1024x16L target, a third of the events.
# 'wd' grows width and depth together so a single event is a real developmental step.
# Each event now also grows DIN, the RETRIEVAL width. DIN was fixed at 8 from the infant through all ten growths
# of the last run, so a D=1024 model was still routing every key-query association through the infant's 45 Taylor
# features -- a transformer at D=1024 has 1024 dimensions of attention. MEASURED (exp/din_scaling.py): raising
# DIN 8->32 is worth 0.133 nats at D=384 and 0.196 at D=768, i.e. the penalty GROWS with width, exactly the regime
# the 235M model plateaued in. DIN costs almost no parameters (Wk,Wq are D x DIN); it costs state and attention
# compute, and at DIN=24 the state is still only 1.3 MiB per layer, so constant-memory context is unaffected.
PLAN=[('wdi',(128,3,4)),('wdi',(128,2,4))]   # 768x11L/DIN16 90M -> 896x14L/20 142M -> 1024x16L/24 202M
CORPUS=os.environ.get("CORPUS","window2")                    # window2 only: educational text, 170B tokens, least repetitive
dm=DataMix(ctx=DOC+1,verbose=True,only=CORPUS)
RD=RV=None
_rp="/mnt/kv_cache/pragnosia_data/recall_32k.bin"; _ra="/mnt/kv_cache/pragnosia_data/recall_32k_ans.npy"
if os.path.exists(_rp) and os.path.exists(_ra):
    RD=np.memmap(_rp,dtype=np.uint16,mode="r"); RV=np.load(_ra)
    print(f"[masked-recall {RECALL_FRAC:.0%}] {len(RV)} answer-masked docs",flush=True)
def recall_batch(bs,L):
    """answer-masked recall: loss ONLY after 'Answer:' so the model must retrieve, not paraphrase."""
    X=np.zeros((bs,L),np.int64); Y=np.full((bs,L),-100,np.int64)
    for b in range(bs):
        s,e,ans=RV[np.random.randint(len(RV))]; n=min(int(e-s),L)
        X[b,:n]=RD[s:s+n]; a=max(1,int(ans-s))
        for t in range(a-1,n-1): Y[b,t]=int(RD[s+t+1])
    return torch.from_numpy(X).to(DEV),torch.from_numpy(Y).to(DEV)
def zero_state(m,bs):
    """explicit zero memory state: keeps the `states` argument a LIST on every call so torch.compile does not
    recompile (and eventually bail to eager) just because the type alternated None/list."""
    cfg=m.cfg
    return [(torch.zeros(bs,cfg.FE,cfg.DV,device=DEV),torch.zeros(bs,cfg.FE,device=DEV)) for _ in range(cfg.NL)]
def main():
    step=0
    # TIED EMBEDDINGS: one matrix for input and output. At vocab 32768 / D=1024 this saves 33.5M parameters (14%
    # of a 235M model) and MEASURES BETTER: -0.108 nats at 37% fewer params (exp/tied_emb.py). Growth keeps the tie
    # via cfg.head_scale and is loss-neutral in practice (+0.0001 across a growth event vs untied -0.0000).
    cfg=C.Config(D=D0,DIN=DIN0,DV=D0,NL=NL0,T=CH,res_scale=TARGET_NL**-0.5,tie_emb=True)
    # THE TWO MEASURED WINS. Gating replaces the hand-set DECAY with a LEARNED per-token forget rate: worth
    # -0.28 nats, it is what separates Mamba-2/GLA/RWKV-6 from plain linear attention, and it improved our own
    # memory gain 18x (+0.0016 -> +0.0286). qk-norm bounds the quartic growth of A=qf.kf: -0.017 nats AND faster.
    cfg.gated=True; cfg.qknorm=True
    m=C.Core(cfg).to(DEV); k=0; sd=None
    if os.path.exists(CKPT):
        sd=torch.load(CKPT,map_location=DEV)
        from util import cfg_from_ckpt
        cfg=cfg_from_ckpt(sd,T=CH,res_scale=TARGET_NL**-0.5)
        cfg.tie_emb=sd.get('tie_emb',False); cfg.head_scale=sd.get('head_scale',1.0)
        cfg.gated=sd.get('gated',False); cfg.qknorm=sd.get('qknorm',False)
        m=C.Core(cfg).to(DEV); m.load_state_dict(sd['model']); step=sd['step']; k=sd['k']
        print(f"RESUMED step {step}: D={cfg.D} NL={cfg.NL} ({sum(p.numel() for p in m.parameters())/1e6:.0f}M), grows done={k}",flush=True)
    opt=torch.optim.Adam(m.parameters(),lr=LR)
    # torch.compile: measured 1.58x on top of bf16 (kernel fusion -> less memory traffic; we are at the 400W cap,
    # so the gain comes from doing LESS work). `m` stays the raw module for growth/params; `fwd` is the compiled call.
    USE_COMPILE=os.environ.get("COMPILE","1")=="1"
    fwd=torch.compile(m) if USE_COMPILE else m
    TOK_PER_PARAM=float(os.environ.get("TOK_PER_PARAM","22"))     # Chinchilla-style floor per stage (user's request)
    sensor=G.SaturationSensor(window=400,t_stat=1.0,min_gap=2500)  # smoothed signal + real spacing
    # RESTORE THE GROWTH CLOCK. tokens_at_size drives BOTH the earned-growth gate (>= 22 tok/param) and the LR
    # schedule, but it was initialised to 0 and never checkpointed -- so every resume silently reset the model's
    # earned progress and pushed growth further away. Same class of bug as the gate starting from zero after a
    # growth, which the user caught earlier.
    tokens_at_size=int(sd.get('tokens_at_size',0)) if sd is not None else 0
    P=lambda: sum(p.numel() for p in m.parameters())
    print(f"PRAGNOSIA: infant D{cfg.D}x{cfg.NL} ({P()/1e6:.0f}M) -> target D{TARGET_D}x{TARGET_NL}; docs {DOC} chunk {CH} B{B}",flush=True)
    bar=pbar(range(step,STEPS),"pragnosia",total=STEPS-step)
    ema=None; t0=time.time(); last_grow_step=0; seen_tok=0
    steps_at_size=int(sd.get('steps_at_size',0)) if sd is not None else 0   # LR schedule position
    def set_lr(step_in_stage,frac_of_budget):
        """warmup after each grow, then cosine-decay against progress toward this stage's 22 tok/param budget."""
        if step_in_stage<WARM: lr=LR*(step_in_stage+1)/WARM
        else:
            f=min(1.0,max(0.0,frac_of_budget))
            lr=LR*(LR_MIN_MULT+(1-LR_MIN_MULT)*0.5*(1+math.cos(math.pi*f)))
        for g in opt.param_groups: g['lr']=lr
        return lr
    for it in bar:
        if RD is not None and np.random.rand()<RECALL_FRAC:          # masked recall (no carry: short QA docs)
            # MASKED RECALL, now THROUGH MEMORY: half the batches put the fact in the carried state (so retrieval
            # must come from memory), half keep it in-window. Previously all of them used a ZERO state, so the
            # persistent memory was never trained to answer a query -- the memory worked but was never asked to.
            X,Y=recall_batch(B,CH)
            opt.zero_grad()
            if np.random.rand()<0.5:
                half=CH//2
                Xa=torch.cat([X[:,:half],torch.zeros_like(X[:,:half])],1)      # context first, through memory
                with torch.autocast("cuda",dtype=torch.bfloat16):
                    _,st_r=fwd(Xa,states=zero_state(m,B),return_states=True)
                st_r=[(a.detach(),b.detach()) for a,b in st_r]
                with torch.autocast("cuda",dtype=torch.bfloat16):
                    hs=fwd(X,states=st_r,return_hidden=True)                    # answer supervised WITH memory live
                    loss=C.ce_from_hidden(m,hs,Y)
            else:
                with torch.autocast("cuda",dtype=torch.bfloat16):
                    hs=fwd(X,states=zero_state(m,B),return_hidden=True)
                    loss=C.ce_from_hidden(m,hs,Y)
            loss.backward(); opt.step(); ent=None
        else:                                                        # long document with memory carried across chunks
            Xd=torch.tensor(dm.batch(B),device=DEV); st=zero_state(m,B); pv=None; opt.zero_grad(); last=None
            for i in range(NCH):
                xi=Xd[:,i*CH:(i+1)*CH]; yi=Xd[:,i*CH+1:(i+1)*CH+1]
                with torch.autocast("cuda",dtype=torch.bfloat16):        # 2.82x faster, verified same quality
                    # `prev` carries the last normed activation across the boundary. Without it _shift fed a ZERO
                    # into the first key of every chunk -- 16 corrupted positions per document, and the chunked
                    # recurrence was 8.1e-02 WRONG in fp64 (now exact to 1e-15).
                    # return_hidden + ce_from_hidden: the [B,T,32768] logits tensor is never built. At B=8/CH=4096
                    # it is 2 GiB in bf16 with autograd holding it, and the old `lg.detach().float()` below added
                    # 4 GiB more. Verified EXACT in value and identical in gradients (1.09e-13 worst relative).
                    hs,st,pv=fwd(xi,states=st,return_states=True,prev=pv,return_prev=True,return_hidden=True)
                    l=C.ce_from_hidden(m,hs,yi)/NCH
                # DECAY IS NOW LEARNED. With gating the state already decays per token inside the layer
                # (M_t = g_t*M_{t-1} + k_t (x) v_t); applying the hand-set DECAY on top would decay it TWICE.
                # The gate is the "selective memory" the vision describes -- the model chooses, not a constant.
                if not cfg.gated: st=m.decay_states(st,DECAY)
                st=[(a.detach(),b.detach()) for a,b in st]; pv=[p.detach() for p in pv]
                l.backward(); last=hs.detach()                       # [B,T,D] ~50 MiB, not [B,T,vocab] at 4 GiB
            opt.step()
            with torch.no_grad(), torch.autocast("cuda",dtype=torch.bfloat16):
                loss=C.ce_from_hidden(m,last,Xd[:,(NCH-1)*CH+1:NCH*CH+1])
                ent=C.entropy_from_hidden(m,last)                    # projects only the 4096 rows it samples
        ema=loss.item() if ema is None else 0.995*ema+0.005*loss.item()
        free=torch.cuda.mem_get_info()[0]/2**30
        tokens_at_size+=B*(CH if ent is None else DOC); steps_at_size+=1; seen_tok+=B*(CH if ent is None else DOC)
        cur_lr=set_lr(steps_at_size,tokens_at_size/(TOK_PER_PARAM*P()))
        # a grow needs BOTH: the model's own saturation signal AND enough tokens spent at this size (22/param).
        earned = tokens_at_size >= TOK_PER_PARAM*P()
        if ent is not None and k<len(PLAN) and free>VRAM_MIN and earned and sensor.observe(ent,it):
            kind,amt=PLAN[k]; P_before=P()
            # TRUE PRE-GROWTH SNAPSHOT, taken at the growth instant so before/after is exactly measurable.
            torch.save({'model':m.state_dict(),'D':m.cfg.D,'DV':m.cfg.DV,'DIN':m.cfg.DIN,'NL':m.cfg.NL,
                        'step':it,'k':k,'tie_emb':m.cfg.tie_emb,'head_scale':m.cfg.head_scale,
                        'gated':m.cfg.gated,'qknorm':m.cfg.qknorm,
                'tokens_at_size':tokens_at_size,'steps_at_size':steps_at_size}, f"{CKPT[:-3]}_before_grow{k+1}.pt")
            if kind=='wdi':                                                 # width + depth + RETRIEVAL together
                G.grow_width(m,amt[0]); G.grow_depth(m,amt[1]); G.grow_din(m,amt[2])
            elif kind=='wd': G.grow_width(m,amt[0]); G.grow_depth(m,amt[1])
            elif kind=='w': G.grow_width(m,amt)
            else: G.grow_depth(m,amt)
            earned_tpp=tokens_at_size/max(1,P_before)            # capture BEFORE resetting (my first edit printed 0)
            opt=torch.optim.Adam(m.parameters(),lr=LR); k+=1; tokens_at_size=0; steps_at_size=0; last_grow_step=it
            # Return the OLD shape's cached blocks to the driver. At growth the process briefly holds the old
            # params + old Adam state + old compiled graph alongside the new ones, and the allocator keeps every
            # freed block: measured 34.5 GB reserved where only ~18.9 GB is live. Harmless (PyTorch frees its
            # cache before it would ever OOM) but it eats the headroom the NEXT growth needs to compile in.
            torch.cuda.empty_cache()
            # RECOMPILE, BUT DO NOT DIE IF THE COMPILER CANNOT HANDLE THE NEW SHAPE. Growth changes FE from
            # DIN^2+DIN+1 to a new value, and inductor hit an internal tiling bug on FE=157
            # (CantSplit.__init__() missing 2 required positional arguments) which killed run 4 right after a
            # SUCCESSFUL growth. Eager is ~2x slower but correct, and a lost run costs far more than that.
            if USE_COMPILE:
                # Warm up at the REAL batch AND with a carried state -- inductor compiles per input SHAPE, so a
                # B=1 stateless probe compiles a different variant and the real one still explodes later. That is
                # exactly how the previous guard failed: it passed, then the training step died on the same
                # CantSplit tiling bug. Both call signatures the loop uses are exercised here.
                try:
                    fwd=torch.compile(m)
                    # Exercise EVERY call signature the loop uses -- inductor compiles per signature+shape, so a
                    # warm-up that misses one lets the real call explode later. That is how the last two guards
                    # failed: first the batch differed (B=1 vs B=8), then return_hidden differed.
                    _z=zero_state(m,B); _x=torch.zeros(B,CH,dtype=torch.long,device=DEV)
                    with torch.autocast("cuda",dtype=torch.bfloat16):
                        _h,_s,_p=fwd(_x,states=_z,return_states=True,prev=None,return_prev=True,return_hidden=True)
                        _l=C.ce_from_hidden(m,_h,_x)
                    _l.backward()          # THE BACKWARD IS WHERE CantSplit FIRES -- forward alone compiles fine
                    _s=[(a.detach(),b.detach()) for a,b in _s]; _p=[q.detach() for q in _p]
                    with torch.autocast("cuda",dtype=torch.bfloat16):
                        _h2,_,_=fwd(_x,states=_s,return_states=True,prev=_p,return_prev=True,return_hidden=True)
                        _l2=C.ce_from_hidden(m,_h2,_x)
                    _l2.backward()                                            # the carried-state signature
                    with torch.autocast("cuda",dtype=torch.bfloat16):
                        C.ce_from_hidden(m,fwd(_x,states=_z,return_hidden=True),_x).backward()   # recall branch
                    m.zero_grad(set_to_none=True); opt.zero_grad(set_to_none=True)
                    del _z,_x,_h,_s,_p,_l,_h2,_l2
                except Exception as e:
                    bar.write(f"    compile failed after growth ({type(e).__name__}) -- continuing EAGER")
                    fwd=m
            else: fwd=m
            bar.write(f"    >>> SELF-GREW #{k} at step {it}: D={m.cfg.D}x{m.cfg.NL}L mlp{m.layers[0].m1.weight.shape[0]} -> {P()/1e6:.0f}M DIN{m.cfg.DIN} (earned {earned_tpp:.0f} tok/param at the old size, free {free:.0f}G)")
        if (it+1)%100==0:
            # COUNT WHAT WE ACTUALLY PROCESSED. This used B*DOC for every iteration, but RECALL_FRAC of them are
            # short recall batches of B*CH -- at 15% that OVERSTATED throughput by 1.164x (a reported 1022 ktok/s
            # was really 879). `seen_tok` accumulates the true count at the point each batch is built.
            tps=seen_tok/(time.time()-t0)
            bar.set_postfix_str(f"loss={ema:.3f} ppl={math.exp(min(20,ema)):.0f} D={m.cfg.D}x{m.cfg.NL}L {P()/1e6:.0f}M {tokens_at_size/P():.0f}tok/param lr={cur_lr:.1e} {tps/1e3:.0f}ktok/s free={free:.0f}G")
        if (it+1)%200==0: torch.cuda.empty_cache()   # bound the reservation, not just the live peak
        if (it+1)%500==0:
            ck={'model':m.state_dict(),'D':m.cfg.D,'DV':m.cfg.DV,'DIN':m.cfg.DIN,'NL':m.cfg.NL,'step':it+1,'k':k,
                'tie_emb':m.cfg.tie_emb,'head_scale':m.cfg.head_scale,'gated':m.cfg.gated,'qknorm':m.cfg.qknorm,
                'tokens_at_size':tokens_at_size,'steps_at_size':steps_at_size}
            torch.save(ck,CKPT)
            # keep the LAST save before each grow and the TWO after it, so growth can be probed after the fact
            if last_grow_step and (it+1)-last_grow_step<=1000: torch.save(ck,f"{CKPT[:-3]}_probe{it+1}.pt")
            # (the pre-growth snapshot is taken AT the growth event instead -- see below. Saving it every 500
            #  steps only duplicated the latest checkpoint and gave no true BEFORE to compare against.)
    torch.save({'model':m.state_dict(),'D':m.cfg.D,'DV':m.cfg.DV,'NL':m.cfg.NL,'step':STEPS,'k':k},CKPT)
if __name__=="__main__": main()
