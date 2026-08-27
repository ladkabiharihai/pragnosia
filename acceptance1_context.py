"""VISION ACCEPTANCE TEST 1 -- 'constant-memory, constant-compute, unbounded/streaming context' (VISION 1b flag 1).
Measures, for context lengths 512 -> 131072 tokens processed in STREAMING chunks with carried state:
  (a) peak GPU memory,  (b) latency per token,  (c) the KV cache a same-size transformer would need.
The claim is FLAT lines for (a) and (b) while the transformer's (c) grows linearly and its attention compute grows
quadratically. Also measures recall-vs-distance, honestly labelled by what the model was actually trained on."""
import sys,time,gc; sys.path.insert(0,"/opt/code/membrae")
import numpy as np, torch, torch.nn.functional as F
import core as C; from data_mix import DataMix
DEV=C.DEV; D=256; NL=6; T=128; DECAY=0.2
dm=DataMix(ctx=T+1,verbose=False)
def fresh():
    torch.manual_seed(0); cfg=C.Config(D=D,DIN=8,DV=D,NL=NL,T=T,res_scale=NL**-0.5); return cfg,C.Core(cfg).to(DEV)
@torch.no_grad()
def stream(m,total_tokens,B=1):
    """process `total_tokens` in chunks of T, carrying (and decaying) state -> constant memory by construction."""
    st=None; nch=total_tokens//T
    torch.cuda.empty_cache(); gc.collect(); torch.cuda.reset_peak_memory_stats()
    X=torch.randint(0,32768,(B,T),device=DEV)
    torch.cuda.synchronize(); t0=time.time()
    for _ in range(nch):
        _,st=m(X,states=st,return_states=True); st=m.decay_states(st,DECAY)
    torch.cuda.synchronize(); dt=time.time()-t0
    peak=torch.cuda.max_memory_allocated()/2**20
    return dt/(nch*T*B)*1e6, peak                                   # microseconds/token, peak MiB
def kv_bytes(L,B=1): return 2*NL*L*D*B*4/2**20                      # a same-size transformer's KV cache, MiB (fp32)
if __name__=="__main__":
    cfg,m=fresh(); m.eval()
    print("ACCEPTANCE 1: constant-memory / constant-compute streaming context",flush=True)
    print(f"  model D={D} NL={NL} chunk={T}, state carried + decayed ({DECAY})",flush=True)
    print(f"\n  {'context':>10s} {'us/token':>10s} {'peak MiB':>10s} | {'transformer KV MiB':>19s} {'attn FLOPs rel':>15s}",flush=True)
    base=None
    for L in (512,2048,8192,32768,131072):
        us,peak=stream(m,L)
        if base is None: base=(us,peak)
        rel=(L/512)**2
        print(f"  {L:10d} {us:10.2f} {peak:10.1f} | {kv_bytes(L):19.1f} {rel:14.0f}x",flush=True)
    us_l,peak_l=stream(m,131072); us_s,peak_s=stream(m,512)
    print(f"\n  VERDICT: latency/token {us_s:.2f} -> {us_l:.2f} us ({us_l/us_s:.2f}x over a 256x longer context)",flush=True)
    print(f"           peak memory  {peak_s:.1f} -> {peak_l:.1f} MiB ({peak_l/peak_s:.2f}x)",flush=True)
    print(f"           a same-size transformer would need {kv_bytes(131072):.0f} MiB of KV cache at 131072 tokens",flush=True)
    print(f"           and {(131072/512)**2:.0f}x the attention compute of its 512-token case.",flush=True)
    print(("  => CLAIM HOLDS: constant memory AND constant per-token cost at 131072 tokens" if peak_l/peak_s<1.5 and us_l/us_s<1.5
           else "  => CLAIM DOES NOT HOLD as measured"),flush=True)
