"""ACCEPTANCE 4 -- ANTI-POISONING: does verification let TRUE facts through, REJECT false ones, and HOLD novel ones?"""
import sys,time; sys.path.insert(0,"/opt/code/membrae")
import verify as VF
CLAIMS=[("TRUE ","The capital of France is"," Paris"),
        ("TRUE ","The largest planet in the solar system is"," Jupiter"),
        ("FALSE","The capital of France is"," Berlin"),
        ("FALSE","The largest planet in the solar system is"," Mercury"),
        ("NOVEL","The tallest tower in Vashmere is called"," Kelmoor")]
EXPECT={"TRUE ":"ACCEPT","FALSE":"REJECT","NOVEL":"HOLD"}
if __name__=="__main__":
    print("ACCEPTANCE 4: verification gate v2 (evidence retrieval from the trusted corpus)",flush=True)
    ok=0
    for lab,ctx,ans in CLAIMS:
        t0=time.time(); v,ev=VF.verify(ctx,ans); good=(v==EXPECT[lab]); ok+=good
        top=", ".join(f"{w!r}:{p:.2f}" for w,_,p in ev['top'][:3])
        print(f"  {lab} {ans!r:11s} -> {v:7s} {'OK ' if good else 'BAD'} | evidence n={ev['n']:5d} claim-share={ev['claim_share']:.3f} rank={ev['claim_rank']} | corpus says: {top}  ({time.time()-t0:.1f}s)",flush=True)
    print(f"\n  {ok}/{len(CLAIMS)} correct  -> {'ANTI-POISONING WORKS' if ok==len(CLAIMS) else 'needs tuning'}",flush=True)
