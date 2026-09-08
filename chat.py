"""CHAT WITH PRAGNOSIA.

    python chat.py                          # the finished 125M
    python chat.py --ckpt <path>            # any checkpoint, e.g. the 400M as it trains
    python chat.py --raw                    # raw continuation instead of chat framing

WHAT TO EXPECT, honestly. This is a BASE model, not an instruction-tuned assistant. It was never trained to
follow instructions, so it continues text rather than answering questions. It uses <user>/<assistant> framing
because that markup IS in its training data (window2, hermes, tulu3), which is why that format works better here
than a bare question -- but it is imitating the shape of a conversation, not having one. GPT-2 at 124M and even
at 774M behaves the same way. Judge it on continuation, not on answers.

STATE IS NOT CARRIED BETWEEN TURNS BY DEFAULT, and that default was earned by measurement. Carrying it locks the
model into whatever it said first -- one Java snippet in turn 2 and every later answer is Java, whatever you ask.
Measured across six turns: no-carry answers each question independently; half-carry degrades into quoted
fragments; three-quarter carry reproduces the code loop. The model was never trained on multi-turn dialogue, so
carrying its own output across turns just accumulates the self-write attractor.
`/carry` turns it on anyway if you want to see that failure. The place the fixed state genuinely shines is a long
DOCUMENT, not a chat transcript -- use `/doc <file>` to stream one in and then ask about it.

  /reset     start a new conversation (clears the state)
  /doc PATH  stream a long file into the state, then ask about it -- this is the real long-context demo
  /carry     toggle carrying state between turns (default OFF; ON reproduces the code-loop failure)
  /state     show state size vs what a transformer's KV cache would cost here
  /temp 0.8  set temperature      /raw   toggle chat framing
  /quit
"""
import sys, os, time, argparse
sys.path.insert(0,"/opt/code/membrae")
from util import cfg_from_ckpt
import torch, core as C
from tokenizers import Tokenizer
DEV=C.DEV; TOK=Tokenizer.from_file("/opt/code/membrae/data/bpe32k.json")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--ckpt",default="/mnt/kv_cache/pragnosia_data/p124f.pt")
    ap.add_argument("--temp",type=float,default=0.8)
    ap.add_argument("--top_p",type=float,default=0.92)
    ap.add_argument("--n",type=int,default=120)
    ap.add_argument("--raw",action="store_true")
    ap.add_argument("--carry",action="store_true",help="carry state between turns (measured WORSE; see docstring)")
    a=ap.parse_args()
    sd=torch.load(a.ckpt,map_location=DEV)
    cfg=cfg_from_ckpt(sd,T=2048,res_scale=sd['res_scale'])
    m=C.Core(cfg).to(DEV); m.load_state_dict(sd['model']); m.eval()
    P=sum(p.numel() for p in m.parameters())
    state_mib=cfg.FE*cfg.DV*4*cfg.NL/2**20
    print(f"PRAGNOSIA {P/1e6:.0f}M  |  D{cfg.D}x{cfg.NL}L DIN{cfg.DIN}  |  {sd['step']*8*2048/1e9:.2f}B tokens")
    print(f"fixed state: {state_mib:.0f} MiB, constant regardless of how long this conversation gets")
    print(f"base model -- it continues text, it does not follow instructions. /quit to exit, /reset to clear.\n")
    st=None; pv=None; turns=0; ntok=0
    while True:
        try: u=input("\033[1myou:\033[0m ").strip()
        except (EOFError,KeyboardInterrupt): print(); break
        if not u: continue
        if u=="/quit": break
        if u=="/reset": st=None; pv=None; turns=0; ntok=0; print("  [state cleared]\n"); continue
        if u=="/state":
            kv=2*cfg.D*cfg.NL*ntok*4/2**20
            print(f"  ours: {state_mib:.0f} MiB fixed  |  a transformer's KV cache at {ntok:,} tokens: "
                  f"{kv:.1f} MiB and growing\n"); continue
        if u.startswith("/temp"): a.temp=float(u.split()[1]); print(f"  [temp={a.temp}]\n"); continue
        if u=="/raw": a.raw=not a.raw; print(f"  [raw={a.raw}]\n"); continue
        if u=="/carry":
            a.carry=not a.carry; st=None; pv=None
            print(f"  [carry={a.carry}] state cleared. Carry is measured WORSE for chat -- it locks the model "
                  f"into its first answer.\n"); continue
        if u.startswith("/doc"):
            path=u.split(maxsplit=1)[1] if len(u.split())>1 else ""
            try: txt=open(path).read()
            except Exception as e: print(f"  [cannot read {path}: {e}]\n"); continue
            dids=TOK.encode(txt).ids
            t0=time.time(); st=None; pv=None
            with torch.no_grad():
                st,pv=m.prefill(torch.tensor([dids],device=DEV)) if len(dids)<=2048 else (None,None)
                if st is None:
                    st=[(torch.zeros(1,cfg.FE,cfg.DV,device=DEV),torch.zeros(1,cfg.FE,device=DEV))
                        for _ in range(cfg.NL)]
                    for i in range(0,len(dids),2048):
                        _,st,pv=m(torch.tensor([dids[i:i+2048]],device=DEV),states=st,
                                  return_states=True,prev=pv,return_prev=True)
            ntok+=len(dids)
            print(f"  [read {len(dids):,} tokens in {time.time()-t0:.1f}s -- state still {state_mib:.0f} MiB, "
                  f"a transformer's KV cache would be {2*cfg.D*cfg.NL*len(dids)*4/2**20:.0f} MiB]\n"); continue
        p = u if a.raw else f"<user> {u}\n<assistant>"
        ids=TOK.encode(p).ids; ntok+=len(ids)
        t0=time.time()
        with torch.no_grad():
            out,ns=m.generate(torch.tensor([ids],device=DEV),n=a.n,temp=a.temp,top_p=a.top_p,states=st)
        st = ns if a.carry else None      # default OFF: carrying its own output across turns is measured worse
        txt=TOK.decode(out)
        if not a.raw:
            for stop in ("<user>","\n<","<|"):
                if stop in txt: txt=txt.split(stop)[0]
        ntok+=len(out); turns+=1
        dt=time.time()-t0
        print(f"\033[36mpragnosia:\033[0m {txt.strip()}")
        print(f"  \033[2m[{len(out)} tok, {1000*dt/max(1,len(out)):.0f} ms/tok, turn {turns}, "
              f"{ntok:,} tokens of history, state still {state_mib:.0f} MiB]\033[0m\n")

if __name__=="__main__": main()
