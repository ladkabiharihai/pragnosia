"""HELD-OUT BITS PER BYTE OF CHECKPOINTS, ON IDENTICAL TEXT (2026-10-08). The trainer's eval halves its batch and retries when it
runs short of memory, and the retry draws from a sampler the failed attempt already advanced: the long-span run (30 GiB of
graphs) scored 975k bytes of DIFFERENT text from the 952k every earlier run scored, so its 1.0558 compared with nothing.
This is the trainer's own measurement (src/common.eval_bpb: 8 x 4 x 8,192 tokens of the five training sources, sampler seed
1234, the checkpoint's averaged weights), run alone so it never retries. usage: eval/bpb_ckpt.py <ckpt> ..."""
import sys, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, "/opt/code/membrae")
from src.common import cap_gpu, eval_bpb
cap_gpu(7.0)
import numpy as np, torch
from tokenizers import Tokenizer
from src.model import LM
from src.train import loaders
import src.assoc as A; A.FUSED_SCAN = True
tok = Tokenizer.from_file("/mnt/kv_cache/tok_v8192.json")
import os
# PROBE_SOURCES=wikipedia (one source, 2026-10-09): which text a mix change helped or hurt; default the five training sources
bins, ks, ps = loaders("/mnt/kv_cache/tokfull_v8192", only=os.environ.get("PROBE_SOURCES", "dclm,fineweb-edu,wikipedia,finemath,arxiv").split(","))
for ck in sys.argv[1:]:
    sd = torch.load(ck, map_location="cpu", weights_only=False)
    m = LM(**sd["cfg"]); m.load_state_dict(sd["model"]); m = m.cuda().eval()
    # ONE ROW AT A TIME (2026-10-09): the rows are drawn one by one from the sampler, so 32 batches of 1 are the same 32
    # rows as 8 of 4 -- and a 168M model's batch of 4 ran out of memory under the 6 GiB side-job cap, whose retry at 2 then
    # scored different text (974k bytes)
    b, nb = eval_bpb(m, tok, bins, ks, ps, np.random.default_rng(1234), n_batches=32, B=1, T=8192, vocab=8192)
    print(f"{ck}: step {sd['step']}  BPB {b:.4f}  ({nb / 1e3:.0f}k bytes)", flush=True)
    del m, sd; torch.cuda.empty_cache()
