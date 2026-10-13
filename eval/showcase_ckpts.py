"""THE SAME QUESTIONS AS eval/showcase.py, ON BASE CHECKPOINTS, SCORED (2026-10-09). Greedy, "User: <q>\\nAssistant:", the first
line; right = the expected answer appears in it. usage: eval/showcase_ckpts.py <ckpt>=<label> ..."""
import os, sys, re, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, "/opt/code/membrae")
from src.common import cap_gpu
cap_gpu(6.0)
import torch
from src.bench import Ours
import src.assoc as A
Q = {
 "FACTS": [("What is the capital of Japan?", ["tokyo"]), ("What is the capital of Australia?", ["canberra"]),
           ("Who wrote Hamlet?", ["shakespeare"]), ("What is the chemical symbol for gold?", [" au", "au."]),
           ("Who painted the Mona Lisa?", ["leonardo", "da vinci"]), ("What is the largest ocean on Earth?", ["pacific"]),
           ("How many continents are there?", ["seven", "7"]), ("At what temperature does water boil at sea level?", ["100"]),
           ("Who was the first person to walk on the Moon?", ["armstrong"]), ("What gas do plants absorb from the air?", ["carbon dioxide", "co2"])],
 "ARITHMETIC": [("What is 2 plus 2?", ["4"]), ("What is 7 plus 8?", ["15"]), ("What is 15 plus 27?", ["42"]), ("What is 9 times 6?", ["54"]),
                ("What is 12 times 12?", ["144"]), ("What is 100 minus 37?", ["63"]), ("What is 144 divided by 12?", ["12"]),
                ("Tom has 3 apples and buys 5 more. How many apples does he have?", ["8"])],
 "REASONING": [("All cats are animals. Tom is a cat. Is Tom an animal?", ["yes"]), ("What comes next in the sequence 2, 4, 6, 8?", ["10"]),
               ("If today is Monday, what day is tomorrow?", ["tuesday"]), ("Which is heavier, a kilogram of iron or a kilogram of feathers?", ["same", "neither", "equal", "both"]),
               ("Anna is taller than Ben, and Ben is taller than Carl. Who is the tallest?", ["anna"]),
               ("If it is raining, the ground gets wet. The ground is dry. Is it raining?", ["no"])],
}
def first(t):
    t = t.split("\nUser:")[0].strip().split("\n")[0]; m = re.match(r"(.+?[.!?])(\s|$)", t)
    return (m.group(1) if m else t)[:140]
def hit(ans, keys, q):
    a = " " + ans.lower()
    for k in keys:
        if k.strip().isdigit():
            if re.search(r"(?<![0-9])" + k.strip() + r"(?![0-9])", a.replace(",", "")): return True
        elif k in a: return True
    return False
for spec in sys.argv[1:]:
    ck, lab = spec.split("=")
    o = Ours(ck, "/mnt/kv_cache/tok_v8192.json"); A.FUSED_SCAN = True
    for mx in o.m.mix:
        if hasattr(mx, "need_sent"): mx.need_sent = False
    print(f"\n######## {lab} ({ck}, {sum(p.numel() for p in o.m.parameters()) / 1e6:.0f}M)", flush=True)
    tot = {}
    for cat, qs in Q.items():
        ok = 0
        for q, keys in qs:
            out = o.generate(o.encode(f"User: {q}\nAssistant:"), 40, temp=0.0, seed=0, stop=8191)
            if 8191 in out: out = out[:out.index(8191)]
            a = first(o.decode(out)); h = hit(a, keys, q); ok += h
            print(f"  [{'+' if h else ' '}] {q[:60]:<60s} -> {a!r}", flush=True)
        tot[cat] = (ok, len(qs))
    print("  SCORE " + "  ".join(f"{c} {k}/{n}" for c, (k, n) in tot.items()), flush=True)
    del o; torch.cuda.empty_cache()
