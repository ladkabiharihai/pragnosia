"""WHAT THE MODELS ACTUALLY SAY (2026-10-08; user: "show me the generation quality on facts, arithmetic and other things").
Every model gets the same "User: <question>\\nAssistant:" template and answers greedily (eval/qa.py's protocol), first
sentence shown; GPT-2 and SmolLM2 are base models, ours is fine-tuned on factual Q&A (the fine-tune of the final pipeline;
the 382k one is still running, so the 375k pipeline's). Then each model continues the same opening as free text, and one
reading example: three invented facts, 2,000 tokens of unrelated prose, a question.
usage: eval/showcase.py <our fine-tuned ckpt> <our base ckpt>"""
import os, sys, re, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, "/opt/code/membrae")
os.environ.setdefault("HF_HOME", "/mnt/kv_cache/hf_home")
from src.common import cap_gpu
cap_gpu(7.0)
import numpy as np, torch
import src.bench as SB
from src.bench import Ours
from transformers import AutoModelForCausalLM, AutoTokenizer

CATS = {
    "FACTS": ["What is the capital of Japan?", "What is the capital of Australia?", "Who wrote Hamlet?",
              "What is the chemical symbol for gold?", "Who painted the Mona Lisa?", "What is the largest ocean on Earth?",
              "How many continents are there?", "At what temperature does water boil at sea level?",
              "Who was the first person to walk on the Moon?", "What gas do plants absorb from the air?"],
    "ARITHMETIC": ["What is 2 plus 2?", "What is 7 plus 8?", "What is 15 plus 27?", "What is 9 times 6?",
                   "What is 12 times 12?", "What is 100 minus 37?", "What is 144 divided by 12?",
                   "Tom has 3 apples and buys 5 more. How many apples does he have?"],
    "REASONING": ["All cats are animals. Tom is a cat. Is Tom an animal?",
                  "What comes next in the sequence 2, 4, 6, 8?",
                  "If today is Monday, what day is tomorrow?",
                  "Which is heavier, a kilogram of iron or a kilogram of feathers?",
                  "Anna is taller than Ben, and Ben is taller than Carl. Who is the tallest?",
                  "If it is raining, the ground gets wet. The ground is dry. Is it raining?"],
    "EXPLAIN": ["Why is the sky blue?", "What is photosynthesis?", "What causes the seasons on Earth?",
                "How does a vaccine work?"],
}

class HF:
    def __init__(self, name, lab):
        self.tok = AutoTokenizer.from_pretrained(name, cache_dir="/mnt/kv_cache/hf_cache")
        self.m = AutoModelForCausalLM.from_pretrained(name, cache_dir="/mnt/kv_cache/hf_cache", torch_dtype=torch.bfloat16).cuda().eval()
        self.lab, self.eos = lab, self.tok.eos_token_id
        self.max_ctx = getattr(self.m.config, "max_position_embeddings", 1024)
    @torch.no_grad()
    def gen(self, text, n, temp=0.0, seed=0):
        ids = self.tok.encode(text, add_special_tokens=False)[-(self.max_ctx - n):]; x = torch.tensor([ids], device="cuda")
        torch.manual_seed(seed)
        kw = dict(do_sample=True, temperature=temp, top_k=0) if temp > 0 else dict(do_sample=False)
        o = self.m.generate(x, max_new_tokens=n, pad_token_id=self.eos, eos_token_id=self.eos, **kw)
        return self.tok.decode(o[0][len(ids):], skip_special_tokens=True)

class Mine:
    def __init__(self, ck, lab):
        self.o = Ours(ck, "/mnt/kv_cache/tok_v8192.json"); self.lab = lab
        import src.assoc as A; A.FUSED_SCAN = True
        for mx in self.o.m.mix:
            if hasattr(mx, "need_sent"): mx.need_sent = False
    @torch.no_grad()
    def gen(self, text, n, temp=0.0, seed=0):
        out = self.o.generate(self.o.encode(text), n, temp=temp, seed=seed, stop=8191)
        if 8191 in out: out = out[:out.index(8191)]
        return self.o.decode(out)

def first(t):
    t = t.split("\nUser:")[0].strip().split("\n")[0]
    m = re.match(r"(.+?[.!?])(\s|$)", t)
    return (m.group(1) if m else t)[:160]

ft, base = sys.argv[1], sys.argv[2]
M = [Mine(ft, "ours (fine-tuned)")] + ([] if os.environ.get("SHOW_HF") == "0" else   # SHOW_HF=0: ours only (2026-10-12)
     [HF("gpt2", "GPT-2 124M"), HF("HuggingFaceTB/SmolLM2-135M", "SmolLM2-135M")])
for cat, qs in CATS.items():
    print(f"\n==================== {cat} ====================", flush=True)
    for q in qs:
        print(f"\nQ: {q}")
        for m in M: print(f"   {m.lab:<19s} {first(m.gen(f'User: {q}' + chr(10) + 'Assistant:', 48))!r}", flush=True)

print("\n==================== FREE TEXT (the same opening, 120 tokens, sampled) ====================", flush=True)
B = Mine(base, "ours (base)")
SB.DRY_MULT = 0.4
for p in ["The history of the city begins", "Photosynthesis is the process by which"]:
    print(f"\nPROMPT: {p!r}")
    print(f"   ours (base), temp 0.85, DRY 0.4:\n      {(p + B.gen(p, 120, temp=0.85, seed=1)).strip()!r}", flush=True)
    SB.DRY_MULT = 0.0
    for m in M[1:]: print(f"   {m.lab}, temp 0.9:\n      {(p + m.gen(p, 120, temp=0.9, seed=1)).strip()!r}", flush=True)
    SB.DRY_MULT = 0.4

print("\n==================== READING (3 invented facts, 2,000 tokens of unrelated Wikipedia, a question) ====================", flush=True)
wiki = np.memmap("/mnt/kv_cache/tokfull_v8192/wikipedia.bin", dtype=np.uint16, mode="r")
f = np.asarray(wiki[7_000_000:7_006_000]); f = f[f != 8191][:2000].tolist()
filler = B.o.decode(f)
facts = "The capital of Velmora is Drathis. The Oskel engine was invented by Mirta Quell. The code for the Zarn vault is 4719."
for q, want in [("What is the capital of Velmora?", "Drathis"), ("Who invented the Oskel engine?", "Mirta Quell"),
                ("What is the code for the Zarn vault?", "4719")]:
    text = facts + "\n\n" + filler + "\n\nQuestion: " + q + " Answer:"
    print(f"\nQ: {q}   (right answer: {want})")
    for m in [B] + M[1:]:
        print(f"   {m.lab:<19s} {first(m.gen(text, 12))!r}", flush=True)
if len(M) > 1: print("\n(GPT-2 sees only its last 1,024 tokens here, so the facts are outside its view; SmolLM2 and ours see everything.)")

# KINDS OF FACT IT NEVER SAW (2026-10-12; SHOW_LOOK=<ckpt>, the lookup fine-tune over many kinds): 10 invented facts of one kind
# no training data or round ever wrote, a question about one of them, greedy; the base beside the lookup-tuned model
if os.environ.get("SHOW_LOOK"):
    import random
    L_ = Mine(os.environ["SHOW_LOOK"], "ours (lookup-tuned)"); r_ = random.Random(11)
    SY_ = ["vel", "dra", "osk", "arth", "mir", "tan", "quel", "lin", "zor", "va", "bre", "nth", "kal", "iso", "rum"]
    nm_ = lambda n: "".join(r_.choice(SY_) for _ in range(n)).capitalize()
    print("\n==================== KINDS OF FACT IT NEVER SAW (10 invented facts read, then a question; greedy) ====================", flush=True)
    for f_, q_, two in [("The {X} river flows into the {Y} Sea.", "Into which sea does the {X} river flow?", False),
                        ("{Y} leads the {X} guild.", "Who is the leader of the {X} guild?", True),
                        ("The {X} bridge was built by {Y}.", "Who built the {X} bridge?", True),
                        ("The favourite fruit of {X} is {Y}.", "What is the favourite fruit of {X}?", False)]:
        for t_ in range(2):
            facts_ = [(nm_(3), f"{nm_(2)} {nm_(2)}" if two else nm_(2)) for _ in range(10)]
            x_, y_ = r_.choice(facts_)
            text = " ".join(f_.format(X=a, Y=b) for a, b in facts_) + "\n\nQuestion: " + q_.format(X=x_) + " Answer:"
            print(f"\nQ: {q_.format(X=x_)}   (right answer: {y_})")
            for m in [B, L_]: print(f"   {m.lab:<21s} {first(m.gen(text, 12))!r}", flush=True)
    print("\n" + text.split("\n\n")[0][:400] + " ...   (the last context, as the models read it)")
