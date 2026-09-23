"""QUESTION ANSWERING, the way an investor will test it: type a question, read the answer, check it.

Every model gets the same template -- "User: <question>\\nAssistant:" -- and answers by GREEDY decoding, so each
answer is reproducible to the token. An open answer counts as correct when an accepted form of the right answer
appears in its first sentence (so "Paris is the capital of France." and "The capital is Paris." both count, and
a correct word buried after a wrong claim does not).
  EVERYDAY   48 common facts asked as plain questions
  SCIQ       1,000 held-out science questions (never in the fine-tuning data), open answer
  ARC-EASY   held-out grade-school science, four-way, scored by total log-probability
Plus the actual answers to ten everyday questions, printed in full.
usage: eval/qa.py <ckpt>[=label] [<ckpt>[=label] ...]      baselines GPT-2 124M and SmolLM2-135M always included"""
import sys, re, glob, warnings, os
warnings.filterwarnings("ignore"); sys.path.insert(0, "/opt/code/membrae")
os.environ.setdefault("HF_HOME", "/mnt/kv_cache/hf_home")
from src.common import cap_gpu
cap_gpu(16.0)
import numpy as np, torch, datasets
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.bench import Ours

EVERYDAY = [
 ("What is the capital of France?", ["paris"]), ("What is the capital of Japan?", ["tokyo"]),
 ("What is the capital of Italy?", ["rome"]), ("What is the capital of Egypt?", ["cairo"]),
 ("What is the capital of Canada?", ["ottawa"]), ("What is the capital of Russia?", ["moscow"]),
 ("What is the largest planet in our solar system?", ["jupiter"]), ("Which planet is closest to the Sun?", ["mercury"]),
 ("What is the chemical symbol for gold?", ["au"]), ("What is the chemical symbol for iron?", ["fe"]),
 ("Water is made of hydrogen and which other element?", ["oxygen"]), ("What gas do plants absorb from the air?", ["carbon dioxide", "co2"]),
 ("How many chambers does the human heart have?", ["four", "4"]), ("At what temperature in Celsius does water freeze?", ["zero", "0"]),
 ("At what temperature in Celsius does water boil?", ["100", "one hundred"]), ("What is the largest ocean on Earth?", ["pacific"]),
 ("What is the longest river in the world?", ["nile", "amazon"]), ("What is the tallest mountain in the world?", ["everest"]),
 ("What is the currency of the United States?", ["dollar"]), ("What is the currency of Japan?", ["yen"]),
 ("How many days are there in a week?", ["seven", "7"]), ("How many months are there in a year?", ["twelve", "12"]),
 ("How many continents are there on Earth?", ["seven", "7"]), ("Who wrote Romeo and Juliet?", ["shakespeare"]),
 ("Who developed the theory of relativity?", ["einstein"]), ("Who was the first president of the United States?", ["washington"]),
 ("What language is spoken in Brazil?", ["portuguese"]), ("What language is spoken in Austria?", ["german"]),
 ("How many sides does a triangle have?", ["three", "3"]), ("What is two plus two?", ["four", "4"]),
 ("What is ten times ten?", ["100", "one hundred"]), ("What colour is the sky on a clear day?", ["blue"]),
 ("What colour is grass?", ["green"]), ("What do bees produce?", ["honey"]), ("What do cows produce?", ["milk"]),
 ("Which animal is known as the king of the jungle?", ["lion"]), ("What can penguins not do that most birds can?", ["fly", "flight"]),
 ("In which country is the Great Wall?", ["china"]), ("In which city is the Eiffel Tower?", ["paris"]),
 ("In which city is the Statue of Liberty?", ["new york"]), ("In which direction does the sun rise?", ["east"]),
 ("What is ice?", ["frozen water", "solid water", "water in its solid", "solid form of water"]),
 ("Which organ pumps blood around the body?", ["heart"]), ("Which organ do we use to breathe?", ["lung"]),
 ("What are doctors who treat teeth called?", ["dentist"]), ("Which season comes after winter?", ["spring"]),
 ("What is a baby dog called?", ["puppy", "pup"]), ("What is the freezing point of water in Fahrenheit?", ["32"]),
]
SHOW = ["What is the capital of France?", "Who wrote Romeo and Juliet?", "Why is the sky blue?",
        "What causes the seasons on Earth?", "How do plants make their food?", "What is the largest planet in our solar system?",
        "Explain what gravity is in one sentence.", "What is photosynthesis?", "Who was the first president of the United States?",
        "What is two plus two?"]
BASE = "/mnt/kv_cache/hf_home/datasets"
def arrow(name, hint):
    fs = [f for f in glob.glob(f"{BASE}/{name}/**/*.arrow", recursive=True) if hint in os.path.basename(f)]
    return datasets.concatenate_datasets([datasets.Dataset.from_file(f) for f in sorted(fs)])
SCIQ = arrow("sciq", "test")
ARC = arrow("allenai___ai2_arc", "test")
ARC = [r for r in ARC if "ARC-Easy" in r["id"] or True]
ARCE = [r for r in arrow("allenai___ai2_arc", "test") if len(r["choices"]["label"]) == 4]


class HF:
    def __init__(self, name, cache):
        self.tok = AutoTokenizer.from_pretrained(name, cache_dir=cache)
        self.m = AutoModelForCausalLM.from_pretrained(name, cache_dir=cache, torch_dtype=torch.bfloat16).cuda().eval()
        self.eos = self.tok.eos_token_id
    def encode(self, s): return self.tok.encode(s, add_special_tokens=False)
    @torch.no_grad()
    def answer(self, prompt, n=48):
        ids = self.encode(prompt); x = torch.tensor([ids], device="cuda")
        o = self.m.generate(x, max_new_tokens=n, do_sample=False, pad_token_id=self.eos, eos_token_id=self.eos)
        return self.tok.decode(o[0][len(ids):], skip_special_tokens=True)
    @torch.no_grad()
    def logprob(self, ctx, cont):
        a, b = self.encode(ctx), self.encode(cont); x = torch.tensor([a + b], device="cuda")
        lg = self.m(x).logits.float().log_softmax(-1)
        return float(lg[0, len(a) - 1:-1].gather(-1, x[0, len(a):].unsqueeze(-1)).sum())


class Mine:
    def __init__(self, ck):
        self.o = Ours(ck, "/mnt/kv_cache/tok_v8192.json")
    def encode(self, s): return self.o.encode(s)
    @torch.no_grad()
    def answer(self, prompt, n=48):
        out = self.o.generate(self.o.encode(prompt), n, temp=0.0, seed=0)
        if 8191 in out: out = out[:out.index(8191)]
        return self.o.decode(out)
    @torch.no_grad()
    def logprob(self, ctx, cont):
        a, b = self.encode(ctx), self.encode(cont)
        lg = self.o.logits_last(a + b, 1).float().log_softmax(-1)
        n = min(len(b), lg.shape[1] - 1)
        y = torch.tensor((a + b)[-n:], device=lg.device)
        return float(lg[0, -n - 1:-1].gather(-1, y.unsqueeze(-1)).sum())


def first_sentence(t):
    t = t.split("\nUser:")[0].strip()
    m = re.match(r"(.+?[.!?])(\s|$)", t, re.S)
    return (m.group(1) if m else t)[:300]


def prompt(q): return f"User: {q}\nAssistant:"


def run(lab, model):
    ev = [any(v in first_sentence(model.answer(prompt(q))).lower() for v in acc) for q, acc in EVERYDAY]
    sq = []
    for r in list(SCIQ)[:1000]:
        a = first_sentence(model.answer(prompt(r["question"]), 32)).lower()
        sq.append(r["correct_answer"].lower().strip() in a)
    arc = []
    for r in ARCE:
        lab_, txt = r["choices"]["label"], r["choices"]["text"]
        if r["answerKey"] not in lab_: continue
        lps = [model.logprob(prompt(r["question"]), " " + t) for t in txt]
        arc.append(int(np.argmax(lps)) == lab_.index(r["answerKey"]))
    print(f"{lab:<26s} {100*np.mean(ev):8.1f}% {100*np.mean(sq):8.1f}% {100*np.mean(arc):9.1f}%", flush=True)
    return model


specs = [(a.split("=")[0], a.split("=")[1] if "=" in a else os.path.basename(os.path.dirname(a))) for a in sys.argv[1:]]
print(f"{'model':<26s} {'EVERYDAY':>9s} {'SCIQ':>9s} {'ARC(4-way)':>10s}   (48 / 1,000 / {len(ARCE)} questions; greedy)")
shown = {}
for ck, lab in specs:
    m = run(lab, Mine(ck)); shown[lab] = [m.answer(prompt(q)) for q in SHOW]; del m; torch.cuda.empty_cache()
for name, cache, lab in (("gpt2", "/mnt/kv_cache/hf_cache", "GPT-2 124M"), ("HuggingFaceTB/SmolLM2-135M", "/mnt/kv_cache/hf_cache", "SmolLM2-135M")):
    m = run(lab, HF(name, cache)); shown[lab] = [m.answer(prompt(q)) for q in SHOW]; del m; torch.cuda.empty_cache()
print("\nTHE ANSWERS THEMSELVES (greedy, first 160 characters):")
for i, q in enumerate(SHOW):
    print(f"\n  Q: {q}")
    for lab, ans in shown.items():
        print(f"     {lab:<24s} {first_sentence(ans)[:160]!r}")
