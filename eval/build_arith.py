"""ARITHMETIC AND NUMBER SENSE AS TRAINING TEXT (2026-10-08; the knowledge + arithmetic phase the user approved). The final base
answered 0 of 8 arithmetic questions (2 + 2: "1"; 12 x 12: "12 is equal to 2") and 8.3% on four-way arithmetic, below chance:
the ~25B tokens of web text it read hold too little arithmetic for a 98M model to learn it, and the tokenizer splits numbers
irregularly (144 -> "14" "4"), so every number format has to be seen often. Generated, never copied from a test:
  facts of arithmetic   + - x / on 1-4 digit numbers, in many phrasings ("15 + 27 = 42", "15 plus 27 is 42.", "The product of
                        12 and 12 is 144.", "User: What is 7 plus 8?\\nAssistant: 7 plus 8 is 15.")
  worked steps          column addition and multiplication written out, digit by digit
  word problems         templated stories (buying, sharing, saving, travelling) with the arithmetic stated
  number sense          comparisons, counting, sequences, even/odd, days of the week, units
Documents of 8-40 items, each followed by end-of-document. usage: eval/build_arith.py <out_dir> <million_tokens>"""
import os, sys, random
import numpy as np
from tokenizers import Tokenizer
OUT, MTOK = sys.argv[1], float(sys.argv[2]); os.makedirs(OUT, exist_ok=True)
tok = Tokenizer.from_file("/mnt/kv_cache/tok_v8192.json"); EOD = 8191
r = random.Random(20261008)
WORD = {"+": ["plus", "added to", "and"], "-": ["minus", "take away", "less"], "*": ["times", "multiplied by"], "/": ["divided by", "over"]}
NAMES = ["Tom", "Sara", "Ravi", "Mei", "Ali", "Ana", "Ben", "Priya", "Omar", "Lena", "Kofi", "Yuki", "Diego", "Nora", "Ivan", "Zara"]
ITEMS = ["apples", "books", "pencils", "coins", "stickers", "marbles", "cards", "eggs", "cookies", "flowers", "stamps", "shells"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
SMALL = {0: "zero", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten",
         11: "eleven", 12: "twelve", 15: "fifteen", 20: "twenty"}

def num(d): return r.randint(10 ** (d - 1) if d > 1 else 0, 10 ** d - 1)
def digits(): return r.choices([1, 2, 3, 4], weights=[30, 40, 20, 10])[0]

def problem():
    op = r.choices("+-*/", weights=[35, 25, 25, 15])[0]
    if op == "+": a, b = num(digits()), num(digits()); c = a + b
    elif op == "-": a, b = num(digits()), num(digits()); a, b = max(a, b), min(a, b); c = a - b
    elif op == "*": a = num(r.choice([1, 1, 2, 2, 3])); b = num(r.choice([1, 1, 2])); c = a * b
    else: b = r.randint(2, 12 if r.random() < .7 else 99); c = num(r.choice([1, 2, 2, 3])); a = b * c
    return op, a, b, c

def fact_line():
    op, a, b, c = problem(); sym = {"+": "+", "-": "-", "*": "x" if r.random() < .3 else "*", "/": "/"}[op]; w = r.choice(WORD[op])
    name = {"+": "sum", "-": "difference", "*": "product", "/": "quotient"}[op]
    return r.choice([
        f"{a} {sym} {b} = {c}", f"{a} {w} {b} is {c}.", f"{a} {w} {b} equals {c}.", f"What is {a} {w} {b}? {c}",
        f"What is {a} {w} {b}? The answer is {c}.", f"Q: {a} {sym} {b} = ?\nA: {c}", f"The {name} of {a} and {b} is {c}.",
        f"User: What is {a} {w} {b}?\nAssistant: {a} {w} {b} is {c}.", f"User: What is {a} {sym} {b}?\nAssistant: {c}",
        f"Question: What is {a} {w} {b}? Answer: {c}", f"{c} = {a} {sym} {b}",
        f"If you compute {a} {sym} {b}, you get {c}."])

def worked():
    if r.random() < .6:                                   # column addition, digit by digit
        a, b = num(r.choice([2, 3, 4])), num(r.choice([2, 3])); da, db = str(a)[::-1], str(b)[::-1]; carry = 0; steps = []
        for i in range(max(len(da), len(db))):
            x = int(da[i]) if i < len(da) else 0; y = int(db[i]) if i < len(db) else 0; s = x + y + carry
            steps.append(f"{x} + {y}" + (f" + {carry}" if carry else "") + f" = {s}, write {s % 10}" + (", carry 1" if s >= 10 else ""))
            carry = s // 10
        if carry: steps.append("write the last carry 1")
        return f"Add {a} and {b}: " + "; ".join(steps) + f". So {a} + {b} = {a + b}."
    a, b = num(r.choice([2, 3])), r.randint(2, 9)           # multiplication by one digit, written out
    parts = [f"{b} x {int(d) * 10 ** i} = {b * int(d) * 10 ** i}" for i, d in enumerate(str(a)[::-1])]
    return f"Multiply {a} by {b}: " + "; ".join(parts[::-1]) + f". Adding them: {a} x {b} = {a * b}."

def word_problem():
    n, it = r.choice(NAMES), r.choice(ITEMS); a, b = r.randint(2, 60), r.randint(2, 40); k = r.randint(2, 12)
    return r.choice([
        f"{n} has {a} {it} and gets {b} more. Now {n} has {a + b} {it}.",
        f"{n} had {a + b} {it} and gave away {b}. {n} has {a} {it} left.",
        f"{n} buys {k} bags with {b} {it} in each bag. That is {k} x {b} = {k * b} {it}.",
        f"{n} shares {k * b} {it} equally among {k} friends. Each friend gets {b} {it}.",
        f"User: {n} has {a} {it} and buys {b} more. How many {it} does {n} have?\nAssistant: {n} has {a} + {b} = {a + b} {it}.",
        f"A train travels {b} kilometres every hour. In {k} hours it travels {b * k} kilometres.",
        f"{n} saves {b} dollars a week. After {k} weeks {n} has saved {b * k} dollars.",
        f"Question: {n} had {a + b} {it} and lost {a}. How many are left? Answer: {b}"])

def sense():
    t = r.randrange(7)
    if t == 0: a, b = r.sample(range(1, 1000), 2); return f"Which is larger, {a} or {b}? {max(a, b)} is larger."
    if t == 1: s, d = r.randint(1, 50), r.randint(1, 10); seq = [s + d * i for i in range(5)]; return f"What comes next in {', '.join(map(str, seq[:4]))}? {seq[4]}"
    if t == 2: s = r.randint(1, 90); return f"Counting from {s} to {s + 6}: " + ", ".join(str(s + i) for i in range(7)) + "."
    if t == 3: x = r.randint(1, 999); return f"{x} is an {'even' if x % 2 == 0 else 'odd'} number."
    if t == 4: d, k = r.randrange(7), r.randint(1, 10); return f"If today is {DAYS[d]}, then in {k} day{'s' if k > 1 else ''} it will be {DAYS[(d + k) % 7]}."
    if t == 5: m, k = r.randrange(12), r.randint(1, 6); return f"{k} month{'s' if k > 1 else ''} after {MONTHS[m]} comes {MONTHS[(m + k) % 12]}."
    x = r.choice(list(SMALL)); y = r.choice(list(SMALL)); return f"{SMALL[x].capitalize()} plus {SMALL[y]} is {x + y}."

def document():
    kinds = [fact_line] * 6 + [worked] * 2 + [word_problem] * 3 + [sense] * 2
    return "\n".join(r.choice(kinds)() for _ in range(r.randint(8, 40)))

target = int(MTOK * 1e6); out = []; total = 0
while total < target:
    for e in tok.encode_batch([document() for _ in range(4000)]):
        out.append(np.array(e.ids + [EOD], dtype=np.uint16)); total += len(e.ids) + 1
    print(f"  {total / 1e6:.1f}M tokens", flush=True)
arr = np.concatenate(out)[:target]; arr.tofile(f"{OUT}/arith.bin")
print(f"done: {len(arr):,} tokens -> {OUT}/arith.bin; {len(out):,} documents")
print("SAMPLE:\n" + document()[:1200])
