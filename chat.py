"""Ask the model questions.

Uses the fine-tuned checkpoint and the template it was trained on -- "User: <question>\\nAssistant:" -- and stops
at the end of the answer. Greedy by default, so the same question always gets the same answer and anyone can
check it; '!temp 0.8' samples instead.

usage: .venv/bin/python chat.py [checkpoint]
       then type a question; a blank line quits.  '!temp 0.8' to sample, '!temp 0' for greedy,
       '!tokens 200' for longer answers, '!raw' to toggle plain text continuation (no template)."""
import sys, os, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.common import cap_gpu
cap_gpu(float(os.environ.get("CHAT_CAP", "6")))
from src.bench import Ours

CK = sys.argv[1] if len(sys.argv) > 1 else "/mnt/kv_cache/run46_sft/ckpt.pt"
m = Ours(CK, "/mnt/kv_cache/tok_v8192.json")
EOD = 8191
temp, n_tok, raw, seed = 0.0, 160, False, 0
print(f"model {os.path.basename(os.path.dirname(CK))} · greedy · type a question, blank line to quit")
print("commands: !temp 0.8   !tokens 200   !raw\n")
while True:
    try: q = input("you> ").strip()
    except (EOFError, KeyboardInterrupt): break
    if not q: break
    if q.startswith("!temp"): temp = float(q.split()[1]); print(f"  temperature {temp}"); continue
    if q.startswith("!tokens"): n_tok = int(q.split()[1]); print(f"  {n_tok} tokens"); continue
    if q == "!raw": raw = not raw; print(f"  {'plain continuation' if raw else 'question template'}"); continue
    seed += 1
    ids = m.encode(q if raw else f"User: {q}\nAssistant:")
    out = m.generate(ids, n_tok, temp=temp, seed=seed)
    if EOD in out: out = out[:out.index(EOD)]
    text = m.decode(out).split("\nUser:")[0].strip()
    print(f"model> {text}\n")
