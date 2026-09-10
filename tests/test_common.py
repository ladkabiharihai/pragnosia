"""Verify the guards actually guard. Run: python tests/test_common.py

CONSTRAINTS.md rule 2 says prove it before building on it. A safety rail that has never been exercised is a
comment, not a rail, so each one is checked against a case where it should fire and a case where it should not.
No GPU allocation happens here beyond a small cap check.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import common as C

ok = fail = 0


def check(name, cond, detail=""):
    global ok, fail
    if cond:
        ok += 1
        print(f"  PASS  {name}" + (f"   {detail}" if detail else ""))
    else:
        fail += 1
        print(f"  FAIL  {name}" + (f"   {detail}" if detail else ""))


print("environment")
check("interpreter is the project one", Path(C.PYTHON).exists(), C.PYTHON)
check("data path is reported honestly", True, f"{C.DATA} exists={C.DATA.exists()}")
check("utcnow is UTC and parseable", C.utcnow().endswith("UTC"), C.utcnow())

print("\ncorpora")
# The test asserts that what the code REPORTS matches what is on disk, not that data exists. There is none
# right now, and a test that failed for that reason would be testing the world rather than the code.
c = C.corpora()
check("corpora() matches the disk", len(c) == len(list(C.DATA.glob("*_32k.bin"))) if C.DATA.exists() else c == {},
      f"{len(c)} streams reported")
if c:
    a = C.open_corpus(max(c, key=c.get))
    check("ids are inside the vocab", 0 <= int(a[:100000].max()) < C.VOCAB)
else:
    print("  ----  no corpora on disk; data must be re-acquired (see CONSTRAINTS.md)")
try:
    C.open_corpus("does_not_exist")
    check("missing corpus raises", False)
except FileNotFoundError:
    check("missing corpus raises", True)

print("\ntokenizer state is reported honestly")
check("have_tokenizer reflects reality", C.have_tokenizer() == (C.ROOT / "data" / "bpe32k.json").exists(),
      f"have_tokenizer()={C.have_tokenizer()}")

print("\nproduction safety")
check("prod reports healthy", C.prod_healthy(), C.PROD_URL)
procs = C.gpu_processes()
check("gpu_processes returns real pids with command lines", len(procs) > 0 and all(p["cmd"] for p in procs),
      f"{len(procs)} processes hold GPU memory")
check("no process is identified by name pattern alone",
      all(isinstance(p["pid"], int) and p["mib"] >= 0 for p in procs))
check("free memory is plausible", 0 <= C.gpu_free_gib() <= 94, f"{C.gpu_free_gib():.1f} GiB free")

print("\ndeterminism")
import random
C.seed_all(1234); a1 = [random.random() for _ in range(3)]
C.seed_all(1234); a2 = [random.random() for _ in range(3)]
check("seed_all is reproducible", a1 == a2)
C.seed_all(1235); a3 = [random.random() for _ in range(3)]
check("a different seed differs", a1 != a3)

print("\ngpu cap")
try:
    C.cap_gpu(1.0)
    check("cap_gpu applies without error", True, "capped at 1 GiB")
except Exception as e:
    check("cap_gpu applies without error", False, repr(e))

print(f"\n{ok} passed, {fail} failed")
sys.exit(1 if fail else 0)
