"""Shared infrastructure: environment guards, determinism, logging, and the progress journal.

Everything here enforces a rule from CONSTRAINTS.md in code rather than leaving it to discipline. The previous
effort lost more time to process and measurement mistakes than to modelling ones: caps guessed instead of
measured, the wrong PID killed, results quoted from a single seed, a criterion printed that differed from the
criterion tested. Those are the failures this module exists to make harder.

Deliberately one file. A new module needs a reason.
"""
from __future__ import annotations
import json
import os
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "logs"
PROGRESS = ROOT / "PROGRESS.md"
PYTHON = str(ROOT / ".venv" / "bin" / "python")   # this project's own venv, not parakeet_env
DATA = Path("/mnt/kv_cache/pragnosia_data")
VOCAB = 32768
PROD_URL = "http://localhost:8000/v1/models"


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


# --------------------------------------------------------------------------------------- production safety
def prod_healthy(timeout: int = 5) -> bool:
    """Is the shared production endpoint answering? Checked BEFORE and AFTER anything that touches the GPU."""
    try:
        out = subprocess.run(["curl", "-s", "-m", str(timeout), "-o", "/dev/null",
                              "-w", "%{http_code}", PROD_URL],
                             capture_output=True, text=True, timeout=timeout + 5)
        return out.stdout.strip() == "200"
    except Exception:
        return False


def require_prod() -> None:
    """Refuse to start if production is already unhealthy: otherwise we get blamed for a pre-existing outage,
    and worse, we cannot tell whether we caused one."""
    if not prod_healthy():
        raise RuntimeError(f"production at {PROD_URL} is not healthy; refusing to start GPU work")


def gpu_processes() -> list[dict]:
    """Who actually holds GPU memory, with their real command line.

    This is the ONLY safe basis for deciding what to kill. `pgrep -f name` matches the searching command's own
    arguments and has, in this project, matched itself and led to a wrapper being killed while the real trainer
    kept running, after which a second trainer was launched on top of the first."""
    try:
        q = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20)
    except Exception:
        return []
    out = []
    for line in q.stdout.strip().splitlines():
        if not line.strip():
            continue
        pid_s, mib_s = (x.strip() for x in line.split(","))
        try:
            cmd = Path(f"/proc/{pid_s}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except OSError:
            cmd = "<gone>"
        out.append({"pid": int(pid_s), "mib": int(mib_s), "cmd": cmd.strip()})
    return out


def gpu_free_gib() -> float:
    q = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total",
                        "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20)
    used, total = (int(x) for x in q.stdout.strip().splitlines()[0].split(","))
    return (total - used) / 1024


def cap_gpu(gib: float) -> None:
    """Hard per-process memory ceiling. Import torch late so this is usable from non-torch code.

    A cap is not the same as checking free VRAM before launching: free VRAM is a race against every other
    process on a shared card, and a cap is a guarantee. Measure what the job needs first; a cap chosen by guess
    gets discovered by crashing, which happened three times in one afternoon."""
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("no CUDA device")
    total = torch.cuda.get_device_properties(0).total_memory / 2**30
    torch.cuda.set_per_process_memory_fraction(min(1.0, gib / total), 0)
    print(f"[gpu] capped at {gib:.1f} GiB of {total:.0f} GiB; {gpu_free_gib():.1f} GiB free on the card",
          flush=True)


# --------------------------------------------------------------------------------------------- determinism
def seed_all(seed: int) -> int:
    """Seed every generator we use and return the seed so the caller can record it."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    return seed


# ------------------------------------------------------------------------------------------------- logging
class Run:
    """One run, one log file, one progress entry. Not one file per thought.

    Opening a Run records the start in PROGRESS.md; closing it records the outcome, including a failure and its
    traceback. A run that crashed leaves a journal entry saying so."""

    def __init__(self, name: str, note: str = "", seed: int | None = None):
        LOGS.mkdir(exist_ok=True)
        self.name = name
        self.note = note
        self.t0 = time.time()
        self.path = LOGS / f"{datetime.now(timezone.utc):%Y%m%d-%H%M%S}-{name}.log"
        self.fh = self.path.open("w", buffering=1)
        self.seed = seed_all(seed) if seed is not None else None
        self.log(f"START {name}  {utcnow()}")
        if note:
            self.log(f"  {note}")
        if self.seed is not None:
            self.log(f"  seed {self.seed}")

    def log(self, msg: str) -> None:
        line = f"[{time.time()-self.t0:8.1f}s] {msg}"
        print(line, flush=True)
        self.fh.write(line + "\n")

    def close(self, outcome: str) -> None:
        mins = (time.time() - self.t0) / 60
        self.log(f"END {self.name}  {outcome}  ({mins:.1f} min)")
        self.fh.close()
        journal(f"**{self.name}** ({mins:.1f} min) {outcome}",
                detail=self.note, log=self.path.name, seed=self.seed)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close("FAILED: " + repr(exc) if exc else "ok")
        return False


def journal(summary: str, detail: str = "", log: str = "", seed: int | None = None) -> None:
    """Append a dated entry to PROGRESS.md. Every event goes here, including the ones that went wrong."""
    parts = [f"- `{utcnow()}` {summary}"]
    if detail:
        parts.append(f"  {detail}")
    tail = []
    if log:
        tail.append(f"log `logs/{log}`")
    if seed is not None:
        tail.append(f"seed {seed}")
    if tail:
        parts.append("  (" + ", ".join(tail) + ")")
    with PROGRESS.open("a") as f:
        f.write("\n".join(parts) + "\n")


# ------------------------------------------------------------------------------------------------- corpora
def corpora() -> dict[str, int]:
    """Available token streams and their lengths in tokens. uint16 ids, vocab 32768, no tokenizer required."""
    return {p.stem.replace("_32k", ""): p.stat().st_size // 2 for p in sorted(DATA.glob("*_32k.bin"))}


def open_corpus(name: str):
    import numpy as np
    p = DATA / f"{name}_32k.bin"
    if not p.exists():
        raise FileNotFoundError(f"{p} not found; available: {sorted(corpora())}")
    return np.memmap(p, dtype=np.uint16, mode="r")


def have_tokenizer() -> bool:
    """False as of 2026-09-10. Anything needing TEXT (bits-per-byte, decoding, chat) is blocked until this is
    True; anything training on ids is not."""
    return (ROOT / "data" / "bpe32k.json").exists()


# ------------------------------------------------------------------------------------- bits per byte
def bpb_from_nats(total_nats: float, total_bytes: int) -> float:
    """Bits per byte. The only loss number worth comparing across models.

    Cross-entropy per TOKEN depends on the vocabulary: a model with a 8k vocab and one with 32k can report
    very different per-token losses while predicting the same text equally well, because they cut it into
    different numbers of pieces. Dividing the total information by the number of BYTES removes the tokenizer
    from the comparison, which is why every cross-model claim in this project must be in bits per byte."""
    import math
    return total_nats / (math.log(2) * max(1, total_bytes))


@torch.no_grad() if False else (lambda f: f)
def eval_bpb(model, tok, bins, ks, ps, rng, n_batches=20, B=8, T=512, vocab=8192):
    """Held-out bits per byte. Decodes the targets to count real bytes rather than assuming a ratio."""
    import numpy as np, torch, torch.nn.functional as F
    model.eval()
    nats = 0.0; nbytes = 0
    # HALVE AND RETRY. The eval runs while the training graph pool still holds its reserve, so the batch that
    # fits for one architecture does not fit for another: the 1-in-4 hybrid died here at ctx 8192 with 9 GiB
    # already reserved (2026-09-18), and the chunked-logits fix of the day before did not cover the trunk's
    # own activations. An evaluation that changes its batch size measures the same thing; one that dies does
    # not measure anything.
    while True:
        try:
            return _eval_bpb(model, tok, bins, ks, ps, rng, n_batches, B, T, vocab)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            if B == 1:
                raise
            B = max(1, B // 2); n_batches = n_batches * 2
            print(f"[eval] out of memory; retrying at batch {B}", flush=True)


def _eval_bpb(model, tok, bins, ks, ps, rng, n_batches, B, T, vocab):
    import numpy as np, torch, torch.nn.functional as F
    model.eval()
    nats = 0.0; nbytes = 0
    with torch.no_grad():
        for _ in range(n_batches):
            o = np.empty((B, T + 1), dtype=np.int64)
            for i in range(B):
                a = bins[ks[rng.choice(len(ks), p=ps)]]
                j = rng.integers(0, len(a) - T - 2)
                o[i] = a[j:j + T + 1]
            x = torch.from_numpy(o).cuda()
            y = x[:, 1:T + 1]
            # CHUNKED, like the training loss: all the logits at once are B x T x V in fp32, which at B8
            # T8192 V8192 is 2.0 GiB and is exactly what killed run20's eval at ctx 8192 inside a 14 GiB cap
            # on 2026-09-17. Rows in blocks of 4096 hold one block of logits at 128 MiB instead.
            with torch.autocast("cuda", dtype=torch.bfloat16):
                h = model.trunk(x[:, :T])
            hf = h.reshape(-1, h.shape[-1]); yf = y.reshape(-1)
            for s0 in range(0, hf.shape[0], 4096):
                hc = hf[s0:s0 + 4096]
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    lg = F.linear(hc, model.e.weight)
                nats += F.cross_entropy(lg.float(), yf[s0:s0 + 4096], reduction="sum").item()
            for b in range(B):
                nbytes += len(tok.decode([int(t) for t in y[b]]).encode("utf-8"))
    model.train()
    return bpb_from_nats(nats, nbytes), nbytes
