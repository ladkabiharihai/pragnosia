"""CAN THE CARRIER RECALL PAST THE ATTENTION WINDOW? Multi-query associative recall (Arora et al. 2023, "Zoology"), the
standard capability test for recurrent memories, made long: every sequence holds key-value pairs, single tokens, at
random places among random filler, and ends with queries -- a key, then the value the model must predict. A query
whose pair sits 512 or more tokens back cannot be answered by the windowed attention (block-local, 512): only the
carrier's state can. Trained on this task alone, small models learn it in minutes if the architecture can.

  eval/mqar.py build <out_dir> <million_tokens>      writes <out_dir>/mqar.bin (sequences of 2,048 + end token)
  eval/mqar.py test <ckpt> [n_sequences]             accuracy on fresh sequences by key-to-query distance
Built 2026-10-02 after needles were retrieved only inside the attention window (PROGRESS.md)."""
import os, sys
import numpy as np

T, NPAIR, EOD = 2048, 32, 8191
KEYS, VALS, FILL = (1000, 2000), (2000, 3000), (3000, 8000)


def sequence(rng):
    """(tokens, [(query position of the value, distance back to the pair)])"""
    keys = rng.choice(np.arange(*KEYS), NPAIR, replace=False)
    vals = rng.integers(*VALS, NPAIR)
    # FILLER IS A PREDICTABLE CYCLE (64 tokens from a random phase), not random tokens: the trainer scores every
    # token, and random filler made 97% of the loss irreducible noise -- the first run's loss sat at log(5,000) and the
    # spike guard stopped it (2026-10-02). Zoology scores only the answers; a learnable filler does the same here.
    seq = (FILL[0] + (rng.integers(0, 64) + np.arange(T)) % 64).astype(np.int64)
    q0 = T - 2 * NPAIR                                    # the queries fill the end
    spots = np.sort(rng.choice(np.arange(0, q0 - 2, 2), NPAIR, replace=False))
    order = rng.permutation(NPAIR)
    for i, s in zip(order, spots): seq[s], seq[s + 1] = keys[i], vals[i]
    where = {int(keys[i]): int(s) for i, s in zip(order, spots)}
    ans = []
    for j, i in enumerate(rng.permutation(NPAIR)):
        p = q0 + 2 * j
        seq[p], seq[p + 1] = keys[i], vals[i]
        ans.append((p + 1, p + 1 - (where[int(keys[i])] + 1)))
    return seq, ans


if __name__ == "__main__" and sys.argv[1] == "build":
    out, mtok = sys.argv[2], float(sys.argv[3])
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(20261002); n = int(mtok * 1e6) // (T + 1)
    arr = np.empty(n * (T + 1), dtype=np.uint16)
    for k in range(n):
        s, _ = sequence(rng); arr[k * (T + 1):(k + 1) * (T + 1) - 1] = s; arr[(k + 1) * (T + 1) - 1] = EOD
    arr.tofile(f"{out}/mqar.bin"); print(f"{len(arr):,} tokens, {n:,} sequences -> {out}/mqar.bin")

if __name__ == "__main__" and sys.argv[1] == "test":
    sys.path.insert(0, "/opt/code/membrae")
    import torch
    from src.common import cap_gpu
    cap_gpu(8.0)
    from src.model import LM
    sd = torch.load(sys.argv[2], map_location="cuda", weights_only=False)
    m = LM(**sd["cfg"]).cuda(); m.load_state_dict(sd["model"]); m.eval()
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 200
    rng = np.random.default_rng(777)                      # sequences never trained on
    hits = {b: [0, 0] for b in ("<512", "512-1023", "1024+")}
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for _ in range(n):
            s, ans = sequence(rng)
            x = torch.tensor(s[None, :T], device="cuda")
            pred = m.trunk(x)                                     # (1, T, d): block-local attention, as trained
            lg = torch.nn.functional.linear(pred[0], m.e.weight)
            for p, dist in ans:
                b = "<512" if dist < 512 else "512-1023" if dist < 1024 else "1024+"
                hits[b][0] += int(lg[p - 1].argmax()) == int(s[p]); hits[b][1] += 1
    print(f"MQAR, {NPAIR} pairs among random filler, {T} tokens, {n} fresh sequences; {sys.argv[2]}")
    for b, (k, t) in hits.items():
        print(f"  key {b:>9s} tokens back: {100 * k / max(1, t):5.1f}%  ({k}/{t})   chance {100 / (VALS[1] - VALS[0]):.2f}%")
