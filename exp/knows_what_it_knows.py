"""VISION 4 AND 5, MEASURED FOR THE FIRST TIME: does the model know what it knows, and abstain otherwise?

These are two of the eight vision goals and neither has ever been measured on this model. Everything else in
the claim sheet is about capability; this is about CALIBRATION, which is a different property: a model that is
right 50% of the time and KNOWS which 50% is far more useful than one that is right 60% of the time and cannot
tell. It is also the precondition for abstention, which is goal 5.

THREE MEASUREMENTS.

  1. Confidence separates right from wrong (goal 4). For each item take the model's probability on the answer
     it chose, and ask whether that number predicts whether it was correct. Reported as AUROC: 0.5 is a model
     whose confidence carries no information at all, 1.0 is perfect self-knowledge.

  2. Confidence separates real from fabricated (goal 5). The same stems are asked about entities that do not
     exist ("the capital of Zorvania"). A model that knows what it does not know should be markedly less
     confident here. If it is equally confident, it cannot abstain, because there is no signal to abstain ON.

  3. Expected calibration error. Of the times it says 80%, is it right 80% of the time?

Measured on the base 125M, the SFT 125M and the 405M, because if SFT or scale moves calibration that matters
more than the small accuracy differences we have been chasing.
"""
import sys, os, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, "/opt/code/membrae"); sys.path.insert(0, "/opt/code/membrae/exp")
from util import setup_log, cfg_from_ckpt, cap_gpu
setup_log("knows_what_it_knows")
cap_gpu(float(os.environ.get("CAP", "14")))
import numpy as np, torch
import core as C
from tokenizers import Tokenizer
from facts_set import FACTS

DEV = C.DEV; TOK = Tokenizer.from_file("/opt/code/membrae/data/bpe32k.json")
D = "/mnt/kv_cache/pragnosia_data"

# CONTROLLED FABRICATION. A first version gave the fake questions invented candidate answers (" Kelmark",
# " Vasmere"), which meant the real and fake sets differed in BOTH the subject and the rarity of the options,
# so confidence could have been tracking token frequency rather than knowledge. These reuse the REAL questions'
# candidate sets verbatim and change only the SUBJECT to one that does not exist, so the option statistics are
# identical by construction and the only difference is whether the thing being asked about is real.
FAKE = [(stem.replace("France", "Zorvania").replace("Japan", "Brelandia")
             .replace("Italy", "Novaquia").replace("Egypt", "Ferrovia")
             .replace("Canada", "Ristonia").replace("Russia", "Vandoria")
             .replace("Brazil", "Kelmoor").replace("Austria", "Tavrenne")
             .replace("the world", "Zorvania").replace("Earth", "Zorvania")
             .replace("our solar system", "the Zorvanian system")
             .replace("the United States", "Zorvania").replace("China", "Brelandia"),
         [right] + wrongs)
        for stem, right, wrongs in FACTS
        if any(w in stem for w in ("France","Japan","Italy","Egypt","Canada","Russia","Brazil","Austria",
                                   "the world","Earth","our solar system","the United States","China"))]
_OLD_FAKE = [
 ("The capital of Zorvania is", [" Kelmark", " Trenholm", " Vasmere", " Oldrin"]),
 ("The capital of Brelandia is", [" Astoril", " Vernhold", " Quillane", " Merrow"]),
 ("The chemical symbol for florium is", [" Fl", " Fo", " Fm", " Fr"]),
 ("The chemical symbol for zenthium is", [" Zn", " Ze", " Zt", " Zh"]),
 ("The largest moon of Kepler-9 is", [" Tarvos", " Orindel", " Kessin", " Malbec"]),
 ("The longest river in Vandoria is the", [" Solane", " Merrick", " Tavish", " Corran"]),
 ("The currency of Ristonia is the", [" drachos", " marlen", " veskin", " tolar"]),
 ("The author of The Crimson Tessellation is Aldous", [" Verrin", " Kastel", " Morrow", " Blaine"]),
 ("The tallest mountain in Ferrovia is Mount", [" Calder", " Orsin", " Thawn", " Belmar"]),
 ("The language spoken in Novaquia is", [" Novaquian", " Rellish", " Tamoran", " Vessik"]),
 ("The inventor of the tessarion engine was Hugo", [" Marlin", " Cresswell", " Danner", " Voight"]),
 ("The primary export of Kelmoor is", [" grain", " copper", " timber", " salt"]),
]

def auroc(scores, labels):
    """Probability a positive outranks a negative. Rank-based, so it needs no threshold."""
    s = np.asarray(scores, float); y = np.asarray(labels, int)
    if y.sum() == 0 or y.sum() == len(y): return float('nan')
    order = np.argsort(s); ranks = np.empty(len(s), float); ranks[order] = np.arange(1, len(s) + 1)
    npos = y.sum(); nneg = len(y) - npos
    return float((ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))

@torch.no_grad()
def probs(m, cfg, stem, cands):
    """Softmax over the candidate set of mean log-prob: the model's distribution over the options it was given."""
    lps = []
    for c in cands:
        si = TOK.encode(stem).ids; ci = TOK.encode(c).ids
        x = torch.tensor([si + ci], device=DEV)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lp = m(x[:, :-1]).float().log_softmax(-1)
        lps.append(float(lp[0, -len(ci):].gather(-1, x[0, 1:][-len(ci):].unsqueeze(-1)).mean()))
    a = np.array(lps); a = np.exp(a - a.max()); return a / a.sum()

print(f"{'model':>18s}{'acc':>7s}{'AUROC conf|correct':>21s}{'conf real':>11s}{'conf fake':>11s}"
      f"{'AUROC real|fake':>18s}{'ECE':>8s}")
print("-" * 96)
for tag, path in [("125M base", f"{D}/p124f.pt"), ("125M SFT", f"{D}/p124f_sft.pt"),
                  ("405M", f"{D}/p400f_405M_snap.pt")]:
    if not os.path.exists(path): continue
    sd = torch.load(path, map_location=DEV)
    cfg = cfg_from_ckpt(sd, T=1024, res_scale=sd['res_scale'])
    m = C.Core(cfg).to(DEV); m.load_state_dict(sd['model']); m.eval()

    conf, corr = [], []
    for stem, right, wrongs in FACTS:
        p = probs(m, cfg, stem, [right] + wrongs)
        conf.append(float(p.max())); corr.append(int(np.argmax(p) == 0))
    fake_conf = [float(probs(m, cfg, stem, cands).max()) for stem, cands in FAKE]
    if not fake_conf: raise RuntimeError("no fabricated stems built -- the substitution list matched nothing")

    acc = float(np.mean(corr))
    a1 = auroc(conf, corr)
    a2 = auroc([-c for c in conf] + [-c for c in fake_conf],
               [1]*len(conf) + [0]*len(fake_conf))          # can low confidence flag the fabricated ones
    bins = np.linspace(0.25, 1.0, 6); ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        idx = [i for i, c in enumerate(conf) if lo <= c < hi]
        if idx: ece += len(idx)/len(conf) * abs(np.mean([corr[i] for i in idx]) - np.mean([conf[i] for i in idx]))
    print(f"{tag:>18s}{100*acc:>6.1f}%{a1:>21.3f}{np.mean(conf):>11.3f}{np.mean(fake_conf):>11.3f}"
          f"{1-a2:>18.3f}{ece:>8.3f}", flush=True)
    del m; torch.cuda.empty_cache()

print("""
  AUROC conf|correct : does its confidence predict its own correctness? 0.5 = no self-knowledge (goal 4)
  AUROC real|fake    : does confidence separate real questions from fabricated ones? 0.5 = cannot abstain (goal 5)
  ECE                : gap between stated confidence and actual accuracy, lower is better
  Chance accuracy is 25%. Confidence is over 4 options so 0.25 is uniform and 1.0 is certain.""")
