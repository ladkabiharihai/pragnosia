"""Shared utilities: standard logging location and a tqdm progress bar for every training loop.
All logs live in logs/ (never scattered in the workspace root). Usage:
    from util import LOGS, pbar
    for i in pbar(range(STEPS), "substrate", loss=lambda: cur):   ...
"""
import os,sys
from tqdm import tqdm
ROOT=os.path.dirname(os.path.abspath(__file__)); LOGS=os.path.join(ROOT,"logs")
os.makedirs(LOGS,exist_ok=True)
def logpath(name): return os.path.join(LOGS,f"{name}.log")
def pbar(it,desc="train",total=None,every=5.0):
    """tqdm bar that is readable in a log file: updates at most every `every` seconds, fixed width, to stdout."""
    return tqdm(it,desc=desc,total=total,file=sys.stdout,ncols=110,mininterval=every,
                smoothing=0.1,bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}")
