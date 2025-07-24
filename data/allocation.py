# data/allocation.py
# --------------------------------------------------------------------------- #
#  Spawn N threads, each grabbing a non‑overlapping slice of ShareGPT and
#  dumping hidden‑state snapshots with *ge_data_all_vicuna.py*.               #
# --------------------------------------------------------------------------- #
import argparse, os, subprocess
from concurrent.futures import ThreadPoolExecutor

parser = argparse.ArgumentParser(description="Shard ShareGPT allocation")
parser.add_argument("--outdir", type=str, default="./Data")
parser.add_argument("--start",  type=int, default=0)
parser.add_argument("--end",    type=int, default=68000)
parser.add_argument("--gpus",   type=str, default="0,1,2,3,4,5,6,7",
                    help="Comma‑sep physical GPU ids; one thread per entry.")
args = parser.parse_args()

gpu_lists = [[g] for g in args.gpus.split(",")]
n_workers = len(gpu_lists)

def split_range(lo, hi, n):
    step = (hi - lo) // n
    extra = (hi - lo) % n
    pts = []
    cur = lo
    for k in range(n):
        nxt = cur + step + (1 if k < extra else 0)
        pts.append((cur, nxt))
        cur = nxt
    return pts

ranges = split_range(args.start, args.end, n_workers)
os.makedirs(args.outdir, exist_ok=True)

def launch(gid, r):
    cmd = [
        "python", "data/ge_data_all_vicuna.py",
        f"--start={r[0]}",
        f"--end={r[1]}",
        f"--index={gid}",
        f"--gpu_index", *map(str, gpu_lists[gid]),
        f"--outdir={args.outdir}"
    ]
    print(" ".join(cmd))
    subprocess.call(cmd)

with ThreadPoolExecutor(max_workers=n_workers) as pool:
    for idx, r in enumerate(ranges):
        pool.submit(launch, idx, r)
