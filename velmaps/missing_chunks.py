"""Rows of match_fix/full_20kpc_nn100.csv whose velocity maps are not all written yet, split into N chunks.

The first production run (submit_velmaps.sh, 25 jobs x 12 h) hit the time limit: rows took ~15 min, not 2-3 min.
A row is done when <outroot>/<model>/<snap>/<halo>_{x,y,z}.pkl all exist (vel_maps.py skips those anyway).
Chunks are interleaved over the remaining rows (mixes models and snapshots, similar run times).
usage (from create_obs/): python match_fix/velmaps/missing_chunks.py --outroot <dir> [--n 20]
  -> match_fix/velmaps/missing_<i>.csv, and prints the number of remaining rows per model
"""
import csv, os, argparse
from collections import Counter
HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser(); ap.add_argument("--outroot", required=True); ap.add_argument("--n", type=int, default=20)
ap.add_argument("--table", default=os.path.join(os.path.dirname(HERE), "full_20kpc_nn100.csv")); a = ap.parse_args()
rows = list(csv.DictReader(open(a.table)))
done = lambda r: all(os.path.isfile(f"{a.outroot}/{r['model']}/{int(float(r['snap']))}/{r['halo']}_{ax}.pkl") for ax in "xyz")
todo = [r for r in rows if not done(r)]
print(f"{len(rows)} rows in {a.table}; {len(rows) - len(todo)} done, {len(todo)} to do")
print("to do per model:", dict(sorted(Counter(r["model"] for r in todo).items())))
for f in os.listdir(HERE):
    if f.startswith("missing_") and f.endswith(".csv"): os.remove(os.path.join(HERE, f))
n = min(a.n, len(todo))
for i in range(n):
    with open(f"{HERE}/missing_{i:02d}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, rows[0].keys()); w.writeheader(); w.writerows(todo[i::n])
print(f"wrote {n} chunk tables missing_00..{n - 1:02d}.csv, ~{len(todo) / max(n, 1):.0f} rows each")
