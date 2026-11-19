"""Re-match every zoom with match_halo.py and write rematch.csv.

usage: python run_rematch.py <base_sets.pkl> <out.csv> [halo ...]
"""
import sys, time, pickle
import csv
from match_halo import match_halo

MODELS = ["SIDM0.01", "SIDM0.05", "SIDM0.1", "SIDM0.2"]
SNAPS = [7, 8, 9, 10]

base = pickle.load(open(sys.argv[1], "rb"))
halos = sys.argv[3:] or sorted(base)
rows, t0 = [], time.time()
for n, h in enumerate(halos):
    try:
        rows += match_halo(h, MODELS, SNAPS, base[h])
    except Exception as e:  # keep going; record the failure
        print(f"ERROR {h}: {e!r}", flush=True)
        rows.append(dict(halo=h, model="ALL", status=f"error: {e!r}"[:200]))
    print(f"{n+1}/{len(halos)} {h} {time.time()-t0:.0f}s", flush=True)
keys = list(dict.fromkeys(k for r in rows for k in r))
with open(sys.argv[2], 'w', newline='') as f:
    w = csv.DictWriter(f, keys); w.writeheader(); w.writerows(rows)
