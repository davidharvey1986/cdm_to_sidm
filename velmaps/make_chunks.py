"""Split match_fix/full_20kpc_nn100.csv (2000 rows: 100 halos x 4 snaps x 5 models; the table that made
tot_2mpc_20kpc_nn_100) into N chunks for vel_maps.py. Rows of one halo/snap stay together; chunks are interleaved
so each mixes models and snapshots (similar run times).
usage (from create_obs/): python match_fix/velmaps/make_chunks.py [N=25]"""
import csv, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); N = int(sys.argv[1]) if len(sys.argv) > 1 else 25
rows = list(csv.DictReader(open(os.path.join(os.path.dirname(HERE), "full_20kpc_nn100.csv"))))
for i in range(N):
    with open(f"{HERE}/chunk_{i:02d}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, rows[0].keys()); w.writeheader(); w.writerows(rows[i::N])
print(f"{len(rows)} rows -> {N} chunks of ~{len(rows) // N}")
