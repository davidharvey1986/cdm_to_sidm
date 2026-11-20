"""Chunk tables for baryon_maps.py: rows of match_fix/full_20kpc_nn100.csv (the table behind the mass and velocity maps)
for the SIDM models in MODELS, plus the CDM rows of the same halo/snap (the chain inputs). Rows of one halo/snap stay
together (CDM first), chunks interleaved so each mixes models and snapshots.
usage (from create_obs/): python match_fix/velmaps/make_baryon_chunks.py [--n 5] [--models SIDM0.1,SIDM0.2]"""
import csv, os, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=5); ap.add_argument("--models", default="SIDM0.1,SIDM0.2"); a = ap.parse_args()
rows = list(csv.DictReader(open(os.path.join(os.path.dirname(HERE), "full_20kpc_nn100.csv")))); M = a.models.split(",")
keep = {(r["halo"], r["snap"]) for r in rows if r["model"] in M}
groups = {}
for r in rows:
    if (r["halo"], r["snap"]) in keep and (r["model"] in M or r["model"] == "CDM"):
        groups.setdefault((r["halo"], r["snap"]), []).append(r)
G = [sorted(g, key=lambda r: r["model"] != "CDM") for _, g in sorted(groups.items())]
for f in os.listdir(HERE):
    if f.startswith("baryon_chunk_") and f.endswith(".csv"): os.remove(os.path.join(HERE, f))
for i in range(a.n):
    with open(f"{HERE}/baryon_chunk_{i:02d}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, rows[0].keys()); w.writeheader(); [w.writerows(g) for g in G[i::a.n]]
n = sum(len(g) for g in G)
print(f"{n} rows ({len(G)} halo/snaps; models CDM + {M}) -> {a.n} chunks of ~{n // a.n} rows (~{n / a.n * 0.25:.0f} h each at 15 min/row)")
