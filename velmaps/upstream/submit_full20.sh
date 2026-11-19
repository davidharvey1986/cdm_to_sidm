#!/bin/bash
# Regenerate the whole 20 kpc catalogue with nn=100 smoothing (current
# darkskies_images.py), keeping each stamp's halo (full_20kpc_nn100.csv).
# Splits the 2000 entries round-robin into NCHUNK jobs and submits them.
# Output: cluster_postage_stamps/tot_2mpc_20kpc_nn_100. Resumable: finished
# stamps are skipped, so rerunning this resubmits only what is left.
NCHUNK=${1:-30}
cd /srv/astro/projects/dark_skies/zooms/scratch_code/zooms/runs/utils/create_obs
python3 - "$NCHUNK" <<'PY'
import csv, sys
n = int(sys.argv[1])
rows = list(csv.DictReader(open("match_fix/full_20kpc_nn100.csv")))
for i in range(n):
    with open(f"match_fix/remake_full20/chunk_{i:02d}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, list(rows[0])); w.writeheader(); w.writerows(rows[i::n])
print(len(rows), "entries in", n, "chunks of", len(rows[0::n]), "or fewer")
PY
for ((i = 0; i < NCHUNK; i++)); do
    c=$(printf %02d $i)
    sed "s/CHUNK/$c/g" match_fix/remake_full20/remake_full20.slurm > match_fix/remake_full20/full20_$c.slurm
    sbatch match_fix/remake_full20/full20_$c.slurm
done
