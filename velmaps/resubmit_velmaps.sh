#!/bin/bash
# Finish the darkskies velocity maps on bonsai: the 25 production jobs (submit_velmaps.sh) hit the 12 h limit
# (~15 min per row, not 2-3), leaving ~1/4 of the rows (mostly SIDM0.1 / SIDM0.2 haloes) unmade.
# 1) missing_chunks.py lists the rows whose x/y/z maps are not all written and splits them into N chunk tables;
# 2) one SLURM file per chunk, same settings as before except the job name; vel_maps.py skips maps that exist and now
#    logs a failed row and carries on (FAILED lines at the end of each log).
# With ~500 rows left, N=20 gives ~25 rows (~6 h) per job inside the 12 h limit.
# Writes the .slurm files only; they are submitted only with SUBMIT=1. Run from anywhere:
#   bash match_fix/velmaps/resubmit_velmaps.sh            # write tables + .slurm files, print what would be submitted
#   SUBMIT=1 bash match_fix/velmaps/resubmit_velmaps.sh   # ... and submit them
#   N=10 TIME=24:00:00 bash match_fix/velmaps/resubmit_velmaps.sh
set -e
cd /srv/astro/projects/dark_skies/zooms/scratch_code/zooms/runs/utils/create_obs
OUT=/srv/astro/projects/dark_skies/zooms/cluster_postage_stamps/veldisp_dm_2mpc_20kpc_bin
N=${N:-20}; TIME=${TIME:-12:00:00}
module load GCCcore/11.3.0 Python/3.10.4 2>/dev/null || true
python match_fix/velmaps/missing_chunks.py --outroot ${OUT} --n ${N}
rm -f match_fix/velmaps/velmaps_miss_*.slurm
for c in match_fix/velmaps/missing_*.csv; do
  i=$(basename $c .csv | sed 's/missing_//')
  f=match_fix/velmaps/velmaps_miss_${i}.slurm
  cat > $f <<EOS
#!/bin/bash -l
#SBATCH --job-name=velmiss_${i}
#SBATCH --time=${TIME}
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH -p public-cpu
#SBATCH --mem 64G
#SBATCH --exclude=cpu003
#SBATCH --output=match_fix/velmaps/velmaps_miss_${i}_%j.out
  module load GCCcore/11.3.0
  module load Python/3.10.4

cd /srv/astro/projects/dark_skies/zooms/scratch_code/zooms/runs/utils/create_obs
python match_fix/velmaps/vel_maps.py --table ${c} --outroot ${OUT}
EOS
  if [ -n "${SUBMIT:-}" ]; then sbatch $f; else echo "would submit $f ($(($(wc -l < $c) - 1)) rows, ${TIME})"; fi
done
