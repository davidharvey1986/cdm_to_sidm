#!/bin/bash
# Production on bonsai: writes one SLURM file per chunk (same style as remake_full20) and submits them.
# 25 jobs, each ~80 halo/snap/model rows x 3 projections (binning, ~2-3 min per row expected). Run from create_obs/ AFTER the test job looks right:
#   bash match_fix/velmaps/submit_velmaps.sh            # write + submit
#   DRY_RUN=1 bash match_fix/velmaps/submit_velmaps.sh  # write the .slurm files only
set -e
cd /srv/astro/projects/dark_skies/zooms/scratch_code/zooms/runs/utils/create_obs
OUT=/srv/astro/projects/dark_skies/zooms/cluster_postage_stamps/veldisp_dm_2mpc_20kpc_bin
for c in match_fix/velmaps/chunk_*.csv; do
  i=$(basename $c .csv | sed 's/chunk_//')
  f=match_fix/velmaps/velmaps_${i}.slurm
  cat > $f <<EOS
#!/bin/bash -l
#SBATCH --job-name=velmaps_${i}
#SBATCH --time=12:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH -p public-cpu
#SBATCH --mem 64G
#SBATCH --exclude=cpu003
#SBATCH --output=match_fix/velmaps/velmaps_${i}_%j.out
  module load GCCcore/11.3.0
  module load Python/3.10.4

cd /srv/astro/projects/dark_skies/zooms/scratch_code/zooms/runs/utils/create_obs
python match_fix/velmaps/vel_maps.py --table ${c} --outroot ${OUT}
EOS
  if [ -z "${DRY_RUN:-}" ]; then sbatch $f; fi
done
