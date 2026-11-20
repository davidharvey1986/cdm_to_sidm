#!/bin/bash
# Darkskies gas + stellar maps on bonsai (baryon_maps.py), for the BAHAMAS-trained DM -> stars -> gas chain test.
# Builds the chunk tables (CDM + MODELS, default all four SIDM models) and one SLURM file per chunk. Writes only; submits
# only with SUBMIT=1. Each job bins 4 rows in parallel (--procs = cpus-per-task). TEST=1 writes (and with SUBMIT=1 submits) a single 1-row test job instead.
#   bash match_fix/velmaps/submit_baryon_maps.sh                 # tables + .slurm files, print what would be submitted
#   TEST=1 SUBMIT=1 bash match_fix/velmaps/submit_baryon_maps.sh # 1 test job (first row of chunk 00, ~15 min)
#   SUBMIT=1 bash match_fix/velmaps/submit_baryon_maps.sh        # N production jobs
set -e
cd /srv/astro/projects/dark_skies/zooms/scratch_code/zooms/runs/utils/create_obs
OUT=/srv/astro/projects/dark_skies/zooms/cluster_postage_stamps/baryon_2mpc_20kpc_bin
N=${N:-5}; TIME=${TIME:-48:00:00}; MODELS=${MODELS:-SIDM0.01,SIDM0.05,SIDM0.1,SIDM0.2}
module load GCCcore/11.3.0 Python/3.10.4 2>/dev/null || true
python match_fix/velmaps/make_baryon_chunks.py --n ${N} --models ${MODELS}
rm -f match_fix/velmaps/baryon_*.slurm
if [ -n "${TEST:-}" ]; then chunks=match_fix/velmaps/baryon_chunk_00.csv; ONLY="--only 1"; OUT=${OUT}_test; TIME=02:00:00
else chunks=$(ls match_fix/velmaps/baryon_chunk_*.csv); ONLY=""; fi
for c in $chunks; do
  i=$(basename $c .csv | sed 's/baryon_chunk_//'); [ -n "${TEST:-}" ] && i=test
  f=match_fix/velmaps/baryon_${i}.slurm
  cat > $f <<EOS
#!/bin/bash -l
#SBATCH --job-name=baryon_${i}
#SBATCH --time=${TIME}
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH -p public-cpu
#SBATCH --mem 96G
#SBATCH --exclude=cpu003
#SBATCH --output=match_fix/velmaps/baryon_${i}_%j.out
  module load GCCcore/11.3.0
  module load Python/3.10.4
cd /srv/astro/projects/dark_skies/zooms/scratch_code/zooms/runs/utils/create_obs
python match_fix/velmaps/baryon_maps.py --table ${c} --outroot ${OUT} ${ONLY}
EOS
  if [ -n "${SUBMIT:-}" ]; then sbatch $f; else echo "would submit $f ($(($(wc -l < $c) - 1)) rows, ${TIME})"; fi
done
