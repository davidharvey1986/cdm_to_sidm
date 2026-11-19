#!/bin/bash
# Production gas + stellar maps: one job per model x snapshot = 4 x 4 = 16 jobs (as the velocity maps), each reads the
# gas and star particles once and writes all 300 clusters x 3 line-of-sight axes to maps_baryons/.
# Test (12376194, CDM z=0, 5 clusters): 82 s, max RSS 40 GB. Run on a COSMA login node AFTER the test (test_baryons.slurm) checks out:
#   bash /cosma7/data/dp004/dc-harv1/bahamas_velmaps/code/submit_baryons.sh            # write + submit
#   DRY_RUN=1 bash /cosma7/data/dp004/dc-harv1/bahamas_velmaps/code/submit_baryons.sh  # write the .slurm files only
set -e
R=/cosma7/data/dp004/dc-harv1/bahamas_velmaps
mkdir -p $R/logs $R/jobs
for model in CDM+baryons SIDM0.1+baryons SIDM0.3+baryons SIDM1+baryons; do
  for snap in 029 030 031 032; do
    f=$R/jobs/bbm_${model}_${snap}.slurm
    cat > $f <<EOS
#!/bin/bash -l
#SBATCH --job-name=bbm_${model}_${snap}
#SBATCH -p cosma7
#SBATCH -A dp004
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=$R/logs/bbm_${model}_${snap}_%j.out
module load python
cd $R/code
python3 bahamas_baryon_maps.py --model ${model} --snap ${snap} --catdir $R/catalogs --out $R/maps_baryons
EOS
    if [ -z "${DRY_RUN:-}" ]; then sbatch $f; fi
  done
done
