#!/bin/bash
# Production: one job per model x snapshot = 4 x 4 = 16 jobs, each reads its particle data once and writes all
# 300 clusters x 3 line-of-sight axes. Run on a COSMA login node AFTER the test checks out:
#   bash /cosma7/data/dp004/dc-harv1/bahamas_velmaps/code/submit_all.sh            # write + submit
#   DRY_RUN=1 bash /cosma7/data/dp004/dc-harv1/bahamas_velmaps/code/submit_all.sh  # write the .slurm files only
set -e
R=/cosma7/data/dp004/dc-harv1/bahamas_velmaps
mkdir -p $R/logs $R/jobs
for model in CDM+baryons SIDM0.1+baryons SIDM0.3+baryons SIDM1+baryons; do
  for snap in 029 030 031 032; do
    f=$R/jobs/bvm_${model}_${snap}.slurm
    cat > $f <<EOS
#!/bin/bash -l
#SBATCH --job-name=bvm_${model}_${snap}
#SBATCH -p cosma7
#SBATCH -A dp004
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=200G
#SBATCH --time=04:00:00
#SBATCH --output=$R/logs/bvm_${model}_${snap}_%j.out
module load python
cd $R
python3 code/bahamas_vel_maps.py --model ${model} --snap ${snap} --catdir $R/catalogs --out $R/maps_v2
EOS
    if [ -z "${DRY_RUN:-}" ]; then sbatch $f; fi
  done
done
