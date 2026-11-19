#!/bin/bash -l
#SBATCH --job-name=bvm_test
#SBATCH -p cosma7
#SBATCH -A dp004
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=200G
#SBATCH --time=02:00:00
#SBATCH --output=/cosma7/data/dp004/dc-harv1/bahamas_velmaps/logs/bvm_test_%j.out
# Test (v2, matched selection): CDM+baryons, snapshot 032 (z=0), first 5 clusters, all three line-of-sight axes.
module load python
cd /cosma7/data/dp004/dc-harv1/bahamas_velmaps
python3 code/bahamas_vel_maps.py --model CDM+baryons --snap 032 --catdir /cosma7/data/dp004/dc-harv1/bahamas_velmaps/catalogs --out /cosma7/data/dp004/dc-harv1/bahamas_velmaps/test_v2 --ncl 5
