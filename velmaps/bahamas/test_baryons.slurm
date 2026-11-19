#!/bin/bash -l
#SBATCH --job-name=bbm_test
#SBATCH -p cosma7
#SBATCH -A dp004
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=/cosma7/data/dp004/dc-harv1/bahamas_velmaps/logs/bbm_test_%j.out
# Test: gas + stellar maps, CDM+baryons, snapshot 032 (z=0), first 5 clusters, all three line-of-sight axes.
module load python
cd /cosma7/data/dp004/dc-harv1/bahamas_velmaps/code
python3 bahamas_baryon_maps.py --model CDM+baryons --snap 032 --catdir /cosma7/data/dp004/dc-harv1/bahamas_velmaps/catalogs --out /cosma7/data/dp004/dc-harv1/bahamas_velmaps/test_baryons --ncl 5
