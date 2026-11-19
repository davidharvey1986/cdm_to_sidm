#!/bin/bash -l
#SBATCH --job-name=velmaps_test
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem 64G
#SBATCH -p public-cpu
#SBATCH --exclude=cpu003
#SBATCH --output=match_fix/velmaps/velmaps_test_%j.out
# Test: first row of chunk_00 (one halo/snap/model), all three projections, to a separate test directory.
  module load GCCcore/11.3.0
  module load Python/3.10.4
cd /srv/astro/projects/dark_skies/zooms/scratch_code/zooms/runs/utils/create_obs
python match_fix/velmaps/vel_maps.py --table match_fix/velmaps/chunk_00.csv --only 1 \
    --outroot /srv/astro/projects/dark_skies/zooms/cluster_postage_stamps/veldisp_dm_2mpc_20kpc_bin_test
