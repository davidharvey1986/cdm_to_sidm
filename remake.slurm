#!/bin/bash -l
#SBATCH --job-name=remake_MODEL
#SBATCH --time=24:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH -p public-cpu
#SBATCH --mem 70G
#SBATCH --exclude=cpu003
#SBATCH --output=match_fix/remake_MODEL_%j.out
  module load GCCcore/11.3.0
  module load Python/3.10.4

cd /srv/astro/projects/dark_skies/zooms/scratch_code/zooms/runs/utils/create_obs
python match_fix/remake_images.py MODEL
