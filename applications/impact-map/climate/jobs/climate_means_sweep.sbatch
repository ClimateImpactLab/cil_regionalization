#!/bin/bash
#SBATCH --job-name=clim_sweep
#SBATCH --account=cil
#SBATCH --partition=cil
#SBATCH --array=0-64%8
#SBATCH --time=01:00:00
#SBATCH --mem=12G
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/clim_sweep_%A_%a.log

source /project/cil/home_dirs/rcc/envs/cil_regionalization_py311/bin/activate
cd /project/cil/home_dirs/scadavidsanchez/repos/climate-and-damages-aggregations/examples/montecarlo/runs

python climate_means_monthly.py \
  --task-id "$SLURM_ARRAY_TASK_ID" \
  --levels adm0,adm1,adm2 \
  --out climate_means
