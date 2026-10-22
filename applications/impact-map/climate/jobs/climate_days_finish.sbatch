#!/bin/bash
#SBATCH --job-name=cdays_fin
#SBATCH --account=cil
#SBATCH --partition=cil
#SBATCH --time=04:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=2
#SBATCH --output=cdays_finish_%j.log

source /project/cil/home_dirs/rcc/envs/cil_regionalization_py311/bin/activate
cd /project/cil/home_dirs/scadavidsanchez/repos/climate-and-damages-aggregations/examples/montecarlo/runs

python climate_days_finish.py --src climate_days --levels adm0,adm1,adm2
