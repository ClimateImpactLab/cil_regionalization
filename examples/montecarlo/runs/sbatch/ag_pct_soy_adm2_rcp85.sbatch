#!/bin/bash
#SBATCH --partition=cil
#SBATCH --account=cil
#SBATCH --job-name=ag_pct_soy_adm2_rcp85
#SBATCH --time=36:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --output=ag_pct_soy_adm2_rcp85_%j.log

/project/cil/home_dirs/rcc/envs/cil_regionalization_py311/bin/python /project/cil/home_dirs/scadavidsanchez/repos/cil_regionalization/examples/montecarlo/runs/run_percent_agriculture.py --crop soy --adm adm2 --rcp rcp85 --workers 16
