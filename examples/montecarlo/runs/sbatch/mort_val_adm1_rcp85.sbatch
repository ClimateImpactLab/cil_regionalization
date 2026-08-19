#!/bin/bash
#SBATCH --partition=cil
#SBATCH --account=cil
#SBATCH --job-name=mort_val_adm1_rcp85
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --output=mort_val_adm1_rcp85_%j.log

/project/cil/home_dirs/rcc/envs/cil_regionalization_py311/bin/python /project/cil/home_dirs/scadavidsanchez/repos/cil_regionalization/examples/montecarlo/runs/run_valued_mortality.py --adm adm1 --rcp rcp85 --workers 16
