#!/bin/bash
#SBATCH --partition=cil
#SBATCH --account=cil
#SBATCH --job-name=ag_caloric_prep
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=ag_caloric_prep_%j.log

# Caloric yield draws for the analysis plan: incadapt, SSP2, IIASA.
# Writes one parquet of window-mean calorie changes per (batch, rcp)
# plus a manifest; the warming-level cut, statistics, and spatial
# aggregation run downstream of these draws once the plan's two open
# definitions are settled.

/project/cil/home_dirs/rcc/envs/cil_regionalization_py311/bin/python /project/cil/home_dirs/scadavidsanchez/repos/cil_regionalization/examples/montecarlo/runs/run_caloric_prep.py --workers 15
