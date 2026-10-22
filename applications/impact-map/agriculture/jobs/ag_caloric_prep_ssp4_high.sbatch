#!/bin/bash
#SBATCH --partition=cil
#SBATCH --account=cil
#SBATCH --job-name=ag_caloric_prep_ssp4_high
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=ag_caloric_prep_ssp4_high_%j.log

# Caloric yield draws, incadapt, SSP4, high IAM. One parquet of
# window-mean calorie changes per (batch, rcp) plus a manifest, in a
# per-scenario output directory; the warming-level cut, statistics,
# and spatial aggregation run downstream once the plan's two open
# definitions are settled.

/project/cil/home_dirs/rcc/envs/cil_regionalization_py311/bin/python /project/cil/home_dirs/scadavidsanchez/repos/cil_regionalization/examples/montecarlo/runs/run_caloric_prep.py --ssp SSP4 --iam high --workers 15
