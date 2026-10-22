#!/bin/bash
#SBATCH --partition=cil
#SBATCH --account=cil
#SBATCH --job-name=pct_gdp
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=pct_gdp_%j.log

# Percent-of-GDP twins for valued mortality and agriculture at every
# level, dividing by integration-econ-bc39.zarr with the window mean
# over window mean convention. Labor is deliberately excluded (2005
# PPP dollars against a 2019-dollar GDP); the manifests say why.

/project/cil/home_dirs/rcc/envs/cil_regionalization_py311/bin/python /project/cil/home_dirs/scadavidsanchez/repos/cil_regionalization/examples/montecarlo/runs/run_pct_gdp.py
