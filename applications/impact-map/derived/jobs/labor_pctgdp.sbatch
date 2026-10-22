#!/bin/bash
#SBATCH --partition=cil
#SBATCH --account=cil
#SBATCH --job-name=labor_pctgdp
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=labor_pctgdp_%j.log

# Labor disutility as a share of GDP: at impact region level the
# physical product times 0.00487 (the denominator cancels), at ADM
# levels dollars over unit GDP built at matching 2005 PPP vintage
# from the projection system's own income and population series.

/project/cil/home_dirs/rcc/envs/cil_regionalization_py311/bin/python /project/cil/home_dirs/scadavidsanchez/repos/cil_regionalization/examples/montecarlo/runs/run_labor_pctgdp.py
