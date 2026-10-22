#!/bin/bash
#SBATCH --job-name=cdays_scen
#SBATCH --account=cil
#SBATCH --partition=cil
#SBATCH --array=0-64%6
#SBATCH --time=06:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=2
#SBATCH --output=logs/cdays_scen_%A_%a.log

source /project/cil/home_dirs/rcc/envs/cil_regionalization_py311/bin/activate
export SSL_CERT_FILE=$(python -c "import certifi; print(certifi.where())")
export REQUESTS_CA_BUNDLE=$SSL_CERT_FILE
cd /project/cil/home_dirs/scadavidsanchez/repos/climate-and-damages-aggregations/examples/montecarlo/runs

python climate_days_daily.py --mode scen \
  --task-id "$SLURM_ARRAY_TASK_ID" \
  --levels adm0,adm1,adm2 --out climate_days --grid-out climate_grid
