#!/bin/bash
#SBATCH --job-name=cdays_hist
#SBATCH --account=cil
#SBATCH --partition=cil
#SBATCH --array=0-24%6
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=2
#SBATCH --output=logs/cdays_hist_%A_%a.log

source /project/cil/home_dirs/rcc/envs/cil_regionalization_py311/bin/activate
export SSL_CERT_FILE=$(python -c "import certifi; print(certifi.where())")
export REQUESTS_CA_BUNDLE=$SSL_CERT_FILE
cd /project/cil/home_dirs/scadavidsanchez/repos/climate-and-damages-aggregations/examples/montecarlo/runs

python climate_days_daily.py --mode hist \
  --task-id "$SLURM_ARRAY_TASK_ID" \
  --levels adm0,adm1,adm2 --out climate_days --grid-out climate_grid
