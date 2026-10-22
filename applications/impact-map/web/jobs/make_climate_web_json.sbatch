#!/bin/bash
#SBATCH --partition=cil
#SBATCH --account=cil
#SBATCH --job-name=make_climate_web_json
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --output=make_climate_web_json_%j.log

# Writes the 120 climate documents into the published tree: 5 variables
# x 4 emissions scenarios x 3 admin levels, in each of web/ and
# web-lite/. Roughly 74 MB and 37 MB respectively.
#
# Strictly additive. The converter only creates or replaces files under
# <out>/climate_*, so the mortality ADM2 documents already sitting in
# web/ and web-lite/ are untouched and both products rsync to Box
# together afterwards.
#
# Memory is set by the widest read: the ADM2 ensemble reduced to the
# population weight is about 2.1 million rows, plus one document's
# payload at a time. 32G has substantial headroom. Runtime is dominated
# by the 20 ADM2 documents at roughly half a minute each.

set -euo pipefail

/project/cil/home_dirs/rcc/envs/cil_regionalization_py311/bin/python \
  /project/cil/home_dirs/scadavidsanchez/repos/cil_regionalization/examples/montecarlo/runs/make_climate_web_json.py \
  --root /project/cil/home_dirs/scadavidsanchez/repos/climate-and-damages-aggregations/examples/montecarlo/runs/climate_days \
  --out /project/cil/gcp/impact_map/web
