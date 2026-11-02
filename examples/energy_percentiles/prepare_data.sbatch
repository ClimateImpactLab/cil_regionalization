#!/bin/bash
#SBATCH --job-name=energy-pct-example-data
#SBATCH --account=cil
#SBATCH --partition=cil
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=energy_pct_example_data_%j.log

# Rebuilds examples/energy_percentiles/data/ and checks it against the
# Impact Map. Submit from the repository root, with a python that has
# netCDF4, pandas, pyarrow and geopandas; the library comes from this
# checkout.
#
#   CILREG_PYTHON=/path/to/python sbatch examples/energy_percentiles/prepare_data.sbatch
#
# Measured on 31 October 2026 (job 61914026): 2:34 and 2.8 GB peak.

set -euo pipefail
: "${CILREG_PYTHON:?set CILREG_PYTHON to a python with netCDF4, pandas, pyarrow and geopandas}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

/usr/bin/time -v "$CILREG_PYTHON" examples/energy_percentiles/prepare_data.py
