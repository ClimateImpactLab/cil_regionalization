#!/bin/bash
#SBATCH --job-name=cmip6-example-data
#SBATCH --account=cil
#SBATCH --partition=cil
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:45:00
#SBATCH --output=/scratch/midway3/%u/cmip6_example_data_%j.log

# Rebuilds examples/cmip6/data/ and checks the daily files against the
# per-cell fields production wrote. Reads GDPCIR from the Planetary
# Computer, so the node needs outbound network. The log goes to your
# Midway3 scratch, not the repository. Submit from the
# repository root, with a python that has netCDF4, xarray, zarr,
# adlfs, planetary_computer, pystac_client, pandas, pyarrow and
# geopandas; the library comes from this checkout.
#
#   CILREG_PYTHON=/path/to/python sbatch examples/cmip6/prepare_data.sbatch

set -euo pipefail
: "${CILREG_PYTHON:?set CILREG_PYTHON to a python with the packages listed above}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
# the system CA store is too thin for the Planetary Computer's async reads
export SSL_CERT_FILE=$("$CILREG_PYTHON" -c "import certifi; print(certifi.where())")
export REQUESTS_CA_BUNDLE=$SSL_CERT_FILE

cd examples/cmip6
/usr/bin/time -v "$CILREG_PYTHON" prepare_data.py
