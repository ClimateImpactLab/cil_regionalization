# Cluster environment

This directory is for running the full pipeline runs on the
University of Chicago RCC cluster (midway3). It is not the normal way
to install this package; that is `pip install`, described in the main
README. Most readers never need anything here.

The environment it builds lives at
`/project/cil/home_dirs/rcc/envs/segment_weights_py311` (a shared group
path on that cluster), pinned and built fresh with uv rather than
layered over the group's shared conda stack.

## Why fresh rather than layered

A layered venv inherits whatever the shared stack becomes, so a rebuild
months later is not the environment that was validated. Mixing
conda-built and pip-built geospatial packages broke during validation
(conflicting PROJ databases between a conda pyproj and a pip rasterio).
And the shared stack was missing pieces this repo needs
(`exactextract` in some environments, the BigQuery client in others),
so completeness depended on which shared environment happened to be
active.

The pins in `requirements-cluster.txt` are the versions the full test
suite passed against, including pandas 3.0.3 and numpy 2.4.6.
`requirements-cluster.freeze.txt`, written by the setup script after the
suite passes, is the fully resolved snapshot including transitive
dependencies; that file is the rebuild contract.

## Build

    bash envs/setup_cluster_env.sh

The script creates the venv on Python 3.11, installs `exactextract`
first in its own step (the one package that may lack a platform wheel;
the script fails immediately and says so if that is where it breaks),
installs the pins, installs `segment_weights` editable without
re-resolving dependencies, runs an import check, runs the full test
suite, and captures the freeze file. It refuses to overwrite an
existing environment.

## Rebuild

Remove the environment directory and rerun the script. For an exact
rebuild, install from `requirements-cluster.freeze.txt` instead of
`requirements-cluster.txt` (same script flow, swap the file). If pins
ever change, the suite run at the end of the script is the gate: a
build whose suite does not pass is not an environment, and the freeze
file should only ever be committed from a passing build.

## Scope

One environment runs the whole repo: weight generation (local backend),
the application and statistics stages, the Monte Carlo pipeline with
NetCDF leaves, and the offline BigQuery tests (the client libraries are
included so the 54 offline tests run instead of skipping; real BigQuery
runs additionally need credentials). Dask packages are installed but
the dask code paths remain unvalidated; their tests stay marker-skipped
by default.
