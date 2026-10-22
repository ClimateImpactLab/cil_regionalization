#!/bin/bash
#SBATCH --partition=cil
#SBATCH --account=cil
#SBATCH --job-name=web_json_ag_percent
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --output=web_json_ag_percent_%j.log

# Regenerate the web JSON for the six *_physical_percent products from
# the corrected 2026-09-29 parquets. The published Aug 24 documents were
# built before the log-space aggregation fix and carry values up to six
# orders of magnitude out of range.
#
# REPLACES PUBLISHED OUTPUT. 216 documents per tree, 432 in total across
# web/ and web-lite/, every one of which already exists in Box. Unit
# counts, periods and statistics are unchanged between the published and
# corrected parquets, so this is a value-for-value replacement: no new
# filenames, no orphans.
#
# A bare make_web_json.py run cannot be used: 78 of the 180 adm-level
# parquets in the tree carry product names absent from its PRODUCTS
# registry, and the first file in sorted order is one of them. Staging a
# root holding only the wanted parquets sidesteps the registry.
#
# The agriculture tree is one level deeper than mortality --
# agriculture/<crop>/<level>/ rather than mortality/<level>/ -- but the
# discovery code only requires that a parquet's immediate parent be named
# adm0, adm1 or adm2; rglob handles the depth. The stage mirrors
# <crop>/<level>/ so the job log reads like the real tree. The stage path
# must contain no path component equal to "web", which the discovery
# filter excludes.

set -euo pipefail

SRC=/project/cil/gcp/impact_map/agriculture
STAGE=/project/cil/home_dirs/scadavidsanchez/scratch/ag_percent_fix
OUT=/project/cil/gcp/impact_map/web
PY=/project/cil/home_dirs/rcc/envs/cil_regionalization_py311/bin/python
SCRIPT=/project/cil/home_dirs/scadavidsanchez/repos/cil_regionalization/examples/montecarlo/runs/make_web_json.py

CROPS="cassava corn rice sorghum soy wheat"
LEVELS="adm0 adm1 adm2"
RCPS="rcp45 rcp85"

rm -rf "$STAGE"
for crop in $CROPS; do
  for lv in $LEVELS; do
    mkdir -p "$STAGE/$crop/$lv"
    for rcp in $RCPS; do
      f="$SRC/$crop/$lv/${crop}_physical_percent_${rcp}.parquet"
      test -e "$f" || { echo "missing input: $f" >&2; exit 1; }
      ln -s "$f" "$STAGE/$crop/$lv/"
    done
  done
done

staged=$(find "$STAGE" -name '*.parquet' | wc -l)
test "$staged" -eq 36 || { echo "staged $staged parquets, expected 36" >&2; exit 1; }
echo "staged $staged parquets under $STAGE"

# --out fixes the full tree; the lite tree follows as <out>-lite with the
# default statistics (mean,q17,q50,q83), matching the published set.
"$PY" "$SCRIPT" --root "$STAGE" --out "$OUT"

rm -rf "$STAGE"

for crop in $CROPS; do
  echo "$crop full: $(ls -1 "$OUT/${crop}_physical_percent" | wc -l)"
  echo "$crop lite: $(ls -1 "${OUT}-lite/${crop}_physical_percent" | wc -l)"
done
