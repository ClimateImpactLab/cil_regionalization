"""Percent yield products converted from the log products at the end.

Replaces the percent-space-aggregated products, whose construction was
flawed: they converted each region and draw to percent before any
averaging, and the exponential's convexity both shifted the median
target upward and let winsorized log tails explode into unbounded
percent tails (districts at thousands of percent). The sector's own
construction, confirmed against the code that produced the paper's
tables, aggregates and pools in log space and converts only at the
end:

    pct = 100 * (exp(v) * exp(0.5 * residvcv) - 1) - baseline

where v is a statistic of the log product and baseline is the
target's own 2001-2010 mean of the converted per-year statistic, per
statistic column. The recentring matters: log-space rebasing zeroes
the baseline in logs but not in converted space (a few points on the
mean, tens of points on upper quantiles), so the baseline-year
statistics come from dedicated pipeline runs (the *_base* configs)
and are subtracted here.

Monotonicity makes the conversion exact for quantiles; for the mean
it reproduces the paper's object (the converted mean of logs), which
is deliberately not the mean of converted draws.

rcp85 statistics use the agriculture sector's model weights: the SMME
table with ACCESS1-0 (absent from the trees) zeroed and its weight
moved uniformly to the five other 0.036-group models, matching the
published extractions rather than renormalizing proportionally.

Run after the log product and baseline runs for a given rcp:

    python run_percent_from_logs.py --rcp rcp85
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path("/project/cil/gcp/impact_map/agriculture")
IR_BASE = Path("/project/cil/gcp/impact_map/impact_region/agriculture")
CSVV_DIR = Path("/project/cil/home_dirs/egrenier/repos/agriculture/1_code/"
                "3_projections/2_run_projection/projection/csvv")
CROPS = ["cassava", "corn", "rice", "sorghum", "soy", "wheat"]
# the plotting code takes wheat's residual variance from wheat_spring
CSVVS = {"cassava": "cassava-110221", "corn": "corn-160221",
         "rice": "rice-160221", "sorghum": "sorghum-160221",
         "soy": "soy-160221", "wheat": "wheat_spring-280823"}
BASELINE_YEARS = [str(y) for y in range(2001, 2011)]
KEY_COLUMNS = {"hierid", "ISO", "ID_1", "ID_2", "rcp", "iam", "ssp",
               "window", "statistic"}


def read_residvcv(crop: str) -> float:
    path = CSVV_DIR / f"{CSVVS[crop]}.csvv"
    lines = path.read_text().splitlines()
    return float(lines[lines.index("residvcv") + 1])


def convert(v: pd.Series, bias: float) -> pd.Series:
    return 100 * (np.exp(v) * bias - 1)


def value_column(frame: pd.DataFrame) -> str:
    candidates = [c for c in frame.columns if c not in KEY_COLUMNS]
    if len(candidates) != 1:
        raise ValueError(f"expected one value column, found {candidates}")
    return candidates[0]


def load_baseline(directory: Path, crop: str, level: str, rcp: str,
                  bias: float) -> pd.DataFrame:
    """The recentring constant per target, scenario and statistic: the
    mean over 2001-2010 of the converted per-year statistics."""
    if level in ("ir", "adm2"):
        stems = [f"{crop}_physical_log_basea_{rcp}",
                 f"{crop}_physical_log_baseb_{rcp}"]
    else:
        stems = [f"{crop}_physical_log_base_{rcp}"]
    frames = [pd.read_parquet(directory / f"{s}.parquet") for s in stems]
    base = pd.concat(frames, ignore_index=True)
    missing = set(BASELINE_YEARS) - set(base["window"].astype(str))
    if missing:
        raise ValueError(f"{crop} {level} {rcp}: baseline years missing "
                         f"{sorted(missing)}")
    valcol = value_column(base)
    base["converted"] = convert(base[valcol].astype("float64"), bias)
    group = [c for c in base.columns
             if c in KEY_COLUMNS and c not in ("window", "rcp")]
    const = (base.groupby(group, sort=False, observed=True)["converted"]
             .mean().rename("baseline").reset_index())
    counts = base.groupby(group, sort=False, observed=True)["converted"].size()
    if (counts != len(BASELINE_YEARS)).any():
        raise ValueError(f"{crop} {level} {rcp}: a target lacks some "
                         f"baseline years")
    return const


def convert_product(crop: str, level: str, rcp: str) -> None:
    directory = IR_BASE / crop if level == "ir" else BASE / crop / level
    bias = float(np.exp(0.5 * read_residvcv(crop)))
    stats = pd.read_parquet(directory / f"{crop}_physical_log_{rcp}.parquet")
    valcol = value_column(stats)
    const = load_baseline(directory, crop, level, rcp, bias)

    merged = stats.merge(const, on=[c for c in const.columns if c != "baseline"],
                         how="left")
    if len(merged) != len(stats):
        raise ValueError(f"{crop} {level} {rcp}: baseline merge changed "
                         f"the row count")
    if merged["baseline"].isna().any():
        n = int(merged["baseline"].isna().sum())
        raise ValueError(f"{crop} {level} {rcp}: {n} rows lack a baseline")
    merged["pct"] = convert(merged[valcol].astype("float64"), bias) - merged["baseline"]
    out = merged.drop(columns=[valcol, "baseline"])

    stem = f"{crop}_physical_percent_{rcp}"
    out.to_parquet(directory / f"{stem}.parquet", index=False)
    manifest = {
        "source": str(directory / f"{crop}_physical_log_{rcp}.parquet"),
        "baseline_source": "the *_base* pipeline runs: per-year statistics "
                           "for 2001-2010 at the same targets",
        "construction": "100 * (exp(statistic of logs) * exp(0.5*residvcv) "
                        "- 1), recentred on the target's own 2001-2010 mean "
                        "of the converted per-year statistic, per statistic "
                        "column: the sector's published ordering, converted "
                        "at the very end",
        "residvcv_bias": bias,
        "replaces": "a percent-space-aggregated product whose ordering was "
                    "flawed: converting per region and draw before "
                    "averaging shifted the median target by several points "
                    "and let winsorized log tails explode into unbounded "
                    "percent tails. Quantiles convert exactly "
                    "(monotonicity); the mean here is the converted mean "
                    "of logs, the paper's object, deliberately not the "
                    "mean of converted draws.",
        "model_weights_note": "rcp85 statistics use the agriculture "
                              "sector's reweighted SMME table (ACCESS1-0 "
                              "zeroed, its mass moved uniformly to the "
                              "five 0.036-group models), matching the "
                              "published extractions; rcp45 conventions "
                              "are identical either way",
        "units": "percentage points of yield relative to the target's "
                 "2001-2010 baseline",
        "coverage": "identical to the log product: targets with no "
                    "cropland for this crop are absent",
    }
    (directory / f"{stem}.manifest.json").write_text(
        json.dumps(manifest, indent=2))
    print(f"wrote {directory}/{stem}.parquet rows={len(out)}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rcp", choices=["rcp45", "rcp85"], required=True)
    args = parser.parse_args()
    for crop in CROPS:
        for level in ("ir", "adm0", "adm1", "adm2"):
            convert_product(crop, level, args.rcp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
