"""ADM0 physical mortality from the adm1 reduced frame.

The adm1 pipeline runs keep their reduced frame (one row per target,
window, and sample member). Summing it by ISO before pooling gives
ADM0 exactly, since impact regions do not cross national borders and
both sides use the same weight file; verified to 2e-16 on a real
leaf. Statistics cannot be summed, which is why this works from the
reduced frame and not from the statistics file.

Run after the adm1 pipeline run for the same rcp:

    python make_adm0_physical.py --rcp rcp85
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from cil_regionalization.stats import pooled_statistics

BASE = Path("/project/cil/gcp/impact_map/mortality")
SMME = "/project/cil/gcp/climate/SMME-weights/{rcp}_SMME_weights.tsv"
QUANTILES = [0.05, 0.17, 0.25, 0.5, 0.75, 0.83, 0.95]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rcp", choices=["rcp45", "rcp85"], required=True)
    args = parser.parse_args()

    reduced_path = BASE / "adm1" / f"mortality_physical_{args.rcp}.reduced.parquet"
    reduced = pd.read_parquet(reduced_path)
    keys = ["ISO", "window", "batch", "gcm", "iam", "ssp"]
    adm0 = reduced.groupby(keys, sort=False)["rebased"].sum().reset_index()

    smme = pd.read_csv(SMME.format(rcp=args.rcp), sep="\t")[["model", "weight"]]
    smme["key"] = smme["model"].str.replace("*", "", regex=False).str.lower()
    adm0 = adm0.assign(
        key=adm0["gcm"].str.replace("surrogate_", "", regex=False).str.lower()
    ).merge(
        smme[["key", "weight"]].rename(columns={"weight": "model_weight"}),
        on="key",
    ).drop(columns="key")

    stats = pooled_statistics(
        adm0,
        sample_dims=["batch", "gcm"],
        value_col="rebased",
        quantiles=QUANTILES,
        weight_col="model_weight",
    )
    out = BASE / "adm0"
    out.mkdir(parents=True, exist_ok=True)
    stem = f"mortality_physical_{args.rcp}"
    stats.to_parquet(out / f"{stem}.parquet", index=False)
    (out / f"{stem}.manifest.json").write_text(json.dumps(
        {"derived_from": str(reduced_path),
         "method": "ISO sum of the adm1 reduced frame before pooling",
         "units": "deaths/year",
         "quantiles": QUANTILES,
         "model_weights": SMME.format(rcp=args.rcp)}, indent=2))
    print(f"wrote {out}/{stem}.parquet rows={len(stats)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
