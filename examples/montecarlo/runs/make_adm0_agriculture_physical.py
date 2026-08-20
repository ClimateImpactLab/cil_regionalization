"""ADM0 physical agriculture from the adm1 reduced frames, per crop.

The physical effect is an intensive quantity (log kg/Ha), so ADM0 is
the crop-area-weighted mean of the adm1 values, not their sum. Crop
area is static, so the weights come straight from the adm1 artifact:
each target's area is the sum of its sources' allocated crop area.
Impact regions stay within one country and both sides use the same
weight file, so this regroup is exact.

Targets with no cropland for a crop are absent from the adm1 frame
already; a country with no cropland for the crop is therefore absent
here too, not zero. Each crop has its own set of covered countries.

Run after the adm1 pipeline runs for the same rcp; covers all six
crops in one pass:

    python make_adm0_agriculture_physical.py --rcp rcp85
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from cil_regionalization.stats import pooled_statistics

BASE = Path("/project/cil/gcp/impact_map/agriculture")
WEIGHTS = ("/project/cil/home_dirs/scadavidsanchez/repos/"
           "climate-and-damages-aggregations/data/weights")
SMME = "/project/cil/gcp/climate/SMME-weights/{rcp}_SMME_weights.tsv"
CROPS = ["cassava", "corn", "rice", "sorghum", "soy", "wheat"]
QUANTILES = [0.05, 0.17, 0.25, 0.5, 0.75, 0.83, 0.95]


def crop_adm0(crop: str, rcp: str, smme: pd.DataFrame) -> None:
    reduced_path = BASE / crop / "adm1" / f"{crop}_physical_log_{rcp}.reduced.parquet"
    reduced = pd.read_parquet(reduced_path)

    art = pd.read_parquet(
        f"{WEIGHTS}/ir_adm1_croparea_{crop}/weights.parquet"
    ).dropna(subset=["croparea_raw"])
    area = (art.groupby(["ISO", "ID_1"], sort=False)["croparea_raw"]
            .sum().reset_index().rename(columns={"croparea_raw": "area"}))

    n = len(reduced)
    frame = reduced.merge(area, on=["ISO", "ID_1"], validate="many_to_one")
    if len(frame) != n:
        raise ValueError(f"{crop}: {n - len(frame)} reduced rows have no "
                         f"area in the artifact")

    keys = ["ISO", "window", "batch", "gcm", "iam", "ssp"]
    frame["wx"] = frame["rebased"] * frame["area"]
    adm0 = frame.groupby(keys, sort=False)[["wx", "area"]].sum().reset_index()
    adm0["rebased"] = adm0["wx"] / adm0["area"]
    adm0 = adm0.drop(columns=["wx", "area"])

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
    out = BASE / crop / "adm0"
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{crop}_physical_log_{rcp}"
    stats.to_parquet(out / f"{stem}.parquet", index=False)
    (out / f"{stem}.manifest.json").write_text(json.dumps(
        {"derived_from": str(reduced_path),
         "method": "crop-area-weighted mean of the adm1 reduced frame by "
                   "ISO before pooling; area from the adm1 crop artifact",
         "units": "log kg/Ha, full adaptation minus histclim",
         "coverage": "countries with no cropland for this crop are absent, "
                     "not zero; each crop has its own set of covered units",
         "window_note": "last window ends at 2098; NEX-GDDP lacks "
                        "precipitation for some models after that",
         "quantiles": QUANTILES,
         "model_weights": SMME.format(rcp=rcp)}, indent=2))
    print(f"wrote {out}/{stem}.parquet rows={len(stats)} "
          f"countries={stats['ISO'].nunique()}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rcp", choices=["rcp45", "rcp85"], required=True)
    args = parser.parse_args()
    smme = pd.read_csv(SMME.format(rcp=args.rcp), sep="\t")[["model", "weight"]]
    smme["key"] = smme["model"].str.replace("*", "", regex=False).str.lower()
    for crop in CROPS:
        crop_adm0(crop, args.rcp, smme)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
