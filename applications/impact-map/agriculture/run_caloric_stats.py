"""Collapse the caloric draws to mappable statistics, one scenario per run.

Reads the draws run_caloric_prep.py wrote for one (ssp, iam) scenario
and produces the SMME-weighted mean and the standard seven quantiles
per impact region, crop, and window, keyed by rcp: one statistics
parquet per rcp beside the draws. This is the rcp-keyed product the
draws support today; the warming-level cut and any spatial aggregation
remain downstream questions and take nothing from this file.

Both value columns are kept, because they answer different questions:
``cal_total_change`` (calories, extensive) and ``cal_per_ha_change``
(calories per hectare, a rate). The six-crop 'all' rate is undefined
in regions planting none of the six crops; those draws are NaN and
their identity groups are absent from the rate statistics while the
total statistics (zero change) remain, so the two columns do not share
exact coverage and the merge is outer.

Run on the cluster:

    python run_caloric_stats.py --ssp SSP2 --iam low
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from cil_regionalization.stats import pooled_statistics

ROOT = Path("/project/cil/gcp/impact_map/agriculture/calories")
SMME = "/project/cil/gcp/climate/SMME-weights/{rcp}_SMME_weights.tsv"
BATCHES = [f"batch{i}" for i in range(15)]
RCPS = ["rcp45", "rcp85"]
QUANTILES = [0.05, 0.17, 0.25, 0.5, 0.75, 0.83, 0.95]
VALUE_COLS = ["cal_per_ha_change", "cal_total_change"]


def smme_map(rcp: str) -> dict[str, float]:
    smme = pd.read_csv(SMME.format(rcp=rcp), sep="\t")
    keys = smme["model"].str.replace("*", "", regex=False).str.lower()
    return dict(zip(keys, smme["weight"].astype(float)))


def load_rcp(scen_dir: Path, rcp: str) -> pd.DataFrame:
    frames = [pd.read_parquet(scen_dir / f"draws_{b}_{rcp}.parquet")
              for b in BATCHES]
    df = pd.concat(frames, ignore_index=True)
    weights = smme_map(rcp)
    key = df["gcm"].str.replace("surrogate_", "", regex=False).str.lower()
    df["model_weight"] = key.map(weights)
    if df["model_weight"].isna().any():
        missing = sorted(df.loc[df["model_weight"].isna(), "gcm"].unique())
        raise ValueError(f"no SMME weight for {missing}")
    return df


def rcp_statistics(scen_dir: Path, rcp: str) -> tuple[pd.DataFrame, int]:
    df = load_rcp(scen_dir, rcp)
    n_members = df.groupby(["batch", "gcm"], sort=False, observed=True).ngroups
    pieces = []
    for col in VALUE_COLS:
        sub = df.dropna(subset=[col])[
            ["hierid", "crop", "window", "batch", "gcm", col, "model_weight"]]
        stats = pooled_statistics(
            sub,
            sample_dims=["batch", "gcm"],
            value_col=col,
            quantiles=QUANTILES,
            weight_col="model_weight",
        )
        pieces.append(stats)
    out = pieces[0].merge(
        pieces[1], on=["hierid", "crop", "window", "statistic"], how="outer")
    for c in out.columns:
        if isinstance(out[c].dtype, pd.CategoricalDtype):
            out[c] = out[c].astype(str)
    return out, n_members


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ssp", choices=["SSP2", "SSP3", "SSP4"],
                        default="SSP2")
    parser.add_argument("--iam", choices=["low", "high"], default="low")
    args = parser.parse_args()
    scen_dir = ROOT / f"draws_incadapt_{args.ssp.lower()}_{args.iam}"
    prep = json.loads((scen_dir / "draws.manifest.json").read_text())

    for rcp in RCPS:
        stats, n_members = rcp_statistics(scen_dir, rcp)
        stem = f"caloric_change_{rcp}"
        stats.to_parquet(scen_dir / f"{stem}.parquet", index=False)
        manifest = {
            "source": str(scen_dir),
            "ssp": args.ssp,
            "iam": f"{args.iam} ({'IIASA' if args.iam == 'low' else 'OECD Env-Growth'})",
            "rcp": rcp,
            "adaptation": "incadapt: this is not the paper's headline "
                          "adaptation object (fulladapt with costs); it is "
                          "the income-only adaptation scenario the analysis "
                          "plan requested",
            "validation": "no published figure reports caloric change per "
                          "impact region, so there is nothing to validate "
                          "this against; the calorie conversion and areas "
                          "are byte-identical to the valuation's inputs, "
                          "and the draw arithmetic was checked by hand "
                          "against the yield files",
            "statistics": "SMME-weighted mean and quantiles pooled over "
                          "batch x gcm, the same construction as every "
                          "other impact_map product",
            "n_members": n_members,
            "windows": prep["windows"],
            "window_note": prep["window_note"],
            "baseline": prep["baseline"],
            "units": prep["units"],
            "coverage": "cal_total_change covers every region and crop; "
                        "cal_per_ha_change for crop 'all' is absent where "
                        "a region plants none of the six crops (the rate "
                        "is undefined there), so the two columns do not "
                        "share exact coverage",
            "quantiles": QUANTILES,
            "model_weights": SMME.format(rcp=rcp),
        }
        (scen_dir / f"{stem}.manifest.json").write_text(
            json.dumps(manifest, indent=2))
        print(f"wrote {scen_dir}/{stem}.parquet rows={len(stats)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
