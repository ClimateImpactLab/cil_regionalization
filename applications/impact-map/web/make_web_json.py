"""Convert impact_map statistics parquets to the front end's JSON.

One gzipped file per product, admin level, RCP, SSP and IAM, in the
shape the Impact Map front end already reads: a metadata block and a
data dict keyed by composite GADM 2.0 unit id ({ISO}, {ISO}.{ID_1},
{ISO}.{ID_1}.{ID_2}). The id rule and the display names were verified
against the front end's own sample: every sample id resolves in the
GADM 2.0 dissolved layer with an exact name match. Names come from
gadm20_names.csv next to this script, committed so conversion does not
need the geometry.

Values are rounded to four significant digits before writing; with
gzip that takes an ADM2 file from ~25 MB to a few MB. The full set
under web/ carries every computed statistic and stays the complete
record. A second, lighter set under web-lite/ carries only the
statistics the front end plots (default mean, q17, q50, q83;
--lite-stats changes it without editing code), about half the size.

The script processes every statistics parquet it finds under the
impact_map root and prints what it skipped, so it can rerun as more
production jobs finish.

Run on the cluster:

    python make_web_json.py
"""
from __future__ import annotations

import argparse
import datetime
import gzip
import json
import math
from pathlib import Path

import pandas as pd

ROOT = Path("/project/cil/gcp/impact_map")
NAMES = Path(__file__).resolve().parent / "gadm20_names.csv"
IDENTITY = {"ISO", "ID_1", "ID_2", "rcp", "iam", "ssp", "window", "statistic"}
RCP_LABEL = {"rcp45": "RCP4.5", "rcp85": "RCP8.5"}
STAT_DESCRIPTIONS = {
    "mean": "SMME-weighted ensemble mean",
    "q05": "5th percentile", "q17": "17th percentile",
    "q25": "25th percentile", "q50": "median", "q75": "75th percentile",
    "q83": "83rd percentile", "q95": "95th percentile",
}

# Facts about what each product is. The ensemble model count differs
# by rcp because the trees do: mortality and labor lack
# surrogate_GFDL-ESM2G_06 under rcp45 only; agriculture additionally
# lacks ACCESS1-0 under rcp85 (skipped by the winsorization step).
AG_MODELS = {"rcp45": 32, "rcp85": 32}
ML_MODELS = {"rcp45": 32, "rcp85": 33}
ADM0_SUM = "ISO sum of the ADM1 sample members before pooling"
PRODUCTS = {
    "mortality_physical": {
        "variable": "mortality_deaths_change",
        "units": "deaths per year",
        "description": "Change in annual deaths from climate change, full "
                       "adaptation minus the histclim counterfactual, from "
                       "the 2021 mortality Monte Carlo projections.",
        "aggregation": "impact-region totals allocated to units by "
                       "population share (extensive sum)",
        "adm0": ADM0_SUM,
        "models": ML_MODELS,
    },
    "mortality_valued": {
        "variable": "mortality_damages",
        "units": "2019 USD per year",
        "description": "Monetized change in annual deaths (VSL, EPA "
                       "scaling), full adaptation minus histclim.",
        "aggregation": "impact-region totals allocated to units by "
                       "population share (extensive sum)",
        "adm0": ADM0_SUM,
        "models": ML_MODELS,
    },
    "labor_physical": {
        "variable": "labor_minutes_change",
        "units": "minutes per worker per day",
        "description": "Change in daily labor supply from climate change, "
                       "full adaptation minus histclim, from the 2025 labor "
                       "Monte Carlo projections.",
        "aggregation": "population-weighted mean (impact-region minutes "
                       "times population and population aggregate "
                       "separately; division at the unit)",
        "adm0": "population-weighted mean via ISO sums of numerator and "
                "denominator",
        "models": ML_MODELS,
    },
    "labor_valued": {
        "variable": "labor_damages",
        "units": "2005 PPP USD per year",
        "description": "Wage value of lost work time, full adaptation "
                       "minus histclim.",
        "aggregation": "impact-region totals allocated to units by "
                       "population share (extensive sum)",
        "adm0": ADM0_SUM,
        "models": ML_MODELS,
    },
    "physical_log": {
        "variable": "yield_log_change",
        "units": "log kg/Ha",
        "description": "Log yield change from climate change, full "
                       "adaptation minus histclim, winsorized Monte Carlo "
                       "trees. Units with no cropland for this crop are "
                       "absent; each crop has its own covered set.",
        "aggregation": "crop-area-weighted mean over impact regions with "
                       "cropland for this crop",
        "adm0": "crop-area-weighted mean of ADM1 members by ISO",
        "models": AG_MODELS,
    },
    "physical_percent": {
        "variable": "yield_percent_change",
        "units": "percentage points of yield relative to the unit's "
                 "2001-2010 average",
        "description": "Percent yield change from climate change: "
                       "exp(fulladapt - histclim) with the lognormal bias "
                       "correction from the crop's csvv residual variance, "
                       "recentered on 2001-2010, following the sector's "
                       "own conversion. Units with no cropland for this "
                       "crop are absent.",
        "aggregation": "crop-area-weighted mean over impact regions with "
                       "cropland for this crop",
        "adm0": "crop-area-weighted mean of ADM1 members by ISO",
        "models": AG_MODELS,
    },
    "agriculture_valued": {
        "variable": "agriculture_damages",
        "units": "2019 USD per year",
        "description": "Welfare change (consumer plus producer surplus) "
                       "from the six-crop constant-elasticity market "
                       "model, the delta-only object the agriculture paper "
                       "published; the no-climate-change counterfactual is "
                       "inside the valuation.",
        "aggregation": "market welfare redistributed to impact regions by "
                       "the valuation's weights, then allocated to units "
                       "by population share (extensive sum)",
        "adm0": ADM0_SUM,
        "models": AG_MODELS,
    },
}
CROPS = {"cassava", "corn", "rice", "sorghum", "soy", "wheat"}


def product_key(stem_base: str) -> str:
    """Registry key for a stem like corn_physical_log or labor_valued."""
    parts = stem_base.split("_", 1)
    if parts[0] in CROPS:
        return parts[1]
    return stem_base


def round_sig(v: float) -> float:
    if v == 0 or not math.isfinite(v):
        return v
    return float(f"{v:.4g}")


def composite_id(row, level: str) -> str:
    if level == "adm0":
        return row.ISO
    if level == "adm1":
        return f"{row.ISO}.{row.ID_1}"
    return f"{row.ISO}.{row.ID_1}.{row.ID_2}"


def write_doc(out_dir: Path, filename: str, metadata: dict, data: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with gzip.open(out_dir / filename, "wt", encoding="utf-8") as f:
        json.dump({"metadata": metadata, "data": data}, f,
                  separators=(",", ":"))


def lite_variant(metadata: dict, data: dict, stats: list[str],
                 drop_names: bool) -> tuple[dict, dict]:
    """The same document restricted to the requested statistics.

    The front end plots a subset of the percentiles; carrying all of
    them costs about 45 percent of the file. Dropping the name fields
    would save another ~15 percent but changes the unit shape the
    front end reads, so that is a flag, off by default.
    """
    missing = [s for s in stats if s not in metadata["statistics"]]
    if missing:
        raise ValueError(
            f"lite statistics {missing} are not in this product "
            f"(available: {sorted(metadata['statistics'])})"
        )
    meta = dict(metadata)
    meta["statistics"] = {s: metadata["statistics"][s] for s in stats}
    if drop_names:
        meta["names"] = ("omitted; take display names from the GADM 2.0 "
                         "boundary files via the id")
    out = {}
    for uid, unit in data.items():
        u = {
            "id": unit["id"],
            "values": {w: {s: v for s, v in vals.items() if s in stats}
                       for w, vals in unit["values"].items()},
        }
        if not drop_names:
            u = {"id": unit["id"], "name": unit["name"],
                 "country_iso": unit["country_iso"], "values": u["values"]}
        out[uid] = u
    return meta, out


def convert_file(path: Path, names: pd.DataFrame, out_root: Path,
                 generated: str, lite_out: Path | None = None,
                 lite_stats: list[str] | None = None,
                 lite_drop_names: bool = False,
                 lite_only: bool = False) -> int:
    level = path.parent.name
    stem = path.stem
    base, rcp = stem.rsplit("_", 1)
    key = product_key(base)
    meta_src = PRODUCTS[key]
    crop = base.split("_", 1)[0] if base.split("_", 1)[0] in CROPS else None

    frame = pd.read_parquet(path)
    value_col = [c for c in frame.columns if c not in IDENTITY][0]
    lookup = names[names["level"] == level].set_index("id")["name"].to_dict()

    frame = frame.copy()
    if level == "adm0":
        frame["_id"] = frame["ISO"]
    elif level == "adm1":
        frame["_id"] = frame["ISO"] + "." + frame["ID_1"].astype(str)
    else:
        frame["_id"] = (frame["ISO"] + "." + frame["ID_1"].astype(str)
                        + "." + frame["ID_2"].astype(str))
    unknown = sorted(set(frame["_id"]) - set(lookup))
    if unknown:
        raise ValueError(
            f"{path}: {len(unknown)} unit ids not in the GADM 2.0 name "
            f"table (first: {unknown[:5]}); the id rule is broken"
        )

    windows = sorted(frame["window"].unique())
    periods = [w.replace("-", "_") for w in windows]
    n_files = 0
    for (iam, ssp), block in frame.groupby(["iam", "ssp"], sort=True):
        data: dict = {}
        for uid, unit in block.groupby("_id", sort=True):
            values: dict = {}
            for window, wb in unit.groupby("window", sort=True):
                values[window.replace("-", "_")] = {
                    s: round_sig(float(v))
                    for s, v in zip(wb["statistic"], wb[value_col])
                }
            data[uid] = {
                "id": uid,
                "name": lookup[uid],
                "country_iso": uid.split(".")[0],
                "values": values,
            }
        stats_present = sorted(block["statistic"].unique())
        metadata = {
            "source": "Climate Impact Lab",
            "variable": meta_src["variable"],
            "units": meta_src["units"],
            "description": meta_src["description"],
            "method_reference":
                "https://github.com/ClimateImpactLab/cil_regionalization",
            "scenario": {"ssp": ssp, "rcp": RCP_LABEL[rcp], "model": iam},
            "climate_framework": "CMIP5 SMME",
            "impact_regions": 24378,
            "ensemble": (
                f"CMIP5 SMME ensemble, {meta_src['models'][rcp]} climate "
                f"models under {RCP_LABEL[rcp]}, 15 Monte Carlo batches, "
                f"SMME-weighted statistics"
            ),
            "aggregation": (meta_src["adm0"] if level == "adm0"
                            else meta_src["aggregation"]),
            "boundaries": {"source": "GADM", "version": "2.0"},
            "statistics": {s: STAT_DESCRIPTIONS.get(s, s)
                           for s in stats_present},
            "admin_level": level,
            "merge_key": "id",
            "periods": periods,
            "count": len(data),
            "generated": generated,
        }
        if crop is not None:
            metadata["crop"] = crop
        filename = f"{base}_{level}_{rcp}_{ssp}_{iam}.json.gz"
        if not lite_only:
            write_doc(out_root / base, filename, metadata, data)
            n_files += 1
        if lite_out is not None and lite_stats:
            meta_l, data_l = lite_variant(metadata, data, lite_stats,
                                          lite_drop_names)
            write_doc(lite_out / base, filename, meta_l, data_l)
            n_files += 1
    return n_files


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--names", type=Path, default=NAMES)
    parser.add_argument(
        "--lite-stats", default="mean,q17,q50,q83",
        help="statistics kept in the lite set (comma separated; "
             "empty disables the lite output)")
    parser.add_argument("--lite-out", type=Path, default=None)
    parser.add_argument(
        "--lite-drop-names", action="store_true",
        help="omit name and country_iso from lite units; saves ~15%% "
             "more but the front end must then take names from the "
             "boundary files")
    parser.add_argument(
        "--lite-only", action="store_true",
        help="write only the lite set, leaving existing full files "
             "untouched")
    args = parser.parse_args()
    out_root = args.out if args.out is not None else args.root / "web"
    lite_stats = [s for s in args.lite_stats.split(",") if s]
    lite_out = None
    if lite_stats:
        lite_out = (args.lite_out if args.lite_out is not None
                    else out_root.parent / f"{out_root.name}-lite")
    names = pd.read_csv(args.names)
    generated = datetime.date.today().isoformat()

    stats_files = sorted(
        p for p in args.root.rglob("*.parquet")
        if not p.name.endswith(".reduced.parquet")
        and p.parent.name in ("adm0", "adm1", "adm2")
        and "web" not in p.parts
    )
    written = 0
    for path in stats_files:
        n = convert_file(path, names, out_root, generated,
                         lite_out=lite_out, lite_stats=lite_stats,
                         lite_drop_names=args.lite_drop_names,
                         lite_only=args.lite_only)
        print(f"{path.relative_to(args.root)}: {n} files", flush=True)
        written += n
    print(f"done: {written} web files from {len(stats_files)} parquets")
    if not stats_files:
        print("nothing found; is the impact_map root right?")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
