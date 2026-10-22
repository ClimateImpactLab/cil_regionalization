"""Convert the CMIP6 climate ensemble parquets to Impact Map web JSON.

Companion to make_web_json.py, which does the same job for the damage
statistics parquets. The two share the published contract through
``cil_regionalization.web_json`` and nothing else, because the source
schemas have almost nothing in common: the damage parquets carry one
product per file with separate ISO / ID_1 / ID_2 columns and a long
``statistic`` dimension, while the climate ensemble stacks five
variables and two weights in one file per admin level, keyed by a
pipe-joined region id, with the statistics as columns.

Why this is a separate script rather than entries in that script's
PRODUCTS registry: the registry only parameterises metadata text. The
conversion itself groups by ``["iam", "ssp"]`` and indexes the model
count by rcp, so a climate product would have to defeat the grouping,
the filename template and the metadata builder all at once. That is a
rewrite of the function, not a registry entry.

The scenario collision
----------------------
The published filename is ``<base>_<level>_<rcp>_<ssp>_<iam>``, where
ssp is the socioeconomic pathway (SSP2/SSP3/SSP4) and iam the income
scenario (high/low). The climate variables have neither. Their
``ssp245`` is an emissions scenario -- the CMIP6 analogue of rcp45 --
so it belongs in the rcp slot, and the two socioeconomic slots are
written as the literal ``na``:

    climate_tas_annual_adm2_ssp245_na_na.json.gz

Keeping five fields means a consumer splitting from the right lands on
the same dimensions it would for a damage product, instead of slicing
through the variable name and getting plausible garbage with no error.
``na`` is distinguishable from every real value in those slots, so a
consumer validating against a whitelist fails loudly rather than
reading a wrong scenario quietly.

The metadata block records the truth rather than the sentinel: every
key of the damage ``scenario`` block is preserved and explicitly null,
with the emissions scenario alongside under its own name.

Ensemble size differs by scenario
---------------------------------
This is the unweighted GDPCIR ensemble, not SMME weighted, because no
CMIP6 SMME weight table exists. Its size differs by scenario -- 25
models historical, 23 ssp245, 20 ssp370, 22 ssp585 -- so a comparison
across scenarios is not a comparison over one ensemble. ``n_models``
is published as its own metadata field for that reason, not only in
the prose.

Writes are strictly additive: see `_check_additive_target`.

Run on the cluster via sbatch/make_climate_web_json.sbatch.
"""
from __future__ import annotations

import argparse
import datetime
from pathlib import Path

import pandas as pd

from cil_regionalization.web_json import (
    ID_SEPARATOR,
    LITE_STATISTICS,
    QUANTILE_DESCRIPTIONS,
    check_ids_resolve,
    lite_document,
    round_significant,
    write_document,
)


ROOT = Path("/project/cil/home_dirs/scadavidsanchez/repos/"
            "climate-and-damages-aggregations/examples/montecarlo/runs/"
            "climate_days")
WEB = Path("/project/cil/gcp/impact_map/web")
NAMES = Path(__file__).resolve().parent / "gadm20_names.csv"

LEVELS = ("adm0", "adm1", "adm2")

# Every published product directory carries this prefix. It namespaces
# climate away from the damage products in a flat tree, and it is what
# `_check_additive_target` enforces.
PRODUCT_PREFIX = "climate_"

# The socioeconomic slots the climate variables do not have. Not a real
# value in either slot, so a whitelist check rejects it.
ABSENT = "na"

# The methodology specifies population weighting; the ensemble also
# carries area weights as an internal diagnostic, which are not
# published.
PUBLISHED_WEIGHT = "pop"

# The ensemble's region column joins the GADM id fields with this; the
# front end's composite id joins them with ID_SEPARATOR. Confirmed
# against id_fields in the adm2 weights manifest rather than assumed.
SOURCE_KEY_SEPARATOR = "|"

# Source column -> published key, in the order they appear in a
# document. The one rename is q5 -> q05: the published damage
# documents spell the 5th percentile q05, and a climate document
# spelling it q5 would be the only file in the tree where the same
# statistic has a different key.
STATISTIC_COLUMNS: tuple[tuple[str, str], ...] = (
    ("mean", "mean"),
    ("min", "min"),
    ("q5", "q05"),
    ("q17", "q17"),
    ("q25", "q25"),
    ("q50", "q50"),
    ("q75", "q75"),
    ("q83", "q83"),
    ("q95", "q95"),
    ("max", "max"),
)

MEAN_DESCRIPTION = "unweighted ensemble mean"

EMISSIONS_LABEL = {
    "historical": "historical",
    "ssp245": "SSP2-4.5",
    "ssp370": "SSP3-7.0",
    "ssp585": "SSP5-8.5",
}

METHOD_REFERENCE = "https://github.com/ClimateImpactLab/cil_regionalization"

SOURCE_GRID = ("0.25 degree global grid, GDPCIR daily tasmax and tasmin, "
               "exact-fraction cell coverage")

AGGREGATION = ("population-weighted mean over the 0.25 degree cells "
               "intersecting the unit (intensive; weights normalized per "
               "destination unit)")

# Thresholds and the (tasmax + tasmin) / 2 construction come from
# cil_regionalization.bcsd_io.derive_daily, which is what produced
# these fields; the archive carries no daily tas.
VARIABLES = {
    "tas_annual": {
        "units": "degrees Fahrenheit",
        "description": "Annual mean daily temperature, the mean over the "
                       "365-day year of (tasmax + tasmin) / 2.",
    },
    "tas_jja": {
        "units": "degrees Fahrenheit",
        "description": "June-July-August mean daily temperature, the mean "
                       "over JJA days of (tasmax + tasmin) / 2.",
    },
    "tas_djf": {
        "units": "degrees Fahrenheit",
        "description": "December-January-February mean daily temperature, "
                       "the mean over DJF days of (tasmax + tasmin) / 2, "
                       "taking December from the preceding year.",
    },
    "days_above_95F": {
        "units": "days per year",
        "description": "Count of days per year with daily maximum "
                       "temperature above 95F.",
    },
    "days_below_32F": {
        "units": "days per year",
        "description": "Count of days per year with daily minimum "
                       "temperature below 32F.",
    },
}

IDENTITY_COLUMNS = ("weight", "scenario", "window", "region", "variable",
                    "n_models", "level")


def product_base(variable: str) -> str:
    """Published product directory and filename stem for `variable`."""
    return f"{PRODUCT_PREFIX}{variable}"


def document_filename(variable: str, level: str, scenario: str) -> str:
    """Five-field published name with the socioeconomic slots marked absent.

    The emissions scenario goes in the rcp slot because that is what it
    is. See "The scenario collision" in the module docstring.
    """
    return (f"{product_base(variable)}_{level}_{scenario}"
            f"_{ABSENT}_{ABSENT}.json.gz")


def _check_additive_target(base: str) -> None:
    """Refuse to write outside this converter's own product directories.

    The output tree is shared. The mortality ADM2 documents written
    today sit in sibling directories under the same root and are what
    the next rsync to Box carries, alongside whatever else accumulates
    there. This converter only ever creates or replaces files inside
    ``<out>/climate_*`` and never removes anything, so a rerun -- or a
    mistyped ``--out`` -- cannot damage another product. The check is
    here rather than implied by the naming so that it survives someone
    later changing how names are built.
    """
    if not base.startswith(PRODUCT_PREFIX):
        raise ValueError(
            f"make_climate_web_json: refusing to write product {base!r}: "
            f"this converter writes only {PRODUCT_PREFIX}* directories, "
            f"because the output tree holds other products that are live"
        )


def _ensemble_sentence(n_models: int, scenario: str) -> str:
    where = ("over the historical period" if scenario == "historical"
             else f"under {EMISSIONS_LABEL[scenario]}")
    return (f"GDPCIR CMIP6 ensemble, {n_models} climate models {where}, "
            f"unweighted: no CMIP6 SMME weight table exists, so every "
            f"model counts equally and the ensemble size differs by "
            f"scenario")


def _statistics_block() -> dict[str, str]:
    return {
        published: (MEAN_DESCRIPTION if published == "mean"
                    else QUANTILE_DESCRIPTIONS[published])
        for _, published in STATISTIC_COLUMNS
    }


def read_level(root: Path, level: str, known_ids) -> pd.DataFrame:
    """Load one admin level, reduced to the published weight and validated.

    Every check here fails the run rather than dropping rows. A climate
    document that is quietly short a scenario, a variable or a set of
    units is indistinguishable downstream from one that is complete,
    because the front end draws what it is given.
    """
    path = root / f"ensemble_{level}.parquet"
    if not path.exists():
        raise ValueError(
            f"make_climate_web_json: {path} does not exist; --root should "
            f"be the directory holding ensemble_<level>.parquet"
        )
    frame = pd.read_parquet(path)

    source_columns = [s for s, _ in STATISTIC_COLUMNS]
    missing = [c for c in (*IDENTITY_COLUMNS, *source_columns)
               if c not in frame.columns]
    if missing:
        raise ValueError(
            f"make_climate_web_json: {path.name} is missing columns "
            f"{missing} (got {list(frame.columns)})"
        )

    weights = sorted(frame["weight"].astype(str).unique())
    if PUBLISHED_WEIGHT not in weights:
        raise ValueError(
            f"make_climate_web_json: {path.name} has no "
            f"{PUBLISHED_WEIGHT!r} weight to publish (got {weights})"
        )
    frame = frame[frame["weight"].astype(str) == PUBLISHED_WEIGHT].copy()

    levels = sorted(frame["level"].astype(str).unique())
    if levels != [level]:
        raise ValueError(
            f"make_climate_web_json: {path.name} declares level(s) "
            f"{levels}, expected [{level!r}]; the file and its name disagree"
        )

    unknown_variables = sorted(set(frame["variable"].astype(str))
                               - set(VARIABLES))
    if unknown_variables:
        raise ValueError(
            f"make_climate_web_json: {path.name} carries variables "
            f"{unknown_variables} that have no entry in VARIABLES; add "
            f"their units and description rather than publishing them bare"
        )

    unknown_scenarios = sorted(set(frame["scenario"].astype(str))
                               - set(EMISSIONS_LABEL))
    if unknown_scenarios:
        raise ValueError(
            f"make_climate_web_json: {path.name} carries scenarios "
            f"{unknown_scenarios} with no emissions label; they would be "
            f"published with the scenario slot unexplained"
        )

    key = ["region", "variable", "scenario", "window"]
    duplicated = int(frame.duplicated(key).sum())
    if duplicated:
        raise ValueError(
            f"make_climate_web_json: {path.name} has {duplicated} duplicate "
            f"rows on {key}; one row per combination is assumed when "
            f"building the per-window payload"
        )

    null_counts = {c: int(frame[c].isna().sum()) for c in source_columns}
    nulls = {c: n for c, n in null_counts.items() if n}
    if nulls:
        raise ValueError(
            f"make_climate_web_json: {path.name} has null statistics "
            f"{nulls}; JSON has no null-float spelling the front end reads, "
            f"so these would ship as the literal NaN and break strict parsers"
        )

    frame["_id"] = (frame["region"].astype(str)
                    .str.replace(SOURCE_KEY_SEPARATOR, ID_SEPARATOR,
                                 regex=False))
    check_ids_resolve(frame["_id"], known_ids,
                      context=f"{path.name} at {level}")
    return frame


def build_document(level: str, variable: str, scenario: str,
                   block: pd.DataFrame, names: dict[str, str],
                   generated: str) -> tuple[dict, dict]:
    """Metadata and data for one variable, level and emissions scenario."""
    model_counts = sorted(block["n_models"].unique().tolist())
    if len(model_counts) != 1:
        raise ValueError(
            f"make_climate_web_json: {variable} {level} {scenario} mixes "
            f"ensemble sizes {model_counts}; the document declares one "
            f"n_models and cannot describe more than one"
        )
    n_models = int(model_counts[0])

    source_columns = [s for s, _ in STATISTIC_COLUMNS]
    published = [p for _, p in STATISTIC_COLUMNS]
    block = block.sort_values(["_id", "window"])

    data: dict = {}
    for uid, unit in block.groupby("_id", sort=True, observed=True):
        values: dict = {}
        rows = unit[["window", *source_columns]].itertuples(index=False,
                                                            name=None)
        for window, *statistics in rows:
            values[str(window).replace("-", "_")] = {
                key: round_significant(float(v))
                for key, v in zip(published, statistics)
            }
        data[uid] = {
            "id": uid,
            "name": names[uid],
            "country_iso": uid.split(ID_SEPARATOR)[0],
            "values": values,
        }

    windows = sorted(str(w) for w in block["window"].unique())
    metadata = {
        "source": "Climate Impact Lab",
        "variable": variable,
        "units": VARIABLES[variable]["units"],
        "description": VARIABLES[variable]["description"],
        "method_reference": METHOD_REFERENCE,
        # Every key of the damage scenario block is kept and explicitly
        # null, so a reader indexing .ssp gets nothing rather than the
        # emissions scenario misread as a socioeconomic pathway.
        "scenario": {
            "emissions": EMISSIONS_LABEL[scenario],
            "rcp": None,
            "ssp": None,
            "model": None,
        },
        "climate_framework": "CMIP6 GDPCIR",
        "source_grid": SOURCE_GRID,
        "ensemble": _ensemble_sentence(n_models, scenario),
        "n_models": n_models,
        "aggregation": AGGREGATION,
        "boundaries": {"source": "GADM", "version": "2.0"},
        "statistics": _statistics_block(),
        "admin_level": level,
        "merge_key": "id",
        "periods": [w.replace("-", "_") for w in windows],
        "count": len(data),
        "generated": generated,
    }
    return metadata, data


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path, default=WEB)
    parser.add_argument("--names", type=Path, default=NAMES)
    parser.add_argument("--levels", nargs="+", default=list(LEVELS),
                        choices=list(LEVELS))
    parser.add_argument(
        "--lite-stats", default=",".join(LITE_STATISTICS),
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

    out_root = args.out
    lite_stats = [s for s in args.lite_stats.split(",") if s]
    lite_out = None
    if lite_stats:
        lite_out = (args.lite_out if args.lite_out is not None
                    else out_root.parent / f"{out_root.name}-lite")

    names = pd.read_csv(args.names)
    generated = datetime.date.today().isoformat()

    # Name what is already in the tree and will be left alone, so a run
    # that is about to write next to live products says so out loud.
    existing = sorted(p.name for p in out_root.glob("*")
                      if p.is_dir() and not p.name.startswith(PRODUCT_PREFIX))
    if existing:
        print(f"leaving {len(existing)} existing product(s) untouched: "
              f"{', '.join(existing)}", flush=True)

    written = 0
    for level in args.levels:
        lookup = (names[names["level"] == level]
                  .set_index("id")["name"].to_dict())
        frame = read_level(args.root, level, lookup)
        grouped = frame.groupby(["variable", "scenario"], sort=True,
                                observed=True)
        for (variable, scenario), block in grouped:
            base = product_base(str(variable))
            _check_additive_target(base)
            metadata, data = build_document(level, str(variable),
                                            str(scenario), block, lookup,
                                            generated)
            filename = document_filename(str(variable), level, str(scenario))
            if not args.lite_only:
                write_document(out_root / base, filename, metadata, data)
                written += 1
            if lite_out is not None:
                meta_l, data_l = lite_document(
                    metadata, data, lite_stats,
                    drop_names=args.lite_drop_names)
                write_document(lite_out / base, filename, meta_l, data_l)
                written += 1
            print(f"{level} {variable} {scenario}: {len(data)} units, "
                  f"{len(metadata['periods'])} period(s), "
                  f"{metadata['n_models']} models", flush=True)

    print(f"done: {written} web files", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
