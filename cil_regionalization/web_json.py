"""The gzipped JSON documents the Impact Map front end reads, one per
product, admin level and scenario.

A document holds a metadata block and a data dict keyed by composite
GADM 2.0 unit id. This module gives every converter the rounding rule,
the gzip framing, the lite subsetting, and the checks that stop a
broken document before it reaches Box. Each converter builds its own
composite ids, from the columns its source has, joined by
`ID_SEPARATOR`. Writes only add or replace the one file asked for, and
nothing here removes, truncates or prunes anything else.
"""

# A publication tree accumulates products across runs, and its sibling
# directories are usually live, since they are what the next rsync to
# Box carries. There is no "clean the output directory" helper here on
# purpose, and adding one would be a mistake.
from __future__ import annotations

import gzip
import json
import math
from pathlib import Path
from typing import Iterable, Mapping


# The front end's composite unit id is the GADM 2.0 id fields joined by
# this separator: "{ISO}", "{ISO}.{ID_1}", "{ISO}.{ID_1}.{ID_2}".
ID_SEPARATOR = "."

# Values carry far fewer meaningful digits than a float64 prints, and
# the difference is most of the file. Four is what the published tree
# already uses; changing it silently rewrites every document.
SIGNIFICANT_DIGITS = 4

# The statistics the front end plots. Carrying the rest costs roughly
# half the file again, which is the whole reason the lite tree exists.
LITE_STATISTICS: tuple[str, ...] = ("mean", "q17", "q50", "q83")

# Shared spellings for the distribution statistics. `mean` is absent on
# purpose: how the ensemble mean was taken differs by product (SMME
# weighted for the damage projections, unweighted for the CMIP6 climate
# ensemble, which has no weight table), and a description there is a
# claim about method. Each converter supplies its own.
QUANTILE_DESCRIPTIONS: dict[str, str] = {
    "min": "ensemble minimum",
    "q05": "5th percentile",
    "q17": "17th percentile",
    "q25": "25th percentile",
    "q50": "median",
    "q75": "75th percentile",
    "q83": "83rd percentile",
    "q95": "95th percentile",
    "max": "ensemble maximum",
}

# What `--lite-drop-names` puts in the metadata in place of the per-unit
# name fields, so a consumer of a lite document is told where the names
# went rather than finding them simply missing.
NAMES_OMITTED = (
    "omitted; take display names from the GADM 2.0 boundary files via the id"
)


def round_significant(value: float) -> float:
    """`value` to `SIGNIFICANT_DIGITS`, leaving 0 and non-finite alone."""
    # `%g` rather than `round(value, n)`, since the magnitudes inside one
    # document span many orders, and deaths per year run from fractions
    # to hundreds of thousands across units. A fixed number of decimal
    # places would flatten the small units to zero or keep meaningless
    # digits on the large ones.
    if value == 0 or not math.isfinite(value):
        return value
    return float(f"{value:.{SIGNIFICANT_DIGITS}g}")


def check_ids_resolve(
    ids: Iterable[str], known: Iterable[str], *, context: str
) -> None:
    """Raises unless every id in `ids` appears in `known`, the ids of the
    GADM 2.0 name table. `context` labels the error."""
    # An id the name table does not carry means the converter's id rule
    # and the boundary layer have diverged. Published that way the unit
    # raises nothing downstream and fails to draw, so this stops the run
    # instead of shipping a map with holes.
    unknown = sorted(set(ids) - set(known))
    if unknown:
        raise ValueError(
            f"web_json: {context}: {len(unknown)} unit ids are not in the "
            f"GADM 2.0 name table (first: {unknown[:5]}); the id rule and "
            f"the boundary layer have diverged"
        )


def check_statistics_available(
    statistics: Iterable[str], available: Iterable[str], *, context: str
) -> None:
    """Raises unless every requested statistic is one the product has.
    `context` labels the error."""
    # A statistic the product never computed, if skipped, leaves the lite
    # document short a series and the front end draws an empty layer.
    have = set(available)
    missing = [s for s in statistics if s not in have]
    if missing:
        raise ValueError(
            f"web_json: {context}: requested statistics {missing} are not "
            f"in this product (available: {sorted(have)})"
        )


def lite_document(
    metadata: Mapping,
    data: Mapping,
    statistics: Iterable[str] = LITE_STATISTICS,
    *,
    drop_names: bool = False,
) -> tuple[dict, dict]:
    """A new (metadata, data) pair of the same document restricted to
    `statistics`, with the inputs left unchanged.

    `drop_names` also omits the per-unit `name` and `country_iso`
    fields, and the metadata's `names` then says where the names went.
    Raises if a statistic is not one the metadata lists.
    """
    # drop_names saves roughly another sixth of the file but changes the
    # unit shape the front end reads, so it is off by default.
    statistics = list(statistics)
    check_statistics_available(
        statistics, metadata["statistics"], context="lite document"
    )

    meta = dict(metadata)
    meta["statistics"] = {s: metadata["statistics"][s] for s in statistics}
    if drop_names:
        meta["names"] = NAMES_OMITTED

    keep = set(statistics)
    out: dict = {}
    for uid, unit in data.items():
        values = {
            window: {s: v for s, v in stats.items() if s in keep}
            for window, stats in unit["values"].items()
        }
        if drop_names:
            out[uid] = {"id": unit["id"], "values": values}
        else:
            out[uid] = {
                "id": unit["id"],
                "name": unit["name"],
                "country_iso": unit["country_iso"],
                "values": values,
            }
    return meta, out


def write_document(
    out_dir: str | Path, filename: str, metadata: Mapping, data: Mapping
) -> Path:
    """Writes one gzipped web document as `filename` in `out_dir` and
    returns its path.

    `out_dir` is created if absent. A document of the same name is
    replaced, so a rerun gives the same tree, and no other file is
    touched. The JSON has compact separators.
    """
    # Every sibling, including other products staged for the same rsync,
    # is left alone, as the comment at the top of the module says.
    # Whitespace is about a tenth of the uncompressed document and
    # survives gzip badly, hence the compact separators.
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / filename
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump({"metadata": dict(metadata), "data": data}, f, separators=(",", ":"))
    return path
