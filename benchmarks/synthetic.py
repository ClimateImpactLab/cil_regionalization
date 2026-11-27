"""Synthetic frames shaped like one leaf of the agriculture ADM2 run.

The source side has the shape of a real corn leaf as `read_netcdf_leaf`
returns it, which is 24,378 impact regions by 118 years, ordered by
year then region. The target side has the shape of the corn ADM2 crop
area weights, built from the counts in `weights_counts`. The values
and the weights are random.
"""

from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd

import cil_regionalization as cilreg
from benchmarks import weights_counts as counts
from cil_regionalization import SourceUnitPolicies
from cil_regionalization.apply import AppliedResult, WeightsArtifact
from cil_regionalization.schema import OutputSchema, SourceUnits, weight_column

WINDOWS = [(2020, 2039), (2040, 2059), (2080, 2098)]
YEARS = np.arange(1981, 2099, dtype=np.int32)
LABELS = {
    "batch": "batch7",
    "gcm": "surrogate_GFDL-ESM2G_11",
    "iam": "high",
    "ssp": "SSP3",
}
ID_FIELDS = ["ISO", "ID_1", "ID_2"]
N_SOURCE, N_TARGET = 24_378, sum(counts.COUNTRY_TARGETS)
WEIGHT = "croparea"

# BENCH_SCALE divides the leaf for the single-process benchmarks.
SCALE = int(os.environ.get("BENCH_SCALE", "1"))


# the country codes `region_ids` starts its ids with
_CODES = np.array(
    ["".join(chr(65 + c // k % 26) for k in (100, 10, 1)) for c in range(253)]
)


def _target_ids(n_target: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Id columns of ``n_target`` targets, numbered as GADM 2.0 numbers
    most countries. ID_1 and ID_2 restart at 1 in each country, and a
    level the country does not have is 0. A target's regions do not
    share its country code."""
    order = rng.permutation(len(_CODES))
    size = np.array(counts.COUNTRY_TARGETS)[order]
    adm1 = np.array(counts.COUNTRY_ADM1)[order]
    start = np.cumsum(size) - size
    t = np.arange(n_target) * N_TARGET // n_target
    c = np.searchsorted(start, t, side="right") - 1
    j = t - start[c]
    return {
        "ISO": _CODES[c],
        "ID_1": np.where(size[c] == 1, 0, j * adm1[c] // size[c] + 1),
        "ID_2": np.where(size[c] == adm1[c], 0, j + 1),
    }


def region_ids(n: int) -> np.ndarray:
    """Ids built like world-combo hierids and about as long as the real
    ones. Each is a three-letter code and dotted numbers, and about a
    fifth of them end in a 16 hex digit hash."""
    # the length is kept because string hashing costs by the character
    out = []
    for i in range(n):
        c = i * 253 // n
        code = "".join(chr(65 + c // k % 26) for k in (100, 10, 1))
        j = i - -(-c * n // 253)  # position within the code's block
        a, b = i % 37 + 1, i % 2003 + 1
        if i % 16 < 3:
            digest = f"R{i * 0x9E3779B97F4A7C15 % 2**64:016x}"
            pick = i // 16 % 20
            if pick == 0:
                tail = digest
            elif pick < 12:
                tail = f"{a}.{digest}"
            elif pick < 19:
                tail = f"{a}.{b}.{digest}"
            else:
                tail = f"{a}.{b}.{i}.{digest}"
        elif i % 25 == 0:
            tail = f"{j}"
        elif i % 2:
            tail = f"{a}.{100 + j}"
        else:
            tail = f"{a}.{b % 300}.{31 * j}"
        out.append(f"{code}.{tail}")
    return np.array(out)


def applied_leaf(scale: int = SCALE, seed: int = 0) -> pd.DataFrame:
    """A leaf as `apply_weights` returns it, one row per year for each
    target that has crop area, in the order of the weights file. The
    targets with none are not in it."""
    frame = weights_frame(scale)
    with_area = frame[weight_column(WEIGHT)].notna()
    targets = frame.loc[with_area, ID_FIELDS].drop_duplicates()
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            **{c: np.repeat(targets[c].to_numpy(), len(YEARS)) for c in ID_FIELDS},
            "year": np.tile(YEARS, len(targets)),
            "pct": rng.normal(size=len(targets) * len(YEARS)),
        }
    )


def source_leaf(scale: int = SCALE, seed: int = 0) -> pd.DataFrame:
    """A leaf as it is read, one row per year and impact region."""
    n_source = N_SOURCE // scale
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "year": np.repeat(YEARS, n_source),
            "hierid": np.tile(region_ids(n_source), len(YEARS)),
            "pct": rng.normal(size=n_source * len(YEARS)).astype(np.float32),
        }
    )


def _sizes(histogram: dict[int, int], scale: int) -> np.ndarray:
    """One size per unit of a `weights_counts` histogram, keeping every
    ``scale``-th unit."""
    sizes = np.repeat(list(histogram), list(histogram.values()))
    return sizes[scale // 2 :: scale]


def weights_frame(scale: int = SCALE) -> pd.DataFrame:
    """Crop area weights shaped like the corn ADM2 file.

    Almost every row is one of two cases, a target that draws on a
    single region or a region that feeds a single target. 27 of the
    real 51,216 rows are neither and are folded into those. Regions
    with no crop area carry NaN, which leaves 6,639 targets with no
    weight at full size. Rows are sorted by target then region, as the
    real file is.
    """
    n_source = N_SOURCE // scale
    rng = np.random.default_rng(1)

    def total(histogram: dict[int, int]) -> int:
        return sum(size * n for size, n in histogram.items())

    # regions that feed several targets, by the targets they feed
    shared_empty = _sizes(counts.TARGETS_PER_EMPTY_REGION, scale)
    shared = _sizes(
        {
            size: n - counts.TARGETS_PER_EMPTY_REGION.get(size, 0)
            for size, n in counts.TARGETS_PER_REGION.items()
        },
        scale,
    )
    # targets that draw on several regions, by the regions they draw on
    split = _sizes(counts.SOURCES_WITH_AREA, scale)
    split_empty = _sizes(counts.SOURCES_WITHOUT_AREA, scale)
    split_mixed = _sizes(counts.SOURCES_MIXED, scale)
    # A mixed target has at least one empty region and one with crop
    # area. The file's other empty rows are spread over the regions
    # each has beyond those two.
    n_mixed = sum(counts.SOURCES_MIXED.values())
    spread = counts.MIXED_EMPTY_ROWS - n_mixed
    room = total(counts.SOURCES_MIXED) - 2 * n_mixed
    mixed_empty = 1 + np.diff(np.cumsum(split_mixed - 2) * spread // room, prepend=0)
    # the regions left over pair with one target each
    n_pairs = n_source - (
        len(shared)
        + len(shared_empty)
        + split.sum()
        + split_empty.sum()
        + split_mixed.sum()
    )
    n_pairs_empty = min(
        n_pairs,
        (counts.EMPTY_SINGLE_TARGETS - total(counts.TARGETS_PER_EMPTY_REGION)) // scale,
    )

    def ones(n):
        return np.ones(n, dtype=np.int64)

    groups = [  # targets, regions, regions with no crop area
        (ones(n_pairs - n_pairs_empty), 1, 0),
        (ones(n_pairs_empty), 1, 1),
        (shared, 1, 0),
        (shared_empty, 1, 1),
        (1, split, 0),
        (1, split_empty, split_empty),
        (1, split_mixed, mixed_empty),
    ]
    n_targets, n_regions, n_empty = (
        np.concatenate(column)
        for column in zip(*(np.broadcast_arrays(*group) for group in groups))
    )
    # Targets and regions are numbered group by group in a shuffled
    # order, so the targets of a shared region sit next to each other
    # and neighbouring targets draw on neighbouring regions.
    order = rng.permutation(len(n_targets))
    n_targets, n_regions, n_empty = n_targets[order], n_regions[order], n_empty[order]

    def first(sizes: np.ndarray) -> np.ndarray:
        return np.cumsum(sizes) - sizes

    # one row per (target, region) pair of each group
    n_rows = np.maximum(n_targets, n_regions)
    group = np.repeat(np.arange(len(n_rows)), n_rows)
    step = np.arange(len(group)) - first(n_rows)[group]
    target = first(n_targets)[group] + np.where(n_targets[group] > 1, step, 0)
    nth_region = np.where(n_regions[group] > 1, step, 0)
    region = first(n_regions)[group] + nth_region
    empty = nth_region < n_empty[group]

    ids = _target_ids(int(n_targets.sum()), rng)
    raw = np.where(empty, np.nan, rng.uniform(0.1, 1.0, len(group)))
    frame = pd.DataFrame(
        {
            **{c: ids[c][target] for c in ID_FIELDS},
            "hierid": region_ids(n_source)[region],
            "croparea_raw": raw,
        }
    )
    area = frame.groupby(ID_FIELDS)["croparea_raw"].transform("sum")
    frame[weight_column(WEIGHT)] = frame["croparea_raw"] / area
    frame["croparea_method"] = np.where(empty, "nan", "native")
    return frame.sort_values(ID_FIELDS + ["hierid"], ignore_index=True)


def artifact(scale: int = SCALE) -> WeightsArtifact:
    """The weights of `weights_frame` as `apply_weights` takes them."""
    frame = weights_frame(scale)
    schema = OutputSchema(
        id_fields=tuple(ID_FIELDS),
        weight_names=(WEIGHT,),
        source_units=SourceUnits.from_string_ids(["hierid"]),
        normalization="per_destination",
    )
    return WeightsArtifact(
        frame=frame,
        schema=schema,
        regions_version="synthetic-adm2",
        source_version="synthetic-ir",
    )


def apply_result(art: WeightsArtifact, leaf: pd.DataFrame) -> AppliedResult:
    """Weight a leaf onto the targets, skipping the regions with no crop
    area. A target whose regions all have none is missing from the
    frame and listed in the result's targets."""
    return cilreg.apply_weights(
        art,
        leaf,
        kind="intensive",
        weight=WEIGHT,
        value_col="pct",
        data_version="synthetic-ir",
        policies=SourceUnitPolicies(on_zero_weight="skip"),
    )


def apply(art: WeightsArtifact, leaf: pd.DataFrame) -> pd.DataFrame:
    """The frame of `apply_result` without its share column, as the
    pipeline reduces it."""
    return apply_result(art, leaf).frame.drop(columns=["weight_share"])


def best(fn, repeat: int = 3):
    """Best wall time of ``repeat`` calls, and the last result."""
    seconds = float("inf")
    for _ in range(repeat):
        start = time.perf_counter()
        out = fn()
        seconds = min(seconds, time.perf_counter() - start)
    return seconds, out


def report(line: str) -> None:
    print(f"[benchmark] {line}", flush=True)
