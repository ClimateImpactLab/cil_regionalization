"""Synthetic frames shaped like one leaf of the agriculture ADM2 run.

The source side has the shape of a real corn leaf as `read_netcdf_leaf`
returns it, which is 24,378 impact regions by 118 years, ordered by
year then region. The target side, 40,000 ADM2 units, is not calibrated
to a real weights file yet. The values are random.
"""
from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd

import cil_regionalization as cilreg
from cil_regionalization import SourceUnitPolicies
from cil_regionalization.apply import WeightsArtifact
from cil_regionalization.schema import OutputSchema, SourceUnits

WINDOWS = [(2020, 2039), (2040, 2059), (2080, 2098)]
YEARS = np.arange(1981, 2099, dtype=np.int32)
LABELS = {
    "batch": "batch7",
    "gcm": "surrogate_GFDL-ESM2G_11",
    "iam": "high",
    "ssp": "SSP3",
}
ID_FIELDS = ["ISO", "ID_1", "ID_2"]
N_SOURCE, N_TARGET = 24_378, 40_000

# BENCH_SCALE divides the leaf for the single-process benchmarks.
SCALE = int(os.environ.get("BENCH_SCALE", "1"))


def _target_ids(t: np.ndarray, n_target: int) -> dict[str, np.ndarray]:
    iso = np.array([f"C{i:03d}" for i in range(250)])
    return {"ISO": iso[(t * 250) // n_target], "ID_1": (t % 60) + 1, "ID_2": t}


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
    """A leaf as `apply_weights` returns it, one row per target and
    year."""
    n_target = N_TARGET // scale
    rng = np.random.default_rng(seed)
    t = np.repeat(np.arange(n_target), len(YEARS))
    return pd.DataFrame(
        {
            **_target_ids(t, n_target),
            "year": np.tile(YEARS, n_target),
            "pct": rng.normal(size=len(t)),
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


def artifact(scale: int = SCALE) -> WeightsArtifact:
    """Crop area weights in which every target overlaps one to three
    impact regions and its weights sum to one."""
    n_source, n_target = N_SOURCE // scale, N_TARGET // scale
    rng = np.random.default_rng(1)
    n_parts = rng.integers(1, 4, n_target)
    t = np.repeat(np.arange(n_target), n_parts)
    first = (np.arange(n_target) * n_source // n_target)[t]
    offset = np.concatenate([np.arange(k) for k in n_parts])
    hierid = region_ids(n_source)
    frame = pd.DataFrame(
        {
            **_target_ids(t, n_target),
            "hierid": hierid[(first + offset) % n_source],
            "croparea_raw": rng.uniform(0.1, 1.0, len(t)),
            "croparea_method": "native",
        }
    )
    total = frame.groupby(ID_FIELDS)["croparea_raw"].transform("sum")
    frame["croparea" + "wt"] = frame["croparea_raw"] / total
    schema = OutputSchema(
        id_fields=tuple(ID_FIELDS),
        weight_names=("croparea",),
        source_units=SourceUnits.from_string_ids(["hierid"]),
        normalization="per_destination",
    )
    return WeightsArtifact(
        frame=frame,
        schema=schema,
        regions_version="synthetic-adm2",
        source_version="synthetic-ir",
    )


def apply(art: WeightsArtifact, leaf: pd.DataFrame) -> pd.DataFrame:
    return cilreg.apply_weights(
        art,
        leaf,
        kind="intensive",
        weight="croparea",
        value_col="pct",
        data_version="synthetic-ir",
        policies=SourceUnitPolicies(on_zero_weight="skip"),
    ).frame


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
