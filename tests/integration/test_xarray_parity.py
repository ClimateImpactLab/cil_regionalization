"""The statistics that pool or keep the years, against xarray.

The pipeline runs for real, by blocks, on two workers. The oracle reads
the leaf files it wrote with pyarrow into one xarray DataArray and takes
each statistic over the pooled dimensions, the way the Inequality
Paper's valued path does, so it shares no code with the reduce backend.

Compared to the bit:

- "linear" and "inverted_cdf" quantiles without weights, against
  xarray's ``.quantile`` with the same method.
- "lower_step" quantiles, and "inverted_cdf" with weights, against the
  rule each is defined by, written here with numpy. xarray has neither.
- The "exact" mean, against an exactly rounded sum over the same
  values, weighted or not.
- ``n_members``, ``n_sample`` and ``n_bin_members``.

Compared within a bound: the "fast" mean, against xarray's mean. Two
correct sums of the same values part in their last bits, so the
difference may reach twice the count of the group times the float64
epsilon times the largest magnitude in it. A member or a year too many
or too few moves a mean by far more than that.

These tests check code that already exists, so they cannot go red by
reverting it. They failed under mutations of the backend: the year
left out of the pooled sample, the last year of a span left out, and a
member pooled under a level it is not at.
"""
from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

xr = pytest.importorskip("xarray")

# unweighted runs pool a level named gcm without weights on purpose
pytestmark = pytest.mark.filterwarnings(
    "ignore::cil_regionalization.stats.UnweightedModelWeightsWarning"
)

from cil_regionalization.pipelines.montecarlo import (  # noqa: E402
    MonteCarloPipelineConfig,
    run_pipeline,
)
from tests.integration.test_montecarlo_pipeline import (  # noqa: E402,F401
    SOURCE_VERSION,
    one_target_a_group,
    weights_dir,
)

LEVELS = {
    "batch": ["b0", "b1", "b2", "b3"],
    "rcp": ["r45", "r85"],
    "gcm": ["m0", "m1", "surrogate_m0_06"],
}
POOLED = list(LEVELS)
YEARS = list(range(2040, 2100))
QUANTILES = [0.05, 0.17, 0.25, 0.5, 0.75, 0.83, 0.95]
# Per year a member of weight 0.5 is a sixteenth of the weight, so the
# step has nothing below 0.05 there and gives -inf, which "exact"
# refuses. The quantiles of a year start above that.
YEAR_QUANTILES = QUANTILES[1:]
LABELS = {q: f"q{int(round(q * 100)):02d}" for q in QUANTILES}
# The weight of each climate model, matched as the weights table
# matches. They are exact in binary, so that a cumulative share that
# equals q in exact arithmetic equals it in both implementations and the
# rule decides it, not the order the weights were summed in. With 0.3
# and 0.2 one in a few hundred quantiles fell on such a tie.
GCM_WEIGHTS = {"m0": 0.5, "m1": 0.375, "surrogate_m0_06": 0.125}
# the leaf of b1/r85/m1 lacks these years
GAP = [2050, 2051, 2052, 2090]


def _build_tree(root: Path, *, gaps: bool) -> None:
    """One leaf a batch, scenario and model, every year from 2040 to
    2099. Values of five magnitudes, and in the band "young" values
    rounded to tenths, so that some members tie."""
    rng = np.random.default_rng(11)
    for combo in itertools.product(*LEVELS.values()):
        rows = []
        for unit in ("u1", "u2"):
            for band in ("old", "young"):
                for year in YEARS:
                    value = float(rng.normal()) * 10.0 ** int(rng.integers(-2, 3))
                    if band == "young":
                        value = round(value, 1)
                    rows.append((unit, band, year, value))
        frame = pd.DataFrame(rows, columns=["unit_id", "band", "year", "value"])
        if gaps and combo == ("b1", "r85", "m1"):
            frame = frame[~frame["year"].isin(GAP)]
        leaf_dir = root.joinpath(*combo)
        leaf_dir.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(leaf_dir / "damages.parquet", index=False)


def _raw(
    weights_dir, tmp_path, *, methods, weighted=False, gaps=True,
    quantiles=QUANTILES, memory_gb=0.002, **stats,
):
    tree, out = tmp_path / "tree", tmp_path / "out"
    _build_tree(tree, gaps=gaps)
    raw = {
        "weights": {"dir": str(weights_dir), "weight": "area"},
        "tree": {
            "root": str(tree),
            "levels": list(LEVELS),
            "filename": "damages.parquet",
            "expect": LEVELS,
        },
        "data": {"kind": "extensive", "version": SOURCE_VERSION, "value_col": "value"},
        "stats": {
            "sample_dims": POOLED,
            "time_col": "year",
            "quantiles": list(quantiles),
            "quantile_method": methods[0],
            "summation": methods[1],
            "include_n_members": True,
            **stats,
        },
        "output": {"dir": str(out), "write_leaves": True},
        "run": {"n_workers": 2},
        # one row group a worker and not two, so that there are two blocks
        "reduce": {"pandas": {"memory_gb": memory_gb, "n_workers": 2}},
    }
    if weighted:
        table = tmp_path / "model_weights.tsv"
        table.write_text("model\tweight\nm0\t0.5\nm1\t0.375\nm0_06\t0.125\n")
        raw["stats"].update(model_weights=str(table), model_weight_level="gcm")
    return raw, out


def _run(raw: dict, out: Path) -> pd.DataFrame:
    run_pipeline(MonteCarloPipelineConfig.model_validate(raw))
    manifest = json.loads((out / "statistics.manifest.json").read_text())
    assert manifest["reduce"]["blocks"] == 2
    assert manifest["reduce"]["options"]["n_workers"] == 2
    return pd.read_parquet(out / "statistics.parquet")


def _leaves(out: Path):
    """Every leaf file the run wrote, as one DataArray over the target,
    the band, the tree levels and the year. A year a leaf lacks is NaN."""
    frames = []
    for combo in itertools.product(*LEVELS.values()):
        path = out / "leaves" / Path(*combo) / "damages.aggregated.parquet"
        frame = pq.read_table(path).to_pandas().drop(columns=["target_ordinal"])
        frames.append(frame.assign(**dict(zip(LEVELS, combo))))
    whole = pd.concat(frames, ignore_index=True)
    return whole.set_index(["target_id", "band", *LEVELS, "year"])["value"].to_xarray()


# the rules the step and the weighted inverse are defined by


def _lower_step(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    """The largest value whose cumulative weight share is strictly
    below q, or -inf if there is none. With equal weights the 0.5
    quantile is the median."""
    if q == 0.5 and (weights == weights[0]).all():
        return float(np.median(values))
    order = np.argsort(values, kind="stable")
    share = np.cumsum(weights[order]) / weights.sum()
    below = values[order][share < q]
    return float(below.max()) if len(below) else -np.inf


def _inverted_cdf(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    """The smallest value of positive weight whose cumulative weight
    share reaches q."""
    positive = weights > 0
    values, weights = values[positive], weights[positive]
    order = np.argsort(values, kind="stable")
    share = np.cumsum(weights[order]) / weights.sum()
    reached = values[order][share >= q]
    return float(reached.min() if len(reached) else values.max()) + 0.0


def _oracle(
    da, pooled: list[str], *, method: str, weighted: bool, quantiles=QUANTILES
) -> pd.DataFrame:
    """Every statistic of ``da`` over ``pooled``, one row a group and
    statistic, with the counts."""
    sample = [d for d in pooled if d in da.dims]
    stacked = da.stack(member=sample).transpose(..., "member")
    weights = (
        np.array([GCM_WEIGHTS[g] for g in stacked["gcm"].values])
        if weighted
        else np.ones(stacked.sizes["member"])
    )
    keys = [d for d in stacked.dims if d != "member"]
    if weighted or method == "lower_step":
        rule = _lower_step if method == "lower_step" else _inverted_cdf
        table = {}
        flat = stacked.values.reshape(-1, stacked.sizes["member"])
        for q in quantiles:
            out = np.full(len(flat), np.nan)
            for i, row in enumerate(flat):
                kept = ~np.isnan(row)
                if kept.any():
                    out[i] = rule(row[kept], weights[kept], q)
            table[LABELS[q]] = out.reshape(stacked.shape[:-1])
        quantiles = {
            label: xr.DataArray(values, dims=keys, coords={k: stacked[k] for k in keys})
            for label, values in table.items()
        }
    else:
        q = da.quantile(quantiles, dim=sample, method=method, skipna=True)
        quantiles = {LABELS[v]: q.sel(quantile=v).drop_vars("quantile") for v in quantiles}

    flat = stacked.values.reshape(-1, stacked.sizes["member"])
    exact = np.full(len(flat), np.nan)
    for i, row in enumerate(flat):
        kept = ~np.isnan(row)
        if kept.any():
            w = weights[kept]
            with np.errstate(over="ignore"):
                products = row[kept] * w
            exact[i] = math.fsum(products.tolist()) / math.fsum(w.tolist())
    coords = {k: stacked[k] for k in keys}
    exact = xr.DataArray(exact.reshape(stacked.shape[:-1]), dims=keys, coords=coords)
    if weighted:
        fast = da.weighted(xr.DataArray(
            [GCM_WEIGHTS[g] for g in da["gcm"].values], dims="gcm",
            coords={"gcm": da["gcm"]},
        )).mean(dim=sample, skipna=True)
    else:
        fast = da.mean(dim=sample, skipna=True)
    largest = abs(da).max(dim=sample, skipna=True)
    count = da.count(dim=sample)

    frames = []
    for label, values in [("mean", exact)] + list(quantiles.items()):
        frame = values.to_series().rename("oracle").reset_index()
        frames.append(frame.assign(statistic=label))
    whole = pd.concat(frames, ignore_index=True)
    extra = xr.Dataset({"fast": fast, "largest": largest, "count": count}).to_dataframe()
    return whole.merge(extra.reset_index(), on=keys, how="left")


def _compare(ours: pd.DataFrame, oracle: pd.DataFrame, keys: list[str], summation: str):
    """The same rows, the quantiles and the counts to the bit, the
    exact mean to the bit and the fast mean within the bound."""
    oracle = oracle[oracle["count"] > 0]
    both = ours.merge(oracle, on=keys + ["statistic"], how="outer", indicator=True)
    assert (both["_merge"] == "both").all(), both.loc[both["_merge"] != "both", keys]
    assert len(both) == len(ours)
    is_mean = (both["statistic"] == "mean").to_numpy()
    value = both["value"].to_numpy(np.float64)
    if summation == "exact":
        to_the_bit = np.ones(len(both), dtype=bool)
    else:
        to_the_bit = ~is_mean
        bound = 2 * both["count"] * np.finfo(np.float64).eps * both["largest"]
        gap = (both["value"] - both["fast"]).abs()
        assert (gap[is_mean] <= bound[is_mean]).all(), both.loc[is_mean & (gap > bound)]
    want = both["oracle"].to_numpy(np.float64)
    same = value.view(np.int64) == want.view(np.int64)
    assert same[to_the_bit].all(), both.loc[to_the_bit & ~same, keys + ["statistic", "value", "oracle"]]
    assert (both["n_sample" if "n_sample" in both else "n_members"] == both["count"]).all()


METHODS = [
    pytest.param(("linear", "fast"), False, id="linear-fast"),
    pytest.param(("linear", "exact"), False, id="linear-exact"),
    pytest.param(("inverted_cdf", "fast"), False, id="inverted_cdf-fast"),
    pytest.param(("inverted_cdf", "exact"), False, id="inverted_cdf-exact"),
    pytest.param(("lower_step", "fast"), False, id="lower_step-fast"),
    pytest.param(("lower_step", "exact"), False, id="lower_step-exact"),
    pytest.param(("lower_step", "fast"), True, id="lower_step-fast-weighted"),
    pytest.param(("lower_step", "exact"), True, id="lower_step-exact-weighted"),
    pytest.param(("inverted_cdf", "fast"), True, id="inverted_cdf-fast-weighted"),
    pytest.param(("inverted_cdf", "exact"), True, id="inverted_cdf-exact-weighted"),
]


@pytest.mark.usefixtures("one_target_a_group")
@pytest.mark.parametrize("methods, weighted", METHODS)
def test_pooled_years_are_xarrays_statistics(weights_dir, tmp_path, methods, weighted):
    """Three windows, the second over half of the first, and a leaf
    that lacks four years."""
    windows = [(2040, 2059), (2050, 2069), (2080, 2099)]
    raw, out = _raw(
        weights_dir, tmp_path, methods=methods, weighted=weighted,
        time_reduction="pooled", windows=[list(w) for w in windows],
    )
    ours = _run(raw, out)
    da = _leaves(out)
    pieces = []
    for lo, hi in windows:
        oracle = _oracle(
            da.sel(year=slice(lo, hi)), POOLED + ["year"],
            method=methods[0], weighted=weighted,
        )
        span = da.sel(year=slice(lo, hi))
        members = span.notnull().any("year").sum(POOLED).to_series().rename("members")
        oracle = oracle.merge(members.reset_index(), on=["target_id", "band"])
        pieces.append(oracle.assign(window=f"{lo}-{hi}"))
    oracle = pd.concat(pieces, ignore_index=True)
    keys = ["target_id", "band", "window"]
    _compare(ours, oracle, keys, methods[1])
    both = ours.merge(oracle, on=keys + ["statistic"])
    assert (both["n_members"] == both["members"]).all()
    # the leaf with the gap is one member short in the years it lacks,
    # not in its windows
    assert set(ours["n_members"]) == {24}


@pytest.mark.usefixtures("one_target_a_group")
@pytest.mark.parametrize("methods, weighted", METHODS)
def test_each_year_is_xarrays_statistics(weights_dir, tmp_path, methods, weighted):
    raw, out = _raw(
        weights_dir, tmp_path, methods=methods, weighted=weighted,
        time_reduction="per_year", years=[2041, 2098], quantiles=YEAR_QUANTILES,
    )
    ours = _run(raw, out)
    da = _leaves(out).sel(year=slice(2041, 2098))
    oracle = _oracle(
        da, POOLED, method=methods[0], weighted=weighted, quantiles=YEAR_QUANTILES
    )
    _compare(ours, oracle, ["target_id", "band", "year"], methods[1])
    counts = ours.groupby("year")["n_members"].min()
    assert counts[2051] == 23 and counts[2060] == 24
    assert sorted(ours["year"].unique()) == list(range(2041, 2099))


# the temperature of each model and scenario in the 2040s and the 2080s,
# before the half a degree of the offset; m1 under r85 is at no level
TEMPERATURES = {
    ("m0", "r45"): (1.0, 1.5),
    ("m1", "r45"): (1.0, 2.5),
    ("surrogate_m0_06", "r45"): (1.5, 2.5),
    ("m0", "r85"): (1.5, 2.5),
    ("m1", "r85"): (9.0, 9.0),
    ("surrogate_m0_06", "r85"): (1.0, 1.5),
}
WARMING = {"levels": [1.5, 2.0, 3.0], "half_width": 0.25, "offset": 0.5}
PERIODS = [(2040, 2059), (2080, 2099)]


def _level(temperature: float):
    """The level a mean temperature is at, by the rule the levels are
    defined by: within the half-width, the level declared first."""
    mean = temperature + WARMING["offset"]
    for level in WARMING["levels"]:
        if abs(mean - level) <= WARMING["half_width"]:
            return level
    return None


@pytest.mark.usefixtures("one_target_a_group")
@pytest.mark.parametrize("methods", [
    ("linear", "fast"), ("linear", "exact"), ("inverted_cdf", "fast"),
    ("inverted_cdf", "exact"), ("lower_step", "fast"), ("lower_step", "exact"),
], ids="-".join)
def test_warming_levels_pooled_are_xarrays_statistics(weights_dir, tmp_path, methods):
    """The paper's valued expression: the members at a level, their
    years pooled with the models and the draws."""
    gmst = tmp_path / "gmst.csv"
    rows = ["gcm,rcp,year,temp"]
    for (gcm, rcp), (mid, late) in TEMPERATURES.items():
        for year in range(1986, 2100):
            temp = 0.0 if year < 2040 else (mid if year < 2070 else late)
            rows.append(f"{gcm},{rcp},{year},{temp}")
    gmst.write_text("\n".join(rows) + "\n")
    raw, out = _raw(
        weights_dir, tmp_path, methods=methods, gaps=False,
        time_reduction="pooled",
        # the windows keep about half the rows, so the budget is smaller
        memory_gb=0.0012,
        warming_levels={
            "gmst": str(gmst), "member": ["gcm", "rcp"],
            "reference_period": [1986, 2005], "smoothing": "unknown",
            "periods": [{"gmst_span": list(p), "impact_span": list(p)} for p in PERIODS],
            **WARMING,
        },
    )
    ours = _run(raw, out)
    da = _leaves(out)
    pieces = []
    for i, (lo, hi) in enumerate(PERIODS):
        level = xr.DataArray(
            [[_level(TEMPERATURES[(g, r)][i]) or np.nan for r in LEVELS["rcp"]]
             for g in LEVELS["gcm"]],
            dims=["gcm", "rcp"], coords={"gcm": LEVELS["gcm"], "rcp": LEVELS["rcp"]},
        )
        span = da.sel(year=slice(lo, hi))
        for at in WARMING["levels"]:
            members = span.where(level == at)
            if not members.notnull().any():
                continue
            oracle = _oracle(members, POOLED + ["year"], method=methods[0], weighted=False)
            has = members.notnull().any("year")
            counts = pd.DataFrame({
                "members": has.sum(POOLED).to_series(),
                "bins": has.any("batch").sum(["rcp", "gcm"]).to_series(),
            }).reset_index()
            oracle = oracle.merge(counts, on=["target_id", "band"])
            pieces.append(oracle.assign(window=f"{lo}-{hi}", warming_level=at))
    oracle = pd.concat(pieces, ignore_index=True)
    keys = ["target_id", "band", "warming_level", "window"]
    _compare(ours, oracle, keys, methods[1])
    both = ours.merge(oracle, on=keys + ["statistic"])
    assert (both["n_members"] == both["members"]).all()
    assert (both["n_bin_members"] == both["bins"]).all()
    assert sorted(set(zip(ours["window"], ours["warming_level"]))) == [
        ("2040-2059", 1.5), ("2040-2059", 2.0), ("2080-2099", 2.0), ("2080-2099", 3.0),
    ]
