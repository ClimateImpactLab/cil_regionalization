"""`window_means` against the plain pandas path it replaced.

The plain path checks duplicates with `DataFrame.duplicated` and runs a
groupby over the key columns for each window. The library works both
out from one set of integer group codes.
"""
from __future__ import annotations

import pandas as pd
import pytest

from benchmarks.synthetic import LABELS, WINDOWS, applied_leaf, best, report
from cil_regionalization.stats import _window_label, window_means

pytestmark = pytest.mark.benchmark


def plain_window_means(data: pd.DataFrame) -> pd.DataFrame:
    group = [c for c in data.columns if c not in ("year", "pct")]
    if data.duplicated(subset=group + ["year"]).any():
        raise ValueError("duplicated rows")
    pieces = []
    for lo, hi in WINDOWS:
        piece = (
            data.loc[data["year"].between(lo, hi)]
            .groupby(group, sort=False, observed=True)["pct"]
            .mean()
            .reset_index()
        )
        piece["window"] = _window_label(lo, hi)
        pieces.append(piece)
    out = pd.concat(pieces, ignore_index=True)
    return out[group + ["window", "pct"]]


def test_group_codes_match_the_plain_path():
    leaf = applied_leaf().assign(**LABELS)
    plain_s, plain = best(lambda: plain_window_means(leaf))
    codes_s, codes = best(
        lambda: window_means(
            leaf, time_col="year", windows=WINDOWS, value_col="pct", summation="fast",
        )
    )
    pd.testing.assert_frame_equal(codes, plain, check_exact=True)
    report(f"window_means, {len(leaf):,} rows to {len(codes):,}")
    report(f"  plain pandas path: {plain_s:.3f} s")
    report(f"  library:           {codes_s:.3f} s ({plain_s / codes_s:.1f}x)")
