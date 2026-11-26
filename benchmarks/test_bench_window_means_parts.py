"""Where `window_means` spends its time, and what constant label
columns cost it.

A caller can label a leaf with its member (batch, gcm, iam, ssp) before
reducing or after. Before, each label is hashed over every row to learn
it has one value.
"""
from __future__ import annotations

import pandas as pd
import pytest

from benchmarks.synthetic import (
    ID_FIELDS,
    LABELS,
    WINDOWS,
    applied_leaf,
    best,
    report,
)
from cil_regionalization.stats import (
    _has_duplicates,
    _identity_groups,
    window_means,
)

pytestmark = pytest.mark.benchmark

_KW = dict(time_col="year", windows=WINDOWS, value_col="pct", summation="fast")


def test_labels_after_reducing_match_labels_before():
    bare = applied_leaf()
    labelled = bare.assign(**LABELS)
    columns = ID_FIELDS + list(LABELS) + ["window", "pct"]

    before_s, before = best(lambda: window_means(labelled, **_KW))
    after_s, after = best(
        lambda: window_means(bare, **_KW).assign(**LABELS)[columns]
    )
    pd.testing.assert_frame_equal(after, before, check_exact=True)

    group = ID_FIELDS + list(LABELS)
    codes_s, groups = best(lambda: _identity_groups(labelled, group))
    check_s, duplicated = best(lambda: _has_duplicates(labelled, groups, ["year"]))
    assert not duplicated
    rest_s = before_s - codes_s - check_s

    report(f"window_means, {len(labelled):,} rows, labels on the leaf: {before_s:.3f} s")
    report(f"  group codes:           {codes_s:.3f} s ({codes_s / before_s:.0%})")
    report(f"  duplicate check:       {check_s:.3f} s ({check_s / before_s:.0%})")
    report(f"  window loop and rest:  {rest_s:.3f} s ({rest_s / before_s:.0%})")
    report(f"labels attached after reducing: {after_s:.3f} s ({before_s / after_s:.1f}x)")
