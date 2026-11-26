"""Two reduce backends have to give the same statistics, to the bit.

`assert_same_statistics` is the test a new backend has to pass before
it reduces a product that must match what is published: same leaves,
same request, two backends. Today both sides are the pandas backend, so
the comparison cannot go red on its own. The stub backends below are
its red: each changes one thing in the statistics and has to make the
comparison fail.

The leaves are synthetic and written with `leaves.write_leaf`.

One thing here is not about backends. The mean pandas takes of a group
can change in its last bit with what else is in the frame: on pandas
3.1.1 a missing value in one group moves the mean of others. A block
holds part of a leaf, so under the "fast" sum, which is that mean,
a block and a whole leaf can differ by a bit when the leaf has missing
values. Under "exact" they cannot. So the tests that compare block
sizes, or a block with the whole leaf, ask for equality to the bit
under "exact" and on leaves without missing values, and for agreement
to twelve digits under "fast" on leaves with them.
"""
from __future__ import annotations

import dataclasses
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cil_regionalization.leaves import TargetTable, write_leaf
from cil_regionalization.reduce import (
    LeafRequest,
    PandasReduceOptions,
    ReduceRequest,
)
from cil_regionalization.reduce.pandas_blocks import PandasBlocks
from cil_regionalization.stats import pooled_statistics, window_means

TARGETS = ["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8"]
MODELS = {"m0": 0.5, "m1": 0.3, "m2": 0.0}
DRAWS = ["d0", "d1"]
YEARS = [2050, 2051, 2052, 2080, 2081]
WINDOWS = (("2050-2052", 2050, 2052), ("2080-2081", 2080, 2081))
METHODS = list(
    itertools.product(["linear", "lower_step", "inverted_cdf"], ["fast", "exact"])
)
# "linear" has no weighted form, so it is tried without weights only
METHODS_AND_WEIGHTS = pytest.mark.parametrize(
    "methods, weighted",
    [
        (methods, weighted)
        for methods in METHODS
        for weighted in (False, True)
        if not (weighted and methods[0] == "linear")
    ],
    ids=lambda value: (
        "-".join(value) if isinstance(value, tuple)
        else "weighted" if value else "unweighted"
    ),
)


def _leaf_frame(
    model: str, draw: str, special: str, missing: bool = True
) -> pd.DataFrame:
    """One leaf. With two targets a row group:

    - T7 and T8 are in no leaf, so their group is empty everywhere.
    - T3 is in no leaf and T4 only in the leaves of m0 and m1, where it
      stops in 2052. A block of that group alone has rows and none in
      the second window.
    - m1/d0 has no T1 and T2, and m2/d1 no T6.

    Values tie across leaves, some are missing unless ``missing`` is
    False, and T5 is zero."""
    seed = list(MODELS).index(model) * 2 + DRAWS.index(draw)
    rng = np.random.default_rng(seed)
    rows = []
    for gid in ["T6", "T1", "T5", "T4", "T2"]:
        if (model, draw) == ("m1", "d0") and gid in ("T1", "T2"):
            continue
        if (model, draw) == ("m2", "d1") and gid == "T6":
            continue
        if gid == "T4" and model == "m2":
            continue
        for band in ("old", "young"):
            for year in YEARS:
                if gid == "T4" and year > 2052:
                    continue
                value = float(rng.normal()) / 7.0
                if gid == "T2" and band == "old":
                    # the same in every leaf
                    value = 0.25 + (year - 2050) / 3.0
                if gid == "T5":
                    value = 0.0
                if missing and gid == "T1" and year == 2051 and seed % 2:
                    value = np.nan
                if special == "all_nan" and gid == "T6" and year >= 2080:
                    value = np.nan
                if special == "infinite" and gid == "T1" and year == 2080 and seed == 0:
                    value = np.inf
                rows.append((gid, band, year, value))
    return pd.DataFrame(rows, columns=["gid", "band", "year", "value"])


def build_request(
    root: Path,
    *,
    weighted: bool,
    n_members: bool,
    methods=None,
    special: str = "",
    missing: bool = True,
    windows_of=None,
    quantiles=(0.5, 0.95),
) -> tuple[ReduceRequest, pd.DataFrame]:
    """Write the leaves under ``root`` and return the request for them,
    with the frame of every leaf joined by hand and labelled."""
    if methods is None:
        methods = ("lower_step" if weighted else "linear", "fast")
    targets = TargetTable.from_frame(pd.DataFrame({"gid": TARGETS}), ["gid"])
    targets.write(root)
    footer = {
        "format_version": 1,
        "targets_per_group": 2,
        "n_targets": len(targets),
        "target_digest": targets.digest,
        "aggregation_digest": "synthetic",
    }
    leaves, joined, rows_per_group, schema = [], [], np.zeros(4), None
    for model, draw in itertools.product(MODELS, DRAWS):
        frame = _leaf_frame(model, draw, special, missing)
        file = f"{model}/{draw}/leaf.aggregated.parquet"
        written = write_leaf(
            frame, root / file, targets=targets, value_col="value",
            identity={"aggregation_digest": "synthetic"}, targets_per_group=2,
        )
        schema = written.schema
        rows_per_group += written.rows_per_group
        windows = WINDOWS if windows_of is None else windows_of(model, draw)
        leaves.append(
            LeafRequest(
                file=file,
                levels=(model, draw),
                windows=tuple(windows),
                rows=len(frame),
                profile=written.profile["year"],
                model_weight=MODELS[model] if weighted else None,
            )
        )
        joined.append(frame.assign(model=model, draw=draw))
    request = ReduceRequest(
        leaves_root=str(root),
        leaves=tuple(leaves),
        footer=footer,
        targets_file=targets.filename,
        rows_per_group=tuple(rows_per_group.tolist()),
        schema=tuple(schema),
        id_fields=("gid",),
        level_names=("model", "draw"),
        time_col="year",
        value_col="value",
        window_labels=tuple(label for label, _, _ in WINDOWS),
        sample_dims=("model", "draw"),
        quantiles=tuple(quantiles),
        include_n_members=n_members,
        quantile_method=methods[0],
        summation=methods[1],
    )
    return request, pd.concat(joined, ignore_index=True)


def _sorted(frame: pd.DataFrame, request: ReduceRequest) -> pd.DataFrame:
    keys = [c for c in request.columns if c not in (request.value_col, "n_members")]
    return frame.sort_values(keys, kind="stable").reset_index(drop=True)


def assert_same_statistics(one, two, request: ReduceRequest) -> None:
    """Fail unless two backends give the same statistics for
    ``request``: the same rows, columns and dtypes, and values equal to
    the bit, so that -0.0 is not 0.0. Two backends that both raise have
    to raise the same type."""
    outcomes = []
    for backend in (one, two):
        try:
            outcomes.append(backend.run(request)[0])
        except Exception as error:  # noqa: BLE001 - compared below
            outcomes.append(error)
    first, second = outcomes
    if isinstance(first, Exception) or isinstance(second, Exception):
        assert type(first) is type(second), (first, second)
        return
    assert list(first.columns) == list(second.columns) == request.columns
    first, second = _sorted(first, request), _sorted(second, request)
    pd.testing.assert_frame_equal(first, second, check_exact=True)
    assert (
        first[request.value_col].to_numpy(dtype="float64").tobytes()
        == second[request.value_col].to_numpy(dtype="float64").tobytes()
    )


def _assert_matches(got, expected, request: ReduceRequest, *, to_the_bit: bool) -> None:
    """Equal to the bit, or to twelve digits where the module docstring
    says a bit can move."""
    got, expected = _sorted(got, request), _sorted(expected, request)
    if not to_the_bit:
        pd.testing.assert_frame_equal(got, expected, check_exact=False, rtol=1e-12)
        return
    pd.testing.assert_frame_equal(got, expected, check_exact=True)
    assert (
        got[request.value_col].to_numpy(dtype="float64").tobytes()
        == expected[request.value_col].to_numpy(dtype="float64").tobytes()
    )


def _backend(memory_gb: float = 1.0, n_workers: int = 1) -> PandasBlocks:
    return PandasBlocks(PandasReduceOptions(memory_gb=memory_gb, n_workers=n_workers))


def _one_group_a_block(request: ReduceRequest, n_workers: int = 1) -> PandasBlocks:
    """A budget that fits one row group a worker and no more, so every
    group is a block of its own."""
    from cil_regionalization.reduce.pandas_blocks import BYTES_PER_ROW, kept_share

    largest = max(request.rows_per_group) * kept_share(request) * BYTES_PER_ROW
    return _backend(memory_gb=largest * 1.01 * n_workers / 1e9, n_workers=n_workers)


def _by_hand(joined: pd.DataFrame, request: ReduceRequest, weighted: bool) -> pd.DataFrame:
    """The statistics as one call over every leaf gives them."""
    reduced = window_means(
        joined, time_col="year", windows=[(lo, hi) for _, lo, hi in WINDOWS],
        value_col="value", summation=request.summation,
    )
    if weighted:
        reduced["model_weight"] = reduced["model"].map(MODELS)
    stats = pooled_statistics(
        reduced,
        sample_dims=["model", "draw"],
        value_col="value",
        quantiles=list(request.quantiles),
        weight_col="model_weight" if weighted else None,
        include_n_members=request.include_n_members,
        quantile_method=request.quantile_method,
        summation=request.summation,
    )
    return stats[request.columns]


@METHODS_AND_WEIGHTS
@pytest.mark.parametrize("n_members", [False, True], ids=["plain", "n_members"])
class TestParity:
    def test_two_backends_agree(self, tmp_path, methods, weighted, n_members):
        request, _ = build_request(
            tmp_path, weighted=weighted, n_members=n_members, methods=methods
        )
        assert_same_statistics(
            _one_group_a_block(request), _one_group_a_block(request), request
        )

    @pytest.mark.parametrize("missing", [False, True], ids=["full", "missing"])
    def test_the_pandas_backend_gives_the_statistics_of_the_joined_frame(
        self, tmp_path, methods, weighted, n_members, missing
    ):
        """One block, one group a block, and two workers, against one
        call of `pooled_statistics` over every leaf."""
        request, joined = build_request(
            tmp_path, weighted=weighted, n_members=n_members, methods=methods,
            missing=missing,
        )
        expected = _by_hand(joined, request, weighted)
        # windows, bands, statistics: four targets in both windows, T4 in one
        assert len(expected) == 2 * 2 * 3 * 4 + 1 * 2 * 3
        to_the_bit = request.summation == "exact" or not missing
        backends = [_backend(), _one_group_a_block(request)]
        if weighted and n_members:
            # a pool of two, in a quarter of the cases, for the time it takes
            backends.append(_one_group_a_block(request, n_workers=2))
        for backend in backends:
            stats, report = backend.run(request)
            _assert_matches(stats, expected, request, to_the_bit=to_the_bit)
            assert report["failed_opens"] == 0
        assert report["blocks"] == 3

    @pytest.mark.parametrize("special", ["all_nan", "infinite"])
    def test_what_one_side_refuses_the_other_refuses(
        self, tmp_path, methods, weighted, n_members, special
    ):
        """A group whose members are all missing is a NaN row without
        weights and an error with them. An infinite statistic is
        returned under "linear" and "lower_step" with the fast sum, and
        refused otherwise."""
        request, joined = build_request(
            tmp_path, weighted=weighted, n_members=n_members, methods=methods,
            special=special,
        )
        assert_same_statistics(
            _one_group_a_block(request), _one_group_a_block(request), request
        )
        raises = (special == "all_nan" and weighted) or (
            special == "infinite"
            and (methods[0] == "inverted_cdf" or methods[1] == "exact")
        )
        if raises:
            for backend in (_backend(), _one_group_a_block(request)):
                with pytest.raises(ValueError):
                    backend.run(request)
            with pytest.raises(ValueError):
                _by_hand(joined, request, weighted)
        else:
            expected = _by_hand(joined, request, weighted)
            for backend in (_backend(), _one_group_a_block(request)):
                stats, _ = backend.run(request)
                _assert_matches(
                    stats, expected, request,
                    to_the_bit=request.summation == "exact",
                )
            assert (
                stats["value"].isna().any() if special == "all_nan"
                else np.isinf(stats["value"]).any()
            )


class _Changed(PandasBlocks):
    """The pandas backend with one thing changed in what it returns."""

    def __init__(self, change) -> None:
        super().__init__(PandasReduceOptions(memory_gb=1.0))
        self.change = change

    def run(self, request):
        stats, report = super().run(request)
        return self.change(stats.copy()), report


def _one_ulp(stats):
    row = stats.index[stats["value"].abs() > 0.1][0]
    stats.loc[row, "value"] = np.nextafter(stats.loc[row, "value"], np.inf)
    return stats


def _minus_zero(stats):
    row = stats.index[stats["value"] == 0.0][0]
    stats.loc[row, "value"] = -0.0
    return stats


def _narrow_counts(stats):
    return stats.astype({"n_members": "int32"})


_CHANGES = {
    "one ulp": _one_ulp,
    "minus zero": _minus_zero,
    "a row dropped": lambda stats: stats.iloc[1:],
    "a dtype changed": _narrow_counts,
    "a label changed": lambda stats: stats.assign(
        window=stats["window"].replace("2050-2052", "2050-2051")
    ),
    "a column moved": lambda stats: stats[list(stats.columns[::-1])],
}


class TestTheHarnessFails:
    """The red of `assert_same_statistics`."""

    @pytest.mark.parametrize("change", list(_CHANGES))
    @pytest.mark.parametrize("weighted", [False, True], ids=["unweighted", "weighted"])
    def test_a_backend_that_differs_in_one_thing_is_caught(
        self, tmp_path, change, weighted
    ):
        request, _ = build_request(tmp_path, weighted=weighted, n_members=True)
        assert_same_statistics(_backend(), _Changed(lambda stats: stats), request)
        with pytest.raises(AssertionError):
            assert_same_statistics(_backend(), _Changed(_CHANGES[change]), request)

    def test_minus_zero_is_what_the_frame_comparison_alone_lets_through(
        self, tmp_path
    ):
        """This is why the bits are compared as well."""
        request, _ = build_request(tmp_path, weighted=False, n_members=False)
        one, _ = _backend().run(request)
        two = _minus_zero(one.copy())
        pd.testing.assert_frame_equal(one, two, check_exact=True)
        assert one["value"].to_numpy().tobytes() != two["value"].to_numpy().tobytes()

    def test_an_error_of_another_type_is_caught(self, tmp_path):
        class Raises(PandasBlocks):
            def run(self, request):
                raise RuntimeError("another engine, another error")

        request, _ = build_request(
            tmp_path, weighted=True, n_members=False, special="all_nan"
        )
        with pytest.raises(ValueError):
            _backend().run(request)
        with pytest.raises(AssertionError):
            assert_same_statistics(_backend(), Raises(), request)


class TestWindowsOfTheLeaf:
    """A leaf brings its own windows. Nothing in a config does this
    yet, so the requests here are built by hand."""

    @staticmethod
    def _windows_of(model, draw):
        """One label with a span that moves with the model, a second
        label m2 does not have, and for m0 a third with the span of the
        first."""
        start = 2050 + list(MODELS).index(model)
        windows = [("warm", start, start + 1)]
        if model != "m2":
            windows.append(("late", 2080, 2081))
        if model == "m0":
            windows.append(("also", start, start + 1))
        return windows

    @METHODS_AND_WEIGHTS
    def test_each_leaf_is_reduced_over_its_own_spans(
        self, tmp_path, weighted, methods
    ):
        request, joined = build_request(
            tmp_path, weighted=weighted, n_members=True, methods=methods,
            windows_of=self._windows_of,
        )
        request = dataclasses.replace(
            request, window_labels=("warm", "late", "also")
        )
        # by hand: every leaf over its own spans, under the request's labels
        pieces = []
        for (model, draw), leaf in joined.groupby(["model", "draw"], sort=False):
            for label, lo, hi in self._windows_of(model, draw):
                piece = window_means(
                    leaf, time_col="year", windows=[(lo, hi)], value_col="value",
                    summation=request.summation,
                )
                pieces.append(piece.assign(window=label))
        reduced = pd.concat(pieces, ignore_index=True)
        if weighted:
            reduced["model_weight"] = reduced["model"].map(MODELS)
        expected = pooled_statistics(
            reduced, sample_dims=["model", "draw"], value_col="value",
            quantiles=list(request.quantiles),
            weight_col="model_weight" if weighted else None,
            include_n_members=True, quantile_method=request.quantile_method,
            summation=request.summation,
        )[request.columns]

        for backend in (_backend(), _one_group_a_block(request)):
            stats, _ = backend.run(request)
            _assert_matches(
                stats, expected, request, to_the_bit=request.summation == "exact"
            )
        got = _sorted(stats, request)
        # T5 is in every leaf. m2 gives no member to "late", only m0
        # gives any to "also", and with weights m2 counts for nothing.
        counts = got[(got["gid"] == "T5") & (got["statistic"] == "mean")]
        counts = counts.groupby("window")["n_members"].max().to_dict()
        assert counts == {"warm": 4 if weighted else 6, "late": 4, "also": 2}

    @pytest.mark.parametrize("weighted", [False, True], ids=["unweighted", "weighted"])
    def test_a_block_with_rows_and_none_in_any_window_gives_no_statistics(
        self, tmp_path, weighted
    ):
        """T4 stops in 2052 and the one window here starts in 2080. With
        one row group a block, the block of T4 reads rows and keeps
        none. Failed with the statistics asked of that block all the
        same."""
        request, joined = build_request(
            tmp_path, weighted=weighted, n_members=True, missing=False,
            windows_of=lambda model, draw: [WINDOWS[1]],
        )
        reduced = window_means(
            joined, time_col="year", windows=[WINDOWS[1][1:]], value_col="value",
            summation="fast"
        )
        if weighted:
            reduced["model_weight"] = reduced["model"].map(MODELS)
        expected = pooled_statistics(
            reduced, sample_dims=["model", "draw"], value_col="value",
            quantiles=list(request.quantiles),
            weight_col="model_weight" if weighted else None,
            include_n_members=True,
            quantile_method="lower_step" if weighted else "linear",
            summation="fast",
        )[request.columns]
        stats, report = _one_group_a_block(request).run(request)
        assert report["blocks"] == 3
        assert "T4" not in set(stats["gid"])
        pd.testing.assert_frame_equal(
            _sorted(stats, request), _sorted(expected, request), check_exact=True
        )

    def test_a_block_is_sized_by_what_each_leaf_keeps(self, tmp_path):
        """Every leaf has five time values. With the same two windows
        each keeps two rows in five. Here m0 has three windows, m1 two
        and m2 one, and two of the leaves are shorter than the rest."""
        from cil_regionalization.reduce.pandas_blocks import kept_share

        same, _ = build_request(tmp_path / "same", weighted=False, n_members=False)
        assert kept_share(same) == pytest.approx(2 / 5)
        own, _ = build_request(
            tmp_path / "own", weighted=False, n_members=False,
            windows_of=self._windows_of,
        )
        rows = {leaf.levels: leaf.rows for leaf in own.leaves}
        assert len(set(rows.values())) > 1
        windows = {"m0": 3, "m1": 2, "m2": 1}
        kept = sum(n / 5 * windows[levels[0]] for levels, n in rows.items())
        assert kept_share(own) == pytest.approx(kept / sum(rows.values()))
        assert kept_share(own) != pytest.approx(2 / 5)


class TestBlocks:
    def test_groups_are_packed_in_order_up_to_the_budget(self):
        from cil_regionalization.reduce.pandas_blocks import (
            BYTES_PER_ROW, choose_blocks,
        )

        rows = [10, 10, 0, 0, 30, 5, 0, 20]
        assert choose_blocks(rows, 1.0, 1000 * BYTES_PER_ROW) == [(0, 8)]
        assert choose_blocks(rows, 1.0, 30 * BYTES_PER_ROW) == [
            (0, 4), (4, 5), (5, 8),
        ]
        assert choose_blocks(rows, 1.0, 35 * BYTES_PER_ROW) == [
            (0, 4), (4, 7), (7, 8),
        ]
        # half the rows kept, so twice the groups fit
        assert choose_blocks(rows, 0.5, 30 * BYTES_PER_ROW) == [(0, 7), (7, 8)]
        # groups that hold nothing start no block
        assert choose_blocks([0, 0, 7, 0, 0], 1.0, 7 * BYTES_PER_ROW) == [(2, 5)]
        assert choose_blocks([0, 0, 0], 1.0, 7 * BYTES_PER_ROW) == []

    def test_a_group_that_does_not_fit_is_refused(self):
        from cil_regionalization.reduce.pandas_blocks import (
            BYTES_PER_ROW, choose_blocks,
        )

        with pytest.raises(ValueError):
            choose_blocks([10, 31, 5], 1.0, 30 * BYTES_PER_ROW)

    def test_a_worker_that_dies_stops_the_run(self, tmp_path, monkeypatch):
        """Failed with a `multiprocessing.Pool`, which waits for the
        dead worker's block for ever. The workers here are forked, so
        that they see the block function this test puts in."""
        import functools
        import multiprocessing

        from cil_regionalization.reduce import pandas_blocks

        if "fork" not in multiprocessing.get_all_start_methods():
            pytest.skip("needs forked workers")
        request, _ = build_request(tmp_path, weighted=False, n_members=False)
        monkeypatch.setattr(
            pandas_blocks,
            "ProcessPoolExecutor",
            functools.partial(
                pandas_blocks.ProcessPoolExecutor,
                mp_context=multiprocessing.get_context("fork"),
            ),
        )
        monkeypatch.setattr(pandas_blocks, "_reduce_block", _die)
        with pytest.raises(RuntimeError, match="a worker died"):
            _one_group_a_block(request, n_workers=2).run(request)


def _die(block):
    import os

    os._exit(1)
