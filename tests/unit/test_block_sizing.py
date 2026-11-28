"""How the pandas backend sizes its blocks, without leaves."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from cil_regionalization.reduce.base import PandasReduceOptions
from cil_regionalization.reduce.pandas_blocks import PandasBlocks, fixed_blocks


def _request(rows_per_group):
    return SimpleNamespace(
        leaves=[],
        time_reduction="window_mean",
        dimensions=(),
        footer={"targets_per_group": 128},
        rows_per_group=rows_per_group,
        n_workers=1,
    )


class TestDeclaredBlocks:
    def test_runs_of_whole_row_groups_from_the_first_that_holds_rows(self):
        assert fixed_blocks([0, 0, 5, 5, 0, 5, 5], 2) == [(2, 4), (4, 6), (6, 7)]
        assert fixed_blocks([0, 0], 2) == []

    def test_a_declared_size_is_a_whole_number_of_row_groups(self):
        plan = PandasBlocks(PandasReduceOptions(targets_per_block=256))._plan(
            _request([10, 10, 10])
        )
        assert plan["by"] == "declared" and plan["blocks"] == [(0, 2), (2, 3)]
        with pytest.raises(
            ValueError,
            match="targets_per_block = 200 is not a whole number of the "
            "leaves' row groups, which hold 128 targets each",
        ):
            PandasBlocks(PandasReduceOptions(targets_per_block=200))._plan(
                _request([10, 10])
            )

    def test_a_budget_or_a_declared_size_is_needed(self):
        with pytest.raises(ValueError, match="or declare the size of a block"):
            PandasBlocks(PandasReduceOptions())._plan(_request([10]))


class TestGrowth:
    """When the growth of a peak says something of the first block."""

    def test_a_peak_that_rose_gives_its_growth(self):
        from cil_regionalization.reduce.pandas_blocks import _growth

        assert _growth(10**9, 10**9 + 5, own_process=True) == (5, None)

    def test_a_peak_that_did_not_rise_gives_no_measure(self):
        """As when the block fits in memory the worker already held. It
        went red with a growth of zero recorded."""
        from cil_regionalization.reduce.pandas_blocks import _FLAT_PEAK, _growth

        assert _growth(10**9, 10**9, own_process=True) == (None, _FLAT_PEAK)
        assert "did not rise over the block" in _FLAT_PEAK

    def test_the_process_that_planned_the_run_gives_no_measure(self):
        from cil_regionalization.reduce.pandas_blocks import _SHARED_PEAK, _growth

        assert _growth(10**9, 2 * 10**9, own_process=False) == (None, _SHARED_PEAK)

    def test_a_peak_not_read_gives_no_measure_and_says_why(self, monkeypatch):
        import sys

        from cil_regionalization.reduce.pandas_blocks import _growth

        monkeypatch.setattr(sys, "platform", "freebsd14")
        assert _growth(None, None, own_process=True) == (
            None,
            "the peak resident memory of a process is not read on freebsd14",
        )

    def test_the_growth_leaves_out_what_the_worker_set_up(self, monkeypatch):
        """The setup raises the peak by 100 and the block by 10, and the
        growth is the block's. It went red with the peak taken as the
        worker started, which gave 110."""
        from cil_regionalization.reduce import pandas_blocks

        peak = [1_000]

        def read(path):
            peak[0] += 100
            return SimpleNamespace(digest="d")

        def block(rows):
            peak[0] += 10
            return {"block": rows, "rows": 1}

        monkeypatch.setattr(pandas_blocks, "_STATE", {})
        monkeypatch.setattr(pandas_blocks, "_peak_resident_bytes", lambda: peak[0])
        monkeypatch.setattr(pandas_blocks, "_reduce_block", block)
        monkeypatch.setattr(
            pandas_blocks.leaf_format.TargetTable, "read", staticmethod(read)
        )
        pandas_blocks._init_worker(
            SimpleNamespace(
                leaves_root="/", targets_file="t", footer={"target_digest": "d"}
            )
        )
        got = pandas_blocks._measured_block((0, 1))
        assert got["worker_setup_peak_bytes"] == 1_100
        assert got["resident_growth_bytes"] == 10


def _result(rows=100, before=1000, resident=None, after=1600, nth=2, own=True):
    return {
        "block": (4, 5),
        "rows": rows,
        "pid": 7,
        "reading": {
            "peak_before_bytes": before,
            "resident_before_bytes": resident,
            "peak_after_bytes": after,
            "nth_in_process": nth,
            "own_process": own,
        },
    }


class TestEveryBlock:
    """What `_block_reading` records of each block, none of it judged."""

    def test_a_block_that_raised_the_peak_grew_from_the_resident_memory(self):
        """Not from the peak before it, which an earlier block set. It went
        red with the growth taken from that peak, 600 here."""
        from cil_regionalization.reduce.pandas_blocks import _block_reading

        got = _block_reading(_result(resident=400))
        assert got["growth_bytes"] == 1200 and got["bytes_per_row"] == 12
        assert got["growth_from"] == "the resident memory before it"
        assert got["unavailable"] is None and got["block"] == [4, 5]

    def test_a_block_that_did_not_raise_the_peak_has_no_growth(self):
        from cil_regionalization.reduce.pandas_blocks import _FLAT_PEAK, _block_reading

        got = _block_reading(_result(resident=400, after=1000))
        assert got["growth_bytes"] is None and got["unavailable"] == _FLAT_PEAK

    def test_without_the_resident_memory_a_worker_s_first_block_alone(self):
        """As on macOS. A later block there would have been read from the
        peak an earlier one set."""
        from cil_regionalization.reduce.pandas_blocks import _block_reading

        first = _block_reading(_result(nth=1))
        assert first["growth_bytes"] == 600
        assert first["growth_from"] == "the peak once the worker had set itself up"
        later = _block_reading(_result(nth=2))
        assert later["growth_bytes"] is None
        assert (
            "the resident memory before the block is not read" in (later["unavailable"])
        )
        planned_in = _block_reading(_result(nth=1, own=False))
        assert planned_in["growth_bytes"] is None

    def test_a_block_without_rows_or_without_a_reading_has_no_growth(self):
        from cil_regionalization.reduce.pandas_blocks import _block_reading

        assert _block_reading(_result(rows=0))["unavailable"] == (
            "the block kept no rows"
        )
        assert _block_reading(_result(before=None))["growth_bytes"] is None

    def test_the_summary_of_every_block(self):
        """Over the measured blocks only, the first being the one that
        starts the run. The median is not the mean here."""
        from cil_regionalization.reduce.pandas_blocks import _over_every_block

        entries = [
            {"block": [0, 1], "bytes_per_row": 100.0},
            {"block": [1, 2], "bytes_per_row": None},
            {"block": [2, 3], "bytes_per_row": 400.0},
            {"block": [3, 4], "bytes_per_row": 200.0},
        ]
        assert _over_every_block(entries, 250) == {
            "blocks": 4,
            "measured": 3,
            "bytes_per_row_min": 100.0,
            "bytes_per_row_median": 200.0,
            "bytes_per_row_max": 400.0,
            "heaviest_block": [2, 3],
            "heaviest_over_estimate": 1.6,
            "first_over_heaviest": 0.25,
        }

    def test_a_summary_with_nothing_measured_says_so(self):
        from cil_regionalization.reduce.pandas_blocks import _over_every_block

        got = _over_every_block([{"block": [0, 1], "bytes_per_row": None}], 250)
        assert got["blocks"] == 1 and got["measured"] == 0
        assert got["heaviest_over_estimate"] is None


class TestJudged:
    """What `_block_size_record` makes of a first block."""

    PLAN = {"by": "estimate", "row_bytes": 280, "groups_in_largest_block": 1}

    @staticmethod
    def _first(rows, growth, unavailable=None):
        return {
            "block": (0, 4),
            "rows": rows,
            "resident_growth_bytes": growth,
            "worker_setup_peak_bytes": None if growth is None else 10**9,
            "peak_after_block_bytes": None if growth is None else 10**9 + growth,
            "measure_unavailable": unavailable,
            "start_method": "fork",
        }

    def _record(self, first, plan=None):
        from cil_regionalization.reduce.pandas_blocks import _block_size_record

        return _block_size_record(_request([10]), plan or self.PLAN, first)

    def test_more_than_twice_the_estimate_is_refused(self):
        got = self._record(self._first(200_000, 200_000 * 561))
        assert got["judged"] and got["refused"] and got["refused_above"] == 2
        assert got["measured_bytes_per_row"] == 561

    @pytest.mark.parametrize("per_row", [281, 560])
    def test_up_to_twice_the_estimate_is_judged_and_passes(self, per_row):
        """A tolerance on a noisy measure. It went red when once the
        estimate was refused."""
        got = self._record(self._first(200_000, 200_000 * per_row))
        assert got["judged"] and not got["refused"]

    def test_below_the_judged_rows_is_recorded_and_not_judged(self):
        got = self._record(self._first(99_999, 99_999 * 10_000))
        assert not got["judged"] and not got["refused"]
        assert got["measured_bytes_per_row"] == 10_000

    def test_without_a_measure_nothing_is_judged(self):
        got = self._record(self._first(200_000, None, unavailable="why"))
        assert got["measured_bytes_per_row"] is None and got["unavailable"] == "why"
        assert not got["judged"] and not got["refused"]

    def test_a_declared_size_is_never_judged(self):
        plan = {**self.PLAN, "by": "declared"}
        got = self._record(self._first(200_000, 200_000 * 10_000), plan)
        assert got["by"] == "declared" and not got["judged"]
        assert got["targets_per_block"] == 128

    def test_the_refusal_names_a_block_that_fits(self):
        from cil_regionalization.reduce.pandas_blocks import _too_heavy

        sizing = self._record(self._first(200_000, 200_000 * 600))
        # 4 row groups of 128 targets took 120 MB, so a target 234,375
        # bytes, and a worker of 1 GB holds 4,266 targets, 33 row groups
        message = _too_heavy(
            _request([10]), {**self.PLAN, "per_worker_gb": 1.0}, sizing
        )
        assert message == (
            "reduce: the first block, row groups 0 to 3, took 600 bytes a kept "
            "row over 200,000 rows, more than 2 times the 280 bytes a row the "
            "blocks were sized at, so a block can take more than a worker's 1 "
            "GB. No statistics were written. The measure is the growth of the "
            "worker's peak resident memory over the block. Declare the size of "
            "a block in [reduce.pandas] targets_per_block, in place of "
            "memory_gb. At the measured size a block of 4224 targets fits a "
            "worker."
        )
