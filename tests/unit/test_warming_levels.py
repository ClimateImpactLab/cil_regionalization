"""Warming levels from a temperature table.

Every table here is synthetic. The values are exact in binary, and the
tests of a band's ends use a period of one year, so a mean is the one
subtraction and the one addition that make it and nothing is rounded
on the way.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cil_regionalization.warming_levels import (
    EVERY_DECLARED,
    GMSTTable,
    LevelWindow,
    Period,
    assign_warming_levels,
    members_that_join,
    read_gmst,
)

REFERENCE = (1986, 2005)
MID = Period(gmst_span=(2040, 2059), impact_span=(2040, 2058))
LATE = Period(gmst_span=(2080, 2099), impact_span=(2080, 2099))
ONE_YEAR = Period(gmst_span=(2050, 2050), impact_span=(2040, 2058))


def _series(reference: float, mid: float, late: float) -> dict[int, float]:
    """A value over the reference period, one over 2040 to 2059 and one
    over 2080 to 2099, with other values in the years between."""
    out = {year: -9.0 for year in range(1981, 2101)}
    out.update({year: reference for year in range(1986, 2006)})
    out.update({year: mid for year in range(2040, 2060)})
    out.update({year: late for year in range(2080, 2100)})
    return out


def _table(members: dict, member_cols=("gcm", "rcp")) -> GMSTTable:
    rows = [
        (*member, year, value)
        for member, series in members.items()
        for year, value in series.items()
    ]
    frame = pd.DataFrame(rows, columns=[*member_cols, "year", "temp"])
    return GMSTTable.from_frame(frame, member_cols=member_cols)


def _assign(table, **changes):
    """With every member of the table joining unless ``members`` says
    otherwise."""
    settings = dict(
        reference_period=REFERENCE,
        offset=0.5,
        levels=[1.5, 2.0, 3.0],
        half_width=0.25,
        periods=[MID, LATE],
        smoothing="unknown",
        members=table.members,
    )
    settings.update(changes)
    return assign_warming_levels(table, **settings)


class TestRebasing:
    def test_a_member_is_rebased_to_its_own_reference_mean_and_offset(self):
        """Two members with the same temperatures after 2040 and other
        reference means land on other levels."""
        table = _table(
            {
                ("a", "rcp45"): _series(0.25, 1.25, 2.75),
                ("b", "rcp45"): _series(-0.25, 1.25, 2.75),
            }
        )
        got = _assign(table)
        # 1.25 - 0.25 + 0.5 and 2.75 - 0.25 + 0.5
        assert got.means[("a", "rcp45")] == (1.5, 3.0)
        assert got.levels[("a", "rcp45")] == (1.5, 3.0)
        # 1.25 + 0.25 + 0.5, and 3.5, which no band holds
        assert got.means[("b", "rcp45")] == (2.0, 3.5)
        assert got.levels[("b", "rcp45")] == (2.0, None)

    def test_the_reference_is_the_mean_of_every_year_of_the_period(self):
        """The reference years go -0.75, -0.25, 0.25 and 0.75 five times
        over, so their mean is 0 and their first year is -0.75."""
        series = _series(0.0, 1.0, 2.5)
        series.update({1986 + i: (-0.75, -0.25, 0.25, 0.75)[i % 4] for i in range(20)})
        got = _assign(_table({("a", "rcp45"): series}))
        assert got.means[("a", "rcp45")] == (1.5, 3.0)
        assert got.levels[("a", "rcp45")] == (1.5, 3.0)

    def test_another_reference_period_and_offset_give_other_levels(self):
        """The same table against 2040 to 2059 with no offset. Both are
        values, and neither has a default."""
        table = _table({("a", "rcp45"): _series(0.25, 1.25, 2.75)})
        got = _assign(table, reference_period=(2040, 2059), offset=0.0)
        assert got.means[("a", "rcp45")] == (0.0, 1.5)
        assert got.levels[("a", "rcp45")] == (None, 1.5)
        assert got.reference_period == (2040, 2059) and got.offset == 0.0

    @pytest.mark.parametrize(
        "left_out",
        [
            "reference_period",
            "offset",
            "levels",
            "half_width",
            "periods",
            "smoothing",
            "members",
        ],
    )
    def test_nothing_that_decides_a_level_has_a_default(self, left_out):
        table = _table({("a", "rcp45"): _series(0.25, 1.25, 2.75)})
        settings = dict(
            reference_period=REFERENCE,
            offset=0.5,
            levels=[1.5],
            half_width=0.25,
            periods=[MID],
            smoothing="unknown",
            members=[("a", "rcp45")],
        )
        del settings[left_out]
        with pytest.raises(TypeError, match=left_out):
            assign_warming_levels(table, **settings)

    def test_the_series_is_taken_as_it_is(self):
        """A jump of four degrees in one year stays in the mean. Nothing
        is smoothed."""
        series = _series(0.0, 1.0, 1.0)
        series[2059] = 6.0
        got = _assign(
            _table({("a", "rcp45"): series}),
            offset=0.0,
            levels=[1.25],
            periods=[MID],
        )
        assert got.means[("a", "rcp45")] == (1.25,)

    def test_the_order_of_the_rows_changes_nothing(self):
        rng = np.random.default_rng(3)
        members = {
            (f"m{i}", rcp): {year: float(rng.normal()) for year in range(1981, 2101)}
            for i in range(4)
            for rcp in ("rcp45", "rcp85")
        }
        table = _table(members)
        shuffled = GMSTTable.from_frame(
            table.frame.sample(frac=1.0, random_state=1).rename(
                columns={"value": "temp"}
            ),
            member_cols=("gcm", "rcp"),
        )
        one, two = _assign(table), _assign(shuffled)
        assert one.means == {member: two.means[member] for member in one.means}
        assert list(two.means) != list(one.means)


class TestBands:
    @staticmethod
    def _level_at(mean: float, **changes):
        """The level of a member whose rebased temperature in the one
        year of the period is ``mean``."""
        series = _series(0.0, 0.0, 0.0)
        series[2050] = mean
        got = _assign(
            _table({("a", "rcp45"): series}),
            offset=0.0,
            periods=[ONE_YEAR],
            **changes,
        )
        assert got.means[("a", "rcp45")] == (mean,)
        return got.levels[("a", "rcp45")][0]

    def test_a_band_holds_both_its_ends(self):
        assert self._level_at(2.75) == 3.0
        assert self._level_at(3.25) == 3.0
        assert self._level_at(math.nextafter(2.75, 0.0)) is None
        assert self._level_at(math.nextafter(3.25, 9.0)) is None

    def test_a_mean_between_two_bands_has_no_level(self):
        assert self._level_at(2.5) is None
        assert self._level_at(math.nextafter(2.25, 9.0)) is None
        assert self._level_at(0.0) is None

    def test_a_mean_on_an_end_two_bands_share_goes_to_the_level_declared_first(self):
        """1.75 ends the band of 1.5 and starts the band of 2."""
        assert self._level_at(1.75, levels=[1.5, 2.0, 3.0]) == 1.5
        assert self._level_at(1.75, levels=[2.0, 1.5, 3.0]) == 2.0
        assert self._level_at(math.nextafter(1.75, 9.0)) == 2.0
        assert self._level_at(math.nextafter(1.75, 0.0)) == 1.5

    def test_bands_that_overlap_are_refused(self):
        table = _table({("a", "rcp45"): _series(0.25, 1.25, 2.75)})
        with pytest.raises(
            ValueError,
            match="the bands of levels 1.5 and 2 overlap with half_width 0.3",
        ):
            _assign(table, half_width=0.3)
        # the order they are declared in does not hide it
        with pytest.raises(ValueError, match="levels 1.5 and 2 overlap"):
            _assign(table, levels=[3.0, 2.0, 1.5], half_width=0.3)
        with pytest.raises(ValueError, match="distinct numbers"):
            _assign(table, levels=[1.5, 2.0, 1.5])
        with pytest.raises(ValueError, match="at least one warming level"):
            _assign(table, levels=[])
        with pytest.raises(ValueError, match="half_width has to be above zero"):
            _assign(table, half_width=0.0)

    def test_the_smoothing_has_to_be_stated(self):
        table = _table({("a", "rcp45"): _series(0.25, 1.25, 2.75)})
        for nothing in ("", "   ", None):
            with pytest.raises(ValueError, match="Write 'unknown' if nobody knows"):
                _assign(table, smoothing=nothing)
        assert _assign(table, smoothing="unknown").record()["smoothing"] == "unknown"


class TestPeriods:
    def test_a_member_has_one_level_a_period_and_they_may_differ(self):
        table = _table(
            {
                ("a", "rcp45"): _series(0.25, 1.25, 2.75),
                ("a", "rcp85"): _series(0.25, 1.75, 5.0),
                ("b", "rcp85"): _series(0.25, 9.0, 9.0),
            }
        )
        got = _assign(table)
        assert got.windows_of(("a", "rcp45")) == (
            LevelWindow(1.5, "2040-2058", 2040, 2058),
            LevelWindow(3.0, "2080-2099", 2080, 2099),
        )
        # a level in the first period and none in the second
        assert got.windows_of(("a", "rcp85")) == (
            LevelWindow(2.0, "2040-2058", 2040, 2058),
        )
        assert got.windows_of(("b", "rcp85")) == ()
        frame = got.assignments
        assert list(frame.columns) == [
            "gcm",
            "rcp",
            "window",
            "gmst_mean",
            "warming_level",
        ]
        assert len(frame) == 6
        assert frame["warming_level"].dtype == np.float64
        assert frame["warming_level"].isna().sum() == 3

    def test_the_level_comes_from_the_gmst_years_and_the_window_from_the_impact_years(
        self,
    ):
        """2059 is four degrees warmer here. With it the mean is 1.25,
        and over the nineteen years before it the mean is 1."""
        series = _series(0.0, 1.0, 1.0)
        series[2059] = 6.0
        table = _table({("a", "rcp45"): series})
        twenty = Period(gmst_span=(2040, 2059), impact_span=(2040, 2058))
        nineteen = Period(gmst_span=(2040, 2058), impact_span=(2040, 2059))
        settings = dict(offset=0.0, levels=[1.0, 1.25], half_width=0.0625)
        got = _assign(table, periods=[twenty], **settings)
        assert got.windows_of(("a", "rcp45")) == (
            LevelWindow(1.25, "2040-2058", 2040, 2058),
        )
        got = _assign(table, periods=[nineteen], **settings)
        assert got.windows_of(("a", "rcp45")) == (
            LevelWindow(1.0, "2040-2059", 2040, 2059),
        )

    def test_a_period_of_one_year_is_labelled_by_the_year(self):
        assert Period(gmst_span=(2090, 2099), impact_span=(2099, 2099)).label == "2099"
        assert MID.label == "2040-2058"

    def test_two_periods_with_the_same_impact_years_are_refused(self):
        table = _table({("a", "rcp45"): _series(0.25, 1.25, 2.75)})
        again = Period(gmst_span=(2041, 2060), impact_span=MID.impact_span)
        with pytest.raises(ValueError, match="the same label"):
            _assign(table, periods=[MID, again])
        with pytest.raises(ValueError, match="at least one period"):
            _assign(table, periods=[])
        backwards = Period(gmst_span=(2059, 2040), impact_span=(2040, 2058))
        with pytest.raises(ValueError, match="gmst_span starts in 2059"):
            _assign(table, periods=[backwards])

    def test_a_member_without_a_year_it_needs_is_refused_by_name(self):
        gap = _series(0.25, 1.25, 2.75)
        del gap[1990]
        with pytest.raises(
            ValueError,
            match=r"\{'gcm': 'a', 'rcp': 'rcp45'\} has no temperature for 1 of "
            r"the 20 years of the reference period 1986 to 2005 \(first: \[1990\]\)",
        ):
            _assign(_table({("a", "rcp45"): gap}))
        nan = _series(0.25, 1.25, 2.75)
        nan[2085] = float("nan")
        nan[2086] = float("nan")
        with pytest.raises(
            ValueError,
            match=r"no temperature for 2 of the 20 years of the period 2080 to "
            r"2099 \(first: \[2085, 2086\]\)",
        ):
            _assign(_table({("b", "rcp85"): nan}))

    def test_a_year_no_span_needs_may_be_missing(self):
        series = _series(0.25, 1.25, 2.75)
        del series[2070]
        series[2071] = float("nan")
        got = _assign(_table({("a", "rcp45"): series}))
        assert got.levels[("a", "rcp45")] == (1.5, 3.0)


class TestMembers:
    def test_the_member_key_is_whatever_columns_the_table_has(self):
        """A model and an ssp, and a key of three columns."""
        table = _table(
            {("a", "ssp245"): _series(0.25, 1.25, 2.75)}, member_cols=("model", "ssp")
        )
        got = _assign(table)
        assert got.member_cols == ("model", "ssp")
        assert got.windows_of(("a", "ssp245"))[0].level == 1.5
        assert got.record()["members"][0]["member"] == {"model": "a", "ssp": "ssp245"}
        three = _table(
            {("a", "ssp245", "r1"): _series(0.25, 1.25, 2.75)},
            member_cols=("model", "ssp", "run"),
        )
        assert _assign(three).levels[("a", "ssp245", "r1")] == (1.5, 3.0)

    def test_a_name_is_matched_as_it_is_written(self):
        """A member named after another is a member of its own, and the
        other's row does not stand in for it."""
        table = _table(
            {
                ("GFDL-ESM2G", "rcp85"): _series(0.25, 1.25, 2.75),
                ("surrogate_GFDL-ESM2G_06", "rcp85"): _series(0.25, 1.75, 4.75),
            }
        )
        got = _assign(table)
        assert got.levels[("GFDL-ESM2G", "rcp85")] == (1.5, 3.0)
        assert got.levels[("surrogate_GFDL-ESM2G_06", "rcp85")] == (2.0, None)
        assert got.missing(
            [
                ("gfdl-esm2g", "rcp85"),
                ("surrogate_GFDL-ESM2G_11", "rcp85"),
                ("GFDL-ESM2G", "rcp85"),
            ]
        ) == [("gfdl-esm2g", "rcp85"), ("surrogate_GFDL-ESM2G_11", "rcp85")]
        with pytest.raises(
            ValueError,
            match=r"\{'gcm': 'surrogate_GFDL-ESM2G_11', 'rcp': 'rcp85'\} has no "
            r"row in the temperature table",
        ):
            got.windows_of(("surrogate_GFDL-ESM2G_11", "rcp85"))

    def test_a_table_with_a_member_and_a_year_twice_is_refused(self):
        frame = pd.DataFrame(
            {
                "gcm": ["a", "a"],
                "rcp": ["rcp45", "rcp45"],
                "year": [2000, 2000],
                "temp": [0.1, 0.2],
            }
        )
        with pytest.raises(ValueError, match=r"repeat a member and a year"):
            GMSTTable.from_frame(frame, member_cols=("gcm", "rcp"))
        with pytest.raises(ValueError, match=r"has no column \['ssp'\]"):
            GMSTTable.from_frame(frame.iloc[:1], member_cols=("gcm", "ssp"))
        with pytest.raises(ValueError, match="name no member or no year"):
            GMSTTable.from_frame(
                frame.iloc[:1].assign(gcm=None), member_cols=("gcm", "rcp")
            )
        with pytest.raises(ValueError, match="whole years"):
            GMSTTable.from_frame(
                frame.iloc[:1].assign(year=2000.5), member_cols=("gcm", "rcp")
            )
        with pytest.raises(ValueError, match="at least one column"):
            GMSTTable.from_frame(frame.iloc[:1], member_cols=())


class TestRecord:
    @staticmethod
    def _levels(**changes):
        table = _table(
            {
                ("a", "rcp45"): _series(0.25, 1.25, 2.75),
                ("a", "rcp85"): _series(0.25, 1.75, 5.0),
                ("b", "rcp85"): _series(0.25, 9.0, 9.0),
            }
        )
        return _assign(
            table,
            levels=[1.5, 2.0, 3.0, 4.0],
            smoothing="as delivered, window not recorded",
            **changes,
        )

    def test_it_says_what_decided_the_levels_and_what_they_are(self):
        record = self._levels().record()
        assert json.loads(json.dumps(record)) == record
        assert record["path"] is None and record["sha256"] is None
        assert record["member_cols"] == ["gcm", "rcp"]
        assert record["smoothing"] == "as delivered, window not recorded"
        assert record["reference_period"] == [1986, 2005]
        assert record["offset"] == 0.5
        assert record["half_width"] == 0.25
        assert record["levels"] == [1.5, 2.0, 3.0, 4.0]
        assert record["tie"] == "the first level declared whose band holds the mean"
        assert record["periods"] == [
            {
                "window": "2040-2058",
                "gmst_span": [2040, 2059],
                "impact_span": [2040, 2058],
            },
            {
                "window": "2080-2099",
                "gmst_span": [2080, 2099],
                "impact_span": [2080, 2099],
            },
        ]
        assert record["members"][1] == {
            "member": {"gcm": "a", "rcp": "rcp85"},
            "gmst_mean": {"2040-2058": 2.0, "2080-2099": 5.25},
            "warming_level": {"2040-2058": 2.0, "2080-2099": None},
        }

    def test_it_counts_every_declared_level_and_lists_who_has_none(self):
        """A level nobody reached is there with a zero, in a period and
        in every period, as 4 is here."""
        record = self._levels().record()
        assert record["n_members"] == {
            "2040-2058": {"1.5": 1, "2": 1, "3": 0, "4": 0},
            "2080-2099": {"1.5": 0, "2": 0, "3": 1, "4": 0},
        }
        assert record["unassigned"] == {
            "2040-2058": [{"gcm": "b", "rcp": "rcp85"}],
            "2080-2099": [
                {"gcm": "a", "rcp": "rcp85"},
                {"gcm": "b", "rcp": "rcp85"},
            ],
        }
        assert record["in_no_bin"] == [{"gcm": "b", "rcp": "rcp85"}]
        assert record["excluded"] == []

    def test_a_member_that_does_not_join_is_listed_with_the_reason(self):
        """Apart from the members that join, in the run or not."""
        levels = self._levels(members=[("a", "rcp85"), ("a", "rcp45")])
        b = {"gcm": "b", "rcp": "rcp85"}
        record = levels.record([("a", "rcp45"), ("b", "rcp85"), ("a", "rcp85")])
        assert [entry["member"]["rcp"] for entry in record["members"]] == [
            "rcp85",
            "rcp45",
        ]
        assert record["n_members"]["2040-2058"] == {
            "1.5": 1,
            "2": 1,
            "3": 0,
            "4": 0,
        }
        assert record["unassigned"]["2040-2058"] == []
        assert record["in_no_bin"] == []
        assert record["excluded"] == [
            {"member": b, "reason": "in the run and not among the members that join"}
        ]
        assert levels.record()["excluded"] == [
            {"member": b, "reason": "in the temperature table and not in the run"}
        ]
        with pytest.raises(ValueError, match="has no row in the temperature table"):
            levels.record([("a", "rcp85"), ("a", "rcp45"), ("c", "rcp45")])
        with pytest.raises(
            ValueError,
            match=r"1 members that join are not in the run \(first: "
            r"\{'gcm': 'a', 'rcp': 'rcp45'\}\)",
        ):
            levels.record([("a", "rcp85")])


class TestRead:
    def test_the_file_is_read_once_and_hashed(self, tmp_path):
        import hashlib

        rows = ["gcm,rcp,year,temp"]
        for year, value in _series(0.25, 1.25, 2.75).items():
            rows.append(f"06,rcp45,{year},{value}")
        path = tmp_path / "gmst.csv"
        # line ends as an old spreadsheet export writes them
        path.write_bytes("\r\n".join(rows).encode())
        table = read_gmst(path, member_cols=["gcm", "rcp"])
        assert table.path == str(path)
        assert table.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
        # a name of digits stays text, with its zero
        assert table.members == [("06", "rcp45")]
        assert list(table.frame.columns) == ["gcm", "rcp", "year", "value"]
        assert table.frame["year"].dtype == np.int64
        got = _assign(table)
        assert got.levels[("06", "rcp45")] == (1.5, 3.0)
        assert got.record()["sha256"] == table.sha256

    def test_other_column_names_and_another_separator(self, tmp_path):
        path = tmp_path / "gmst.tsv"
        path.write_text(
            "model\tssp\ttime\ttas\n"
            + "".join(
                f"a\tssp245\t{year}\t{value}\n"
                for year, value in _series(0.25, 1.25, 2.75).items()
            )
        )
        table = read_gmst(
            path,
            member_cols=["model", "ssp"],
            time_col="time",
            value_col="tas",
            sep="\t",
        )
        assert _assign(table).levels[("a", "ssp245")] == (1.5, 3.0)


class TestMembersThatJoin:
    """The caller names the members that join, and no name decides
    anything."""

    @staticmethod
    def _table():
        series = _series(0.25, 1.25, 2.75)
        empty = {year: float("nan") for year in series}
        return _table(
            {
                ("GFDL-ESM2G", "rcp45"): series,
                ("surrogate_GFDL-ESM2G_06", "rcp45"): empty,
                ("surrogate_GFDL-ESM2G_06", "rcp85"): _series(0.25, 1.75, 4.75),
            }
        )

    def test_only_the_members_that_join_are_assessed(self):
        """The rest of the table is at no level and recorded, the member
        with no temperature among them."""
        table = self._table()
        got = _assign(table, members=[("GFDL-ESM2G", "rcp45")])
        assert list(got.levels) == [("GFDL-ESM2G", "rcp45")]
        assert got.excluded == (
            ("surrogate_GFDL-ESM2G_06", "rcp45"),
            ("surrogate_GFDL-ESM2G_06", "rcp85"),
        )
        assert got.windows_of(("surrogate_GFDL-ESM2G_06", "rcp85")) == ()
        assert [entry["member"] for entry in got.record()["excluded"]] == [
            {"gcm": "surrogate_GFDL-ESM2G_06", "rcp": "rcp45"},
            {"gcm": "surrogate_GFDL-ESM2G_06", "rcp": "rcp85"},
        ]

    def test_a_member_named_for_a_surrogate_joins_when_it_is_named(self):
        """It went red on the code that read the prefix, where a member
        whose name began with surrogate_ could be left out by a
        setting."""
        got = _assign(self._table(), members=[("surrogate_GFDL-ESM2G_06", "rcp85")])
        assert got.levels == {("surrogate_GFDL-ESM2G_06", "rcp85"): (2.0, None)}
        assert ("GFDL-ESM2G", "rcp45") in got.excluded

    def test_the_module_reads_no_member_name(self):
        """No prefix and no name convention, in the library or in the
        pipeline. It went red on the code that recognised surrogate_."""
        import cil_regionalization.pipelines.montecarlo as pipeline
        import cil_regionalization.warming_levels as module

        for source in (module, pipeline):
            assert "surrogate" not in Path(source.__file__).read_text().lower()

    def test_a_member_that_joins_without_a_temperature_is_refused_by_name(self):
        with pytest.raises(
            ValueError,
            match=r"\{'gcm': 'surrogate_GFDL-ESM2G_06', 'rcp': 'rcp45'\} has no "
            r"temperature for 20 of the 20",
        ):
            _assign(self._table())

    def test_a_member_that_joins_without_a_row_is_refused_by_name(self):
        with pytest.raises(
            ValueError,
            match=r"1 of the members that join have no row in the temperature "
            r"table \(first: \{'gcm': 'CCSM4', 'rcp': 'rcp45'\}\)",
        ):
            _assign(
                self._table(), members=[("CCSM4", "rcp45"), ("GFDL-ESM2G", "rcp45")]
            )

    @pytest.mark.parametrize(
        "members, message",
        [
            ([], "at least one member that joins"),
            ([("GFDL-ESM2G",)], r"one value for each of \['gcm', 'rcp'\]"),
            ("GFDL-ESM2G", "got the string 'GFDL-ESM2G'"),
        ],
    )
    def test_members_that_name_no_member_are_refused(self, members, message):
        with pytest.raises(ValueError, match=message):
            _assign(self._table(), members=members)


class TestJoin:
    """`members_that_join` turns what a run declares into the members."""

    COLS = ["gcm", "rcp"]
    DECLARED = [("a", "rcp45"), ("b", "rcp45"), ("a", "rcp85"), ("s_a", "rcp85")]

    def test_declared_is_every_member_declared_in_their_order(self):
        assert EVERY_DECLARED == "declared"
        assert members_that_join("declared", self.DECLARED, self.COLS) == self.DECLARED

    def test_a_list_by_column_takes_its_values_under_every_other_value(self):
        """The models listed join under each scenario the tree has."""
        assert members_that_join({"gcm": ["b", "a"]}, self.DECLARED, self.COLS) == [
            ("a", "rcp45"),
            ("b", "rcp45"),
            ("a", "rcp85"),
        ]
        assert members_that_join(
            {"gcm": ["a"], "rcp": ["rcp85"]}, self.DECLARED, self.COLS
        ) == [("a", "rcp85")]

    @pytest.mark.parametrize(
        "join, message",
        [
            ("all", "join is 'declared' or a mapping"),
            ({}, "join names no member column"),
            (
                {"model": ["a"]},
                r"join names \['model'\], which are not among the member "
                r"columns \['gcm', 'rcp'\]",
            ),
            ({"gcm": "a"}, "join.gcm has to be a list of values"),
            ({"gcm": []}, "join.gcm lists no value"),
            ({"gcm": ["a", "b", "a"]}, r"join.gcm lists \['a'\] more than once"),
        ],
    )
    def test_a_join_that_is_not_precise_is_refused(self, join, message):
        with pytest.raises(ValueError, match=message):
            members_that_join(join, self.DECLARED, self.COLS)

    def test_a_value_listed_that_no_member_declared_holds_is_refused_by_name(self):
        with pytest.raises(
            ValueError,
            match=re.escape(
                "2 values listed to join are held by no member declared: "
                "[\"gcm = 'c'\", \"rcp = 'rcp26'\"]"
            ),
        ):
            members_that_join(
                {"gcm": ["a", "c"], "rcp": ["rcp26", "rcp45"]},
                self.DECLARED,
                self.COLS,
            )
