"""Warming levels from a table of global mean surface temperature.

A member is one temperature series, named by the columns of the table
that are not the year or the value, such as a model and a scenario. For
each period a member's series is rebased to a reference period, an
offset is added, and the mean over the period's years is taken. The
member is given the level whose band holds that mean. A mean that lies
in no band gives the member no level in that period.

Nothing here smooths. The series is taken as it is in the table, and
the caller has to say how it was smoothed, so the record of a run says
so too. The reference period and the offset have no default, because
the offset is only right for one reference period and one baseline.
"""

from __future__ import annotations

import hashlib
import io
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, NamedTuple, Optional, Sequence

import numpy as np
import pandas as pd

from cil_regionalization.stats import _window_label

_YEAR, _VALUE = "year", "value"
_TIE_RULE = "the first level declared whose band holds the mean"


class LevelWindow(NamedTuple):
    """A level a member has in a period, with the label of the period
    and the first and last year of the impacts that go under it."""

    level: float
    window: str
    start: int
    end: int


@dataclass(frozen=True)
class Period:
    """A calendar period. ``gmst_span`` is the years the temperature is
    averaged over to pick a member's level, and ``impact_span`` the
    years of the impacts that go under that level. Both include their
    first and last year. They are given apart because a product may
    average its impacts over other years than its temperature."""

    gmst_span: tuple[int, int]
    impact_span: tuple[int, int]

    @property
    def label(self) -> str:
        """The period as a window label, from the years of the impacts."""
        return _window_label(*self.impact_span)


@dataclass(frozen=True)
class GMSTTable:
    """A temperature table with one row for a member and a year.
    ``frame`` has the member columns as strings, then an integer 'year'
    and a float 'value'. ``path`` and ``sha256`` are those of the file
    it was read from, or None if it was built from a frame."""

    frame: pd.DataFrame
    member_cols: tuple[str, ...]
    path: Optional[str] = None
    sha256: Optional[str] = None

    @classmethod
    def from_frame(
        cls,
        frame: pd.DataFrame,
        *,
        member_cols: Sequence[str],
        time_col: str = "year",
        value_col: str = "temp",
        path: Optional[str] = None,
        sha256: Optional[str] = None,
    ) -> "GMSTTable":
        """Check a table and keep its member, year and value columns.
        Raises if a column is missing, a member or a year is missing
        from a row, a year is not a whole number, or a member has two
        rows for one year."""
        member_cols = tuple(member_cols)
        if not member_cols:
            raise ValueError("member_cols must name at least one column")
        wanted = [*member_cols, time_col, value_col]
        if len(set(wanted)) != len(wanted):
            raise ValueError(f"the columns {wanted} must be distinct")
        absent = [c for c in wanted if c not in frame.columns]
        if absent:
            raise ValueError(
                f"the temperature table has no column {absent}; it has "
                f"{list(frame.columns)}"
            )
        kept = frame[wanted].reset_index(drop=True)
        empty = kept[[*member_cols, time_col]].isna().any(axis=1)
        if empty.any():
            raise ValueError(
                f"{int(empty.sum())} rows of the temperature table name no "
                f"member or no year"
            )
        years = pd.to_numeric(kept[time_col]).to_numpy(dtype=np.float64)
        if (years != np.floor(years)).any():
            raise ValueError(
                f"column {time_col!r} of the temperature table has to hold whole years"
            )
        out = pd.DataFrame({name: kept[name].astype(str) for name in member_cols})
        out[_YEAR] = years.astype(np.int64)
        out[_VALUE] = pd.to_numeric(kept[value_col]).to_numpy(dtype=np.float64)
        repeated = out.duplicated(subset=[*member_cols, _YEAR])
        if repeated.any():
            first = out.loc[repeated, [*member_cols, _YEAR]].iloc[0].tolist()
            raise ValueError(
                f"{int(repeated.sum())} rows of the temperature table repeat "
                f"a member and a year (first: {first})"
            )
        return cls(frame=out, member_cols=member_cols, path=path, sha256=sha256)

    @property
    def members(self) -> list[tuple[str, ...]]:
        """The members, in the order they first appear in the table."""
        distinct = self.frame[list(self.member_cols)].drop_duplicates()
        return list(distinct.itertuples(index=False, name=None))

    def series(self) -> dict[tuple[str, ...], dict[int, float]]:
        """The value at each year of each member."""
        out: dict[tuple[str, ...], dict[int, float]] = {}
        columns = [self.frame[c] for c in self.member_cols]
        for *member, year, value in zip(
            *columns, self.frame[_YEAR], self.frame[_VALUE]
        ):
            out.setdefault(tuple(member), {})[int(year)] = float(value)
        return out


def read_gmst(
    path: str | Path,
    *,
    member_cols: Sequence[str],
    time_col: str = "year",
    value_col: str = "temp",
    sep: str = ",",
) -> GMSTTable:
    """Read a temperature table. The file is read once, and the hash is
    of the bytes the rows were parsed from. The member columns are read
    as text, so a name like '06' keeps its zero."""
    raw = Path(path).read_bytes()
    frame = pd.read_csv(
        io.BytesIO(raw), sep=sep, dtype={name: str for name in member_cols}
    )
    return GMSTTable.from_frame(
        frame,
        member_cols=member_cols,
        time_col=time_col,
        value_col=value_col,
        path=str(path),
        sha256=hashlib.sha256(raw).hexdigest(),
    )


def _level_label(level: float) -> str:
    return f"{level:g}"


def _span(name: str, span) -> tuple[int, int]:
    try:
        first, last = span
    except (TypeError, ValueError):
        raise ValueError(f"{name} has to be a first and a last year; got {span!r}")
    if first != int(first) or last != int(last):
        raise ValueError(f"{name} has to hold whole years; got {span!r}")
    if first > last:
        raise ValueError(f"{name} starts in {first}, after it ends in {last}")
    return int(first), int(last)


@dataclass(frozen=True)
class WarmingLevels:
    """The level of every member of a temperature table in every
    period, and what decided it. ``means`` and ``levels`` hold one
    value a period for each member, in the order of ``periods``. A
    level is None where the member's mean lies in no band."""

    table: GMSTTable
    reference_period: tuple[int, int]
    offset: float
    levels_declared: tuple[float, ...]
    half_width: float
    periods: tuple[Period, ...]
    smoothing: str
    means: dict[tuple[str, ...], tuple[float, ...]]
    levels: dict[tuple[str, ...], tuple[Optional[float], ...]]

    @property
    def member_cols(self) -> tuple[str, ...]:
        return self.table.member_cols

    @property
    def assignments(self) -> pd.DataFrame:
        """One row for a member and a period, with the member columns,
        'window', 'gmst_mean' and 'warming_level', which is NaN where
        the member has no level."""
        rows = [
            (*member, period.label, self.means[member][i], self.levels[member][i])
            for member in self.means
            for i, period in enumerate(self.periods)
        ]
        frame = pd.DataFrame(
            rows, columns=[*self.member_cols, "window", "gmst_mean", "warming_level"]
        )
        frame["warming_level"] = frame["warming_level"].astype(np.float64)
        return frame

    def _known(self, member: Sequence[str]) -> tuple[str, ...]:
        member = tuple(member)
        if member not in self.levels:
            raise ValueError(
                f"{dict(zip(self.member_cols, member))} has no row in the "
                f"temperature table"
                + (f" {self.table.path}" if self.table.path else "")
            )
        return member

    def missing(self, members: Iterable[Sequence[str]]) -> list[tuple[str, ...]]:
        """The members that have no row in the table, sorted."""
        return sorted({tuple(m) for m in members} - set(self.levels))

    def windows_of(self, member: Sequence[str]) -> tuple[LevelWindow, ...]:
        """The periods in which a member has a level, in the order of
        ``periods``. Raises if the member has no row in the table."""
        member = self._known(member)
        return tuple(
            LevelWindow(level, period.label, *period.impact_span)
            for period, level in zip(self.periods, self.levels[member])
            if level is not None
        )

    def record(self, members: Optional[Iterable[Sequence[str]]] = None) -> dict:
        """What a manifest says of a run, as plain values. ``members``
        are those the run used, and every member of the table if it is
        left out. The rest of the table is listed under
        ``unused_members``. Raises if a member has no row."""
        used = (
            list(self.levels)
            if members is None
            else list(dict.fromkeys(self._known(m) for m in members))
        )
        in_run = set(used)
        labels = [period.label for period in self.periods]

        def named(member: tuple[str, ...]) -> dict:
            return dict(zip(self.member_cols, member))

        counts = {
            label: {_level_label(level): 0 for level in self.levels_declared}
            for label in labels
        }
        unassigned: dict[str, list[dict]] = {label: [] for label in labels}
        entries = []
        for member in used:
            for label, level in zip(labels, self.levels[member]):
                if level is None:
                    unassigned[label].append(named(member))
                else:
                    counts[label][_level_label(level)] += 1
            entries.append(
                {
                    "member": named(member),
                    "gmst_mean": dict(zip(labels, self.means[member])),
                    "warming_level": dict(zip(labels, self.levels[member])),
                }
            )
        return {
            "path": self.table.path,
            "sha256": self.table.sha256,
            "member_cols": list(self.member_cols),
            "smoothing": self.smoothing,
            "reference_period": list(self.reference_period),
            "offset": self.offset,
            "half_width": self.half_width,
            "levels": list(self.levels_declared),
            "tie": _TIE_RULE,
            "periods": [
                {
                    "window": period.label,
                    "gmst_span": list(period.gmst_span),
                    "impact_span": list(period.impact_span),
                }
                for period in self.periods
            ],
            "members": entries,
            "n_members": counts,
            "unassigned": unassigned,
            "in_no_bin": [
                named(member)
                for member in used
                if all(level is None for level in self.levels[member])
            ],
            "unused_members": [
                named(member) for member in self.levels if member not in in_run
            ],
        }


def assign_warming_levels(
    table: GMSTTable,
    *,
    reference_period: tuple[int, int],
    offset: float,
    levels: Sequence[float],
    half_width: float,
    periods: Sequence[Period],
    smoothing: str,
) -> WarmingLevels:
    """The level of every member in every period.

    A member's series is rebased by taking off its own mean over
    ``reference_period`` and adding ``offset``. Its mean over a
    period's ``gmst_span`` then picks the level. A level holds a mean
    from ``half_width`` below it to ``half_width`` above it, both ends
    included. Two bands may share an end and may not overlap. A mean
    on a shared end goes to the level declared first.

    ``smoothing`` says how the series in the table was smoothed, in
    words, and goes into the record as it is. Nothing is smoothed
    here. Raises if a member lacks a value for a year of the reference
    period or of a ``gmst_span``.
    """
    reference_period = _span("reference_period", reference_period)
    offset = float(offset)
    half_width = float(half_width)
    if not math.isfinite(offset):
        raise ValueError(f"offset has to be a number; got {offset}")
    if not math.isfinite(half_width) or half_width <= 0:
        raise ValueError(f"half_width has to be above zero; got {half_width}")
    if not isinstance(smoothing, str) or not smoothing.strip():
        raise ValueError(
            "smoothing has to say how the series in the table was smoothed. "
            "Write 'unknown' if nobody knows."
        )
    levels = tuple(float(level) for level in levels)
    if not levels:
        raise ValueError("levels must list at least one warming level")
    if len(set(levels)) != len(levels) or not all(map(math.isfinite, levels)):
        raise ValueError(f"levels have to be distinct numbers; got {list(levels)}")
    ordered = sorted(levels)
    for below, above in zip(ordered, ordered[1:]):
        if below + half_width > above - half_width:
            raise ValueError(
                f"the bands of levels {below:g} and {above:g} overlap with "
                f"half_width {half_width:g}. A mean in both would have two "
                f"levels."
            )
    periods = tuple(periods)
    if not periods:
        raise ValueError("periods must list at least one period")
    for period in periods:
        _span("gmst_span", period.gmst_span)
        _span("impact_span", period.impact_span)
    labels = [period.label for period in periods]
    if len(set(labels)) != len(labels):
        raise ValueError(
            f"two periods have the same impact years, so the same label: {labels}"
        )

    def years_of(span) -> list[int]:
        return list(range(int(span[0]), int(span[1]) + 1))

    def values(member, series: dict[int, float], span, what: str) -> list[float]:
        years = years_of(span)
        lacking = [y for y in years if y not in series or math.isnan(series[y])]
        if lacking:
            raise ValueError(
                f"{dict(zip(table.member_cols, member))} has no temperature "
                f"for {len(lacking)} of the {len(years)} years of {what} "
                f"{span[0]} to {span[1]} (first: {lacking[:5]})"
            )
        return [series[y] for y in years]

    means: dict[tuple[str, ...], tuple[float, ...]] = {}
    assigned: dict[tuple[str, ...], tuple[Optional[float], ...]] = {}
    for member, series in table.series().items():
        base = values(member, series, reference_period, "the reference period")
        reference = math.fsum(base) / len(base)
        member_means, member_levels = [], []
        for period in periods:
            span = values(member, series, period.gmst_span, "the period")
            # each year is rebased, and then the mean is taken
            rebased = [(value - reference) + offset for value in span]
            mean = math.fsum(rebased) / len(rebased)
            member_means.append(mean)
            member_levels.append(
                next(
                    (
                        level
                        for level in levels
                        if level - half_width <= mean <= level + half_width
                    ),
                    None,
                )
            )
        means[member] = tuple(member_means)
        assigned[member] = tuple(member_levels)
    return WarmingLevels(
        table=table,
        reference_period=reference_period,
        offset=offset,
        levels_declared=levels,
        half_width=half_width,
        periods=periods,
        smoothing=smoothing,
        means=means,
        levels=assigned,
    )
