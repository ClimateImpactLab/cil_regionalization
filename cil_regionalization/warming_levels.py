"""Warming levels from a table of global mean surface temperature.

A member is one temperature series, named by the columns of the table
that are not the year or the value, such as a model and a scenario. For
each period a member's series is rebased to a reference period, an
offset is added, and the mean over the period's years is taken. The
member is given the level whose band holds that mean. A mean that lies
in no band gives the member no level in that period.

Nothing here smooths. The series is taken as it is in the table, and
the caller has to say how it was smoothed, so the record of a run says
so too. The reference period and the offset have no default, and
neither have the members that join. Nothing here reads a member's name
to decide what the member is.
"""

# The reference period and the offset have no default because the
# offset is only right for one reference period and one baseline.

from __future__ import annotations

import hashlib
import io
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, NamedTuple, Optional, Sequence

import numpy as np
import pandas as pd

from cil_regionalization.stats import _window_label

_YEAR, _VALUE = "year", "value"
_TIE_RULE = "the first level declared whose band holds the mean"
# the value of `join` under which every member declared joins
EVERY_DECLARED = "declared"
# why a member of the table is at no level, as the record says it
IN_RUN_NOT_JOINING = "in the run and not among the members that join"
NOT_IN_RUN = "in the temperature table and not in the run"


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
    first and last year, and the two may differ."""

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
    """The level of each assessed member of a temperature table in
    every period, and what decided it. ``means`` and ``levels`` hold
    one value a period for each assessed member, in the order of
    ``periods``. A level is None where the member's mean lies in no
    band. ``excluded`` are the members of the table that do not join,
    which are at no level."""

    table: GMSTTable
    reference_period: tuple[int, int]
    offset: float
    levels_declared: tuple[float, ...]
    half_width: float
    periods: tuple[Period, ...]
    smoothing: str
    means: dict[tuple[str, ...], tuple[float, ...]]
    levels: dict[tuple[str, ...], tuple[Optional[float], ...]]
    excluded: tuple[tuple[str, ...], ...] = ()

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
        if member in self.levels or member in self.excluded:
            return member
        raise ValueError(
            f"{dict(zip(self.member_cols, member))} has no row in the "
            f"temperature table" + (f" {self.table.path}" if self.table.path else "")
        )

    def missing(self, members: Iterable[Sequence[str]]) -> list[tuple[str, ...]]:
        """The members that have no row in the table, sorted."""
        return sorted({tuple(m) for m in members} - set(self.table.members))

    def windows_of(self, member: Sequence[str]) -> tuple[LevelWindow, ...]:
        """The periods in which a member has a level, in the order of
        ``periods``, none for a member that does not join. Raises if the
        member has no row in the table."""
        member = self._known(member)
        if member in self.excluded:
            return ()
        return tuple(
            LevelWindow(level, period.label, *period.impact_span)
            for period, level in zip(self.periods, self.levels[member])
            if level is not None
        )

    def record(self, members: Optional[Iterable[Sequence[str]]] = None) -> dict:
        """What a manifest says of a run, as plain values. ``members``
        are the members of the run, the members that join if it is left
        out. Every member of the table that does not join is under
        ``excluded`` with the reason. Raises if a member of the run has
        no row in the table, or a member that joins is not in the run."""
        run = set(
            self.levels
            if members is None
            else (self._known(member) for member in members)
        )
        used = list(self.levels)
        outside = [member for member in used if member not in run]
        if outside:
            raise ValueError(
                f"{len(outside)} members that join are not in the run "
                f"(first: {dict(zip(self.member_cols, outside[0]))})"
            )
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
            "excluded": [
                {
                    "member": named(member),
                    "reason": IN_RUN_NOT_JOINING if member in run else NOT_IN_RUN,
                }
                for member in self.excluded
            ],
        }


def _listed(
    join: str | Mapping[str, Sequence[str]], member_cols: Sequence[str]
) -> Optional[dict[str, set[str]]]:
    """The values listed to join by column, None under "declared"."""
    member_cols = list(member_cols)
    if isinstance(join, str):
        if join != EVERY_DECLARED:
            raise ValueError(
                f"join is {EVERY_DECLARED!r} or a mapping of member columns "
                f"to the values that join; got {join!r}"
            )
        return None
    if not join:
        raise ValueError("join names no member column")
    unknown = [col for col in join if col not in member_cols]
    if unknown:
        raise ValueError(
            f"join names {unknown}, which are not among the member columns "
            f"{member_cols}"
        )
    listed: dict[str, set[str]] = {}
    for col, values in join.items():
        if isinstance(values, str):
            raise ValueError(f"join.{col} has to be a list of values; got {values!r}")
        names = [str(value) for value in values]
        if not names:
            raise ValueError(f"join.{col} lists no value")
        twice = sorted({name for name in names if names.count(name) > 1})
        if twice:
            raise ValueError(f"join.{col} lists {twice} more than once")
        listed[col] = set(names)
    return listed


def members_that_join(
    join: str | Mapping[str, Sequence[str]],
    declared: Iterable[Sequence[str]],
    member_cols: Sequence[str],
) -> list[tuple[str, ...]]:
    """The members of ``declared`` that join a warming level, in their
    order. ``join`` is "declared", under which every one of them joins,
    or a mapping of member columns to the values that join, under which
    a member joins when its value in each column named is listed.
    Raises if a column named is not a member column, if a list is empty
    or repeats a value, or if a value listed is held by no member of
    ``declared``."""
    member_cols = list(member_cols)
    declared = list(dict.fromkeys(tuple(str(name) for name in m) for m in declared))
    listed = _listed(join, member_cols)
    if listed is None:
        return declared
    at = {col: member_cols.index(col) for col in join}
    held = {col: {member[at[col]] for member in declared} for col in join}
    unheld = [
        f"{col} = {name!r}"
        for col, values in join.items()
        for name in (str(value) for value in values)
        if name not in held[col]
    ]
    if unheld:
        raise ValueError(
            f"{len(unheld)} values listed to join are held by no member "
            f"declared: {unheld}"
        )
    return [
        member
        for member in declared
        if all(member[at[col]] in listed[col] for col in join)
    ]


def assign_warming_levels(
    table: GMSTTable,
    *,
    reference_period: tuple[int, int],
    offset: float,
    levels: Sequence[float],
    half_width: float,
    periods: Sequence[Period],
    smoothing: str,
    members: Iterable[Sequence[str]],
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
    here. ``members`` are the members that join, each one value for
    each member column, and have no default. A member of the table that
    is not among them is at no level, and the record lists it. Raises if a
    member that joins has no row, or lacks a value for a year of the
    reference period or of a ``gmst_span``.
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

    every = table.series()
    if isinstance(members, str):
        raise ValueError(
            f"members are the members that join, each one value for each of "
            f"{list(table.member_cols)}; got the string {members!r}"
        )
    asked = list(dict.fromkeys(tuple(str(name) for name in m) for m in members))
    if not asked:
        raise ValueError("members must name at least one member that joins")
    width = len(table.member_cols)
    if any(len(m) != width for m in asked):
        bad = next(m for m in asked if len(m) != width)
        raise ValueError(
            f"a member is one value for each of {list(table.member_cols)}; got "
            f"{list(bad)}"
        )
    absent = [m for m in asked if m not in every]
    if absent:
        raise ValueError(
            f"{len(absent)} of the members that join have no row in the "
            f"temperature table "
            f"(first: {dict(zip(table.member_cols, absent[0]))})"
        )
    joining = set(asked)
    excluded = tuple(m for m in table.members if m not in joining)
    means: dict[tuple[str, ...], tuple[float, ...]] = {}
    assigned: dict[tuple[str, ...], tuple[Optional[float], ...]] = {}
    for member in asked:
        series = every[member]
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
        excluded=excluded,
    )
