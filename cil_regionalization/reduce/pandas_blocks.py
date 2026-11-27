"""The pandas backend, with the statistics taken one block of targets at
a time.

A block is a run of consecutive row groups, sized so that a worker is
planned to hold no more than its share of ``memory_gb``. A worker reads
its block from every leaf file and pools it with
`stats.pooled_statistics`, so the numbers are those of `stats`.
"""

# `stats.pooled_statistics` needs every member of a target at once and
# nothing of any other target, which is why the targets can be taken a
# block at a time.

from __future__ import annotations

import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

import numpy as np
import pandas as pd

from cil_regionalization import leaves as leaf_format
from cil_regionalization.manifest import collect_package_versions
from cil_regionalization.reduce.base import (
    PandasReduceOptions,
    ReduceBackend,
    ReduceRequest,
)
from cil_regionalization.stats import (
    GroupError,
    UnweightedModelWeightsWarning,
    _window_label,
    pooled_statistics,
    window_means,
)

# What a block is planned to take for a row it keeps, and for each
# dimension that row has of its own. They come from the resident memory
# of a worker, which grew by at most 155 bytes a kept row on leaves
# with no dimension, and by 168, 194 and 218 with one, two and three.
# That is the largest over window means and pooled years, the four
# method pairs, with and without model weights, blocks of 1,300 to
# 41,000 kept rows a leaf, and pandas 2.3.3 and 3.1.1, on macOS. The
# two figures cover each of those by half again or more. A worker
# killed for memory costs the pass, and a block that is smaller than
# it had to be costs opens.
BYTES_PER_ROW = 240
BYTES_PER_DIMENSION = 40


def bytes_per_row(request: ReduceRequest) -> int:
    """The bytes a block is planned to take for each row it keeps,
    `BYTES_PER_ROW` and `BYTES_PER_DIMENSION` more for each dimension
    of the leaf."""
    return BYTES_PER_ROW + BYTES_PER_DIMENSION * len(request.dimensions)


def kept_share(request: ReduceRequest) -> float:
    """The rows a block keeps of a row group, as a share of the group's
    rows in the leaf files. Under "window_mean" a leaf keeps its rows
    at one time value once for each of its windows, counted by its own
    windows and time values, since two leaves may differ in both.
    Otherwise a leaf keeps its rows inside its windows, and a row inside
    two windows counts twice."""
    kept = total = 0.0
    for leaf in request.leaves:
        if request.time_reduction != "window_mean":
            values = np.asarray(leaf.profile["values"], dtype=np.float64)
            rows = np.asarray(leaf.profile["rows"], dtype=np.float64)
            kept += sum(
                rows[(values >= start) & (values <= end)].sum()
                for _, start, end in leaf.windows
            )
        else:
            n_times = max(len(leaf.profile["values"]), 1)
            kept += leaf.rows / n_times * len(leaf.windows)
        total += leaf.rows
    return kept / total if total else 0.0


def choose_blocks(
    rows_per_group, share: float, bytes_per_worker: float, row_bytes: float
) -> list[tuple[int, int]]:
    """Consecutive row groups packed into blocks of at most
    ``bytes_per_worker``, as (first, stop) pairs, with a kept row
    counted as ``row_bytes``. A block never starts on a group that holds
    nothing, so the empty groups before the first that holds rows are
    left out, and a later empty group joins the block before it.
    Raises if one group alone does not fit."""
    need = np.asarray(rows_per_group, dtype=np.float64) * share * row_bytes
    if len(need) and need.max() > bytes_per_worker:
        raise _TooSmall(float(need.max()))
    blocks: list[tuple[int, int]] = []
    first, held = None, 0.0
    for g, size in enumerate(need.tolist()):
        if first is not None and held + size > bytes_per_worker:
            blocks.append((first, g))
            first, held = None, 0.0
        if size == 0.0 and first is None:
            continue
        if first is None:
            first = g
        held += size
    if first is not None:
        blocks.append((first, len(need)))
    return blocks


class _TooSmall(ValueError):
    def __init__(self, largest_group: float) -> None:
        super().__init__(largest_group)
        self.largest_group = largest_group


class PandasBlocks(ReduceBackend):
    kind = "pandas"

    def __init__(self, options: PandasReduceOptions | None = None) -> None:
        self.options = options or PandasReduceOptions()

    def check_options(self) -> None:
        if self.options.memory_gb is None:
            raise ValueError(
                "reducing leaves by blocks needs a memory budget. Set "
                "[reduce.pandas] memory_gb to what all the workers may "
                "hold together."
            )

    def _plan(self, request: ReduceRequest) -> dict:
        self.check_options()
        n_workers = self.options.n_workers or request.n_workers
        per_worker = self.options.memory_gb * 1e9 / n_workers
        share = kept_share(request)
        row_bytes = bytes_per_row(request)
        try:
            blocks = choose_blocks(request.rows_per_group, share, per_worker, row_bytes)
        except _TooSmall as error:
            needed = error.largest_group * n_workers / 1e9
            raise ValueError(
                f"[reduce.pandas] memory_gb = {self.options.memory_gb:g} gives "
                f"each of {n_workers} workers {per_worker / 1e9:.3g} GB, and "
                f"one row group of {request.footer['targets_per_group']} "
                f"targets needs {error.largest_group / 1e9:.3g} GB. The "
                f"smallest budget that runs with {n_workers} workers is "
                f"{needed:.3g} GB."
            ) from None
        kept = np.asarray(request.rows_per_group, dtype=np.float64) * share
        return {
            "blocks": blocks,
            "n_workers": max(1, min(n_workers, len(blocks))),
            "per_worker_gb": per_worker / 1e9,
            "share": share,
            "row_bytes": row_bytes,
            "rows_in_largest_block": int(
                max((kept[a:b].sum() for a, b in blocks), default=0)
            ),
            "groups_in_largest_block": max((b - a for a, b in blocks), default=0),
        }

    def describe(self, request: ReduceRequest) -> dict:
        plan = self._plan(request)
        per_group = request.footer["targets_per_group"]
        n_blocks, n_leaves = len(plan["blocks"]), len(request.leaves)
        return {
            "kind": self.kind,
            "engine": collect_package_versions(("pandas", "numpy", "pyarrow")),
            "options": {
                "memory_gb": self.options.memory_gb,
                "n_workers": plan["n_workers"],
            },
            "blocks": n_blocks,
            "targets_in_largest_block": plan["groups_in_largest_block"] * per_group,
            "rows_in_largest_block": plan["rows_in_largest_block"],
            "opens_per_leaf": n_blocks,
            "opens": n_blocks * n_leaves,
            "per_worker_gb": plan["per_worker_gb"],
            "bytes_per_row": plan["row_bytes"],
            "kept_share": plan["share"],
            "text": [
                f"reduce: pandas, {self.options.memory_gb:g} GB over "
                f"{plan['n_workers']} workers, "
                f"{plan['per_worker_gb']:.3g} GB a worker",
                f"  {n_blocks} blocks of up to "
                f"{plan['groups_in_largest_block'] * per_group} targets and "
                f"{plan['rows_in_largest_block']:,} rows",
                f"  {n_blocks} opens a leaf, {n_blocks * n_leaves:,} in all",
                f"  sized at {plan['row_bytes']} bytes a row, "
                + (
                    f"{BYTES_PER_ROW} and {BYTES_PER_DIMENSION} for each "
                    f"dimension of a leaf, "
                    if request.dimensions
                    else ""
                )
                + f"with "
                f"{plan['share']:.4g} of a group's rows kept"
                f"{', each with its year' if request.time_reduction != 'window_mean' else ''}. "
                f"The rows of a "
                f"group are known summed over the leaves, so this takes "
                f"every leaf to spread its rows over the groups alike.",
            ],
        }

    def run(self, request: ReduceRequest) -> tuple[pd.DataFrame, dict]:
        plan = self._plan(request)
        results = _run_blocks(request, plan["blocks"], plan["n_workers"])
        frames = [r["stats"] for r in results if len(r["stats"])]
        if not frames:
            raise ValueError("reduce: the leaves hold no row inside any window")
        stats = pd.concat(frames, ignore_index=True)
        report = {
            **{k: v for k, v in self.describe(request).items() if k != "text"},
            "failed_opens": sum(r["failed_opens"] for r in results),
            "max_worker_peak_gb": max(r["peak_gb"] for r in results),
            "rows_pooled": sum(r["rows"] for r in results),
        }
        return stats[request.columns], report


def _run_blocks(request: ReduceRequest, blocks, n_workers: int) -> list[dict]:
    """Every block's result, in any order. The first block that fails
    stops the run, and a worker that dies raises `RuntimeError`."""
    # A process pool executor, because a `multiprocessing.Pool` would
    # wait for ever for a worker that died.
    if n_workers == 1:
        _init_worker(request)
        results = []
        for block in blocks:
            results.append(_reduce_block(block))
            _log_block(results, len(blocks))
        return results
    results = []
    executor = ProcessPoolExecutor(
        max_workers=n_workers, initializer=_init_worker, initargs=(request,)
    )
    try:
        futures = {executor.submit(_reduce_block, block): block for block in blocks}
        for future in as_completed(futures):
            first, stop = futures[future]
            try:
                results.append(future.result())
            except BrokenProcessPool:
                raise RuntimeError(
                    f"reduce: a worker died while row groups {first} to "
                    f"{stop - 1} were being reduced, most likely killed for "
                    f"memory. No statistics were written. Raise "
                    f"[reduce.pandas] memory_gb only if the node has it. "
                    f"Otherwise lower it, which makes the blocks smaller."
                ) from None
            _log_block(results, len(blocks))
    finally:
        executor.shutdown(wait=False, cancel_futures=True)
    return results


def _log_block(results: list[dict], n_blocks: int) -> None:
    last = results[-1]
    print(
        f"reduce: block {len(results)}/{n_blocks}, row groups "
        f"{last['block'][0]}-{last['block'][1] - 1}, {last['rows']:,} rows, "
        f"worker peak {last['peak_gb']:.2f}G",
        flush=True,
    )


_STATE: dict = {}


def _init_worker(request: ReduceRequest) -> None:
    targets = leaf_format.TargetTable.read(
        Path(request.leaves_root) / request.targets_file
    )
    if targets.digest != request.footer["target_digest"]:
        raise ValueError(
            f"the target table {request.targets_file} is not the one the "
            f"leaves were written with"
        )
    _STATE.update(request=request, targets=targets)


def _peak_gb() -> float:
    import resource

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 1e6 if sys.platform.startswith("linux") else peak / 1e9


def _leaf_rows(
    request: ReduceRequest, leaf, frame: pd.DataFrame
) -> pd.DataFrame | None:
    """One leaf's rows of a block reduced to its window means, under the
    labels the request gives its windows. None if no window of the leaf
    has a row here."""
    times = frame[request.time_col]
    # one call for the distinct spans that have rows in this slice
    spans: dict[tuple[float, float], list[str]] = {}
    for label, start, end in leaf.windows:
        spans.setdefault((start, end), []).append(label)
    present = [span for span in spans if times.between(span[0], span[1]).any()]
    if not present:
        return None
    reduced = window_means(
        frame,
        time_col=request.time_col,
        windows=present,
        value_col=request.value_col,
        summation=request.summation,
    )
    named = {_window_label(*span): spans[span] for span in present}
    if any(labels != [made] for made, labels in named.items()):
        pieces = []
        for label, start, end in leaf.windows:
            if (start, end) not in present:
                continue
            piece = reduced[reduced["window"] == _window_label(start, end)]
            pieces.append(piece.assign(window=label))
        reduced = pd.concat(pieces, ignore_index=True)
    return reduced


def _leaf_years(
    request: ReduceRequest, leaf, frame: pd.DataFrame
) -> pd.DataFrame | None:
    """One leaf's rows of a block that lie inside its windows, each
    with its time value and under the label the request gives its
    window. A row inside two windows comes back twice. None if no
    window of the leaf has a row here."""
    time_col, value_col = request.time_col, request.value_col
    times = frame[time_col]
    labels = list(request.window_labels)
    pieces = []
    for label, start, end in leaf.windows:
        inside = times.between(start, end).to_numpy()
        if not inside.any():
            continue
        piece = frame[inside]
        # one code a row, as for the levels
        piece.insert(
            len(piece.columns),
            "window",
            pd.Categorical.from_codes(
                np.full(len(piece), labels.index(label), dtype=np.int32),
                categories=labels,
            ),
        )
        pieces.append(piece)
    if not pieces:
        return None
    kept = pieces[0] if len(pieces) == 1 else pd.concat(pieces, ignore_index=True)
    group = [c for c in frame.columns if c not in (time_col, value_col)]
    return kept[group + ["window", time_col, value_col]]


# the column of a member row that says which bin member its leaf is
_BIN = "bin_member_code"


def _by_id(request: ReduceRequest, group: dict) -> dict:
    """``group`` with the target's ordinal replaced by its id fields,
    from the target table."""
    if leaf_format.ORDINAL_COLUMN not in group:
        return group
    rest = {k: v for k, v in group.items() if k != leaf_format.ORDINAL_COLUMN}
    ids = _STATE["targets"].ids[int(group[leaf_format.ORDINAL_COLUMN])]
    return {**dict(zip(request.id_fields, ids)), **rest}


def _leaf_span(
    request: ReduceRequest, leaf, frame: pd.DataFrame
) -> pd.DataFrame | None:
    """One leaf's rows of a block inside its one window, with their time
    value, which stays a column of the statistics. None if the window
    has no row here."""
    ((_, start, end),) = leaf.windows
    inside = frame[request.time_col].between(start, end).to_numpy()
    if not inside.any():
        return None
    group = [c for c in frame.columns if c not in (request.time_col, request.value_col)]
    return frame.loc[inside, group + [request.time_col, request.value_col]]


def _with_counts(
    stats: pd.DataFrame, members: list[pd.DataFrame], request: ReduceRequest
) -> pd.DataFrame:
    """The statistics with the counts that come from ``members``, which
    holds one row for each member with a value in a window. Under
    "pooled" the count `pooled_statistics` gave is the rows
    behind a statistic and becomes 'n_sample', and 'n_members' is the
    rows of ``members`` behind it. 'n_bin_members' is the distinct bin
    members among those rows."""
    if request.time_reduction == "pooled":
        stats = stats.rename(columns={"n_members": "n_sample"})
    identity = [
        c
        for c in stats.columns
        if c
        not in ("statistic", request.value_col, "n_members", "n_sample", "n_models")
    ]
    grouped = _plain_labels(pd.concat(members, ignore_index=True)).groupby(
        identity, sort=False, observed=True
    )
    counts = []
    if request.time_reduction == "pooled":
        counts.append(grouped.size().rename("n_members"))
    if request.include_n_bin_members:
        counts.append(grouped[_BIN].nunique().rename("n_bin_members"))
    names = [count.name for count in counts]
    stats = stats.merge(
        pd.concat(counts, axis=1).reset_index(),
        on=identity,
        how="left",
        validate="many_to_one",
    )
    for name in names:
        stats[name] = stats[name].fillna(0).astype(np.int64)
    return stats


def _check_whole_spans(request: ReduceRequest, leaf, frame: pd.DataFrame) -> None:
    """Raise if a group of a leaf's rows has a value for some years of
    one of the leaf's windows and not for all of them. Its window mean
    would be over fewer years than the window says."""
    times = frame[request.time_col]
    has_value = frame[request.value_col].notna().to_numpy()
    group = [c for c in frame.columns if c not in (request.time_col, request.value_col)]
    for _, start, end in leaf.windows:
        valid = times.between(start, end).to_numpy() & has_value
        n_years = int(end) - int(start) + 1
        sizes = frame.loc[valid, group].groupby(group, sort=False).size()
        short = sizes[sizes != n_years]
        if not len(short):
            continue
        first = short.index[0] if len(group) > 1 else (short.index[0],)
        where = dict(zip(group, first))
        ordinal = where.pop(leaf_format.ORDINAL_COLUMN)
        where = {
            **dict(zip(request.id_fields, _STATE["targets"].ids[ordinal])),
            **where,
        }
        raise ValueError(
            f"leaf {'/'.join(leaf.levels)} has a value for "
            f"{int(short.iloc[0])} of the {n_years} years {start:g} to "
            f"{end:g} at {where}, and so for {len(short)} of its "
            f"{len(sizes)} groups in this block. A mean over fewer years "
            f"than the span is refused."
        )


def _plain_labels(frame: pd.DataFrame) -> pd.DataFrame:
    for name in frame.columns:
        if isinstance(frame[name].dtype, pd.CategoricalDtype):
            frame[name] = frame[name].astype(str)
    return frame


def _reduce_block(block: tuple[int, int]) -> dict:
    """The statistics of one block of targets, as `_pool_block` gives
    them. A refusal names a target by its id fields."""
    try:
        return _pool_block(block)
    except GroupError as error:
        # the frames name a target by its ordinal, the reader by its id
        raise error.naming(_by_id(_STATE["request"], error.group)) from None


def _pool_block(block: tuple[int, int]) -> dict:
    request: ReduceRequest = _STATE["request"]
    first, stop = block
    root = Path(request.leaves_root)
    read = [leaf_format.ORDINAL_COLUMN] + [
        c["name"]
        for c in request.schema
        if c["name"] not in request.id_fields
        and c["name"] != leaf_format.ORDINAL_COLUMN
    ]
    level_values = [
        list(dict.fromkeys(leaf.levels[i] for leaf in request.leaves))
        for i in range(len(request.level_names))
    ]
    weighted = request.weighted
    pooled = request.time_reduction == "pooled"
    per_year = request.time_reduction == "per_year"
    count_bins = request.include_n_bin_members
    count_members = (pooled and request.include_n_members) or count_bins
    bins = list(dict.fromkeys(leaf.bin_member for leaf in request.leaves))
    frames, members, failed_opens = [], [], 0
    for leaf in request.leaves:
        table, failed = leaf_format.read_groups(
            root / leaf.file,
            first,
            stop,
            expect={**request.footer, "rows": leaf.rows},
            columns=read,
        )
        failed_opens += failed
        if table.num_rows == 0:
            # nothing to turn into a frame
            continue
        frame = table.to_pandas()
        if request.whole_spans:
            _check_whole_spans(request, leaf, frame)
        if per_year:
            reduced = _leaf_span(request, leaf, frame)
        else:
            reduced = (_leaf_years if pooled else _leaf_rows)(request, leaf, frame)
        if reduced is None:
            continue
        pieces = [reduced]
        if count_members and (not weighted or leaf.model_weight > 0):
            # A leaf's rows are members of their own, since every level
            # is either pooled or kept. So the members are counted leaf
            # by leaf, from far fewer rows than the block pools.
            has_value = reduced[request.value_col].notna().to_numpy()
            with_value = reduced.loc[
                has_value,
                [
                    c
                    for c in reduced.columns
                    if c not in (request.time_col, request.value_col)
                ],
            ]
            # a window mean is one row a member already
            pieces.append(with_value.drop_duplicates() if pooled else with_value.copy())
        for piece in pieces:
            at = list(piece.columns).index(request.time_col if per_year else "window")
            for i, name in enumerate(request.level_names):
                # one code a row and the leaf's value once, not a string
                # a row
                piece.insert(
                    at + i,
                    name,
                    pd.Categorical.from_codes(
                        np.full(
                            len(piece),
                            level_values[i].index(leaf.levels[i]),
                            dtype=np.int32,
                        ),
                        categories=level_values[i],
                    ),
                )
        if len(pieces) > 1:
            if count_bins:
                pieces[1].insert(
                    len(pieces[1].columns), _BIN, bins.index(leaf.bin_member)
                )
            members.append(pieces[1])
        if request.time_reduction == "window_mean":
            reduced["window"] = pd.Categorical(
                reduced["window"], categories=list(request.window_labels)
            )
        if weighted:
            reduced["model_weight"] = leaf.model_weight
        frames.append(reduced)

    rows = sum(len(f) for f in frames)
    if not frames:
        # `pooled_statistics` is not asked about nothing
        return {
            "block": block,
            "stats": pd.DataFrame(),
            "rows": 0,
            "failed_opens": failed_opens,
            "peak_gb": round(_peak_gb(), 2),
        }
    combined = pd.concat(frames, ignore_index=True)
    del frames
    with warnings.catch_warnings():
        # the parent says it once
        warnings.simplefilter("ignore", UnweightedModelWeightsWarning)
        stats = pooled_statistics(
            combined,
            sample_dims=list(request.sample_dims)
            + ([request.time_col] if pooled else []),
            value_col=request.value_col,
            quantiles=list(request.quantiles),
            include_mean=request.include_mean,
            weight_col="model_weight" if weighted else None,
            include_n_members=request.include_n_members,
            model_dims=list(request.model_dims) if request.include_n_members else None,
            quantile_method=request.quantile_method,
            summation=request.summation,
        )
    del combined
    stats = _plain_labels(stats)
    if count_members:
        stats = _with_counts(stats, members, request)
    # the ids go on last, one lookup a statistics row
    per_group = request.footer["targets_per_group"]
    block_ids = _STATE["targets"].ids[first * per_group : stop * per_group]
    ordinals = stats.pop(leaf_format.ORDINAL_COLUMN).to_numpy() - first * per_group
    for i, field in enumerate(request.id_fields):
        ids = np.array([row[i] for row in block_ids], dtype=object)
        stats.insert(i, field, ids[ordinals])
    return {
        "block": block,
        "stats": stats,
        "rows": rows,
        "failed_opens": failed_opens,
        "peak_gb": round(_peak_gb(), 2),
        "pid": os.getpid(),
    }
