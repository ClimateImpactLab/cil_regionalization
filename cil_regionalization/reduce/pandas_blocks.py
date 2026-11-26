"""The pandas backend: statistics one block of targets at a time.

`stats.pooled_statistics` needs every member of a target at once and
nothing of any other target. So the targets are cut into blocks of
consecutive row groups, a worker reads one block from every leaf file,
and no process ever holds more than its share of the memory budget.
The numbers are those of `stats`, which this module calls and does not
repeat.
"""
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
    UnweightedModelWeightsWarning,
    _window_label,
    pooled_statistics,
    window_means,
)

# Bytes a pooled row is planned at. Traced on a block of the corn ADM2
# shape: 43 held, 64 at the peak inside the statistics, 69 while the
# Arrow tables are handed over.
BYTES_PER_ROW = 130


def kept_share(request: ReduceRequest) -> float:
    """The rows a block keeps of a row group, as a share of the group's
    rows in the leaf files. A leaf's rows shrink to one a window where
    they were one a time value, and each leaf counts by its own windows
    and its own time values, since two leaves may differ in both."""
    kept = total = 0.0
    for leaf in request.leaves:
        n_times = max(len(leaf.profile["values"]), 1)
        kept += leaf.rows / n_times * len(leaf.windows)
        total += leaf.rows
    return kept / total if total else 0.0


def choose_blocks(
    rows_per_group, share: float, bytes_per_worker: float
) -> list[tuple[int, int]]:
    """Consecutive row groups packed into blocks of at most
    ``bytes_per_worker``, as (first, stop) pairs. Groups that hold
    nothing are left out. Raises if one group alone does not fit."""
    need = np.asarray(rows_per_group, dtype=np.float64) * share * BYTES_PER_ROW
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

    def _plan(self, request: ReduceRequest) -> dict:
        if self.options.memory_gb is None:
            raise ValueError(
                "reducing leaves by blocks needs a memory budget. Set "
                "[reduce.pandas] memory_gb to what all the workers may "
                "hold together."
            )
        n_workers = self.options.n_workers or request.n_workers
        per_worker = self.options.memory_gb * 1e9 / n_workers
        share = kept_share(request)
        try:
            blocks = choose_blocks(request.rows_per_group, share, per_worker)
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
            "bytes_per_row": BYTES_PER_ROW,
            "kept_share": plan["share"],
            "text": [
                f"reduce: pandas, {self.options.memory_gb:g} GB over "
                f"{plan['n_workers']} workers, "
                f"{plan['per_worker_gb']:.3g} GB a worker",
                f"  {n_blocks} blocks of up to "
                f"{plan['groups_in_largest_block'] * per_group} targets and "
                f"{plan['rows_in_largest_block']:,} rows",
                f"  {n_blocks} opens a leaf, {n_blocks * n_leaves:,} in all",
                f"  sized at {BYTES_PER_ROW} bytes a row, with "
                f"{plan['share']:.4g} of a group's rows kept. The rows of a "
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
    stops the run. A worker that dies raises too, where a
    `multiprocessing.Pool` would wait for it for ever."""
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
    present = [
        span for span in spans if times.between(span[0], span[1]).any()
    ]
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


def _reduce_block(block: tuple[int, int]) -> dict:
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
    frames, failed_opens = [], 0
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
        reduced = _leaf_rows(request, leaf, table.to_pandas())
        if reduced is None:
            continue
        n = len(reduced)
        at = list(reduced.columns).index("window")
        for i, name in enumerate(request.level_names):
            # one code a row and the leaf's value once, not a string a row
            reduced.insert(
                at + i,
                name,
                pd.Categorical.from_codes(
                    np.full(n, level_values[i].index(leaf.levels[i]), dtype=np.int32),
                    categories=level_values[i],
                ),
            )
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
            "block": block, "stats": pd.DataFrame(), "rows": 0,
            "failed_opens": failed_opens, "peak_gb": round(_peak_gb(), 2),
        }
    combined = pd.concat(frames, ignore_index=True)
    del frames
    with warnings.catch_warnings():
        # the parent says it once
        warnings.simplefilter("ignore", UnweightedModelWeightsWarning)
        stats = pooled_statistics(
            combined,
            sample_dims=list(request.sample_dims),
            value_col=request.value_col,
            quantiles=list(request.quantiles),
            include_mean=request.include_mean,
            weight_col="model_weight" if weighted else None,
            include_n_members=request.include_n_members,
            quantile_method=request.quantile_method,
            summation=request.summation,
        )
    del combined
    # the ids go on last, one lookup a statistics row
    per_group = request.footer["targets_per_group"]
    block_ids = _STATE["targets"].ids[first * per_group:stop * per_group]
    ordinals = stats.pop(leaf_format.ORDINAL_COLUMN).to_numpy() - first * per_group
    for i, field in enumerate(request.id_fields):
        ids = np.array([row[i] for row in block_ids], dtype=object)
        stats.insert(i, field, ids[ordinals])
    for name in stats.columns:
        if isinstance(stats[name].dtype, pd.CategoricalDtype):
            stats[name] = stats[name].astype(str)
    return {
        "block": block,
        "stats": stats,
        "rows": rows,
        "failed_opens": failed_opens,
        "peak_gb": round(_peak_gb(), 2),
        "pid": os.getpid(),
    }
