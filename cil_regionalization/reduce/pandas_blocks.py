"""The pandas backend, with the statistics taken one block of targets at
a time.

A block is a run of consecutive row groups, sized so that a worker is
planned to hold no more than its share of ``memory_gb``, or of the size
``targets_per_block`` declares. A worker reads its block from every leaf
file and pools it with `stats.pooled_statistics`, so the numbers are
those of `stats`.

The size of a block rests on an estimate of the bytes a kept row takes,
calibrated on resident memory. The first block is built alone, in a
worker of its own, and measured as the growth of that worker's peak
resident memory, and the manifest records the estimate and the
measurement side by side. A run sized by the estimate is refused when
the measurement of a block large enough to judge is more than twice the
estimate. In a single process that has already done other work the peak
cannot be put down to the block, so there is no measurement and no
judgement.
"""

# `stats.pooled_statistics` needs every member of a target at once and
# nothing of any other target, which is why the targets can be taken a
# block at a time.

from __future__ import annotations

import multiprocessing
import os
import statistics
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

import numpy as np
import pandas as pd

from cil_regionalization import leaves as leaf_format
from cil_regionalization import resident
from cil_regionalization.manifest import collect_package_versions
from cil_regionalization.reduce.base import (
    PandasReduceOptions,
    ReduceBackend,
    ReduceRequest,
)
from cil_regionalization.resident import peak_resident_bytes as _peak_resident_bytes
from cil_regionalization.resident import resident_bytes as _resident_bytes
from cil_regionalization.stats import (
    GroupError,
    UnweightedModelWeightsWarning,
    _window_label,
    pooled_statistics,
    window_means,
)
from cil_regionalization.workers import WorkerDied, shut_down_broken

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
# The kept rows a first block needs for its measurement to be judged.
# Below it the fixed cost of pooling a block outweighs what its rows
# take, and the measurement is recorded and not judged.
JUDGED_ROWS = 100_000
# A first block is refused when it takes more than this many times the
# estimate a kept row. It is a tolerance on a measure with known noise,
# not a correction of the estimate. One synthetic first block, measured
# on macOS in one fresh worker after another, read from a fifth of the
# estimate to nine tenths of it. Memory sampled through the block peaked
# where the measure did, so the spread is in what the block holds
# resident and not in the reading. Twice refuses a block far heavier
# than the estimate and not that spread. The manifest records the
# measure of every run, so runs on real data can tighten it.
REFUSED_ABOVE = 2
# Why a first block run in the process that planned the run has no
# measurement.
_SHARED_PEAK = (
    "the first block ran in the process that planned the run, whose peak "
    "resident memory includes what it did before, so the block's share of "
    "the peak cannot be told"
)
# Why a worker whose peak did not rise has none either. A block that
# fits in memory the worker already held, as a small one can in what the
# worker's setup freed, leaves the peak where it was.
_FLAT_PEAK = (
    "the worker's peak resident memory did not rise over the block, so the "
    "block fit in memory the worker already held and the peak says nothing "
    "of it"
)


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


def fixed_blocks(rows_per_group, groups_per_block: int) -> list[tuple[int, int]]:
    """Consecutive row groups in blocks of ``groups_per_block``, as
    (first, stop) pairs. The empty groups before the first that holds
    rows are left out, as in `choose_blocks`, and the last block may be
    shorter."""
    rows = np.asarray(rows_per_group, dtype=np.float64)
    held = np.flatnonzero(rows > 0)
    if not len(held):
        return []
    start, end = int(held[0]), len(rows)
    return [
        (first, min(first + groups_per_block, end))
        for first in range(start, end, groups_per_block)
    ]


class _TooSmall(ValueError):
    def __init__(self, largest_group: float) -> None:
        super().__init__(largest_group)
        self.largest_group = largest_group


class PandasBlocks(ReduceBackend):
    kind = "pandas"

    def __init__(self, options: PandasReduceOptions | None = None) -> None:
        self.options = options or PandasReduceOptions()

    def check_options(self) -> None:
        if self.options.memory_gb is None and self.options.targets_per_block is None:
            raise ValueError(
                "reducing leaves by blocks needs a memory budget. Set "
                "[reduce.pandas] memory_gb to what all the workers may "
                "hold together, or declare the size of a block in "
                "targets_per_block."
            )

    def _plan(self, request: ReduceRequest) -> dict:
        self.check_options()
        n_workers = self.options.n_workers or request.n_workers
        share = kept_share(request)
        row_bytes = bytes_per_row(request)
        kept = np.asarray(request.rows_per_group, dtype=np.float64) * share
        declared = self.options.targets_per_block
        if declared is not None:
            per_group = request.footer["targets_per_group"]
            if declared % per_group:
                raise ValueError(
                    f"[reduce.pandas] targets_per_block = {declared} is not a "
                    f"whole number of the leaves' row groups, which hold "
                    f"{per_group} targets each"
                )
            blocks = fixed_blocks(request.rows_per_group, declared // per_group)
            return {
                "by": "declared",
                "blocks": blocks,
                "n_workers": max(1, min(n_workers, len(blocks))),
                "per_worker_gb": None,
                "share": share,
                "row_bytes": row_bytes,
                "rows_in_largest_block": int(
                    max((kept[a:b].sum() for a, b in blocks), default=0)
                ),
                "groups_in_largest_block": max((b - a for a, b in blocks), default=0),
            }
        per_worker = self.options.memory_gb * 1e9 / n_workers
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
        return {
            "by": "estimate",
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
        declared = plan["by"] == "declared"
        if declared:
            budget = (
                f"reduce: pandas, blocks of {self.options.targets_per_block} "
                f"targets as declared, over {plan['n_workers']} workers"
            )
            sized = (
                "  sized as declared, not from memory_gb. The first block is "
                "measured, and the measurement is recorded and not judged."
            )
        else:
            budget = (
                f"reduce: pandas, {self.options.memory_gb:g} GB over "
                f"{plan['n_workers']} workers, "
                f"{plan['per_worker_gb']:.3g} GB a worker"
            )
            sized = (
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
                f"every leaf to spread its rows over the groups alike. The "
                f"first block is measured in a worker of its own, and the run "
                f"is refused if it takes more than {REFUSED_ABOVE} times that."
            )
        return {
            "kind": self.kind,
            "engine": collect_package_versions(("pandas", "numpy", "pyarrow")),
            "options": {
                "memory_gb": self.options.memory_gb,
                "targets_per_block": self.options.targets_per_block,
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
                budget,
                f"  {n_blocks} blocks of up to "
                f"{plan['groups_in_largest_block'] * per_group} targets and "
                f"{plan['rows_in_largest_block']:,} rows",
                f"  {n_blocks} opens a leaf, {n_blocks * n_leaves:,} in all",
                sized,
            ],
        }

    def run(self, request: ReduceRequest) -> tuple[pd.DataFrame, dict]:
        plan = self._plan(request)
        if plan["by"] == "declared":
            warnings.warn(
                f"[reduce.pandas] targets_per_block = "
                f"{self.options.targets_per_block} declares the size of a "
                f"block, so the blocks are not sized from a memory budget, "
                f"and the measurement of the first block is recorded and not "
                f"judged",
                UserWarning,
                stacklevel=2,
            )
        sizing: dict = {}

        def judge(first: dict) -> None:
            sizing.update(_block_size_record(request, plan, first))
            if sizing["refused"]:
                raise ValueError(_too_heavy(request, plan, sizing))

        results = _run_blocks(request, plan["blocks"], plan["n_workers"], judge)
        frames = [r["stats"] for r in results if len(r["stats"])]
        if not frames:
            raise ValueError("reduce: the leaves hold no row inside any window")
        stats = pd.concat(frames, ignore_index=True)
        sizing.pop("refused", None)
        # every block measured and recorded, and none of it judged
        every = sorted(
            (_block_reading(r) for r in results), key=lambda e: e["block"][0]
        )
        sizing.update(
            every_block=every,
            over_every_block=_over_every_block(every, plan["row_bytes"]),
        )
        report = {
            **{k: v for k, v in self.describe(request).items() if k != "text"},
            "block_size": sizing,
            "failed_opens": sum(r["failed_opens"] for r in results),
            "max_worker_peak_gb": max(r["peak_gb"] for r in results),
            "rows_pooled": sum(r["rows"] for r in results),
        }
        return stats[request.columns], report


def _block_size_record(request: ReduceRequest, plan: dict, first: dict) -> dict:
    """The manifest's account of how the blocks were sized, the estimate
    and the measurement of the first block side by side, and whether the
    run is refused for it. Without a measurement the run is not
    judged."""
    rows, growth = first["rows"], first["resident_growth_bytes"]
    unavailable = first["measure_unavailable"]
    measured = growth / rows if rows and growth is not None else None
    judged = (
        plan["by"] == "estimate"
        and unavailable is None
        and measured is not None
        and rows >= JUDGED_ROWS
    )
    return {
        "by": plan["by"],
        "targets_per_block": (
            plan["groups_in_largest_block"] * request.footer["targets_per_group"]
            if plan["by"] == "declared"
            else None
        ),
        "estimate_bytes_per_row": plan["row_bytes"],
        "measured_bytes_per_row": measured,
        "measured_rows": rows,
        "measured_growth_bytes": growth,
        # Under fork a worker counts the pages it shares with the process
        # that started it as its own, so its peaks overstate what it alone
        # holds. That errs on the side of the budget, and the growth, the
        # difference of two peaks that both count them, is exact.
        "worker_setup_peak_bytes": first["worker_setup_peak_bytes"],
        "peak_after_block_bytes": first["peak_after_block_bytes"],
        "measured_block": list(first["block"]),
        "start_method": first["start_method"],
        "measure": (
            f"the growth of the peak resident memory of the worker that built "
            f"the first block, read as {resident.reading() or 'nothing'} on "
            f"{sys.platform}, from the peak once it had set itself up and "
            f"before it read the block, to the end of the block. Under fork "
            f"both peaks count the pages the worker shares with the process "
            f"that started it, so they overstate what it alone holds, and the "
            f"growth is unaffected"
        ),
        "unavailable": unavailable,
        "judged": judged,
        "judged_from_rows": JUDGED_ROWS,
        "refused_above": REFUSED_ABOVE,
        "refused": judged and measured > REFUSED_ABOVE * plan["row_bytes"],
    }


def _too_heavy(request: ReduceRequest, plan: dict, sizing: dict) -> str:
    """The refusal of a run whose first block took more than
    `REFUSED_ABOVE` times the estimate, with the largest declared block
    that fits at the measured size."""
    per_group = request.footer["targets_per_group"]
    first, stop = sizing["measured_block"]
    targets = (stop - first) * per_group
    per_target = sizing["measured_growth_bytes"] / targets
    fits = int(plan["per_worker_gb"] * 1e9 // per_target) // per_group * per_group
    way_out = (
        f"At the measured size a block of {fits} targets fits a worker."
        if fits
        else f"At the measured size not one row group of {per_group} targets "
        f"fits a worker, so memory_gb has to be raised."
    )
    return (
        f"reduce: the first block, row groups {first} to {stop - 1}, took "
        f"{sizing['measured_bytes_per_row']:.0f} bytes a kept row over "
        f"{sizing['measured_rows']:,} rows, more than {REFUSED_ABOVE} times "
        f"the {sizing['estimate_bytes_per_row']} bytes a row the blocks were "
        f"sized at, so a block can take more than a worker's "
        f"{plan['per_worker_gb']:.3g} GB. No statistics were written. The "
        f"measure is the growth of the worker's peak resident memory over "
        f"the block. Declare the size of a block in "
        f"[reduce.pandas] targets_per_block, in place of memory_gb. "
        f"{way_out}"
    )


def _run_blocks(request: ReduceRequest, blocks, n_workers: int, judge) -> list[dict]:
    """Every block's result, in any order. The first block is built
    alone and measured, and handed to ``judge``, which may raise, before
    any other is. The first block that fails stops the run, and a
    worker that dies raises `WorkerDied`."""
    # A process pool executor, because a `multiprocessing.Pool` would
    # wait for ever for a worker that died.
    if not blocks:
        return []
    first, rest = blocks[0], blocks[1:]
    if n_workers == 1:
        _init_worker(request, own_process=False)
        results = [_measured_block(first)]
        results[0]["start_method"] = None
        judge(results[0])
        _log_block(results, len(blocks))
        for block in rest:
            results.append(_read_block(block))
            _log_block(results, len(blocks))
        return results
    results = []
    # the platform's own, fork on Linux, named so the manifest can say it
    context = multiprocessing.get_context()
    executor = ProcessPoolExecutor(
        max_workers=n_workers,
        mp_context=context,
        initializer=_init_worker,
        initargs=(request,),
    )
    try:
        results.append(
            _collect(executor.submit(_measured_block, first), first, executor)
        )
        results[0]["start_method"] = context.get_start_method()
        judge(results[0])
        _log_block(results, len(blocks))
        futures = {executor.submit(_read_block, block): block for block in rest}
        for future in as_completed(futures):
            results.append(_collect(future, futures[future], executor))
            _log_block(results, len(blocks))
    finally:
        executor.shutdown(wait=False, cancel_futures=True)
    return results


def _collect(future, block, executor) -> dict:
    """The result of a block's future. Raises `WorkerDied` if a worker
    died, saying how the workers ended, which is not always memory."""
    first, stop = block
    try:
        return future.result()
    except BrokenProcessPool:
        ended, killed = shut_down_broken(executor)
        memory = (
            " Raise [reduce.pandas] memory_gb only if the node has it. "
            "Otherwise lower it, which makes the blocks smaller."
            if killed
            else ""
        )
        # Every block not back fails at once, so this one was not
        # necessarily the dead worker's.
        raise WorkerDied(
            f"reduce: a worker died with row groups {first} to {stop - 1} "
            f"among the blocks not back, {ended}. No statistics were "
            f"written.{memory}"
        ) from None


def _measured_block(block: tuple[int, int]) -> dict:
    """`_read_block`, with the growth of the process's peak resident
    memory over the block under 'resident_growth_bytes', or None and the
    reason under 'measure_unavailable'. The growth is taken from the peak
    once the worker has set itself up and before it reads the block, so
    what the worker holds for every block, its imports and its target
    table, is outside it."""
    result = _read_block(block)
    setup = result["reading"]["peak_before_bytes"]
    peak = result["reading"]["peak_after_bytes"]
    growth, unavailable = _growth(setup, peak, _STATE["own_process"])
    result.update(
        worker_setup_peak_bytes=setup,
        peak_after_block_bytes=peak,
        resident_growth_bytes=growth,
        measure_unavailable=unavailable,
    )
    return result


def _read_block(block: tuple[int, int]) -> dict:
    """`_reduce_block`, with the process's peak resident memory before and
    after the block and its resident memory before it under 'reading'."""
    peak_before = _peak_resident_bytes()
    resident_before = _resident_bytes()
    result = _reduce_block(block)
    peak_after = _peak_resident_bytes()
    _STATE["blocks_read"] = _STATE.get("blocks_read", 0) + 1
    result["reading"] = {
        "peak_before_bytes": peak_before,
        "resident_before_bytes": resident_before,
        "peak_after_bytes": peak_after,
        "nth_in_process": _STATE["blocks_read"],
        "own_process": _STATE["own_process"],
    }
    return result


def _block_reading(result: dict) -> dict:
    """A block's entry in the manifest, with what the block itself grew
    the resident memory by, or None and why. The growth is read off the
    peak after the block, so only a block that raised its process's peak
    has one. It is taken from the resident memory before the block where
    that is read. Elsewhere only a worker's first block has a growth,
    from the peak once the worker had set itself up."""
    reading, rows = result["reading"], result["rows"]
    before = reading["peak_before_bytes"]
    resident_before = reading["resident_before_bytes"]
    after = reading["peak_after_bytes"]
    growth = base = unavailable = None
    if not rows:
        unavailable = "the block kept no rows"
    elif before is None or after is None:
        unavailable = resident.why_not_read()
    elif after <= before:
        unavailable = _FLAT_PEAK
    elif resident_before is not None:
        growth, base = after - resident_before, "the resident memory before it"
    elif reading["nth_in_process"] == 1 and reading["own_process"]:
        growth, base = after - before, "the peak once the worker had set itself up"
    else:
        unavailable = (
            f"the resident memory before the block is not read on "
            f"{sys.platform}, and the peak before it is that of what the "
            f"process did earlier"
        )
    return {
        "block": list(result["block"]),
        "rows": rows,
        "pid": result.get("pid"),
        "nth_in_process": reading["nth_in_process"],
        "peak_before_bytes": before,
        "resident_before_bytes": resident_before,
        "peak_after_bytes": after,
        "growth_bytes": growth,
        "bytes_per_row": growth / rows if growth is not None else None,
        "growth_from": base,
        "unavailable": unavailable,
    }


def _over_every_block(entries: list[dict], estimate: float) -> dict:
    """What the measured blocks of a run say beside the estimate, the
    spread of their bytes a kept row and how far the first block sits
    from the heaviest. None of it is judged."""
    measured = [e for e in entries if e["bytes_per_row"] is not None]
    summary = {
        "blocks": len(entries),
        "measured": len(measured),
        "bytes_per_row_min": None,
        "bytes_per_row_median": None,
        "bytes_per_row_max": None,
        "heaviest_block": None,
        "heaviest_over_estimate": None,
        "first_over_heaviest": None,
    }
    if not measured:
        return summary
    per_row = sorted(e["bytes_per_row"] for e in measured)
    heaviest = max(measured, key=lambda e: e["bytes_per_row"])
    first = entries[0]["bytes_per_row"]
    summary.update(
        bytes_per_row_min=per_row[0],
        bytes_per_row_median=statistics.median(per_row),
        bytes_per_row_max=per_row[-1],
        heaviest_block=heaviest["block"],
        heaviest_over_estimate=per_row[-1] / estimate if estimate else None,
        first_over_heaviest=first / per_row[-1] if first is not None else None,
    )
    return summary


def _growth(
    started: int | None, peak: int | None, own_process: bool
) -> tuple[int | None, str | None]:
    """The growth of the peak resident memory over the first block, or
    None and why it cannot be read off the peak."""
    if not own_process:
        return None, _SHARED_PEAK
    if started is None or peak is None:
        return None, resident.why_not_read()
    if peak <= started:
        return None, _FLAT_PEAK
    return peak - started, None


def _log_block(results: list[dict], n_blocks: int) -> None:
    last = results[-1]
    per_row = _block_reading(last)["bytes_per_row"]
    grew = "" if per_row is None else f", {per_row:.0f} bytes a kept row"
    print(
        f"reduce: block {len(results)}/{n_blocks}, row groups "
        f"{last['block'][0]}-{last['block'][1] - 1}, {last['rows']:,} rows, "
        f"worker peak {last['peak_gb']:.2f}G{grew}",
        flush=True,
    )


_STATE: dict = {}


def _init_worker(request: ReduceRequest, own_process: bool = True) -> None:
    """Load what every block needs. ``own_process`` is False where the
    blocks run in the process that planned the run, whose peak resident
    memory is not the worker's own."""
    # a forked worker starts with the state of the process it came from
    _STATE.update(own_process=own_process, blocks_read=0)
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
    peak = _peak_resident_bytes()
    return float("nan") if peak is None else peak / 1e9


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
    id_types = _STATE["targets"].id_types
    for i, field in enumerate(request.id_fields):
        # in the type the weights artifact gave the field
        kind = object if id_types[i] == "string" else np.dtype(id_types[i])
        ids = np.array([row[i] for row in block_ids], dtype=kind)
        stats.insert(i, field, ids[ordinals])
    return {
        "block": block,
        "stats": stats,
        "rows": rows,
        "failed_opens": failed_opens,
        "peak_gb": round(_peak_gb(), 2),
        "pid": os.getpid(),
    }
