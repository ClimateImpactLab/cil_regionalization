"""One leaf through `apply_weights` and `window_means`, a process pool
run with a coarse and a fine task split, and the aggregate stage's pool
against the waves that replaced it.

The pool runs the same leaves twice, once with about 1.4 tasks a worker
and once with every task halved, at whatever worker count the machine
has.
"""

from __future__ import annotations

import logging
import math
import multiprocessing as mp
import os
import time

import pandas as pd
import pytest

from benchmarks.synthetic import (
    ID_FIELDS,
    LABELS,
    WINDOWS,
    YEARS,
    apply,
    apply_result,
    artifact,
    best,
    report,
    source_leaf,
)
from cil_regionalization.pipelines.montecarlo import LEAVES_PER_WORKER
from cil_regionalization.stats import window_means
from cil_regionalization.workers import run_in_waves

pytestmark = pytest.mark.benchmark

_KW = dict(time_col="year", windows=WINDOWS, value_col="pct", summation="fast")
_COLUMNS = ID_FIELDS + list(LABELS) + ["window", "pct"]

# Eight full-size workers need about 16 GiB, so the pool runs on leaves
# an eighth of the size unless told otherwise.
POOL_SCALE = int(os.environ.get("BENCH_POOL_SCALE", "8"))
POOL_WORKERS = int(os.environ.get("BENCH_WORKERS", min(8, os.cpu_count() or 1)))
# leaves in a coarse task; shorter tasks drown in pool start-up
POOL_LEAVES = int(os.environ.get("BENCH_LEAVES", "48"))


def test_label_placement_gives_the_same_leaf():
    art, leaf = artifact(), source_leaf()

    apply_s, result = best(lambda: apply_result(art, leaf))
    # the pipeline takes the share off before the window means
    applied = result.frame.drop(columns=["weight_share"])
    labelled = applied.assign(**LABELS)
    before_s, before = best(lambda: window_means(labelled, **_KW))
    after_s, after = best(
        lambda: window_means(applied, **_KW).assign(**LABELS)[_COLUMNS]
    )
    # labels on the leaf before the weights, which the pipeline avoids
    labelled_leaf = leaf.assign(**LABELS)
    first_s, applied_first = best(lambda: apply(art, labelled_leaf))
    first = window_means(applied_first, **_KW)[_COLUMNS]

    pd.testing.assert_frame_equal(after, before, check_exact=True)
    pd.testing.assert_frame_equal(first, before, check_exact=True)

    n_targets = len(art.frame[ID_FIELDS].drop_duplicates())
    n_absent = len(result.targets.left_out["no weight"])
    assert n_absent == n_targets - len(applied) // len(YEARS)
    report(
        f"leaf: {len(leaf):,} rows read, {len(applied):,} after "
        f"apply_weights, {len(before):,} reduced"
    )
    report(
        f"  skipped: {len(result.coverage.zero_weight):,} regions with no crop "
        f"area, so {n_absent:,} of {n_targets:,} targets are not in the output"
    )
    report(f"  apply_weights, bare leaf:               {apply_s:.3f} s")
    report(f"  apply_weights, labels on the leaf:      {first_s:.3f} s")
    report(f"  window_means, labels before reducing:   {before_s:.3f} s")
    report(f"  window_means, labels after reducing:    {after_s:.3f} s")


def _task(args: tuple[int, int]) -> list[tuple[int, int, float]]:
    """Reduce ``n_leaves`` leaves starting at ``first_seed``. The weights
    are built inside the task, as the runners load them."""
    first_seed, n_leaves = args
    # `apply_weights` warns about the skipped regions on every call, and
    # a spawned worker has no handler to capture it: one line a leaf on
    # stderr. The single-leaf test above prints the count once.
    logging.getLogger("cil_regionalization.validate").setLevel(logging.ERROR)
    art = artifact(POOL_SCALE)
    out = []
    for seed in range(first_seed, first_seed + n_leaves):
        applied = apply(art, source_leaf(POOL_SCALE, seed))
        reduced = window_means(applied, **_KW)
        out.append((seed, len(reduced), float(reduced["pct"].sum())))
    return out


def _pool_run(n_tasks: int, n_leaves: int) -> tuple[float, list]:
    tasks = [(i * n_leaves, n_leaves) for i in range(n_tasks)]
    start = time.perf_counter()
    with mp.Pool(processes=POOL_WORKERS) as pool:
        chunks = pool.map(_task, tasks, chunksize=1)
    seconds = time.perf_counter() - start
    return seconds, sorted(row for chunk in chunks for row in chunk)


def test_finer_tasks_give_the_same_leaves():
    n_coarse = math.ceil(1.375 * POOL_WORKERS)
    half = max(1, POOL_LEAVES // 2)

    coarse_s, coarse = _pool_run(n_coarse, 2 * half)
    fine_s, fine = _pool_run(2 * n_coarse, half)
    assert fine == coarse

    report(
        f"pool of {POOL_WORKERS} workers, {len(coarse)} leaves at 1/{POOL_SCALE} size"
    )
    report(f"  {n_coarse} tasks of {2 * half} leaves:  {coarse_s:.1f} s")
    report(
        f"  {2 * n_coarse} tasks of {half} leaves:  {fine_s:.1f} s "
        f"({fine_s / coarse_s:.0%} of the coarse run)"
    )


_WORKER: dict = {}


def _load_weights() -> None:
    """The initializer of both pools. The weights load once a worker, as
    the pipeline's do."""
    logging.getLogger("cil_regionalization.validate").setLevel(logging.ERROR)
    _WORKER["art"] = artifact(POOL_SCALE)


def _one_leaf(seed: int) -> tuple[int, int, float, float]:
    """The leaf's row count and sum, and the seconds it took in the
    worker."""
    start = time.perf_counter()
    reduced = window_means(apply(_WORKER["art"], source_leaf(POOL_SCALE, seed)), **_KW)
    return seed, len(reduced), float(reduced["pct"].sum()), time.perf_counter() - start


def _recycled_pool(seeds: list[int]) -> list:
    """What the aggregate stage ran before the waves."""
    with mp.Pool(
        processes=POOL_WORKERS,
        initializer=_load_weights,
        maxtasksperchild=LEAVES_PER_WORKER,
    ) as pool:
        return sorted(pool.imap_unordered(_one_leaf, seeds, chunksize=1))


def _waves(seeds: list[int]) -> list:
    got: list = []
    run_in_waves(
        _one_leaf,
        seeds,
        n_workers=POOL_WORKERS,
        per_worker=LEAVES_PER_WORKER,
        initializer=_load_weights,
        initargs=(),
        on_result=got.append,
        name=str,
        what="benchmark",
    )
    return sorted(got)


def test_waves_give_what_the_recycled_pool_gave():
    """Three waves of leaves, one a task, under the platform's start
    method, each run twice and alternating. The time a leaf takes moves
    from one run to the next on a laptop by more than the pools differ,
    so what is compared is the wall time not spent on leaves, the wall
    less the leaves' own seconds over the workers."""
    seeds = list(range(3 * POOL_WORKERS * LEAVES_PER_WORKER))
    runs: dict[str, list[tuple[float, float]]] = {"pool": [], "waves": []}
    results = {}
    for _ in range(2):
        for key, run in (("pool", _recycled_pool), ("waves", _waves)):
            start = time.perf_counter()
            got = run(seeds)
            wall = time.perf_counter() - start
            runs[key].append((wall, wall - sum(r[3] for r in got) / POOL_WORKERS))
            results[key] = [r[:3] for r in got]
    assert results["waves"] == results["pool"]

    report(
        f"{len(seeds)} leaves at 1/{POOL_SCALE} size on {POOL_WORKERS} "
        f"workers, {LEAVES_PER_WORKER} a worker, "
        f"{mp.get_start_method()} start"
    )
    for key, label in (
        ("pool", "pool renewing each worker after its leaves"),
        ("waves", "waves of new workers"),
    ):
        report(
            f"  {label + ':':<44} wall "
            + ", ".join(f"{wall:.1f}" for wall, _ in runs[key])
            + " s, not on leaves "
            + ", ".join(f"{other:.1f}" for _, other in runs[key])
            + " s"
        )
