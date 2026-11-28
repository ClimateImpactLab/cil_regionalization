"""Process pools that stop when a worker dies.

A worker that dies outright, killed by a signal or ending while it sets
itself up, leaves a `multiprocessing.Pool` waiting for ever for the task
it held. A `ProcessPoolExecutor` raises `BrokenProcessPool` instead.
`run_in_waves` runs tasks on one, under the platform's own start method,
with new workers for every wave of tasks so that none lives for more
than its wave. `shut_down_broken` ends a broken pool and says how its
workers ended, from their exit codes.
"""

# The executor's own recycling, max_tasks_per_child, refuses the fork
# start method and does not exist on Python 3.10, so the workers are
# renewed by starting a new executor for each wave instead.

from __future__ import annotations

import itertools
import multiprocessing
import signal
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from typing import Any, Callable, Sequence


class WorkerDied(RuntimeError):
    """A worker of a pool ended without handing back its task."""


def run_in_waves(
    fn: Callable[[Any], Any],
    tasks: Sequence[Any],
    *,
    n_workers: int,
    per_worker: int,
    initializer: Callable[..., None],
    initargs: tuple,
    on_result: Callable[[Any], None],
    name: Callable[[Any], str],
    what: str,
) -> None:
    """``fn`` of every task, each result handed to ``on_result`` as it
    comes back. The tasks run in waves of ``n_workers * per_worker``,
    each on new workers set up by ``initializer``, so a worker takes
    ``per_worker`` tasks on average and at most a wave. No more tasks
    are handed out than there are workers, so a worker that dies raises
    `WorkerDied` naming by ``name`` every task in flight, and saying how
    the workers ended."""
    wave = n_workers * per_worker
    context = multiprocessing.get_context()
    for start in range(0, len(tasks), wave):
        batch = iter(tasks[start : start + wave])
        executor = ProcessPoolExecutor(
            max_workers=min(n_workers, len(tasks) - start),
            mp_context=context,
            initializer=initializer,
            initargs=initargs,
        )
        try:
            out: dict = {}
            for task in itertools.islice(batch, n_workers):
                out[executor.submit(fn, task)] = task
            while out:
                done, _ = wait(out, return_when=FIRST_COMPLETED)
                for future in done:
                    if not isinstance(future.exception(), BrokenProcessPool):
                        del out[future]
                        on_result(future.result())
                if any(future in out for future in done):
                    ended, _ = shut_down_broken(executor)
                    held = [name(task) for task in out.values()]
                    raise WorkerDied(
                        f"{what}: a worker died with {_and(held)} in flight, "
                        f"{ended}. The run stopped there."
                    ) from None
                for task in itertools.islice(batch, n_workers - len(out)):
                    out[executor.submit(fn, task)] = task
        finally:
            executor.shutdown(wait=True, cancel_futures=True)


def _and(names: list[str]) -> str:
    return names[0] if len(names) == 1 else f"{', '.join(names[:-1])} and {names[-1]}"


def shut_down_broken(executor: ProcessPoolExecutor) -> tuple[str, bool]:
    """Shut a broken executor down, and say how its workers ended from
    their exit codes, with whether one was killed by SIGKILL. The pool
    stops the workers still alive with SIGTERM when one dies, so those
    are told apart."""
    # The process table is private and shutting down empties it, so the
    # workers are taken from it first and their exit codes read after.
    workers = list((getattr(executor, "_processes", None) or {}).values())
    executor.shutdown(wait=True)
    for worker in workers:
        worker.join(timeout=5)
    codes = [w.exitcode for w in workers if w.exitcode is not None]
    return how_workers_ended(codes), -signal.SIGKILL in codes


def how_workers_ended(codes: Sequence[int]) -> str:
    """How the workers of a broken pool ended, from their exit codes."""
    stopped = list(codes).count(-signal.SIGTERM)
    ended = [code for code in codes if code != -signal.SIGTERM]
    if not codes:
        return "and its exit code could not be read"
    if not ended:
        return f"{_ended(-signal.SIGTERM)}, as {stopped} workers did"
    told = "; ".join(_ended(code) for code in ended)
    if stopped:
        told += f", and the pool then stopped the other {stopped} with SIGTERM"
    return told


def _ended(code: int) -> str:
    if code == 0:
        return (
            "exiting with code 0 before it handed back a task, as a worker "
            "does whose setup raised, which logs the traceback"
        )
    if code > 0:
        return f"exiting with code {code}"
    try:
        name = signal.Signals(-code).name
    except ValueError:
        name = f"signal {-code}"
    if -code == signal.SIGKILL:
        return f"killed by {name}, which is how an out-of-memory kill ends a process"
    return f"killed by {name}"
