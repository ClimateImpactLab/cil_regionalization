"""Pools that stop when a worker dies, in waves of new workers."""

from __future__ import annotations

import os
import signal
import time

import pytest

from cil_regionalization.workers import WorkerDied, how_workers_ended, run_in_waves


def _setup() -> None:
    pass


def _failing_setup() -> None:
    raise ValueError("this setup fails")


def _pid_of(task: int) -> tuple[int, int]:
    return task, os.getpid()


def _killed_on_three(task: int) -> int:
    if task == 3:
        os.kill(os.getpid(), signal.SIGKILL)
    time.sleep(0.05)
    return task


def _run(fn, tasks, *, initializer=_setup, n_workers=2, per_worker=2):
    got = []
    run_in_waves(
        fn,
        tasks,
        n_workers=n_workers,
        per_worker=per_worker,
        initializer=initializer,
        initargs=(),
        on_result=got.append,
        name=lambda task: f"task {task}",
        what="test",
    )
    return got


def test_every_task_runs_once_on_new_workers_each_wave():
    """Waves of four, so the workers of one wave are none of the next.
    It went red on one pool for every task."""
    got = _run(_pid_of, list(range(10)))
    assert sorted(task for task, _ in got) == list(range(10))
    waves = [
        {pid for task, pid in got if start <= task < start + 4} for start in (0, 4, 8)
    ]
    assert not waves[0] & waves[1] and not waves[1] & waves[2]


def test_a_worker_killed_stops_the_run_at_once_and_says_how():
    """It hung on the pool before, waiting for the task the dead worker
    held."""
    began = time.monotonic()
    with pytest.raises(WorkerDied) as error:
        _run(_killed_on_three, list(range(8)), per_worker=4)
    assert time.monotonic() - began < 60
    message = str(error.value)
    # two workers, so task 3 and one of 2 and 4 were out
    assert message.startswith("test: a worker died with task "), message
    assert " in flight, killed by SIGKILL, which is how an out-of-memory " in message
    held = message.split("died with ")[1].split(" in flight")[0]
    assert "task 3" in held and len(held.replace(" and ", ", ").split(", ")) == 2, held


def test_a_failing_setup_is_said_to_be_one():
    """Not blamed on memory. Its exit code is 0, and its traceback goes
    to the log."""
    with pytest.raises(WorkerDied) as error:
        _run(_pid_of, list(range(4)), initializer=_failing_setup)
    assert "task 0 and task 1 in flight, exiting with code 0 before it " in str(
        error.value
    )
    assert "out-of-memory" not in str(error.value)


@pytest.mark.parametrize(
    "codes, told",
    [
        (
            [-9, -15],
            "killed by SIGKILL, which is how an out-of-memory kill ends a process, "
            "and the pool then stopped the other 1 with SIGTERM",
        ),
        ([1], "exiting with code 1"),
        (
            [-11, -15, -15],
            "killed by SIGSEGV, and the pool then stopped the other 2 with SIGTERM",
        ),
        ([-15, -15], "killed by SIGTERM, as 2 workers did"),
        ([], "and its exit code could not be read"),
    ],
)
def test_the_exit_codes_are_told_as_they_are(codes, told):
    assert how_workers_ended(codes) == told
