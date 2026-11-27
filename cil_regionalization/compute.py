"""The Dask client the local backend spreads its per-region work over.

`dask_client_for` is the one entry point, used as a context manager:

    with dask_client_for(cfg.backend.local) as client:
        ...                # `client` is None when dask = "off"

The config's `n_workers` and `threads_per_worker` override the local
defaults. A SLURM cluster takes its queue, the cores, memory and
walltime of a job and the number of jobs from the config alone, which
has no default for any of them.
"""

from __future__ import annotations

import contextlib
import logging
from typing import Iterator, Optional

from cil_regionalization.config import LocalBackendOptions


logger = logging.getLogger(__name__)


# LocalCluster reference shape: one worker per process so each gets
# its own Python interpreter (geometry ops release the GIL only
# partially); single thread per worker keeps the scheduler simple.
_LOCAL_DEFAULT_N_WORKERS = 4
_LOCAL_DEFAULT_THREADS = 1


@contextlib.contextmanager
def dask_client_for(opts: LocalBackendOptions) -> Iterator[Optional[object]]:
    """Yield a Dask distributed client, or None when ``opts.dask`` is
    ``"off"``.

    Under ``"off"`` the caller runs serially and Dask is not imported, so
    this works without the optional dependency. ``"local"`` starts a
    ``LocalCluster`` on this machine. ``"slurm"`` starts a
    ``dask_jobqueue.SLURMCluster`` with the queue, cores, memory and
    walltime of ``opts`` and scales it to ``opts.n_workers`` jobs. The
    client and cluster are closed when the ``with`` block exits. Raises
    if ``opts.dask`` is any other value, or a SLURM setting is missing.
    """
    if opts.dask == "off":
        yield None
        return

    if opts.dask == "local":
        from dask.distributed import Client, LocalCluster

        n_workers = opts.n_workers or _LOCAL_DEFAULT_N_WORKERS
        threads = opts.threads_per_worker or _LOCAL_DEFAULT_THREADS
        cluster = LocalCluster(
            n_workers=n_workers,
            threads_per_worker=threads,
            processes=True,
            dashboard_address=None,
        )
        client = Client(cluster)
        logger.info(
            "dask LocalCluster ready: %d workers x %d threads",
            n_workers,
            threads,
        )
        try:
            yield client
        finally:
            client.close()
            cluster.close()
        return

    if opts.dask == "slurm":
        # checked again, before anything is imported, for options built
        # without the model's check
        type(opts).model_validate(opts.model_dump())
        from dask.distributed import Client
        from dask_jobqueue import SLURMCluster

        cluster = SLURMCluster(
            queue=opts.slurm_queue,
            cores=opts.slurm_cores,
            memory=opts.slurm_memory,
            walltime=opts.slurm_walltime,
            processes=opts.threads_per_worker or 1,
        )
        cluster.scale(jobs=opts.n_workers)
        client = Client(cluster)
        logger.info(
            "dask SLURMCluster requested: %d jobs on queue %r",
            opts.n_workers,
            opts.slurm_queue,
        )
        try:
            yield client
        finally:
            client.close()
            cluster.close()
        return

    raise ValueError(f"unknown backend.local.dask mode: {opts.dask!r}")
