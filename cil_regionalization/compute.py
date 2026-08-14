"""Dask cluster context for the local backend.

Local-backend per-region work is embarrassingly parallel: each region's
segment building hits an independent slice of the grid, and the raster
zonal-stats step is per-segment. Wrapping that work in a Dask Client
lets a single config switch fan it out across cores (LocalCluster) or
across HPC nodes (SLURMCluster on the CIL 'cil' partition).

The context manager is the only public seam. Callers do:

    with dask_client_for(cfg.backend.local) as client:
        ...                # `client` is None when dask = "off"

so the serial path is a true no-op: no cluster is ever instantiated,
no port is opened, no Dask import happens.

SLURM defaults match the CIL 'cil' partition (64c / 251 GB / 2 nodes
typical). The TOML's `n_workers`, `threads_per_worker`, `slurm_queue`,
`slurm_walltime` override the defaults; everything else stays fixed
to keep the surface area small.
"""
from __future__ import annotations

import contextlib
import logging
from typing import Iterator, Optional

from cil_regionalization.config import LocalBackendOptions


logger = logging.getLogger(__name__)


# CIL 'cil' partition reference shape: 64 cores, 251 GB RAM per node,
# typical run uses 2 nodes. One Dask worker per node by default so
# in-process geometry caches are shared across threads.
_SLURM_DEFAULT_QUEUE = "cil"
_SLURM_DEFAULT_CORES = 64
_SLURM_DEFAULT_MEMORY = "251GB"
_SLURM_DEFAULT_WALLTIME = "02:00:00"
_SLURM_DEFAULT_N_WORKERS = 2

# LocalCluster reference shape: one worker per process so each gets
# its own Python interpreter (geometry ops release the GIL only
# partially); single thread per worker keeps the scheduler simple.
_LOCAL_DEFAULT_N_WORKERS = 4
_LOCAL_DEFAULT_THREADS = 1


@contextlib.contextmanager
def dask_client_for(opts: LocalBackendOptions) -> Iterator[Optional[object]]:
    """Yield a Dask distributed Client, or None when dask is disabled.

    ``opts.dask``:
        ``"off"``  -> yields ``None``; the caller runs serially. Dask is
                       NOT imported, so this works on installs without the
                       optional dependency.
        ``"local"`` -> spins up a ``LocalCluster`` + ``Client``.
        ``"slurm"`` -> spins up a ``dask_jobqueue.SLURMCluster`` shaped
                       for the CIL 'cil' partition and scales to
                       ``opts.n_workers`` jobs.

    The cluster and client are closed deterministically on exit; no
    background threads or sockets survive the ``with`` block.
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
        from dask.distributed import Client
        from dask_jobqueue import SLURMCluster

        n_workers = opts.n_workers or _SLURM_DEFAULT_N_WORKERS
        cluster = SLURMCluster(
            queue=opts.slurm_queue or _SLURM_DEFAULT_QUEUE,
            cores=_SLURM_DEFAULT_CORES,
            memory=_SLURM_DEFAULT_MEMORY,
            walltime=opts.slurm_walltime or _SLURM_DEFAULT_WALLTIME,
            processes=opts.threads_per_worker or 1,
        )
        cluster.scale(jobs=n_workers)
        client = Client(cluster)
        logger.info(
            "dask SLURMCluster requested: %d jobs on queue %r",
            n_workers,
            opts.slurm_queue or _SLURM_DEFAULT_QUEUE,
        )
        try:
            yield client
        finally:
            client.close()
            cluster.close()
        return

    raise ValueError(f"unknown backend.local.dask mode: {opts.dask!r}")
