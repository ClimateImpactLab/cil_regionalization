"""Dask cluster context: off mode is a no-op; local mode yields a real Client."""
from __future__ import annotations

import pytest

from cil_regionalization.compute import dask_client_for
from cil_regionalization.config import LocalBackendOptions


class TestDaskOff:
    def test_off_yields_none_and_does_not_import_dask(self, monkeypatch):
        """The 'off' branch must not import dask; installs that skip the
        optional dependency must still work for the serial backend."""
        import sys

        # Remove any cached dask modules so a stray import would re-import.
        for mod in list(sys.modules):
            if mod.startswith(("dask", "distributed", "dask_jobqueue")):
                monkeypatch.delitem(sys.modules, mod, raising=False)

        def boom(*a, **k):
            raise AssertionError("dask should not be imported in 'off' mode")

        monkeypatch.setattr(
            "builtins.__import__",
            lambda name, *a, **k: boom() if name.startswith(("dask", "distributed")) else __import__(name, *a, **k),
        )

        opts = LocalBackendOptions(dask="off")
        with dask_client_for(opts) as client:
            assert client is None

    def test_default_is_off(self):
        opts = LocalBackendOptions()
        assert opts.dask == "off"
        with dask_client_for(opts) as client:
            assert client is None


@pytest.mark.dask
class TestDaskLocal:
    def test_local_cluster_yields_real_client(self):
        from dask.distributed import Client

        opts = LocalBackendOptions(dask="local", n_workers=2, threads_per_worker=1)
        with dask_client_for(opts) as client:
            assert isinstance(client, Client)
            info = client.scheduler_info()
            assert len(info["workers"]) == 2
            # Round-trip a trivial future to prove the workers are alive.
            fut = client.submit(lambda x: x * 2, 21)
            assert fut.result() == 42

    def test_client_is_closed_on_exit(self):
        from dask.distributed import Client

        opts = LocalBackendOptions(dask="local", n_workers=1)
        with dask_client_for(opts) as client:
            captured: Client = client
        # After exit, the client + cluster are closed; futures fail fast.
        assert captured.status in {"closed", "closing"}
