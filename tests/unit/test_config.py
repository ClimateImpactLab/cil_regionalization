"""Config validation tests.

The error messages produced by config.py are part of its public interface:
users see them when their TOML is wrong, and they need to identify the
offending key. These tests pin the relevant substrings of those messages.
"""

from __future__ import annotations

import copy

import pytest
from pydantic import ValidationError

from cil_regionalization.config import Config, LocalBackendOptions, load_config


def _local_cfg() -> dict:
    return {
        "project": {"name": "test"},
        "regions": {
            "path": "/tmp/regions.shp",
            "id_fields": ["hierid"],
            "on_invalid_geometry": "repair",
        },
        "grid": {
            "mode": "generate",
            "resolution": 1.0,
            "offset": "center",
            "lon_convention": "[-180,180)",
        },
        "weights": [
            {
                "name": "pop",
                "raster": "/tmp/pop.tif",
                "fallback": "area",
                "kind": "point_sum",
            },
            {"name": "area"},
        ],
        "backend": {"kind": "local", "coverage": "exact_fraction"},
        "output": {"dir": "/tmp/out"},
        "normalization": "per_destination",
    }


def _bq_cfg() -> dict:
    return {
        "project": {"name": "test"},
        "regions": {
            "table": "compute-impactlab.spatial_aggregation.regions",
            "id_fields": ["hierid"],
            "on_invalid_geometry": "repair",
        },
        "grid": {
            "mode": "generate",
            "resolution": 1.0,
            "lon_convention": "[0,360)",
            "offset": "center",
        },
        "weights": [
            {
                "name": "pop",
                "table": "compute-impactlab.gpw.pop",
                "column": "population",
                "fallback": "area",
                "kind": "point_sum",
            },
            {"name": "area"},
        ],
        "backend": {
            "kind": "bigquery",
            "bigquery": {
                "project": "compute-impactlab",
                "temp_dataset": "temp_workspace",
                "staging_uri": "gs://example-staging/segment-weights/",
            },
            "coverage": "exact_fraction",
        },
        "output": {"dir": "gs://impactlab-data-scratch/test"},
        "normalization": "per_destination",
    }


def _polygon_cfg() -> dict:
    d = _local_cfg()
    del d["grid"]
    d["regions"]["version"] = "targets-v1"
    d["source"] = {
        "path": "/tmp/units.parquet",
        "id_fields": ["unit_id"],
        "version": "units-v1",
        "on_invalid_geometry": "repair",
    }
    return d


def _without(d: dict, path: str) -> dict:
    """``d`` with the key at the dotted ``path`` deleted, a weight named
    in brackets."""
    d = copy.deepcopy(d)
    *parents, key = path.split(".")
    node = d
    for part in parents:
        if part.startswith("weights["):
            name = part[len("weights[") : -1]
            node = next(w for w in node["weights"] if w["name"] == name)
        else:
            node = node[part]
    del node[key]
    return d


class TestRequired:
    """Keys whose wrong value would give another number in silence, which
    a config states and which are refused by name when left out. Each
    went red with the key defaulted."""

    @pytest.mark.parametrize(
        "config, path, named",
        [
            (_local_cfg, "normalization", "normalization"),
            (_local_cfg, "regions.on_invalid_geometry", "regions.on_invalid_geometry"),
            (_polygon_cfg, "source.on_invalid_geometry", "source.on_invalid_geometry"),
            (_local_cfg, "grid.offset", "grid.offset"),
            (_local_cfg, "grid.lon_convention", "grid.lon_convention"),
            (_local_cfg, "backend.coverage", "backend.coverage"),
            (_local_cfg, "weights[pop].fallback", "weights.pop.fallback"),
            (_local_cfg, "weights[pop].kind", "weights.pop.kind"),
            (_bq_cfg, "weights[pop].column", "weights.pop.column"),
        ],
    )
    def test_a_config_without_the_key_is_refused_naming_it(self, config, path, named):
        assert Config.model_validate(config())
        with pytest.raises(ValidationError) as exc:
            Config.model_validate(_without(config(), path))
        assert named in str(exc.value)

    @pytest.mark.parametrize(
        "key, value", [("fallback", "area"), ("kind", "point_sum")]
    )
    def test_the_area_weight_takes_neither_fallback_nor_kind(self, key, value):
        """Its total is the region's own area and it reads no raster, so
        either would claim a choice that is never made."""
        d = _local_cfg()
        d["weights"][1][key] = value
        with pytest.raises(ValidationError, match=f"weights.area takes no {key}"):
            Config.model_validate(d)

    def test_a_local_weight_with_a_column_is_refused(self):
        d = _local_cfg()
        d["weights"][0]["column"] = "population"
        with pytest.raises(ValidationError, match="names a column of a BigQuery"):
            Config.model_validate(d)


class TestValidConfigs:
    def test_minimal_local_applies_defaults(self):
        cfg = Config.model_validate(_local_cfg())
        assert cfg.backend.kind == "local"
        assert cfg.backend.local.dask == "off"
        assert cfg.output.format == "parquet"
        assert cfg.validation.sum_tolerance == 1e-6

    def test_minimal_bigquery_applies_defaults(self):
        cfg = Config.model_validate(_bq_cfg())
        assert cfg.backend.kind == "bigquery"
        # the project and the temp dataset are declared, as the fixture does
        assert cfg.backend.bigquery.project == "compute-impactlab"
        assert cfg.backend.bigquery.temp_dataset == "temp_workspace"
        assert cfg.backend.bigquery.cache_temp_tables is False
        # staging_uri has no default (a writable bucket is a site fact);
        # the fixture supplies one and it comes through verbatim.
        assert cfg.backend.bigquery.staging_uri == (
            "gs://example-staging/segment-weights/"
        )
        assert cfg.backend.bigquery.temp_table_expiration_hours == 24
        # Default off: an install with the bigquery-storage package present
        # is inert rather than a 403 trap for accounts without the
        # bigquery.readSessionUser role.
        assert cfg.backend.bigquery.use_bqstorage is False
        assert cfg.grid.lon_convention == "[0,360)"

    def test_weight_kind_area_weighted_sum_allowed(self):
        data = _local_cfg()
        data["weights"] = [
            {
                "name": "crop",
                "raster": "/tmp/cropland2000_area.tif",
                "kind": "area_weighted_sum",
                "fallback": "nan",
            },
            {"name": "area"},
        ]
        cfg = Config.model_validate(data)
        crop = next(w for w in cfg.weights if w.name == "crop")
        assert crop.kind == "area_weighted_sum"
        assert crop.fallback == "nan"

    def test_on_null_geometry_default_is_error(self):
        # The library default must protect: silently dropping NULL hierids
        # would lose 17 of 24,378 regions on s51 without anyone noticing.
        # s51-specific configs flip to 'skip' explicitly with comments.
        cfg = Config.model_validate(_local_cfg())
        assert cfg.regions.on_null_geometry == "error"

    def test_on_null_geometry_skip_allowed(self):
        d = _local_cfg()
        d["regions"]["on_null_geometry"] = "skip"
        cfg = Config.model_validate(d)
        assert cfg.regions.on_null_geometry == "skip"

    def test_on_unknown_id_default_is_error(self):
        # Same protection rationale as on_null_geometry: a silent skip
        # at the library level would hide request-side typos / stale
        # configs (AND/BMU not in this impact region version was caught by the
        # runner's coverage check, not by anyone reviewing the diff).
        cfg = Config.model_validate(_local_cfg())
        assert cfg.regions.on_unknown_id == "error"

    def test_on_unknown_id_skip_allowed(self):
        d = _local_cfg()
        d["regions"]["on_unknown_id"] = "skip"
        cfg = Config.model_validate(d)
        assert cfg.regions.on_unknown_id == "skip"

    def test_confirm_cost_default_is_true(self):
        # Default-True keeps runners safe under nbconvert / cron / SLURM
        # by refusing to proceed without an explicit acknowledgement.
        cfg = Config.model_validate(_local_cfg())
        assert cfg.output.confirm_cost is True

    def test_confirm_cost_false_allowed(self):
        d = _local_cfg()
        d["output"]["confirm_cost"] = False
        cfg = Config.model_validate(d)
        assert cfg.output.confirm_cost is False

    def test_dry_run_byte_ceiling_default_is_10gb(self):
        # Default must protect: the s51 case's worst-case GPW scan is ~6 GB,
        # so 10 GB is the smallest ceiling that lets the s51 workload through
        # without an override. Anything larger (the previous 500 GB) is a
        # defense-in-depth regression. The measurement justifying the number
        # is recorded in examples/configs/s51.toml: 5,257,870,075 bytes
        # dry-run estimate for the full 24,361-valid-region GPW scan.
        cfg = Config.model_validate(_bq_cfg())
        assert cfg.backend.bigquery.dry_run_byte_ceiling == 10_000_000_000

    def test_dry_run_byte_ceiling_overridable(self):
        d = _bq_cfg()
        d["backend"]["bigquery"]["dry_run_byte_ceiling"] = 50_000_000_000
        cfg = Config.model_validate(d)
        assert cfg.backend.bigquery.dry_run_byte_ceiling == 50_000_000_000

    def test_prebuilt_grid_from_path(self):
        d = _local_cfg()
        d["grid"] = {
            "mode": "prebuilt",
            "path": "/tmp/grid.parquet",
            "offset": "center",
            "lon_convention": "[-180,180)",
        }
        cfg = Config.model_validate(d)
        assert cfg.grid.mode == "prebuilt"
        assert cfg.grid.path == "/tmp/grid.parquet"

    def test_prebuilt_grid_from_table(self):
        d = _bq_cfg()
        d["grid"] = {
            "mode": "prebuilt",
            "table": "compute-impactlab.foo.grid",
            "offset": "center",
            "lon_convention": "[-180,180)",
        }
        cfg = Config.model_validate(d)
        assert cfg.grid.table == "compute-impactlab.foo.grid"

    def test_load_from_toml(self, tmp_path):
        toml_text = """
normalization = "per_destination"

[project]
name = "demo"
description = "exercise"

[regions]
path = "/tmp/regions.shp"
id_fields = ["hierid"]
on_invalid_geometry = "repair"

[grid]
mode = "generate"
resolution = 0.25
offset = "edge"
lon_convention = "[-180,180)"

[[weights]]
name = "pop"
raster = "/tmp/pop.tif"
fallback = "area"
kind = "point_sum"

[[weights]]
name = "area"

[backend]
kind = "local"
coverage = "exact_fraction"

[backend.local]
dask = "local"
n_workers = 4

[output]
dir = "./data/out"
format = "both"

[validation]
sum_tolerance = 0.0001
"""
        p = tmp_path / "cfg.toml"
        p.write_text(toml_text)
        cfg = load_config(p)
        assert cfg.project.name == "demo"
        assert cfg.grid.resolution == 0.25
        assert cfg.grid.offset == "edge"
        assert cfg.backend.local.dask == "local"
        assert cfg.backend.local.n_workers == 4
        assert cfg.output.format == "both"
        assert cfg.validation.sum_tolerance == 0.0001

    def test_load_config_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError) as exc:
            load_config(tmp_path / "nope.toml")
        assert "nope.toml" in str(exc.value)


class TestGridErrors:
    def test_generate_with_path_is_ambiguous(self):
        d = _local_cfg()
        d["grid"] = {
            "mode": "generate",
            "resolution": 1.0,
            "path": "/tmp/g.shp",
            "offset": "center",
            "lon_convention": "[-180,180)",
        }
        with pytest.raises(ValidationError) as exc:
            Config.model_validate(d)
        msg = str(exc.value)
        assert "grid.path" in msg
        assert "ambiguous" in msg

    def test_generate_with_table_is_ambiguous(self):
        d = _bq_cfg()
        d["grid"] = {
            "mode": "generate",
            "resolution": 1.0,
            "table": "x.y.z",
            "offset": "center",
            "lon_convention": "[-180,180)",
        }
        with pytest.raises(ValidationError) as exc:
            Config.model_validate(d)
        msg = str(exc.value)
        assert "grid.table" in msg
        assert "ambiguous" in msg

    def test_generate_without_resolution(self):
        d = _local_cfg()
        d["grid"] = {
            "mode": "generate",
            "offset": "center",
            "lon_convention": "[-180,180)",
        }
        with pytest.raises(ValidationError) as exc:
            Config.model_validate(d)
        assert "grid.resolution" in str(exc.value)

    def test_prebuilt_with_resolution_is_ambiguous(self):
        d = _local_cfg()
        d["grid"] = {
            "mode": "prebuilt",
            "resolution": 1.0,
            "path": "/tmp/g.shp",
            "offset": "center",
            "lon_convention": "[-180,180)",
        }
        with pytest.raises(ValidationError) as exc:
            Config.model_validate(d)
        msg = str(exc.value)
        assert "grid.resolution" in msg
        assert "ambiguous" in msg

    def test_prebuilt_without_source(self):
        d = _local_cfg()
        d["grid"] = {
            "mode": "prebuilt",
            "offset": "center",
            "lon_convention": "[-180,180)",
        }
        with pytest.raises(ValidationError) as exc:
            Config.model_validate(d)
        msg = str(exc.value)
        assert "grid.path" in msg
        assert "grid.table" in msg

    def test_prebuilt_with_both_path_and_table(self):
        d = _local_cfg()
        d["grid"] = {
            "mode": "prebuilt",
            "path": "/tmp/g.shp",
            "table": "x.y.z",
            "offset": "center",
            "lon_convention": "[-180,180)",
        }
        with pytest.raises(ValidationError) as exc:
            Config.model_validate(d)
        msg = str(exc.value)
        assert "grid.path" in msg
        assert "grid.table" in msg
        assert "mutually exclusive" in msg

    def test_negative_resolution(self):
        d = _local_cfg()
        d["grid"] = {
            "mode": "generate",
            "resolution": -1.0,
            "offset": "center",
            "lon_convention": "[-180,180)",
        }
        with pytest.raises(ValidationError) as exc:
            Config.model_validate(d)
        assert "grid.resolution" in str(exc.value) or "resolution" in str(exc.value)

    def test_unknown_lon_convention(self):
        d = _local_cfg()
        d["grid"]["lon_convention"] = "[0,180)"
        with pytest.raises(ValidationError) as exc:
            Config.model_validate(d)
        assert "lon_convention" in str(exc.value)


class TestWeightErrors:
    def test_local_non_area_without_raster(self):
        d = _local_cfg()
        d["weights"] = [
            {"name": "pop", "fallback": "area", "kind": "point_sum"},
            {"name": "area"},
        ]
        with pytest.raises(ValidationError) as exc:
            Config.model_validate(d)
        msg = str(exc.value)
        assert "weights.pop" in msg
        assert "raster" in msg
        assert "local" in msg

    def test_bigquery_non_area_without_table(self):
        d = _bq_cfg()
        d["weights"] = [
            {"name": "pop", "fallback": "area", "kind": "point_sum"},
            {"name": "area"},
        ]
        with pytest.raises(ValidationError) as exc:
            Config.model_validate(d)
        msg = str(exc.value)
        assert "weights.pop" in msg
        assert "table" in msg
        assert "bigquery" in msg

    def test_area_weight_rejects_raster(self):
        d = _local_cfg()
        d["weights"] = [{"name": "area", "raster": "/tmp/a.tif"}]
        with pytest.raises(ValidationError) as exc:
            Config.model_validate(d)
        msg = str(exc.value)
        assert "area" in msg
        assert "does not accept" in msg

    def test_area_weight_rejects_table(self):
        d = _bq_cfg()
        d["weights"] = [{"name": "area", "table": "x.y.z"}]
        with pytest.raises(ValidationError) as exc:
            Config.model_validate(d)
        assert "area" in str(exc.value)

    def test_duplicate_weight_names(self):
        d = _local_cfg()
        d["weights"] = [
            {
                "name": "pop",
                "raster": "/tmp/p1.tif",
                "fallback": "area",
                "kind": "point_sum",
            },
            {
                "name": "pop",
                "raster": "/tmp/p2.tif",
                "fallback": "area",
                "kind": "point_sum",
            },
        ]
        with pytest.raises(ValidationError) as exc:
            Config.model_validate(d)
        msg = str(exc.value)
        assert "duplicate" in msg
        assert "pop" in msg

    def test_empty_weights_list(self):
        d = _local_cfg()
        d["weights"] = []
        with pytest.raises(ValidationError) as exc:
            Config.model_validate(d)
        assert "weights" in str(exc.value)


class TestRegionsErrors:
    def test_both_path_and_table(self):
        d = _local_cfg()
        d["regions"] = {
            "path": "/tmp/r.shp",
            "table": "x.y.z",
            "id_fields": ["hierid"],
            "on_invalid_geometry": "repair",
        }
        with pytest.raises(ValidationError) as exc:
            Config.model_validate(d)
        msg = str(exc.value)
        assert "regions.path" in msg
        assert "regions.table" in msg
        assert "mutually exclusive" in msg

    def test_neither_path_nor_table(self):
        d = _local_cfg()
        d["regions"] = {"id_fields": ["hierid"], "on_invalid_geometry": "repair"}
        with pytest.raises(ValidationError) as exc:
            Config.model_validate(d)
        msg = str(exc.value)
        assert "regions.path" in msg
        assert "regions.table" in msg

    def test_empty_id_fields(self):
        d = _local_cfg()
        d["regions"] = {
            "path": "/tmp/r.shp",
            "id_fields": [],
            "on_invalid_geometry": "repair",
        }
        with pytest.raises(ValidationError) as exc:
            Config.model_validate(d)
        assert "id_fields" in str(exc.value)

    def test_unknown_on_invalid_geometry(self):
        d = _local_cfg()
        d["regions"]["on_invalid_geometry"] = "ignore"
        with pytest.raises(ValidationError) as exc:
            Config.model_validate(d)
        assert "on_invalid_geometry" in str(exc.value)


class TestBackendErrors:
    def test_unknown_backend_kind(self):
        d = _local_cfg()
        d["backend"]["kind"] = "spark"
        with pytest.raises(ValidationError) as exc:
            Config.model_validate(d)
        assert "kind" in str(exc.value)

    def test_unknown_coverage_mode(self):
        d = _local_cfg()
        d["backend"]["coverage"] = "midpoint"
        with pytest.raises(ValidationError) as exc:
            Config.model_validate(d)
        assert "coverage" in str(exc.value)

    def test_unknown_dask_mode(self):
        d = _local_cfg()
        d["backend"]["local"] = {"dask": "ray"}
        with pytest.raises(ValidationError) as exc:
            Config.model_validate(d)
        assert "dask" in str(exc.value)


class TestMissingRequiredSections:
    @pytest.mark.parametrize(
        "key", ["project", "regions", "grid", "weights", "backend", "output"]
    )
    def test_missing_section_named(self, key):
        d = _local_cfg()
        del d[key]
        with pytest.raises(ValidationError) as exc:
            Config.model_validate(d)
        assert key in str(exc.value)


class TestBigQueryStagingUri:
    def test_missing_staging_uri_rejected(self):
        d = _bq_cfg()
        del d["backend"]["bigquery"]["staging_uri"]
        with pytest.raises(ValidationError) as exc:
            Config.model_validate(d)
        assert "staging_uri" in str(exc.value)
        assert "site fact" in str(exc.value)

    def test_local_backend_needs_no_staging_uri(self):
        cfg = Config.model_validate(_local_cfg())
        assert cfg.backend.bigquery.staging_uri is None


class TestSiteFactsAreDeclared:
    """A queue, the shape of a job, a billing project and a dataset belong
    to one cluster or one cloud account, and the config says them."""

    @pytest.mark.parametrize(
        "missing", [["project"], ["temp_dataset"], ["project", "temp_dataset"]]
    )
    def test_bigquery_needs_its_project_and_temp_dataset(self, missing):
        """They defaulted to one cloud account's, and this went red on
        that."""
        d = _bq_cfg()
        for key in missing:
            del d["backend"]["bigquery"][key]
        named = " and ".join(f"backend.bigquery.{key}" for key in missing)
        with pytest.raises(ValidationError) as exc:
            Config.model_validate(d)
        assert (
            f"backend.kind = 'bigquery' needs {named}, the billing project "
            f"and the dataset in the location of the weight tables that temp "
            f"tables go in. They belong to a cloud account, so neither has a "
            f"default."
        ) in str(exc.value)

    def test_the_local_backend_needs_neither(self):
        cfg = Config.model_validate(_local_cfg())
        assert cfg.backend.bigquery.project is None
        assert cfg.backend.bigquery.temp_dataset is None

    def test_slurm_needs_every_setting_of_its_jobs(self):
        """They defaulted to one cluster's partition, and this went red
        on that."""
        with pytest.raises(ValidationError) as exc:
            LocalBackendOptions(dask="slurm")
        assert (
            "backend.local.dask = 'slurm' needs backend.local.slurm_queue, "
            "backend.local.slurm_cores, backend.local.slurm_memory, "
            "backend.local.slurm_walltime and backend.local.n_workers. The "
            "queue, the cores, memory and walltime of a job and the number of "
            "jobs belong to a cluster, so none has a default."
        ) in str(exc.value)

    @pytest.mark.parametrize(
        "left_out",
        ["slurm_queue", "slurm_cores", "slurm_memory", "slurm_walltime", "n_workers"],
    )
    def test_slurm_needs_each_of_them(self, left_out):
        settings = dict(
            dask="slurm",
            slurm_queue="general",
            slurm_cores=8,
            slurm_memory="32GB",
            slurm_walltime="01:00:00",
            n_workers=3,
        )
        LocalBackendOptions(**settings)
        del settings[left_out]
        with pytest.raises(ValidationError, match=f"needs backend.local.{left_out}"):
            LocalBackendOptions(**settings)

    def test_the_other_dask_modes_need_none_of_them(self):
        for mode in ("off", "local"):
            assert LocalBackendOptions(dask=mode).slurm_queue is None


class TestBigQueryGridOffset:
    def test_an_edge_offset_grid_is_refused(self):
        """The SQL places cells for the centre offset. It went red with
        the edge offset accepted."""
        d = _bq_cfg()
        d["grid"]["offset"] = "edge"
        with pytest.raises(ValidationError) as exc:
            Config.model_validate(d)
        assert (
            "backend.kind = 'bigquery' takes grid.offset = 'center' only, and "
            "got 'edge'. Its SQL handles centre offsets only, so the weights of "
            "an edge-offset grid would come out half a cell off. Use "
            "backend.kind = 'local' for an edge-offset grid."
        ) in str(exc.value)

    def test_the_local_backend_takes_an_edge_offset_grid(self):
        d = _local_cfg()
        d["grid"]["offset"] = "edge"
        assert Config.model_validate(d).grid.offset == "edge"


class TestUnknownKeys:
    def test_typo_in_grid_rejected(self):
        d = _local_cfg()
        d["grid"]["resolutoin"] = 1.0
        with pytest.raises(ValidationError) as exc:
            Config.model_validate(d)
        assert "resolutoin" in str(exc.value)

    def test_typo_in_top_level_rejected(self):
        d = _local_cfg()
        d["putput"] = {"dir": "/tmp"}
        with pytest.raises(ValidationError) as exc:
            Config.model_validate(d)
        assert "putput" in str(exc.value)

    def test_unknown_weight_key_rejected(self):
        d = _local_cfg()
        d["weights"][0]["weihgt"] = 1.0
        with pytest.raises(ValidationError) as exc:
            Config.model_validate(d)
        assert "weihgt" in str(exc.value)


class TestOutputErrors:
    def test_empty_dir(self):
        d = _local_cfg()
        d["output"]["dir"] = ""
        with pytest.raises(ValidationError) as exc:
            Config.model_validate(d)
        assert "dir" in str(exc.value)

    def test_unknown_format(self):
        d = _local_cfg()
        d["output"]["format"] = "json"
        with pytest.raises(ValidationError) as exc:
            Config.model_validate(d)
        assert "format" in str(exc.value)


class TestImmutability:
    def test_frozen_default_factory_distinct_per_instance(self):
        # ValidationConfig default uses default_factory so each Config gets its own.
        a = Config.model_validate(_local_cfg())
        b = Config.model_validate(_local_cfg())
        assert a.validation is not b.validation

    def test_input_dict_not_mutated(self):
        d = _local_cfg()
        snapshot = copy.deepcopy(d)
        Config.model_validate(d)
        assert d == snapshot
