"""Web document contract: rounding, guards, lite subsetting, additive writes."""
from __future__ import annotations

import gzip
import json
import math

import pytest

from cil_regionalization.web_json import (
    ID_SEPARATOR,
    LITE_STATISTICS,
    NAMES_OMITTED,
    QUANTILE_DESCRIPTIONS,
    check_ids_resolve,
    check_statistics_available,
    lite_document,
    round_significant,
    write_document,
)


def _metadata() -> dict:
    """A document header with the statistics a damage product carries."""
    return {
        "source": "Climate Impact Lab",
        "variable": "mortality_deaths_change",
        "scenario": {"ssp": "SSP2", "rcp": "RCP4.5", "model": "high"},
        "statistics": {
            "mean": "SMME-weighted ensemble mean",
            "q17": QUANTILE_DESCRIPTIONS["q17"],
            "q50": QUANTILE_DESCRIPTIONS["q50"],
            "q83": QUANTILE_DESCRIPTIONS["q83"],
            "q95": QUANTILE_DESCRIPTIONS["q95"],
        },
        "admin_level": "adm1",
        "count": 2,
    }


def _data() -> dict:
    """Two units over two windows, keyed by composite GADM id."""
    def unit(uid, name, base):
        return {
            "id": uid,
            "name": name,
            "country_iso": uid.split(ID_SEPARATOR)[0],
            "values": {
                "2020_2039": {"mean": base, "q17": base - 2.0,
                              "q50": base - 0.5, "q83": base + 2.0,
                              "q95": base + 4.0},
                "2040_2059": {"mean": base * 2, "q17": base, "q50": base * 1.8,
                              "q83": base * 2.4, "q95": base * 3.0},
            },
        }
    return {"AFG.1": unit("AFG.1", "Badakhshan", 10.0),
            "USA.5": unit("USA.5", "California", -3.0)}


class TestRoundSignificant:
    def test_four_significant_digits_large(self):
        assert round_significant(1234567.0) == 1235000.0

    def test_four_significant_digits_small(self):
        assert round_significant(0.000123456) == 0.0001235

    def test_negative_rounded_not_truncated(self):
        assert round_significant(-58.04999) == -58.05

    def test_zero_passes_through(self):
        assert round_significant(0.0) == 0.0

    def test_nan_passes_through(self):
        assert math.isnan(round_significant(float("nan")))

    def test_infinity_passes_through(self):
        assert round_significant(float("inf")) == float("inf")

    def test_magnitudes_in_one_document_keep_their_own_precision(self):
        # The reason for %g over round(x, n): a fixed decimal count would
        # flatten the small unit to zero or pad the large one.
        assert round_significant(0.0004321987) == 0.0004322
        assert round_significant(432198.7) == 432200.0


class TestCheckIdsResolve:
    def test_all_known_passes(self):
        check_ids_resolve(["AFG.1", "USA.5"], ["AFG.1", "USA.5", "BRA.2"],
                          context="adm1")

    def test_unknown_id_raises(self):
        with pytest.raises(ValueError, match="not in the GADM 2.0 name table"):
            check_ids_resolve(["AFG.1", "ZZZ.9"], ["AFG.1"], context="adm1")

    def test_error_names_the_count_and_the_offenders(self):
        with pytest.raises(ValueError) as exc:
            check_ids_resolve(["ZZZ.9", "QQQ.1"], [], context="adm1")
        message = str(exc.value)
        assert "2 unit ids" in message
        assert "QQQ.1" in message

    def test_error_carries_the_caller_context(self):
        with pytest.raises(ValueError, match="ensemble_adm2.parquet"):
            check_ids_resolve(["ZZZ.9"], [],
                              context="ensemble_adm2.parquet at adm2")

    def test_duplicate_ids_do_not_trip_the_guard(self):
        check_ids_resolve(["AFG.1", "AFG.1"], ["AFG.1"], context="adm1")


class TestCheckStatisticsAvailable:
    def test_subset_passes(self):
        check_statistics_available(["mean", "q50"],
                                   ["mean", "q17", "q50"], context="t")

    def test_missing_statistic_raises_naming_both_sides(self):
        with pytest.raises(ValueError) as exc:
            check_statistics_available(["mean", "q99"], ["mean", "q50"],
                                       context="t")
        message = str(exc.value)
        assert "q99" in message
        assert "q50" in message


class TestLiteDocument:
    def test_metadata_statistics_narrowed_to_request(self):
        meta, _ = lite_document(_metadata(), _data(), ["mean", "q50"])
        assert list(meta["statistics"]) == ["mean", "q50"]

    def test_payload_narrowed_in_every_window(self):
        _, data = lite_document(_metadata(), _data(), ["mean", "q50"])
        for unit in data.values():
            for stats in unit["values"].values():
                assert list(stats) == ["mean", "q50"]

    def test_values_are_unchanged_not_recomputed(self):
        full = _data()
        _, data = lite_document(_metadata(), full, ["mean"])
        assert (data["AFG.1"]["values"]["2020_2039"]["mean"]
                == full["AFG.1"]["values"]["2020_2039"]["mean"])

    def test_unit_shape_matches_the_full_document(self):
        full = _data()
        _, data = lite_document(_metadata(), full, ["mean"])
        assert list(data["AFG.1"]) == list(full["AFG.1"])

    def test_windows_all_survive(self):
        _, data = lite_document(_metadata(), _data(), ["mean"])
        assert list(data["AFG.1"]["values"]) == ["2020_2039", "2040_2059"]

    def test_unrelated_metadata_keys_preserved(self):
        meta, _ = lite_document(_metadata(), _data(), ["mean"])
        assert meta["scenario"] == {"ssp": "SSP2", "rcp": "RCP4.5",
                                    "model": "high"}
        assert meta["admin_level"] == "adm1"

    def test_inputs_not_mutated(self):
        # The caller writes the full document from the same objects
        # afterwards, so narrowing must not reach back into them.
        meta_in, data_in = _metadata(), _data()
        lite_document(meta_in, data_in, ["mean"])
        assert list(meta_in["statistics"]) == ["mean", "q17", "q50", "q83",
                                               "q95"]
        assert list(data_in["AFG.1"]["values"]["2020_2039"]) == [
            "mean", "q17", "q50", "q83", "q95"]

    def test_default_is_the_published_lite_set(self):
        meta, data = lite_document(_metadata(), _data())
        assert list(meta["statistics"]) == list(LITE_STATISTICS)
        assert list(data["AFG.1"]["values"]["2020_2039"]) == list(
            LITE_STATISTICS)

    def test_statistic_absent_from_product_raises(self):
        with pytest.raises(ValueError, match="q99"):
            lite_document(_metadata(), _data(), ["mean", "q99"])

    def test_drop_names_omits_the_name_fields(self):
        _, data = lite_document(_metadata(), _data(), ["mean"],
                                drop_names=True)
        assert list(data["AFG.1"]) == ["id", "values"]

    def test_drop_names_explains_itself_in_metadata(self):
        meta, _ = lite_document(_metadata(), _data(), ["mean"],
                                drop_names=True)
        assert meta["names"] == NAMES_OMITTED

    def test_names_kept_by_default(self):
        _, data = lite_document(_metadata(), _data(), ["mean"])
        assert data["USA.5"]["name"] == "California"
        assert data["USA.5"]["country_iso"] == "USA"


class TestWriteDocument:
    def test_round_trips_to_the_published_envelope(self, tmp_path):
        path = write_document(tmp_path / "p", "d.json.gz", _metadata(),
                              _data())
        doc = json.loads(gzip.open(path, "rt", encoding="utf-8").read())
        assert list(doc) == ["metadata", "data"]
        assert doc["data"]["AFG.1"]["name"] == "Badakhshan"
        assert doc["metadata"]["admin_level"] == "adm1"

    def test_creates_missing_parent_directories(self, tmp_path):
        path = write_document(tmp_path / "web" / "mortality_physical",
                              "d.json.gz", _metadata(), _data())
        assert path.exists()

    def test_returns_the_written_path(self, tmp_path):
        path = write_document(tmp_path, "d.json.gz", _metadata(), _data())
        assert path == tmp_path / "d.json.gz"

    def test_output_is_gzip_not_plain_json(self, tmp_path):
        path = write_document(tmp_path, "d.json.gz", _metadata(), _data())
        assert path.read_bytes()[:2] == b"\x1f\x8b"

    def test_separators_are_compact(self, tmp_path):
        path = write_document(tmp_path, "d.json.gz", _metadata(), _data())
        text = gzip.open(path, "rt", encoding="utf-8").read()
        assert text.startswith('{"metadata":{')
        assert ", " not in text
        assert ": " not in text

    def test_does_not_mutate_the_data_passed_in(self, tmp_path):
        data = _data()
        write_document(tmp_path, "d.json.gz", _metadata(), data)
        assert list(data["AFG.1"]["values"]["2020_2039"]) == [
            "mean", "q17", "q50", "q83", "q95"]


class TestWritesAreAdditive:
    """The published tree accumulates; a write must not prune siblings.

    This is the property the whole module is built around: other
    products staged for the same rsync live in the same tree.
    """

    def test_sibling_file_in_same_directory_survives(self, tmp_path):
        sibling = tmp_path / "mortality_physical.json.gz"
        sibling.write_bytes(b"untouched")
        write_document(tmp_path, "climate_tas_annual.json.gz", _metadata(),
                       _data())
        assert sibling.read_bytes() == b"untouched"

    def test_sibling_product_directory_survives(self, tmp_path):
        other = tmp_path / "mortality_physical"
        other.mkdir()
        (other / "keep.json.gz").write_bytes(b"untouched")
        write_document(tmp_path / "climate_tas_annual", "d.json.gz",
                       _metadata(), _data())
        assert (other / "keep.json.gz").read_bytes() == b"untouched"

    def test_existing_directory_contents_survive_a_rerun(self, tmp_path):
        write_document(tmp_path, "first.json.gz", _metadata(), _data())
        write_document(tmp_path, "second.json.gz", _metadata(), _data())
        assert sorted(p.name for p in tmp_path.iterdir()) == [
            "first.json.gz", "second.json.gz"]

    def test_rewriting_the_same_name_replaces_only_that_file(self, tmp_path):
        write_document(tmp_path, "d.json.gz", _metadata(), _data())
        keep = tmp_path / "other.json.gz"
        keep.write_bytes(b"untouched")
        narrowed = dict(_metadata())
        narrowed["count"] = 1
        write_document(tmp_path, "d.json.gz", narrowed, _data())
        doc = json.loads(gzip.open(tmp_path / "d.json.gz", "rt",
                                   encoding="utf-8").read())
        assert doc["metadata"]["count"] == 1
        assert keep.read_bytes() == b"untouched"


class TestFullAndLiteAgree:
    def test_lite_is_a_strict_subset_of_the_full_document(self, tmp_path):
        metadata, data = _metadata(), _data()
        full_path = write_document(tmp_path / "web", "d.json.gz", metadata,
                                   data)
        meta_l, data_l = lite_document(metadata, data)
        lite_path = write_document(tmp_path / "web-lite", "d.json.gz",
                                   meta_l, data_l)

        full = json.loads(gzip.open(full_path, "rt", encoding="utf-8").read())
        lite = json.loads(gzip.open(lite_path, "rt", encoding="utf-8").read())

        assert set(lite["data"]) == set(full["data"])
        for uid, unit in lite["data"].items():
            for window, stats in unit["values"].items():
                for name, value in stats.items():
                    assert value == full["data"][uid]["values"][window][name]

    def test_lite_is_smaller_on_disk(self, tmp_path):
        metadata, data = _metadata(), _data()
        full_path = write_document(tmp_path / "web", "d.json.gz", metadata,
                                   data)
        meta_l, data_l = lite_document(metadata, data)
        lite_path = write_document(tmp_path / "web-lite", "d.json.gz",
                                   meta_l, data_l)
        assert lite_path.stat().st_size < full_path.stat().st_size
