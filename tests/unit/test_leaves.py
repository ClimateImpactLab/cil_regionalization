"""The layout of an aggregated leaf on disk.

The frames are synthetic. Every test here fails without the module,
because the import fails. The tests that name a mutation in their
docstring were also seen to fail under it.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cil_regionalization import leaves
from cil_regionalization.leaves import (
    ORDINAL_COLUMN,
    TargetTable,
    read_leaf_footer,
    write_leaf,
)

IDS = ["T3", "T1", "T5", "T2", "T4"]
YEARS = [2051, 2050, 2052]


def _targets(ids=IDS) -> TargetTable:
    return TargetTable.from_frame(pd.DataFrame({"gid": ids}), ["gid"])


def _leaf(ids=IDS, years=YEARS) -> pd.DataFrame:
    """One row per target and year, in the order given, with a value
    that names its row."""
    rows = [(gid, year) for gid in ids for year in years]
    return pd.DataFrame(
        {
            "gid": [gid for gid, _ in rows],
            "year": [year for _, year in rows],
            "value": [int(gid[1:]) + year / 10_000 for gid, year in rows],
        }
    )


def _write(frame, tmp_path, *, targets=None, per_group=2, **kwargs):
    path = tmp_path / "k0" / "leaf.aggregated.parquet"
    written = write_leaf(
        frame,
        path,
        targets=targets or _targets(),
        value_col="value",
        identity=kwargs.pop("identity", {"levels": {"kumquat": "k0"}}),
        targets_per_group=per_group,
        **kwargs,
    )
    return path, written


class TestTargetTable:
    def test_the_ordinal_is_the_rank_of_the_sorted_ids(self):
        """Not the order of the artifact's rows. Failed with the ids
        kept in the order they first appear."""
        table = _targets()
        assert table.ids == (("T1",), ("T2",), ("T3",), ("T4",), ("T5",))
        again = TargetTable.from_frame(
            pd.DataFrame({"gid": IDS[::-1] + IDS, "weight": 1.0}), ["gid"]
        )
        assert again == table

    def test_ids_sort_by_code_point_and_by_field(self):
        frame = pd.DataFrame(
            {
                "iso": ["b", "a", "a", "B", "é"],
                "adm": ["1", "10", "9", "1", "1"],
            }
        )
        table = TargetTable.from_frame(frame, ["iso", "adm"])
        assert table.ids == (
            ("B", "1"), ("a", "10"), ("a", "9"), ("b", "1"), ("é", "1"),
        )
        assert table.ordinals(frame).tolist() == [3, 1, 2, 0, 4]
        assert table.ordinals(frame).dtype == np.int32

    def test_the_digest_is_of_the_names_and_the_ids(self):
        """The number of fields goes in first, then each name, then the
        number of targets, then each id. A name or an id goes in as its
        length in eight bytes and then its UTF-8 bytes. The literal is
        what every environment has to give."""
        table = _targets()
        expected = hashlib.sha256(b"cil_regionalization targets 1\n")
        for part in (1, "gid", 5, "T1", "T2", "T3", "T4", "T5"):
            if isinstance(part, str):
                raw = part.encode("utf-8")
                expected.update(len(raw).to_bytes(8, "big") + raw)
            else:
                expected.update(part.to_bytes(8, "big"))
        assert table.digest == expected.hexdigest()
        assert table.digest == (
            "696923e5e4bc85244b67517226e48687d1907f0830f0245d215a08e306df5a0d"
        )
        assert _targets(IDS[:4]).digest != table.digest
        other_name = TargetTable.from_frame(pd.DataFrame({"hid": IDS}), ["hid"])
        assert other_name.digest != table.digest
        # two fields are not one field with the values run together
        joined = TargetTable.from_frame(pd.DataFrame({"gid": ["ab"]}), ["gid"])
        split = TargetTable.from_frame(
            pd.DataFrame({"gid": ["a"], "x": ["b"]}), ["gid", "x"]
        )
        assert joined.digest != split.digest

    @pytest.mark.parametrize("dtype", ["object", "str", "string"])
    def test_any_string_dtype_gives_the_same_table(self, dtype):
        frame = pd.DataFrame({"gid": pd.array(IDS, dtype=dtype)})
        assert TargetTable.from_frame(frame, ["gid"]) == _targets()

    def test_an_id_that_is_not_a_string_is_refused(self):
        with pytest.raises(ValueError, match="'gid' has dtype int64"):
            TargetTable.from_frame(pd.DataFrame({"gid": [1, 2]}), ["gid"])

    def test_a_missing_id_is_refused(self):
        with pytest.raises(ValueError, match="'gid' has missing values"):
            TargetTable.from_frame(pd.DataFrame({"gid": ["T1", None]}), ["gid"])

    def test_a_target_the_table_lacks_is_refused(self):
        with pytest.raises(ValueError, match="1 targets are not in the target"):
            _targets().ordinals(pd.DataFrame({"gid": ["T1", "T9", "T9"]}))

    def test_it_is_written_under_its_digest_and_read_back(self, tmp_path):
        table = _targets()
        path = table.write(tmp_path / "leaves")
        assert path.name == f"targets.{table.digest[:16]}.parquet"
        assert TargetTable.read(path) == table
        schema = pq.read_schema(path)
        assert [(f.name, str(f.type)) for f in schema] == [
            (ORDINAL_COLUMN, "int32"), ("gid", "string"),
        ]
        assert b"pandas" not in schema.metadata

        stamp = path.stat().st_mtime_ns
        assert table.write(tmp_path / "leaves") == path
        assert path.stat().st_mtime_ns == stamp
        assert [p.name for p in path.parent.iterdir()] == [path.name]

    def test_a_file_of_other_targets_under_its_name_is_refused(self, tmp_path):
        table = _targets()
        other = _targets(IDS[:4])
        other.write(tmp_path)
        (tmp_path / other.filename).rename(tmp_path / table.filename)
        with pytest.raises(ValueError, match="does not hold the targets"):
            table.write(tmp_path)


class TestLayout:
    def test_rows_are_sorted_by_target_and_keep_their_order_within_one(
        self, tmp_path
    ):
        """The years of a target stay as the leaf had them, 2051, 2050,
        2052. Failed with a sort that is not stable, on the second leaf
        here, which is long enough for such a sort to move rows."""
        many = [f"T{i:04d}" for i in np.random.default_rng(7).permutation(2000)]
        path, _ = _write(_leaf(many), tmp_path, targets=_targets(many))
        back = pq.read_table(path).to_pandas()
        assert back["gid"].tolist() == [g for g in sorted(many) for _ in YEARS]
        assert back["year"].tolist() == YEARS * 2000

        frame = _leaf()
        path, _ = _write(frame, tmp_path)
        back = pq.read_table(path).to_pandas()
        assert list(back.columns) == [ORDINAL_COLUMN, "gid", "year", "value"]
        assert back[ORDINAL_COLUMN].tolist() == [o for o in range(5) for _ in YEARS]
        assert back["gid"].tolist() == [f"T{o + 1}" for o in range(5) for _ in YEARS]
        assert back["year"].tolist() == YEARS * 5
        expected = frame.set_index(["gid", "year"])["value"]
        got = back.set_index(["gid", "year"])["value"]
        assert got.to_dict() == expected.to_dict()

    def test_a_group_holds_a_fixed_range_of_targets(self, tmp_path):
        """T2 and T3 are not in this leaf, so the group of ordinals 2
        and 3 holds T4 alone and no group moves up to fill a gap."""
        frame = _leaf(["T5", "T1", "T4"])
        path, written = _write(frame, tmp_path)
        metadata = pq.read_metadata(path)
        assert metadata.num_row_groups == 3
        assert written.rows_per_group == [3, 3, 3]
        file = pq.ParquetFile(path)
        assert file.read_row_group(0)["gid"].to_pylist() == ["T1"] * 3
        assert file.read_row_group(1)["gid"].to_pylist() == ["T4"] * 3
        assert file.read_row_group(2)["gid"].to_pylist() == ["T5"] * 3

    def test_a_group_the_leaf_has_nothing_of_is_empty(self, tmp_path):
        """Failed with the empty groups left out, which moves every
        later group up."""
        path, written = _write(_leaf(["T5"]), tmp_path)
        metadata = pq.read_metadata(path)
        assert written.rows_per_group == [0, 0, 3]
        assert [metadata.row_group(g).num_rows for g in range(3)] == [0, 0, 3]
        assert pq.ParquetFile(path).read_row_group(1).num_rows == 0

    def test_a_leaf_with_no_rows_has_every_group_empty(self, tmp_path):
        path, written = _write(_leaf([]).astype({"gid": object}), tmp_path)
        assert written.rows_per_group == [0, 0, 0]
        assert pq.read_metadata(path).num_row_groups == 3

    def test_the_footer_statistics_give_each_group_its_ordinals(self, tmp_path):
        path, _ = _write(_leaf(), tmp_path)
        metadata = pq.read_metadata(path)
        assert metadata.schema.names[0] == ORDINAL_COLUMN
        ranges = [
            (
                metadata.row_group(g).column(0).statistics.min,
                metadata.row_group(g).column(0).statistics.max,
            )
            for g in range(metadata.num_row_groups)
        ]
        assert ranges == [(0, 1), (2, 3), (4, 4)]

    def test_a_group_above_the_default_size_stays_one_group(self, tmp_path):
        """pyarrow cuts a write above 1,048,576 rows in two unless it
        is told the size. Failed with the size left out."""
        n = 1_100_000
        frame = pd.DataFrame(
            {
                "gid": ["T2"] * n + ["T4"],
                "year": list(range(n)) + [0],
                "value": np.zeros(n + 1),
            }
        )
        path, written = _write(frame, tmp_path)
        assert written.rows_per_group == [n, 1, 0]
        metadata = pq.read_metadata(path)
        assert [metadata.row_group(g).num_rows for g in range(3)] == [n, 1, 0]

    def test_the_group_size_is_128_targets_unless_told(self, tmp_path):
        assert leaves.TARGETS_PER_GROUP == 128
        ids = [f"T{i:03d}" for i in range(130)]
        path = tmp_path / "leaf.parquet"
        written = write_leaf(
            _leaf(ids, [2050]), path, targets=_targets(ids), value_col="value",
            identity={},
        )
        assert written.rows_per_group == [128, 2]
        assert read_leaf_footer(path)["targets_per_group"] == 128


class TestSchema:
    def test_the_arrow_schema_is_fixed(self, tmp_path):
        """The same under pandas 2 and pandas 3, which write a string
        column of their own as two different Arrow types."""
        frame = _leaf().assign(band="old", flag=True, depth=np.float32(1.5))
        frame["year"] = frame["year"].astype("int32")
        frame = frame[["gid", "band", "flag", "depth", "year", "value"]]
        path, written = _write(frame, tmp_path)
        schema = pq.read_schema(path)
        assert [(f.name, str(f.type)) for f in schema] == [
            (ORDINAL_COLUMN, "int32"),
            ("gid", "string"),
            ("band", "string"),
            ("flag", "bool"),
            ("depth", "float"),
            ("year", "int32"),
            ("value", "double"),
        ]
        assert sorted(schema.metadata) == [b"cil_regionalization.leaf"]
        assert [(c["name"], c["arrow"]) for c in written.schema] == [
            (f.name, str(f.type)) for f in schema
        ]
        assert [c["pandas"] for c in written.schema] == ["int32"] + [
            str(dtype) for dtype in frame.dtypes
        ]

    def test_the_value_is_stored_plain_and_the_rest_by_dictionary(self, tmp_path):
        """Failed with a dictionary on every column."""
        path, _ = _write(_leaf(), tmp_path)
        group = pq.read_metadata(path).row_group(0)
        by_name = {group.column(i).path_in_schema: group.column(i) for i in range(4)}
        assert not any("DICTIONARY" in e for e in by_name["value"].encodings)
        for name in (ORDINAL_COLUMN, "gid", "year"):
            assert any("DICTIONARY" in e for e in by_name[name].encodings)
        assert {c.compression for c in by_name.values()} == {"SNAPPY"}

    def test_the_values_come_back_to_the_last_bit(self, tmp_path):
        frame = _leaf(["T1", "T2"])
        frame["value"] = [0.1, 1 / 3, np.nan, -0.0, 5e-324, 1.7976931348623157e308]
        path, _ = _write(frame, tmp_path)
        back = pq.read_table(path).to_pandas()["value"]
        assert back.to_numpy().tobytes() == frame["value"].to_numpy().tobytes()

    @pytest.mark.parametrize(
        "column",
        [
            pd.Categorical(["a", "b", "a"]),
            pd.to_datetime(["2050-01-01", "2051-01-01", "2052-01-01"]),
            pd.array([1, 2, 3], dtype="Int64"),
            pd.array([1.0, 2.0, None], dtype="Float64"),
            pd.array(["a", "b", "a"], dtype="string"),
            pd.array([True, False, None], dtype="boolean"),
            np.array([1, "a", 2.5], dtype=object),
        ],
        ids=["categorical", "datetime", "Int64", "Float64", "string", "boolean",
             "mixed"],
    )
    def test_a_dimension_pandas_writes_two_ways_is_refused(self, tmp_path, column):
        frame = _leaf(["T1"]).assign(band=column)[["gid", "band", "year", "value"]]
        with pytest.raises(ValueError, match="leaf column 'band' has dtype"):
            _write(frame, tmp_path)
        assert not (tmp_path / "k0").exists() or not list((tmp_path / "k0").iterdir())

    def test_a_value_that_is_not_float64_is_refused(self, tmp_path):
        frame = _leaf().astype({"value": "float32"})
        with pytest.raises(ValueError, match="'value' is float32"):
            _write(frame, tmp_path)

    def test_a_column_named_like_the_ordinal_is_refused(self, tmp_path):
        frame = _leaf().rename(columns={"year": ORDINAL_COLUMN})
        with pytest.raises(ValueError, match="already has a column named"):
            _write(frame, tmp_path)


class TestFooter:
    def test_it_holds_the_layout_and_what_it_was_given(self, tmp_path):
        identity = {
            "aggregation_digest": "abc",
            "levels": {"kumquat": "k0", "zeppelin": "z1"},
            "sources": ["/tree/k0/z1/damages.nc4"],
        }
        path, _ = _write(_leaf(), tmp_path, identity=identity)
        assert read_leaf_footer(path) == {
            **identity,
            "format_version": 1,
            "targets_per_group": 2,
            "n_targets": 5,
            "target_digest": _targets().digest,
        }
        # under one key of the parquet file, as JSON, for any reader
        raw = pq.read_metadata(path).metadata[b"cil_regionalization.leaf"]
        assert json.loads(raw) == read_leaf_footer(path)

    def test_a_file_without_the_footer_is_refused(self, tmp_path):
        _leaf().to_parquet(tmp_path / "aggregated_damages.parquet", index=False)
        with pytest.raises(ValueError, match="has no cil_regionalization.leaf"):
            read_leaf_footer(tmp_path / "aggregated_damages.parquet")


class TestWrite:
    def test_the_size_and_the_hash_are_those_of_the_file(self, tmp_path):
        path, written = _write(_leaf(), tmp_path)
        assert written.size == path.stat().st_size
        assert written.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
        assert leaves.file_sha256(path) == written.sha256

    def test_a_write_that_fails_leaves_the_file_that_was_there(
        self, tmp_path, monkeypatch
    ):
        """And nothing else in the directory. Failed with the file
        written straight under its name."""
        path, first = _write(_leaf(), tmp_path)
        before = path.read_bytes()

        def fail(tmp, final):
            raise OSError("the volume dropped")

        monkeypatch.setattr(leaves, "_replace", fail)
        with pytest.raises(OSError, match="the volume dropped"):
            _write(_leaf(["T1"]), tmp_path)
        assert path.read_bytes() == before
        assert [p.name for p in path.parent.iterdir()] == [path.name]

    def test_the_profile_counts_the_rows_at_each_value(self, tmp_path, monkeypatch):
        frame = _leaf().assign(band="old")
        frame = frame[frame["year"] != 2052].iloc[:-1]
        _, written = _write(frame[["gid", "band", "year", "value"]], tmp_path)
        assert written.profile == {"year": {"values": [2050, 2051], "rows": [4, 5]}}
        assert json.loads(json.dumps(written.as_dict())) == written.as_dict()

        monkeypatch.setattr(leaves, "PROFILE_LIMIT", 1)
        _, written = _write(frame[["gid", "band", "year", "value"]], tmp_path)
        assert written.profile == {"year": None}

    def test_the_digest_of_a_document_ignores_the_order_of_its_keys(self):
        one = leaves.canonical_digest({"a": 1, "b": {"c": [1, 2], "d": None}})
        two = leaves.canonical_digest({"b": {"d": None, "c": [1, 2]}, "a": 1})
        assert one == two
        assert one != leaves.canonical_digest({"a": 1, "b": {"c": [2, 1], "d": None}})


class TestReadGroups:
    """`read_groups` checks a leaf file against what the leaves manifest
    says of it before it reads a row."""

    @staticmethod
    def _expect(rows: int = 15, **changed) -> dict:
        return {
            "format_version": 1,
            "targets_per_group": 2,
            "n_targets": 5,
            "target_digest": _targets().digest,
            "aggregation_digest": "abc",
            "rows": rows,
            **changed,
        }

    @staticmethod
    def _file(tmp_path, frame=None):
        path, _ = _write(
            _leaf() if frame is None else frame,
            tmp_path,
            identity={"aggregation_digest": "abc"},
        )
        return path

    def test_it_reads_the_groups_asked_for_and_no_others(self, tmp_path):
        path = self._file(tmp_path)
        table, failed = leaves.read_groups(path, 1, 3, expect=self._expect())
        assert failed == 0
        assert table.column("gid").to_pylist() == ["T3"] * 3 + ["T4"] * 3 + ["T5"] * 3
        table, _ = leaves.read_groups(
            path, 0, 1, expect=self._expect(), columns=[ORDINAL_COLUMN, "value"]
        )
        assert table.column_names == [ORDINAL_COLUMN, "value"]
        assert table.column(ORDINAL_COLUMN).to_pylist() == [0] * 3 + [1] * 3
        table, _ = leaves.read_groups(path, 1, 1, expect=self._expect())
        assert table.num_rows == 0

    @pytest.mark.parametrize(
        "changed",
        [
            {"format_version": 2},
            {"targets_per_group": 4},
            {"n_targets": 6},
            {"target_digest": "0" * 64},
            {"aggregation_digest": "abd"},
        ],
        ids=lambda changed: next(iter(changed)),
    )
    def test_a_footer_that_says_otherwise_is_refused(self, tmp_path, changed):
        key = next(iter(changed))
        with pytest.raises(ValueError, match=f"its footer has {key} = "):
            leaves.read_groups(
                self._file(tmp_path), 0, 1, expect=self._expect(**changed)
            )

    def test_a_file_with_other_rows_than_the_manifest_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="has 15 rows and the leaves manifest"):
            leaves.read_groups(self._file(tmp_path), 0, 1, expect=self._expect(14))

    def test_a_file_without_the_footer_is_refused(self, tmp_path):
        _leaf().to_parquet(tmp_path / "old.parquet", index=False)
        with pytest.raises(ValueError, match="has no cil_regionalization.leaf"):
            leaves.read_groups(tmp_path / "old.parquet", 0, 1, expect=self._expect())

    def _by_hand(self, tmp_path, groups: list[list[int]]):
        """A file with the footer of a leaf and the row groups given,
        each a list of ordinals, written without `write_leaf`."""
        schema = pa.schema(
            [(ORDINAL_COLUMN, pa.int32()), ("value", pa.float64())],
            metadata={
                b"cil_regionalization.leaf": json.dumps(
                    {k: v for k, v in self._expect().items() if k != "rows"}
                ).encode()
            },
        )
        path = tmp_path / "by_hand.parquet"
        with pq.ParquetWriter(path, schema) as writer:
            for ordinals in groups:
                writer.write_table(
                    pa.table(
                        [pa.array(ordinals, pa.int32()), pa.array([0.0] * len(ordinals))],
                        schema=schema,
                    ),
                    row_group_size=max(len(ordinals), 1),
                )
        return path

    def test_a_file_with_other_groups_than_its_layout_is_refused(self, tmp_path):
        """Five targets at two a group make three groups. This file has
        its rows in two, as a writer that skips an empty group leaves
        them."""
        path = self._by_hand(tmp_path, [[0, 1], [4]])
        with pytest.raises(ValueError, match="has 2 row groups where its layout has 3"):
            leaves.read_groups(path, 0, 1, expect=self._expect(3))

    def test_a_group_with_an_ordinal_of_another_group_is_refused(self, tmp_path):
        """Only for a group that is read. The check costs nothing,
        because the range of a group is in the footer."""
        path = self._by_hand(tmp_path, [[0, 1], [2, 4], [4]])
        with pytest.raises(
            ValueError, match="row group 1 holds ordinals 2 to 4, outside its range"
        ):
            leaves.read_groups(path, 0, 2, expect=self._expect(5))
        table, _ = leaves.read_groups(path, 0, 1, expect=self._expect(5))
        assert table.num_rows == 2

    def test_an_open_that_fails_is_tried_again(self, tmp_path, monkeypatch):
        """Twice here, and the third open is the one that counts. Failed
        with one open and no more."""
        path = self._file(tmp_path)
        real, opens = pq.ParquetFile, []

        def flaky(*args, **kwargs):
            opens.append(args)
            if len(opens) <= 2:
                raise OSError("the volume dropped")
            return real(*args, **kwargs)

        monkeypatch.setattr(leaves.pq, "ParquetFile", flaky)
        table, failed = leaves.read_groups(
            path, 0, 1, expect=self._expect(), pause=0.0
        )
        assert (failed, len(opens), table.num_rows) == (2, 3, 6)

    def test_three_opens_that_fail_are_the_end(self, tmp_path, monkeypatch):
        path = self._file(tmp_path)
        opens = []

        def gone(*args, **kwargs):
            opens.append(args)
            raise OSError("the volume dropped")

        monkeypatch.setattr(leaves.pq, "ParquetFile", gone)
        with pytest.raises(OSError, match="the volume dropped"):
            leaves.read_groups(path, 0, 1, expect=self._expect(), pause=0.0)
        assert len(opens) == 3

    def test_a_file_that_is_wrong_is_not_tried_again(self, tmp_path, monkeypatch):
        path = self._file(tmp_path)
        real, opens = pq.ParquetFile, []

        def counted(*args, **kwargs):
            opens.append(args)
            return real(*args, **kwargs)

        monkeypatch.setattr(leaves.pq, "ParquetFile", counted)
        with pytest.raises(ValueError, match="rows and the leaves manifest"):
            leaves.read_groups(path, 0, 1, expect=self._expect(1), pause=0.0)
        assert len(opens) == 1
