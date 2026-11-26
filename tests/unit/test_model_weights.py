"""Model weights are read once, refused when ambiguous, and recorded.

The tables are synthetic. Every test here fails without the module,
because the import fails. The tests that name a mutation in their
docstring were also seen to fail under it.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path

import pandas as pd
import pytest

import cil_regionalization as cilreg
from cil_regionalization.model_weights import ModelWeights, read_model_weights

MODELS = [
    "mri-cgcm3_01",
    "gfdl-esm2g",
    "gfdl-esm2g_01",
    "inmcm4*",
    "access1-0*",
    "mri-cgcm3*",
]
WEIGHTS = [0.1, 1 / 3, 0.036, 0.2, 0.0, 0.3306666666666667]
# the tree has no leaf of the last model but one
MEMBERS = [
    "surrogate_MRI-CGCM3_01",
    "GFDL-ESM2G",
    "surrogate_GFDL-ESM2G_01",
    "inmcm4",
    "MRI-CGCM3",
]


def _table(tmp_path: Path, models=MODELS, weights=WEIGHTS) -> Path:
    path = tmp_path / "weights.tsv"
    pd.DataFrame({"model": models, "weight": weights}).to_csv(
        path, sep="\t", index=False
    )
    return path


def _draws(members: list[str], n_batch: int = 3) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "batch": [f"batch{b}" for b in range(n_batch) for _ in members],
            "gcm": members * n_batch,
            "value": [float(i) for i in range(n_batch * len(members))],
        }
    )


class TestRead:
    def test_rows_come_back_as_written(self, tmp_path):
        path = _table(tmp_path)
        got = read_model_weights(path)
        assert isinstance(got, ModelWeights)
        assert got.path == str(path)
        assert got.names == tuple(MODELS)
        assert got.keys == (
            "mri-cgcm3_01",
            "gfdl-esm2g",
            "gfdl-esm2g_01",
            "inmcm4",
            "access1-0",
            "mri-cgcm3",
        )
        assert got.weights == tuple(WEIGHTS)
        assert all(type(w) is float for w in got.weights)

    def test_a_table_laid_out_like_the_published_files(self, tmp_path):
        """The published tables have a leading column, year columns
        after the weight, and lines that end in a carriage return
        alone."""
        path = tmp_path / "weights.tsv"
        lines = ["quantile\tmodel\tweight\t1950\t1951"] + [
            f"{i / 10}\t{model}\t{weight!r}\t{i + 0.5}\t{i + 1.5}"
            for i, (model, weight) in enumerate(zip(MODELS, WEIGHTS))
        ]
        path.write_bytes("\r".join(lines).encode() + b"\r")
        got = read_model_weights(path)
        assert got.names == tuple(MODELS)
        assert got.weights == tuple(WEIGHTS)

    def test_the_hash_is_of_the_bytes_parsed(self, tmp_path, monkeypatch):
        """The file is rewritten as soon as it has been read once. Both
        the rows and the hash are those of the first bytes. Failed with
        the hash taken from a second read, and with the rows parsed from
        the path."""
        path = _table(tmp_path)
        original = path.read_bytes()
        read_bytes = Path.read_bytes

        def read_then_edit(self):
            raw = read_bytes(self)
            if self == path:
                _table(tmp_path, weights=WEIGHTS[::-1])
            return raw

        monkeypatch.setattr(Path, "read_bytes", read_then_edit)
        got = read_model_weights(path)
        monkeypatch.undo()

        assert path.read_bytes() != original
        assert got.sha256 == hashlib.sha256(original).hexdigest()
        assert got.weights == tuple(WEIGHTS)

    def test_two_rows_for_one_model_are_refused(self, tmp_path):
        """Both rows are named as written. Failed with the check taken
        out, which lets the last row win."""
        path = _table(
            tmp_path,
            ["access1-0*", "inmcm4", "ACCESS1-0", "Inmcm4*", "ccsm4"],
            [0.0, 0.2, 0.4, 0.3, 0.1],
        )
        with pytest.raises(ValueError) as error:
            read_model_weights(path)
        message = str(error.value)
        assert "['access1-0*', 'ACCESS1-0']" in message
        assert "['inmcm4', 'Inmcm4*']" in message
        assert "ccsm4" not in message

    def test_a_surrogate_suffix_keeps_two_names_apart(self, tmp_path):
        """The published tables hold both mri-cgcm3* and mri-cgcm3_01,
        and both gfdl-esm2g and gfdl-esm2g_01."""
        got = read_model_weights(_table(tmp_path))
        assert len(set(got.keys)) == len(MODELS)

    def test_a_table_without_the_two_columns_is_refused(self, tmp_path):
        path = tmp_path / "weights.tsv"
        pd.DataFrame({"model": ["a"], "w": [1.0]}).to_csv(path, sep="\t", index=False)
        with pytest.raises(ValueError, match="needs 'model' and 'weight'"):
            read_model_weights(path)

    def test_it_is_exported(self):
        assert cilreg.read_model_weights is read_model_weights
        assert cilreg.ModelWeights is ModelWeights


class TestMatching:
    def test_members_match_as_the_pipeline_matched_them(self, tmp_path):
        """The lookup the pipeline did before, written out here."""
        path = _table(tmp_path)
        table = pd.read_csv(path, sep="\t")
        keys = table["model"].str.replace("*", "", regex=False).str.lower()
        before = dict(zip(keys, table["weight"].astype(float)))

        got = read_model_weights(path)
        for member in MEMBERS + ["ACCESS1-0"]:
            expected = before[member.replace("surrogate_", "").lower()]
            assert got.weight_of(member) == expected
        assert got.missing(MEMBERS + ["ACCESS1-3", "surrogate_CCSM4_01"]) == [
            "ACCESS1-3",
            "surrogate_CCSM4_01",
        ]

    @pytest.mark.parametrize("dtype", ["object", "category"])
    def test_attach_gives_the_column_the_merge_gave(self, tmp_path, dtype):
        """The merge is the one the README shows. A model at weight zero
        gets 0.0 like any other."""
        path = _table(tmp_path)
        draws = _draws(MEMBERS + ["ACCESS1-0"])

        table = pd.read_csv(path, sep="\t")[["model", "weight"]]
        table["key"] = table["model"].str.replace("*", "", regex=False).str.lower()
        merged = draws.assign(
            key=draws["gcm"].str.replace("surrogate_", "", regex=False).str.lower()
        ).merge(table[["key", "weight"]], on="key", how="left")

        members = draws["gcm"].astype(dtype)
        members.index = range(100, 100 + len(members))
        got = read_model_weights(path).attach(members)
        assert got.dtype == "float64"
        assert got.index.equals(members.index)
        assert got.tolist() == merged["weight"].tolist()
        assert 0.0 in got.tolist()

    def test_attach_refuses_a_member_with_no_row(self, tmp_path):
        """Failed with the weight of such a member left as NaN."""
        weights = read_model_weights(_table(tmp_path))
        draws = _draws(MEMBERS + ["ACCESS1-3"])
        with pytest.raises(ValueError, match=r"1 members have no row.*ACCESS1-3"):
            weights.attach(draws["gcm"])

    def test_weight_of_refuses_a_member_with_no_row(self, tmp_path):
        weights = read_model_weights(_table(tmp_path))
        with pytest.raises(ValueError, match="'ACCESS1-3' has no row"):
            weights.weight_of("ACCESS1-3")


class TestRecord:
    def test_it_counts_leaves_and_lists_the_rows_left_over(self, tmp_path):
        """Failed with each member counted once, and with the rows left
        over listed without their weights."""
        path = _table(tmp_path)
        weights = read_model_weights(path)
        leaves = MEMBERS * 15 + ["inmcm4"]
        record = weights.record(leaves)
        assert record["path"] == str(path)
        assert record["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        assert record["members"] == {
            "GFDL-ESM2G": {"row": "gfdl-esm2g", "weight": 1 / 3, "n_leaves": 15},
            "MRI-CGCM3": {
                "row": "mri-cgcm3*",
                "weight": 0.3306666666666667,
                "n_leaves": 15,
            },
            "inmcm4": {"row": "inmcm4*", "weight": 0.2, "n_leaves": 16},
            "surrogate_GFDL-ESM2G_01": {
                "row": "gfdl-esm2g_01",
                "weight": 0.036,
                "n_leaves": 15,
            },
            "surrogate_MRI-CGCM3_01": {
                "row": "mri-cgcm3_01",
                "weight": 0.1,
                "n_leaves": 15,
            },
        }
        assert record["unmatched_rows"] == {"access1-0*": 0.0}

    def test_the_record_gives_the_table_back(self, tmp_path):
        """The weights read back from JSON are the floats that were
        used, to the last bit."""
        path = _table(tmp_path)
        weights = read_model_weights(path)
        record = json.loads(json.dumps(weights.record(MEMBERS), sort_keys=True))
        rebuilt = {
            entry["row"]: entry["weight"] for entry in record["members"].values()
        }
        rebuilt.update(record["unmatched_rows"])
        assert {name: w.hex() for name, w in rebuilt.items()} == {
            name: w.hex() for name, w in zip(MODELS, WEIGHTS)
        }

    def test_a_member_with_no_row_is_refused(self, tmp_path):
        weights = read_model_weights(_table(tmp_path))
        with pytest.raises(ValueError, match=r"1 members have no row.*ACCESS1-3"):
            weights.record(MEMBERS + ["ACCESS1-3"])

    def test_the_rows_survive_the_trip_to_a_worker(self, tmp_path):
        weights = read_model_weights(_table(tmp_path))
        assert pickle.loads(pickle.dumps(weights)) == weights
