"""Model weights are read once, matched by the exact name or a declared
row, refused when a member has none, and recorded.

The tables are synthetic. Every test here fails without the module,
because the import fails. The tests that name a mutation in their
docstring were also seen to fail under it.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import re
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
# spelt as the trees spell them, which the rows above are not, and the
# tree has no leaf of the last model but one
MEMBERS = [
    "surrogate_MRI-CGCM3_01",
    "GFDL-ESM2G",
    "surrogate_GFDL-ESM2G_01",
    "inmcm4",
    "MRI-CGCM3",
]
# the row of each member, written out, and of a model the tree lacks
NAMES = {
    "surrogate_MRI-CGCM3_01": "mri-cgcm3_01",
    "GFDL-ESM2G": "gfdl-esm2g",
    "surrogate_GFDL-ESM2G_01": "gfdl-esm2g_01",
    "inmcm4": "inmcm4*",
    "MRI-CGCM3": "mri-cgcm3*",
    "ACCESS1-0": "access1-0*",
}


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
        got = read_model_weights(path, table_names=NAMES)
        assert isinstance(got, ModelWeights)
        assert got.path == str(path)
        assert got.names == tuple(MODELS)
        assert got.weights == tuple(WEIGHTS)
        assert all(type(w) is float for w in got.weights)
        assert got.table_names == NAMES
        assert read_model_weights(path).table_names == {}

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

    def test_another_separator(self, tmp_path):
        """As the copies of the examples keep the two columns."""
        path = tmp_path / "weights.csv"
        pd.DataFrame({"model": MODELS, "weight": WEIGHTS}).to_csv(path, index=False)
        assert read_model_weights(path, sep=",").weights == tuple(WEIGHTS)

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

    def test_a_name_in_the_table_twice_is_refused(self, tmp_path):
        """Failed with the check taken out, which lets the last row
        win."""
        path = _table(
            tmp_path,
            ["access1-0*", "inmcm4", "access1-0*", "ccsm4", "ccsm4"],
            [0.0, 0.2, 0.4, 0.3, 0.1],
        )
        with pytest.raises(
            ValueError,
            match=re.escape(
                "has the rows 'access1-0*' and 'ccsm4' more than once. Each "
                "model needs exactly one row."
            ),
        ):
            read_model_weights(path)

    def test_names_that_differ_by_a_letter_are_two_rows(self, tmp_path):
        """access1-0* and ACCESS1-0 were one model to the rule that
        lowercased both and dropped the star, which refused the table.
        It went red on that rule."""
        path = _table(tmp_path, ["access1-0*", "ACCESS1-0"], [0.25, 0.75])
        got = read_model_weights(path)
        assert got.weight_of("ACCESS1-0") == 0.75
        assert got.weight_of("access1-0*") == 0.25

    def test_two_members_declared_to_one_row_are_refused(self, tmp_path):
        with pytest.raises(
            ValueError,
            match=re.escape(
                "table_names declares one row of the model weights file "
                f"{tmp_path / 'weights.tsv'} for more than one member, the row "
                "'inmcm4*' for 'inmcm4' and 'INMCM4'. Each member needs a row "
                "of its own."
            ),
        ):
            read_model_weights(
                _table(tmp_path), table_names={"inmcm4": "inmcm4*", "INMCM4": "inmcm4*"}
            )

    @pytest.mark.parametrize("row", ["", 3])
    def test_a_declared_row_that_is_not_a_name_is_refused(self, tmp_path, row):
        with pytest.raises(ValueError, match="both strings"):
            read_model_weights(_table(tmp_path), table_names={"inmcm4": row})

    def test_a_table_without_the_two_columns_is_refused(self, tmp_path):
        path = tmp_path / "weights.tsv"
        pd.DataFrame({"model": ["a"], "w": [1.0]}).to_csv(path, sep="\t", index=False)
        with pytest.raises(ValueError, match="needs 'model' and 'weight'"):
            read_model_weights(path)

    def test_it_is_exported(self):
        assert cilreg.read_model_weights is read_model_weights
        assert cilreg.ModelWeights is ModelWeights


class TestMatching:
    def test_a_member_takes_the_row_of_its_own_name_and_no_other(self, tmp_path):
        """Not lowercased, not without its prefix, not without the
        table's star. It went red on the rule that did all three."""
        got = read_model_weights(_table(tmp_path))
        assert got.weight_of("inmcm4*") == 0.2
        assert got.missing(MEMBERS + ["ACCESS1-0", "inmcm4*"]) == sorted(
            MEMBERS + ["ACCESS1-0"]
        )

    def test_a_declared_row_is_the_member_s_row(self, tmp_path):
        got = read_model_weights(_table(tmp_path), table_names=NAMES)
        weight = dict(zip(MODELS, WEIGHTS))
        for member, row in NAMES.items():
            assert got.weight_of(member) == weight[row]
        assert got.missing(MEMBERS + ["ACCESS1-3"]) == ["ACCESS1-3"]

    def test_the_refusal_names_both_sides_and_the_ways_out(self, tmp_path):
        """The member as the tree spells it, the row it comes closest
        to, what to change, and why the published tables fail."""
        weights = read_model_weights(_table(tmp_path))
        with pytest.raises(ValueError) as error:
            weights.attach(_draws(["ACCESS1-0", "surrogate_GFDL-ESM2G_01"])["gcm"])
        assert str(error.value) == (
            f"2 members have no row in the model weights file "
            f"{tmp_path / 'weights.tsv'}: 'ACCESS1-0' (closest row "
            f"'access1-0*') and 'surrogate_GFDL-ESM2G_01' (closest row "
            f"'gfdl-esm2g_01'). A member takes the row of its own name, letter "
            f"for letter, or the row declared for it. Write the members' "
            f"names into the table, or declare each member's row in "
            f"stats.model_weight_names, or in table_names of "
            f'read_model_weights, as in "ACCESS1-0" = "access1-0*".'
        )

    def test_a_declared_row_the_table_lacks_is_named_with_its_closest(self, tmp_path):
        weights = read_model_weights(
            _table(tmp_path), table_names={"ACCESS1-0": "acess1-0*"}
        )
        with pytest.raises(
            ValueError,
            match=re.escape(
                "'ACCESS1-0' declared as 'acess1-0*' (closest row 'access1-0*')"
            ),
        ):
            weights.weight_of("ACCESS1-0")

    def test_past_five_the_refusal_counts_the_rest(self, tmp_path):
        weights = read_model_weights(_table(tmp_path))
        with pytest.raises(ValueError, match=r"\(closest row '[^']+'\) and 2 more\."):
            weights.check(MEMBERS + ["ACCESS1-0", "ACCESS1-3"])

    def test_two_members_on_one_row_are_refused(self, tmp_path):
        """One by its own name and one by a declared row. Failed with
        both given the row's weight."""
        weights = read_model_weights(
            _table(tmp_path), table_names={"INMCM4": "inmcm4*"}
        )
        with pytest.raises(
            ValueError,
            match=re.escape(
                "would give one row to more than one member, the row "
                "'inmcm4*' to 'inmcm4*' and 'INMCM4'. Each member needs a row "
                "of its own."
            ),
        ):
            weights.attach(_draws(["inmcm4*", "INMCM4"])["gcm"])

    @pytest.mark.parametrize("dtype", ["object", "category"])
    def test_attach_gives_each_draw_the_weight_of_its_row(self, tmp_path, dtype):
        """A model at weight zero gets 0.0 like any other."""
        weight = dict(zip(MODELS, WEIGHTS))
        draws = _draws(MEMBERS + ["ACCESS1-0"])
        members = draws["gcm"].astype(dtype)
        members.index = range(100, 100 + len(members))
        got = read_model_weights(_table(tmp_path), table_names=NAMES).attach(members)
        assert got.dtype == "float64"
        assert got.index.equals(members.index)
        assert got.tolist() == [weight[NAMES[m]] for m in draws["gcm"]]
        assert 0.0 in got.tolist()

    def test_attach_refuses_a_member_with_no_row(self, tmp_path):
        """Failed with the weight of such a member left as NaN."""
        weights = read_model_weights(_table(tmp_path), table_names=NAMES)
        draws = _draws(MEMBERS + ["ACCESS1-3"])
        with pytest.raises(ValueError, match=r"1 member has no row.*'ACCESS1-3'"):
            weights.attach(draws["gcm"])

    def test_attach_refuses_a_draw_with_no_member(self, tmp_path):
        weights = read_model_weights(_table(tmp_path), table_names=NAMES)
        with pytest.raises(ValueError, match="1 row names no member"):
            weights.attach(pd.Series(["inmcm4", None]))

    def test_weight_of_refuses_a_member_with_no_row(self, tmp_path):
        weights = read_model_weights(_table(tmp_path), table_names=NAMES)
        with pytest.raises(ValueError, match="1 member has no row.*'ACCESS1-3'"):
            weights.weight_of("ACCESS1-3")


class TestRecord:
    def test_it_counts_leaves_and_lists_the_rows_left_over(self, tmp_path):
        """Failed with each member counted once, and with the rows left
        over listed without their weights."""
        path = _table(tmp_path)
        weights = read_model_weights(path, table_names=NAMES)
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

    def test_it_lists_the_rows_declared_and_those_no_member_used(self, tmp_path):
        record = read_model_weights(_table(tmp_path), table_names=NAMES).record(MEMBERS)
        assert record["table_names"] == NAMES
        assert record["table_names_unused"] == {"ACCESS1-0": "access1-0*"}

    def test_the_record_gives_the_table_back(self, tmp_path):
        """The weights read back from JSON are the floats that were
        used, to the last bit."""
        weights = read_model_weights(_table(tmp_path), table_names=NAMES)
        record = json.loads(json.dumps(weights.record(MEMBERS), sort_keys=True))
        rebuilt = {
            entry["row"]: entry["weight"] for entry in record["members"].values()
        }
        rebuilt.update(record["unmatched_rows"])
        assert {name: w.hex() for name, w in rebuilt.items()} == {
            name: w.hex() for name, w in zip(MODELS, WEIGHTS)
        }

    def test_a_member_with_no_row_is_refused(self, tmp_path):
        weights = read_model_weights(_table(tmp_path), table_names=NAMES)
        with pytest.raises(ValueError, match=r"1 member has no row.*'ACCESS1-3'"):
            weights.record(MEMBERS + ["ACCESS1-3"])

    def test_the_rows_survive_the_trip_to_a_worker(self, tmp_path):
        weights = read_model_weights(_table(tmp_path), table_names=NAMES)
        assert pickle.loads(pickle.dumps(weights)) == weights
