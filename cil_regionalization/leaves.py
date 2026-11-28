"""The aggregated leaf on disk.

A leaf file holds the frame `apply.apply_weights` returns for one leaf
of a Monte Carlo tree. Its rows are sorted by target and cut into row
groups of a fixed number of targets, the same targets in the same group
of every leaf, so the rows of a range of targets can be read from every
leaf without reading the rest.

The file is plain parquet with a fixed Arrow schema and no pandas
metadata. Its footer says which aggregation produced it, under the key
``cil_regionalization.leaf``.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# 2: the target ids keep their type, a string or an integer of its width,
# where version 1 took strings alone
FORMAT_VERSION = 2
TARGETS_PER_GROUP = 128
ORDINAL_COLUMN = "target_ordinal"
LEAF_KEY = b"cil_regionalization.leaf"
TARGETS_KEY = b"cil_regionalization.targets"
# a numeric dimension with more distinct values than this gets no profile
PROFILE_LIMIT = 4096


def canonical_digest(document) -> str:
    """SHA256 of ``document`` written as JSON with sorted keys and no
    spaces."""
    text = json.dumps(document, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_string(column: pd.Series) -> bool:
    if isinstance(column.dtype, pd.StringDtype):
        return True
    return column.dtype == object and pd.api.types.infer_dtype(column, skipna=True) in (
        "string",
        "empty",
    )


def id_type(column: pd.Series, field: str) -> str:
    """The canonical name of an id field's type, "string" for text in any
    storage and the numpy name of an integer, such as "int64". Raises on
    any other type."""
    # A string column's storage is not part of the id. Arrow keeps the
    # width of a string's offsets, 32 bits under string and 64 under
    # large_string, and two copies of the same targets can come with
    # either (ir_adm1_full has large_string and ir_adm2_croparea_corn
    # string), as pandas can hold either as object or as a string dtype.
    # Were the storage hashed, the digest of identical ids would differ
    # and a leaf written from one copy would be refused against the
    # other. An integer's width is kept, because an int32 and an int64 id
    # are different identities once the values outgrow int32.
    dtype = column.dtype
    if isinstance(dtype, np.dtype) and dtype.kind in "iu":
        return dtype.name
    if _is_string(column):
        return "string"
    raise ValueError(
        f"id field {field!r} has dtype {dtype}. A leaf file takes ids that "
        f"are strings or integers."
    )


def _arrow_id_type(name: str) -> pa.DataType:
    return pa.string() if name == "string" else pa.from_numpy_dtype(np.dtype(name))


def _canonical_arrow(arrow: pa.DataType, field: str) -> str:
    """`id_type` of an id column read back from a file."""
    if pa.types.is_string(arrow) or pa.types.is_large_string(arrow):
        return "string"
    if pa.types.is_integer(arrow):
        return np.dtype(arrow.to_pandas_dtype()).name
    raise ValueError(f"id field {field!r} is stored as {arrow}")


def _is_nullable_string(column: pd.Series) -> bool:
    return (
        isinstance(column.dtype, pd.StringDtype)
        and getattr(column.dtype, "na_value", pd.NA) is pd.NA
    )


def _replace(tmp: Path, path: Path) -> None:
    os.replace(tmp, path)


def _temporary(path: Path) -> Path:
    # unique to the process and the call, so two writers never share one
    return path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")


def write_json(path: Path, document) -> None:
    """Write ``document`` under a temporary name, then rename it."""
    tmp = _temporary(path)
    try:
        tmp.write_text(json.dumps(document, indent=2, sort_keys=True))
        _replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


@dataclass(frozen=True)
class TargetTable:
    """The targets of a weights artifact in the order leaf files use.

    A target's ordinal is its position among the distinct id tuples
    sorted ascending, integers by value and strings by code point. It
    does not depend on the order of the artifact's rows. ``id_types``
    are the fields' types as `id_type` names them, and ``digest`` is a
    SHA256 over the field names, their types and the ids, not over any
    file, so an integer 1 and a string "1" are different targets.
    """

    id_fields: tuple[str, ...]
    id_types: tuple[str, ...]
    ids: tuple[tuple, ...]
    digest: str

    @classmethod
    def from_frame(cls, frame: pd.DataFrame, id_fields) -> "TargetTable":
        """The table of the targets in ``frame``, each id kept in the type
        of its column. Raises if an id field is neither strings nor
        integers, or has a missing value."""
        id_fields = tuple(id_fields)
        id_types = tuple(id_type(frame[field], field) for field in id_fields)
        for field in id_fields:
            if frame[field].isna().any():
                raise ValueError(
                    f"id field {field!r} has missing values. A target "
                    f"with no id cannot be placed in a leaf file."
                )
        distinct = frame[list(id_fields)].drop_duplicates()
        # Python ints and strs, so a field sorts by value and hashes alike
        # whatever width or storage its column had
        ids = tuple(
            sorted(
                zip(
                    *(
                        [int(v) for v in distinct[f].tolist()]
                        if kind != "string"
                        else [str(v) for v in distinct[f].tolist()]
                        for f, kind in zip(id_fields, id_types)
                    )
                )
            )
        )
        if not ids:
            raise ValueError("the weights artifact has no targets")
        return cls(id_fields, id_types, ids, cls._digest(id_fields, id_types, ids))

    @staticmethod
    def _digest(id_fields, id_types, ids) -> str:
        digest = hashlib.sha256(b"cil_regionalization targets 2\n")

        def put(text: str) -> None:
            raw = text.encode("utf-8")
            digest.update(len(raw).to_bytes(8, "big"))
            digest.update(raw)

        digest.update(len(id_fields).to_bytes(8, "big"))
        for field, kind in zip(id_fields, id_types):
            put(field)
            put(kind)
        digest.update(len(ids).to_bytes(8, "big"))
        for row in ids:
            for value, kind in zip(row, id_types):
                put(value if kind == "string" else str(int(value)))
        return digest.hexdigest()

    def __len__(self) -> int:
        return len(self.ids)

    @property
    def filename(self) -> str:
        return f"targets.{self.digest[:16]}.parquet"

    def n_groups(self, targets_per_group: int) -> int:
        return -(-len(self.ids) // targets_per_group)

    def ordinals(self, frame: pd.DataFrame) -> np.ndarray:
        """The ordinal of each row of ``frame``, as int32. Raises if a
        row names a target the table does not have."""
        keys = frame[list(self.id_fields)]
        if len(keys) == 0:
            return np.zeros(0, dtype=np.int32)
        codes = (
            keys.groupby(list(self.id_fields), sort=False, dropna=False)
            .ngroup()
            .to_numpy()
        )
        # one lookup per distinct target, in the order the codes number them
        distinct = keys.drop_duplicates()
        position = {row: i for i, row in enumerate(self.ids)}
        of_code = np.array(
            [
                position.get(row, -1)
                for row in zip(
                    *(distinct[f].astype(object).tolist() for f in self.id_fields)
                )
            ],
            dtype=np.int32,
        )
        if (of_code < 0).any():
            unknown = distinct[of_code < 0].head(5).to_dict("records")
            raise ValueError(
                f"{int((of_code < 0).sum())} targets are not in the target "
                f"table (first: {unknown})"
            )
        return of_code[codes]

    def write(self, leaves_dir: Path) -> Path:
        """Write the table as ``targets.<digest>.parquet`` unless a file
        of that name is already there, and return its path. Raises if
        the file there holds other targets."""
        path = Path(leaves_dir) / self.filename
        if path.exists():
            found = TargetTable.read(path)
            if found != self:
                raise ValueError(f"{path} does not hold the targets its name says")
            return path
        schema = pa.schema(
            [(ORDINAL_COLUMN, pa.int32())]
            + [
                (field, _arrow_id_type(kind))
                for field, kind in zip(self.id_fields, self.id_types)
            ],
            metadata={
                TARGETS_KEY: json.dumps(
                    {
                        "format_version": FORMAT_VERSION,
                        "digest": self.digest,
                        "id_fields": list(self.id_fields),
                        "id_types": list(self.id_types),
                        "n_targets": len(self.ids),
                    },
                    sort_keys=True,
                ).encode("utf-8")
            },
        )
        table = pa.table(
            [pa.array(np.arange(len(self.ids), dtype=np.int32))]
            + [
                pa.array([row[i] for row in self.ids], type=_arrow_id_type(kind))
                for i, kind in enumerate(self.id_types)
            ],
            schema=schema,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = _temporary(path)
        try:
            pq.write_table(table, tmp, compression="snappy")
            _replace(tmp, path)
        finally:
            tmp.unlink(missing_ok=True)
        return path

    @classmethod
    def read(cls, path: str | Path) -> "TargetTable":
        """Read a table back. The digest is worked out again from the
        ids, and a file whose footer says another one is refused."""
        table = pq.read_table(path)
        metadata = table.schema.metadata or {}
        if TARGETS_KEY not in metadata:
            raise ValueError(f"{path} is not a target table")
        footer = json.loads(metadata[TARGETS_KEY])
        if footer.get("format_version") != FORMAT_VERSION:
            raise ValueError(
                f"{path} is a target table of format version "
                f"{footer.get('format_version')}, and this version of the library "
                f"reads version {FORMAT_VERSION}. Write the leaves again."
            )
        id_fields = tuple(footer["id_fields"])
        id_types = tuple(
            _canonical_arrow(table.schema.field(f).type, f) for f in id_fields
        )
        if list(id_types) != footer["id_types"]:
            raise ValueError(
                f"{path} stores its ids as {list(id_types)}, and its footer "
                f"says {footer['id_types']}"
            )
        ordinals = table.column(ORDINAL_COLUMN).to_pylist()
        if ordinals != list(range(len(ordinals))):
            raise ValueError(f"{path} is not in ordinal order")
        ids = tuple(zip(*(table.column(f).to_pylist() for f in id_fields)))
        found = cls(id_fields, id_types, ids, cls._digest(id_fields, id_types, ids))
        if found.digest != footer["digest"]:
            raise ValueError(
                f"{path} holds targets with digest {found.digest}, and its "
                f"footer says {footer['digest']}"
            )
        return found


def _string_array(column: pd.Series) -> pa.Array:
    # one Python string per distinct value, not per row
    codes, distinct = pd.factorize(column)
    values = pa.array(distinct.astype(object), type=pa.string(), from_pandas=True)
    return values.take(pa.array(codes, mask=codes < 0))


def _dimension_array(name: str, column: pd.Series) -> pa.Array:
    """A leaf's own dimension as an Arrow array. Integers, floats and
    booleans keep their width, and plain strings become Arrow strings.
    Raises on any other dtype, such as a categorical, datetime or
    nullable one."""
    # a categorical, datetime or nullable column is refused because
    # pandas 2 and pandas 3 write it differently
    dtype = column.dtype
    if isinstance(dtype, np.dtype) and dtype.kind in "iufb":
        return pa.array(column.to_numpy())
    if _is_string(column) and not _is_nullable_string(column):
        return _string_array(column)
    raise ValueError(
        f"leaf column {name!r} has dtype {dtype}. A leaf file takes "
        f"integer, float, boolean and plain string dimensions. A "
        f"categorical, datetime or nullable column is refused, because "
        f"pandas 2 and pandas 3 write it differently. Convert the column "
        f"in the tree, or set output.write_leaves = false."
    )


def _profile(frame: pd.DataFrame, dimensions: list[str]) -> dict:
    """The rows at each value of each numeric dimension. A dimension
    with more than `PROFILE_LIMIT` distinct values is recorded as None."""
    out = {}
    for name in dimensions:
        column = frame[name]
        if not (isinstance(column.dtype, np.dtype) and column.dtype.kind in "iuf"):
            continue
        counts = column.value_counts(sort=False).sort_index()
        if len(counts) > PROFILE_LIMIT:
            out[name] = None
            continue
        out[name] = {
            "values": counts.index.to_numpy().tolist(),
            "rows": counts.to_numpy().tolist(),
        }
    return out


@dataclass(frozen=True)
class WrittenLeaf:
    """What `write_leaf` wrote, for the leaves manifest."""

    rows_per_group: list[int]
    size: int
    sha256: str
    schema: list[dict]
    profile: dict

    def as_dict(self) -> dict:
        return {
            "rows_per_group": self.rows_per_group,
            "size": self.size,
            "sha256": self.sha256,
            "schema": self.schema,
            "profile": self.profile,
        }


def write_leaf(
    frame: pd.DataFrame,
    path: str | Path,
    *,
    targets: TargetTable,
    value_col: str,
    identity: dict,
    targets_per_group: int = TARGETS_PER_GROUP,
) -> WrittenLeaf:
    """Write one aggregated leaf and return what was written, for the
    leaves manifest.

    ``frame`` is what `apply_weights` returned. The file has an int32
    `ORDINAL_COLUMN` first, then the columns of ``frame`` in their
    order. Row group g holds ordinals ``g * targets_per_group`` up to
    the next multiple and is empty where the leaf has none of them.
    ``identity`` goes into the footer with the layout. The file is
    written under a temporary name and renamed, so a file under ``path``
    is always whole. Raises if the value column is not float64, or a
    row names a target the table does not have.
    """
    path = Path(path)
    if ORDINAL_COLUMN in frame.columns:
        raise ValueError(
            f"the leaf already has a column named {ORDINAL_COLUMN!r}, which "
            f"a leaf file uses for the target's ordinal"
        )
    if frame[value_col].dtype != np.float64:
        raise ValueError(
            f"value column {value_col!r} is {frame[value_col].dtype}; an "
            f"aggregated leaf holds float64"
        )
    ordinals = targets.ordinals(frame)
    if len(ordinals) > 1 and (ordinals[1:] < ordinals[:-1]).any():
        # stable, so the rows of one target keep their order
        order = np.argsort(ordinals, kind="stable")
        frame, ordinals = frame.iloc[order], ordinals[order]

    dimensions = [
        c for c in frame.columns if c not in targets.id_fields and c != value_col
    ]
    arrays = [pa.array(ordinals, type=pa.int32())]
    pandas_dtypes = ["int32"]
    taken = pa.array(ordinals)
    for name in frame.columns:
        column = frame[name]
        if name in targets.id_fields:
            # the ordinal already says which id a row has
            field = targets.id_fields.index(name)
            array = pa.array(
                [row[field] for row in targets.ids],
                type=_arrow_id_type(targets.id_types[field]),
            ).take(taken)
        elif name == value_col:
            array = pa.array(column.to_numpy(), type=pa.float64())
        else:
            array = _dimension_array(name, column)
        arrays.append(array)
        pandas_dtypes.append(str(column.dtype))
    names = [ORDINAL_COLUMN] + list(frame.columns)
    footer = {
        **identity,
        "format_version": FORMAT_VERSION,
        "targets_per_group": targets_per_group,
        "n_targets": len(targets),
        "target_digest": targets.digest,
    }
    schema = pa.schema(
        [(name, array.type) for name, array in zip(names, arrays)],
        metadata={LEAF_KEY: json.dumps(footer, sort_keys=True).encode("utf-8")},
    )
    table = pa.table(arrays, schema=schema)

    n_groups = targets.n_groups(targets_per_group)
    bounds = np.searchsorted(
        ordinals, np.arange(n_groups + 1, dtype=np.int64) * targets_per_group
    )
    rows = np.diff(bounds).tolist()

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _temporary(path)
    try:
        with pq.ParquetWriter(
            tmp,
            schema,
            compression="snappy",
            # the value is stored plain; a dictionary of floats that are
            # nearly all distinct only adds to the file
            use_dictionary=[name for name in names if name != value_col],
        ) as writer:
            for start, n in zip(bounds[:-1].tolist(), rows):
                # Its own length as the row group size. Left to the
                # default, a group above 1,048,576 rows is split in two
                # and every later group index shifts.
                writer.write_table(table.slice(start, n), row_group_size=max(n, 1))
        metadata = pq.read_metadata(tmp)
        written = [
            metadata.row_group(g).num_rows for g in range(metadata.num_row_groups)
        ]
        if written != rows:
            raise RuntimeError(
                f"leaf {path}: {len(written)} row groups were written where "
                f"{n_groups} were laid out"
            )
        size, sha256 = tmp.stat().st_size, file_sha256(tmp)
        _replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)

    return WrittenLeaf(
        rows_per_group=rows,
        size=size,
        sha256=sha256,
        schema=[
            {"name": name, "arrow": str(array.type), "pandas": dtype}
            for name, array, dtype in zip(names, arrays, pandas_dtypes)
        ],
        profile=_profile(frame, dimensions),
    )


def read_leaf_footer(path: str | Path) -> dict:
    """The footer `write_leaf` put in a leaf file. Raises if the file
    has none, which is how a file of the old layout is told apart."""
    metadata = pq.read_schema(path).metadata or {}
    if LEAF_KEY not in metadata:
        raise ValueError(
            f"{path} has no {LEAF_KEY.decode()} footer, so it is not a leaf "
            f"file of this layout"
        )
    return json.loads(metadata[LEAF_KEY])


# what the footer of a leaf has to share with the leaves manifest
_FOOTER_CHECKED = (
    "format_version",
    "targets_per_group",
    "n_targets",
    "target_digest",
    "aggregation_digest",
)


def read_groups(
    path: str | Path,
    first: int,
    stop: int,
    *,
    expect: dict,
    columns: list[str] | None = None,
    attempts: int = 3,
    pause: float = 0.5,
) -> tuple[pa.Table, int]:
    """Row groups ``first`` up to ``stop`` of a leaf file, and the number
    of opens that failed before the one that worked.

    The file is checked against ``expect`` before anything is read. The
    footer has to agree with it on the layout and the two digests, the
    file has to have the groups and the rows it says, and the ordinals of
    each group read have to lie in that group's range. An `OSError` is
    tried again, up to ``attempts`` opens in all. A file that is wrong is
    not tried again.
    """
    failed = 0
    while True:
        try:
            return _read_groups(Path(path), first, stop, expect, columns), failed
        except OSError:
            failed += 1
            if failed >= attempts:
                raise
            time.sleep(pause * failed)


def _read_groups(path, first, stop, expect, columns) -> pa.Table:
    file = pq.ParquetFile(path)
    metadata = file.metadata
    raw = (file.schema_arrow.metadata or {}).get(LEAF_KEY)
    if raw is None:
        raise ValueError(
            f"{path} has no {LEAF_KEY.decode()} footer, so it is not a leaf "
            f"file of this layout"
        )
    footer = json.loads(raw)
    for key in _FOOTER_CHECKED:
        if footer.get(key) != expect[key]:
            raise ValueError(
                f"leaf {path}: its footer has {key} = {footer.get(key)!r} "
                f"and the leaves manifest has {expect[key]!r}"
            )
    per_group = expect["targets_per_group"]
    n_groups = -(-expect["n_targets"] // per_group)
    if metadata.num_row_groups != n_groups:
        raise ValueError(
            f"leaf {path} has {metadata.num_row_groups} row groups where "
            f"its layout has {n_groups}"
        )
    if metadata.num_rows != expect["rows"]:
        raise ValueError(
            f"leaf {path} has {metadata.num_rows} rows and the leaves "
            f"manifest has {expect['rows']}"
        )
    ordinal = metadata.schema.names.index(ORDINAL_COLUMN)
    for g in range(first, stop):
        group = metadata.row_group(g)
        statistics = group.column(ordinal).statistics
        if group.num_rows == 0 or statistics is None or not statistics.has_min_max:
            continue
        if statistics.min < g * per_group or statistics.max >= (g + 1) * per_group:
            raise ValueError(
                f"leaf {path}: row group {g} holds ordinals {statistics.min} "
                f"to {statistics.max}, outside its range"
            )
    return file.read_row_groups(list(range(first, stop)), columns=columns)
