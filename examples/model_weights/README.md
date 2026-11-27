# Declared rows of the published SMME tables

The two pair files here are a bridge, not part of the library. The
library matches a member of a projection tree to a row of a model
weights table by its exact name, or by the row declared for it, and
refuses a member with neither. These files declare the rows because
the published SMME tables (Zenodo record 22003542) spell the models
differently from the trees. They go away when those tables are
rewritten with the trees' names. The pairs that real runs use belong
beside the tables, in the recipes repository, not in this repository's
examples.

## Why the published tables need them

The published SMME tables do not pass as they stand. They name the
models in lower case, some with a trailing `*`, and the surrogates
without the `surrogate_` prefix of the trees, so `ACCESS1-0` is the row
`access1-0*` and `surrogate_GFDL-CM3_89` the row `gfdl-cm3_89`. Of the
32 members under rcp45 and the 33 under rcp85, only `bcc-csm1-1` is
spelt the same on both sides. Until the tables are rewritten, a run
that reads them declares the row of each member.

## The files

Each holds the `[stats.model_weight_names]` block of one rcp, member
name to table name, for every member that needs one.

- `smme_rcp45_names.toml`, 31 pairs for `rcp45_SMME_weights.tsv`.
- `smme_rcp85_names.toml`, 32 pairs for `rcp85_SMME_weights.tsv`.

A pipeline config takes the block as it is. A script passes it to
`read_model_weights` or `fetch_model_weights` as `table_names`, read
with `load_model_weight_names`:

```python
from cil_regionalization.pipelines.montecarlo import load_model_weight_names

names = load_model_weight_names("examples/model_weights/smme_rcp85_names.toml")
weights = cilreg.fetch_model_weights("rcp85", table_names=names)
```

The members are those the configs under `examples/montecarlo/runs`
declare for each rcp. Each was paired once with the row the library
used to match it to, by dropping `surrogate_` and `*` and lowercasing
both sides. The pairs are written out so that nothing reads a name to
decide what it means. Every member has one row, and every row of the
two tables is the row of one member.
