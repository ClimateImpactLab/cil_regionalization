# Warming levels of the CMIP5 models

`ar5_cmip5_smme.toml` is a `[stats.warming_levels]` block for the CMIP5
models of the temperature table under `data/external/`.
`tests/integration/test_warming_levels_paper.py` reads the same file and
checks that it puts the Inequality Paper's models at each level, so the
settings and the check cannot differ.

```python
from cil_regionalization.pipelines.montecarlo import load_warming_levels

warming = load_warming_levels("examples/warming_levels/ar5_cmip5_smme.toml")
table = warming.read_table()
levels = warming.assign(table, declared=table.members)
```

A pipeline config takes the block as it is, with `gmst` written relative
to that config. `join` lists models, so each of them has to be in the
tree under at least one scenario, and a run over fewer models needs a
list of its own.

## Where each number comes from

`finding_gwl_bins.R` in the inequality repository, branch `elliot_wip`
at `4dd6b9c`, lines 1 to 90, is the code the paper's levels came from.

- **`reference_period` and `offset`.** Each member's series is rebased
  to its own mean over 1986 to 2005, and 0.61 C is added for the warming
  from 1850-1900 to that period. AR5 WG1, Technical Summary TS.2.2.1,
  gives 0.61 [0.55 to 0.67] C, from HadCRUT4 and its 5 to 95%
  uncertainty, and `finding_gwl_bins.R` cites it. The offset is right
  for that reference period only. The range is not used. With 0.55, 12
  of the 42 members change level in a period, and with 0.67, 14 do.
- **`levels` and `half_width`.** Levels 1.5, 2, 3, 4 and 5, each holding
  a mean from 0.25 below it to 0.25 above it with both ends included, as
  `finding_gwl_bins.R` bins them. A mean of 1.75 goes to 1.5, the first
  match of its `case_when`, which is the library's rule for an end two
  bands share.
- **`periods`.** 2040 to 2059 and 2080 to 2099, twenty years each with
  both ends included, as the year filter in `finding_gwl_bins.R` takes
  them. The impacts are taken over the same years. With nineteen years
  of temperature, three members move: MIROC-ESM rcp85 leaves 3 mid
  century, MPI-ESM-LR rcp85 joins 2 mid century, and NorESM1-M rcp85
  leaves 4 at the end of the century.
- **`join`.** The 21 CMIP5 models of the table, under both scenarios.
  The surrogate members are left out. The surrogate mixed model
  ensemble is a weighted construction, and warming levels use no
  weights, so counting a constructed member once beside a CMIP5 run is
  not what that ensemble was built for. The paper's lists hold no
  surrogate. `finding_gwl_bins.R` itself does not filter them, since
  that line is commented out.
- **`gmst`, `member` and `smoothing`.** The table is Carleton et al.
  (2022), and `data/external/README.md` says where it comes from. It
  arrived smoothed, and how is recorded nowhere.
