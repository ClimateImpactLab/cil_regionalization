# Impact Map applications

The record of how the published Impact Map aggregations were produced: four
damage sectors in physical units, valued and as a percent of GDP, at ADM0,
ADM1 and ADM2 and at impact region level, plus the climate ensemble.

These configs and job scripts only run with access to the University of
Chicago RCC cluster: the input trees, the weight files and the scheduler
account they name live there. They are a record, not something to run
elsewhere. For a runnable example, see `examples/aggregation` and
`examples/energy`.

Each TOML drives one `cilreg pipeline` run (one damages tree, one target
level, one RCP); the matching job file submits it. Paths inside them name
cluster locations current at the time of the runs and need updating if the
inputs move.

## What produced what

| Product | Script | Config | Job |
|---|---|---|---|
| Agriculture log yield, ADM1 and ADM2 | `cilreg pipeline` (`pipelines/montecarlo`) | `agriculture/configs/ag_{crop}_{adm}_{rcp}.toml` | `agriculture/jobs/ag_{crop}_{adm}_{rcp}.sbatch` |
| Agriculture log yield, impact region | same | `agriculture/configs/ag_{crop}_ir_{rcp}.toml` | `agriculture/jobs/ag_{crop}_ir_{rcp}.sbatch` |
| Agriculture log yield, ADM0 | `agriculture/make_adm0_agriculture_physical.py` | none | `agriculture/jobs/ag_phys_adm0_{rcp}.sbatch` |
| Agriculture baseline windows | both of the above | `agriculture/configs/*_base*.toml` | `agriculture/jobs/*_base*.sbatch` |
| Agriculture percent change, all levels | `agriculture/run_percent_from_logs.py` | none | `agriculture/jobs/ag_pct_from_logs_{rcp}.sbatch` |
| Agriculture valued | `agriculture/run_valued_agriculture.py` | none | `agriculture/jobs/ag_val_*.sbatch` |
| Agriculture calories | `agriculture/run_caloric_prep.py`, `run_caloric_stats.py` | none | `agriculture/jobs/ag_caloric_*.sbatch` |
| Crop coverage masks | `weights/build_crop_weights.py` | none | run once by hand |
| Mortality physical | `cilreg pipeline`, `mortality/make_adm0_physical.py` | `mortality/configs/mortality_*.toml` | `mortality/jobs/mort_phys_*`, `mortality_adm*` |
| Mortality valued | `mortality/run_valued_mortality.py` | none | `mortality/jobs/mort_val_*.sbatch` |
| Labor physical | `labor/run_physical_labor.py` | `labor/configs/labor_2025_phys_ir_*.toml` | `labor/jobs/labor_phys_*.sbatch` |
| Labor high and low risk | `labor/run_physical_labor_groups.py` | none | `labor/jobs/labor_groups_*.sbatch` |
| Labor valued | `cilreg pipeline`, `labor/make_adm0_labor_valued.py` | `labor/configs/labor_2025_*.toml` | `labor/jobs/labor_val_*.sbatch` |
| Percent of GDP, agriculture and mortality | `derived/run_pct_gdp.py` | none | `derived/jobs/pct_gdp.sbatch` |
| Percent of GDP, labor | `derived/run_labor_pctgdp.py` | none | `derived/jobs/labor_pctgdp.sbatch` |
| Energy regionalization | `energy/run_energy_regionalization.py` | none | `energy/jobs/run_energy_regionalization.sbatch` |
| Energy delta method gradients | projection system | see the job file | `energy/jobs/deltamethod.sbatch` |
| Climate ensemble, five variables | `climate/climate_days_daily.py`, `climate_days_finish.py` | none | `climate/jobs/climate_days_{hist,scen,finish}.sbatch` |
| Climate monthly cross-check | `climate/climate_means_monthly.py`, `climate_means_finish.py` | none | `climate/jobs/climate_means_sweep.sbatch` |
| Grid to GADM weights | `cilreg` weight generation | `weights/configs/grid025_to_gadm20_adm{1,2}_per_destination.toml` | `weights/jobs/climate_weights_gen.sbatch` |
| Web JSON for the map | `web/make_web_json.py`, `web/make_climate_web_json.py` | `web/gadm20_names.csv` | `web/jobs/*` |

## Model weights

`agriculture/smme/rcp85_SMME_weights_agriculture.tsv` is the standard rcp85
SMME table with the agriculture overrides applied: ACCESS1-0 set to zero and
`ipsl-cm5a-lr`, `ipsl-cm5a-mr`, `csiro-mk3-6-0`, `canesm2` and `bnu-esm` set to
0.036, everything else unchanged. The overrides come from the agriculture
repository's prospectus extraction code and are not in the published
replication package.

The tables carry the twelve pattern-scaled surrogates under unprefixed names
(`gfdl-cm3_89`), while the damages trees name those directories
`surrogate_gfdl-cm3_89`. The pipeline strips the prefix when matching. Dropping
that step silently loses twelve of thirty-three models, and they are the ones
covering the tails.

## Order

Weights first, then the sector runs, then the derived measures which read the
sector products, then the web generators which read everything. Within
agriculture, impact region and ADM1 and ADM2 run before ADM0, which reads
their reduced frames.
