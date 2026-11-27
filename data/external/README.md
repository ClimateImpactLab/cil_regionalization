# Reference data

Files read by the test suite and the examples, kept apart from both.

## `GMTanom_all_temp_2001_2010_smooth.csv`

Reference data, built from: https://doi.org/10.5281/zenodo.6416119, `data/4_damage_function/GMST/GMTanom_all_temp_2001_2010_smooth`. Carleton et al. (2022), QJE. CC BY 4.0.

Columns `gcm,rcp,year,temp`, 33 GCMs under rcp45 and rcp85, 1981 to 2100. Anomaly relative to 2001-2010. md5 `de9b41cd3b6a55cbe96c4ad14aed9ddf`.

`finding_gwl_bins.R` in the inequality repository rebases each `(gcm, rcp)` to 1986-2005 and adds 0.61 C for the shift to pre-industrial, citing AR5 WG1 TS.2.2.1. Both are settings of `examples/warming_levels/ar5_cmip5_smme.toml`.

Smoothing is recorded nowhere. Largest year-on-year change is 0.22 C, in the surrogates under rcp85 late in the century, and 0.08 C among the 21 models.

`surrogate_GFDL-ESM2G_06` has no temperature under rcp45 in any year, and the script above drops it. The other surrogates have none for 2100.

## `inequality_gwl_bins.csv`

The models at each warming level in the Inequality Paper, as its extraction hardcodes them, one `(gcm, rcp)` member a row. Read on 2026-11-26 from the inequality repository, branch `elliot_wip` at `4dd6b9c`, `3_extraction/extract_physical_impacts/inequality_extract_mort_impacts.sh`, the binlist at line 25 and the case arms over lines 42 to 107. The `a` and `b` suffixes there split one level by rcp, and are merged here. No surrogate appears in any list. `tests/integration/test_warming_levels_paper.py` derives the same members from the table above, with the settings of `examples/warming_levels/ar5_cmip5_smme.toml`.
