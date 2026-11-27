`GMTanom_all_temp_2001_2010_smooth.csv`

Reference data, built from: https://doi.org/10.5281/zenodo.6416119, `data/4_damage_function/GMST/GMTanom_all_temp_2001_2010_smooth`. Carleton et al. (2022), QJE. CC BY 4.0.

Columns `gcm,rcp,year,temp`, 33 GCMs under rcp45 and rcp85, 1981 to 2100. Anomaly relative to 2001-2010. md5 `de9b41cd3b6a55cbe96c4ad14aed9ddf`.

`finding_gwl_bins.R` in the inequality repository rebases each `(gcm, rcp)` to 1986-2005 and adds 0.61 C for the shift to pre-industrial, citing AR5 WG1 TS.2.2.1. Both are parameters here.

Smoothing is recorded nowhere. Largest year-on-year change is 0.22 C, in the surrogates under rcp85 late in the century, and 0.08 C among the 21 models.

`surrogate_GFDL-ESM2G_06` has no temperature under rcp45 in any year, and the script above drops it. The other surrogates have none for 2100.
