# Worked example: Colombia

This example aggregates real Monte Carlo mortality projections for
Colombia from impact regions to departments (ADM1) and municipalities
(ADM2). It uses a subset of the full projections: the monetized
damages sample is one Monte Carlo batch out of fifteen, across all 33
climate models (rcp85, SSP3); the physical rates sample is two batches
of one model (GFDL-ESM2G), because the tree holding the rates is still
being regenerated. The numbers are a demonstration, not results.

Start with the notebook, `aggregation_colombia.ipynb`. It fetches the
weights, aggregates and maps one draw, shows percent of GDP with the
`ratio` kind, pools statistics across the 33 models, maps a pooled
quantile next to the single draw, plots the distribution of draws
for every department, and runs the same aggregation against the GADM
4.1 weights. All output is saved in the notebook, so you can
read it without running anything. Beyond the `[netcdf]` extra it needs
`matplotlib`; there is no separate extra for that.

The script `run_example.py` runs the physical rates (the intensive
kind, which the notebook only describes) plus the damages at both
levels:

```
python examples/aggregation/run_example.py            # fetches weights
python examples/aggregation/run_example.py --offline  # committed slices
```

The default mode fetches the published weights from Zenodo and needs
network access. `--offline` uses the Colombia weight slices committed
under `data/weights/`. Both modes produce identical output.

## Quick example

```python
import cil_regionalization as cilreg

weights = cilreg.fetch_weights("gadm20-adm1-per-source")
data = cilreg.read_netcdf_leaf(leaf_path, variables=["total_damages"],
                               region_dim="region", region_col="hierid",
                               kind="extensive")
result = cilreg.apply_weights(weights, data, kind="extensive", weight="pop",
                              value_col="total_damages",
                              data_version="world-combo-201710",
                              restrict_to_sources={(h,) for h in data["hierid"].unique()})
```

The weight file covers the whole world, while this example only uses
Colombia. `restrict_to_sources` limits the aggregation to the impact
regions in the sample. `data_version` names the impact region geometry
version the data was built on; if it does not match the
`source_version` in the weight file, `apply_weights` raises an error.

## Sample data

Colombia has 500 impact regions, 32 ADM1 departments in this GADM 2.0
version (the country later gained a 33rd), and 1,065 ADM2
municipalities. The committed data (about 23 MB) holds:

- `montecarlo/batch0/rcp85/<gcm>/low/SSP3/mortality_damages_IR_batch.nc4`
  for each of the 33 climate models: monetized mortality damages
  (`total_damages`, 2019 USD) and the GDP embedded in the same files.
  Damages are extensive: totals, split between units with `per_source`
  weights.
- `montecarlo/{batch0,batch1}/rcp85/GFDL-ESM2G/low/SSP3/Agespec_interaction_response-combined.nc4`:
  the physical mortality rate (`rebased`, deaths per person per year).
  Rates are intensive: averages, taken over the regions in each unit
  with `per_destination` weights.
- `weights/`: the Colombia slices of the published weight files, used
  by `--offline`.
- `col_adm1_plot.parquet`: department shapes and names, used for the
  maps and the printed tables. Built from the impact region shapes and
  simplified; the department names come from the GADM 2.0 copy, whose
  accented characters were damaged, and are corrected here.

## Example output

```
The tree this example reads: one damages leaf per climate model,
  montecarlo/batch0/rcp85/<gcm>/low/SSP3/ for 33 models,
plus the physical rate leaves,
  montecarlo/batch0/rcp85/GFDL-ESM2G/low/SSP3/Agespec_interaction_response-combined.nc4
  montecarlo/batch1/rcp85/GFDL-ESM2G/low/SSP3/Agespec_interaction_response-combined.nc4

Physical rate at ADM1 (deaths per person per year, intensive,
population weighted mean over each department's impact regions):
      NAME_1  year   rebased
      Boyacá  2099 -0.002262
Cundinamarca  2099 -0.002181
      Nariño  2099 -0.001536
       Cauca  2099 -0.001118
      Caldas  2099 -0.001107
  ... 32 departments, both batches computed

Monetized damages at ADM1 (2019 USD, extensive, allocated shares
sum to each impact region's total; mass balance checked):
                  NAME_1  year  total_damages        gcm
                 Guainía  2099  -3.912271e+07 GFDL-ESM2G
                  Vaupés  2099  -8.851429e+07 GFDL-ESM2G
                 Vichada  2099  -1.399679e+08 GFDL-ESM2G
                Amazonas  2099  -1.678937e+08 GFDL-ESM2G
San Andrés y Providencia  2099  -1.982496e+08 GFDL-ESM2G
  sum over departments -1.774214e+11 = sum over impact regions -1.774214e+11

Monetized damages at ADM2 (1,065 municipalities from the same
500 impact regions; same kind, same checks, finer weights):
  region COL.6.R2297bdd18341ff36 splits across 90 municipalities;
  its largest population shares:
NAME_1  ID_2    popwt
Boyacá   294 0.095976
Boyacá   275 0.048733
Boyacá   252 0.034370
  shares sum to 1.000000
  2099 sum over municipalities -1.774214e+11 matches the impact region sum

Statistics over the 33 climate models (window mean 2080-2099,
then pooled):
   NAME_1 statistic  total_damages
Antioquia      mean  -1.270419e+10
Antioquia       q05  -4.270336e+10
Antioquia       q50  -1.524614e+10
Antioquia       q95   2.302501e+10
```

The department totals add back up to the impact region totals at both
levels; that check runs inside `apply_weights`. At ADM2 one impact
region splits across 90 municipalities by population share. Rates have
no adding-up identity; the check there is that each department value
lies between the smallest and largest rate of its own impact regions.

Statistics are calculated after the spatial aggregation and the
2080-2099 mean. Each climate model is one draw; with 33 of them the
spread is real, and the quantiles for a department can cross zero.
