# CMIP6 threshold counts example

Days above 95F counted three ways for four Peruvian provinces, from GDPCIR, the CMIP6 downscaled and bias corrected data behind the Impact Map climate tables. Counted from daily tasmax, the number is right. Counted from daily tavg or from monthly tavg, it counts something else, and a monthly mean does not hold its days at all. In one model and province the monthly count lands closer to the right one than the daily tavg count, by accident, because a hot month counts its cool days too. Closer is not right. Beside the counts, the annual mean from the daily and the monthly data agree.

The committed daily files were checked against production's per-cell grids for all six model-years: days above 95F match in all 503 cells, and the annual mean differs by 0.0 F. The numbers here are production's numbers for these cells, not a reconstruction.

The provinces are Ucayali and Madre de Dios in the Amazon, Piura on the coast and Cusco in the Andes. The daily files cover three models, FGOALS-g3, MIROC6 and UKESM1-0-LL, for 2000 (historical) and 2090 (ssp370). That shows the mechanism, not the product, which averages 20-year windows over 20 or more models. A table of the product's own values for all twenty ssp370 models shows the real spread.

The `data/` folder is about 5 MB and holds everything the notebook reads. You don't need cluster access or a network connection, and the outputs are saved in the notebook if you only want to read it.

## Run it

With [uv](https://docs.astral.sh/uv/), at the versions in `uv.lock`:

```bash
git clone https://github.com/ClimateImpactLab/cil_regionalization.git
cd cil_regionalization
uv sync --extra examples
uv run jupyter lab examples/cmip6/threshold_counts.ipynb
```

Without uv, `pip install '.[examples]'` installs the same packages at their newest versions.

## Files

- `threshold_counts.ipynb` is the example.
- `data.py` holds the file paths, the provinces, models and years, with the source of each file.
- `prepare_data.py` rebuilds `data/` on Midway3 and checks the daily files against the per-cell fields production wrote. Its docstring lists the sources. It reads GDPCIR from the Microsoft Planetary Computer, so it needs a compute node with outbound network. Submit it from the repository root:

  ```bash
  CILREG_PYTHON=/path/to/python sbatch examples/cmip6/prepare_data.sbatch
  ```

  The log is written to `/scratch/midway3/$USER/cmip6_example_data_<jobid>.log`.

## Where the weights come from

The three weight tables are the rows of the production tables for these provinces, from `/project/cil/gcp/impact_map_inputs/weights/` on Midway3: `grid025_ir_per_destination`, `grid025_gadm20_adm2_per_destination` and `grid025_gadm20_adm1_per_destination`. They were generated with this library from the impact region shapefile and the GADM 2.0 layers, with the GPW v4 rev 10 population raster adjusted to UN World Population Prospects. The configs are in the impactmap-data-gen repository, one per table, at `weights/configs/grid025_to_ir_per_destination.toml`, `weights/configs/grid025_to_gadm20_adm2_per_destination.toml` and `weights/configs/grid025_to_gadm20_adm1_per_destination.toml`. Each table has its own config, so each manifest's config hash points to the one recipe that made it.

## Data and licences

GDPCIR comes from Rhodium Group and the Climate Impact Lab and is distributed on the Microsoft Planetary Computer, in three collections by licence. FGOALS-g3 is CC0 1.0; MIROC6 and UKESM1-0-LL are CC BY 4.0. Each daily and monthly file states its licence. CanESM5, the hottest of the twenty models, is also listed under CC BY-SA 4.0 and is left out of the daily files; it appears only in the summary table.

The ensemble is unweighted. No SMME weight table exists for CMIP6: those tables name CMIP5 models and are indexed by RCP, and a model's weight is its position in a FaIR distribution of warming, so it does not carry over to another generation of models.
