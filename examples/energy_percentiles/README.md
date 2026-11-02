# Electricity spending percentiles example

How climate change moves electricity spending in Qatar and Iceland over 2080 to 2099, under RCP8.5 and SSP3 (IIASA), with full adaptation. Qatar spends more, Iceland less.

These percentiles need no batches of Monte Carlo projections. The run for each climate projection carries the gradient of the result with respect to the regression coefficients, and their covariance matrix, so each projection gives a normal distribution at any province or country. The notebook mixes the 33 climate projections, from 21 climate models and 12 pattern-scaled surrogates, with their SMME weights and reads the percentiles off the mixture. Every step is a cell, from reading one leaf to the final table. A map at the end shows the change per person in 2020 to 2039 and 2080 to 2099 for two other countries, Chile and Austria, chosen because their units differ so much, at country, impact region, adm1 and adm2 level.

The `data/` folder is about 8 MB and holds everything the notebook reads. You don't need cluster access, and the outputs are saved in the notebook if you only want to read it.

## Run it

With [uv](https://docs.astral.sh/uv/), at the versions in `uv.lock`:

```bash
git clone https://github.com/ClimateImpactLab/cil_regionalization.git
cd cil_regionalization
uv sync --extra examples
uv run jupyter lab examples/energy_percentiles/energy_percentiles.ipynb
```

Without uv, `pip install '.[examples]'` installs the same packages at their newest versions.

## Files

- `energy_percentiles.ipynb` is the example.
- `data.py` holds the file paths and the scenario, with the source of each file.
- `prepare_data.py` rebuilds `data/` from the CIL filesystem on Midway3 and checks it against the Impact Map. Its docstring lists the source files. Submit it from the repository root:

  ```bash
  CILREG_PYTHON=/path/to/python sbatch examples/energy_percentiles/prepare_data.sbatch
  ```
