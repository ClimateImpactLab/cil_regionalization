# Architecture notes

Notes on how this package's statistics relate to the products other
Climate Impact Lab pipelines have published. The package's own design
is documented in the module docstrings; this file records the
cross-cutting facts that do not belong to any one module.

## The statistics ordering, and who follows it

Every statistic this package produces follows one ordering: aggregate
each Monte Carlo draw spatially, reduce time per draw, then compute
statistics over the pooled draws, with climate models weighted by the
SMME weights during the pooling. `cil_regionalization.stats` refuses
input that already carries a statistic dimension precisely so that no
other ordering can slip through. The lab's standard extraction tool,
`quantiles.py` in `prospectus-tools/gcp/extract`, implements the same
construction: it pools gcm x batch as one flat weighted sample
(`do-gcmweights: yes`) and takes quantiles of that pool. The published
mortality, labor, energy, and per-crop agriculture extractions were
produced that way, and results built with this package are comparable
to them.

## The agriculture valued quantiles are a different construction

The files at

    /project/cil/gcp/outputs/agriculture/agval/main_spec/extracted/ag_mc_valued_*.csv

(May 2025, welfare change `wc_no_reallocation` per impact region, with
columns mean and Q1 through Q99) were not produced by `quantiles.py`.
They come from the agval repo's own extraction
(`agval/main/1_aggregate_gcms.py` then `2_get_quantiles.py`, calling
`agval/utils/aggregate.py` and `agval/utils/extract.py`), which takes
the SMME-weighted mean across climate models first and then computes
quantiles across the 15 batches of the ensemble means. The
`2_get_quantiles.py` docstring says it is "doing essentially the same
thing as quantiles.py"; it is not. Quantiles of GCM means carry only
the statistical uncertainty around the ensemble mean: the climate
model spread is averaged away before the distribution is taken, and no
SMME weight enters the quantile step at all. The resulting intervals
are systematically narrower than the published-method intervals.

The consequence: agriculture's valued quantiles must not be compared
against the mortality, labor, or energy quantile products, or against
anything this package produces, as if they were the same object. The
means are comparable (the weighted mean commutes with this reordering
when the sample is balanced); the quantiles are not. Anyone needing
published-method quantiles for valued agriculture should recompute
them from the per-leaf `disaggregated_damages.nc4` draws with this
package or with `quantiles.py`, not reuse the extracted files.

A copy of this note sits beside the files themselves.
