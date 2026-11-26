# Benchmarks

Timing runs on synthetic data shaped like one leaf of the agriculture
ADM2 run: 24,378 impact regions by 118 years, weighted onto 42,726 ADM2
targets, then reduced to three windows.

The source side is calibrated to a real leaf, `corn-160221.nc4` of the
corn-2025 tree, as the pipeline reads it: 2,876,604 rows ordered by year
then region, years 1981 to 2098 as int32, a float32 value, and region
ids of the real lengths (3 to 35 characters, 14 on average).

The target side is calibrated to the corn ADM2 crop area weights,
`ir_adm2_croparea_corn`. `weights_counts.py` holds counts taken from
that file, no ids and no weights, and `synthetic.py` builds a frame
with those counts:

| | |
|---|---|
| Rows | 51,216, sorted by target then region |
| Targets | 42,726 in 253 countries, the largest with 5,504 |
| Regions per target | one for 40,203 (94.09%), mean 1.1987, at most 48 |
| Targets per region | one for 21,070 of the 24,378, at most 184 |
| Rows with no crop area | 8,192, NaN weight, in 4,391 regions |
| Targets with no crop area | 6,639, and 485 more with only some rows empty |
| After `apply_weights` | 36,087 targets, 4,258,266 rows |

A region with no crop area has a NaN weight in every row. The benchmarks
pass `on_zero_weight="skip"`; the default policy raises on those
regions. What the library then reports is the regions: 4,391
on `coverage.zero_weight` and in one warning a call. It does not count
the targets. The 6,639 whose regions are all empty are missing from the
output, and the 485 that keep some rows are averaged over those alone.

Three things differ from the file. `ID_1` has 90 distinct values, not
93: one country there numbers its first-level units out of sequence and
its ids are not copied. A target's regions do not carry its country
code, because the countries of the region ids are all the same size and
the real ones are not. And 27 of the real rows tie a target with
several regions to a region with several targets; here no row does.

    pytest benchmarks -m benchmark -s

`-s` shows the timings, printed on lines that start with `[benchmark]`.
The default suite skips these tests: `pyproject.toml` excludes the
`benchmark` marker.

## What passes and fails

Each timing test runs two code paths and asserts that they give
identical output. `test_artifact_shape.py` asserts the counts in the
table above. Both hold on any machine. No test compares a time to a
threshold, so a slow machine cannot fail one.

The numbers are machine specific. Shared CI runners vary two to three
times between runs, and a busy laptop does the same. Compare two paths
within one run, not one run with another.

## The four files

`test_artifact_shape.py` times nothing. It checks the synthetic weights
against the counts of the real file at full size, that the default
policy raises on the empty regions, what the skip reports, and that the
frames still fit together when scaled down.

`test_bench_window_means.py` runs `window_means` against the plain
pandas path it replaced: `DataFrame.duplicated` for the duplicate check
and a groupby over the key columns for each window.

`test_bench_window_means_parts.py` splits the time inside
`window_means` into group codes, duplicate check and the rest. It also
compares labelling a leaf with its member (batch, gcm, iam, ssp) before
reducing against after. It calls two private helpers of `stats.py` to
time the parts, so a rename there has to be followed here.

`test_bench_leaf.py` takes one leaf through `apply_weights` and
`window_means` with the labels attached at three different points, and
runs a process pool twice over the same leaves: once with 1.4 tasks a
worker and once with every task halved.
It prints the skipped regions and the missing targets once; the pool
workers mute the warning the library logs for every leaf.

## Sizes

| Variable | Default | Meaning |
|---|---|---|
| `BENCH_SCALE` | 1 | divides the leaf in the single-process tests |
| `BENCH_POOL_SCALE` | 8 | divides the leaf in the pool test |
| `BENCH_WORKERS` | cores, at most 8 | pool size |
| `BENCH_LEAVES` | 48 | leaves in one coarse pool task |

A full-size leaf peaks at about 2 GiB in one process. The pool test
uses smaller leaves by default because eight full-size workers swap on
a 16 GB machine, and then the run measures swapping.
