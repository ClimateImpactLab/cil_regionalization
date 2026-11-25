# Benchmarks

Timing runs on synthetic data shaped like one leaf of the agriculture
ADM2 run: 24,378 impact regions by 118 years, weighted onto 40,000 ADM2
targets, then reduced to three windows.

The source side is calibrated to a real leaf, `corn-160221.nc4` of the
corn-2025 tree, as the pipeline reads it: 2,876,604 rows ordered by year
then region, years 1981 to 2098 as int32, a float32 value, and region
ids of the real lengths (3 to 35 characters, 14 on average). The target
side is not calibrated yet: the 40,000 targets, their id columns and
the one to three sources each are placeholders.

    pytest benchmarks -m benchmark -s

`-s` shows the timings, printed on lines that start with `[benchmark]`.
The default suite skips these tests: `pyproject.toml` excludes the
`benchmark` marker.

## What passes and fails

Each test runs two code paths and asserts that they give identical
output. That holds on any machine. No test compares a time to a
threshold, so a slow machine cannot fail one.

The numbers are machine specific. Shared CI runners vary two to three
times between runs, and a busy laptop does the same. Compare two paths
within one run, not one run with another.

## The three files

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
