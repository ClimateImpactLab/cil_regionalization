# Pipeline runs

These configs and job scripts only run with access to the University
of Chicago RCC cluster: the input trees, the weight files, and the
scheduler account they name live there. They are the record of how the
published aggregations were run, not something to run elsewhere. For a
runnable example, see `examples/aggregation`.

Each TOML drives one `segweights pipeline` run (one damages tree, one
target level, one RCP); the matching .sbatch file submits it. Paths
inside them name cluster locations current at the time of the runs and
need updating if the inputs move.
