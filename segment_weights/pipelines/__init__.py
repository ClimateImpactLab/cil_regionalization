"""Pipeline layer: named workflows composed from the library core.

The core (geometry, weights, application, statistics) knows nothing
about file trees, schedulers, or environments. Pipelines add exactly
that composition: resolve a declared data layout, run the core stages
per unit of work, and account for every piece. All site-specific facts
(paths, tree grammars, window bounds, parallelism) arrive through the
pipeline's own config; none live in code.
"""
