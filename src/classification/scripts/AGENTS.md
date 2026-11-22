# Explicit maintenance scripts

Read [script contracts](../../../docs/classification/scripts.md) before changing or
running `smoke_jina.py`, `start_deepseek.sh`, `prepare_matroyshka_sweep.py`, or the
repository-level Slurm and sweep wrappers. `__init__.py` remains side-effect free.
Verify smoke-runner changes with `tests/test_smoke_jina.py`, cached dimension job
preparation and submission with `tests/test_matroyshka_submission.py`, result
collection and plotting with `tests/test_collect_results.py` and
`tests/test_plot_matroyshka_sweep.py`, and other wrapper changes with
`tests/test_classification_cli.py`.
