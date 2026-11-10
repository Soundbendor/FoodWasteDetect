# Explicit maintenance scripts

Read [script contracts](../../../docs/classification/scripts.md) before changing or
running `smoke_jina.py`, `start_deepseek.sh`, or the repository-level Slurm and
sweep wrappers. `__init__.py` remains side-effect free. Verify smoke-runner changes
with `tests/test_smoke_jina.py` and wrapper changes with
`tests/test_classification_cli.py`.
