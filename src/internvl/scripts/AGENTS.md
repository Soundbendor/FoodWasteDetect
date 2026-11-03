# Explicit maintenance scripts

Read [script contracts](../../../docs/internvl/scripts.md) before changing or
running these commands. Repository-level submission wrappers are mapped there too.

- `smoke_jina.py` is an explicit real-model CPU check. It can download weights and
  requires authorization; routine checks use the synthetic worker/process tests.
- Preserve temporary cache isolation, GPU hiding, child shutdown before cache
  removal, and signal-handler restoration in the smoke runner.
- `start_deepseek.sh` is a historical daemon launcher with a local binary path and
  home-directory log. Use the managed CLI lifecycle for current description work.
- `__init__.py` remains side-effect free.

Check changed smoke-runner behavior with `tests/test_smoke_jina.py`. Script or
submission inspection alone does not establish real-model, GPU, or HPC acceptance.
