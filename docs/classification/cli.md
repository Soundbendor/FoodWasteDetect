# CLI and configuration

Read this document when changing command parsing, overrides, stage dispatch, or
the Ollama download entry point. The implementation is
[`cli.py`](../../src/classification/cli.py), reached by both the `classification` console script
and [`__main__.py`](../../src/classification/__main__.py).

## Command flow

`main()` performs one argparse pass. Experiment commands require `--config_file`
and accept repeatable `--set PATH=VALUE` assignments. Help exits before opening
configuration or dataset files. Abbreviated flags are disabled.

| Stage | Dispatch |
| --- | --- |
| `describe` | `Experiment.generate_descriptions()` |
| `index` | `Experiment.build_vector_database()` |
| `evaluate` | `Experiment.evaluate()` |
| `run` | `Experiment.run()` |
| `matroyshka-sweep` | `experiments.matryoshka.run_sweep()` with a tracked evaluation callback |

Preserve the public `matroyshka-sweep` spelling. Experiment modules have no separate
CLI parser. Stage ordering, cache behavior, and the current description-indexing
limitation are defined in [experiment contracts](experiments.md).

The former per-key flags, `finetune-embeddings`, `build-db`, top-level `evaluate`,
`caption-images`, `generate-descriptions`, `merge-captions`, and `convert-captions`
are removed. They produce ordinary argparse errors; there are no compatibility
commands or old flat-module import shims.

## Override semantics

`_load_config()` loads YAML with OmegaConf and requires a mapping. Struct mode
rejects unknown paths; custom YAML must contain a key before `--set` can target it.
Overrides merge in order, and the last assignment wins. Values use YAML parsing
rather than conversion to the old value's type. For a string that resembles
another type, quote inside the assignment: `--set 'output_dir="001"'`.
OmegaConf's path and merge rules apply, including interpolation.

Each assignment requires a nonempty path and `=`. Malformed assignments and
unknown paths fail before experiment construction. The resolved configuration is a
plain nested dictionary passed to `_run_experiment()`; source YAML is never
rewritten. Reports, W&B config, and the configuration artifact retain this effective
dictionary. Overrides do not automatically choose isolated artifact names or
manage services.

The CLI calls `experiments.config.validate_config` before W&B initialization. This
is distinct from backend validation, which can fail later when the lazy encoder
loads. Parsing and shared-validation errors create no run; constructor or stage
errors occur within the native W&B context. See [tracking](tracking.md) for
lifecycle ownership.

## Ollama downloads

`classification download_model MODEL_NAME` reads `cfg/experiment_config_internvl.yml`
relative to the working directory; it has no config flag or experiment overrides.
The wrapper passes its `ollama` section into `OllamaSession`, opens the owned server
context, and streams the requested model pull. Downloads require the connection and
process settings; the explicit model name makes `ollama.llm` unnecessary. The YAML
stays unchanged and no W&B run is created. Server ownership, executable validation,
and cleanup are specified in [descriptions](data.md#class-descriptions-and-ollama).

## Verification

[`tests/test_classification_cli.py`](../../tests/test_classification_cli.py) covers parsing, dispatch,
ordered overrides, YAML types, interpolation, unchanged source files, download
cleanup, and wrapper argument forwarding with mocked backends. Lifecycle and
configuration publication checks also live in
[`tests/test_tracking.py`](../../tests/test_tracking.py). Use the
[development commands](../development.md); no model or service is needed for these
checks.
