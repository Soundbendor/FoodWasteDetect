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

`_load_config()` reads a top-level mapping with `yaml.safe_load()`, applies ordered
overrides, then returns an `ExperimentConfig`. Field definitions and defaults live
in [`config.py`](../../src/classification/experiments/config.py).

Overrides form nested mappings from dotted paths, including fields omitted from
YAML. Mapping values merge recursively; scalar/null values replace the old value.
The last assignment wins, and overrides can repair invalid input values. Values
use YAML syntax; split occurs at the first equals sign, preserving additional
equals signs. Quote strings that resemble other types: `--set 'output_dir="001"'`.

OmegaConf and interpolation have been removed. Strings containing interpolation
syntax receive ordinary YAML/string handling. Source YAML remains unchanged.
YAML, override, and schema errors produce argparse errors with field paths before
W&B initialization or experiment construction, including sweep dispatch.
Backend checks still occur when constructing the lazy encoder.

For cached dimension sweeps, set an explicit ordered list in YAML or use, for
example, `--set 'sweep_dimensions=[256,512,1024]'`. The field is required only
for sweep execution. See the [sweep contract](experiments.md#cached-matryoshka-sweep).

Overrides do not select isolated artifact names or manage services.

## Ollama downloads

`classification download_model MODEL_NAME` reads `cfg/experiment_config_internvl.yml`
relative to the working directory; it has no config flag or experiment overrides.
The wrapper validates only its `ollama` section as `OllamaConfig` and passes it into `OllamaSession`, opens the owned server
context, and streams the requested model pull. Downloads require the connection and
process settings; the explicit model name makes `ollama.llm` unnecessary. The YAML
stays unchanged and no W&B run is created. Server ownership, executable validation,
and cleanup are specified in [descriptions](data.md#class-descriptions-and-ollama).

## Verification

[`tests/test_classification_cli.py`](../../tests/test_classification_cli.py) covers parsing, dispatch,
schema-defined ordered overrides, YAML types, unchanged source files, download
cleanup, and wrapper argument forwarding with mocked backends. Lifecycle and
configuration publication checks also live in
[`tests/test_tracking.py`](../../tests/test_tracking.py). Use the
[development commands](../development.md); no model or service is needed for these
checks.
