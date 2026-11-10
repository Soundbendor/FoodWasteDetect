# Whole-image classification package

`classification` classifies FoodX251 images through multimodal embeddings and LanceDB
retrieval. Reference inputs are class names, visual descriptions, training images,
or a configured combination. The pipeline embeds whole images directly. Ollama
generates class descriptions separately.

## Working in this package

1. Read the responsibility document below for the files or behavior being changed.
2. Follow the relevant submodule guide before editing that directory.
3. Verify the changed contract with synthetic CPU fixtures. See
   [development](../../docs/development.md) for Pixi commands and
   [verification history](../../docs/experiments.md) for the limits of earlier checks.

Import concrete modules directly; package initializers stay lightweight. Keep
backend construction and runtime side effects behind explicit calls. Shared
configuration models belong in `experiments.config.ExperimentConfig`; shared
JSON writing belongs in `experiments.artifacts.write_json`.

## Package responsibilities

| Files | Responsibility and when to read more |
| --- | --- |
| `cli.py`, `__main__.py` | Parse once, resolve YAML overrides, validate shared settings, and dispatch stages within native W&B contexts. Read [CLI](../../docs/classification/cli.md) when changing commands, configuration, or dispatch. |
| `tracking.py` | Publish resolved configuration, reference counts, and completed evaluation results. Read [tracking](../../docs/classification/tracking.md) when changing W&B lifecycle, artifacts, summaries, or progress. |
| `datasets/image_dataset.py` | Load local FoodX251 metadata and select/decode ordered batches. Read [data](../../docs/classification/data.md) when changing splits, identities, image caps, or image ownership. |
| `schemas.py` | Shared reference, neighbor, prediction, and retrieval-metric records. Read [data](../../docs/classification/data.md#shared-records) when changing fields or record/vector associations. |
| `__init__.py` | Package description; no backend imports. |
| `pyproject.toml`, `pixi.lock`, `.python-version`, `MANIFEST.in` | Standalone packaging, environment selection, tasks, and source-distribution inputs. Read [development](../../docs/development.md) before environment or packaging changes. |
| `README.md` | Installation and running commands; edits require explicit approval under the root guide. |

## Submodule map

| Module | Boundary | Guide and detailed reference |
| --- | --- | --- |
| `encoders/` | Shared batch interface, local Sentence Transformers, Vertex requests | [Guide](encoders/AGENTS.md); [model protocols](../../docs/classification/encoders.md) for model, dimension, device, or request changes. |
| `experiments/` | Orchestration, reference/cache construction, artifacts, configuration, dimension sweeps | [Guide](experiments/AGENTS.md); [experiment contracts](../../docs/classification/experiments.md) for stage ordering, resumption, or report changes. |
| `llm/` | Class-description prompts, structured responses, and owned Ollama processes | [Guide](llm/AGENTS.md); [descriptions](../../docs/classification/data.md#class-descriptions-and-ollama) for vocabulary generation or service lifecycle changes. |
| `retrieval/` | LanceDB adapter, voting, classification and ranking metrics | [Guide](retrieval/AGENTS.md); [retrieval contracts](../../docs/classification/retrieval.md) for storage, search, scores, or evaluation changes. |
| `scripts/` | Explicit maintenance and real-model checks | [Guide](scripts/AGENTS.md); [scripts](../../docs/classification/scripts.md) for smoke checks or Slurm submission changes. |

`Experiment` composes these responsibilities: metadata loads during construction;
the encoder loads on first use; `IndexBuilder` prepares and persists references or
validation caches; `LanceDBIndex` owns storage operations, including build-metadata
snapshots and derived prefix tables. The cached dimension sweep in
`experiments.matroyshka` orchestrates prefix-table preparation through `LanceDBIndex`
and delegates validation evaluation to `Experiment.evaluate_cached()`.
`experiments.evaluation` computes predictions from supplied batches; `Experiment`
writes a complete local report before `Tracker` publishes it. Direct
`Experiment(config: ExperimentConfig)` calls have no W&B run unless a tracker is supplied.
