# Food Waste Detection agent guide

## Project goals

Develop the selected food-recognition architecture: YOLO localization, Jina CLIP
image embeddings, and Qdrant similarity classification. `src/classification`
provides a separate whole-image embedding/LanceDB comparison. Keep those
experiment contracts separate; existing box/class outputs do not establish
instance segmentation. Read the [project overview](docs/overview.md) when
choosing or extending a pipeline.

## Working rules

- Inspect Git status before editing and preserve unrelated work. Keep commits
  and pull requests local to this repository.
- Always use `pixi` for your dependency management, environment handling, and
  package management. Don't use `uv` or other tools.
- Use small synthetic fixtures and CPU-only checks. Large model/dataset
  downloads, Ollama startup, GPU/HPC work, live cloud spending, shared-storage
  changes, and model publication require explicit authorization; prior
  authorization retains its scope. Keep credentials, private
  data/media/embeddings, and machine-specific access material out of commits.
- Import dependencies normally, including annotation imports; do not use
  `TYPE_CHECKING` workarounds or tests that forbid dependency imports. Keep
  model construction, downloads, service startup, and file mutations behind
  explicit calls.
- Edit any `README.md` only with explicit approval for that README update. Use
  `create-readme` and `humanizer:humanizer` for approved edits and preserve
  verified commands. READMEs cover installation and usage; agent contracts
  belong here and in the linked docs.
- Before changing a module, read its scoped guide and the task reference below.
  Keep each contract in its owning document; add every new `docs/` file to this
  index, except files under `docs/plans/`, which are exempt. Use
  [development checks](docs/development.md) for dependencies, packaging, tests,
  and documentation maintenance.

## Pixi setup

Run from the repository root. The standalone package manifest and committed
lockfile own these environments; select the model family explicitly:

```sh
# Jina CLIP v2 / Jina Embeddings v4 (default classification YAML)
pixi install --manifest-path src/classification/pyproject.toml --locked -e jina-cpu
# EmbeddingGemma2 / OpenAI CLIP / Qwen
pixi install --manifest-path src/classification/pyproject.toml --locked -e default
# Prepare the corresponding CUDA environment when needed
pixi install --manifest-path src/classification/pyproject.toml --locked -e jina-cu126
pixi install --manifest-path src/classification/pyproject.toml --locked -e cu126
```

Use an installed environment without updating it:
`pixi run --manifest-path src/classification/pyproject.toml --as-is -e jina-cpu classification --help`.
See [development](docs/development.md) for checks and hardware-selection
boundaries.

## Scoped agent guides

Start with [classification](src/classification/AGENTS.md) for package ownership,
CLI, tracking, and top-level files. For changes within a subpackage, also read
its guide: [encoders](src/classification/encoders/AGENTS.md),
[llm](src/classification/llm/AGENTS.md),
[experiments](src/classification/experiments/AGENTS.md),
[retrieval](src/classification/retrieval/AGENTS.md), or
[scripts](src/classification/scripts/AGENTS.md).

## Agent skills

### Issue tracker

Issues live in GitHub Issues for Soundbendor/FoodWasteDetect.
See [issue tracker](docs/agents/issue-tracker.md).

### Triage labels

Use the five default triage labels.
See [label mapping](docs/agents/triage-labels.md).

### Domain docs

Use a single-context layout: root `GLOSSARY.md` and `docs/adr/`.
See [domain rules](docs/agents/domain.md).

## Documentation index

Read the matching reference before working on its task. Current contracts live
in module docs; plans, handoffs, and dated verification describe their recorded
state.

| Task / read when                                                                   | Documentation                                                                |
| ---------------------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| Scope: select a pipeline or interpret branch history                               | [Project overview](docs/overview.md)                                         |
| Development: change dependencies, packaging, tests, or documentation               | [Development and environment checks](docs/development.md)                    |
| Issues: create, read, or update tracked work                                       | [Issue tracker](docs/agents/issue-tracker.md)                                |
| Triage: classify issues or change their readiness                                 | [Triage labels](docs/agents/triage-labels.md)                                 |
| Domain: explore terminology or architectural decisions                            | [Domain rules](docs/agents/domain.md)                                       |
| CLIP design: extend localization, masks, crop retrieval, or evaluation             | [CLIP architecture](docs/clip/architecture.md)                               |
| CLIP repairs: touch legacy annotations, crops, indexing, metrics, or training      | [Source review and correctness gaps](docs/clip/review.md)                    |
| CLI: change parsing, overrides, config rules, command dispatch, or downloads       | [CLI and configuration](docs/classification/cli.md)                          |
| Tracking: change W&B lifecycle, artifacts, summaries, or progress                  | [Tracking and progress](docs/classification/tracking.md)                     |
| Data: change FoodX251 selection, schemas, descriptions, or Ollama ownership        | [Data and descriptions](docs/classification/data.md)                         |
| Encoders: change local/Vertex models, dimensions, image ownership, or requests     | [Encoder contracts](docs/classification/encoders.md)                         |
| Experiments: change stages, resumption, validation caching, reports, or sweeps     | [Experiment contracts](docs/classification/experiments.md)                   |
| Retrieval: change embedding tables, voting, or metric definitions                  | [Embedding tables and retrieval](docs/classification/retrieval.md)           |
| Decision: change embedding table headers, legacy refusal, or migration policy      | [ADR-0001: embedding table header](docs/adr/0001-embedding-table-header.md)  |
| Operations: change smoke helpers or Slurm/service scripts                          | [Script operations](docs/classification/scripts.md)                          |
| Verification: inspect environment checks, historical results, or acceptance limits | [Experiment verification notes](docs/experiments.md)                         |
| Migration history: inspect the Pydantic configuration migration                    | [Pydantic migration plan](docs/plans/pydantic_config_migration.md)             |
| Refactor history: inspect the IndexBuilder simplification plan                     | [IndexBuilder plan](docs/plans/index_builder_refactor.md)                    |
| Proposed comparison: consider direct Ollama image classification                   | [Ollama classification plan](docs/plans/ollama_image_classification.md)      |
| Legacy text: interpret retired caption/retrieval code or artifacts                 | [Retired InternVL workflow](docs/history/internvl-text.md)                   |
| Decision history: inspect the October 19 whole-image handoff                       | [Historical handoff](docs/handoffs/2026-10-19-whole-image-classification.md) |
