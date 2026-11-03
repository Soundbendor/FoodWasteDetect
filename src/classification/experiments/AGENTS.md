# Experiment orchestration

This package runs the whole-image classification baseline. `Experiment` owns
stage orchestration, evaluation, and publication; `IndexBuilder` owns reference
selection and encoding batches. Ordinary database operations use `retrieval`;
`matryoshka.py` directly reads and creates the sweep's derived tables.

| Task | Read before changing |
| --- | --- |
| Stages, configuration validation, or description inputs | [Experiment lifecycle](../../../docs/classification/experiments.md#lifecycle-and-configuration) |
| Reference selection, restart behavior, or validation caching | [Builds and resumption](../../../docs/classification/experiments.md#reference-builds-and-resumption), [validation caching](../../../docs/classification/experiments.md#validation-caching) |
| JSON output or failure handling | [Artifact and failure contract](../../../docs/classification/experiments.md#artifact-and-failure-contract) |
| Cached dimension sweeps | [Matryoshka sweep](../../../docs/classification/experiments.md#cached-matryoshka-sweep) |
| Query depth, voting, or reported metrics | [Retrieval contract](../../../docs/classification/retrieval.md) |
| Progress bars or W&B publication | [Tracking](../../../docs/classification/tracking.md) |

`config.py`, `artifacts.py`, and `batches.py` are the shared validation, JSON, and
ordered metadata/vector helpers. Import concrete modules directly. Keep CLI
parsing in `classification.cli`; package initializers remain free of backend imports.
