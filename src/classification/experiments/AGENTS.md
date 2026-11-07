# Experiment orchestration

This package runs the whole-image classification baseline. `Experiment` owns
stage orchestration, validation-batch preparation, and report publication; `evaluation.py` owns batch prediction,
progress, and completeness checks. `IndexBuilder` owns reference selection and
encoding batches for reference tables and validation caches; table opening,
missing-key scans, and completion checks belong to `EmbeddingTable` in `retrieval`.
Open tables only through `EmbeddingTable` with the matching role. `matryoshka.py`
holds sweep policy (dimensions, `<source>_d<dim>` names, distinct-name and
source-dimension checks) and derives prefix tables through
`EmbeddingTable.derive_prefix()`; it must not import LanceDB or Arrow.

| Task | Read before changing |
| --- | --- |
| Stages, configuration, or description inputs | [Experiment lifecycle](../../../docs/classification/experiments.md#lifecycle-and-configuration) |
| Fingerprinted settings or incompatible-table refusal | [Fingerprints and incompatible tables](../../../docs/classification/experiments.md#fingerprints-and-incompatible-tables) |
| Reference selection, restart behavior, or validation caching | [Builds and resumption](../../../docs/classification/experiments.md#reference-builds-and-resumption), [validation caching](../../../docs/classification/experiments.md#validation-caching) |
| JSON output or failure handling | [Artifact and failure contract](../../../docs/classification/experiments.md#artifact-and-failure-contract) |
| Cached dimension sweeps | [Matryoshka sweep](../../../docs/classification/experiments.md#cached-matryoshka-sweep) |
| Query depth, voting, or reported metrics | [Retrieval contract](../../../docs/classification/retrieval.md) |
| Progress bars or W&B publication | [Tracking](../../../docs/classification/tracking.md) |

`fingerprints.py` alone decides which settings change vectors; pass its
fingerprints to every `EmbeddingTable` resume or evaluation open.
`Experiment.evaluate()` completes and marks an enabled cache; `evaluate_cached()`
reads existing vectors regardless of the cache flag. Evaluation opens tables with
`EmbeddingTable.open_complete()`, supplying expected keys from `IndexBuilder`;
builders return complete tables without scanning or encoding. Cache setup is eager
and vector fetching lazy. `batches.py` owns only the batch type.
