# Retrieval

`lancedb.py` owns database access; `voting.py` selects labels; `metrics.py` scores
predictions and rankings. Shared record types live in `classification.schemas`.
Experiment selection, resumption, and report publication belong to `experiments`.

| Task | Read before changing |
| --- | --- |
| Storage, identity scans, row counts, vector fetching, or optimization | [LanceDB storage](../../../docs/classification/retrieval.md#lancedb-storage) |
| Stored width or sweep prefix tables | [Prefix tables](../../../docs/classification/retrieval.md#prefix-tables) |
| Querying or label assignment | [Exact retrieval and voting](../../../docs/classification/retrieval.md#exact-retrieval-and-voting) |
| Metric definitions, denominators, or abstentions | [Evaluation metrics](../../../docs/classification/retrieval.md#evaluation-metrics) |
| Resume ordering or validation-cache writes | [Experiment persistence](../../../docs/classification/experiments.md#reference-builds-and-resumption) |

Preserve exact cosine search, separate vote fraction from similarity, and keep
pure voting/metric functions independent of database and model loading. Import
concrete modules directly; the package initializer does not load backends.
