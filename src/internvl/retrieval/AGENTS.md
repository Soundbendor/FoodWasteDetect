# Retrieval

`lancedb.py` owns database access; `voting.py` selects labels; `metrics.py` scores
predictions and rankings. Shared record types live in `internvl.schemas`.
Experiment selection, resumption, and report publication belong to `experiments`.

| Task | Read before changing |
| --- | --- |
| Storage, identity scans, vector fetching, or optimization | [LanceDB storage](../../../docs/internvl/retrieval.md#lancedb-storage) |
| Querying or label assignment | [Exact retrieval and voting](../../../docs/internvl/retrieval.md#exact-retrieval-and-voting) |
| Metric definitions, denominators, or abstentions | [Evaluation metrics](../../../docs/internvl/retrieval.md#evaluation-metrics) |
| Resume ordering or validation-cache writes | [Experiment persistence](../../../docs/internvl/experiments.md#reference-builds-and-resumption) |

Preserve exact cosine search, separate vote fraction from similarity, and keep
pure voting/metric functions independent of database and model loading. Import
concrete modules directly; the package initializer does not load backends.
