# Retrieval

`embedding_table.py` owns embedding tables: headers, roles, LanceDB access, column
names, and the record-to-row mapping; `voting.py` selects labels; `metrics.py` scores
predictions and rankings. Shared record types live in `classification.schemas`.
Experiment selection, resumption, and report publication belong to `experiments`.

| Task | Read before changing |
| --- | --- |
| Embedding tables, source-key scans, row counts, vector fetching, or optimization | [Embedding tables](../../../docs/classification/retrieval.md#embedding-tables) |
| Append role guard or duplicate source-key counts | [Rows and appends](../../../docs/classification/retrieval.md#rows-and-appends); [Duplicate source keys](../../../docs/classification/retrieval.md#duplicate-source-keys) |
| Header, roles, or legacy refusal | [Header](../../../docs/classification/retrieval.md#header); [ADR-0001](../../../docs/adr/0001-embedding-table-header.md) |
| Fingerprints, `require_compatible()`, or incompatible-table refusal | [Fingerprints and compatibility](../../../docs/classification/retrieval.md#fingerprints-and-compatibility) |
| Completion markers, `open_complete()`, or incomplete-table refusal | [Completion](../../../docs/classification/retrieval.md#completion) |
| Dimension or sweep prefix tables | [Prefix tables](../../../docs/classification/retrieval.md#prefix-tables) |
| Querying or label assignment | [Exact retrieval and voting](../../../docs/classification/retrieval.md#exact-retrieval-and-voting) |
| Metric definitions, denominators, or abstentions | [Evaluation metrics](../../../docs/classification/retrieval.md#evaluation-metrics) |
| Resume ordering or validation-cache writes | [Experiment persistence](../../../docs/classification/experiments.md#reference-builds-and-resumption) |

Keep LanceDB, Arrow, column names, schema metadata keys, and the completion tag
inside `embedding_table.py`. Refuse legacy and incompatible tables; never migrate
them or add an override. Drop a table only when `derive_prefix()` recreates an
incomplete prefix of the same source at the same dimension; never drop a table
this source did not derive. Compare fingerprints generically: their setting names
and values come from `experiments/fingerprints.py`. Use the
[glossary](../../../GLOSSARY.md) vocabulary. Preserve exact cosine search, separate
vote fraction from similarity, and keep pure voting/metric functions independent
of database and model loading. Import
concrete modules directly; the package initializer does not load backends.
