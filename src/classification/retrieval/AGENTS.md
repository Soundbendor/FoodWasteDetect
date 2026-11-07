# Retrieval

`embedding_table.py` owns embedding tables: headers, roles, LanceDB access, column
names, and the record-to-row mapping; `voting.py` selects labels; `metrics.py` scores
predictions and rankings. Shared record types live in `classification.schemas`.
Experiment selection, resumption, and report publication belong to `experiments`.

| Task | Read before changing |
| --- | --- |
| Embedding tables, source-key scans, row counts, vector fetching, or optimization | [Embedding tables](../../../docs/classification/retrieval.md#embedding-tables) |
| Header, roles, or legacy refusal | [Header](../../../docs/classification/retrieval.md#header); [ADR-0001](../../../docs/adr/0001-embedding-table-header.md) |
| Dimension or sweep prefix tables | [Prefix tables](../../../docs/classification/retrieval.md#prefix-tables) |
| Querying or label assignment | [Exact retrieval and voting](../../../docs/classification/retrieval.md#exact-retrieval-and-voting) |
| Metric definitions, denominators, or abstentions | [Evaluation metrics](../../../docs/classification/retrieval.md#evaluation-metrics) |
| Resume ordering or validation-cache writes | [Experiment persistence](../../../docs/classification/experiments.md#reference-builds-and-resumption) |

Keep LanceDB, Arrow, column names, and schema metadata keys inside
`embedding_table.py`. Refuse legacy tables; never migrate them. Use the
[glossary](../../../GLOSSARY.md) vocabulary. Preserve exact cosine search, separate
vote fraction from similarity, and keep pure voting/metric functions independent
of database and model loading. Import
concrete modules directly; the package initializer does not load backends.
