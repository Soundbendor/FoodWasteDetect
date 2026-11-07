# Embedding tables, voting, and evaluation metrics

Read this document when changing embedding tables, label assignment, or evaluation
definitions. [Experiments](experiments.md) owns selection, persistence orchestration,
and publication; [tracking](tracking.md) owns W&B summaries and tables. Terms
follow the [glossary](../../GLOSSARY.md).

| Module | Responsibility |
| --- | --- |
| `retrieval/embedding_table.py` | `EmbeddingTable`: headed table lifecycle, role checks, fingerprint and dimension compatibility, completion markers, build operations (open or create, missing keys, mark complete), appends, record and vector reads, counts, optimization, search, and prefix-table derivation |
| `retrieval/voting.py` | Pure `classify()` majority vote |
| `retrieval/metrics.py` | Pure per-query ranking and aggregate classification/retrieval metrics |
| `schemas.py` | `PointRecord`, `SourceKey`, neighbor, prediction, and retrieval-metric types |

## Embedding tables

`retrieval/embedding_table.py` is the only module that knows storage column names,
schema metadata keys, the record-to-row mapping, or LanceDB and Arrow calls. Callers,
including the sweep, use `EmbeddingTable` operations and never import LanceDB or
Arrow. The design decision is recorded in
[ADR-0001](../adr/0001-embedding-table-header.md).

`lancedb.uri` is a filesystem path resolved from the working directory with `~`
expansion, independently of `output_dir`. `EmbeddingTable.exists(uri, name)` and
`open()` do not create missing database directories.

| Factory | Contract |
| --- | --- |
| `open(uri, name, role)` | Open an existing table; a missing directory raises `FileNotFoundError`, a missing table propagates LanceDB's `ValueError` |
| `open_complete(uri, name, role, *, expected, encoder=None, dimension=None)` | Open an existing table for reading, as `open()`, require any given encoder fingerprint and dimension ([compatibility](#fingerprints-and-compatibility)), then require it to be [complete](#completion); `expected` is a callable invoked only for an incomplete table |
| `create(uri, name, role, dimension, *, snapshot, encoder, inputs=None)` | Create an empty table with a header, its fingerprints, and the verbatim creation snapshot; an existing name is refused |
| `open_or_create(uri, name, role, dimension, *, snapshot, encoder, inputs=None)` | Resume an existing table under `role` whose fingerprints equal `encoder` and `inputs`, or create one; `dimension` is a callable invoked only when creating, and `snapshot` is ignored for an existing table |

`role` is `"reference"` or `"validation_cache"`.

### Header

Every table created by this module records a store-owned header in its schema
metadata. The header is written once and never rewritten:

- `schema_version` (currently `SCHEMA_VERSION = 2`);
- `role`: reference table or validation cache;
- `dimension`: the fixed vector width;
- `encoder` and `inputs`: the [fingerprints](#fingerprints-and-compatibility);
  `inputs` is present on reference tables only;
- for prefix tables, `source_table` and `source_dimension`.

The creation snapshot is stored verbatim under its own key and never parsed or
rewritten; the retired `classification.build` key is no longer written or read.
`role`, `dimension`, `encoder_fingerprint`, `inputs_fingerprint` (`None` for a
validation cache), `prefix_source` (a `PrefixSource(table, dimension)` or `None`),
and `creation_snapshot` read these values back.

Opening a table fails with `LegacyTableError` when the header is absent, malformed,
records another schema version, lacks a creation snapshot, disagrees with the
stored columns or vector width, or has an inputs fingerprint that does not match
its role. Schema version 2 added the fingerprints, so every table written under
version 1, before fingerprints existed, is refused as legacy. The message tells the researcher to rebuild under a
new table name and cites ADR-0001; there is no migration. Opening a valid table under
the other role fails with `EmbeddingTableError`. An incomplete table fails with
`IncompleteTableError` where completion is required. All three are `ValueError`
subclasses; `LegacyTableError`, `IncompatibleTableError`, and
`IncompleteTableError` subclass `EmbeddingTableError`. The completion marker is not
part of the header, so it does not change the schema version.

### Fingerprints and compatibility

Each table records two fingerprints in its header when it is created. They are
flat mappings from a setting name to a string, integer, boolean, or null value.
The store compares them generically; `experiments/fingerprints.py` computes them
from the experiment configuration
([experiments](experiments.md#fingerprints-and-incompatible-tables)), naming each
setting by its configuration path:

- **Encoder fingerprint**, on every table: `embedding.model_id`,
  `embedding.truncate_dim`, and `embedding.vertex.classification_prompt`.
- **Inputs fingerprint**, on reference tables only: `dataset.name`,
  `embed_images`, `embed_class_labels`, `embed_descriptions`, `n_imgs_per_class`,
  and `vocabulary_sha256`, the SHA-256 of the loaded vocabulary as canonical
  JSON (sorted class keys, description order kept, no insignificant whitespace,
  UTF-8), or null when descriptions are disabled.

`create()` refuses a reference table without an inputs fingerprint and a
validation cache with one, with `EmbeddingTableError`, before writing anything.

`require_compatible(*, encoder=None, inputs=None, dimension=None)` compares the
stored values with each argument that is given. Fingerprints are compared
setting by setting, and a setting absent from one side compares as null. Any
difference raises `IncompatibleTableError`, whose message lists every differing
setting as `<setting>: stored <value>, required <value>` and tells the researcher
to use a new table name. The `differences` attribute maps each setting to its
stored and required values; a dimension mismatch appears as `dimension`. There
is no override.

- `open_or_create()` requires both fingerprints to equal the current ones before
  resuming an existing table, so one table never mixes vectors or reference inputs
  from two configurations.
- `open_complete()` checks any given encoder fingerprint and dimension before
  completion, so an incompatible table fails without a key scan.
- Settings outside the fingerprints never block: device, batch size, Vertex
  project and location, concurrency, attempts, request timeout, dataset root,
  output directory, `k`, validation caching, and table names.

### Rows and appends

Rows hold non-null strings `id`, `class`, `source_kind`, `source_identity`, and
`value`, plus a fixed-length float32 `vector`. `PointRecord` supplies UUIDs and
exposes `source_key`; it does not know column names. The module maps `label` to the
stored `class`.

`append()` checks record/vector counts, pairs rows strictly, and asks the backend
to reject bad vectors. It adds completed batches without application write
retries. Compatibility is checked when a table is opened, not per append;
transaction and rollback guarantees are outside the current scope. Follow
[experiment persistence](experiments.md#reference-builds-and-resumption).

Builds resume and finish through two operations over expected source keys:

- `missing(keys)` performs one metadata-only scan and returns the requested keys
  that are absent, each once, in first-request order.
- `mark_complete(expected)` raises `IncompleteTableError` stating how many of the
  expected keys are missing when any are absent, without optimizing or marking.
  Extra stored rows are allowed. After the check passes, it removes any earlier
  marker, reference tables call `optimize()`, and the
  [completion marker](#completion) is persisted against the resulting version.
  Validation caches are never optimized. The role comes from the header.

### Completion

A table is complete when its completion marker, a LanceDB version tag, points at
the table's current version. `is_complete` reads this without scanning rows. Any
later write, such as `append()` or a delete, moves the version past the marker,
so a changed table is never reported complete. `optimize()` clears the marker
before compacting. Tables written before completion markers existed have none: they open
normally and are incomplete. Marking them complete needs no rebuild.

What complete means per role:

- **Reference table**: every expected source key (each prepared text record and
  selected training image) is present, and the table was optimized before it was
  marked.
- **Validation cache**: every validation-split `validation_image` key is present.
  Caches are marked without optimizing.
- **Prefix table**: the streaming copy from a complete source finished and its
  row count was verified to equal the source's.

`open_complete(uri, name, role, *, expected=None)` opens a complete table without
calling `expected` or scanning stored keys. For an incomplete table it calls
`expected`, performs one `missing()` scan, and raises `IncompleteTableError`
stating how many of the expected keys are missing and how to finish the table:
rerun `index` for a reference table, evaluate with validation caching enabled
for a validation cache, or rerun the sweep for a prefix table. Without
`expected`, it refuses an incomplete table without counting, and the error's
`missing` is `None`. The count is zero when every key is stored but the build
never reached `mark_complete()`. Computing the expected keys can itself fail,
for example when reference inputs no longer cover every class; that error
propagates instead.

| Read operation | Contract |
| --- | --- |
| `class_counts()` | Stream only the class column without a result limit; every source kind and duplicate row counts |
| `source_counts()` | Stream only the source-kind column without a result limit; every stored row, including duplicate source keys, counts |
| `rows()` | Every stored `PointRecord` with its float32 vector, in storage order; for inspection and small tables |
| `fetch_vectors(keys)` | Validation caches only. Return float32 vectors in caller order, including repeated requests; missing keys raise `KeyError` |
| `query(vector, k)` | Reference tables only. Exact cosine neighbours in backend order, as `id`, `label`, and similarity `score` |

`query()` on a validation cache and `fetch_vectors()` on a reference table raise
`EmbeddingTableError`. `fetch_vectors()` escapes single quotes in key predicates
and matches both kind and identity. Empty input returns shape `(0, dimension)`. It
performs no duplicate-row validation; source-key resumption is not content
verification.

### Prefix tables

`derive_prefix(destination, dimension, *, batch_size)` returns a complete prefix
table in the same database:

- The source must be [complete](#completion); otherwise `IncompleteTableError`
  is raised before anything is written. A prefix wider than the source dimension
  raises `EmbeddingTableError`, also before any write. A prefix equal to the
  source dimension copies vectors unchanged.
- An existing destination is opened under the source's role. It is **reused**
  without copying when it is complete, has the source's row count, records the
  source's fingerprints, and is a prefix of this source at the source's current
  dimension, at the requested dimension. It is **dropped and recreated** when it
  matches in the same way but is incomplete, as a crash between creation and
  marking, or any later write, leaves it, or when its row count differs because
  the source was completed again with other rows; a prefix holds only derived
  data, so no paid requests repeat.
- Any other existing destination is **refused and left unchanged**: a legacy
  table raises `LegacyTableError`; one of another role, one that is not a
  prefix table, or one derived from another source, from this source at
  another source dimension, or at another dimension raises
  `EmbeddingTableError`; one recording other fingerprints raises
  `IncompatibleTableError`, checked first, even when it is incomplete. The
  destination is never dropped in these cases.
- A new prefix is written by one streaming `create_table()` call, with no
  query-result limit, retaining every row, record ID, and row metadata value.
  Only vectors change to exact float32 prefixes; no normalization is applied.
  The prefix is marked complete only after its row count equals the source's.
  A different count raises `IncompleteTableError` (its `missing` is the source
  row count minus the copied row count) and leaves the prefix incomplete, so a
  rerun recreates it.
- The new header keeps the source's role and fingerprints, records the prefix
  dimension, and records the source table and source dimension. A prefix of a
  prefix records its immediate source. The creation snapshot is copied verbatim,
  so it is identical to the source's. Because fingerprints are inherited, a
  reference prefix and a validation-cache prefix of the same width derived from
  compatible sources are compatible with each other.

Open, create, and stream failures propagate without scan, repair, overwrite, or
cleanup; recreation happens only on the next `derive_prefix()` call. This is not
a crash durability guarantee: there is one writer, and no locking or transaction
protects a concurrent derivation.

`optimize()` delegates to LanceDB defaults. Reference builds invoke it only
through `mark_complete()` of an incomplete table; validation caches, complete
tables, and evaluation never call it. The
recorded LanceDB 0.40.0 policy compacts files and retains seven days of historical
versions with `delete_unverified=False`; the application supplies no custom policy.
See the [verification log](../experiments.md#unified-experiment-pipeline-november-2-2026)
for that version's observed behavior. Appends and optimization preserve the header
and creation snapshot. LanceDB's cleanup refuses to remove a tagged version, so
`optimize()` and `mark_complete()` delete any marker first; the
[version-tag check](../experiments.md#completion-markers-2026-11-06) records the
observed tag behavior.

Use one completed writer/build before concurrent readers, with a filesystem
visible to every node. Concurrent writes to the same table are unsupported.
The separate CLIP/YOLO Qdrant backend remains outside this package; rebuild new
references instead of migrating or changing old Qdrant collections.

## Exact retrieval and voting

Search explicitly uses cosine distance and `bypass_vector_index()`; returned
similarity is `1 - distance`. Retain negative similarities. The application does
not add embedding normalization.

Before evaluation, stream reference class counts once. For each query let `R` be
the number of stored rows matching its true class and retrieve `max(k, 5, R)`
neighbors. `k` must be a positive integer excluding booleans. Preserve backend
ranking order. Deeper rows support ranking metrics, then are discarded; voting
and saved `neighbors` use only the first `k` returned rows.

Every support row contributes one vote, including repeated labels and duplicate
rows. Ties resolve by the label's best similarity, then lexical class name.
`vote_fraction` is the winning vote count divided by the actual returned support
length; it is not calibrated confidence. Nearest-neighbor prediction uses the
first row. Empty retrieval yields label/nearest `None` and vote fraction zero;
it counts as incorrect in aggregate accuracy.

## Evaluation metrics

Metrics apply equally to local encoders, cached Vertex evaluation, and dimension
sweeps. They score the full labeled validation split. Per-query metrics live in
prediction `retrieval_metrics`; aggregate scalars appear in reports, W&B summaries,
and sweep summary rows.

### Ranking

Relevance is binary equality between a retrieved row's label and the query's true
class. A rank is a vector position, not a distinct class. Let `S` be the actual
first-k support and use one-based ranks.

| Per-query field | Definition | Aggregate field |
| --- | --- | --- |
| `top5_hit` | Any relevant vector in the first five positions | `top5_hit_rate` |
| `precision_at_k` | Relevant rows in `S` divided by `len(S)` | `retrieval_precision_at_k` |
| `reciprocal_rank_at_k` | Reciprocal of the first relevant rank within `S`, else zero | `mrr_at_k` |
| `ndcg_at_k` | Binary DCG on `S`, discount `1/log2(rank+1)`, divided by ideal DCG with `min(k,R)` relevant rows | `ndcg_at_k` |
| `average_precision_at_r` | Sum of Precision@i at relevant ranks through rank `R`, divided by `R` | `map_at_r` |
| `relevant_reference_count` | `R`, including every stored reference type and duplicate row | No separate aggregate |

Empty retrieval and `R=0` produce zero ranking scores. Aggregate effectiveness
metrics average across **every validation query**, including abstentions; they do
not average only successful retrievals. The top-five hit rate remains distinct
from majority-vote accuracy and from a top-five distinct-class prediction.

### Classification

`evaluated` counts predictions. `majority_vote_accuracy` compares the voted label
with truth; `nearest_neighbor_accuracy` compares the first neighbor's label.
Abstentions remain in both accuracy denominators.

Precision, recall, and F1 use majority-vote predictions only. Macro averages cover
the union of true and predicted labels, excluding `None`, independently of the
per-class display list. Macro F1 averages each class's F1 directly. Micro metrics
pool TP/FP/FN; an abstention adds a true-class FN and no predicted-class count.
Undefined ratios are zero.

The aggregate fields are `precision_macro`, `recall_macro`, `f1_macro`,
`precision_micro`, `recall_micro`, and `f1_micro`. `per_class` follows the supplied
display-class order and contains `support`, `correct`, `recall`, `precision`, and
`f1`. Displayed recall is null for zero-support classes; precision/F1 remain zero
when undefined. Removing a class from the display list does not alter aggregates.

### Cosine diagnostics and empty inputs

`mean_top1_cosine_similarity` averages each qualifying query's first score.
`mean_support_cosine_similarity` first averages the saved support per query, then
averages those means equally. Queries with no neighbors are excluded from these
two diagnostics; both are null when none qualify. Negative scores remain present.

The pure metric function permits empty input: accuracies and cosine means are
null, while classification/ranking ratios are zero. Experiment evaluation rejects
an empty validation split before publishing a report.

## Verification

`tests/test_embedding_table.py` uses synthetic vectors and temporary real LanceDB
tables through the `EmbeddingTable` interface. It checks header round trips,
legacy refusal (including schema-version-1 headers), role checks, fingerprint
round trips, inheritance, and mismatch refusal on resume and on open for
evaluation, row preservation, bad-vector rejection, exact cosine
ordering, negative scores, unlimited scans, ordered fetches, concurrent readers,
completion markers (persistence across reopen, invalidation by appends, refusal
with a missing count, tables without markers), optimization, and prefix
derivation: refusal of incomplete sources, row-count-verified completion, reuse
without copying, recreation after a crash or later write, and refusal, without
changes, of legacy, non-prefix, other-source, other-dimension, other-role, and
other-fingerprint destinations. `tests/test_matroyshka_sweep.py` checks the
same reuse, recreation, and refusal through the sweep entry point. `tests/test_retrieval.py` checks
vote ties, abstentions, and hand-calculated ranking denominators.
`tests/test_classification.py` checks that deeper rankings do not expand saved
voting support. `tests/test_matroyshka_sweep.py` and `tests/test_tracking.py` cover
metric propagation. Historical reports remain unchanged when metrics evolve.
