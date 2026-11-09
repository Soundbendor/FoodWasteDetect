# Storage, voting, and evaluation metrics

Read this document when changing LanceDB access, label assignment, or evaluation
definitions. [Experiments](experiments.md) owns selection, persistence orchestration,
and publication; [tracking](tracking.md) owns W&B summaries and tables.

| Module | Responsibility |
| --- | --- |
| `retrieval/lancedb.py` | `LanceDBIndex`: table lifecycle, build snapshots, prefix derivation, appends, identity/vector reads, counts, optimization, and search |
| `retrieval/voting.py` | Pure `classify()` majority vote |
| `retrieval/metrics.py` | Pure per-query ranking and aggregate classification/retrieval metrics |
| `schemas.py` | `PointRecord`, source identity, neighbor, prediction, and retrieval-metric types |

## LanceDB storage

`lancedb.uri` is a filesystem path resolved from the working directory with `~`
expansion, independently of `output_dir`. `exists()` and `open()` do not create
missing database directories. `LanceDBIndex.open(uri, table)` and
`LanceDBIndex.create(uri, table, dimension, *, config=None, metadata=None)` return an index with an open
`LanceTable`; `LanceDBIndex.exists(uri, table)` checks storage without constructing
an index. Each factory uses the local `LanceDBConnection` directly. Index methods
can therefore use their table without an unopened state or repeated guards.
Creation uses an explicit Arrow schema with
non-null string fields `id`, `class`, `source_kind`, `source_identity`, and `value`,
plus a fixed-length float32 `vector`. `mode="create", exist_ok=False` refuses
overwrites. `PointRecord` supplies UUIDs and maps its `label` to stored `class`.
When `config` is supplied to `create()`, its `vector_dimension` is set to `dimension`
and serialized under `classification.build` schema metadata.

`append()` pairs rows strictly and asks the backend
to reject bad vectors. It adds completed batches without application write
retries. Compatibility checks and transaction/rollback guarantees are outside the
current scope; follow [experiment persistence](experiments.md#reference-builds-and-resumption).

| Operation | Contract |
| --- | --- |
| `dimension` | Positive fixed-size vector width from the table schema |
| `create_prefix_table(destination, dimension)` | Reuse a readable destination or shallow-clone the source table and replace its `vector` column with L2-normalized float32 prefixes via PyLance, updating any `classification.build` snapshot's `vector_dimension` |
| `get_record_ids()` | Unlimited metadata-only scan of `(source_kind, source_identity)`; includes every stored row |
| `class_counts()` | Stream only the class column without a result limit; every source type and duplicate row counts |
| `fetch_vectors(identities)` | Select requested identities and return float32 vectors in caller order, including repeated requests; missing identities fail |
| `query(vector, k)` | Exact cosine neighbors in backend order, as `id`, `label`, and similarity `score` |

`fetch_vectors()` escapes single quotes in identity predicates and uses both kind
and identity. Empty input returns shape `(0, table dimension)`. It performs no
duplicate-row or compatibility validation; identity resumption is not content
verification.

`optimize()` delegates to LanceDB defaults. Reference build finalization owns its
invocation; ordinary evaluation never calls it. LanceDB 0.40.0 compacts files and
retains seven days of historical versions with `delete_unverified=False`; the
application supplies no custom policy.

Use one completed writer/build before concurrent readers, with a filesystem
visible to every node. Concurrent writes to the same table are unsupported.
Creation `metadata` attaches string pairs to the Arrow schema without adding row
columns; appends, optimization, and reopening preserve it. For experiment snapshots,
read [table creation metadata](experiments.md#table-creation-metadata).

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

`tests/test_retrieval.py` uses synthetic vectors and temporary real LanceDB
tables. It checks schema/row preservation, bad-vector rejection, exact cosine
ordering, negative scores, unlimited scans, ordered fetches, concurrent readers,
optimization, vote ties, abstentions, and hand-calculated ranking denominators.
`tests/test_classification.py` checks that deeper rankings do not expand saved
voting support. `tests/test_matroyshka_sweep.py` and `tests/test_tracking.py` cover
metric propagation. Historical reports remain unchanged when metrics evolve.
