# Whole-image experiments

This baseline classifies FoodX251 images by embedding pixels and retrieving
labeled references. It runs independently of YOLO localization and InternVL
captioning. The selected project architecture remains the separate CLIP/crop
pipeline described in the root guide.

Read this document when changing experiment stages, persistence, reports, or
cached sweeps. For backend behavior use [encoders](encoders.md); for dataset
selection and image ownership use [data](data.md); for scoring use
[retrieval](retrieval.md). Historical checks and limitations are recorded in
[the experiment verification log](../experiments.md).

## Ownership

| Module              | Responsibility                                                                                            |
| ------------------- | --------------------------------------------------------------------------------------------------------- |
| `classification.py` | `Experiment`: configuration, lazy encoder, stages, summaries, and report publication                      |
| `index_builder.py`  | `IndexBuilder`: input selection, encoding batches, reference-table builds, and validation-cache builds    |
| `descriptions.py`   | Saved-vocabulary reading and ordered text-reference construction                                          |
| `evaluation.py`     | Batch evaluation, metadata-row conversion, predictions, and completeness checks                           |
| `config.py`         | Experiment settings                                                                                      |
| `batches.py`        | `EmbeddingBatch` type alias                                       |
| `artifacts.py`      | Shared `write_json()`                                                                                     |
| `report_types.py`   | Index summary and evaluation report models                                                                |
| `matryoshka.py`     | Cached dimension sweep policy, derived table names, and sweep summary                              |

`EmbeddingTable` ([embedding tables](retrieval.md#embedding-tables)) owns every
table operation, including the streaming derivation of prefix tables for the sweep;
`matryoshka.py` does not import LanceDB or Arrow. Terms follow the
[glossary](../../GLOSSARY.md). CLI parsing stays
in [cli.py](cli.md); neither experiment module has its own command-line parser.
Use concrete module imports; `__init__.py` does not import backends.

## Lifecycle and configuration

The shared CLI exposes `experiment describe`, `index`, `evaluate`, `run`, and
`matroyshka-sweep`. Preserve the existing sweep spelling. The old caption/BGE
workflows and flat-module import paths have no compatibility shims.

`Experiment(config: ExperimentConfig, tracker=None)` resolves paths, loads dataset
metadata and class names, and stores the configured batch size.
Construction loads metadata even for disabled or cached description stages.
It creates directories only during writing stages. The `embedder` property
constructs one backend on first use and reuses it; complete stored references or
validation caches can avoid construction. Direct Python use is untracked by
default; [tracking](tracking.md) defines CLI run ownership.

Field definitions and defaults live in
[`config.py`](../../src/classification/experiments/config.py). Runtime data and
backend checks remain separate; service requirements belong to
[descriptions](data.md#class-descriptions-and-ollama) and [Vertex](encoders.md#vertex).

`run()` calls `generate_descriptions()`, `build_vector_database()`, then
`evaluate()`. The owned Ollama context closes before embedding begins. The
`describe` and `index` CLI stages use those named methods directly.

### Description inputs

For vocabulary paths, generation/cache behavior, and server lifecycle, read
[class descriptions and Ollama](data.md#class-descriptions-and-ollama).
`DescriberLLM` owns the per-class generation loop; the experiment owns
saved-file handling and the server context.

`Experiment` resolves the vocabulary path once. Both generation and indexing
read saved vocabularies through `descriptions.load_descriptions()`. Indexing
accepts an existing explicit or default `output_dir/descriptions.json` file and
never starts Ollama. A missing explicit file fails without fallback or
generation. The `descriptions.text_references()` function builds class-name and
description records from an in-memory mapping, preserving class and sentence
order.

## Reference builds and resumption

Reference switches select any nonempty combination of class names, generated
descriptions, and training images. Text identities are the class name for
`class_name` and `<class>:<description index>` for `generated_description`.
Training records use `training_image` and the dataset source ID. Values retain
the class name, description text, or image filename respectively.

Use the full official training split by default. The optional image cap selects
the first N images per class in annotation order, independently of batch size.
It affects images only. Every class must have a selected training reference when
images are enabled; selected image decoding can still fail later. Compare
complete retrieval strategies: changing sources changes reference counts.

`IndexBuilder.build_reference_table()` sequences `EmbeddingTable` operations on a
reference table; it makes no other storage calls and never mutates the experiment
configuration. New and resumed builds follow one flow:

1. Prepare and validate inputs: the capped unbatched training selection, class
   coverage, and text references (which reads the vocabulary). The expected
   source keys are every prepared text record and selected training image. An
   empty expected set fails before any table exists. Validation failures happen
   before a table is created or the encoder is constructed.
2. `open_or_create(..., "reference", ...)` opens the table, or creates it with a
   header and creation snapshot when absent. Opening a legacy table, or a
   validation cache under the reference name, fails. The dimension is supplied
   lazily as `embedder.dimension` and computed only when creating, so resuming
   never constructs the encoder for that reason.
3. A table already [complete](retrieval.md#completion) is returned after this
   verification: no missing-key scan, no optimization, no encoder, and no write.
4. Otherwise `missing()` scans all source keys once, without
   a result limit, and returns the expected keys that are absent. Only those text
   records and images are embedded, preserving the original dataset splits and
   order. Caps apply before filtering by missing keys. A table with no missing
   keys never constructs the encoder.
5. Encode each text/image batch and append its strictly paired records/vectors
   once. Advance progress only after a successful append.
6. `mark_complete(expected)` verifies that every expected key is present,
   failing with `IncompleteTableError` and the number of missing keys otherwise,
   then optimizes the reference table and marks it complete. A resume with no
   missing keys, such as a table written before completion markers existed or one
   whose optimization failed, is verified, optimized, and marked without
   re-encoding. The builder returns the table to `Experiment` for summary
   publication.

Image records are built directly from column-oriented batches, with strict
pairing of `class`, `source_id`, and `fname` columns. `EmbeddingBatch` is an
ordered metadata dictionary and vector matrix; row conversion belongs to
evaluation and reporting. Keep final partial batches. The dataset and encoder
[image ownership rules](data.md) apply.

Successful batches survive later failures; failed encoding batches save no
vectors. Storage failures propagate without write retries or rollback. Existing
rows and UUIDs remain intact. Completion or optimization failure preserves the
previous summary and leaves the table incomplete; a rerun retries finalization
without re-encoding saved inputs. Use one
writer and finish building/optimization before concurrent readers. Shared jobs
need a filesystem path visible to all nodes and separate evaluation output
directories.

### Creation snapshot

New reference tables and validation caches store the full `ExperimentConfig` JSON
as their creation snapshot, beside the store-owned
[header](retrieval.md#header). The snapshot is the configuration exactly as
supplied; the configuration has no dimension setting, and the dimension comes from
the table header (`EmbeddingTable.dimension`). The snapshot remains unchanged on
resume, even when the current configuration differs, and prefix tables copy it
verbatim. Tables written before the header existed are refused as legacy and must
be rebuilt under a new name ([ADR-0001](../adr/0001-embedding-table-header.md)).

## Validation caching

Enabled `cache_validation_embeddings` requires `lancedb.validation_table`; the
configuration enforces the [validation-table rule](cli.md#validation-table) at
load time. All backends share this path. Cache records use `validation_image`;
they never join reference searches.

Evaluation opens the complete reference table first, then calls
`IndexBuilder.build_validation_cache()`. It uses the same operations as a
reference build on a validation cache: `open_or_create(..., "validation_cache",
...)` with a lazy dimension; an already complete cache is returned without a
scan. Otherwise it runs `missing()` over every validation-split
`validation_image` key, appends the missing batches, and calls
`mark_complete(expected)`, which marks the cache complete and never optimizes it.
Completion fails with the number of missing keys before any vectors are fetched.
Evaluation with caching enabled therefore leaves a complete cache that later
cached evaluations and sweeps can open; it also completes a filled cache written
before completion markers existed without re-encoding. `Experiment` then fetches
vectors in metadata order. Complete caches need neither image decoding nor a model/client.
Disabled caching recomputes validation embeddings. Validation caches are never
optimized; evaluation does not optimize or insert into reference tables.

Cache persistence uses the same completed-batch boundary as reference insertion.
For paid backends, restarting can repeat requests from the unfinished batch;
[encoder retry behavior](encoders.md) does not make storage transactional.

## Artifact and failure contract

`build_vector_database()` returns and writes `index.json` only after the
reference table is marked complete and optimized. It contains `table`, resolved
`uri`, `dimension` (the table's stored vector width), `source_counts` (read from
the table with `EmbeddingTable.source_counts()`, duplicates included), and the
unmodified `config`, including defaults. Writers return an `IndexSummary`.
Evaluation never reads `index.json`: a missing, stale, mismatched, or malformed
summary has no effect on reports. Before evaluating, `evaluate()` and
`evaluate_cached()` read `reference_source_counts` from the reference table with
`EmbeddingTable.source_counts()`, which counts every stored row, duplicates included.
The field is always present and never null.

`evaluate()` and `evaluate_cached()` share `_execute_evaluation()`. The latter
reads existing cached vectors regardless of the cache flag, without completing or
mutating the cache.

Both open the reference table with `EmbeddingTable.open_complete()`. An
incomplete reference table, such as one left by an interrupted `index` run, fails
with `IncompleteTableError` stating how many expected reference source keys are
missing; the expected keys come from `IndexBuilder.expected_reference_keys()`,
which prepares reference inputs without decoding images or constructing the
encoder. Rerun `index` to resume and mark it. `evaluate_cached()` opens the
validation cache the same way, against `IndexBuilder.expected_validation_keys()`, before
it prepares the reference table or fetches any vector. Fill an incomplete cache
by evaluating with `cache_validation_embeddings` enabled. `Experiment` prepares validation batches; it opens or completes
the cache and constructs the metadata iterator eagerly, then fetches vectors lazily.
Evaluation opens the reference table with role `reference` and the cache with role
`validation_cache`, so a table opened under the wrong role fails. Ordinary
evaluation prepares references first; cached-only evaluation prepares the cache
first. `evaluation.evaluate_batches()` consumes
these batches through a supplied query callable, owns progress and completeness
checks, and returns predictions. It performs no report writes or tracking.
`Experiment` assembles an `EvaluationReport` model with typed configuration,
writes its JSON-compatible dump, then publishes it.
Evaluation covers the full labeled validation split (`source_split="val"`), with
a nonempty split and exactly one prediction per image required before saving.
This is validation accuracy, not official test accuracy. Strict metadata/vector
pairing is retained.

A successful evaluation writes only `evaluation_<UTC timestamp>/report.json`,
containing `config`, `table`, `dimension` (the reference table's stored vector
width), `reference_source_counts`, `predictions`, and `metrics`.
Predictions contain image metadata, majority-vote label, vote
fraction, nearest label, first-k neighbors, and per-query retrieval metrics.
[Retrieval](retrieval.md) defines the denominators and abstention behavior.
Reports have no separate prediction/metric files, manifest, prediction `value`,
status/error fields, or fixed retrieval settings.

`write_json()` serializes finite JSON with a trailing newline to a sibling
`.tmp` file and replaces the destination. Tracking uploads reports only after
successful local writing. Loading, encoding, storage, and interruption failures
propagate; there are no partial evaluation reports, exclusions,
application-level retries, or cleanup of partial reference tables. Backend
request retries are separately documented in [encoders](encoders.md).
[Tracking](tracking.md) defines loop-owned progress bars and W&B lifecycle
details.

Keep the explicitly deferred scope: artifact compatibility checks, hashes,
manifest/table-content validation, automatic table naming, rollback,
and Git/model/code revision, dependency-version, or timestamp provenance
metadata are not implemented. This includes nested text-encoder revisions; UTC
output directory names remain in use. Changing models, dimensions, prompts,
descriptions, source inputs, or caps requires fresh tables and output
directories with all enabled references rebuilt. Existing tables are reused by
source key without verifying encoder or input compatibility; tables without a
valid header are refused, never migrated. Preserve historical data and
collections; the separate CLIP/Qdrant experiment is unaffected.

## Cached Matryoshka sweep

`experiment matroyshka-sweep` evaluates cached vector prefixes for any model ID.
This is a cached prefix experiment, not a guarantee of Matryoshka quality.
Set `sweep_dimensions` to a nonempty list of unique positive integers in the
order to evaluate them. Existing configurations must supply this field for the
sweep; other stages and historical snapshots may omit it. The shipped Gemini
configuration uses 768, 1536, and 3072. The sweep opens the reference table
and the validation cache under their roles and requires their header dimensions to
match before anything is derived; all targets must fit
that stored width, including when the sources are already truncated.
The sweep makes no embedding requests. Each per-dimension cached evaluation
requires its reference table and validation cache to be
[complete](retrieval.md#completion); an incomplete table fails with its missing
count instead of being regenerated. Prefix tables of complete sources are marked
complete when derived, so a source changed after completion yields incomplete
prefixes that evaluation refuses. Evaluation uses the configured `k` and
batch size.

Configured `sweep_dimensions` require a validation table under the
[validation-table rule](cli.md#validation-table). The sweep owns experiment policy
only: target dimensions, `<source>_d<dimension>` naming, distinct-name checks,
pairing the reference table with the validation cache, and the loop over
dimensions. Beyond the validation-table rule, both source names and all derived names must
be distinct, even if the stored width is not requested. A target equal to the
stored width uses the original tables; each smaller target calls
`EmbeddingTable.derive_prefix()` on both sources. The
[prefix-table contract](retrieval.md#prefix-tables) defines how prefix tables are
copied, reused, and given headers that record their source table and source
dimension. Reference switches and image caps remain recorded
but do not filter stored rows. The sweep delegates cached validation preparation
to `Experiment.evaluate_cached()`.

Each per-dimension configuration enables validation caching and uses its own
table names and output directory under
`<output_dir>/matroyshka_sweep_<UTC>/d<dimension>`. It leaves
`embedding.truncate_dim` as configured, since the sweep embeds nothing, and
records no dimension; each report's `dimension` comes from that dimension's
reference table. Prefix tables keep the source's creation snapshot verbatim.
The CLI owns one native W&B context per dimension, with no outer run, explicit
identity/group settings, or
resume checks. Each dimension's report and W&B summary carry the source counts of
that dimension's reference table, which a prefix table copies row for row from its
source.

Only complete success writes `summary.json`, with dimensions, tables, output
directories, aggregate metrics, and returned W&B run IDs. Reports from earlier
dimensions survive later failures. Production sweeps require separate compute
authorization; routine checks use synthetic CPU tables and mocked tracking.

## Verification

For evaluation computation, use `tests/test_evaluation.py` with synthetic
batches and query callables. For saved-vocabulary/reference construction, use
`tests/test_descriptions.py`. For orchestration changes, use
`tests/test_classification.py` with synthetic encoders/images, mocked Ollama,
and temporary LanceDB tables. It covers selection, pairing, partial/complete
resumes, completion refusal and marking per role, cache reuse, failures, and
publication ordering. For sweep changes use
`tests/test_matroyshka_sweep.py`; embedding table behavior is covered in
`tests/test_embedding_table.py`, scoring in `tests/test_retrieval.py`, and
tracking in `tests/test_tracking.py`.

These tests establish neither real-checkpoint behavior nor cloud, GPU, HPC, or
shared-storage compatibility. Consult the [verification log](../experiments.md)
for dated test results and historical baseline failures.
