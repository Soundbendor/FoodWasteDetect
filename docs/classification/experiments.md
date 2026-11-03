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
| `index_builder.py`  | `IndexBuilder`: input selection, reference resumption, encoding batches, and validation-cache preparation |
| `descriptions.py`   | Saved-vocabulary reading and ordered text-reference construction                                          |
| `evaluation.py`     | Batch evaluation, metadata-row conversion, predictions, and completeness checks                           |
| `config.py`         | Immutable Pydantic configuration models validated before CLI tracking                               |
| `batches.py`        | `EmbeddingBatch` and ordered cache reads through `cached_batches()`                                       |
| `artifacts.py`      | Shared `write_json()`                                                                                     |
| `table_metadata.py` | Pure construction, serialization, and derivation of table creation snapshots                              |
| `report_types.py`   | Pydantic index summaries and evaluation reports with typed configuration                    |
| `matryoshka.py`     | Cached Gemini dimension sweep, table prefixes, and sweep summary                                          |

`LanceDBIndex` owns ordinary database operations. The sweep's streaming creation
of derived tables is implemented directly in `matryoshka.py`. CLI parsing stays
in [cli.py](cli.md); neither experiment module has its own command-line parser.
Use concrete module imports; `__init__.py` does not import backends.

## Lifecycle and configuration

The shared CLI exposes `experiment describe`, `index`, `evaluate`, `run`, and
`matroyshka-sweep`. Preserve the existing sweep spelling. The old caption/BGE
workflows and flat-module import paths have no compatibility shims.

`Experiment(config, tracker=None)` requires a validated `ExperimentConfig`, resolves
paths, loads dataset metadata and class names, and stores the configured batch size.
Construction loads metadata even for disabled or cached description stages.
It creates directories only during writing stages. The `embedder` property
constructs one backend on first use and reuses it; complete stored references or
validation caches can avoid construction. Direct Python use is untracked by
default; [tracking](tracking.md) defines CLI run ownership.

Construct settings with `ExperimentConfig.model_validate(mapping)` and access them
through attributes. To change settings, modify a fresh `model_dump()` and validate
a new model. Orchestration and sweep helpers accept validated models.

`ExperimentConfig` and its nested models use strict field types, reject unknown
fields, validate defaults, and freeze attributes. Dataset names accept FoodX251
case-insensitively while preserving spelling. Positive integer constraints exclude
booleans; timeouts must be positive finite numbers. Model validators require at
least one reference source, output outside the raw dataset, and distinct nonempty
validation/reference table names when caching is enabled.

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

New builds prepare and validate reference inputs before creating a table or
loading the encoder. Existing builds follow this sequence:

1. Open the table, prepare the capped unbatched training selection, check class
   coverage, and prepare text references.
2. Scan all `(source_kind, source_identity)` metadata once, without a result
   limit. Initialize counts from every stored row, including duplicates and
   extra rows; use the identity set to exclude saved inputs.
3. Batch and decode only remaining images, preserving the original dataset
   splits and order. Caps apply before saved-identity exclusion.
4. Encode each text/image batch and append its strictly paired records/vectors
   once. Advance counts and progress only after successful append.
5. Optimize the reference table, including a no-work resume, then return the
   index and counts to `Experiment` for summary publication.

Image records are built directly from column-oriented batches, with strict
pairing of `class`, `source_id`, and `fname` columns. `EmbeddingBatch` is an
ordered metadata dictionary and vector matrix; row conversion belongs to
evaluation and reporting. Keep final partial batches. The dataset and encoder
[image ownership rules](data.md) apply.

Successful batches survive later failures; failed encoding batches save no
vectors. Storage failures propagate without write retries or rollback. Existing
rows and UUIDs remain intact. Optimization failure preserves the previous
summary; a rerun retries finalization without re-encoding saved inputs. Use one
writer and finish building/optimization before concurrent readers. Shared jobs
need a filesystem path visible to all nodes and separate evaluation output
directories.

### Table creation metadata

New reference and validation tables store UTF-8 JSON under the Arrow schema
metadata key `classification.build`. For fields and defaults, read
[`table_metadata.py`](../../src/classification/experiments/table_metadata.py).
Snapshots select configuration fields explicitly and remain partial mappings,
not complete `ExperimentConfig` models.

## Validation caching

`cache_validation_embeddings` defaults to false when omitted and requires a strict
boolean. Enabled caching requires a nonempty
`lancedb.validation_table` distinct from the reference table. All backends share
this path. Cache records use `validation_image`; they never join reference
searches.

Evaluation opens references first, creates or opens the cache, scans saved
identities, and completes missing validation batches. It then fetches vectors in
metadata order through `cached_batches()`. Complete caches need neither image
decoding nor a model/client. Disabled caching recomputes validation embeddings.
Validation-cache preparation does not optimize tables; evaluation does not
optimize or insert into reference tables.

Cache persistence uses the same completed-batch boundary as reference insertion.
For paid backends, restarting can repeat requests from the unfinished batch;
[encoder retry behavior](encoders.md) does not make storage transactional.

## Artifact and failure contract

`build_vector_database()` returns and writes `index.json` only after successful
optimization. It contains `table`, resolved `uri`, `source_counts`, and
effective `config`, including defaults. Evaluation attaches counts from this file only when both URI
and table match; otherwise `reference_source_counts` is null. Writers return an `IndexSummary` Pydantic model with `config: ExperimentConfig`.
Historical summaries are read only for URI/table matching and source counts;
their configuration is never reconstructed or revalidated. An absent or
mismatched summary is represented by None.

`evaluate()` and `evaluate_from_batches()` share `_execute_evaluation()`. The
latter accepts ordered supplied metadata/vector batches and lets the sweep reuse
standard prediction and report logic. `evaluation.evaluate_batches()` consumes
these batches through a supplied query callable, owns progress and completeness
checks, and returns predictions. It performs no report writes or tracking.
`Experiment` assembles an `EvaluationReport` model with typed configuration,
writes its JSON-compatible dump, then publishes it.
Evaluation covers the full labeled validation split (`source_split="val"`), with
a nonempty split and exactly one prediction per image required before saving.
This is validation accuracy, not official test accuracy. Strict metadata/vector
pairing is retained.

A successful evaluation writes only `evaluation_<UTC timestamp>/report.json`,
containing `config`, `table`, `reference_source_counts`, `predictions`, and
`metrics`. Predictions contain image metadata, majority-vote label, vote
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
manifest/table-content validation, automatic table naming, migration, rollback,
and Git/model/code revision, dependency-version, or timestamp provenance
metadata are not implemented. This includes nested text-encoder revisions; UTC
output directory names remain in use. Changing models, dimensions, prompts,
descriptions, source inputs, or caps requires fresh tables and output
directories with all enabled references rebuilt. Existing tables are reused by
identity without verifying compatibility. Preserve historical data and
collections; the separate CLIP/Qdrant experiment is unaffected.

## Cached Matryoshka sweep

`experiment matroyshka-sweep` assumes complete Gemini 3072-dimensional reference
and validation tables configured for `google/ge-v2`. It makes no embedding
requests and performs no compatibility/completeness validation. Missing cached
vectors fail instead of being regenerated.

The sweep evaluates 768, 1536, and 3072 dimensions with the configured `k` and
batch size. All six reference/validation names must be distinct. The native
dimension reads the original tables; smaller dimensions use `<source>_d768` and
`<source>_d1536`. Each missing destination is created by one streaming
`create_table()` call, with no query-result limit, retaining every row, UUID,
and row metadata value. Only vectors change to exact float32 prefixes and
fixed-size Arrow types; no extra normalization is applied. Reference switches
and image caps remain recorded but do not filter stored rows.

Existing destinations are reused if they open. Open/create/stream failures
propagate without scan, repair, overwrite, or cleanup; this is not a crash
durability guarantee. The sweep reads undecoded validation metadata and fetches
cached vectors by identity.

Each dimension gets a freshly dumped and validated `ExperimentConfig`, selected dimension and
table names, enabled caching, and an output directory under
`<output_dir>/matroyshka_sweep_<UTC>/d<dimension>`. The CLI owns one native W&B
context per dimension, with no outer run, explicit identity/group settings, or
resume checks. Standard reports have null reference counts unless a matching
index summary exists in their output directory.

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
resumes, cache reuse, failures, and publication ordering. For sweep changes use
`tests/test_matroyshka_sweep.py`; scoring is covered in
`tests/test_retrieval.py`, tracking in `tests/test_tracking.py`.

These tests establish neither real-checkpoint behavior nor cloud, GPU, HPC, or
shared-storage compatibility. Consult the [verification log](../experiments.md)
for dated test results and historical baseline failures.
