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
| `config.py`         | Experiment settings                                                                                      |
| `batches.py`        | `EmbeddingBatch` type alias                                       |
| `artifacts.py`      | Shared `write_json()`                                                                                     |
| `report_types.py`   | Index summary and evaluation report models                                                                |
| `matroyshka.py`     | Cached dimension sweep, table prefixes, and sweep summary                                          |

`LanceDBIndex` owns database operations, including build-metadata snapshots,
vector-dimension inspection, and streaming creation of derived prefix tables.
CLI parsing stays in [cli.py](cli.md); neither experiment module has its own
command-line parser. Use concrete module imports; `__init__.py` does not import
backends.

## Lifecycle and configuration

The shared CLI exposes `experiment describe`, `index`, `evaluate`, `run`, and
`matroyshka-sweep`. Preserve the existing sweep spelling.

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

Vocabulary path resolution, generation/cache behavior, and Ollama lifecycle are
defined in [class descriptions and Ollama](data.md#class-descriptions-and-ollama).
`descriptions.load_descriptions()` reads saved vocabularies, and
`descriptions.text_references()` builds ordered class-name and description records.

## Reference builds and resumption

Reference switches select any nonempty combination of class names, generated
descriptions, and training images. Text identities are the class name for
`class_name` and `<class>:<description index>` for `generated_description`.
Training records use `training_image` and the dataset source ID. Values retain
the class name, description text, or image filename respectively. Per-class
training image caps and before-exclusion selection order are defined in
[data](data.md#foodx251-metadata-and-image-batches).

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

New reference and validation tables store the full `ExperimentConfig` as JSON under
the Arrow schema key `classification.build`, with `vector_dimension` set to the
stored vector size. The snapshot records creation settings and remains unchanged
on resume, even when the current configuration differs.

## Validation caching

Enabled `cache_validation_embeddings` requires a nonempty
`lancedb.validation_table` distinct from the reference table. All backends share
this path. Cache records use `validation_image`; they never join reference
searches.

Evaluation opens references first, creates or opens the cache, scans saved
identities, and completes missing validation batches. It then fetches vectors in
metadata order through `Experiment`. Complete caches need neither image
decoding nor a model/client. Disabled caching recomputes validation embeddings.
Validation-cache preparation does not optimize tables; evaluation does not
optimize or insert into reference tables.

Cache persistence uses the same completed-batch boundary as reference insertion.
For paid backends, restarting can repeat requests from the unfinished batch;
[encoder retry behavior](encoders.md) does not make storage transactional.

## Artifact and failure contract

`build_vector_database()` returns and writes `index.json` only after successful
optimization. It contains `table`, resolved `uri`, `source_counts`, and
effective `config`, including defaults. Evaluation attaches counts from this file
only when both URI and table match; otherwise `reference_source_counts` is null.
Writers return an `IndexSummary`. Historical summaries supply only URI/table
matching and source counts.

`evaluate()` and `evaluate_cached()` share `_execute_evaluation()`. The latter
reads existing cached vectors regardless of the cache flag, without completing or
mutating the cache. `Experiment` prepares validation batches; it opens or completes
the cache and constructs the metadata iterator eagerly, then fetches vectors lazily.
Ordinary evaluation prepares references first; cached-only evaluation prepares the
cache first. Missing vectors fail during consumption after reference preparation. `evaluation.evaluate_batches()` consumes
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

`write_json()` serializes finite JSON with a trailing newline to a sibling
`.tmp` file and replaces the destination. Tracking uploads reports only after
successful local writing. Loading, encoding, storage, and interruption failures
propagate without partial evaluation reports, exclusions, application-level
retries, or cleanup of partial reference tables.

Artifact compatibility checks, content hashes, automatic table naming, migrations,
and Git/model/dependency provenance metadata are intentionally omitted. Existing
tables are reused by identity without verifying compatibility; changing models,
dimensions, prompts, descriptions, source inputs, or caps requires fresh tables
and output directories.

## Cached Matryoshka sweep

`experiment matroyshka-sweep` evaluates cached vector prefixes for any model ID.
This is a cached prefix experiment, not a guarantee of Matryoshka quality.
Set `sweep_dimensions` to a nonempty list of unique positive integers in the
order to evaluate them. Existing configurations must supply this field for the
sweep; other stages and historical snapshots may omit it. The shipped Gemini
configuration uses 768, 1536, and 3072. The source reference and validation
vector fields must be fixed-size lists of the same positive width; all targets
must fit that stored width, including when the sources are already truncated.
The sweep makes no embedding requests and performs no completeness validation. Missing cached
vectors fail instead of being regenerated. Evaluation uses the configured `k` and
batch size.

Both source names and all derived names must be distinct, even if the stored
width is not requested. A target equal to the stored width uses the original
tables; smaller targets use `<source>_d<dimension>`. Each missing destination is created by shallow-cloning
the source table via `clone_table()` and replacing its `vector` column through a PyLance batch UDF,
retaining every row, UUID, and row metadata value without copying unmodified columns. Only vectors change to L2-normalized float32 prefixes and
fixed-size Arrow types. Reference switches
and image caps remain recorded but do not filter stored rows.

Derived tables retain the source configuration with only `vector_dimension`
changed. Other schema metadata is preserved; absent snapshots remain absent.

Existing destinations are reused if they open. Open/create/stream failures
propagate without scan, repair, overwrite, or cleanup; this is not a crash
durability guarantee. The sweep delegates cached validation preparation to `Experiment.evaluate_cached()`.

Each dimension enables validation caching and uses its own table names and output
directory under `<output_dir>/matroyshka_sweep_<UTC>/d<dimension>`.
The CLI owns one native W&B context per dimension, with no outer run, explicit
identity/group settings, or
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
