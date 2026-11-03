# Datasets, records, and descriptions

Read this document when changing FoodX251 loading, reference selection, shared
records, visual-description generation, or Ollama ownership. Reference/cache
persistence is specified in [experiments](experiments.md), and vector storage in
[retrieval](retrieval.md).

## FoodX251 metadata and image batches

[`datasets/image_dataset.py`](../../src/classification/datasets/image_dataset.py) reads local class and CSV
annotation files through pandas, independently of `data_wrappers.foodx251`.
`FoodX251ImageDataset` exposes a typed Hugging Face `DatasetDict`: `train` holds
training references and `validation` holds labeled validation images.
`train_set()` returns the former; `validation_set()` returns the latter.
The legacy `test_set()` delegates to `validation_set()` for compatibility.
`test_path` and `val_path` both point at validation images. Official test files are
not read, and resulting accuracy is validation accuracy.

Class IDs map through `annot/class_list.txt`; names are capitalized with underscores
replaced by spaces. Duplicate IDs, empty taxonomy, and duplicate canonical labels
are rejected. Split CSVs preserve annotation order and produce `fname`, `class`,
`source_id`, `source_split`, and an undecoded `image` column. Paths resolve within
their split directory; identities use `foodx251:<train|val>:<relative path>`.
Duplicate identities within a split are rejected. `source_split` remains `val` for
validation records. Loading metadata does not decode images or compare contents or
paths across splits. Content hashing and exclusion artifacts remain removed.

`select_images()` applies the optional per-class cap before saved-identity
exclusions, leaving the input dataset unchanged. This order makes resumed builds
select the same first N images in annotation order. Omitted/null
`n_imgs_per_class` uses all training images; shared configuration validation accepts
only positive integers otherwise, excluding booleans. The reference builder applies
the cap only to training images, with text references and full validation unchanged.
Every class still needs a selected training reference when image references are
enabled. A changed cap needs fresh reference storage; see [experiment
contracts](experiments.md).

`iter_dataset_batches(dataset, batch_size, decode=True)` batches a prepared dataset.
`iter_batches()` combines selection and batching for `train`, `val`, or
`validation`. Both retain the final partial batch through `Dataset.iter()`.
Decoding occurs only after selection; `decode=False` removes the image column and
supports complete-cache evaluation without reading image files. Batches remain
column-oriented and ordered. Image-loading failures propagate. Experiment loops
leave decoded PIL images to garbage collection; encoder ownership of caller images
and path-opened images is defined in [encoders](encoders.md).

## Shared records

[`schemas.py`](../../src/classification/schemas.py) contains shared data contracts without
backend construction:

| Record | Meaning |
| --- | --- |
| `SourceIdentity` | `(source_kind, source_identity)`, the storage/resumption key. |
| `ImageMetadata` | Filename, canonical class, source ID, and source split. |
| `PointRecord` | Frozen reference/cache record with label, source kind/identity, value, and generated UUID. `to_payload()` stores the label under `class`. |
| `Neighbor` | Retrieved ID, label, and cosine similarity score. |
| `Prediction` | Image metadata plus prediction, vote fraction, nearest label, ordered neighbors, and per-query retrieval metrics. |
| `QueryRetrievalMetrics` | Top-five hit, precision, reciprocal rank, nDCG, AP@R, and relevant-reference count. Definitions are in [retrieval](retrieval.md). |

Keep metadata/vector associations ordered through selection, batching, storage,
retrieval, and reporting. `experiments.batches.EmbeddingBatch` carries column
metadata alongside its NumPy vector matrix; conversion to prediction rows happens
at evaluation.

## Class descriptions and Ollama

`Experiment.generate_descriptions()` owns saved vocabulary handling. Disabled
`embed_descriptions` returns without opening Ollama. A non-null `descriptions_file`
selects a read-only JSON file resolved from the working directory with `~`
expansion, independently of `output_dir`; a missing file fails without generation.
Omitted/null selects `output_dir/descriptions.json` for generation or reuse.
Indexing reads the same resolved path through `experiments.descriptions.load_descriptions`;
it never generates missing descriptions. The JSON has a `descriptions` mapping
from labels to lists of sentences. The default classification YAML shares the existing vocabulary under
`assets/classification/descriptions_v1/`. Generation followed by indexing can use
the default vocabulary path without an explicit `descriptions_file`.

[`generate_descriptors.py`](../../src/classification/llm/generate_descriptors.py) owns the
prompt and model requests. `DescriberLLM(client, model)` receives its client and
model explicitly. `DescriberLLM.generate_descriptions(classes, count)`
loops through classes and requests; `describe_class(label)` uses the shared visual
prompt. Pydantic validates the `FoodDescriptor` response, the requested class label
replaces the model's returned label, and asterisks are stripped from the sentence.
Empty or whitespace-only descriptions fail the stage. Errors and interrupts
propagate without retries or partial description files. Progress semantics are in
[tracking](tracking.md).

`llm/ollama.py` owns the `OllamaSession` context manager, which requires
`binary_path`, `host`, `startup_timeout`, and `request_timeout` from the `ollama`
section, with no environment fallback for those settings. Description generation
also requires `ollama.llm`, read before entering the session; downloading uses the
explicit requested model name. Timeouts must be positive. Startup validates an
executable before probing:
bare names resolve through PATH; explicit paths expand `~` and resolve from the
working directory. Installation or OS launch errors provide Ollama setup guidance;
there is no automatic installation.

The context manager rejects an already-served endpoint, then launches the configured
binary directly with `serve` and an inherited environment plus `OLLAMA_HOST`.
`OLLAMA_MODELS` is therefore inherited. Bounded readiness polling requests
`/api/tags` with proxy-environment trust disabled, retries transient connection or
timeout failures, and rejects invalid responses. Exit or startup failure terminates
only the process this instance started, waits, and kills it after the shutdown
timeout if necessary. No pre-existing server is stopped. `run()` finishes this
context before loading the embedder. Downloads use the same lifecycle via
[the CLI](cli.md#ollama-downloads).

## Verification

[`tests/test_classification.py`](../../tests/test_classification.py) covers miniature
local datasets, metadata-only access, identity/path validation, deterministic caps,
partial batches, cache ordering, and vocabulary handling.
[`tests/test_descriptor_lifecycle.py`](../../tests/test_descriptor_lifecycle.py)
uses fake HTTP/process backends to cover owned-server startup, cleanup, downloads,
and description errors. Routine checks never start a real service.
