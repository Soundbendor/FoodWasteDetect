# Datasets, records, and descriptions

Read this document when changing dataset loading, reference selection, shared
records, visual-description generation, or Ollama ownership. Reference/cache
persistence is specified in [experiments](experiments.md), and vector storage in
[retrieval](retrieval.md).

## Dataset loading and image batches

[`datasets/image_dataset.py`](../../src/classification/datasets/image_dataset.py)
defines the abstract `FoodImageDataset`. A subclass sets `name` and supplies
`_read_classes()` (unique canonical labels in taxonomy order) and `_read_split()`
for `train` and `validation`, returning its own source split name, an image
directory, and an ordered `fname`/`class` table. The base class exposes a typed
Hugging Face `DatasetDict` with `fname`, `class`, `source_id`, `source_split`, and
an undecoded `image` column. `train_set()` returns training references;
`validation_set()` returns the labeled evaluation split, whatever the dataset calls
it. Paths resolve within the image directory; identities use
`<name>:<source_split>:<relative path>`, and duplicates within a split are
rejected. Loading metadata does not decode images or compare contents or paths
across splits.

`DATASETS` maps each registered name to its class, and `load_dataset()` builds the
configured one. `experiments.config.DatasetName` lists the same names (lowercase
only); a test keeps the two equal.

`FoodX251ImageDataset` (`foodx251`) reads pandas tables independently of
`data_wrappers.foodx251`. Class IDs map through `annot/class_list.txt`; names are
capitalized with underscores replaced by spaces. Duplicate IDs, empty taxonomy,
and duplicate canonical labels are rejected. `annot/{train,val}_info.csv` index
`train/train_set` and `val/val_set`; identities are
`foodx251:<train|val>:<relative path>`, and `source_split` stays `val` for
validation records. The unlabeled test split is not read, so accuracy is
validation accuracy.

### Downloads

`download(root, client=None)` fetches and unpacks a dataset; the base class
raises `NotImplementedError`. Module helpers serve every subclass:
`_fetch_archive(url, root, client)` streams to `root/.downloads/<name>.part` and
renames on completion, reusing a completed archive without a request;
`_extract(archive, destination)` unpacks `.zip` files or tars with the `data`
filter, which rejects members escaping `destination`. `client` is an optional
`httpx.Client` for tests; the default follows redirects with a 60 s timeout.

`FoodX251ImageDataset.download()` returns early when `annot/class_list.txt`
exists. Otherwise it fetches `https://food-x.s3.amazonaws.com/{train,val,annot}.tar`,
extracts each into `root/<part>`, and deletes the archive. `annot` goes last so
its class list marks a complete download. The real tar layout is unverified;
the loader expects `annot/`, `train/train_set/`, and `val/val_set/`.

### Reference selection

`select_images()` applies the optional per-class cap before saved-identity
exclusions, leaving the input dataset unchanged. This order makes resumed builds
select the same first N images in annotation order. Omitted/null
`n_imgs_per_class` uses all training images; shared configuration validation accepts
only positive integers otherwise, excluding booleans. The reference builder applies
the cap only to training images, with text references and full validation unchanged.
Every class still needs a selected training reference when image references are
enabled. A changed cap needs fresh reference storage; see [experiment
contracts](experiments.md).

`iter_dataset_batches(dataset, batch_size, decode=True)` batches a prepared dataset,
such as `validation_set()` or the output of `select_images()`. It retains the final
partial batch through `Dataset.iter()`.
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
| `EmbeddingBatch` | Column metadata dictionary paired with its float32 vector matrix. |
| `ImageMetadata` | Filename, canonical class, source ID, and source split. |
| `PointRecord` | Frozen reference/cache record with label, source kind/identity, value, and generated UUID. `to_payload()` stores the label under `class`. |
| `Neighbor` | Retrieved ID, label, and cosine similarity score. |
| `Prediction` | Image metadata plus prediction, vote fraction, nearest label, ordered neighbors, and per-query retrieval metrics. |
| `QueryRetrievalMetrics` | Top-five hit, precision, reciprocal rank, nDCG, AP@R, and relevant-reference count. Definitions are in [retrieval](retrieval.md). |

Keep metadata/vector associations ordered through selection, batching, storage,
retrieval, and reporting. Conversion from `EmbeddingBatch` to prediction rows
happens at evaluation.

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

`llm/ollama.py` owns the `OllamaSession` context manager. Callers pass an
`OllamaConfig`, which validates `binary_path`, `host`, and positive finite
`startup_timeout`/`request_timeout` seconds. These settings have no environment
fallback. Before generating descriptions, `Experiment.generate_descriptions()`
also requires `ollama.llm` and `description_count`; downloads use the explicit
requested model name. Cached and disabled description paths need no Ollama settings.

Startup validates an executable before probing:
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
[`tests/test_dataset_download.py`](../../tests/test_dataset_download.py) serves
synthetic tars through `httpx.MockTransport` to cover download, loading,
idempotence, archive reuse, failed fetches, zip extraction, tar escapes, and CLI
dispatch.
[`tests/test_descriptor_lifecycle.py`](../../tests/test_descriptor_lifecycle.py)
uses fake HTTP/process backends to cover owned-server startup, cleanup, downloads,
and description errors. Routine checks never start a real service.
