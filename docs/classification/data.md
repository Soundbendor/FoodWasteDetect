# Datasets, records, and descriptions

Read this document when changing dataset loading, reference selection, shared
records, visual-description generation, or Ollama ownership. Reference/cache
persistence is specified in [experiments](experiments.md), and vector storage in
[retrieval](retrieval.md).

## Dataset loading and image batches

[`datasets/base.py`](../../src/classification/datasets/base.py)
defines the abstract `FoodImageDataset`. Each dataset has its own loader module
under `datasets/`, and `datasets/registry.py` registers them. A subclass sets
`name` and supplies `_read_classes()` (unique canonical labels in taxonomy order) and `_read_split()`
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

`Food101ImageDataset` (`food101`) reads `meta/classes.txt` for the taxonomy and
`meta/{train,test}.txt` for `<class>/<stem>` entries under `images/`, appending
`.jpg`. Labels use the same canonicalization as FoodX-251; a split entry whose
class is missing from `classes.txt` is rejected. Validation is the official
`test` split, so identities are `food101:<train|test>:<class>/<stem>.jpg`.
`cfg/classification_food101.yml` uses its own `output_dir`
(`assets/classification/food101`) and LanceDB table, and leaves
`descriptions_file` null, because no Food-101 vocabulary exists yet;
descriptions generate into `output_dir/descriptions.json`.

`UECFood256ImageDataset` (`uecfood256`) reads the layout verified from the
central directory of Hugging Face `kng-research/uecfood256` `dataset256.zip`
(2026-11-12): one `UECFOOD256/` folder holding `README.txt`, a tab-separated
`category.txt` with an `id`/`name` header for 256 IDs, and `<id>/` directories of
`<number>.jpg` files with a `bb_info.txt` box table (31,395 listings of 28,897
distinct images). Names are canonicalized as above; IDs 40 and 67 are both
`omelet`, so they share one label and the taxonomy has 255 entries. The dataset
has no official split. Per class ID, the loader sorts the `.jpg` names,
shuffles them with `random.Random(f"{SEED}:{class_id}")`, and holds out
`round(n * VALIDATION_RATIO)` (class constants `0` and `0.2`) as validation; the
split is not written to disk. Images listed under several class directories show
several foods and are dropped from both splits (27,043 remain upstream; the
smallest classes keep 19). `bb_info.txt` is not read. Identities are
`uecfood256:<train|validation>:<id>/<number>.jpg`; changing the seed or ratio
reassigns them and needs fresh reference storage. Results are not comparable with
published UEC-Food256 numbers, which use the official 5-fold protocol.
`cfg/classification_uecfood256.yml` mirrors the Food-101 config with its own
`output_dir`, table, and null `descriptions_file`.

`VegFruImageDataset` (`vegfru`) reads `classid_classname.csv` (a
`class_id,class_name` header, then IDs 0-199 for vegetables and 200-291 for
fruits) for the taxonomy, canonicalized as above; the 292 canonical names are
unique, so no IDs merge. Training references come from
`vegfru_list/vegfru_train.txt` and validation from `vegfru_list/vegfru_test.txt`
(`LIST_DIR` class constant); `vegfru_val.txt` is not read. Each line is
`<relative path> <label id>` under `root`; the label is the path's class
directory, the ID is ignored, and a class missing from the CSV is rejected.
Identities are `vegfru:<train|test>:<veg200_images|fru92_images>/<class>/<file>`.
Verified from the head of Hugging Face `xedwin23x/VegFru` `vegfru.tar.gz`
(27.9 GB, ungated; first 5 MB streamed on 2026-11-12): top-level `vegfru/` with
`veg200_images/`, `classid_classname.csv`, and a 14 GB nested `archive.zip`
whose first member is `fru92_images/Chinese_chestnut/f_07_03_0001.jpg`.
Unverified, from the published VegFru layout: the list directory name, list
filenames and line format, and where the lists and `veg200_images/` contents sit
(after the nested zip in the tar, or inside it). Check them on the first
authorized download and adjust `LIST_DIR` or the parser.
`cfg/classification_vegfru.yml` mirrors the other configs with its own
`output_dir`, table, and null `descriptions_file`.

`ISIAFood500ImageDataset` (`isia500`) reads `metadata_ISIAFood_500/` (`META_DIR`):
`ISIAFood500_classLabel.txt` holds headerless `name<TAB>id` lines for the
taxonomy, and `train_finetune.txt` / `test_finetune.txt` hold `<class>/<file> <id>`
lines for training references and validation. Labels come from the ID; an ID
missing from the class file is rejected. Names only have underscores replaced by
spaces (`spaced`), keeping case, parentheses, and non-ASCII text. Images resolve
under `images/` (`IMAGE_DIR`); identities are `isia500:<train|test>:<class>/<file>`.
The label and list formats come from the metadata zip; the metadata zip's nesting
and URL and the image directory produced by `7z` are unverified, so check them on
the first authorized download. `cfg/classification_isia500.yml` has its own
`output_dir`, table, and null `descriptions_file`.

### Downloads

`download(root, client=None)` fetches and unpacks a dataset; the base class
raises `NotImplementedError`. Helpers in `datasets/util.py` serve every subclass:
`fetch_archive(url, root, client)` streams to `root/.downloads/<name>.part` and
renames on completion, reusing a completed archive without a request, and
discards an unencoded body whose size differs from `Content-Length`
(`require_length=True` also rejects a missing header);
`extract(archive, destination)` unpacks `.zip` files or tars with the `data`
filter, which rejects members escaping `destination`. `client` is an optional
`httpx.Client` for tests; the default follows redirects with a 60 s timeout.

`FoodX251ImageDataset.download()` returns early when `annot/class_list.txt`
exists. Otherwise it fetches `https://food-x.s3.amazonaws.com/{train,val,annot}.tar`,
extracts each into `root/<part>`, and deletes the archive. `annot` goes last so
its class list marks a complete download. The real tar layout is unverified;
the loader expects `annot/`, `train/train_set/`, and `val/val_set/`.

`Food101ImageDataset.download()` returns early when `meta/classes.txt` exists.
Otherwise it fetches `https://data.vision.ee.ethz.ch/cvl/food-101.tar.gz` and
calls `install_archive(archive, root, inner, last)`, which extracts to
`root/.downloads/<archive>.staging`, moves the `inner` entries (`food-101/`) into
`root` with `last` (`meta/`) moved last as the completion marker, and deletes the
staging directory and archive.

`UECFood256ImageDataset.download()` returns early when `category.txt` exists.
Otherwise it fetches
`https://huggingface.co/datasets/kng-research/uecfood256/resolve/main/dataset256.zip`
(4.2 GB, ungated) through `fetch_archive`, so tests mock it with `client`, and
installs the `UECFOOD256/` entries with `category.txt` last.

`VegFruImageDataset.download()` returns early when `classid_classname.csv`
exists. Otherwise it fetches
`https://huggingface.co/datasets/xedwin23x/VegFru/resolve/main/vegfru.tar.gz`
through `fetch_archive` and calls `install_archive(..., "vegfru",
"classid_classname.csv", nested="archive.zip")`: the nested zip is unpacked into
the staged `vegfru/` and deleted before the entries move, so the CSV still marks
a complete install. Peak disk use is roughly 70 GB (archive, staged tar, unpacked zip).

`ISIAFood500ImageDataset.download()` returns early when
`metadata_ISIAFood_500/ISIAFood500_classLabel.txt` exists. It requires `7z` on
PATH (conda-forge `p7zip`, in every Pixi environment) and fails before any request
otherwise. From `BASE_URL`
(`http://123.57.42.89/Dataset_ict/ISIA_Food500_Dir/dataset/`, plain HTTP, no
checksums) it fetches `METADATA_URL` (`metadata_ISIAFood_500.zip`, assumed to sit
next to the parts) and the split zip `ISIA_Food500.z01`-`.z10` plus
`ISIA_Food500.zip`, each with `require_length=True`. It runs
`7z x -y -o<root> ISIA_Food500.zip`, then moves the directory holding the class
file out of the extracted metadata zip into `META_DIR`, class file last, and
deletes all archives. A failed `7z` run keeps the parts for a retry. Expect about
90 GB peak disk use (43.6 GB of parts plus unpacked images).

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
