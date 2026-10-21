# Food Waste Detection Research Agent Guide

## Scope and experimental direction

This repository is an experimental food-recognition pipeline. The user has chosen
the `clip` architecture as the basis for future experiments: YOLO localization,
Jina CLIP image embeddings, and classification through Qdrant similarity search.
Keep development aligned with that choice. Direct YOLO-E inference and hosted
vision models are separate possible comparisons, not replacements for this design.

The original CLIP findings below describe `clip` at commit
`204e467696551600f0d57c1485b840505dbf2e58`, dated December 2, 2025, reviewed on
October 19, 2026. They distinguish intended behavior from implemented behavior.
Recheck affected code before acting on a finding and update this guide as fixes
land. `main` contains an older InternVL/text-retrieval baseline; `fw-eval` inherits
much of `clip` and adds other experiments. Its newer date does not make it the
selected architecture.

The `vlm-refactor` branch extracts the caption/text-retrieval experiments into
`src/internvl`. This is a structural extraction with a clean break from the old
imports and commands. The user explicitly deferred runtime bug fixes, new caching
or restart guarantees, training redesign, and model changes. Preserve this scope
unless separately asked to extend it. The package name follows its current
captioner; Ollama class-description generation remains a separate task within it.

Before editing, inspect this repository's Git status and preserve unrelated work.
Keep commits and pull requests local to this repository.

Keep the root README focused on installation, configuration, usage, and a brief
experiment overview. Record implementation contracts and developer constraints
in this guide, and detailed environment checks in `docs/experiments.md`. Keep both READMEs
focused on installation and running commands. Use create-readme and
humanizer:humanizer for README updates; preserve verified commands in the prose pass.

## Data and compute boundaries

- Do not download large models or datasets, start Ollama, use GPU/HPC resources,
  modify shared research storage, or publish model artifacts without explicit
  user authorization. Existing authorization remains applicable to its scope.
- Never commit credentials, private datasets, household media, embeddings
  derived from private data, or machine-specific access material.
- Use small synthetic fixtures and CPU-only checks by default. Do not substitute
  a live production service for an automated test.

## CLIP architecture to develop

1. Localize food instances with an explicitly identified YOLO11-family checkpoint.
   When the experiment requires instance segmentation, use a segmentation-capable
   model and retain its per-instance masks in original-image coordinates.
2. Associate each detected region with a stable image ID, instance ID, box, crop,
   detector score, and mask when available. Keep these associations through
   filtering and batching.
3. Encode reference images and query crops with the same pinned
   `jinaai/jina-clip-v2` revision or explicitly selected fine-tuned checkpoint.
   Record preprocessing, embedding dimensions, and normalization choices.
4. Index labeled reference embeddings in a versioned Qdrant collection and search
   it using query-crop embeddings. Keep evaluation images and their derivatives
   out of the reference set unless a separately declared protocol requires them.
5. Assign categories from retrieved examples using a documented scoring rule.
   Preserve detector confidence and retrieval scores separately. Adding reference
   categories expands the classifier vocabulary; it does not guarantee that the
   localizer will detect previously unseen objects.

The current implementation is a collection of offline experiments, not a complete
implementation of this target. In particular, do not describe existing outputs as
instance segmentation when they contain only boxes and class predictions.

## Configurable whole-image classification (current)

The separately authorized `internvl.experiment` runner supersedes the old BGE
text-retrieval/training commands. `workflows.py` and its legacy workflows have
been deleted without a compatibility module. The
`finetune-embeddings`, `internvl build-db`, and top-level `evaluate` CLI commands
are also removed and produce ordinary argparse errors. Use `experiment index`
and `experiment evaluate` for the current alternatives. `embeddings.py` is inference-only Jina CLIP v2;
the historical BGE training/triplet implementation has been removed. References
to that implementation below describe the retired workflow, not a supported API.
The caption, merge, conversion, and duplicate description CLI commands are removed.
Description generation uses `experiment describe`; Ollama downloads remain in the CLI.
The separate CLIP fine-tuning experiment and legacy image interfaces are unchanged.

Use `python -m internvl experiment {describe,index,evaluate,run} --config_file
cfg/classification.yml`. Parsing and YAML loading live in `internvl.cli`;
`internvl.experiment` has no command-line entry point; use the shared CLI.
This independent whole-image baseline embeds pixels
without InternVL captioning or YOLO. Its three reference booleans select any
nonempty combination of class names, generated descriptions, and training images.
Use all official FoodX251 training references and labeled validation images for
evaluation, recording `source_split="val"`; never claim official test accuracy.
The dataset adapter resolves image paths, preserves legacy names, and rejects
duplicate source identities within each split. It does not compare training and
validation paths or image contents for overlap.

Full 1024-dimensional Jina embeddings use checkpoint preprocessing
and an explicit device. By user choice, this whole-image experiment does not
configure or track model/code revisions, including nested text-encoder revisions;
it uses the library's default model loading and caching behavior, including
downloads when needed during normal application execution. The no-download
restriction applies to local development checks: mock pretrained construction
and use synthetic encoders. Do not download weights or start real services for checks.
Retrieval uses exact cosine top-k search with configurable `k`. Every retrieved
record contributes one vote; ties use best similarity then class name.
`vote_fraction` is winning votes divided by the actual returned neighbor count,
not calibrated confidence. Completed evaluations cover the full validation split,
with an evaluated count and per-class metrics. Empty retrieval abstains and counts
as incorrect. Image-loading, embedding, and infrastructure errors fail the run
without retries or exclusions; stage
failures propagate without saved partial reports. There is no top-five metric.

Git revision, dependency-version, and timestamp provenance tracking has been
removed from the whole-image runner by user request.
Artifact validation is explicitly deferred by user request. Do not add hashes,
compatibility checks, manifest comparisons, or collection-content verification.
The runner preserves strict row/vector pairing and requires a nonempty validation
split with exactly one prediction per image before saving a report.
`Experiment(config)` loads dataset metadata and class names during construction,
including for disabled or cached description commands. It stores batch size once,
defaulting to 32, and creates directories only in writing stages. The embedder
and Qdrant client are constructed on first use and reused. `run()` finishes
Ollama description generation before constructing the embedder. Constructor
injection arguments and `_positive_integer()` have been removed.
`validate_config()` requires boolean reference switches and at least one enabled
source during construction. Numeric settings use backend checks and ordinary
Python errors.
Image references are constructed from each decoded training batch rather than
through the removed `references()` helper. Both image loops rely on garbage
collection for decoded batch images, as requested by the user.
`DescriberLLM.generate_descriptions(classes, count)` owns the per-class generation
loop and empty-description checks, using `describe_class(label)` for the shared
prompt. The experiment owns saved-file handling and the Ollama context. The legacy
`generate_descriptors(ds, save_path)` method has been removed; use
`generate_descriptions(classes, count)` instead.
Failures and interrupts propagate without retries or partial reports. A failed
index build may leave a partial Qdrant collection; no rollback or cleanup is added.
The previous successful index summary remains intact after a failed rebuild.
The `index` CLI command calls `Experiment.build_vector_database()`; the former
Python method `index()` has been renamed. The `describe` CLI command calls
`Experiment.generate_descriptions()`, formerly `describe()`.
CLI commands and YAML keys remain unchanged; the Python constructor and new
report formats intentionally change. Temporary-file JSON replacement remains.
Existing descriptions load directly. `qdrant.collection_prefix` now supplies the
exact collection name without an appended UUID. `build_vector_database()` checks
existence before loading the embedder or reading training inputs. For an existing
collection it warns and returns None, leaving vectors and index.json untouched.
For a missing collection it inserts records and saves collection name, source
counts and configuration in index.json after success. Select a
new name to build again, or an older collection's full name to reuse it.
No compatibility or completeness checks are added; partial collections are also
reused. `run()` still generates or loads descriptions before the build stage.
Evaluation queries the configured collection and uses a local index.json only
for metadata about that same collection. Without a matching summary, reference_source_counts is null. This supports reuse across output directories
without claiming knowledge of the stored vectors. A successful evaluation
writes only `evaluation_<UTC timestamp>/report.json`, with configuration, collection,
reference-source counts, predictions, and metrics. Prediction `value`, evaluation
manifest, fixed retrieval settings, status/error fields, and separate prediction
and metric files have been removed. Metrics retain evaluated count, majority-vote
and nearest-neighbor accuracy, and per-class metrics; no intermediate counts or
coverage are saved. Historical collections and raw data
are not overwritten. Every class requires a usable image reference when image
references are enabled. Evaluation embeddings are recomputed for each run.
`FoodX251ImageDataset` reads the local class list and split CSV annotations
with pandas, independently of `data_wrappers.foodx251`. It builds an explicitly
typed Hugging Face `DatasetDict` (`splits['train']`, `splits['validation']`) and exposes these
through `train_set()` and `test_set()`. Class ID mapping and naturalized names
remain compatible. `train_path` points to the training images; `val_path` and
`test_path` point to labeled validation images. No official test files are read.
`iter_batches(split, batch_size)` accepts `train`, `val`, or `validation`, enables
image decoding, and uses `Dataset.iter()` with the final partial batch retained.
Metadata access leaves images undecoded. The experiment embeds decoded PIL images
and leaves their cleanup to garbage collection. Paths must resolve within their
split's image directory. Image-content hashing and SHA-256 metadata were
removed by user request; do not restore them. Image files are read when batches
are decoded, and loading failures propagate. No exclusion artifacts are written by new runs.
Compare complete retrieval strategies, because changing reference sources also
changes reference counts. No tuning or subsampling occurs.

CPU verification includes fake model/Ollama backends, synthetic images, miniature
FoodX251 fixtures, Hugging Face datasets, and qdrant-client's in-memory backend.
Embedding contracts are isolated in `tests/test_embeddings.py` and require
Sentence Transformers, CPU PyTorch, NumPy, and Pillow. `EmbeddingModel` wraps a
`SentenceTransformer` directly; `encode()` owns inference batching and NumPy
conversion. The wrapper does not request embedding normalization; Qdrant handles
normalization for cosine retrieval. Model settings are recorded only through the saved experiment
configuration; the embedder has no metadata dictionary/property or separate artifact
model field. The experiment iterates decoded dataset batches to bound
image memory; `encode_images()` prepares the supplied batch without rebatching it.
String inputs use Jina's default text/path/URL interpretation without overriding
`assume_text_inputs`.
The RGB helper explicitly closes path-opened originals after conversion, including
on conversion failure, and leaves caller-owned PIL images open. RGB copies are
released through garbage collection after use. Loading calls Sentence Transformers directly,
without temporary configuration copies or enforced offline mode.
Tests exercise real Sentence Transformers
with synthetic vectors and mock pretrained construction; they do not establish
real-model compatibility. `EmbeddingModel` has no backend injection parameter;
encoding tests patch `_load_backend()` to return a fake or synthetic encoder.
Install the CPU environment described in `src/internvl/README.md` for these tests;
retain explicit skips when dependencies are absent.
These tests do not establish real model, dataset, GPU, Ollama, remote-Qdrant, or HPC
compatibility. The historical workflow limitations below describe the removed
legacy source, and must not be attributed to the new experiment runner.

## InternVL text architecture and experiment pipeline (historical)

The following describes the removed workflow. Its source remains in Git history;
the standalone model and description backends remain in the repository.

This is an independent whole-image classification experiment. It does not use
YOLO or produce instance masks. YOLO is needed only by the separate image/crop
pipeline and by explicitly called shared dataset crop/detection methods.

1. Pass known FoodX251 class names to `DescriberLLM` through Ollama. The configured
   model is `gemma3:27b`. It receives class labels, not images, and attempts five
   generic visual descriptions per class. The result is a JSON dictionary from
   class labels to lists of descriptions.
2. Embed the reference sentences with the text `EmbeddingModel`, configured as
   `BAAI/bge-large-en-v1.5`. The database workflow intends to store vectors,
   descriptions, and class labels in Qdrant. Ollama generates text; it does not
   generate these embeddings.
3. Give a whole image and the visual-description prompt to `InternVLM`. The
   configured checkpoint is `InternVL2_5-8B-MPO`, loaded directly with Transformers
   on CUDA. Its prompt does not include the image's ground-truth food label.
4. Embed the returned caption using an embedding model compatible with the
   reference collection, retrieve up to ten Qdrant neighbors, optionally rerank
   their descriptions with a cross-encoder, and classify using their class labels.
   The main evaluation uses majority voting. Ollama is not called for each query
   once reference generation is complete. Collection compatibility is not yet
   validated by the implementation.

Caption generation can also save training-image captions, merge four partition
files, and group captions by ground-truth class into a reference dictionary.
That format is interchangeable with the class-describer dictionary, but their
provenance differs. Preserve that distinction in experiment records. Per-object
InternVL captioning after YOLO detection is a possible extension, not existing
behavior.

### Package ownership and commands

- `src/internvl/model.py` owns the existing `InternVLM` and image preprocessing.
- `src/internvl/generate_descriptors.py` owns `DescriberLLM` and `FoodDescriptor`.
- `src/internvl/embeddings.py` now owns inference-only multimodal embeddings; the
  former text training/triplet implementation is retired.
- `src/internvl/cli.py` owns command dispatch and the Ollama download wrapper.
  The former `src/internvl/workflows.py` module has been deleted.
- Dataset wrappers, `database.vecdb`, and classification utilities remain shared.

Use `internvl download_model MODEL_NAME` or `python -m internvl download_model
MODEL_NAME` for downloads. It reads `cfg/experiment_config_internvl.yml` from the
working directory without a config flag. `caption-images`, `generate-descriptions`,
`merge-captions`, and `convert-captions` are removed and produce argparse errors.
Generate descriptions with `python -m internvl experiment describe --config_file
cfg/classification.yml`: `embed_descriptions` controls generation, cached
`output_dir/descriptions.json` is reused, and new descriptions are written there.
No existing artifacts are migrated. The description Slurm script accepts an
optional config path and defaults to `cfg/classification.yml`.
The old text console commands and imports under `vlm`, `llm`, and
`database.embedding` have no compatibility aliases. The image commands
`build_clip_db`, `test_clip`, and `test_yolo` retain their names.

Descriptor generation now probes Ollama readiness and, for unavailable
endpoints, launches the configured binary directly with bounded polling.
Its context manager cleans up only the server it started, including on startup
failure. The constructor receives `cfg["ollama"]`, requiring `llm`, `binary_path`, `host`,
`startup_timeout`, and `request_timeout` without defaults or environment fallback.
Existing Ollama endpoints are rejected with an error naming the full host;
servers not started by this instance are never stopped. Before probing or launching,
startup validates `ollama.binary_path` as an executable file, expands `~`, and
resolves explicit relative paths against the working directory. Bare executable
names resolve through PATH, allowing `ollama.binary_path: ollama` under Pixi.
Missing/invalid installations and OS launch errors point to README.md: Ollama setup.
No automatic binary or model installation is performed.
Do not invoke real server startup as part of a smoke test.
`internvl download_model MODEL_NAME` reads the binary and host from the experiment
YAML, starts an owned server, pulls the positional model through `Client.pull(stream=True)` with visible progress,
and cleans up on success, failure, or Ctrl-C. It rejects occupied endpoints and
does not change `ollama.llm` or the YAML. Run from the repository root with the
Python environment activated and Ollama installed; optional `OLLAMA_MODELS` is
inherited. Use an approved cluster allocation; Slurm handles forced termination. The historical
`start_deepseek.sh` retains its local-path assumptions.

### Evaluation and known limitations

FoodX251 validation compares the final retrieved labels with ground truth. The
example config samples 1,000 images with seed 42 and reports top-1, top-5, and
voting accuracy. Top-5 means five description records, not five distinct classes.
The former cosine triplet evaluator and BGE embedding-training workflow have
been removed. The historical retrieval evaluation did not isolate caption quality
or measure whether every visual claim in a description was true.

No direct descriptor-factuality or human-reference caption evaluation is
implemented. Ollama JSON validation checks structure, and the supplied label
overwrites the model's class field. The macaron query in database construction is
only a demonstration, not an accuracy benchmark.

The deleted legacy workflows had these defects; this historical list does not
describe the current experiment runner:

- INI-style configuration lookups remain in several YAML-driven workflows.
- Classification passes the anchor-file path as the embedding model name.
- Text insertion passes incompatible class/metadata arguments and omits IDs.
- The shared reranker accesses the Qdrant response incorrectly.
- Caption resume checks Series indices, and final buffered captions are not saved.
- Legacy retrieval callers still expect the removed training-dependent embedding
  interface and are not supported by the inference-only wrapper.

Do not claim successful end-to-end inference from the extraction tests. They
establish command wiring, import boundaries, and selected artifact behavior only.
The README lists command migration, file formats, and preserved limitations.

## Verified CLIP implementation and source map

| Component | Current behavior and source |
| --- | --- |
| Experiment orchestration | [src/yolo.py](src/yolo.py) contains `ExperimentManager`, reference-index construction, crop classification, detector baselines, and several evaluation paths. |
| Localization and crops | [src/database/img_crop.py](src/database/img_crop.py) loads `ultralytics.YOLO(model)`, extracts boxes, saves rectangular crops, and records detector labels and confidence. It does not carry masks into the retrieval pipeline. |
| Image and text embeddings | [src/database/clip_embedding.py](src/database/clip_embedding.py) wraps `SentenceTransformer(model_name, truncate_dim=vec_dim, trust_remote_code=True)`. Its `save_dir` argument does not load a saved checkpoint. |
| Retrieval | [src/database/vecdb.py](src/database/vecdb.py) creates cosine-distance collections, upserts labeled vectors, retrieves ten neighbors, and supports nearest-neighbor and majority-vote classification. |
| Datasets | [src/data_wrappers/dataset.py](src/data_wrappers/dataset.py) handles annotated crops, detected crops, and dataset tables. Wrappers include Food201, FoodSeg103, UECFoodPix, FoodX251, and FwTest. Their schemas are not interchangeable. |
| Fine-tuning | [src/clip_finetune.py](src/clip_finetune.py) explicitly loads `jinaai/jina-clip-v2` for image/text contrastive fine-tuning. Training outputs are not automatically selected by inference. |
| Evaluation | [src/util.py](src/util.py) contains crop-classification counters and the custom box evaluator. These measures have the correctness gaps listed below. |
| Configuration and entry points | [cfg/experiment_config_clip.yml](cfg/experiment_config_clip.yml), [src/app.py](src/app.py), and [pyproject.toml](pyproject.toml) contain incomplete configuration and stale callers. |

The checked-in CLIP configuration selects `clip-vIT-B-32` with 512 dimensions,
not Jina CLIP v2. The other experiment configuration selects BGE text embeddings
for the older caption/description pipeline. Neither proves which model populated
an existing Qdrant collection.

Active detector experiments load an external `best.pt`. The explicit
`yolo11x.pt` reference is commented out, and `finetune_yolov11()` calls validation
only. No tracked checkpoint or artifact manifest establishes that `best.pt` is a
particular YOLO11 model or a segmentation model. Verify provenance rather than
inferring architecture from function names.

## Known gaps in the reviewed clip implementation

1. **Instance annotations are lost during conversion.**
   `read_segments()` in [src/util/seg_to_boxes.py](src/util/seg_to_boxes.py) stores
   polygons in a dictionary keyed by class ID. A second instance of the same
   class overwrites the first. Preserve every instance in derived boxes and crops.

2. **Detection evaluation can match objects across unrelated images.**
   `ExperimentResult.get_map50()` in `src/util.py` pools predictions and targets
   from all images before one `metric_fn.add()` call. It also hard-codes 414
   classes. Both its mAP and precision/recall loops iterate prediction groups,
   excluding images with zero detections from ground-truth accounting.
   `evaluate_pipeline()` in `src/yolo.py` compares class names without spatial
   matching. These outputs do not establish valid end-to-end detection or
   segmentation performance. The separate `get_pr()` implementation does perform
   IoU matching within each visited image; do not confuse it with the different
   broken evaluator on `fw-eval`.

3. **Crop writers and readers disagree on paths.**
   `ImageCropper.crop_dataset()` writes to `<root>/cropped-detections`, while
   `Dataset.get_patches()` reconstructs `<root>/<split>/cropped-detections`.
   `ExperimentManager.evaluate_model()` silently skips embedding exceptions,
   allowing missing or unreadable crops to disappear from accuracy denominators.

4. **Vector IDs and resume logic can omit reference examples.**
   `ExperimentManager.update_vecdb()` returns the last used ID, which callers
   reuse as the next dataset's starting ID. It checks only the first ID of each
   batch and skips the whole batch if that ID exists. `np.array_split(..., 50)`
   creates 50 partitions, not batches of 50 records; small datasets can produce
   empty partitions that fail at `patches.index[0]`. Collection reuse checks
   existence but not embedding-model or taxonomy compatibility.

5. **Missing crops can shift classification results onto different instances.**
   In `clip_update_preds()`, missing files are skipped while collecting
   embeddings, but subsequent lookup uses the original dataframe indices.
   A missing crop or non-contiguous index can misassign embeddings or raise an
   index error. Carry explicit instance IDs alongside embeddings.

6. **Several declared entry points are incompatible with their implementations.**
   `app.eval_img_vdb()` and `app.build_img_vdb()` pass an extra dataset argument
   to the three-argument `CLIPEmbedding` constructor. The latter also calls
   `VectorDB.add_records()` without its required `ids` argument and supplies a
   single class string where the implementation expects per-record labels.
   `test_yolo` invokes `yolo.main()`, whose `YOLOWorld` symbol is not imported.
   The direct `src/yolo.py` path selects FoodX251 using a config key absent from
   the checked-in CLIP config and assumes pre-existing patch files. A declared
   command is not evidence that the experiment runs.

7. **Fine-tuning has data and evaluation errors.**
   `finetune_combined()` reads `row["fpath"]` for evaluation even though its
   loader supplies `patch_pth`. It uses test-set patches for the evaluator that
   selects the best checkpoint. Random negative captions can equal the true
   category while being labeled negative. Preserve a separate validation split,
   exclude true-class negatives, and record random seeds.

8. **Retrieval scores do not define calibrated class confidence.**
   Majority voting always chooses an indexed label and does not handle an empty
   result set. Vote confidence divides by ten even when fewer neighbors return.
   The reported top five are five reference examples, potentially with repeated
   labels, not five distinct candidate classes. Reclassification retains the
   original detector confidence. Define unknown rejection and any combined score
   as explicit experimental choices, fitted on validation data.

9. **Environment and artifact provenance are incomplete.**
   `pyproject.toml` declares no runtime dependencies. `environment.yml` omits
   Ultralytics, Sentence Transformers, and the custom mAP dependency. Paths and
   Qdrant hosts are machine-specific, model revisions are unpinned, and no
   model-level regression suite is committed. The extraction adds CPU tests for
   text-package boundaries and artifacts, not model accuracy. A collection name
   or `best.pt` filename is insufficient to reproduce a result.

10. **Annotation utilities can mutate data during import.**
    [src/util/joint_labels.py](src/util/joint_labels.py) reads and rewrites labels
    in a hard-coded shared directory at module execution. Do not import it for
    inspection or testing. Refactor transformations into explicit functions that
    write separate, versioned outputs before using them in new experiments.

## Implementation and evaluation rules

- Import dependencies normally, including imports used by type annotations.
  Never use `if TYPE_CHECKING` guards or equivalent type-only import workarounds.
  Do not add tests that forbid dependency imports. Keep model construction,
  downloads, service startup, and other runtime side effects behind explicit calls.
- Prefer typed model adapters and small CPU-testable units.
  Construct model and database clients explicitly; avoid new
  script-level downloads, inference, service startup, or file mutations.
- Make paths, model revisions/checksums, taxonomy versions, dataset splits, seeds,
  and artifact metadata explicit before extending experiments. Record the code
  commit, reference-set manifest, crop policy, collection version, retrieval
  settings, thresholds, and dependency versions with each run.
- Keep detector, embedder, reference index, classifier, and evaluator separable.
  A changed embedding model or preprocessing policy requires compatible reference
  embeddings. Validate collection metadata before insertion and querying.
- Use stable identifiers that include dataset, source image, and instance identity.
  Make insertion and resume behavior idempotent and verify individual records.
  Do not infer batch completeness from one existing point.
- Preserve per-image boundaries and count missed objects on images with zero
  predictions. Define box coordinates, mask resolution, class mapping, and score
  semantics explicitly. Match predictions to ground truth one-to-one within an
  image and class using the declared IoU threshold.
- Report crop-classification accuracy, detector recall, box mAP, and mask mAP as
  distinct measurements. Classification on annotated crops does not measure
  localization. Retain failures and exclusions in run reports instead of silently
  evaluating only successful samples.
- Keep source images and their crops in the same dataset split. Use training
  references for retrieval, validation for model/threshold selection, and an
  untouched test set for final reporting. Declare held-out categories explicitly
  when claiming open-vocabulary generalization.

## Verification and repair order

Prioritize annotation integrity and correct per-image evaluation before trusting
comparisons. Then repair crop paths and instance alignment, entry points, vector
index integrity, and reproducible model configuration. Add mask preservation and
mask evaluation when implementing the intended instance-segmentation experiment.

Use focused synthetic regression cases for the behavior being changed:

- Two same-class objects in one image must survive annotation conversion.
- Predictions in one image must never match targets in another; include missed
  images, empty predictions, empty targets, duplicate predictions, and bad boxes.
- Rectangular images must preserve width/height and mask coordinate conventions.
- Crop generation and loading must agree; missing crops must not shift instance
  assignments or silently reduce the evaluated sample count.
- Multiple datasets, partial resumes, empty batches, and fewer than 50 records
  must preserve vector identity and completeness.
- Empty retrieval results, fewer than ten neighbors, repeated class labels, and
  incompatible collection metadata must have defined behavior.

The October 2026 review used source inspection, syntax parsing, and CPU-only
probes with dependency stubs. It reproduced same-class instance loss, crop-path
mismatch, cross-image metric pooling, incompatible constructor calls, and vector
batch skipping. Those probes are not a committed test suite and do not establish
model accuracy or external-library integration.

Run checks appropriate to each change and report what was actually verified.
For the extraction checks, use
`PYTHONPATH=src python -m unittest discover -s tests -v`. Artifact and dataset
checks require the CPU dependencies documented in `src/internvl/README.md`; missing
dependencies produce explicit skips. Keep all fixtures synthetic and local.
Explicitly identify dataset, model, GPU, network, Qdrant, and HPC checks that could
not be run. Do not claim an end-to-end experiment passed based on syntax checks or
stubbed components alone.


## Standalone internvl environment

`src/internvl/pyproject.toml` and its committed `pixi.lock` own the `internvl`
package and console command; root package discovery excludes it. Use standard
CPython 3.13.13 and select `-e default` or `-e cu126` with
`pixi install/run --manifest-path src/internvl/pyproject.toml --locked`.
The environments are separate Linux x86-64 solve groups. They include native
Ollama 0.9.6 (CPU or the pinned CUDA 12.9 build) and Qdrant 1.19.0.
The Ollama SDK remains a separate PyPI dependency. Slurm scripts use `--as-is`
and require the environment to be prepared before submission. The legacy description method and
its `ClassDataset` protocol have been removed. See `src/internvl/README.md`
for installation, Ruff, scoped ty checks, and pytest coverage commands.
`internvl.scripts.smoke_jina` is an explicit real-model download check with
temporary subprocess caches; it is never part of routine synthetic tests.
The user previously authorized running it during the uv environment setup;
the Pixi migration uses synthetic checks without new model downloads.

Real-model acceptance required Transformers `<5` (resolved 4.57.6), Sentence
Transformers `>=5.4,<6` (5.7.0), and datasets `<5` (4.8.5). The initially proposed
Transformers 5.5.4 stack failed image embedding validation in both profiles.
The previous Python 3.14.8 profiles (CPU PyTorch 2.14.1 and CUDA 12.8
PyTorch 2.11.0) passed real text/image encoding on CPU and temporary-cache
cleanup. These historical results do not validate the current PyTorch 2.7 profiles.
CUDA execution and external services were not tested.

The V100 compatibility migration pins both profiles to PyTorch 2.7.1 and
torchvision 0.22.1 with Python >=3.13,<3.14. The GPU profile uses the explicit
CUDA 12.6 index: PyTorch 2.11 CUDA 12.8 binaries dropped Volta support.
Keep CPU and CUDA features in separate environments and select one on every
install/run. See `docs/experiments.md` for the manual V100 kernel check; routine
verification remains synthetic and CPU-only.
Before Pixi, both uv profiles installed and passed 79 synthetic tests with one existing
skip and 93% coverage on Python 3.13.13; the CUDA profile ran with GPUs hidden.
Ruff, scoped ty, and CPU CLI help passed. Imports confirmed the pinned versions
and the CUDA binary includes sm_70. No real-model or GPU execution was performed
for this migration; historical real-model acceptance does not transfer to it.

Pixi migration checks: both environments passed 81 synthetic tests, one existing
skip, and 93% coverage, plus lint, scoped ty, CLI help, and builds. Repeated locked
installation left the lockfile unchanged. Ollama uses the corrected CUDA 12.9
build `cuda_129h63bc95c_4`; the originally selected CUDA 12.6 artifact lacked
backend libraries. The target cluster driver is 580.178.04, which supports CUDA
12.9. PyTorch remains on CUDA 12.6, so the environment name is still `cu126`.
See `docs/experiments.md` for verification. Do not claim GPU Ollama acceptance
from package inspection or synthetic tests. No live services, model downloads,
or GPU kernels were run.
