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
Keep commits and pull requests local to this repository. This code is not the
external segmentation service currently called by `CompostBinServer`; it has no
stable production API or deployment contract. Define and test an inference
request/response contract and artifact lifecycle before connecting the two.

## Data and compute boundaries

- Do not download large models or datasets, start Ollama, use GPU/HPC resources,
  modify shared research storage, or publish model artifacts without explicit
  user authorization. Existing authorization remains applicable to its scope.
- Never commit credentials, private dataset samples, household media, embeddings
  derived from private data, or machine-specific access material.
- Preserve raw data and annotations independently from crops, converted labels,
  embeddings, predictions, and evaluation reports. Write derived artifacts to
  separate destinations with source and transformation metadata.
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

## InternVL text architecture and experiment pipeline

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
- `src/internvl/embeddings.py` owns text embedding, training, and triplet evaluation.
- `src/internvl/workflows.py` owns the text workflows formerly mixed into `app.py`.
  Functions receive configuration dictionaries; captioning also receives a
  partition number. `src/internvl/cli.py` supplies command dispatch.
- Dataset wrappers, `database.vecdb`, and classification utilities remain shared.
  Do not import the image orchestration or CLIP wrapper into the text workflows.
  Keep YOLO imports inside shared crop/detection methods and the optional
  cross-encoder import inside reranker construction.

Use `internvl` or `python -m internvl` with `caption-images`,
`generate-descriptions`, `build-db`, `evaluate`, `finetune-embeddings`,
`merge-captions`, or `convert-captions`. Model workflows take `--config_file`;
captioning also takes `--partition`. The configuration is
`cfg/experiment_config_internvl.yml`. The old text console commands and imports
under `vlm`, `llm`, and `database.embedding` have no compatibility aliases.
The image commands `build_clip_db`, `test_clip`, and `test_yolo` retain their names.

Package import and CLI help must not load models, open datasets, start subprocesses,
or contact services. Concrete backend modules are imported by the selected
workflow. Descriptor generation now probes Ollama readiness and, for unavailable
endpoints, launches the configured binary directly with bounded polling.
Its context manager cleans up only the server it started, including on startup
failure. The constructor receives `cfg["ollama"]`, requiring `llm`, `binary_path`, `host`,
`startup_timeout`, and `request_timeout` without defaults or environment fallback.
Existing Ollama endpoints are rejected with an error naming the full host;
servers not started by this instance are never stopped.
Do not invoke real server startup as part of a smoke test. Scripts under `src/internvl/scripts`
retain historical local-path and model assumptions.

### Evaluation and known limitations

FoodX251 validation compares the final retrieved labels with ground truth. The
example config samples 1,000 images with seed 42 and reports top-1, top-5, and
voting accuracy. Top-5 means five description records, not five distinct classes.
The cosine triplet evaluator compares a caption anchor with same-class and
different-class descriptions sampled from the description training split.
Sampling is unseeded. These evaluations do not isolate caption quality or measure
whether every visual claim in a description is true.

No direct descriptor-factuality or human-reference caption evaluation is
implemented. Ollama JSON validation checks structure, and the supplied label
overwrites the model's class field. The macaron query in database construction is
only a demonstration, not an accuracy benchmark.

The extraction intentionally preserves these defects for separate repair:

- INI-style configuration lookups remain in several YAML-driven workflows.
- Classification passes the anchor-file path as the embedding model name.
- Text insertion passes incompatible class/metadata arguments and omits IDs.
- The shared reranker accesses the Qdrant response incorrectly.
- Caption resume checks Series indices, and final buffered captions are not saved.
- Text embedding construction depends on training artifacts even for inference;
  random splits and triplet sampling are unseeded.

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

- Prefer import-safe package boundaries, typed model adapters, and small
  CPU-testable units. Construct model and database clients explicitly; avoid new
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
checks require the small CPU dependencies documented in README.md; missing
dependencies produce explicit skips. Keep all fixtures synthetic and local.
Explicitly identify dataset, model, GPU, network, Qdrant, and HPC checks that could
not be run. Do not claim an end-to-end experiment passed based on syntax checks or
stubbed components alone.
