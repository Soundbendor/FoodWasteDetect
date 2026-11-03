# CLIP source review and correctness gaps

Read this before editing legacy annotation conversion, crop generation, Qdrant
insertion, detector evaluation, or CLIP fine-tuning. Inspect
`src/util/joint_labels.py` as text only: importing it rewrites labels in a hard-coded
shared directory.

This review describes `clip` at commit
`204e467696551600f0d57c1485b840505dbf2e58` (December 2, 2025), reviewed October 19,
2026. It distinguishes implementation from the [target architecture](architecture.md).
Recheck affected source before repairing a finding and update this document when
fixes land. The October review used source inspection, syntax parsing, and CPU
probes with dependency stubs. It reproduced instance loss, crop-path mismatch,
cross-image metric pooling, incompatible constructors, and vector batch skipping;
those probes were not a committed regression suite or model-accuracy evaluation.

## Source map


| Component | Reviewed behavior and source |
| --- | --- |
| Experiment orchestration | [src/yolo.py](../../src/yolo.py) contains `ExperimentManager`, reference-index construction, crop classification, detector baselines, and several evaluation paths. |
| Localization and crops | [src/database/img_crop.py](../../src/database/img_crop.py) loads `ultralytics.YOLO(model)`, extracts boxes, saves rectangular crops, and records detector labels and confidence. It does not carry masks into the retrieval pipeline. |
| Image and text embeddings | [src/database/clip_embedding.py](../../src/database/clip_embedding.py) wraps `SentenceTransformer(model_name, truncate_dim=vec_dim, trust_remote_code=True)`. Its `save_dir` argument does not load a saved checkpoint. |
| Retrieval | [src/database/vecdb.py](../../src/database/vecdb.py) creates cosine-distance collections, upserts labeled vectors, retrieves ten neighbors, and supports nearest-neighbor and majority-vote classification. |
| Datasets | [src/data_wrappers/dataset.py](../../src/data_wrappers/dataset.py) handles annotated crops, detected crops, and dataset tables. Wrappers include Food201, FoodSeg103, UECFoodPix, FoodX251, and FwTest. Their schemas are not interchangeable. |
| Fine-tuning | [src/clip_finetune.py](../../src/clip_finetune.py) explicitly loads `jinaai/jina-clip-v2` for image/text contrastive fine-tuning. Training outputs are not automatically selected by inference. |
| Evaluation | [src/util.py](../../src/util.py) contains crop-classification counters and the custom box evaluator. These measures have the correctness gaps listed below. |
| Configuration and entry points | [cfg/experiment_config_clip.yml](../../cfg/experiment_config_clip.yml), [src/app.py](../../src/app.py), and [pyproject.toml](../../pyproject.toml) contain incomplete configuration and stale callers. |

The checked-in CLIP configuration selects `clip-vIT-B-32` with 512 dimensions,
not Jina CLIP v2. The other experiment configuration selects BGE text embeddings
for the older caption/description pipeline. Neither proves which model populated
an existing Qdrant collection.

Active detector experiments load an external `best.pt`. The explicit
`yolo11x.pt` reference is commented out, and `finetune_yolov11()` calls validation
only. No tracked checkpoint or artifact manifest establishes that `best.pt` is a
particular YOLO11 model or a segmentation model. Verify provenance rather than
inferring architecture from function names.

## Recorded correctness gaps

1. **Instance annotations are lost during conversion.**
   `read_segments()` in [src/util/seg_to_boxes.py](../../src/util/seg_to_boxes.py) stores
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
   At review time, `pyproject.toml` declared no runtime dependencies. The current
   root manifest declares Sentence Transformers but still does not establish the
   complete legacy runtime. The reviewed `environment.yml` omits
   Ultralytics, Sentence Transformers, and the custom mAP dependency. Paths and
   Qdrant hosts are machine-specific, model revisions are unpinned, and no
   model-level regression suite is committed. The extraction adds CPU tests for
   text-package boundaries and artifacts, not model accuracy. A collection name
   or `best.pt` filename is insufficient to reproduce a result.

10. **Annotation utilities can mutate data during import.**
    [src/util/joint_labels.py](../../src/util/joint_labels.py) reads and rewrites labels
    in a hard-coded shared directory at module execution. Do not import it for
    inspection or testing. Refactor transformations into explicit functions that
    write separate, versioned outputs before using them in new experiments.
