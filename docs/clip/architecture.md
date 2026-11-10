# CLIP localization and retrieval architecture

Read this before extending the legacy YOLO/CLIP/Qdrant experiment. This is the
selected research target, not a claim that the current scripts implement every
step. Consult the [source review](review.md) before changing affected code.

## Target pipeline

1. Localize food with an explicitly identified YOLO11-family checkpoint. Instance
   segmentation requires a segmentation-capable model and per-instance masks
   retained in original-image coordinates.
2. Carry stable dataset, image, and instance identities with each box, crop,
   detector score, and available mask through filtering and batching.
3. Encode references and query crops with the same pinned `jinaai/jina-clip-v2`
   revision or explicitly selected fine-tuned checkpoint. Record preprocessing,
   dimensions, and normalization.
4. Index labeled references in a versioned Qdrant collection. Keep evaluation
   images and derivatives out of references unless a declared protocol requires
   them. Validate model/taxonomy compatibility before insertion and querying.
5. Assign categories using a documented retrieval scoring rule. Keep detector
   confidence and retrieval scores separate. Adding reference categories expands
   classifier vocabulary without guaranteeing localizer recall for unseen objects.

Keep detector, embedder, reference index, classifier, and evaluator separable.
Use typed adapters and small CPU-testable units with explicit client construction.
Insertion/resumption should verify individual stable identities rather than infer
batch completeness from one stored point.

## Data and provenance

Before extending this experiment, make paths, model revisions/checksums, taxonomy
versions, splits, seeds, and artifact metadata explicit. Record the code commit,
reference manifest, crop policy, collection version, retrieval settings,
thresholds, and dependency versions with each run. Changed embedding models or
preprocessing require compatible reference embeddings.

Keep source images and their crops in the same split. Use training references,
validation for model/threshold selection, and an untouched test set for final
reporting. Declare held-out categories for open-vocabulary claims. Preserve raw
data and historical collections; write versioned derived outputs.

These provenance and compatibility requirements belong to the CLIP target.
The separate whole-image runner intentionally defers that machinery; follow its
[experiment contract](../classification/experiments.md) when working in `classification`.

## Evaluation contract

Preserve per-image boundaries and include ground truth from images with no
predictions. Define box coordinates, mask resolution, class mapping, and score
semantics. Match predictions and ground truth one-to-one within the same image
and class at the declared IoU threshold.

Report annotated-crop classification, detector recall, box mAP, and mask mAP
separately. Annotated-crop accuracy does not measure localization. Retain failures
and exclusions in run reports rather than silently shrinking the denominator.
Unknown rejection and any combined detector/retrieval score require an explicit
experimental choice fitted on validation data.

## Repair sequence and regression cases

Prioritize annotation integrity and per-image evaluation before comparisons.
Then repair crop paths/instance alignment, entry points, vector integrity, and
model configuration. Add mask preservation and mask evaluation when implementing
instance segmentation.

Use synthetic cases that establish the changed behavior:

- Two same-class objects survive annotation conversion.
- Cross-image objects cannot match; cover missed images, empty predictions and
  targets, duplicate predictions, and invalid boxes.
- Rectangular images preserve width/height and mask coordinate conventions.
- Crop readers agree with writers; missing crops retain correct instance alignment
  and cannot silently reduce evaluated counts.
- Multiple datasets, partial resumes, empty batches, and fewer than 50 records
  preserve vector identities and completeness.
- Empty retrieval, fewer than ten neighbors, repeated labels, and incompatible
  collection metadata have defined outcomes.

Report what was actually verified. Source inspection and stubbed CPU probes do
not establish real-model accuracy or end-to-end external-library integration.
