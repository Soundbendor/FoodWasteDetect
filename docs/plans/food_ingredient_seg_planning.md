# Food ingredient instance segmentation: dataset planning

Record of the 2026-11-10 planning session. Status: research only. No code changed, no
dataset stored outside `/tmp`.

## Goal

Build a retrieval-based food instance segmentation model: one mask and one embedding
per food instance. A plate with three potatoes should yield three masks.

The legacy pipeline outside `src/classification` uses Food201, UECFoodPix, and
FoodSeg103. All three are semantic segmentation datasets: one mask per class per
image. The question was what it takes to turn Food201 and UECFoodPix into instance
datasets the way FoodInsSeg did for FoodSeg103.

## Findings

### Label granularity decides which datasets fit

| Dataset | Labels | Masks |
| --- | --- | --- |
| FoodSeg103 | 103 ingredient classes (carrot, potato, rice, sauce) | Semantic, hand-annotated |
| FoodInsSeg | Same images and classes | Instance, SAM-derived |
| Food201 | 201 dish classes | Semantic, crowdsourced |
| UECFoodPix Complete | 102 dish classes ("eels on rice", "pilaf") | Semantic, hand-annotated |

Converting Food201 or UECFoodPix can only separate one serving from another. A plate
of potatoes labeled as one dish stays one mask. Mixing those sets with FoodInsSeg
would mix two meanings of "instance". Only the FoodSeg103 family matches the goal.

### FoodInsSeg has no reproducible process

- No published paper found. The README's paper link is empty, the project page has an
  unfilled arXiv placeholder, and Crossref and OpenAlex return nothing for "When
  Segment Anything Model Meets Food Instance Segmentation" (Jiang, Lan, Lyu, Dong,
  Xue; University of Chinese Academy of Sciences).
- A SigPort supplement from February 2024 holds an appendix, 20 examples, and a
  dataset documentation file. The appendix gives MMDetection baseline settings and
  nothing about how masks are assigned or split.
- The annotation tool, InsSAM-Tool, is not released. The repo holds only `eval.py`
  (pycocotools `segm` COCOeval) and `visualize.py`.
- The documentation says InsSAM-Tool plus FoodSeg103 semantic masks produced the
  instances, with manual merging, deletion, and relabeling on the test set only.
- The same group published FoodSAM (IEEE TMM, Apache-2.0 code), which matches SAM
  masks to a predicted semantic map. It is the closest public reference. Its source
  was not read.

### FoodInsSeg inspection

Downloaded from the Google Drive link (1.2 GB) and inspected in `/tmp`. COCO JSON with
polygon masks, 103 categories, CC BY-NC-SA 4.0, academic use only.

| | Train (4,983 images) | Test (2,135 images) |
| --- | --- | --- |
| Annotations | 82,716 | 36,332 |
| Masks per image, mean / median / max | 16.6 / 11 / 126 | 17.0 / 12 / 186 |
| Smallest mask | exactly 150 px | 7 px |
| Labeled pixels under more than one instance | 0.3% | 1.9% |
| Median polygon vertices | 104 | 75 |
| Annotations with several polygons | 27% | 15% |

The hard area floor and near-zero overlap indicate machine output in train. Test looks
hand-edited, which agrees with the documentation.

Twelve rendered images (six per split) showed:

- Tight boundaries, and one mask each for separated items such as fish fillets,
  pretzels, fruit pieces, and pancakes.
- Touching same-class items merged: three stacked cookies are one instance. This is
  the three-potatoes case.
- Fragmentation in crowded scenes: a strawberry salad has 55 masks, many slivers.
- Grouped annotations: about 2.4% of annotations hold two or more separate objects,
  and about 25% of images contain at least one. One "red beans" annotation has 349
  polygons.
- Inherited FoodSeg103 label errors, such as pancakes labeled "pie".

Not checked: agreement between FoodInsSeg instances and the original FoodSeg103 masks.

Verdict: usable as pseudo-labels for training and as a rough test set. It is not a
gold benchmark, and there is no process to copy.

### Legacy code

- `src/data_wrappers/data_loader.py` downloads the three datasets and calls Ultralytics
  `convert_segment_masks_to_yolo_seg`, which writes one polygon per external contour.
  Existing `labels/` are connected-component pseudo-instances without holes.
- `read_segments()` in `src/util/seg_to_boxes.py` keys polygons by class ID and
  collapses them back to one per class (gap 1 in `docs/clip/review.md`).
- UECFoodPix stores the class ID in the R channel only. The Ultralytics converter
  version inspected reads masks as grayscale, so the existing UECFoodPix labels are
  probably wrong. Check the HPC copy.
- Data lives under `/nfs/...` on OSU HPC, not on the development machine.

## Datasets

Ingredient level:

| Dataset | Size | Masks | Notes |
| --- | --- | --- | --- |
| FoodSeg103 | 7,118 images, 103 classes | Semantic | Standard benchmark, images from Recipe1M |
| FoodSeg154 | 9,490 images, 154 classes | Semantic | FoodSeg103 plus 2,372 Asian food images; public availability of the extra images not verified |
| FoodInsSeg | 7,118 images, 119,048 masks | Instance | SAM-derived, test split hand-corrected |
| DishSeg24k | 24,096 images, 278 dish classes | Instance at dish level | Mentions ingredient-level annotations without detail; not released (ACM MM 2026) |

Food-item level, hand-annotated instances:

| Dataset | Size | Notes |
| --- | --- | --- |
| MyFoodRepo-273 | 24,119 images, 273 classes, 39,325 polygons | AIcrowd Food Recognition Benchmark; classes like bread, butter, tomato; dataset license not checked |
| UNIMIB2016 | 1,027 tray images, 73 classes, 3,616 instances | One polygon per food item; access terms not checked |
| COCO / LVIS food categories | General-purpose | About ten food classes in COCO, many more in LVIS |

Dish level, lower priority: Food201, UECFoodPix Complete, SibNet (12,557 images, 138
classes), MixedDishes (9,254 images, 184 classes), SFOOD (6,066 images in its
instance task), SUEC Food. UEC-FOOD100/256 have per-item bounding boxes only.

MyFoodRepo-273 and UNIMIB2016 are the best evaluation sets to add first because their
labels do not come from SAM.

## Draft plan

Build our own semantic-to-instance pipeline and use FoodInsSeg as a comparison output.

1. New standalone Pixi package `src/segmentation/`, separate from `src/classification`
   and the legacy scripts.
2. Semantic readers returning an image path, an `HxW` class-ID array, and the taxonomy.

   ```python
   def read_uecfoodpix_mask(path: Path) -> np.ndarray:
       return np.asarray(Image.open(path).convert("RGB"))[..., 0]   # R channel only
   ```

3. Instance splitting in pure NumPy so it is CPU-testable:

   ```python
   def split_instances(classes, sam_masks, *, min_purity=0.8, min_area=64) -> list[Instance]:
       # 1. label each SAM mask with its majority class; drop background and impure
       #    masks; clip to that class region
       # 2. largest-first containment suppression
       # 3. uncovered semantic pixels become connected-component instances
   ```

   Per class, the union of instances equals the semantic mask, so boundaries stay
   human-drawn and SAM only decides where to cut.
4. SAM adapter behind an explicit constructor, and a COCO writer and loader compatible
   with FoodInsSeg. Use RLE to keep holes.
5. Calibrate on FoodSeg103: run the pipeline and score with COCOeval against
   FoodInsSeg `Train.json` (fidelity to their automatic output) and `Test.json`
   (distance from hand-corrected labels). Tune thresholds on train only.
6. Hand-review test splits in CVAT or X-AnyLabeling.
7. Synthetic CPU tests: two touching same-class blobs, nested part/whole masks,
   uncovered residual pixels, an R-channel mask, a rectangular image, a COCO round
   trip.

## Limitations

1. Touching or occluded same-class items separate only if SAM gets it right. Without
   manual review there is no ground truth to measure that.
2. SAM-derived test labels favor SAM-based models. Report on hand-annotated sets too.
3. FoodInsSeg train labels are unreviewed machine output.
4. Licenses: FoodInsSeg is CC BY-NC-SA 4.0; UECFoodPix is non-commercial research
   only; Food201 has no stated license. None of this clears product use.
5. Food201 masks are coarse and incomplete, and sources disagree on its size (12,093
   images versus 10,100 + 2,525).
6. Datasets and GPUs are on OSU HPC. Downloads, SAM weights, and GPU runs need
   authorization under `AGENTS.md`.

## Open decisions

- Whether serving-level masks from Food201 and UECFoodPix are worth converting at all,
  given the ingredient-level goal.
- Whether to hand-check about 100 FoodInsSeg test images to estimate instance-count
  error before using it for evaluation.
- Whether to ask the FoodInsSeg authors for the manuscript and InsSAM-Tool.
- How much manual review time is available for test splits.
- Whether `src/segmentation/` as a new package is acceptable.

## Sources

- https://github.com/jamesjg/FoodInsSeg
- https://laura990501.github.io/FoodInsSeg_dataset/
- https://sigport.org/documents/when-segment-anything-model-meets-food-instance-segmentation
- https://github.com/jamesjg/FoodSAM and https://arxiv.org/abs/2308.05938
- https://arxiv.org/abs/2105.05409 (FoodSeg103, FoodSeg154)
- https://mm.cs.uec.ac.jp/uecfoodpix/
- https://openaccess.thecvf.com/content_iccv_2015/papers/Meyers_Im2Calories_Towards_an_ICCV_2015_paper.pdf
- https://arxiv.org/abs/2106.14977 (MyFoodRepo-273)
- https://arxiv.org/abs/2607.23070 (DishSeg24k)
- https://arxiv.org/abs/2507.04412 (SFOOD)
- https://ieee-dataport.org/documents/food-instance-counting-and-segmentation (SibNet)
- https://github.com/ultralytics/ultralytics/blob/main/ultralytics/data/converter.py
