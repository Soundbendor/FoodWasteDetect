# Experiment notes

These notes describe current behavior and limitations of the offline experiments.
See the [README](../README.md) for setup, commands, and the module map.

## Text retrieval workflow

Ollama receives known FoodX251 class names and attempts five generic visual
descriptions per class. It receives no image pixels. The example configuration
selects `gemma3:27b`. A separate sentence embedder, configured as
`BAAI/bge-large-en-v1.5`, embeds reference sentences for storage with their labels
and text in Qdrant. Ollama generates text, not embeddings.

For a query, InternVL receives a whole image and a visual-description prompt
without the ground-truth label. The configured `InternVL2_5-8B-MPO` checkpoint
loads directly through Transformers on CUDA. The text embedder encodes its
caption; Qdrant retrieves up to ten reference descriptions. An optional
cross-encoder reranks them before label-based classification. Ollama is not
called for each query once references are populated.

Stored and query embeddings must use compatible models and preprocessing.
Collection compatibility is not validated. This workflow uses whole images;
per-object captioning after detection is not implemented.

## Configuration and artifacts

Run from the repository root. Model workflows receive the YAML selected by
`--config_file`; `download_model MODEL_NAME` always reads
`cfg/experiment_config_internvl.yml` from the working directory and accepts no
configuration flag. It pulls the positional Ollama model with progress and does
not change `ollama.llm` or the YAML. It starts an owned server, rejects existing
endpoints, and cleans up on completion, failure, or Ctrl-C. `OLLAMA_MODELS`, when
set, is inherited. Use an approved allocation for cluster execution.

The required Ollama settings are `llm`, `binary_path`, `host`, `startup_timeout`,
and `request_timeout`. The executable path expands `~` and resolves relative to
the working directory, without PATH discovery. No automatic installation occurs.
See [Ollama setup](../README.md#ollama-setup).

| Artifact | Producer and format |
| --- | --- |
| Configured `paths.dictionary` | `generate-descriptions`: JSON object mapping class labels to lists of descriptions. |
| `assets/foodx251_captions_1.txt` through `_4.txt` | `caption-images`: CSV data despite the extension, with image index (`idx`), class, and caption fields. Each partition selects a quarter of the training table. |
| `assets/foodx251_captions.txt` | `merge-captions`: combines the four fixed partition paths, retaining `idx`, `class`, and `caption`. |
| `foodx251_captions.json` | `convert-captions`: reads the fixed merged path and writes a class-to-caption-list dictionary in the working directory. |
| Configured Qdrant collection | `build-db`: intended destination for reference vectors, descriptions, and class labels; insertion currently needs repair. |

The generated-description and grouped-caption dictionaries share a format but
have different provenance: one derives from class names, the other from labeled
training images. Record which source an experiment uses. Conversion does not
update `paths.dictionary`; select the resulting file explicitly. Preserve source
images and annotations independently from derived captions, crops, and vectors.

## Evaluation

Text classification compares retrieved labels with FoodX251 validation labels.
The example configuration samples 1,000 images with seed 42 and reports top-1,
top-5, and voting accuracy. Top-5 examines five description records, which can
repeat classes. Voting selects the most common label among up to ten records.
These metrics evaluate the combined caption, embedding, and retrieval path.

The BGE embedding-training experiment, its `finetune-embeddings` command, and
cosine triplet evaluator have been removed. The current whole-image experiment
uses pretrained multimodal embeddings without training; see
[whole-image classification](../README.md#whole-image-classification).

No direct visual-factuality or human-reference caption evaluation is implemented.
Ollama response validation checks structure, and the supplied class replaces the
model's class field. Neither establishes factual accuracy. The macaron query in
database construction is a demonstration, not a benchmark.

## Text workflow limitations

- `caption-images` and the retired `build-db` workflow use INI-style
  `cfg.get("Models", ...)` lookups against YAML dictionaries, resolving settings
  incorrectly.
- `evaluate` passes the caption-anchor path as the embedding model name.
- Text database construction omits required IDs and passes incompatible labels
  and metadata to the shared insertion method.
- The shared reranker iterates the Qdrant response before accessing its points.
- Caption resume checks Series indices rather than stored image IDs, and final
  buffered captions are not flushed. Caption files can therefore be incomplete.
- Retired retrieval workflows still call the removed training-dependent embedding
  interface and cannot use the new inference-only wrapper.

## Image workflow limitations

The selected research direction is YOLO localization, Jina CLIP image embeddings,
and Qdrant retrieval. The checked-in CLIP configuration instead selects
`clip-vIT-B-32` with 512 dimensions. The external `best.pt` filename does not
establish detector architecture or checkpoint provenance. Existing crop retrieval
uses rectangular boxes and does not preserve instance masks.

The console entry points do not yet provide a working implementation of this
target:

- `build_clip_db` and `test_clip` pass an extra dataset argument to
  `CLIPEmbedding`. Database construction also omits IDs and passes a single class
  string where per-record labels are expected.
- These entry points expect `paths.dataset`, which the example CLIP YAML lacks.
- `test_yolo` calls a YOLO-World baseline whose `YOLOWorld` symbol is not imported.
  It is separate from the YOLO/CLIP retrieval experiment.

Image evaluation also has correctness gaps: detection mAP pools unrelated images,
missed images can be excluded, and one pipeline evaluator compares class names
without spatial matching. Crop writers/readers disagree on paths; missing crops
can disappear from evaluation or misalign predictions. Vector insertion and
resume logic can skip or overwrite examples. These results do not establish
valid end-to-end detection or segmentation performance.

## Verification boundaries

The CPU suite checks command dispatch, import boundaries, selected artifact
behavior, and mocked Ollama lifecycle behavior. Artifact and dataset checks need
NumPy, pandas, Pillow, tqdm, and PyYAML; unavailable dependencies produce explicit
skips. Passing these checks does not verify real datasets, model accuracy,
network services, Qdrant integration, GPU execution, or HPC jobs.

The annotation scripts under `src/util/` include scripts with hard-coded paths
and import-time data writes, particularly `joint_labels.py`. Inspect them before
use; do not import them merely to discover their behavior.
