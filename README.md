# FoodWasteDetect

FoodWasteDetect explores food recognition for food-waste sensing through two
retrieval-based computer vision pipelines and a whole-image classification
baseline:

- **Image retrieval:** the primary direction is YOLO localization → Jina CLIP
  image embeddings → Qdrant category retrieval. This target architecture is
  partially implemented; the example configuration selects a different CLIP
  model.
- **Whole-image classification:** `internvl.experiment` embeds images with a
  pretrained Jina CLIP v2, storing class names, generated descriptions, and
  training images in a Qdrant vector database. Images are classified at
  inference time by nearest-neighbor search within the vector database.

> [!NOTE] These are offline experiments with known runtime and evaluation
> limitations. See [experiment notes](docs/experiments.md) before running a
> workflow or interpreting its results.

## Setup

From the repository root, activate your research Python environment and install:

```sh
python -m pip install -e .
```

Runtime dependencies are managed separately; installation does not supply them.
The [environment file](environment.yml) is machine-specific and incomplete.
Configure dataset/model paths and Qdrant settings in the
[InternVL configuration](cfg/experiment_config_internvl.yml) or
[CLIP configuration](cfg/experiment_config_clip.yml). InternVL captioning uses
CUDA. Run commands from the repository root.

To view command line options:

```sh
PYTHONPATH=src python -m internvl --help
```

### Ollama setup

Ollama is used to to generate image captions and food item descriptions. Install
the server using the [official instructions](https://docs.ollama.com/linux) In
the `cfg/` file, be sure to set `ollama.binary_path` to the Ollama executable
(the example uses `deps/ollama/bin/ollama`) and `ollama.host` to an unused local
port.

The description and download commands start and stop their own server and reject
occupied endpoints. Download a model with:

```sh
internvl download_model gemma3:27b
```

Optionally set `OLLAMA_MODELS` to choose model storage. This command downloads
an Ollama model; the InternVL captioner checkpoint is configured separately.

## Commands

### Ollama downloads

Run `internvl download_model MODEL_NAME` (or
`python -m internvl download_model MODEL_NAME`). The positional model name is
required. The command always reads `cfg/experiment_config_internvl.yml` from the
working directory and accepts no configuration flag.

Generate class descriptions through the whole-image experiment:

```sh
python -m internvl experiment describe --config_file cfg/classification.yml
```

Description generation follows the experiment configuration:
`embed_descriptions` must be enabled, existing `output_dir/descriptions.json` is
reused, and newly generated descriptions are saved there. The Slurm script
`scripts/generate_descriptions.sbatch` defaults to this configuration and
accepts an optional configuration path as its first argument. Existing artifacts
are not migrated.

### Whole-image classification

Configure [cfg/classification.yml](cfg/classification.yml), then run:

```sh
PYTHONPATH=src python -m internvl experiment run --config_file cfg/classification.yml
```

The same command accepts `describe`, `index`, or `evaluate` in place of `run`.
Argument parsing and YAML loading live in `internvl.cli`. Use the shared CLI;
`internvl.experiment` has no command-line entry point. The `index` CLI command
calls `Experiment.build_vector_database()`; `describe` calls
`Experiment.generate_descriptions()`. `describe` is a no-op when descriptions
are disabled. `index` reads the saved descriptions when that source is enabled.
`run` closes its owned Ollama server before constructing the embedder. Neither
CLI help nor imports load models. Constructing `Experiment` loads dataset
metadata but does not create directories. Disabled and cached description stages
also require dataset metadata. Description generation calls
`DescriberLLM.generate_descriptions(classes, count)`, which loops over class
names using `describe_class(label)` and rejects empty descriptions. The
experiment owns saved-file handling and the Ollama context.

Choose `embed_images`, `embed_class_labels`, and `embed_descriptions`; all seven
nonempty combinations are supported. The default uses five generated
descriptions per class, batches of 32, CPU, and k=10. Each reference record gets
one equal vote in the pooled exact cosine top-k. Compare results as complete
retrieval strategies: changing sources changes reference counts as well as
wording. No partitions, subsampling, parameter tuning, or query-embedding cache
are used. Reference switches must be YAML booleans, with at least one enabled
source; `validate_config()` checks these during construction. Use positive
integers for batch size, evaluation `k`, and generated description count; the
runner does not validate these numeric settings.

FoodX251 references use the complete official training table. Evaluation uses
the complete **labeled validation split** (`source_split="val"`), never the
unlabeled official test set. The adapter preserves legacy filenames, resolves
paths, validates labels, and rejects cross-split path/content duplication.
Images and raw annotations are never rewritten. Set a separate output directory
for every reference strategy you want to keep separately; paths are resolved
from the working directory.

The pretrained embedder uses Sentence Transformers with Jina CLIP v2 and full
1,024-dimensional float32 vectors. The wrapper does not request normalization;
Qdrant handles normalization for cosine retrieval. Its text and image methods
both delegate to `SentenceTransformer.encode()`. Model and code revisions are
not configured or tracked for this experiment. Sentence Transformers handles
model, nested text encoder, tokenizer, and processor loading through its normal
caching and download behavior. The runner does not enforce offline mode.
Attention accelerators are disabled to keep device selection explicit. Local
development tests mock pretrained loading and use synthetic encoders, so they do
not download weights or require a populated model cache. Runtime packages
include PyTorch, Sentence Transformers (>=5.4,<6), Transformers,
huggingface-hub, einops, timm, requests, Pillow, NumPy, pandas, tqdm, datasets,
qdrant-client, PyYAML, and Ollama's Python client. Real-model compatibility
still requires a separately authorized run; fixture tests do not validate it.
See the [Jina model card](https://huggingface.co/jinaai/jina-clip-v2) for the
upstream text/image API.

Artifacts in the output directory:

- `descriptions.json`: generated description dictionary. Existing descriptions
  are loaded directly; remove this file to generate them again.
- `index.json`: latest collection name, reference-source counts, and
  configuration.
- `evaluation_<UTC timestamp>/report.json`: configuration, collection,
  reference-source counts, predictions, and metrics. Folder timestamps use
  `YYYYMMDDTHHMMSS_microsecondsZ`, for example
  `evaluation_20261019T153042_123456Z`.

Artifact validation is deferred. There are no hashes, compatibility checks,
manifest comparisons, or collection-content verification.
`qdrant.collection_prefix` is now the exact collection name; no UUID is
appended. If it exists, `index` warns and returns without loading the embedder,
inserting records, or rewriting `index.json`. Otherwise, it creates that
collection and saves `index.json` after successful insertion. Choose a different
name to build another collection. To reuse an older UUID-suffixed collection,
set the YAML value to its full name. `evaluate` queries the configured
collection. It includes stored source counts from a local `index.json` only when
its collection name matches; otherwise `reference_source_counts` is `null`. This
allows reuse from a new output directory without inventing metadata for
precomputed vectors. No compatibility or completeness checks are performed, so
an existing partial collection is also reused. `run()` still performs the
description stage before checking the collection. `Experiment(config)` loads
dataset metadata and class names during construction, including for disabled or
cached description commands. The batch size defaults to 32 and is stored once.
The embedder and Qdrant client are constructed on first use and reused. `run()`
finishes description generation before loading the embedder. Beyond
reference-switch validation, backend checks and Python errors propagate. The
independent FoodX251 adapter reads `annot/class_list.txt`,
`annot/train_info.csv`, and `annot/val_info.csv` directly. It exposes a Hugging
Face `DatasetDict` as `splits`, with `train` and `validation` keys and explicit
column types. Class IDs map to the existing naturalized class names. It does not
import the legacy dataset adapter. Pandas reads the annotations, and
`Dataset.from_pandas()` converts the resulting tables. `train_path` points to
`train/train_set`; `val_path` and the evaluation-facing `test_path` point to
`val/val_set`. `test_set()` returns labeled validation data, never the unlabeled
official test split. Its `iter_batches()` uses `Dataset.iter()` to load images
in annotation order, including the final partial batch. Images are decoded once
per pass and passed directly to the embedder. Missing, unreadable, or
unembeddable images stop the run without retries or exclusions. Errors and
interrupts propagate without saving partial reports. Failed indexing can leave a
partial collection; no rollback or cleanup is added. The previous successful
`index.json` remains in place if a rebuild or summary write fails. JSON writes
use temporary-file replacement. Old artifacts stay intact. Every class needs a
training image reference when that source is enabled. Generated descriptions
must be nonempty.

Votes tie-break by best individual similarity, then class name. `vote_fraction`
is winning votes / actual returned neighbors, not calibrated confidence. Empty
results abstain and count as incorrect. Completed evaluations use the entire
validation split, with exactly one prediction per image. Reports retain the
evaluated count, majority-vote and nearest-neighbor accuracy, and per-class
support/correct/recall. There is no top-five metric. Prediction records keep
image identity, ground truth, the predicted label, vote fraction, nearest label,
and neighbor details; `fname` is not duplicated as `value`. Strict row/vector
pairing rejects mismatches. Image reference records are constructed directly
from decoded training batches. Indexing and evaluation rely on garbage
collection to release decoded batch images. Qdrant failures also abort; an empty
evaluation split fails with undefined accuracy.

The old `internvl build-db` and top-level `evaluate` commands have been removed.
Use `internvl experiment index` and `internvl experiment evaluate` instead. The
`finetune-embeddings` command and its workflow have been removed, along with BGE
training/triplet evaluation. The legacy `internvl.workflows` module and the
`caption-images`, `merge-captions`, `convert-captions`, and
`generate-descriptions` commands have also been removed without compatibility
aliases. The standalone InternVL model and separate `clip_finetune.py`
experiment remain available.

### Image experiments

These entry points take `--config_file PATH`. Their
[wiring and configuration need repair](docs/experiments.md#image-workflow-limitations).

| Command         | Intended purpose                                                  |
| --------------- | ----------------------------------------------------------------- |
| `build_clip_db` | Index FoodX251 training-image embeddings in Qdrant.               |
| `test_clip`     | Evaluate image-embedding retrieval on FoodX251 validation images. |
| `test_yolo`     | Run a YOLO-World food-class detection baseline.                   |

## Module map

Paths below are relative to `src/`.

| Module                                                                              | Responsibility                                                                                            |
| ----------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| [internvl/model.py](src/internvl/model.py)                                          | InternVL loading, image preprocessing, and caption inference.                                             |
| [internvl/generate_descriptors.py](src/internvl/generate_descriptors.py)            | Ollama lifecycle, model downloads, and class descriptions.                                                |
| [internvl/embeddings.py](src/internvl/embeddings.py)                                | Inference-only Jina CLIP text/image embeddings.                                                           |
| [internvl/experiment.py](src/internvl/experiment.py), [cli.py](src/internvl/cli.py) | Whole-image experiment orchestration, command dispatch, and Ollama download wrapper.                      |
| [app.py](src/app.py), [yolo.py](src/yolo.py)                                        | Image-retrieval entry points, crop experiments, and detector evaluation.                                  |
| [clip_finetune.py](src/clip_finetune.py)                                            | CLIP image/text fine-tuning experiments.                                                                  |
| [database/](src/database)                                                           | Qdrant retrieval/scoring (`vecdb`), image embeddings (`clip_embedding`), and detector crops (`img_crop`). |
| [data_wrappers/](src/data_wrappers)                                                 | Dataset tables, splits, labels, crops, and download utilities.                                            |
| [util.py](src/util.py), [util/](src/util)                                           | Configuration/evaluation helpers and standalone annotation/conversion scripts.                            |

[cfg/](cfg) holds experiment settings; [scripts/](scripts) and [sbatch/](sbatch)
hold cluster jobs; [tests/](tests) contains CPU checks.

## Checks

```sh
PYTHONPATH=src python -m unittest discover -s tests -v
```

Artifact and dataset checks need NumPy, pandas, Pillow, tqdm, and PyYAML;
whole-image tests additionally need datasets, qdrant-client, and the Ollama
Python client; the embedder import and encoding tests require Sentence
Transformers and CPU PyTorch. missing dependencies produce skips. Install the
CPU test dependencies with:

```sh
uv run --no-project --with numpy --with 'pandas<3' --with Pillow --with tqdm \
  --with PyYAML --with datasets --with qdrant-client --with ollama --with requests \
  --with 'sentence-transformers>=5.4,<6' \
  --with 'torch==2.10.0+cpu' --extra-index-url https://download.pytorch.org/whl/cpu \
  --index-strategy unsafe-best-match \
  python -m unittest discover -s tests -v
```

The pandas upper bound supports the separate legacy adapter; the whole-image
loader no longer uses its pandas reader. Run this command with `PYTHONPATH=src`.
The CPU wheel above targets Linux/Windows; use the platform's PyTorch wheel on
macOS. The legacy YAML command test also requires requests because `util`
imports the legacy vector database module. Imports used by annotations are
ordinary runtime imports. Tests do not restrict dependency imports; importing a
library does not authorize model construction, downloads, or service startup.
Tests use synthetic fixtures and mocked backends plus a tiny synthetic Sentence
Transformers module, without loading pretrained models or starting services;
they do not establish model accuracy or real-model/remote-Qdrant/GPU/HPC
integration. New tests use an in-memory Qdrant client, not a live service.
