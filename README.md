# FoodWasteDetect

FoodWasteDetect explores food recognition for food-waste sensing through two
retrieval-based research pipelines:

- **Image retrieval:** the primary direction is YOLO localization → Jina CLIP
  image embeddings → Qdrant category retrieval. This target architecture is
  partially implemented; the example configuration selects a different CLIP
  model.
- **Text retrieval:** the `internvl` package captions whole images, embeds the
  captions, and retrieves labeled descriptions from Qdrant. References come from
  Ollama-generated class descriptions or training-image captions. This pipeline
  performs whole-image classification without boxes or masks.

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

Inspect the text CLI without installing or loading models:

```sh
PYTHONPATH=src python -m internvl --help
```

### Ollama setup

Install the server using the
[official instructions](https://docs.ollama.com/linux); the Python client does
not include it. Set `ollama.binary_path` to its executable (the example uses
`deps/ollama/bin/ollama`) and `ollama.host` to an unused local endpoint. Keep
the model and timeout settings in the configuration.

The description and download commands start and stop their own server and reject
occupied endpoints. Download a model with:

```sh
internvl download_model gemma3:27b
```

Optionally set `OLLAMA_MODELS` to choose model storage. This command downloads
an Ollama model; the InternVL captioner checkpoint is configured separately.

## Commands

### Text experiments

Prefix each command below with `internvl` (or `python -m internvl`). Commands
marked **config** require `--config_file cfg/experiment_config_internvl.yml`.

| Command                     | Arguments                   | Purpose / output                                                                   |
| --------------------------- | --------------------------- | ---------------------------------------------------------------------------------- |
| `download_model MODEL_NAME` | Positional model name       | Download Ollama weights; always reads `cfg/experiment_config_internvl.yml`.        |
| `generate-descriptions`     | config                      | Generate five descriptions per FoodX251 class into the configured JSON dictionary. |
| `caption-images`            | config, `--partition 1`–`4` | Caption one quarter of the training images into a partition CSV.                   |
| `merge-captions`            | None                        | Merge four partition CSVs into `assets/foodx251_captions.txt`.                     |
| `convert-captions`          | None                        | Group merged captions by class into `foodx251_captions.json`.                      |
| `build-db`                  | config                      | Embed reference descriptions and insert them into Qdrant.                          |
| `evaluate`                  | config                      | Caption validation images and report retrieval classification accuracy.            |
| `finetune-embeddings`       | config                      | Train the text embedder using descriptions and image captions.                     |

For example:

```sh
internvl generate-descriptions --config_file cfg/experiment_config_internvl.yml
```

Captioning, indexing, evaluation, and training have
[current limitations](docs/experiments.md#text-workflow-limitations).

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

| Module                                                                            | Responsibility                                                                                            |
| --------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| [internvl/model.py](src/internvl/model.py)                                        | InternVL loading, image preprocessing, and caption inference.                                             |
| [internvl/generate_descriptors.py](src/internvl/generate_descriptors.py)          | Ollama lifecycle, model downloads, and class descriptions.                                                |
| [internvl/embeddings.py](src/internvl/embeddings.py)                              | Text embeddings, fine-tuning, and triplet evaluation.                                                     |
| [internvl/workflows.py](src/internvl/workflows.py), [cli.py](src/internvl/cli.py) | Text experiment orchestration and command dispatch.                                                       |
| [app.py](src/app.py), [yolo.py](src/yolo.py)                                      | Image-retrieval entry points, crop experiments, and detector evaluation.                                  |
| [clip_finetune.py](src/clip_finetune.py)                                          | CLIP image/text fine-tuning experiments.                                                                  |
| [database/](src/database)                                                         | Qdrant retrieval/scoring (`vecdb`), image embeddings (`clip_embedding`), and detector crops (`img_crop`). |
| [data_wrappers/](src/data_wrappers)                                               | Dataset tables, splits, labels, crops, and download utilities.                                            |
| [util.py](src/util.py), [util/](src/util)                                         | Configuration/evaluation helpers and standalone annotation/conversion scripts.                            |

[cfg/](cfg) holds experiment settings; [scripts/](scripts) and [sbatch/](sbatch)
hold cluster jobs; [tests/](tests) contains CPU checks.

## Checks

```sh
PYTHONPATH=src python -m unittest discover -s tests -v
```

Artifact and dataset checks need NumPy, pandas, Pillow, tqdm, and PyYAML;
missing dependencies produce skips. Tests use synthetic fixtures and mocked
backends, without starting models or services; they do not establish model
accuracy or Qdrant/GPU/HPC integration.
