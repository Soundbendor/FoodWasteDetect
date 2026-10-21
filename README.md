# FoodWasteDetect

FoodWasteDetect explores food recognition for food-waste sensing through offline
image-retrieval experiments:

- Image retrieval: Currently, this experiment pipeline supports YOLO for object
  localization localization → Jina CLIP image embeddings → Qdrant vector
  retrieval. (Pending deprecation)
  - Whole-image classification: the `internvl` package uses Jina CLIP v2 and
    Qdrant to classify images using references drawn from training images, class
    names, and generated descriptions. The setup below runs this baseline.

> [!NOTE] These experiments have known runtime and evaluation limitations. See
> [experiment notes](docs/experiments.md) before running a workflow or
> interpreting its results.

## Setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and run
all commands from the repository root.

### Install the environment

```sh
uv python install 3.13.13
uv sync --project src/internvl --locked --extra cpu --managed-python
uv run --project src/internvl --locked --extra cpu internvl --help
```

For GPU experiments, use `--extra cu126` in both sync and run commands to
install CUDA-specific dependencies. Check that PyTorch can access the GPU with:

```sh
uv sync --project src/internvl --locked --extra cu126 --managed-python
uv run --project src/internvl --locked --extra cu126 python -c \
  "import torch; print(torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

The profiles pin PyTorch 2.7.1 and torchvision 0.22.1 with Python 3.13.13. The
CUDA 12.6 profile targets V100 compatibility; run the
[V100 kernel check](src/internvl/README.md#v100-kernel-check) inside your
allocation. The [environment guide](src/internvl/README.md) covers dependencies
and checks.

### Configure the dataset and services

Copy the example for your experiment:

```sh
cp cfg/classification.yml cfg/classification_hpc.yml
```

The whole-image experiment supports FoodX251. Set `dataset.root` to its local
directory, with this layout:

```text
annot/class_list.txt
annot/train_info.csv
annot/val_info.csv
train/train_set/       # training images
val/val_set/           # labeled validation images
```

Update these settings in your copied YAML:

| Setting                    | Value to supply                                            |
| -------------------------- | ---------------------------------------------------------- |
| `embedding.device`         | `cuda` for GPU execution, or `cpu`                         |
| `embedding.batch_size`     | Start with 32; reduce based on available VRAM              |
| `ollama.binary_path`       | Absolute path to the installed Ollama executable           |
| `ollama.host`              | An unused local endpoint, such as `http://127.0.0.1:11434` |
| `qdrant.connection.url`    | Running Qdrant service reachable from the compute node     |
| `qdrant.collection_prefix` | The exact Qdrant collection name                           |
| `output_dir`               | Writable output directory outside the dataset              |

Choose at least one reference source by setting these YAML booleans:

| Setting              | References                          | Default |
| -------------------- | ----------------------------------- | ------- |
| `embed_images`       | All training images                 | `false` |
| `embed_class_labels` | Class names                         | `false` |
| `embed_descriptions` | Ollama-generated class descriptions | `true`  |

The example uses `description_count: 5` descriptions per class, batches of 32,
CPU execution, and `k: 10` retrieved neighbors per query. We recommend using a
separate Qdrant collection and output directory for each experiment.

### Qdrant setup

Qdrant is a vector database that stores reference embeddings and retrieves
similar examples for classification. See the
[Qdrant installation guide on GitHub](https://github.com/qdrant/qdrant#client-server).
Create a persistent storage directory, then choose Docker or Singularity:

```sh
mkdir -p qdrant_storage
```

With Docker:

```sh
docker run --rm -p 6333:6333 \
  -v "$(pwd)/qdrant_storage:/qdrant/storage" qdrant/qdrant:v1.12.0
```

On HPC, pull the Singularity image on a node with internet access, then run it
inside an approved allocation. Replace `{PORT}` with an available port, such as
6333:

```sh
singularity pull qdrant.sif docker://qdrant/qdrant:v1.12.0
singularity exec \
  --bind "$(pwd)/qdrant_storage:/qdrant/storage" \
  --env QDRANT__SERVICE__HTTP_PORT={PORT} qdrant.sif /qdrant/qdrant
```

Keep the service running during indexing and evaluation. Set
`qdrant.connection.url` to `http://<service-node>:<port>`; use `localhost` only
when Qdrant and the experiment run on the same node. These container commands
have not been verified against the project's locked client.

### Ollama setup

Install the [Ollama executable](https://docs.ollama.com/linux) separately.
Choose persistent storage paths for your model cache, and export these variables
in each job (replace the example paths):

```sh
export HF_HOME=/your/storage/huggingface
export OLLAMA_MODELS=/your/storage/ollama
```

Set the Ollama binary and host in both your experiment YAML and
`cfg/experiment_config_internvl.yml`: the download command always reads the
latter and has no configuration flag. Then download the configured model:

```sh
uv run --project src/internvl --locked --extra cu126 internvl download_model gemma3:27b
```

Do not start Ollama manually: the download and description commands start and
stop their own server. The embedding model downloads automatically on first use.
If compute nodes lack internet, populate its cache beforehand on a permitted
node, using the same `HF_HOME` available to the job:

```sh
uv run --project src/internvl --locked --extra cu126 python -c \
  "from internvl.embeddings import EmbeddingModel; EmbeddingModel('jinaai/jina-clip-v2', device='cpu')"
```

The temporary smoke test deletes its downloads and does not prepare this cache.

### Run your first experiment

Inside your allocation, run each stage separately to check setup:

```sh
uv run --project src/internvl --locked --extra cu126 internvl experiment describe --config_file cfg/classification_hpc.yml
uv run --project src/internvl --locked --extra cu126 internvl experiment index --config_file cfg/classification_hpc.yml
uv run --project src/internvl --locked --extra cu126 internvl experiment evaluate --config_file cfg/classification_hpc.yml
```

Use `experiment run` to execute all three stages in one command. For CPU runs,
use `--extra cpu` throughout and set `embedding.device: cpu`.

`describe` skips generation when `embed_descriptions` is disabled and reuses
`output_dir/descriptions.json` when it exists. Remove that file to generate new
descriptions. For Slurm,
[generate_descriptions.sbatch](scripts/generate_descriptions.sbatch) accepts a
configuration path as its first argument and defaults to
`cfg/classification.yml`.

> [!IMPORTANT] Existing collections are reused without inserting records or
> checking compatibility or completeness. After a failed index build, choose a
> new collection name before retrying. CUDA execution and HPC behavior have not
> been verified by the local CPU checks.

## Whole-image classification

The baseline classifies whole images localization:

1. `describe` optionally uses Ollama to generate visual descriptions from known
   class names.
2. `index` embeds the selected reference sources with Jina CLIP v2 and stores
   their vectors and class labels in Qdrant. Image references use the complete
   FoodX251 training split.
3. `evaluate` embeds each validation image with the same model, retrieves up to
   `k` neighbors by cosine similarity, and predicts a class by majority vote.
   Each retrieved record contributes one vote.

## Results

Successful stages save these files under `output_dir`:

| File                                     | Contents                                                                                 |
| ---------------------------------------- | ---------------------------------------------------------------------------------------- |
| `descriptions.json`                      | Generated class descriptions.                                                            |
| `index.json`                             | Collection name, reference-source counts, and indexing configuration.                    |
| `evaluation_<UTC timestamp>/report.json` | Evaluation configuration, collection, reference-source counts, predictions, and metrics. |

Evaluation covers the complete labeled FoodX251 validation split; these results
do not measure official test accuracy. Metrics include the evaluated image
count, majority-vote accuracy, nearest-neighbor accuracy, and per-class support,
correct counts, and recall. Prediction `vote_fraction` is the share of returned
neighbors supporting the winning class, not calibrated confidence.

Image-loading, embedding, or service errors stop the run without a partial
evaluation report.

## Troubleshooting

You can run the unit test suite locally, without downloading CUDA dependencies
with the following command:

```sh
uv run --project src/internvl --locked --extra cpu pytest -c src/internvl/pyproject.toml
```

Routine checks use synthetic inputs, mocked model and Ollama backends, and
in-memory Qdrant. They do not establish model accuracy or GPU/HPC compatibility.

## Further reading

- [Environment guide](src/internvl/README.md): dependency details, linting, type
  checks, and explicit model and GPU checks.
- [Image-workflow limitations](docs/experiments.md#image-workflow-limitations):
  current status of the YOLO/CLIP experiments and legacy commands.
- [Developer guide](AGENTS.md): implementation contracts, source maps, and
  research constraints.
