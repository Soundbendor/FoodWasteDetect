# Maintenance and submission scripts

Read this document when changing explicit smoke checks, service launchers, or Slurm
wrappers. Environment installation and routine verification commands belong in
[development](../development.md); model protocols and hardware requirements belong
in [encoders](encoders.md). Running model downloads, services, GPU/HPC work, or live
cloud requests requires authorization under the root guide.

## Package scripts

[`src/classification/scripts/smoke_jina.py`](../../src/classification/scripts/smoke_jina.py) is an
explicit real-checkpoint CPU check invoked as
`python -m classification.scripts.smoke_jina`. It constructs Jina CLIP v2, encodes one
text and one synthetic RGB image, and requires each result to be a finite, nonzero
NumPy float32 matrix of shape `(1, 1024)`. It can download weights; it is not part
of routine synthetic tests.

The parent creates a temporary directory and starts a worker subprocess using the
same Python executable. Before any ML imports, it assigns Hugging Face, Sentence
Transformers, Torch, XDG, and temporary-cache variables beneath that directory and
hides GPUs with `CUDA_VISIBLE_DEVICES=""`. On completion or interruption it waits
for the worker before removing caches, escalating termination to kill when needed.
It temporarily ignores further shutdown signals during cleanup and restores the
original handlers afterward. Preserve this ownership order when editing it.

[`tests/test_smoke_jina.py`](../../tests/test_smoke_jina.py) verifies cache isolation
and cleanup with lightweight subprocesses; it does not establish checkpoint
compatibility. Historical checkpoint results in [verification history](../experiments.md)
apply only to the recorded stacks, not automatically to current locks.

[`start_deepseek.sh`](../../src/classification/scripts/start_deepseek.sh) is a historical
shell launcher. It assumes `./deps/ollama/bin/ollama`, backgrounds the server, and
writes `~/ollama.log`; its `PORT` assignment does not configure the process. It has
no managed readiness or cleanup contract. Current description and download work
uses [the owned CLI lifecycle](data.md#class-descriptions-and-ollama).

## Slurm wrappers

Submission wrappers run from `SLURM_SUBMIT_DIR`, require prepared Pixi environments,
and use `pixi run --as-is`. The first positional argument, when supplied, is the
config file; following arguments pass through to the CLI. To pass overrides while
keeping the default YAML, supply that YAML explicitly as the first argument.

| Wrapper | Stage and environment |
| --- | --- |
| [`sbatch/run_embedding.sbatch`](../../sbatch/run_embedding.sbatch) | `experiment run`; `INTERNVL_PIXI_ENV`, default `cu126`. |
| [`sbatch/test_embedding.sbatch`](../../sbatch/test_embedding.sbatch) | `experiment evaluate`; `INTERNVL_PIXI_ENV`, default `cu126`. |
| [`scripts/generate_descriptions.sbatch`](../../scripts/generate_descriptions.sbatch) | `experiment describe`; fixed `cu126`. |

All three default to `cfg/classification.yml`. Model caches and Ollama models must
already be prepared when needed. Choose cluster resources and partitions manually;
the run/evaluation wrappers currently default to `dgx2`. The configured LanceDB
path must be accessible from participating nodes. Use separate output directories
for evaluation jobs and follow the writer/reader contract in
[experiments](experiments.md).

## Sweep submitters

[`submit_embedding_sweep.sh`](../../scripts/submit_embedding_sweep.sh) submits five
local models through `run_embedding.sbatch`. It assigns distinct reference tables
and output directories, exports `jina-cu126` for Jina models and `cu126` for the
others, and selects CUDA explicitly. Jina Embeddings v4 uses batch size 1; Qwen uses
8. Select suitable native-BF16 hardware for those models manually, such as through
`SBATCH_PARTITION`. The script does not add hardware detection or a batch fallback.

[`submit_image_count_sweep.sh`](../../scripts/submit_image_count_sweep.sh) defaults
to `cfg/classification_embeddinggemma2_images.yml`. It submits image-only
EmbeddingGemma2 runs with caps 1, 5, 10, 20, 50, and 100, separate tables/outputs,
and `cu126`. Selection semantics are defined in [data](data.md).

Both submitters accept at most one config path and require execution from the
repository root. Existing strict CLI override keys must be present in custom YAML.
Shared saved descriptions keep model comparisons on the same vocabulary; concurrent
jobs that generate descriptions need distinct Ollama endpoints. Wrapper checks in
[`tests/test_classification_cli.py`](../../tests/test_classification_cli.py) use fake executables and do
not submit jobs or validate HPC execution.
