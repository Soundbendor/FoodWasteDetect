# Maintenance and submission scripts

Read this document when changing explicit smoke checks, service launchers, or Slurm
wrappers. Environment installation and routine verification commands belong in
[development](../development.md); model protocols and hardware requirements belong
in [encoders](encoders.md). Running model downloads, services, GPU/HPC work, or live
cloud requests requires authorization under the root guide.

## Package scripts

From the classification Pixi environment, run
`python -m classification.scripts.check_dataset --config-file CONFIG`
to compare the loader's class count, label coverage and referenced paths against
recorded upstream counts and the installed file inventory. It prints each error and
exits nonzero on failure. Image files are inventoried without decoding their contents.

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

## Full-width training and validation caches

From the repository root, submit `sbatch sbatch/cache_full_width_images.sbatch MODEL`
with a suitable GPU partition and account. Choose `qwen`, `jina_v4`, or
`embeddinggemma2`. Prepare model weights and the CUDA Pixi environment first.
The recipes embed all FoodX251 training images at native width, cache validation
images and evaluate. Each model uses distinct reference and validation tables.
Jina uses `jina-cu126`. The other two models use `cu126`.

## Slurm wrappers

Submission wrappers run from `SLURM_SUBMIT_DIR`, require prepared Pixi environments,
and use `pixi run --as-is`. For the first three wrappers, the first positional
argument, when supplied, is the config file; following arguments pass through to
the CLI. To pass overrides while keeping the default YAML, supply that YAML
explicitly as the first argument.

| Wrapper | Stage and environment |
| --- | --- |
| [`sbatch/run_embedding.sbatch`](../../sbatch/run_embedding.sbatch) | `experiment run`; `INTERNVL_PIXI_ENV`, default `cu126`. |
| [`sbatch/test_embedding.sbatch`](../../sbatch/test_embedding.sbatch) | `experiment evaluate`; `INTERNVL_PIXI_ENV`, default `cu126`. |
| [`scripts/generate_descriptions.sbatch`](../../scripts/generate_descriptions.sbatch) | `experiment describe`; fixed `cu126`. |
| [`sbatch/run_matroyshka_sweep.sbatch`](../../sbatch/run_matroyshka_sweep.sbatch) | `experiment matroyshka-sweep`; fixed CPU `default`. Requires one prepared config and passes nothing else through. |

The first three default to `cfg/classification.yml`. Model caches and Ollama models must
already be prepared when needed. Choose cluster resources and partitions manually;
the run/evaluation wrappers currently default to `dgx2`. The configured LanceDB
path must be accessible from participating nodes. Use separate output directories
for evaluation jobs and follow the writer/reader contract in
[experiments](experiments.md).

### Dataset download acceptance

From the repository root, submit `sbatch sbatch/download_datasets.sbatch LOG_DIR`
with the account, partition and CPU architecture appropriate to the installed Pixi
environment. This job downloads five datasets and checks each installation. Use a
new log directory for each run. Read `summary.tsv` for per-dataset outcomes and
`<dataset>.download.log` or `<dataset>.check.log` for failures. This workflow requires authorization for real downloads and storage.

## Sweep submitters

### Cached dimension jobs

Copy `cfg/matroyshka_sources.example.json` and select existing index summaries
and validation tables. From the repository root run
`bash scripts/submit_matroyshka_sweep.sh --dry-run SOURCES.json` to prepare the
configs and preview the commands. Omit `--dry-run` to submit one CPU job per
usable model. Set `SBATCH_ACCOUNT` and `SBATCH_PARTITION` as needed. Both modes
save resolved configs, `jobs.tsv`, and `manifest.json` under `assets/matroyshka`,
configurable via `MATROYSHKA_OUTPUT_ROOT`. The manifest lists every expected run
with its `run`, `source`, `model_id`, and `dimension`; each prepared config's
`result_file` sends that run's record to `results/<run>.json` in the same
directory. Submission records job IDs in `submitted.tsv`, then submits
`plot_matroyshka_sweep.sbatch` with `--dependency=afterany:` on all of them, so
collection and plotting run once every sweep has ended. Prepare the plotting
environment before submitting.
Preparation reads the stored width from the reference table and sweeps the grid
widths below it plus that width. `max_dimension` is an optional assertion. It
runs the sweep's source checks (see [experiments](experiments.md#cached-matryoshka-sweep))
and skips any source that fails them or shares a reference or validation table
with an earlier ready source, since derived table names are shared. Do not run
concurrent sweeps from the same tables across manifests either.

### Embedding and image-count submitters

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

## Collecting and plotting results

`scripts/collect_results.py` and `scripts/plot_matroyshka_sweep.py` run in the
plotting environment and import nothing from the package. A group directory holds
`manifest.json` and `results/<run>.json` records ([format](experiments.md#artifact-and-failure-contract)).

Run `pixi run --manifest-path scripts/plotting/pixi.toml --as-is collect GROUP_DIR`
from the repository root. It writes `results.csv` with one row per manifest run:
the manifest fields, `status` (`done` or `missing`), aggregate metrics, and the
report path. It exits nonzero when runs are missing unless `--allow-partial` is
given, and always when none are done. Dataset, `k`, reference switches, image cap,
and evaluated count must match across runs; a setting named as a manifest field is
a declared axis and may vary.

Run `pixi run --manifest-path scripts/plotting/pixi.toml --as-is plot GROUP_DIR/results.csv --output accuracy.png`
to draw majority-vote accuracy against dimension, one curve per `model_id`. Missing
runs are omitted and the title is marked partial. `sbatch sbatch/plot_matroyshka_sweep.sbatch GROUP_DIR`
runs both steps with `--allow-partial` and writes `GROUP_DIR/plots/matroyshka_accuracy.png`;
the dimension submitter queues it automatically.

To add another sweep type, have its submitter write a manifest and pass each run
`--set result_file=GROUP_DIR/results/<run>.json`.
