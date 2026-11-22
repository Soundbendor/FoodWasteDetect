# Maintenance and submission scripts

Read this document when changing explicit smoke checks, service launchers, or Slurm
wrappers. Environment installation and routine verification commands belong in
[development](../development.md); model protocols and hardware requirements belong
in [encoders](encoders.md). Running model downloads, services, GPU/HPC work, or live
cloud requests requires authorization under the root guide.

## Package scripts

For a VegFru-only download/install retry, submit
`sbatch sbatch/download_vegfru.sbatch` from the repository root. It uses
`soundbendor` for both partition and account, two AVX2 CPUs, 20 GB RAM,
and the prepared Pixi `default` environment. It runs only `download_dataset`
with `cfg/classification_vegfru.yml`; completed archives are reused.

`classification.scripts.check_dataset --config-file FILE --report-file FILE`
checks the configured dataset's complete train/evaluation metadata against raw
annotation counts and an image-file inventory. It never decodes images. Reports
include class coverage, missing files, unused images, and filename/identity digests
for comparison across repeated downloads. UEC counts are additionally compared
with the recorded upstream inventory. Errors produce a failed JSON report and a
nonzero exit status.

Submit `sbatch sbatch/check_datasets.sbatch logs/dataset-acceptance/RUN_ID` from
the repository root for authorized real downloads. It uses the prepared Pixi
`default` CPU environment, the `soundbendor` partition and account, an AVX2-capable CPU, two CPUs,
20 GB memory, and 48 hours. The first real run on Ivy Bridge failed with an
illegal instruction during CLI startup; the wrapper excludes those older CPUs
and disables core dumps.
It processes all five datasets sequentially, checks free disk headroom before
each download, and probes ISIA metadata before its large image payload. Each
successful installation is checked, downloaded again, and checked for unchanged
counts and filename/identity digests. `summary.tsv` records download, validation,
and repeat outcomes; a failure does not prevent subsequent datasets from running.
Free filesystem capacity does not establish user quota. Partial transfers restart;
completed archives are reused. Failed artifacts remain for diagnosis.

These checks establish metadata and referenced-file completeness, not image-content
integrity. Download logs and reports stay under ignored `logs/`; configured dataset
roots stay under ignored `data/`. No models or services are started.

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

### Cached dimension sweep submission

Run `bash scripts/submit_matroyshka_sweep.sh --dry-run` to inspect ready jobs,
or omit `--dry-run` to submit one CPU job per ready model. An optional final JSON
path replaces `cfg/matroyshka_sources.json`. Both modes write resolved JSON configs
and `jobs.tsv` below `assets/matroyshka/<timestamp>_<pid>`; set
`MATROYSHKA_OUTPUT_ROOT` to change that parent. Submission also saves job IDs in
`submitted.tsv`. If submission fails partway through, earlier jobs remain submitted.

The sources file selects each model's saved `index.json` and validation table.
`classification.scripts.prepare_matroyshka_sweep` preserves the saved dataset,
reference strategy, prompt, and retrieval settings. It checks table existence,
nonzero row counts, full vector widths, and available model/prompt build metadata.
Legacy tables without build metadata rely on the explicitly selected index report
and cache; provenance and cache completeness are not established by these checks.
Missing or invalid pairs are skipped with a reason; no ready pairs exits nonzero.
No embeddings, caches, or source tables are created by preparation.

The checked-in selection points Qwen, Jina v4, and EmbeddingGemma2 to new
`*_images_all_full` reference runs and `*_validation_all_full` validation tables.
These use full-width vectors, all training images, and no text references. Run
the GPU jobs described below to populate them before submitting cached sweeps.
Gemini retains its existing compact reference and validation tables; neither
of the GPU preparation jobs nor their configs selects the paid Vertex backend.

Each job evaluates `[128,256,512,768,1024,1536,2048,3072,4096]` filtered to the
model's declared maximum. The CPU wrapper uses `soundbendor` partition/account,
AVX2, eight CPUs, 32 GB RAM, 24 hours, and the prepared Pixi `default` environment.
Standard `SBATCH_*` variables can override submission resources. It hides GPUs,
sets Hugging Face offline mode, and defaults W&B to offline (override with
`WANDB_MODE`). Cached sweeps never construct an encoder, including for Jina.
Jobs create normalized prefix tables and evaluation reports, but do not generate
the combined plot. Avoid overlapping sweeps using the same source tables because
they share derived table names; readable existing derived tables are reused.

Synthetic preparation and fake-submission checks live in
`tests/test_matroyshka_submission.py`.

### Full-width local image cache generation

Submit `sbatch sbatch/cache_full_width_images.sbatch MODEL`, where MODEL is
`qwen`, `jina_v4`, or `embeddinggemma2`. The wrapper selects
`cfg/classification_MODEL_images_all_full.yml` and runs `experiment run`:
build all training-image embeddings, cache all validation-image embeddings, and
evaluate. Configs disable class-name/description references and set
`n_imgs_per_class: null`, `truncate_dim: null`, and validation caching enabled.
All six tables have fresh names; prior mixed text/image and capped tables are
unused. Rerunning resumes these new tables from completed batches.

Jobs request one A40 GPU on `ampere`, account `eecs`, four CPUs, 48 GB host RAM,
and 48 hours. Qwen uses batch size 2, Jina v4 uses 1, and Gemma uses 16.
Prepare locked Pixi `cu126` and `jina-cu126` environments before submission;
the wrapper uses `--as-is`, with `jina-cu126` for Jina and `cu126` otherwise.
Cached model weights must be available because Hugging Face offline mode is
enabled. W&B defaults to offline. Gemini is excluded from the wrapper's model
allowlist; its existing embeddings remain selected for subsequent CPU sweeps.

### Plotting accuracy against dimension

`scripts/plot_matroyshka_sweep.py` reads `report.json` files recursively from
one submission directory or several explicitly selected sweep directories.
It plots `metrics.majority_vote_accuracy` as a percentage, using
`config.vector_dimension` (or `embedding.truncate_dim`) for the x coordinate.
It does not use nearest-neighbor accuracy. Model IDs select the four curves.

The plotting environment is independent of the model environments:

```sh
pixi install --manifest-path scripts/plotting/pixi.toml --locked
pixi run --manifest-path scripts/plotting/pixi.toml --as-is plot \
  assets/matroyshka/RUN_ID --output assets/plots/matroyshka_accuracy.png
```

Replace `RUN_ID` with the directory printed by the sweep submitter. To combine
jobs from different submissions, pass their specific model sweep directories as
multiple positional arguments. Avoid passing all historical experiments together:
duplicate model/dimension points are rejected rather than silently selecting or
averaging runs. Dataset settings, reference switches, image cap, `k`, and evaluated
image count must match across reports. These checks do not establish cache provenance.

By default, all 28 points in the requested grid across Qwen, Jina v4, Gemma, and
Gemini are required. `--allow-partial` permits unfinished sweeps, warns about missing
dimensions, and labels the graph as partial. Available points are sorted numerically;
each curve ends at that model's largest available dimension. Output may be PNG,
SVG, or PDF. A sibling CSV records raw accuracy fractions and source report paths.
The script reads reports only and performs no model inference or cloud requests.

Run its synthetic checks with
`pixi run --manifest-path scripts/plotting/pixi.toml --as-is test`.

To queue the graph after all four CPU sweeps succeed, use
`sbatch --dependency=afterok:QWEN_JOB:JINA_JOB:GEMMA_JOB:GEMINI_JOB sbatch/plot_matroyshka_sweep.sbatch assets/matroyshka/RUN_ID`,
replacing the job IDs and run directory. The wrapper requests one CPU, 4 GB RAM,
and 30 minutes on `soundbendor`, using the installed plotting Pixi environment.
It writes `plots/matroyshka_accuracy.png` and a sibling CSV inside that submission
directory. It requires all 28 sweep points, including the native-width points;
standalone full-width experiment reports are not included again. No GPU or model
inference is involved.
