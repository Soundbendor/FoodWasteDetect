# Experiment notes

These notes describe current behavior and limitations of the offline experiments.
See the [README](../README.md) for setup and commands, and the
[developer guide](../AGENTS.md) for source maps and implementation contracts.

## Text retrieval workflow (removed)

The `internvl.workflows` module and its caption, merge, conversion, and duplicate
description commands have been removed. This section records their historical
design; the standalone InternVL model remains available.

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

Run from the repository root. Whole-image experiment commands receive the YAML selected by
`--config_file`; `download_model MODEL_NAME` always reads
`cfg/experiment_config_internvl.yml` from the working directory and accepts no
configuration flag. It pulls the positional Ollama model with progress and does
not change `ollama.llm` or the YAML. It starts an owned server, rejects existing
endpoints, and cleans up on completion, failure, or Ctrl-C. `OLLAMA_MODELS`, when
set, is inherited. Use an approved allocation for cluster execution.

The required Ollama settings are `llm`, `binary_path`, `host`, `startup_timeout`,
and `request_timeout`. The executable path expands `~` and resolves relative to
the working directory; bare executable names resolve through PATH. The application
does not install binaries automatically.
See [Ollama setup](../README.md#ollama-setup).

Generate descriptions with `python -m internvl experiment describe --config_file
cfg/classification.yml`. This stage follows `embed_descriptions`: when disabled,
it returns without generating descriptions. When enabled, it reuses existing
`output_dir/descriptions.json` or generates and saves that file with a top-level
`descriptions` mapping from class labels to lists of descriptions.
`scripts/generate_descriptions.sbatch` uses this command and configuration by
default; its optional first argument selects another configuration.

The removed commands were `generate-descriptions`, `caption-images`,
`merge-captions`, and `convert-captions`. Existing description dictionaries and
caption CSV files are not migrated. Class-name descriptions and captions of
labeled training images have different provenance; preserve that distinction
when using historical artifacts.

## Whole-image LanceDB storage

The internvl experiment uses embedded LanceDB 0.40 with PyArrow declared directly.
`lancedb.uri` resolves relative to the working directory and expands `~`;
`lancedb.table` names the reference table. Storage is independent of `output_dir`.
Existing Qdrant collections and historical artifacts are untouched. Rebuild
references from inputs into a new LanceDB table. The native Qdrant package,
Pixi task, and server scripts remain for the separate CLIP/YOLO experiments.

Build once, then submit evaluation jobs against the completed table with separate
output directories. Each reader opens one table handle and runs exact cosine
searches without writes, index creation, optimization, or cleanup. Concurrent
writers to the same table are outside the supported workflow. Shared storage
must be accessible from every allocated node; local temporary-directory tests
do not establish cluster filesystem behavior.

The existing full-run and evaluation Slurm wrappers preserve resources and
forward `--set lancedb.uri=...` and `--set lancedb.table=...`. The sweep assigns a
separate table and output directory to each model run and requires no Qdrant
service. After indexing completes, evaluations can share the URI/table while
writing reports under distinct `output_dir` values.

New index summaries store `table`, resolved `uri`, source counts, and configuration.
Evaluation reports use `table` instead of `collection`. Source counts are attached
only when the local summary identifies the same URI and table. Existing tables,
including partial builds, are reused without compatibility or completeness checks.
Failures propagate; no rollback, retries, or partial evaluation reports are added.

Search uses explicit cosine distance and `bypass_vector_index()`, converting each
distance to similarity with `1 - distance`; voting is unchanged. See the
[LanceDB search documentation](https://docs.lancedb.com/search/vector-search).

## Historical text evaluation

The removed text classification workflow compared retrieved labels with FoodX251 validation labels.
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

## Historical text workflow limitations

These defects describe the deleted workflow source, available in Git history.

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


## LanceDB migration verification

Both locked Pixi environments install LanceDB 0.40.0 and PyArrow 26.0.0.
Python 3.13.13, PyTorch 2.7.1, torchvision 0.22.1, CUDA, Ollama, and native Qdrant
constraints remain unchanged. Each environment passed 100 synthetic tests with
one existing Ollama-guard skip and 94% package coverage; GPUs were hidden and
Hugging Face downloads disabled. Ruff, scoped ty, shell syntax and argument
forwarding checks, sweep isolation, and wheel/source builds passed. The root
package also built successfully.

Real temporary LanceDB tests cover persistence, payloads, float32/float64 inputs,
cosine ranking and scores, malformed vectors, empty retrieval, and overwrite
protection. Three spawned processes synchronize before overlapping searches of a
completed table; its contents and version remain unchanged. Evaluation passes
with database directories set to 0555 and files to 0444 under an unprivileged
user, while missing databases/tables fail without creating storage or reports.
The local sandbox stalled in LanceDB connection construction; the synthetic
suites completed outside that sandbox. No real models, datasets, services, GPUs,
or Slurm jobs were used. Shared-filesystem behavior and cluster concurrency
remain separate acceptance checks.

## Previous environment verification

### PyTorch 2.7 migration

Both PyTorch 2.7.1 profiles installed on Linux x86_64 with CPython 3.13.13.
Each passed 79 synthetic tests with one existing skip and 93% package coverage;
the CUDA profile ran with GPUs hidden. Ruff, scoped ty, and CPU console help
passed. Imports confirmed torchvision 0.22.1 in both profiles and CUDA runtime
12.6 in the GPU build. The installed CUDA binary includes `sm_70`.

No real models, datasets, GPU kernels, Ollama, remote Qdrant, or HPC jobs were
run for this migration. Use the V100 kernel check above on the cluster before
retrying the application.

### Earlier Python 3.14 environment

Before the PyTorch 2.7 migration, the CPU 2.14.1 and CUDA 12.8 2.11.0
profiles were installed separately on Linux with CPython 3.14.8.
Each passed 81 tests and 151 subtests with 93% package coverage, one existing
explicit skip, and no missing-dependency skips. Both real-model smoke runs passed
text and image encoding on CPU, and their temporary caches were removed. Synthetic
success, failure, SIGINT, and SIGTERM cleanup checks passed.

Ruff and the scoped ty check passed. Wheel and source distributions built;
non-editable imports and console help worked outside the checkout without
`PYTHONPATH`. Repeated locked syncs completed without dependency changes. The root
wheel retained the three legacy image commands and excluded `internvl`.
CUDA execution, real datasets, Ollama, remote Qdrant, and HPC remain unverified.

These historical real-model results do not validate the new PyTorch 2.7 profiles.

## Pixi environment maintenance

`src/internvl/pyproject.toml` defines two independent solve groups: `default`
(CPU) and `cu126` (PyTorch CUDA 12.6, Ollama CUDA 12.9). Both use standard CPython 3.13.13 and include the
application, dev, and test dependencies. `pixi.lock` records native packages and
PyPI wheels together. Install both environments before submitting Slurm jobs;
`--locked` checks manifest consistency, while `--as-is` runs an already prepared
environment without installation or lockfile updates. Reinstall after pulling
manifest or lockfile changes. Pixi's cache and both environment directories must
be accessible wherever installation takes place; the prepared environment must
be available at the same checkout path to the compute nodes.

Conda-forge supplies Python, Ollama, and native runtime libraries. Qdrant 1.19.0
comes explicitly from the Anaconda channel under strict channel priority. The
Conda-to-PyPI mapping marks native Ollama as a non-Python package so that the
Python Ollama SDK is installed independently. Python application dependencies
remain on PyPI, with package-specific official CPU/CUDA indexes for PyTorch.
See the [Pixi manifest reference](https://pixi.prefix.dev/latest/reference/pixi_manifest/)
and [PyTorch guide](https://pixi.prefix.dev/latest/python/pytorch/).

Ollama is pinned to 0.9.6. The GPU environment pins build `cuda_129h63bc95c_4`
and `cuda-version=12.9`. The platform requires CUDA 12.9 driver support; the
`cu126` environment name still refers to the unchanged PyTorch wheel index.
The CPU build is `cpu_hba3e463_4`. Drivers are supplied by the cluster. PyTorch
2.7.1 and torchvision 0.22.1 preserve the selected Volta/V100 stack. The minimum
glibc version is 2.28. Native package metadata alone does not prove GPU support.

Transformers 4.57.6, Sentence Transformers 5.7.0, and datasets 4.8.5 preserve the
previous compatible Python stack. Transformers 5.5.4 failed real image embedding
validation during earlier setup; Sentence Transformers 6 requires Transformers 5,
and datasets 5 requires a Hugging Face Hub version incompatible with Transformers 4.
These environment constraints do not introduce model or artifact revision tracking.

To update a dependency, edit its constraint if pinned, then run a selective update:

```sh
pixi update --manifest-path src/internvl/pyproject.toml PACKAGE_NAME
pixi install --manifest-path src/internvl/pyproject.toml --locked -e default
pixi install --manifest-path src/internvl/pyproject.toml --locked -e cu126
```

Review the manifest and lockfile diff, check both environments' package versions,
and rerun tests, lint, type checks, and builds before committing. Preserve the
Ollama CUDA build and PyTorch index choices unless intentionally changing GPU
support. Dependency installation does not download models or start services.
Model caches and Qdrant data remain outside `.pixi/`; rebuilding an environment
does not replace them. The retired uv environment can be removed locally once
Pixi works; no script removes it or user data automatically.

The Qdrant task starts a foreground process with fresh storage under
`assets/qdrant/1.19.0/` by default. It binds to loopback unless explicitly
configured otherwise. Before opening older data with a newer server, back it up
and follow Qdrant's consecutive minor-version upgrade procedure. Do not point
this environment at an old database as an installation check; downgrading a
server is not a data rollback strategy. See
[Qdrant upgrades](https://qdrant.tech/documentation/operations/upgrades/).

Live acceptance is separate from synthetic checks: use temporary Qdrant storage
to verify insert/query/restart behavior, then an approved V100 allocation for an
Ollama description and a PyTorch CUDA kernel. Never use production collections
for these checks. Routine tests do not start Ollama or download model weights.

### Packaging and check details

Accelerator selection lives in Pixi features; the former Python `cpu` and `cu126`
extras are removed. With those extras retained, Pixi 0.81.0's locked-install check
expects the default PyPI index for their torch/torchvision requirements and
rejects the feature-specific PyTorch indexes. The package name, console entry
point, and ordinary runtime dependency metadata remain unchanged.

Compared with the previous uv lock, Python application versions are unchanged
for the selected Linux profiles. The new build task adds `build`, `wheel`, and
`pyproject-hooks`; setuptools resolves to 84.2.1 instead of 81.0.0. The Pixi lock
omits uv resolutions for unsupported platforms and unselected accelerator variants.

The enforced ty scope remains `cli.py` and `schemas.py`. A whole-package check
still has two deferred diagnostics: optional Ollama response content passed to
Pydantic and a string passed to datasets' `NamedSplit` parameter. Unresolved
imports are not suppressed.

The explicit real-model smoke test uses fresh subprocess caches and requires
finite, nonzero float32 outputs of shape `(1, 1024)` for both modalities. It waits
for the worker before removing temporary caches on success, failure, Ctrl-C, or
SIGTERM; SIGKILL and machine failure cannot guarantee cleanup. It is excluded
from routine pytest discovery and does not establish GPU or service compatibility.

### V100 kernel check

Run inside an approved V100 allocation after installing `cu126`:

```sh
pixi run --manifest-path src/internvl/pyproject.toml --as-is -e cu126 python - <<'PYTHON'
import torch
import torchvision

print("PyTorch:", torch.__version__, "torchvision:", torchvision.__version__)
print("CUDA runtime:", torch.version.cuda)
assert torch.__version__ == "2.7.1+cu126"
assert torchvision.__version__ == "0.22.1+cu126"
assert torch.cuda.is_available()
print("GPU:", torch.cuda.get_device_name(0))
print("Capability:", torch.cuda.get_device_capability(0))
print("Compiled architectures:", torch.cuda.get_arch_list())
assert torch.cuda.get_device_capability(0) == (7, 0)
assert "sm_70" in torch.cuda.get_arch_list()
x = torch.ones((32, 32), device="cuda")
y = x @ x
assert y.sum().item() == 32768
torch.cuda.synchronize()
print("V100 kernel check passed")
PYTHON
```

### Pixi migration verification

Both environments installed with Pixi 0.81.0. Repeated `--locked` installation
left `pixi.lock` unchanged. Each environment passed 81 synthetic tests with one
existing skip and 93% package coverage, Ruff, scoped ty, CLI help, and package
builds. Tests hid GPUs and disabled Hugging Face downloads. Native Qdrant reports
1.19.0 and its linked libraries resolve. The Qdrant wrapper passed synthetic
checks for defaults, overrides, and argument forwarding without starting a server.
PyTorch reports the selected CPU/CUDA variants; the CUDA binary includes `sm_70`.
No models, datasets, GPU kernels, live Ollama/Qdrant services, or HPC jobs were run.

The original conda-forge Ollama `0.9.6-cuda_126h278ac6b_` package contained only
`bin/ollama`, with no backend libraries. It was replaced by build
`cuda_129h63bc95c_4` after the target cluster driver was identified as
580.178.04. NVIDIA's driver compatibility documentation supports CUDA 12.9
applications on that driver through backward compatibility. The installed package
contains `libggml-cuda.so`, `libggml-base.so`, CUDA runtime, and cuBLAS libraries.
Static inspection with CUDA 12.9 `cuobjdump --list-elf` confirms `sm_70` machine
code in `libggml-cuda.so`. Its dependencies resolve with Ollama's own library
directory on the runner's search path, as configured by upstream Ollama. These
checks do not establish successful Ollama inference on V100.

PyTorch keeps its CUDA 12.6 wheels, including `sm_70`. Ollama is a separate
process with its own packaged CUDA libraries; do not set a global
`LD_LIBRARY_PATH` that mixes its libraries with PyTorch's wheel dependencies.
No cluster CUDA module or system toolkit installation is needed for these
precompiled runtimes. GPU execution remains an explicit cluster acceptance step.

Sources: [conda-forge package metadata](https://api.anaconda.org/package/conda-forge/ollama),
[Ollama CUDA targets](https://github.com/ollama/ollama/blob/v0.9.6/CMakePresets.json),
and [NVIDIA compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html).

## W&B progress verification (October 24, 2026)

The progress-only refactor passed the CPU suite with 158 tests passed, one
existing skip, and 213 subtests passed (94% package coverage). Eight additional
insertion/description failure and interruption cases passed in a targeted run.
Native disabled and offline W&B smoke tests passed through the CLI-owned context.
Targeted Ruff checks, scoped ty checks, and source/wheel builds also passed.
Build outputs are in `/tmp/wandb-progress-dist`.

Checks used `/tmp/pixi run --as-is -e default` with the existing manifest at
`/home/theta/Projects/SmartCompostSensing/FoodWasteDetect-wandb_observability/src/internvl/pyproject.toml`.
`PYTHONPATH` selected this checkout's `src`; GPUs were hidden and
`HF_HUB_OFFLINE=1` prevented model downloads. The sandboxed run stalled in
`lancedb.connect` during fixture setup. The approved run outside the sandbox
completed, including real temporary LanceDB storage and local offline W&B sockets.
No live uploads, Ollama servers, real models, GPUs, or HPC resources were used.
These checks establish synthetic progress/reporting contracts, not model accuracy
or online dashboard latency.


## Jina Embeddings v4 alongside CLIP v2

The `internvl` whole-image runner selects either model using
`embedding.model_id`. CLIP v2 remains the default. V4 uses 2048-dimensional
single vectors, the retrieval adapter, and passage prompts for text references.
Training images and validation images use the same retrieval adapter without
text prompts. LanceDB exact cosine retrieval and voting are unchanged. This
migration does not affect the separate YOLO/CLIP pipeline.

The implementation follows the [Jina model card](https://huggingface.co/jinaai/jina-embeddings-v4)
and [Sentence Transformers module](https://huggingface.co/jinaai/jina-embeddings-v4/blob/main/custom_st.py).
V4 uses checkpoint dtype and SDPA. The upstream module explicitly uses BF16
autocast, so an FP16 loading override alone would not establish V100 compatibility.
V100 real-model execution remains a manual acceptance gate; H100 is the agreed
alternative. No local precision workaround or FlashAttention dependency is added.

### Rebuild references and compare

Prepare the locked `cu126` environment before submitting a job. Run each command
from this worktree's root in an approved GPU allocation. Use a fresh table and
output directory: the build command reuses an existing table without checking
its model or dimensions. Never search old CLIP vectors using v4 query vectors.

For a controlled comparison, copy the baseline's `descriptions.json` into
`assets/classification/descriptions_jina_v4/` before indexing when description
references are enabled. Keep the same reference switches, image cap, descriptions,
and `k`. If no descriptions exist, run the ordinary `describe` stage first and
reuse those same descriptions for both models. All paths below resolve from the
current worktree, which does not inherit ignored datasets or artifacts from
another worktree; override `dataset.root` and `lancedb.uri` as needed with the
intended accessible paths.

```bash
pixi run --manifest-path src/internvl/pyproject.toml --as-is -e cu126 internvl experiment index \
  --config_file cfg/classification.yml \
  --set embedding.model_id=jinaai/jina-embeddings-v4 \
  --set embedding.device=cuda --set embedding.batch_size=1 \
  --set lancedb.table=foodx251_descriptions_jina_v4 \
  --set output_dir=assets/classification/descriptions_jina_v4

pixi run --manifest-path src/internvl/pyproject.toml --as-is -e cu126 internvl experiment evaluate \
  --config_file cfg/classification.yml \
  --set embedding.model_id=jinaai/jina-embeddings-v4 \
  --set embedding.device=cuda --set embedding.batch_size=1 \
  --set lancedb.table=foodx251_descriptions_jina_v4 \
  --set output_dir=assets/classification/descriptions_jina_v4
```

Batch size 1 is a conservative starting point, not a verified memory requirement.
Increase it only after measuring memory use on the selected GPU. Compare the full
labeled validation split; the migration does not establish improved accuracy.

### Manual real-model acceptance

This check may download v4 weights and uses a GPU. Run it only with explicit
authorization and an allocated GPU, after preparing `cu126`. It generates no
descriptions and opens no dataset or database. It checks the same wrapper used
by indexing and evaluation with synthetic text and an image. A successful run
establishes basic encoding on that GPU, not food-recognition accuracy.

```bash
pixi run --manifest-path src/internvl/pyproject.toml --as-is -e cu126 python - <<'PYTHON'
import numpy as np
import torch
from PIL import Image
from internvl.embeddings import EmbeddingModel

print("GPU:", torch.cuda.get_device_name(0))
model = EmbeddingModel("jinaai/jina-embeddings-v4", batch_size=1, device="cuda")
with Image.new("RGB", (224, 224), (160, 80, 40)) as image:
    for modality, vectors in (
        ("text", model.encode_text(["a bowl of rice", "a plate of noodles"])),
        ("image", model.encode_images([image])),
    ):
        expected_rows = 2 if modality == "text" else 1
        assert vectors.shape == (expected_rows, 2048)
        assert vectors.dtype == np.float32
        assert np.isfinite(vectors).all()
        assert (np.linalg.norm(vectors, axis=1) > 0).all()
        print(modality, vectors.shape, vectors.dtype)
torch.cuda.synchronize()
PYTHON
```

Automated checks use mocked pretrained loading, real Sentence Transformers with
synthetic vectors, and temporary LanceDB tables. They cover 1024/2048 dimensions,
v4 passage prompting and task routing, batching and order, invalid outputs,
image ownership, and 2048-dimensional cosine retrieval. They do not download
weights or establish real-model, GPU, shared-filesystem, or HPC compatibility.


### Verification on October 25, 2026

Pixi 0.81.0 added PEFT 0.21.2, Accelerate 1.15.0, and psutil 7.2.2 to both
locked profiles without changing existing package versions. The new worktree's
`default` CPU environment installed with `--locked`; the CUDA profile was resolved
but not installed or executed. Imports confirmed PEFT 0.21.2, PyTorch 2.7.1+cpu,
Transformers 4.57.6, and Sentence Transformers 5.7.0.

With `PYTHONPATH=src`, `CUDA_VISIBLE_DEVICES=""`, and `HF_HUB_OFFLINE=1`, the
complete package suite passed 178 tests with one existing skip. Package coverage
was 95%; the embedding wrapper had 100% statement coverage. The initial focused
run stalled in LanceDB under the restricted sandbox and was terminated; the full
suite passed outside that sandbox using temporary databases. Ruff, the manifest's
scoped ty check, CLI experiment help, wheel/sdist builds, and `git diff --check`
also passed. These checks used no model weights, real datasets, GPU kernels,
Ollama services, or HPC allocations. Real v4 and V100/H100 acceptance remains
pending the manual check above.

## Four-model embedding extension (October 25, 2026)

The whole-image runner now selects four Sentence Transformers models through the
existing `embedding.model_id` setting. Jina CLIP v2 remains the default (1024
values); Jina Embeddings v4 retains its 2048-value retrieval vectors and passage
prompt for text. EmbeddingGemma2 produces 768 values, uses the `Document` text
prompt, and receives each image as a separate image dictionary. OpenAI CLIP
produces 512 projected features for both modalities; text has no prompt template
and truncates to 77 tokens. Its image preprocessing comes from the checkpoint.

EmbeddingGemma2 loads text and vision with `audio_config=None`. It uses BF16 only
when the selected CUDA device supports BF16 natively, otherwise FP32. OpenAI CLIP
loads FP32. Only Jina uses remote code and the missing-modality metadata fallback.
The wrapper adds no normalization. See the upstream
[EmbeddingGemma2 architecture documentation](https://huggingface.co/docs/transformers/en/model_doc/embedding_gemma2)
for modality configuration and per-sample image dictionaries.

Choose a new table and output directory for every model. Rebuild all enabled
references and use the same model for evaluation. Existing tables are reused
without checking model or vector compatibility. Keep the reference inputs,
saved descriptions, image cap, and retrieval settings identical for comparisons.
For example, apply the same overrides to both stages:

```sh
for stage in index evaluate; do
  python -m internvl experiment "$stage" --config_file cfg/classification.yml \
    --set embedding.model_id=google/embeddinggemma-2 \
    --set lancedb.table=foodx_embeddinggemma2 \
    --set output_dir=outputs/embeddinggemma2
done

for stage in index evaluate; do
  python -m internvl experiment "$stage" --config_file cfg/classification.yml \
    --set embedding.model_id=openai/clip-vit-base-patch32 \
    --set lancedb.table=foodx_openai_clip \
    --set output_dir=outputs/openai_clip
done
```

These are normal experiment commands and may download weights. Default YAML,
Slurm scripts, experiment orchestration, and the separate YOLO pipeline are
unchanged. Neither README was edited.

### Shared environment (superseded by isolated Jina profiles below)

The two Pixi solve groups initially pinned Sentence Transformers 6.2.0 and Transformers
5.20.2; project requirements are `sentence-transformers>=6.1,<7` and
`transformers>=5.20.2,<6`. The solve updated huggingface-hub to 1.33.0 and tokenizers
to 0.23.3, with required CLI dependencies. Python 3.13.13, PyTorch 2.7.1 CPU/cu126,
torchvision 0.22.1, and native dependencies remain unchanged. No audio/video
extras were added. The CPU environment was installed with `--locked`; CUDA was
resolved without installing or running it.

The earlier two-Jina verification above records the previous library versions.
Upstream Jina CLIP v2 compatibility repairs for Transformers 5 are explicitly
deferred. Mocked loading and synthetic Jina tests remain; they do not establish
that either real Jina checkpoint runs with the upgraded libraries.

### Extension verification

Using the installed CPU environment with `PYTHONPATH=src`,
`CUDA_VISIBLE_DEVICES=""`, and `HF_HUB_OFFLINE=1`, the full suite passed **198 tests,
one existing skip, and 260 subtests**. Package statement coverage was 95%; the
embedding wrapper had 100% coverage. The suite ran outside the restricted sandbox
for temporary LanceDB databases and local offline W&B sockets.

Coverage includes all four dimensions, prompts and task routing, Gemma image
dictionaries, native selected-device BF16 checks and FP32 fallback, empty and
invalid outputs, image ownership, real Sentence Transformers batching and input
order, and synthetic indexing/evaluation through real temporary LanceDB tables.
A small randomly initialized local CLIP model verifies 512-dimensional projected
text/image outputs and 77-token truncation without pretrained downloads.
EmbeddingGemma2's configuration and model architecture import successfully
without loading weights.

Package Ruff, scoped ty, CLI experiment help, and wheel/sdist builds passed.
Builds are in `/tmp/embedding-extension-dist`. No pretrained weights, datasets,
GPU kernels, live Ollama services, or HPC resources were used. These checks do
not establish checkpoint, GPU, real-dataset, or shared-storage compatibility.


## Isolated Jina environments

The supplied diagnosis records two construction failures from complete W&B console
logs and progress summaries: [royal-bird-17](https://forge.coreweave.com/wandb/soundbendor/food-waste-detection/runs/aispdzi7/logs)
failed importing `clip_loss`, and [dutiful-elevator-14](https://forge.coreweave.com/wandb/soundbendor/food-waste-detection/runs/a3shaxva/logs)
failed at `ROPE_INIT_FUNCTIONS["default"]`. Both completed 0 / 6,275 references.
Their installed dependency versions were not recorded. The implementation change
isolates the Jina backends on older libraries instead of patching remote model code.

| Environments | Models | Sentence Transformers | Transformers | PEFT |
| --- | --- | --- | --- | --- |
| `default`, `cu126` | EmbeddingGemma2, OpenAI CLIP | 6.2.0 | 5.20.2 | 0.21.2 |
| `jina-cpu`, `jina-cu126` | Both Jina models | 5.7.0 | 4.57.6 | 0.17.1 |

The four solve groups preserve Python 3.13.13, PyTorch 2.7.1, torchvision 0.22.1,
and native package resolutions. Existing modern profiles retain their package
resolutions. The loader rejects mismatched library families before pretrained
construction and names the model, installed versions, and suitable environments.
The YAML default remains Jina CLIP v2 and requires an explicit Jina environment.

Prepare environments before submission, from the repository root:

```bash
pixi install --manifest-path src/internvl/pyproject.toml --locked -e jina-cpu
pixi install --manifest-path src/internvl/pyproject.toml --locked -e jina-cu126
pixi run --manifest-path src/internvl/pyproject.toml --locked -e jina-cpu test
```

For an authorized allocation and prepared model caches, use the existing config
and override forwarding:

```bash
INTERNVL_PIXI_ENV=jina-cu126 sbatch sbatch/run_embedding.sbatch cfg/classification.yml \
  --set lancedb.table=jina_clip_retry --set output_dir=outputs/jina_clip_retry
INTERNVL_PIXI_ENV=jina-cu126 sbatch sbatch/test_embedding.sbatch cfg/classification.yml \
  --set lancedb.table=jina_clip_retry --set output_dir=outputs/jina_clip_validation
```

The wrappers retain `cu126` as their default. The four-model sweep exports the
appropriate environment per model, keeps V4 batch size 1, and creates separate
tables and outputs. Scheduling remains manual: choose suitable resources with
Slurm options. V4 requires native BF16 on the selected CUDA device (including
nondefault GPUs); automatic device selection is resolved first. V100 is rejected.
CPU remains available; other accelerator types are rejected. V4 keeps checkpoint
dtype and SDPA, and Gemma keeps its FP32 fallback.

Real-checkpoint acceptance remains pending. Separately authorize real text/image
smoke checks, using native-BF16 hardware for V4, then repeat both failed runs with
the same reference inputs and retrieval settings in fresh tables and outputs.
Acceptance requires completed indexing and validation reports. Routine checks
remain offline, synthetic, and CPU-only; they do not establish checkpoint, GPU,
shared-filesystem, or cluster compatibility.


### Verification of isolated profiles

Both CPU environments were installed from the regenerated lockfile. All 92 modern
and 86 Jina installed PyPI dependencies matched their locked versions. Comparing
lockfile entries confirmed that both modern profiles retained every prior package
resolution and both Jina profiles retained the matching modern native packages.
The existing local `cu126` installation is stale: Sentence Transformers 5.7.0,
Transformers 4.57.6, and no PEFT metadata. It was inspected without GPU execution;
reinstall it with `--locked` before use. `jina-cu126` was resolved but not installed.

With `CUDA_VISIBLE_DEVICES=""`, `HF_HUB_OFFLINE=1`, and `PYTHONPATH=src`, the full
suite passed in both environments: **205 passed, 2 skipped, 278 subtests** in
`default`; **204 passed, 3 skipped, 278 subtests** in `jina-cpu`. The skips comprise
one existing skip plus checks specific to the other dependency family. The initial
sandboxed runs stalled and were interrupted; successful runs used unrestricted
execution for temporary LanceDB databases and local offline W&B sockets.

Checks cover real Sentence Transformers synthetic batching, order, dimensions,
prompts, finite nonzero vectors, image ownership, and shared experiment behavior.
Jina imports include `clip_loss`, the default RoPE initializer, and PEFT. Modern
checks retain the miniature projected CLIP/truncation test and EmbeddingGemma2
architecture import. Mocked checks reject dependency mismatches and unsupported
V4 hardware before pretrained construction, including automatic and nondefault
CUDA selection. Fake Slurm/Pixi tests cover explicit environments, inherited
values, defaults, argument forwarding, V4 batch size, and separate tables/outputs;
the final focused rerun passed all three script tests. Ruff, scoped ty, and
`git diff --check` passed. No model weights, real datasets, GPUs, live services,
or cluster jobs were used.

## Qwen whole-image embedding integration (October 26, 2026)

`Qwen/Qwen3-VL-Embedding-8B` is the fifth supported whole-image model. The
[model card](https://huggingface.co/Qwen/Qwen3-VL-Embedding-8B) specifies native
4096-dimensional embeddings. The loader uses Sentence Transformers with
`torch_dtype="auto"` and `attn_implementation="sdpa"`, leaving the checkpoint's
instruction, preprocessing, last-token pooling, and normalization intact. Text
and RGB PIL images use the existing encoding methods. Shared optional truncation
also applies; changing dimensions requires rebuilding references in a fresh table.

The sweep selects `cu126` and explicitly sets Qwen batch size 8. Select an
A100/H100-class partition manually using `SBATCH_PARTITION` before running the
sweep; the Slurm wrapper still defaults to `dgx2`. The target precision is native
BF16 through checkpoint dtype. No precision adaptation, hardware guard, automatic
batch reduction, quantization, prompt tuning, or dependency changes were added.
Keep reference sources, saved descriptions, image caps, and retrieval settings
identical across models, with separate tables and output directories.

The supplied investigation reports CPU text/image forwards with a tiny randomly
initialized Qwen3-VL model. This integration did not repeat that probe or load the
real checkpoint. Real-checkpoint acceptance requires separate authorization on
the selected GPU: encode text and images at batch size 8, measure peak memory and
throughput with representative and large images, then build fresh references and
evaluate the full labeled validation split. Memory failures propagate normally.

### CPU verification

Checks used the existing modern `default` environment with
`CUDA_VISIBLE_DEVICES=""`, `HF_HUB_OFFLINE=1`, and `PYTHONPATH=src`. Pretrained
loading was mocked. New Qwen tests cover loader options, text/image routing,
input order, RGB ownership, empty inputs, invalid vectors, float32 outputs,
truncation, and real Sentence Transformers batching with eight inputs plus a
final two-input batch. A synthetic normalization module verifies that normalized
outputs are preserved. Temporary LanceDB tests index and evaluate native 4096
and truncated 128-dimensional vectors. Fake `sbatch` checks cover five jobs,
selected environments, batch sizes, and unique tables/output directories.

The pre-change targeted guard baseline reproduced **3 failures and 1 skip**.
Running the original embedding source and tests from `HEAD` in a temporary
directory gave **10 failures, 35 passed, 1 skipped**: the same absent guard also
breaks seven loader tests through their shared setup. These guard repairs remain
out of scope. Earlier notes claiming the guards exist describe prior verification,
not the current source. The sandboxed combined run stalled during LanceDB checks;
verification was resumed outside the sandbox, still offline and CPU-only.

The final full modern CPU suite reported **208 passed, 2 skipped, 10 failed,
339 subtests passed**. All ten failures match the pre-change embedding baseline;
no new failures remain. One warning reports Sentence Transformers' deprecated
`sentence_transformers.models` import in the synthetic normalization test; that
import remains compatible with both installed library families. Production-loader
Ruff and `git diff --check` passed. No model weights, datasets, GPUs, real services,
or cluster jobs were used. README files and dependency manifests were unchanged.

## Vertex Gemini Embedding 2 comparison (October 26, 2026)

Implemented in branch `feat/ge-v2-vertex`, isolated at
`/tmp/FoodWasteDetect-ge-v2` from `28746cb`. The original checkout remained clean
at that commit. The dedicated `cfg/ge_v2.yml` preserves the classification
reference defaults and shared description vocabulary while selecting separate
storage destinations. Both existing classification YAML files, README files,
Slurm scripts, and sweeps remain unchanged.

The configuration alias `google/ge-v2` sends requests to `gemini-embedding-2`
through Vertex AI. The [Google multimodal embedding contract](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/embeddings/get-multimodal-embeddings)
provides the service model name, dimensionality setting, and classification
instruction format. The wrapper uses inline RGB PNG images and individual text
records; it leaves preprocessing beyond RGB conversion and service normalization
to the API. `truncate_dim` selects 128–3072 output dimensions, with 3072 by default.
Optional classification instructions apply to both reference and evaluation data.
Use fresh tables and output directories when changing this setting or dimensions.

`google-genai==2.33.0` was added with `pixi add --pypi` from this worktree.
The lock change adds the SDK and its dependencies to all four environments;
existing dependency versions and modern/Jina feature pins are unchanged. The
[SDK retry implementation](https://github.com/googleapis/python-genai/blob/v2.33.0/google/genai/_api_client.py)
owns retry selection and exponential backoff. The application configures five
total attempts, a 60-second timeout per attempt, and four concurrent requests by
default. It cancels queued requests on failure and waits for running calls before
closing the shared client. It does not return partial batches.

After separate cloud authorization, run from the repository root with ADC
configured outside YAML:

```bash
python -m internvl experiment run --config_file cfg/ge_v2.yml \
  --set embedding.vertex.project=YOUR_PROJECT_ID
```

The default references are saved descriptions; evaluation uses labeled validation
images. Add `--set embed_images=true` for training-image references. No live API
acceptance, cloud quota, latency, cost, or classification-quality claim is made.

### CPU verification

All checks ran in the isolated worktree. Both locked CPU suites used
`CUDA_VISIBLE_DEVICES=""` and `HF_HUB_OFFLINE=1` through the Pixi `test` task.
No model weights or datasets were downloaded, and no live cloud requests, Ollama
servers, GPU allocations, or cluster jobs were used. SDK tests supplied synthetic
credentials and an in-memory HTTP transport; retry delays were zeroed only in the
test client while production retry settings were asserted separately.

The 59 new tests pass. They cover lazy alias routing, unchanged local forwarding,
text/image payloads and prompts, original image dimensions and ownership, empty
inputs, dimension and settings validation, invalid response cardinality/vectors,
ordered concurrent results, worker limits, every retryable HTTP status, connection
and timeout recovery/exhaustion, nonretryable errors, interruption, queued-work
cancellation, and client cleanup. A miniature local FoodX251 fixture and real
LanceDB exercise all three reference types, partial batches, cached builds with no
client construction, full evaluation, and a later failed batch with no new report.
CLI tests load the standalone YAML and exercise strict nested overrides.

Final commands and results:

- `pixi run --manifest-path src/internvl/pyproject.toml --locked -e default test --tb=short -q`:
  **267 passed, 2 skipped, 10 failed; 339 subtests passed**.
- `pixi run --manifest-path src/internvl/pyproject.toml --locked -e jina-cpu test --tb=short -q`:
  **267 passed, 3 skipped, 9 failed; 339 subtests passed**.
- Focused Vertex suite: **59 passed**; wrapper statement coverage is 100% in both
  full suites. This does not establish real-service acceptance.
- Ruff on the changed Python files, scoped ty including the new wrapper, and
  `git diff --check` passed.

All full-suite failures are in the unchanged `tests/test_embeddings.py` against
unchanged `embeddings.py`: the already documented absent dependency-family and
V4 hardware guards, including loader setup that patches the absent guard.
The Jina stack skips the EmbeddingGemma2 architecture test, producing one fewer
failure. Both suites also retain the existing synthetic Sentence Transformers
import deprecation warning. Guard restoration remains outside this feature.


## Vertex embedding resumption (October 26, 2026)

Vertex reference builds now resume by `(source_kind, source_identity)` and save
each successful response individually. Existing UUIDs and rows remain intact.
The training-image cap is applied before saved identities are excluded; exclusion
happens before decoding. Metadata scans use an explicit unlimited query and omit
vectors. Completed builds regenerate `index.json` from stored source counts.

The standalone Vertex YAML enables `cache_validation_embeddings` and selects
`foodx251_ge_v2_validation` as its separate validation table. Omitted caching
remains disabled. Cached evaluation first completes missing embeddings, then reads
bounded vector batches in annotation order for reference-table retrieval. Complete
caches work without image files, credentials, client construction, or API requests.
Changing retrieval settings can reuse vectors. Local embedding behavior is unchanged.

Callbacks run serially on the coordinating thread. After API failure or interruption,
queued work is cancelled and successful in-flight responses are saved before the
error propagates. Storage failures are not retried, and failed saves do not mark
identities complete. Reports are still published only after full evaluation.
Progress includes reused requested records and advances after completed batches;
a failed batch can commit vectors without a matching final progress update.

Use one writer per table and fresh tables for changes to inputs, descriptions,
model, dimension, or prompt mode. There is no compatibility validation or migration.
The guarantee covers committed vectors; a crash before saving or a timeout after
remote processing can still cause a repeated request.

Verification used synthetic PNGs, temporary real LanceDB tables, and the real
Google SDK with mocked HTTP transports. Tests cover all reference source types,
more than ten saved rows, capped training selection, out-of-order callbacks,
in-flight success draining on failure and interruption, storage failures without
write retries, exact missing-request counts on restart, complete caches without
image files or credentials, validation ordering and table separation, report
failure reuse, progress accounting, strict CLI overrides, configuration defaults,
and unchanged local existing-table behavior.

The locked environments' `pytest` launchers contain an obsolete `/tmp` shebang,
so verification used `python -m pytest` without changing dependencies or launchers:

- `pixi run --manifest-path src/internvl/pyproject.toml --locked -e default python -m pytest -c src/internvl/pyproject.toml --tb=short -q`:
  **283 passed, 2 skipped, 10 failed; 339 subtests passed**.
- The same command with `-e jina-cpu`:
  **283 passed, 3 skipped, 9 failed; 339 subtests passed**.
- All 75 Vertex tests pass in both suites; Vertex wrapper statement coverage is
  **100%**, and overall package coverage is **94%**.
- Ruff on `src/internvl` and `tests/test_vertex_embeddings.py`, scoped ty, and
  `git diff --check` passed.

All failures remain in unchanged local `test_embeddings.py` guard tests and loader
setup: missing dependency-family and V4 hardware guards. The Jina stack skips the
EmbeddingGemma2 architecture test. These failures remain outside this change.
An initial sandboxed run stalled in LanceDB connection setup; the completed tests
ran outside that sandbox. No real cloud requests, spending, model downloads, GPU
work, or live services were used. These checks do not establish cloud acceptance
or concurrent-writer safety.

### Cache orchestration simplification (October 26, 2026)

Reference construction now uses one workflow for both backend families. Existing
local tables return before input preparation or embedder construction. Vertex
builds scan stored identities once, using an immutable snapshot for filtering and
an initial counter that includes every stored row. Capped training metadata supplies
requested and reused counts and validates class coverage before a missing table is
created. Only missing decoded images become records. Successful batches extend the
initial source counts; summary publication does not rescan the table under the
single-writer contract.

Typed text and image saving helpers share the callback that pairs each response
with its original record. Committed rows determine restart behavior; insertion no
longer maintains a mutable saved-identity set. Training references and validation
cache population share an image insertion loop with an explicit split. Evaluation
completes `_prepare_validation_cache()` before classification, then passes the
prepared table to an iterator that reads metadata and fetches vectors in annotation
order. Uncached evaluation still encodes images directly.

Regression tests compare exact submitted text and image inputs against unsaved
identities. Storage tests use 26 distinct vectors, shuffled insertion, reversed
requests, repeated requests, quoted identities, and matching identity strings
across source kinds. A two-class partial-cache test verifies predicted labels and
annotation ordering, then repeats evaluation with image files removed. Additional
tests verify one identity scan, counts including duplicate and unrequested rows,
input validation before table creation, and cache completion before classification.
The local-backend test also checks batch insertion and ignored cache settings.

Verification used offline CPU environments, synthetic data, temporary LanceDB
tables, and mocked HTTP transports:

- The focused Vertex, retrieval, and classification suite passed **132 tests and
  111 subtests** before the final three contract tests were added.
- `CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 pixi run --manifest-path src/internvl/pyproject.toml --locked -e default python -m pytest -c src/internvl/pyproject.toml --tb=short -q`:
  **288 passed, 2 skipped, 10 failed; 339 subtests passed**.
- The same full command with `-e jina-cpu`:
  **288 passed, 3 skipped, 9 failed; 339 subtests passed**.
- Full suites include all five added tests. Package statement coverage is **95%**;
  the unchanged Vertex callback implementation retains **100%** coverage.
- Ruff and the expanded scoped ty check pass. The type-check scope now includes
  experiment, retrieval, and dataset modules; `NamedSplit`, concrete local LanceDB
  connection casts, and narrow dimension/array annotations resolve their diagnostics.

Full-suite failures remain confined to the existing local embedding dependency
and hardware guard tests and their loader setup. No model weights, live cloud
requests, cloud spending, or GPU resources were used. Dependencies, lockfiles,
READMEs, scripts, sweeps, configuration, and storage schemas were unchanged by
this refactor.

### Vertex utility extraction (October 26, 2026)

`internvl.vertex_utils` now owns cache configuration, response persistence,
reference resumption, validation-cache preparation, and cached batch reading.
Module-level functions accept explicit inputs and callbacks without receiving an
experiment object. `VertexExperiment(Experiment)` in this module now owns Vertex
validation, construction, persistence, and the build/evaluation overrides. The
base module imports neither Vertex module and retains shared text preparation,
capped training validation, insertion loops, progress, classification, reports,
and one lazy encoder cache. Direct callers explicitly select the class; the CLI
selects once and validates through that class before starting W&B. Cached and uncached evaluation yield metadata batches
and vectors; prediction-row construction remains in the experiment.

Reference preparation scans identities once, retains an immutable membership
snapshot, and counts all stored rows, including duplicates and unrequested records.
Training caps precede exclusion and decoding. Validation-cache preparation receives
lazy dimension lookup and shared insertion/progress callbacks. A complete cache
needs neither images nor an embedder. Transport retries, cancellation, and draining
remain unchanged in `VertexEmbeddingModel`.

Cache toggles follow Python truthiness. Omitted and falsey values disable caching;
truthy values require a nonempty validation-table name distinct from the reference
table. Tests exercise nonboolean values through configuration and full evaluation.
READMEs, public stage methods, YAML keys, schemas, and report formats are unchanged.

Verification used Pixi 0.83.0 with the unchanged lockfile and both CPU dependency
environments. GPU visibility and Hugging Face downloads were disabled. Vertex
requests used mocked transports and storage used temporary real LanceDB tables.
The sandboxed focused run stalled at LanceDB and was interrupted; completed suites
ran outside the sandbox. No live Vertex requests, model weights, or services were
used.

Commands used `pixi run --manifest-path src/internvl/pyproject.toml --locked -e ENV
python -m pytest -c src/internvl/pyproject.toml --tb=short -q`, with
`CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1`. The focused run additionally selected
`tests/test_vertex_embeddings.py tests/test_classification.py
tests/test_retrieval.py tests/test_tracking.py`.

- Focused modern suite: **201 passed; 111 subtests passed**.
- Full modern suite: **293 passed, 2 skipped, 10 failed; 339 subtests passed**.
- Full Jina CPU suite: **293 passed, 3 skipped, 9 failed; 339 subtests passed**.
- `vertex_utils.py` and the unchanged Vertex transport each have **100% statement
  coverage**; package coverage is **95%**.
- Ruff on the package and Vertex tests, scoped ty in both environments, and
  `git diff --check` passed.

All full-suite failures remain in unchanged `tests/test_embeddings.py`: missing
`_check_dependency_family`/`version` and loader setup for the previously removed
dependency and V4 hardware guards. No refactor regressions were observed.

### Explicit VertexExperiment subclass (October 26, 2026)

`VertexExperiment` now overrides build and evaluation orchestration in
`vertex_utils.py`. It also supplies cache validation, encoder construction, and
per-response batch persistence. `Experiment` exposes shared preparation,
insertion, summary publication, and evaluation execution methods. Its lazy cache
uses a private structural encoder protocol. The inherited `run()` dispatches to
the subclass overrides. Local encoder options are selected explicitly and local
batch persistence remains unchanged.

CLI selection happens once from `embedding.model_id`; class validation precedes
W&B initialization and construction remains inside the native context. Direct
Python callers must select `VertexExperiment(config)` for Vertex. Importing the
base module loads neither Vertex module. Tests now exercise this import boundary,
both CLI classes across all four stages, validation before tracking, inherited
stage dispatch, and the existing persistence, restart, cache, and progress cases.

The focused modern CPU suite passed **212 tests and 111 subtests**; the separate
CLI suite passed **25 tests and 71 subtests**. Checks use synthetic inputs, mocked
Vertex transports, temporary LanceDB tables, disabled GPUs, and offline Hugging
Face loading. Sandboxed LanceDB runs stalled and were interrupted; completed
LanceDB suites ran outside the sandbox. No live requests or model downloads were
used. READMEs, dependencies, lockfile, transport behavior, YAML keys, table schemas,
and report formats were not changed by this subclass refactor.

Final full-suite verification uses the same locked Pixi commands documented above:

- Modern CPU: **304 passed, 2 skipped, 10 failed; 339 subtests passed**.
- Jina CPU: **304 passed, 3 skipped, 9 failed; 339 subtests passed**.
- Ruff on the package and changed tests, scoped ty in both environments, and
  `git diff --check` passed.

The remaining failures are confined to unchanged `tests/test_embeddings.py`:
missing `_check_dependency_family`/`version` and loader setup for the previously
removed dependency and V4 hardware guards. The refactor does not restore those
guards. Vertex utilities and transport retain **100% statement coverage**; overall
package coverage is **95%**.

### Shared Embedder abstract base (October 26, 2026)

`internvl.embedder.Embedder` replaces the private experiment protocol. Both concrete
encoders inherit its abstract text/image methods and ordinary typed `dimension`
attribute. The interface has no constructor or backend imports. `ImageInput` is
defined there and explicitly re-exported from `embeddings.py`. The lazy experiment
cache and Vertex persistence callbacks and concrete casts remain unchanged.
READMEs, dependencies, configuration, and transport behavior are unchanged.

Verification used synthetic encoders, mocked Vertex transports, temporary LanceDB
databases, `CUDA_VISIBLE_DEVICES=''`, and `HF_HUB_OFFLINE=1`. Pixi's editable-package
cache required execution outside the sandbox; completed CPU suites also ran outside
the sandbox. No live cloud calls, model downloads, or GPU checks were performed.

- Focused interface, classification, and Vertex tests: **147 passed, 74 subtests passed**.
- Locked modern CPU suite: **312 passed, 2 skipped, 10 failed; 339 subtests passed**.
- Locked Jina CPU suite: **312 passed, 3 skipped, 9 failed; 339 subtests passed**.
- Ruff on the package and new interface tests, scoped ty in both CPU environments
  (now including `embedder.py`), and `git diff --check` passed.

The full suites used `pixi run --manifest-path src/internvl/pyproject.toml --locked
-e ENV python -m pytest -c src/internvl/pyproject.toml --tb=short -q`. All failures
remain in the existing embedding-guard tests listed above, concerning removed
dependency/hardware guards and their loader setup. The new tests cover abstract
instantiation rejection, incomplete subclasses, concrete implementations, the
image-input re-export, and interface import isolation. Existing lazy reuse,
dimension, image-input, callback persistence, restart, and complete-cache tests
remain in place and passed.

### Caller-owned progress counts (October 26, 2026)

`_progress_callback` now returns a stateless `functools.partial` that forwards
cumulative completed counts to the tracker, or None when tracking is disabled.
Insertion loops use their existing source counters plus `initial_completed`.
Reference orchestration carries that offset from text into images; Vertex uses
selected reused identities rather than all stored source rows. Validation-cache
preparation passes its reused count into image insertion. Initial zero/reused
reports, successful-batch reporting, failure propagation, and per-response Vertex
persistence remain unchanged. No README, configuration, or dependency changes were
made for this refactor.

Synthetic verification used offline model loading, mocked Vertex transports, and
temporary LanceDB databases. CPU suites ran outside the sandbox; no live cloud
requests, model downloads, or GPU checks were performed.

- Focused tracking, classification, and Vertex tests: **203 passed, 74 subtests passed**.
- Locked modern CPU suite: **315 passed, 2 skipped, 10 failed; 339 subtests passed**.
- Locked Jina CPU suite: **315 passed, 3 skipped, 9 failed; 339 subtests passed**.
- Ruff on the package and changed tests, scoped ty in both CPU environments, and
  `git diff --check` passed.

Full suites used the locked Pixi commands above with `CUDA_VISIBLE_DEVICES=''`
and `HF_HUB_OFFLINE=1`, plus separate temporary coverage files. All failures remain
in the existing embedding dependency/hardware guard tests. New checks verify
stateless forwarding and disabled tracking; expanded integration assertions cover
resumed text-to-image counts, partial validation caches, final partial batches,
and duplicate/unrequested stored rows. Existing cache, interruption, insertion
failure, and logging failure tests passed.

### Loop-owned tqdm progress (October 26, 2026)

Progress reporting now uses context-managed tqdm bars in description generation,
text insertion, image insertion, and evaluation. W&B retains configuration,
source-count summaries, final metrics/tables, and report artifacts; intermediate
progress metrics and their callbacks, counters, offsets, and custom axes are removed.
Vertex's per-response persistence callbacks are unchanged. Reference/cache
preparation no longer computes totals or reused counts just for reporting.

Bars count returned description requests or records/images in successful batches,
including final partial batches. Resumed insertion shows newly processed work only.
Image insertion uses a running count without a total, keeping decoding streaming
and avoiding an extra metadata scan. Other bars have locally known totals. Bars
refresh at a five-second minimum interval, remain enabled without a TTY, and close
on exceptions and interrupts. Previously saved records are excluded before decoding
as before. Complete caches still avoid constructing an encoder or making requests.

Verification used synthetic encoders, mocked Vertex transports, temporary LanceDB
databases, disabled GPUs, and offline Hugging Face loading. Tests ran outside the
sandbox for LanceDB and the local W&B service. No model downloads or live cloud
requests were used. The existing offline SDK test now checks saved `output_raw`
records for a completed tqdm bar. This verifies native console capture locally;
hosted W&B Logs-tab rendering remains untested.

- Focused tracking, classification, Vertex, and descriptor lifecycle suite:
  **215 passed, 1 skipped; 95 subtests passed**.
- Full locked modern CPU suite: **306 passed, 2 skipped, 10 failed; 339 subtests passed**.
- Full locked Jina CPU suite: **306 passed, 3 skipped, 9 failed; 339 subtests passed**.
- Ruff on the package and changed tests, scoped ty in both CPU environments, and
  `git diff --check` passed.

Full suites used the locked Pixi commands above with `CUDA_VISIBLE_DEVICES=''`,
`HF_HUB_OFFLINE=1`, and separate temporary coverage files. All failures remain in
the existing embedding dependency/hardware guard tests. Retired progress-metric
tests were replaced by real tqdm counter/closure checks, batch failure and
interruption checks, pending-work counts for restarted and cached Vertex runs,
and assertions that W&B receives only final reporting. READMEs, dependencies,
configuration, model behavior, persistence, and report formats were unchanged.

### Optional validation-cache reuse in dimension sweeps (October 28, 2026)

`experiment matroyshka-sweep` now accepts the configured local or Vertex model.
`sweep.source_dim` describes the existing reference table and any validation
cache; `sweep.dimensions` sets evaluation order. Targets must be a nonempty list
of unique positive integers no greater than the source dimension, with booleans
rejected. Gemini retains 3072 and `[768, 1536, 3072]` as omitted-setting defaults.
Other models require explicit settings. The shipped local YAML files contain
null placeholders so strict CLI overrides can set both fields.

For example, given an existing 4096-dimensional Qwen reference table, the local
configuration can select `embedding.model_id=Qwen/Qwen3-VL-Embedding-8B`,
`sweep.source_dim=4096`, and `sweep.dimensions=[256,512,1024,2048,4096]` through
`--set` assignments. Select the table and output directory explicitly. These
settings do not build references or verify their model/dimension compatibility.

The sweep uses `lancedb.validation_table` as its only cache toggle:

- A supplied nonempty name requires the existing cache. Prefix tables are
  prepared for both reference and validation sources. Evaluation reads vectors
  by identity in annotation order, without model construction or image decoding.
  Missing tables or identities fail without inference or repair.
- Omitted or null prepares only reference prefixes. Each dimension constructs
  the normal local or Vertex experiment, encodes validation images in bounded
  batches at that dimension, and discards vectors after evaluation. Effective
  configurations disable validation caching and record `validation_table: null`,
  as do combined summary entries. This overrides the ordinary Vertex cache flag
  within the sweep only.

The source dimension reuses the original table names. Smaller dimensions retain
the `<source>_d<dimension>` naming and existing destination reuse/failure behavior.
All source and destination names must be distinct, even when the target list
omits the source dimension. The prebuilt reference table remains mandatory.
Every dimension keeps its own W&B context, effective configuration, and report;
only complete success writes the combined summary. Earlier reports survive a
later encoding or evaluation failure. Ordinary evaluation and Vertex cache
resumption are unchanged.

Implementation and verification took place in a separate worktree based on
`a143af5`, using the existing modern CPU environment with `PYTHONPATH=src` pointing
at the worktree. Checks used synthetic 3072/4096-dimensional vectors, PIL images,
temporary LanceDB databases, mocked encoders/W&B, and the existing offline W&B
tests. `CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1` remained set. The sandboxed test
process stalled without output and was interrupted; completed tests ran outside
the sandbox. No weights, datasets, cloud requests, GPU jobs, or services were used.

Verification results:

- Sweep-only suite: **65 passed**.
- Focused sweep, classification, Vertex, CLI, tracking, and retrieval suites:
  **303 passed, 182 subtests passed**. The existing YAML-default comparison now
  explicitly checks Gemini's sweep defaults and the local null placeholders.
- Package and changed-test Ruff checks, scoped ty, and `git diff --check` passed.

The focused pytest command used the existing environment's Python with
`-m pytest -c src/internvl/pyproject.toml tests/test_matroyshka_sweep.py
tests/test_classification.py tests/test_vertex_embeddings.py tests/test_internvl.py
tests/test_tracking.py tests/test_retrieval.py --no-cov -q` from the worktree.
Ruff used `--config src/internvl/pyproject.toml`; ty used `--project src/internvl`
and the existing CPU environment as its explicit `--python` path.

Coverage includes cached Gemini regression behavior, Qwen prefixes and original
4096-dimensional table reuse, annotation order, omitted/null caches, separate
encoding passes with partial batches at every dimension, no validation-table
writes, missing source tables/vectors, invalid dimensions and names, ordered CLI
overrides, unchanged source tables, and retained reports without a combined
summary after later failures. Source compatibility remains an assumption; these
checks do not establish real-model or GPU acceptance. README files are unchanged.

### Cached Gemini dimension sweep (October 27, 2026)

The shared CLI now accepts `experiment matroyshka-sweep` with `cfg/ge_v2.yml`
and the usual ordered strict overrides. It assumes complete 3072-dimensional
Gemini reference and validation tables, derives float32 prefix tables for 768 and
1536 dimensions, and evaluates the original tables at 3072. Each missing table
uses one streaming creation call; successfully opened destinations are reused.
No compatibility scans, cache filling, source filtering, normalization, repair,
or cleanup are introduced. Native storage failure tests do not establish
crash durability.

Each dimension has its own native W&B context and effective configuration artifact.
Standard reports live under `matroyshka_sweep_<UTC>/d<dimension>/evaluation_<UTC>/`;
a combined `summary.json` is written only after all dimensions succeed. Cache
lookup restores annotation order without decoding images or constructing encoders.
Reference counts remain null when no matching local index summary exists.

Verification used the existing modern CPU environment, synthetic 3072-dimensional
vectors, temporary LanceDB databases, mocked W&B for sweep runs, and the existing
native offline W&B tests. GPUs were disabled with `CUDA_VISIBLE_DEVICES=''` and
Hugging Face loading was offline with `HF_HUB_OFFLINE=1`. The sandboxed test run
stalled without output and was interrupted; completed tests ran outside the sandbox.

- Focused sweep, shared CLI, retrieval, classification, Vertex, and tracking suite:
  **244 passed; 182 subtests passed**.
- Final sweep-only suite, including additional unlimited-scan and native streaming
  failure cases: **15 passed**.
- Package and new-test Ruff checks, scoped ty (including the new module), and
  `git diff --check` passed.

Checks cover exact prefixes and fixed-size schemas, metadata/ID preservation,
multiple batches beyond default query limits, unchanged sources, reuse without a
new table version, interrupted readers, storage/open failures, collision rejection,
missing vectors, and preservation of earlier reports after a later failure.
Independent NumPy cosine calculations predict different winning neighbors across
dimensions. CLI checks cover ordered overrides, help without YAML loading, three
tracking contexts, effective configuration artifacts, and uploaded report contents.
No production tables, real dataset, models, GPU, Gemini requests, hosted W&B runs,
or HPC jobs were used. README files, dependency versions, and the lockfile are unchanged.


## Classification and ranked retrieval metrics (2026-10-27)

The shared evaluator now reports majority-vote macro/micro precision, recall, and
F1, top-five vector hit rate, Precision@k, MRR@k, nDCG@k, MAP@R, and two cosine
means. Existing majority-vote and nearest-neighbor accuracies remain unchanged.
Local evaluation, cached Vertex evaluation, W&B, and cached dimension sweeps share
the same calculations. Every aggregate scalar is published to each sweep row and
W&B's `evaluation/` summary; per-class tables add precision and F1.

For each query, R counts all stored reference rows with exactly the true label,
regardless of reference type. One unlimited streamed class-column scan supplies
these counts. Search depth is max(k,5,R), but voting and saved support remain the
first k rows. Repeated labels occupy separate ranks. Top-five hit rate is the
fraction of queries with a match at ranks 1–5. Precision@k divides matches in the
first k by the actual returned support length. MRR@k averages the reciprocal of
the first matching rank within k. Binary DCG sums relevance/log2(rank+1); nDCG@k
divides by ideal DCG with min(k,R) matches. AP@R sums Precision@i at matching ranks
i through R and divides by R; MAP@R averages these values. Empty retrieval or R=0
contributes zero, retaining every validation query in the denominator.

Classification uses majority votes. Macro averages cover labels in truth or
predictions, independently of the display list; macro F1 averages class F1s.
Micro scores pool TP, FP, and FN. Abstentions contribute FN without a synthetic
class. Undefined ratios are zero, except displayed recall stays null for absent
support. Without abstentions, micro scores equal single-label accuracy, following
[scikit-learn's definitions](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html).
The nearest cosine mean and mean of per-query support cosine means exclude empty
retrievals, retain negative scores, and return null if no query qualifies.

Use macro F1 to compare class performance and nDCG@k/MAP@R to inspect neighborhood
quality. [BEIR](https://arxiv.org/html/2104.08663v4#S3.SS3) uses nDCG for ranked
retrieval evaluation. [Musgrave et al.](https://arxiv.org/html/2003.08505v3#S3.SS2)
motivate MAP@R as more informative about surrounding retrievals than nearest-only
accuracy. Keep reference inputs and retrieval settings fixed across comparisons.
Cosine averages are diagnostics: larger values alone do not demonstrate improved
semantic separation; see [Steck et al.](https://arxiv.org/abs/2403.05440).
Hubness analysis and class-pair confusion matrices remain follow-up work.

No metric flags, dependencies, compatibility validation, or historical report
rewrites are introduced. Deeper retrieval can increase evaluation cost when R is
large. Checks use synthetic CPU fixtures, temporary LanceDB tables, and mocked
model/cloud/tracking backends; no real checkpoint, cloud, GPU, or HPC acceptance
is implied.

Verification results:

- Focused retrieval/classification/tracking/Vertex/sweep suites: modern CPU **226
  passed, 111 subtests passed**; Jina CPU **227 passed, 111 subtests passed**.
  The Jina run also included the subsequently added invalid-k test.
- Final classification/Vertex/CLI regression suites, including the new scan-depth,
  empty-table, and cached-metric parity assertions: **167 passed, 145 subtests
  passed** in each CPU environment.
- Package Ruff, scoped ty in both environments, and `git diff --check` passed.
- Tests cover hand-calculated multiclass scores, abstentions, predicted-only and
  unused labels, empty inputs, ranks five/six, repeated labels, R greater than
  max(k,5), short/empty support, negative and query-weighted cosine means, a
  3,300-row mixed-reference count scan, and report/W&B/sweep publication.
- Commands used each installed `src/internvl/.pixi/envs/{default,jina-cpu}/bin/python
  -m pytest -c src/internvl/pyproject.toml ... --no-cov -q`, with
  `CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1`. Sandbox runs stalled at LanceDB and
  were interrupted; completed suites ran outside the sandbox. No new dependencies,
  model downloads, live services, or GPU work were used. Unrelated embedding-guard
  suites were outside this focused verification.
