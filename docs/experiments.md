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
