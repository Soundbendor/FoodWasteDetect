# Standalone internvl environment

This project installs the whole-image Jina CLIP/Qdrant experiment independently
of the legacy image pipeline. Run the commands below from the repository root:
`--project` selects the environment without changing the application's working
directory, so existing `cfg/` paths retain their meaning.

## Installation

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then select
one profile. Standard CPython 3.14.8 is pinned; free-threaded Python is not used.

```sh
uv sync --project src/internvl --locked --extra cpu
uv run --project src/internvl --locked --extra cpu internvl --help
uv run --project src/internvl --locked --extra cpu internvl experiment run --config_file cfg/classification.yml
```

For CUDA 12.8:

```sh
uv sync --project src/internvl --locked --extra cu128
uv run --project src/internvl --locked --extra cu128 internvl --help
```

Always select the same extra for sync and run. The extras are mutually exclusive;
changing the extra replaces the PyTorch installation in `src/internvl/.venv`.
For a separate CUDA environment, prefix **both** commands with
`UV_PROJECT_ENVIRONMENT=.venv-cu128` (resolved relative to the nested project).
The locked accelerator profiles target Linux/Windows systems with matching wheels;
CUDA execution additionally requires a compatible NVIDIA driver and GPU.

| Dependency | CPU | CUDA 12.8 |
| --- | --- | --- |
| Python | 3.14.8 | 3.14.8 |
| torch | 2.14.1+cpu | 2.11.0+cu128 |
| torchvision | 0.29.1+cpu | 0.26.0+cu128 |
| sentence-transformers | 5.7.0 | 5.7.0 |
| transformers | 4.57.6 | 4.57.6 |
| datasets | 4.8.5 | 4.8.5 |

`uv.lock` records all exact resolutions. PyTorch uses explicit package-specific
[uv indexes](https://docs.astral.sh/uv/guides/integration/pytorch/).
Transformers is constrained to `<5`: the initial 5.5.4 installation passed
synthetic tests but failed the adapter's image embedding shape/finite-value check
with both PyTorch profiles. The initially proposed `<5.6` constraint was therefore
insufficient. Sentence Transformers is constrained to `>=5.4,<6` because version 6
requires Transformers 5. Datasets is constrained to `<5` because version 5 requires
Hugging Face Hub 1.x, while Transformers 4 requires Hub below 1. The resolver selects
the newest compatible releases within these ranges. Model code is not patched. `requests`, `einops`, and `timm` are explicit
dependencies of Jina's remote code. Model choices, revision policy, configuration,
and normal application caches are unchanged. Ollama's executable/models and a
remote Qdrant service are separate infrastructure; see the repository README.

The repository-root package still owns the legacy image commands. It no longer
installs `internvl`; install this nested project for that command.

## Development

The default `dev` and `test` dependency groups install Ruff, ty, pytest, and
pytest-cov. Existing unittest tests run through pytest without conversion.

```sh
uv run --project src/internvl --locked --extra cpu ruff check src/internvl
uv run --project src/internvl --locked --extra cpu ty check --project src/internvl
uv run --project src/internvl --locked --extra cpu pytest -c src/internvl/pyproject.toml
uv build --project src/internvl --out-dir src/internvl/dist
```

Pytest collects the repository's `tests/` and reports `internvl` coverage. Routine
tests use synthetic fixtures and mocked model construction, plus in-memory Qdrant.
They do not download models, start services, or use a GPU. An existing explicitly
skipped remote-endpoint lifecycle test remains; dependency skips are not expected
in the complete environment.

The enforced ty scope is `cli.py` and `schemas.py`. To inspect the whole package:

```sh
uv run --project src/internvl --locked --extra cpu ty check --project src/internvl --config 'src.include=["**"]'
```

Two existing diagnostics are deferred: optional Ollama response content passed to
Pydantic in `generate_descriptors.py`, and a string passed to the datasets
`NamedSplit` parameter in `image_dataset.py`. Unresolved imports are not suppressed.

## Explicit real-model smoke test

This command downloads Jina weights and remote implementation code, generates a
synthetic image, and encodes one text and one image on CPU:

```sh
uv run --project src/internvl --locked --extra cpu python -m internvl.scripts.smoke_jina
```

For the separate CUDA library environment, still using CPU execution:

```sh
UV_PROJECT_ENVIRONMENT=.venv-cu128 uv run --project src/internvl --locked --extra cu128 python -m internvl.scripts.smoke_jina
```

Each modality must return a finite, nonzero float32 array of shape `(1, 1024)`.
The parent establishes fresh Hugging Face, dynamic-module, Sentence Transformers,
PyTorch, Xet, and temporary-file caches before the child imports ML libraries.
It stops and waits for the worker before deleting its temporary directory on
success, failure, Ctrl-C, or SIGTERM. SIGKILL and machine failure cannot guarantee
cleanup. Every invocation downloads again; installed libraries and the uv
environment persist. This is deliberately outside routine pytest discovery.

Passing this test establishes one real-model CPU encoding per modality; it does
not establish accuracy, real dataset compatibility, CUDA execution, Ollama,
remote Qdrant, or HPC behavior.

## Verified setup

Both locked profiles were installed separately on Linux with CPython 3.14.8.
Each passed 81 tests and 151 subtests with 93% package coverage, one existing
explicit skip, and no missing-dependency skips. Both real-model smoke runs passed
text and image encoding on CPU, and their temporary caches were removed. Synthetic
success, failure, SIGINT, and SIGTERM cleanup checks passed.

Ruff and the scoped ty check passed. Wheel and source distributions built;
non-editable imports and console help worked outside the checkout without
`PYTHONPATH`. Repeated locked syncs completed without dependency changes. The root
wheel retained the three legacy image commands and excluded `internvl`.
CUDA execution, real datasets, Ollama, remote Qdrant, and HPC remain unverified.
