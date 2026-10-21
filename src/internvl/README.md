# Standalone internvl environment

This project installs the whole-image Jina CLIP/Qdrant experiment independently
of the legacy image pipeline. Run the commands below from the repository root:
`--project` selects the environment without changing the application's working
directory, so existing `cfg/` paths retain their meaning.

## Installation

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then select
one profile. Standard CPython 3.13.13 is pinned; free-threaded Python is not used.
PyTorch 2.7 supports Python through 3.13, so Python 3.14 cannot be used with
these profiles. If uv cannot find the interpreter, see
[Python installation](#python-installation).

```sh
uv sync --project src/internvl --locked --extra cpu
uv run --project src/internvl --locked --extra cpu internvl --help
uv run --project src/internvl --locked --extra cpu internvl experiment run --config_file cfg/classification.yml
```

For CUDA 12.6:

```sh
uv sync --project src/internvl --locked --extra cu126
uv run --project src/internvl --locked --extra cu126 internvl --help
```

Always select the same extra for sync and run. The extras are mutually exclusive;
changing the extra replaces the PyTorch installation in `src/internvl/.venv`.
For a separate CUDA environment, prefix **both** commands with
`UV_PROJECT_ENVIRONMENT=.venv-cu126` (resolved relative to the nested project).
The locked accelerator profiles target Linux/Windows systems with matching wheels;
CUDA execution additionally requires a compatible NVIDIA driver and GPU.

| Dependency | CPU | CUDA 12.6 |
| --- | --- | --- |
| Python | 3.13.13 | 3.13.13 |
| torch | 2.7.1+cpu | 2.7.1+cu126 |
| torchvision | 0.22.1+cpu | 0.22.1+cu126 |
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

## Python installation

Install the pinned interpreter before syncing if needed:

```sh
uv python install 3.13.13
uv sync --project src/internvl --locked --extra cpu --managed-python
```

If uv reports **No download found**, check `uv --version` and
`uv python list 3.13 --all-versions`. Update uv's interpreter catalog by updating
uv, or use an installed standard CPython 3.13.13 interpreter. The previous
Python 3.14 workaround is incompatible with PyTorch 2.7. Keep the project's
`.python-version`, `requires-python`, and `uv.lock` consistent. See
[uv's interpreter documentation](https://docs.astral.sh/uv/concepts/python-versions/).

## V100 kernel check

The previous PyTorch 2.11 CUDA 12.8 build omitted Volta (`sm_70`, including V100),
which can cause `no kernel image is available for execution on the device`.
Both profiles now pin PyTorch 2.7.1 and its matching torchvision 0.22.1;
the GPU extra is renamed to `cu126`. See the
[PyTorch release notes](https://github.com/pytorch/pytorch/releases/tag/v2.11.0)
and [version matrix](https://github.com/pytorch/pytorch/blob/v2.7.1/RELEASE.md).

On a Linux x86_64 V100 compute node, check `nvidia-smi` and `ldd --version`:
the environment requires a CUDA 12.6-compatible driver and glibc 2.28 or newer.
Run this inside an approved GPU allocation after syncing `--extra cu126`:

```sh
uv run --project src/internvl --locked --extra cu126 python - <<'PYTHON'
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

Retry the application after this check. CUDA availability alone does not verify
kernel execution, and this small check does not establish model compatibility.

For dataset layout, Ollama/Qdrant setup, persistent caches, and the first
experiment, follow the [repository setup guide](../../README.md#setup).

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
UV_PROJECT_ENVIRONMENT=.venv-cu126 uv run --project src/internvl --locked --extra cu126 python -m internvl.scripts.smoke_jina
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

## Current verification

Both PyTorch 2.7.1 profiles installed on Linux x86_64 with CPython 3.13.13.
Each passed 79 synthetic tests with one existing skip and 93% package coverage;
the CUDA profile ran with GPUs hidden. Ruff, scoped ty, and CPU console help
passed. Imports confirmed torchvision 0.22.1 in both profiles and CUDA runtime
12.6 in the GPU build. The installed CUDA binary includes `sm_70`.

No real models, datasets, GPU kernels, Ollama, remote Qdrant, or HPC jobs were
run for this migration. Use the V100 kernel check above on the cluster before
retrying the application.

## Historical verification

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
