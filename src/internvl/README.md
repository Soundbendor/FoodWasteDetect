# Standalone internvl environment

This environment installs the whole-image Jina CLIP/Qdrant experiment independently
of the legacy image pipeline. Run these commands from the repository root.

## Installation

Install [Pixi](https://pixi.prefix.dev/latest/installation/) 0.81.0 or newer.
Linux x86-64 and glibc 2.28 or newer are required. Pixi installs Python 3.13.13,
Python dependencies, Ollama, and Qdrant from the committed `pixi.lock`.

For CPU execution:

```sh
pixi install --manifest-path src/internvl/pyproject.toml --locked -e default
pixi run --manifest-path src/internvl/pyproject.toml --locked -e default internvl --help
pixi run --manifest-path src/internvl/pyproject.toml --locked -e default internvl experiment run --config_file cfg/classification.yml
```

For GPU execution, including V100 GPUs:

```sh
pixi install --manifest-path src/internvl/pyproject.toml --locked -e cu126
pixi run --manifest-path src/internvl/pyproject.toml --locked -e cu126 internvl --help
```

Select `-e default` or `-e cu126` for each command. Both environments can coexist.
The GPU environment uses PyTorch with CUDA 12.6 and Ollama with CUDA 12.9.
It requires a CUDA 12.9-compatible NVIDIA driver, such as 580.178.04. Prepare the
environment on a permitted network-connected node before submitting jobs, then
use `pixi run --as-is` in the job to avoid installation there.

For dataset configuration, persistent model caches, starting Qdrant, downloading
Ollama models, and running an experiment, follow the
[repository setup guide](../../README.md#setup).

## V100 kernel check

Run the [V100 kernel check](../../docs/experiments.md#v100-kernel-check) inside an
approved GPU allocation before running experiments. CUDA availability alone does
not verify kernel execution. Local synthetic checks do not establish GPU or HPC
compatibility.

## Development

Both environments include Ruff, ty, pytest, coverage, and package build tools.

```sh
pixi run --manifest-path src/internvl/pyproject.toml --locked -e default lint
pixi run --manifest-path src/internvl/pyproject.toml --locked -e default typecheck
pixi run --manifest-path src/internvl/pyproject.toml --locked -e default test
pixi run --manifest-path src/internvl/pyproject.toml --locked -e default build
```

The test task uses synthetic fixtures, mocked model construction, and in-memory
Qdrant with GPUs hidden and Hugging Face offline. It does not download models or
start servers. The type check covers `cli.py` and `schemas.py`. Builds are written
to `src/internvl/dist/`.

## Explicit real-model smoke test

This separate check downloads Jina weights and remote code, then encodes one
synthetic image and one text on CPU:

```sh
pixi run --manifest-path src/internvl/pyproject.toml --locked -e default python -m internvl.scripts.smoke_jina
pixi run --manifest-path src/internvl/pyproject.toml --locked -e cu126 python -m internvl.scripts.smoke_jina
```

Each invocation downloads into temporary caches that are removed when the worker
exits normally or handles an interrupt. These downloads do not prepare the
application's persistent cache. Run only where model downloads are permitted.

See [environment maintenance](../../docs/experiments.md#pixi-environment-maintenance)
for dependency updates, compatibility constraints, and service acceptance checks.
