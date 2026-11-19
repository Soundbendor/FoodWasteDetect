# Development and environment checks

Read this before changing dependencies, packaging, or test commands. For initial
Pixi installation, use the commands in [AGENTS.md](../AGENTS.md#pixi-setup).
The manifest and lockfile are authoritative for exact dependency versions.

## Environment selection

`src/classification/pyproject.toml` defines independent Linux x86-64 solve groups:

| Environment | Model family | Runtime |
| --- | --- | --- |
| `default` | EmbeddingGemma2, OpenAI CLIP, Qwen | CPU, modern dependencies |
| `cu126` | EmbeddingGemma2, OpenAI CLIP, Qwen | CUDA, modern dependencies |
| `jina-cpu` | Jina CLIP v2, Jina Embeddings v4 | CPU, Jina dependencies |
| `jina-cu126` | Jina CLIP v2, Jina Embeddings v4 | CUDA, Jina dependencies |

The default classification YAML selects Jina CLIP v2, so select a Jina environment
explicitly. CPU and CUDA features remain separate. The current stack uses CPython
3.13.13 and PyTorch 2.7.1; the CUDA environment name refers to PyTorch's CUDA 12.6
wheels. Its separate native Ollama package bundles CUDA 12.9. Qdrant remains
installed for the legacy CLIP pipeline; whole-image retrieval uses embedded LanceDB.
The Ollama Python SDK is a separate dependency from the native binary.
Every environment includes conda-forge `p7zip` 16.02, whose `7z` unpacks the ISIA
Food-500 split zip; run dataset downloads inside a Pixi environment.

Slurm wrappers use `--as-is`, so prepare the selected environment before submission.
Their defaults and model-family selection are described in
[script operations](classification/scripts.md). Environment/hardware selection is manual;
dependency-family and Jina V4 hardware guards are absent. See
[encoder contracts](classification/encoders.md) before choosing model hardware.

## Routine checks

Run from the repository root in an already prepared CPU environment:

```sh
pixi run --manifest-path src/classification/pyproject.toml --as-is -e default lint
pixi run --manifest-path src/classification/pyproject.toml --as-is -e default format-check
pixi run --manifest-path src/classification/pyproject.toml --as-is -e default typecheck
pixi run --manifest-path src/classification/pyproject.toml --as-is -e default test
pixi run --manifest-path src/classification/pyproject.toml --as-is -e default classification --help
```

Use `-e jina-cpu` for changes involving the Jina dependency family. The test task
hides GPUs and sets Hugging Face offline mode. It uses mocked pretrained loading
and service transports, synthetic images/FoodX251 fixtures, and temporary real
LanceDB databases. Keep explicit dependency skips where tests support them.
The scoped ty task uses `$CONDA_PREFIX` to avoid resolving an old `.venv`.
Use the `build` task for packaging changes; it writes distributions under
`src/classification/dist`. Run focused checks appropriate to the change and
report failures against the unchanged baseline.

## Evidence and acceptance

[Experiment notes](experiments.md) retain environment constraints and cluster
acceptance procedures, including the V100 kernel check. Synthetic tests establish
contracts for their fixtures; they do not establish model accuracy, GPU support,
Ollama/cloud availability, shared-filesystem behavior, or HPC throughput.

The real-model helper `classification.scripts.smoke_jina` downloads checkpoints into
temporary subprocess caches and is excluded from routine checks. Preserve the root
guide's authorization boundaries and identify unrun checks explicitly.

## Documentation maintenance

Keep durable behavior in the task-specific docs, scope and ownership in module
`AGENTS.md` files, and hardware/acceptance procedures in `docs/experiments.md`.
Documents under `docs/`, except those under `docs/plans/`, must have a direct entry
in the root documentation index with a task trigger. Update the owning document
instead of appending a new copy of the same contract to the root guide.
