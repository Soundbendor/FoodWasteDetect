# Experiment verification notes

Read [development](development.md) for environment selection and routine checks,
[encoder contracts](classification/encoders.md) for model protocols, and
[experiment contracts](classification/experiments.md) for persistence and evaluation.
Procedures here retain the [root guide](../AGENTS.md)'s explicit authorization
requirements.

## Verification baseline and boundaries

The CPU test suite checks command dispatch, import boundaries, dataset/reference
selection, LanceDB persistence/prefix derivation, evaluation/retrieval metrics,
and mocked Ollama/Vertex/W&B lifecycles. Restricted sandboxes can stall inside
initial `lancedb.connect()` setup; run LanceDB suites in an unrestricted local
CPU session with `CUDA_VISIBLE_DEVICES=""` and `HF_HUB_OFFLINE=1`.

The legacy annotation scripts under `src/util/` include scripts with hard-coded
paths and import-time data writes, particularly `joint_labels.py`. Inspect them as
text before use; do not import them to discover their behavior.

## Environment and native package constraints

`src/classification/pyproject.toml` and `src/classification/pixi.lock` pin two
library families on CPython 3.13.13, PyTorch 2.7.1, and torchvision 0.22.1:

| Environments | Models | Sentence Transformers | Transformers | PEFT |
| --- | --- | --- | --- | --- |
| `default`, `cu126` | EmbeddingGemma2, OpenAI CLIP, Qwen | 6.2.0 | 5.20.2 | 0.21.2 |
| `jina-cpu`, `jina-cu126` | Jina CLIP v2, Jina Embeddings v4 | 5.7.0 | 4.57.6 | 0.17.1 |

- **Jina isolation**: Jina remote code fails on Transformers 5 (`clip_loss` import
  and `ROPE_INIT_FUNCTIONS["default"]`), so the Jina environments retain the
  compatible 4.57.6 stack instead of patching remote model code.
- **Ollama and PyTorch CUDA**: Conda-forge supplies native Ollama 0.9.6 (`cpu_hba3e463_4`
  on CPU; `cuda_129h63bc95c_4` with `cuda-version=12.9` on GPU, which includes
  `sm_70` in `libggml-cuda.so` and runs on cluster driver 580.178.04). PyTorch
  keeps its CUDA 12.6 wheels (`sm_70` included). Do not set a global
  `LD_LIBRARY_PATH` that mixes Ollama's bundled CUDA libraries with PyTorch's wheels.
- **Qdrant**: Pinned to 1.19.0 from the Anaconda channel for the legacy CLIP
  pipeline. Back up older storage and follow Qdrant's minor-version upgrade path
  before opening existing collections.

To update a dependency selectively, edit its constraint if pinned and refresh
the locked environments:

```sh
pixi update --manifest-path src/classification/pyproject.toml PACKAGE_NAME
pixi install --manifest-path src/classification/pyproject.toml --locked -e default
pixi install --manifest-path src/classification/pyproject.toml --locked -e jina-cpu
pixi install --manifest-path src/classification/pyproject.toml --locked -e cu126
pixi install --manifest-path src/classification/pyproject.toml --locked -e jina-cu126
```

## Cluster acceptance checks

Run these only inside an explicitly authorized GPU allocation after installing the
corresponding CUDA environment.

### V100 kernel check

```sh
pixi run --manifest-path src/classification/pyproject.toml --as-is -e cu126 python - <<'PYTHON'
import torch
import torchvision

assert torch.__version__ == "2.7.1+cu126"
assert torchvision.__version__ == "0.22.1+cu126"
assert torch.cuda.is_available()
assert torch.cuda.get_device_capability(0) == (7, 0)
assert "sm_70" in torch.cuda.get_arch_list()
x = torch.ones((32, 32), device="cuda")
assert (x @ x).sum().item() == 32768
torch.cuda.synchronize()
print("V100 kernel check passed on", torch.cuda.get_device_name(0))
PYTHON
```

### Real-model GPU smoke check (Jina v4 example)

Requires native BF16 hardware (e.g., A100/H100) and `jina-cu126`:

```sh
pixi run --manifest-path src/classification/pyproject.toml --as-is -e jina-cu126 python - <<'PYTHON'
import numpy as np
import torch
from PIL import Image
from classification.encoders.sentence_transformers import EmbeddingModel

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
torch.cuda.synchronize()
print("Real-model GPU smoke check passed on", torch.cuda.get_device_name(0))
PYTHON
```
