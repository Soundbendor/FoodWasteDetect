"""Inference-only multimodal embeddings. No import-time backends."""

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image
from sentence_transformers import SentenceTransformer

# Keep model-specific behavior here; batching remains owned by Sentence Transformers.
_MODEL_SPECS = {
    "jinaai/jina-clip-v2": {
        "dimension": 1024,
        "load": {
            "trust_remote_code": True,
            "model_kwargs": {"torch_dtype": torch.float32},
            "config_kwargs": {"use_text_flash_attn": False, "use_vision_xformers": False},
        },
    },
    "jinaai/jina-embeddings-v4": {
        "dimension": 2048,
        "load": {
            "trust_remote_code": True,
            "model_kwargs": {"torch_dtype": "auto", "attn_implementation": "sdpa"},
        },
        "encode": {"task": "retrieval"},
        "text": {"prompt_name": "passage"},
    },
    "google/embeddinggemma-2": {
        "dimension": 768,
        "load": {"config_kwargs": {"audio_config": None}},
        "text": {"prompt_name": "Document"},
        "image_dict": True,
    },
    "openai/clip-vit-base-patch32": {
        "dimension": 512,
        "load": {
            "model_kwargs": {"torch_dtype": torch.float32},
            "processor_kwargs": {"model_max_length": 77},
        },
    },
}
ImageInput = str | Path | Image.Image


def _load_backend(model_id: str, device: str) -> SentenceTransformer:
    """Load directly, using the library's normal caching and downloads."""
    if device == "auto":
        from sentence_transformers.util import get_device_name

        device = get_device_name()
    options = dict(_MODEL_SPECS[model_id]["load"])
    if model_id == "google/embeddinggemma-2":
        # Check native support on the selected GPU, not the default CUDA device.
        dtype = torch.float32
        if torch.device(device).type == "cuda":
            with torch.cuda.device(device):
                if torch.cuda.is_bf16_supported(including_emulation=False):
                    dtype = torch.bfloat16
        options["model_kwargs"] = {"torch_dtype": dtype}
    backend = SentenceTransformer(model_id, device=device, truncate_dim=None, **options)
    if model_id.startswith("jinaai/"):
        # Custom Jina modules may lack Sentence Transformers modality metadata.
        module: Any = backend[0]
        if not hasattr(module, "modalities"):
            module.modalities = ["text", "image"]
    return backend


def _prepare_rgb_image(image: ImageInput) -> Image.Image:
    """Convert images to RGB colorspace, loading the image if not already open."""
    if isinstance(image, (str, Path)):
        original = Image.open(image)
        try:
            return original.convert("RGB")
        finally:
            original.close()
    return image.convert("RGB")


class EmbeddingModel:
    """Batch text and images into full-size model vectors."""

    def __init__(
        self,
        model_id: str,
        batch_size: int = 32,
        *,
        device: str = "auto",
    ) -> None:
        """Load a supported model with the requested device and batch size."""
        if model_id not in _MODEL_SPECS:
            raise ValueError(
                f"Unsupported model {model_id!r}. Supported models: {list(_MODEL_SPECS)}"
            )
        self.batch_size = batch_size
        self.model_id = model_id
        self.dimension = _MODEL_SPECS[model_id]["dimension"]
        self.model = _load_backend(model_id, device)

    def _encode(
        self, inputs: list[Any], **options: Any
    ) -> NDArray[np.float32]:
        """Delegate batching and NumPy conversion to the library."""
        if not inputs:
            return np.empty((0, self.dimension), dtype=np.float32)
        options = {**_MODEL_SPECS[self.model_id].get("encode", {}), **options}
        values = self.model.encode(
            inputs,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            precision="float32",
            show_progress_bar=False,
            **options,
        )
        values = np.asarray(values, dtype=np.float32)
        if (
            values.shape != (len(inputs), self.dimension)
            or not np.isfinite(values).all()
        ):
            raise ValueError("Invalid embedding shape or nonfinite values")
        norms = np.linalg.norm(values, axis=1, keepdims=True)
        if (norms == 0).any():
            raise ValueError("Model returned zero-length embeddings.")
        return values

    def encode_text(self, texts: Iterable[str]) -> NDArray[np.float32]:
        """Generate embedding vectors from text inputs."""
        options = _MODEL_SPECS[self.model_id].get("text", {})
        return self._encode(list(texts), **options)

    def encode_images(self, images: Iterable[ImageInput]) -> NDArray[np.float32]:
        """Load images into memory and convert to embedding vectors."""
        prepared = [_prepare_rgb_image(image) for image in images]
        if _MODEL_SPECS[self.model_id].get("image_dict"):
            return self._encode([{"image": image} for image in prepared])
        return self._encode(prepared)
