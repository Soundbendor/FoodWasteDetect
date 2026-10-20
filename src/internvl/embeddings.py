"""Inference-only Jina CLIP v2 embeddings. No import-time backends."""

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image
from sentence_transformers import SentenceTransformer

_SUPPORTED_MODELS = ["jinaai/jina-clip-v2"]
_DIMENSION = 1024
ImageInput = str | Path | Image.Image


def _load_backend(model_id: str, device: str) -> SentenceTransformer:
    """Load Jina directly, using the library's normal caching and downloads."""
    backend = SentenceTransformer(
        model_id,
        device=device,
        trust_remote_code=True,
        truncate_dim=None,
        model_kwargs={"torch_dtype": torch.float32},
        config_kwargs={
            "use_text_flash_attn": False,
            "use_vision_xformers": False,
        },
    )
    # Jina-clip-v2 lacks modality metadata - inject it.
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
    """Batch text and images into full-size Jina CLIP vectors."""

    def __init__(
        self,
        model_id: str,
        batch_size: int = 32,
        *,
        device: str = "auto",
    ) -> None:
        """Load a supported model with the requested device and batch size."""
        if model_id not in _SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported model {model_id!r}. Supported models: {_SUPPORTED_MODELS}"
            )
        self.batch_size = batch_size
        self.dimension = _DIMENSION
        self.model = _load_backend(model_id, device)

    def _encode(self, inputs: list[str] | list[Image.Image]) -> NDArray[np.float32]:
        """Delegate batching and NumPy conversion to the library."""
        if not inputs:
            return np.empty((0, self.dimension), dtype=np.float32)
        values = self.model.encode(
            inputs,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            precision="float32",
            show_progress_bar=False,
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
        return self._encode(list(texts))

    def encode_images(self, images: Iterable[ImageInput]) -> NDArray[np.float32]:
        """Load images into memory and convert to embedding vectors."""
        return self._encode([_prepare_rgb_image(image) for image in images])
