"""Shared interface for text and image embedding backends."""

from abc import ABC, abstractmethod
from collections.abc import Iterable
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image

ImageInput = str | Path | Image.Image


class Embedder(ABC):
    """Encode inputs as ordered float32 matrices with ``dimension`` columns.

    Each input produces one finite, nonzero vector. Empty calls return shape
    ``(0, dimension)``. Errors propagate without returning partial matrices.
    Backends preserve their model's normalization and keep caller-owned images open.
    """

    dimension: int

    @abstractmethod
    def encode_text(self, texts: Iterable[str]) -> NDArray[np.float32]:
        """Encode text records in input order."""

    @abstractmethod
    def encode_images(self, images: Iterable[ImageInput]) -> NDArray[np.float32]:
        """Encode paths or PIL images in input order, preserving caller-owned images."""
