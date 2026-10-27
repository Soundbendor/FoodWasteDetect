"""Shared interface for text and image embedding backends."""

from abc import ABC, abstractmethod
from collections.abc import Iterable
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image

ImageInput = str | Path | Image.Image


class Embedder(ABC):
    """Encode text and images into vectors with a declared dimension."""

    dimension: int

    @abstractmethod
    def encode_text(self, texts: Iterable[str]) -> NDArray[np.float32]:
        """Encode text records in input order."""

    @abstractmethod
    def encode_images(self, images: Iterable[ImageInput]) -> NDArray[np.float32]:
        """Encode image records in input order."""
