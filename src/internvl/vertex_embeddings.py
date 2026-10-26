"""Vertex Gemini Embedding 2 requests with bounded per-batch concurrency."""

from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from math import ceil, isfinite
from pathlib import Path
from typing import TypeVar

import numpy as np
from google import genai
from google.genai import types
from numpy.typing import NDArray
from PIL import Image

_Input = TypeVar("_Input")
_PREFIX = "task: classification | query: "


class VertexEmbeddingModel:
    """Encode individual records through Vertex using Application Default Credentials."""

    def __init__(
        self,
        *,
        project: str,
        location: str = "global",
        max_concurrency: int = 4,
        max_attempts: int = 5,
        request_timeout: float = 60,
        classification_prompt: bool = False,
        truncate_dim: int | None = None,
    ) -> None:
        for name, value in (("project", project), ("location", location)):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"vertex.{name} must be a nonempty string")
        for name, value in (("max_concurrency", max_concurrency), ("max_attempts", max_attempts)):
            if type(value) is not int or value <= 0:
                raise ValueError(f"vertex.{name} must be a positive integer")
        if (
            isinstance(request_timeout, bool)
            or not isinstance(request_timeout, (int, float))
            or not isfinite(request_timeout)
            or request_timeout <= 0
        ):
            raise ValueError("vertex.request_timeout must be positive finite seconds")
        if type(classification_prompt) is not bool:
            raise ValueError("vertex.classification_prompt must be a boolean")
        if truncate_dim is not None and (
            type(truncate_dim) is not int or not 128 <= truncate_dim <= 3072
        ):
            raise ValueError("truncate_dim must be None or an integer between 128 and 3072")
        self.dimension = 3072 if truncate_dim is None else truncate_dim
        self.project = project
        self.location = location
        self.max_concurrency = max_concurrency
        self.classification_prompt = classification_prompt
        self.http_options = types.HttpOptions(
            api_version="v1",
            timeout=ceil(request_timeout * 1000),
            retry_options=types.HttpRetryOptions(
                attempts=max_attempts,
                initial_delay=1,
                exp_base=2,
                jitter=1,
                max_delay=60,
                http_status_codes=[408, 429, 500, 502, 503, 504],
            ),
        )

    def _request(self, client: genai.Client, content: types.Content) -> NDArray[np.float32]:
        response = client.models.embed_content(
            model="gemini-embedding-2",
            contents=[content],
            config=types.EmbedContentConfig(output_dimensionality=self.dimension),
        )
        if response.embeddings is None or len(response.embeddings) != 1:
            raise ValueError("Vertex must return exactly one embedding per record")
        vector = np.asarray(response.embeddings[0].values, dtype=np.float32)
        if vector.shape != (self.dimension,) or not np.isfinite(vector).all() or not np.any(vector):
            raise ValueError(
                "Vertex embedding must have the selected dimension and finite, nonzero values"
            )
        return vector

    def _encode(
        self, inputs: Iterable[_Input], prepare: Callable[[_Input], types.Content]
    ) -> NDArray[np.float32]:
        records = list(inputs)
        if not records:
            return np.empty((0, self.dimension), dtype=np.float32)
        vectors = np.empty((len(records), self.dimension), dtype=np.float32)
        with genai.Client(
            vertexai=True,
            project=self.project,
            location=self.location,
            http_options=self.http_options,
        ) as client:
            executor = ThreadPoolExecutor(max_workers=self.max_concurrency)
            futures = {}
            try:
                # Prepare images inside workers to bound serialized image memory too.
                def encode(record: _Input) -> NDArray[np.float32]:
                    return self._request(client, prepare(record))

                for index, record in enumerate(records):
                    futures[executor.submit(encode, record)] = index
                for future in as_completed(futures):
                    vectors[futures[future]] = future.result()
            finally:
                executor.shutdown(wait=True, cancel_futures=True)
        return vectors

    def encode_text(self, texts: Iterable[str]) -> NDArray[np.float32]:
        """Encode class names or descriptions, preserving source order."""

        def prepare(text: str) -> types.Content:
            if self.classification_prompt:
                text = _PREFIX + text
            return types.Content(parts=[types.Part.from_text(text=text)])

        return self._encode(texts, prepare)

    def encode_images(self, images: Iterable[str | Path | Image.Image]) -> NDArray[np.float32]:
        """Send RGB PNG pixels without labels, filenames, or resizing."""

        def prepare(image: str | Path | Image.Image) -> types.Content:
            if isinstance(image, (str, Path)):
                with Image.open(image) as original:
                    rgb = original.convert("RGB")
            else:
                rgb = image.convert("RGB")
            with rgb, BytesIO() as buffer:
                rgb.save(buffer, format="PNG")
                part = types.Part.from_bytes(data=buffer.getvalue(), mime_type="image/png")
            parts = [part]
            if self.classification_prompt:
                parts.insert(0, types.Part.from_text(text=_PREFIX))
            return types.Content(parts=parts)

        return self._encode(images, prepare)
