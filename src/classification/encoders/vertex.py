"""Interface for Vertex API to yield vectors from Gemini Embedding 2."""

from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from math import ceil
from typing import TypeVar

import numpy as np
from google import genai
from google.genai import types
from numpy.typing import NDArray

from classification.encoders.base import Embedder, ImageInput, load_rgb

_Input = TypeVar("_Input")
_PREFIX = "task: classification | query: "


class VertexEmbeddingModel(Embedder):
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
        """Configure Vertex requests; each nonempty encode call owns its client.

        Args:
            project: Google Cloud project used for Vertex requests.
            location: Vertex endpoint location.
            max_concurrency: Maximum number of concurrent record requests.
            max_attempts: Total SDK attempts per request, including the first.
            request_timeout: Positive finite seconds allowed per attempt.
            classification_prompt: Prefix text and images with the classification prompt.
            truncate_dim: Output dimension from 128 through 3072, or None for 3072.

        Raises:
            ValueError: The project is empty or truncate_dim is out of range.
        """
        if not project:
            raise ValueError("vertex.project must be a nonempty string")
        if truncate_dim is not None and not 128 <= truncate_dim <= 3072:
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
        self,
        inputs: Iterable[_Input],
        prepare: Callable[[_Input], types.Content],
    ) -> NDArray[np.float32]:
        records = list(inputs)
        if not records:
            return np.empty((0, self.dimension), dtype=np.float32)
        with (
            genai.Client(
                vertexai=True,
                project=self.project,
                location=self.location,
                http_options=self.http_options,
            ) as client,
            ThreadPoolExecutor(max_workers=self.max_concurrency) as executor,
        ):
            # Prepare images inside workers to bound serialized image memory too.
            # A failed request cancels the pending ones when map's iterator closes.
            return np.stack(
                list(executor.map(lambda record: self._request(client, prepare(record)), records))
            )

    def encode_text(
        self,
        texts: Iterable[str],
    ) -> NDArray[np.float32]:
        """Encode class names or descriptions, preserving source order."""

        def prepare(text: str) -> types.Content:
            if self.classification_prompt:
                text = _PREFIX + text
            return types.Content(parts=[types.Part.from_text(text=text)])

        return self._encode(texts, prepare)

    def encode_images(
        self,
        images: Iterable[ImageInput],
    ) -> NDArray[np.float32]:
        """Send RGB PNG pixels without labels, filenames, or resizing."""

        def prepare(image: ImageInput) -> types.Content:
            with load_rgb(image) as rgb, BytesIO() as buffer:
                rgb.save(buffer, format="PNG")
                part = types.Part.from_bytes(data=buffer.getvalue(), mime_type="image/png")
            parts = [part]
            if self.classification_prompt:
                parts.insert(0, types.Part.from_text(text=_PREFIX))
            return types.Content(parts=parts)

        return self._encode(images, prepare)
