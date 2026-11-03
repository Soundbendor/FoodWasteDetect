"""Ordered metadata and embedding batches."""

from collections.abc import Iterable, Iterator

import numpy as np

from classification.retrieval.lancedb import LanceDBIndex

type EmbeddingBatch = tuple[dict[str, list], np.ndarray[tuple[int, int], np.dtype[np.float32]]]


def cached_batches(
    cache: LanceDBIndex,
    metadata_batches: Iterable[dict[str, list]],
) -> Iterator[EmbeddingBatch]:
    """Pair metadata with cached vectors in source identity order."""
    for batch in metadata_batches:
        yield (
            batch,
            cache.fetch_vectors(
                [("validation_image", identity) for identity in batch["source_id"]]
            ),
        )


