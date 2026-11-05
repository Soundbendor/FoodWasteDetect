"""Ordered metadata and embedding batches."""

import numpy as np

type EmbeddingBatch = tuple[dict[str, list], np.ndarray[tuple[int, int], np.dtype[np.float32]]]
