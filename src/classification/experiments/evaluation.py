"""Evaluate ordered embedding batches before experiment report publication."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import cast

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

from classification.retrieval.metrics import ranking_metrics
from classification.retrieval.voting import classify
from classification.schemas import EmbeddingBatch, ImageMetadata, Neighbor, Prediction


def extract_batch_metadata(batch: dict[str, list]) -> list[ImageMetadata]:
    """Convert metadata-only columns into ordered report rows."""
    return [
        cast(ImageMetadata, dict(zip(batch, values, strict=True)))
        for values in zip(*batch.values(), strict=True)
    ]


def prediction(
    record: ImageMetadata,
    ranked: Sequence[Neighbor],
    k: int,
    relevant_count: int,
) -> Prediction:
    """Combine image metadata, the vote result, and retrieved neighbors."""
    neighbors = ranked[:k]
    label, fraction = classify(neighbors)
    return {
        **record,
        "prediction": label,
        "vote_fraction": fraction,
        "nearest": neighbors[0]["label"] if neighbors else None,
        "neighbors": list(neighbors),
        "retrieval_metrics": ranking_metrics(ranked, record["class"], k, relevant_count),
    }


def evaluate_batches(
    batches: Iterable[EmbeddingBatch],
    *,
    query: Callable[[NDArray[np.float32], Sequence[int]], Sequence[Sequence[Neighbor]]],
    class_counts: Mapping[str, int],
    k: int,
    expected_count: int,
) -> list[Prediction]:
    """Query each ordered batch once and require a complete, nonempty evaluation."""
    predictions: list[Prediction] = []
    with tqdm(total=expected_count, desc="Evaluation", unit="image", mininterval=5) as progress:
        for batch, vectors in batches:
            rows = extract_batch_metadata(batch)
            if len(rows) != len(vectors):
                raise ValueError("Batch metadata and vectors must pair")
            counts = [class_counts.get(r["class"], 0) for r in rows]
            ranked = query(vectors, [max(k, 5, c) for c in counts])
            for record, neighbors, count in zip(rows, ranked, counts, strict=True):
                predictions.append(prediction(record, neighbors, k, count))
            progress.update(len(rows))
    if not expected_count:
        raise ValueError("No evaluation images; accuracy is undefined")
    if expected_count != len(predictions):
        raise ValueError("Incomplete evaluation: validation and prediction counts differ")
    return predictions
