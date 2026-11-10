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
    """Convert metadata columns into ordered report rows, omitting images."""
    metadata = batch.copy()
    metadata.pop("image", None)
    return [
        cast(ImageMetadata, dict(zip(metadata, values, strict=True)))
        for values in zip(*metadata.values(), strict=True)
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
    query: Callable[[NDArray[np.float32], int], Sequence[Neighbor]],
    class_counts: Mapping[str, int],
    k: int,
    expected_count: int,
) -> list[Prediction]:
    """Query ordered batches and require a complete, nonempty evaluation."""
    if type(k) is not int or k <= 0:
        raise ValueError("k must be a positive integer")
    predictions: list[Prediction] = []
    with tqdm(total=expected_count, desc="Evaluation", unit="image", mininterval=5) as progress:
        for batch, vectors in batches:
            rows = extract_batch_metadata(batch)
            for record, vector in zip(rows, vectors, strict=True):
                relevant_count = class_counts.get(record["class"], 0)
                ranked = query(vector, max(k, 5, relevant_count))
                predictions.append(prediction(record, ranked, k, relevant_count))
            progress.update(len(rows))
    if not expected_count:
        raise ValueError("No evaluation images; accuracy is undefined")
    if expected_count != len(predictions):
        raise ValueError("Incomplete evaluation: validation and prediction counts differ")
    return predictions
