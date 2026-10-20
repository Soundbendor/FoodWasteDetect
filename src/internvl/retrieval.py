"""Qdrant boundary and backend-independent voting/metrics."""

from collections import Counter
from collections.abc import Iterable, Sequence
from typing import TypedDict, cast

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, SearchParams, VectorParams

from .schemas import Neighbor, PointRecord, Prediction


class ClassMetrics(TypedDict):
    support: int
    correct: int
    recall: float | None


class RetrievalMetrics(TypedDict):
    evaluated: int
    majority_vote_accuracy: float | None
    nearest_neighbor_accuracy: float | None
    per_class: dict[str, ClassMetrics]


def classify(neighbors: Sequence[Neighbor]) -> tuple[str | None, float]:
    """Vote by count, then best similarity, then class name."""
    if not neighbors:
        return None, 0.0
    votes = Counter(n["label"] for n in neighbors)
    best = {
        label: max(n["score"] for n in neighbors if n["label"] == label)
        for label in votes
    }
    winner = min(votes, key=lambda label: (-votes[label], -best[label], label))
    return winner, votes[winner] / len(neighbors)


def metrics(
    predictions: Sequence[Prediction], classes: Iterable[str]
) -> RetrievalMetrics:
    """Calculate accuracy and per-class recall for completed predictions."""
    evaluated = len(predictions)
    support: Counter[str] = Counter()
    hits: Counter[str] = Counter()
    correct = nearest = 0
    for prediction in predictions:
        label = prediction["class"]
        support[label] += 1
        if prediction["prediction"] == label:
            hits[label] += 1
            correct += 1
        nearest += prediction["nearest"] == label
    per_class: dict[str, ClassMetrics] = {}
    for label in classes:
        count = support[label]
        per_class[label] = {
            "support": count,
            "correct": hits[label],
            "recall": hits[label] / count if count else None,
        }
    return {
        "evaluated": evaluated,
        "majority_vote_accuracy": correct / evaluated if evaluated else None,
        "nearest_neighbor_accuracy": nearest / evaluated if evaluated else None,
        "per_class": per_class,
    }


class QdrantIndex:
    def __init__(self, client: QdrantClient, collection: str, dimension: int) -> None:
        """Store the client and collection settings."""
        self.client, self.collection, self.dimension = client, collection, dimension

    def create(self) -> None:
        """Create a new cosine collection without overwriting one."""
        if self.client.collection_exists(self.collection):
            raise ValueError("Refusing to overwrite an existing collection")
        self.client.create_collection(
            self.collection,
            vectors_config=VectorParams(size=self.dimension, distance=Distance.COSINE),
        )

    def upsert(
        self,
        records: Sequence[PointRecord],
        vectors: np.ndarray[tuple[int, int], np.dtype[np.float32 | np.float64]],
    ) -> None:
        """Insert paired reference records and vectors."""
        if len(records) != len(vectors):
            raise ValueError("Record/vector count mismatch")
        self.client.upsert(
            self.collection,
            points=[
                PointStruct(id=r.id, vector=v.tolist(), payload=r.to_payload())
                for r, v in zip(records, vectors, strict=True)
            ],
            wait=True,
        )

    def query(
        self, vector: np.ndarray[tuple[int], np.dtype[np.float32 | np.float64]], k: int
    ) -> list[Neighbor]:
        """Retrieve exact cosine neighbors with their labels."""
        points = self.client.query_points(
            self.collection,
            query=vector.tolist(),
            limit=k,
            search_params=SearchParams(exact=True),
            with_payload=True,
        ).points
        return [
            Neighbor(
                id=str(p.id),
                label=cast(dict[str, str], p.payload)["class"],
                score=float(p.score),
            )
            for p in points
        ]
