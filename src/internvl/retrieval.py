"""Qdrant boundary and backend-independent voting/metrics."""
from collections import Counter
from dataclasses import dataclass
from typing import Sequence

from .records import ReferenceRecord


@dataclass(frozen=True)
class Neighbor:
    id: str
    label: str
    score: float


def classify(neighbors):
    if not neighbors:
        return None, 0.0
    votes = Counter(n.label for n in neighbors)
    best = {label: max(n.score for n in neighbors if n.label == label) for label in votes}
    winner = min(votes, key=lambda label: (-votes[label], -best[label], label))
    return winner, votes[winner] / len(neighbors)


def metrics(predictions, selected, available, embedded, classes):
    evaluated = len(predictions)
    correct = sum(p['prediction'] == p['class'] for p in predictions)
    nearest = sum(p['nearest'] == p['class'] for p in predictions)
    per_class = {}
    for label in classes:
        rows = [p for p in predictions if p['class'] == label]
        hits = sum(p['prediction'] == label for p in rows)
        per_class[label] = dict(support=len(rows), correct=hits,
                                recall=hits / len(rows) if rows else None)
    return dict(selected=selected, available=available, successfully_embedded=embedded,
                evaluated=evaluated, coverage=evaluated / selected if selected else 0.0,
                majority_vote_accuracy=correct / evaluated if evaluated else None,
                nearest_neighbor_accuracy=nearest / evaluated if evaluated else None,
                per_class=per_class)


class QdrantIndex:
    def __init__(self, client, collection, dimension):
        self.client, self.collection, self.dimension = client, collection, dimension

    def create(self):
        from qdrant_client.models import Distance, VectorParams
        if self.client.collection_exists(self.collection):
            raise ValueError('Refusing to overwrite an existing collection')
        self.client.create_collection(self.collection, vectors_config=VectorParams(
            size=self.dimension, distance=Distance.COSINE))

    def upsert(self, records: Sequence[ReferenceRecord], vectors):
        from qdrant_client.models import PointStruct
        if len(records) != len(vectors):
            raise ValueError('Record/vector count mismatch')
        self.client.upsert(self.collection, points=[
            PointStruct(id=r.id, vector=v.tolist(), payload=r.to_payload())
            for r, v in zip(records, vectors)], wait=True)

    def query(self, vector, k):
        from qdrant_client.models import SearchParams
        points = self.client.query_points(self.collection, query=vector.tolist(), limit=k,
                                          search_params=SearchParams(exact=True), with_payload=True).points
        return [Neighbor(str(p.id), p.payload['class'], float(p.score)) for p in points]
