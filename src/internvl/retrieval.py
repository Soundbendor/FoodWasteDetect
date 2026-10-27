"""LanceDB boundary and backend-independent voting/metrics."""

from collections import Counter
from collections.abc import Iterable, Sequence
from math import log2
from pathlib import Path
from typing import Literal, TypedDict, cast

import lancedb
import numpy as np
import pyarrow as pa
from lancedb.db import LanceDBConnection
from numpy.typing import NDArray

from .schemas import Neighbor, PointRecord, Prediction, QueryRetrievalMetrics, SourceIdentity


class ClassMetrics(TypedDict):
    support: int
    correct: int
    recall: float | None
    precision: float
    f1: float


class RetrievalMetrics(TypedDict):
    evaluated: int
    majority_vote_accuracy: float | None
    nearest_neighbor_accuracy: float | None
    top5_hit_rate: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    precision_micro: float
    recall_micro: float
    f1_micro: float
    mean_top1_cosine_similarity: float | None
    mean_support_cosine_similarity: float | None
    retrieval_precision_at_k: float
    mrr_at_k: float
    ndcg_at_k: float
    map_at_r: float
    per_class: dict[str, ClassMetrics]


def classify(neighbors: Sequence[Neighbor]) -> tuple[str | None, float]:
    """Vote by count, then best similarity, then class name."""
    if not neighbors:
        return None, 0.0
    votes = Counter(n["label"] for n in neighbors)
    best = {label: max(n["score"] for n in neighbors if n["label"] == label) for label in votes}
    winner = min(votes, key=lambda label: (-votes[label], -best[label], label))
    return winner, votes[winner] / len(neighbors)


def ranking_metrics(
    neighbors: Sequence[Neighbor], label: str, k: int, relevant_count: int,
) -> QueryRetrievalMetrics:
    """Score the ranked rows before discarding neighbors beyond the voting support."""
    relevance = [n["label"] == label for n in neighbors] if relevant_count else []
    support = relevance[:k]
    ideal = sum(1 / log2(i + 1) for i in range(1, min(k, relevant_count) + 1))
    dcg = sum(hit / log2(i + 1) for i, hit in enumerate(support, 1))
    hits = 0
    ap = 0.0
    for rank, hit in enumerate(relevance[:relevant_count], 1):
        hits += hit
        if hit:
            ap += hits / rank
    return {
        "top5_hit": any(relevance[:5]),
        "precision_at_k": sum(support) / len(support) if support else 0.0,
        "reciprocal_rank_at_k": next((1 / i for i, hit in enumerate(support, 1) if hit), 0.0),
        "ndcg_at_k": dcg / ideal if ideal else 0.0,
        "average_precision_at_r": ap / relevant_count if relevant_count else 0.0,
        "relevant_reference_count": relevant_count,
    }


def metrics(predictions: Sequence[Prediction], classes: Iterable[str]) -> RetrievalMetrics:
    """Aggregate majority-vote classification and query-weighted retrieval scores."""
    evaluated = len(predictions)
    support = Counter(p["class"] for p in predictions)
    predicted = Counter(p["prediction"] for p in predictions if p["prediction"] is not None)
    hits = Counter(p["class"] for p in predictions if p["prediction"] == p["class"])
    correct = sum(hits.values())
    labels = support.keys() | predicted.keys()

    def ratio(numerator: float, denominator: float) -> float:
        return numerator / denominator if denominator else 0.0

    def scores(label: str) -> tuple[float, float, float]:
        return (ratio(hits[label], predicted[label]), ratio(hits[label], support[label]),
                ratio(2 * hits[label], predicted[label] + support[label]))

    per_class: dict[str, ClassMetrics] = {}
    for label in classes:
        precision, recall, f1 = scores(label)
        per_class[label] = dict(support=support[label], correct=hits[label],
                                recall=recall if support[label] else None,
                                precision=precision, f1=f1)
    macro = [ratio(sum(scores(label)[i] for label in labels), len(labels)) for i in range(3)]
    with_neighbors = [p["neighbors"] for p in predictions if p["neighbors"]]

    def ranking_mean(key: Literal[
        "top5_hit", "precision_at_k", "reciprocal_rank_at_k", "ndcg_at_k", "average_precision_at_r",
    ]) -> float:
        return ratio(sum(p["retrieval_metrics"][key] for p in predictions), evaluated)

    return {
        "evaluated": evaluated,
        "majority_vote_accuracy": correct / evaluated if evaluated else None,
        "nearest_neighbor_accuracy": (
            sum(p["nearest"] == p["class"] for p in predictions) / evaluated if evaluated else None
        ),
        "per_class": per_class,
        "precision_macro": macro[0], "recall_macro": macro[1], "f1_macro": macro[2],
        "precision_micro": ratio(correct, sum(predicted.values())),
        "recall_micro": ratio(correct, evaluated),
        "f1_micro": ratio(2 * correct, sum(predicted.values()) + evaluated),
        "top5_hit_rate": ranking_mean("top5_hit"),
        "retrieval_precision_at_k": ranking_mean("precision_at_k"),
        "mrr_at_k": ranking_mean("reciprocal_rank_at_k"),
        "ndcg_at_k": ranking_mean("ndcg_at_k"),
        "map_at_r": ranking_mean("average_precision_at_r"),
        "mean_top1_cosine_similarity": (
            sum(n[0]["score"] for n in with_neighbors) / len(with_neighbors)
            if with_neighbors else None
        ),
        "mean_support_cosine_similarity": (
            sum(sum(n["score"] for n in neighbors) / len(neighbors)
                for neighbors in with_neighbors) / len(with_neighbors) if with_neighbors else None
        ),
    }


class LanceDBIndex:
    def __init__(self, uri: str | Path, table: str) -> None:
        self.uri = Path(uri).expanduser().resolve()
        self.name = table
        self.table = None

    def exists(self) -> bool:
        """Check without creating a missing database directory."""
        return self.uri.is_dir() and self.name in cast(LanceDBConnection, lancedb.connect(self.uri))

    def open(self) -> None:
        """Open a completed table for reads without creating storage."""
        if not self.uri.is_dir():
            raise FileNotFoundError(f"LanceDB database does not exist: {self.uri}")
        self.table = cast(LanceDBConnection, lancedb.connect(self.uri)).open_table(self.name)

    def create(self, dimension: int) -> None:
        """Create a typed table; the backend refuses existing names."""
        schema = pa.schema(
            [
                pa.field(name, pa.string(), nullable=False)
                for name in ("id", "class", "source_kind", "source_identity", "value")
            ]
            + [pa.field("vector", pa.list_(pa.float32(), dimension), nullable=False)]
        )
        self.table = cast(LanceDBConnection, lancedb.connect(self.uri)).create_table(
            self.name, schema=schema, mode="create", exist_ok=False
        )

    def append(
        self,
        records: Sequence[PointRecord],
        vectors: np.ndarray[tuple[int, int], np.dtype[np.float32 | np.float64]],
    ) -> None:
        """Append paired records; malformed vectors fail backend validation."""
        if len(records) != len(vectors):
            raise ValueError("Record/vector count mismatch")
        if self.table is None:
            raise RuntimeError("Open or create the table before insertion")
        self.table.add(
            [
                dict(r.to_payload(), vector=v.tolist())
                for r, v in zip(records, vectors, strict=True)
            ],
            on_bad_vectors="error",
        )

    def class_counts(self) -> Counter[str]:
        """Stream all class metadata, counting every stored row without loading vectors."""
        if self.table is None:
            raise RuntimeError("Open or create the table before scanning")
        counts: Counter[str] = Counter()
        for batch in self.table.search().select(["class"]).limit(None).to_batches():
            counts.update(batch.column(0).to_pylist())
        return counts

    def identities(self) -> list[SourceIdentity]:
        """Scan all identity metadata without reading vectors."""
        if self.table is None:
            raise RuntimeError("Open or create the table before scanning")
        rows = self.table.search().select(
            ["source_kind", "source_identity"]
        ).limit(None).to_list()
        return [(row["source_kind"], row["source_identity"]) for row in rows]

    def fetch_vectors(self, identities: Sequence[SourceIdentity]) -> NDArray[np.float32]:
        """Fetch a bounded selection and restore the caller's order."""
        if self.table is None:
            raise RuntimeError("Open or create the table before fetching")
        if not identities:
            return np.empty((0, self.table.schema.field("vector").type.list_size), dtype=np.float32)

        def literal(value: str) -> str:
            return "'" + value.replace("'", "''") + "'"

        predicate = " OR ".join(
            f"(source_kind = {literal(kind)} AND source_identity = {literal(identity)})"
            for kind, identity in identities
        )
        rows = self.table.search().where(predicate).select(
            ["source_kind", "source_identity", "vector"]
        ).limit(None).to_list()
        vectors = {(row["source_kind"], row["source_identity"]): row["vector"] for row in rows}
        return np.asarray([vectors[key] for key in identities], dtype=np.float32)

    def query(
        self, vector: np.ndarray[tuple[int], np.dtype[np.float32 | np.float64]], k: int
    ) -> list[Neighbor]:
        """Retrieve exact cosine neighbors, converting distance to similarity."""
        if self.table is None:
            raise RuntimeError("Open or create the table before querying")
        points = (
            self.table.search(vector.tolist())
            .distance_type("cosine")
            .bypass_vector_index()
            .limit(k)
            .to_list()
        )
        return [
            Neighbor(id=p["id"], label=p["class"], score=1 - float(p["_distance"])) for p in points
        ]
