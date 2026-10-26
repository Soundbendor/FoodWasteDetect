"""LanceDB boundary and backend-independent voting/metrics."""

from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import TypedDict, cast

import lancedb
import numpy as np
import pyarrow as pa
from lancedb.db import LanceDBConnection
from numpy.typing import NDArray

from .schemas import Neighbor, PointRecord, Prediction, SourceIdentity


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
    best = {label: max(n["score"] for n in neighbors if n["label"] == label) for label in votes}
    winner = min(votes, key=lambda label: (-votes[label], -best[label], label))
    return winner, votes[winner] / len(neighbors)


def metrics(predictions: Sequence[Prediction], classes: Iterable[str]) -> RetrievalMetrics:
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
