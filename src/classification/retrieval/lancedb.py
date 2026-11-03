"""LanceDB storage and exact cosine retrieval."""

from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Self

import numpy as np
import pyarrow as pa
from lancedb.db import LanceDBConnection
from lancedb.table import LanceTable
from numpy.typing import NDArray

from classification.schemas import Neighbor, PointRecord, SourceIdentity


class LanceDBIndex:
    def __init__(self, uri: str | Path, table: LanceTable) -> None:
        self.uri = Path(uri).expanduser().resolve()
        self.name = table.name
        self.table: LanceTable = table

    @staticmethod
    def exists(uri: str | Path, table: str) -> bool:
        """Check without creating a missing database directory."""
        path = Path(uri).expanduser().resolve()
        return path.is_dir() and table in LanceDBConnection(path)

    @classmethod
    def open(cls, uri: str | Path, table: str) -> Self:
        """Open a completed table for reads without creating storage."""
        path = Path(uri).expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"LanceDB database does not exist: {path}")
        return cls(path, LanceDBConnection(path).open_table(table))

    @classmethod
    def create(cls, uri: str | Path, table: str, dimension: int) -> Self:
        """Create a typed table; the backend refuses existing names."""
        schema = pa.schema(
            [
                pa.field(name, pa.string(), nullable=False)
                for name in ("id", "class", "source_kind", "source_identity", "value")
            ]
            + [pa.field("vector", pa.list_(pa.float32(), dimension), nullable=False)]
        )
        path = Path(uri).expanduser().resolve()
        opened = LanceDBConnection(path).create_table(
            table, schema=schema, mode="create", exist_ok=False
        )
        return cls(path, opened)

    def append(
        self,
        records: Sequence[PointRecord],
        vectors: np.ndarray[tuple[int, int], np.dtype[np.float32 | np.float64]],
    ) -> None:
        """Append paired records; malformed vectors fail backend validation."""
        if len(records) != len(vectors):
            raise ValueError("Record/vector count mismatch")
        self.table.add(
            [
                dict(r.to_payload(), vector=v.tolist())
                for r, v in zip(records, vectors, strict=True)
            ],
            on_bad_vectors="error",
        )

    def optimize(self) -> None:
        """Compact with LanceDB's default maintenance and version-retention policy."""
        self.table.optimize()

    def class_counts(self) -> Counter[str]:
        """Stream all class metadata, counting every stored row without loading vectors."""
        counts: Counter[str] = Counter()
        for batch in self.table.search().select(["class"]).limit(None).to_batches():
            counts.update(batch.column(0).to_pylist())
        return counts

    def get_record_ids(self) -> list[SourceIdentity]:
        """Scan all identity metadata without reading vectors."""
        rows = self.table.search().select(
            ["source_kind", "source_identity"]
        ).limit(None).to_list()
        return [(row["source_kind"], row["source_identity"]) for row in rows]

    def fetch_vectors(self, identities: Sequence[SourceIdentity]) -> NDArray[np.float32]:
        """Fetch a bounded selection and restore the caller's order."""
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
