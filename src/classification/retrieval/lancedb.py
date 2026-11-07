"""LanceDB storage and exact cosine retrieval."""

from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Self

import numpy as np
import pyarrow as pa
from lancedb.db import LanceDBConnection
from lancedb.table import LanceTable
from numpy.typing import NDArray

from classification.experiments.config import ExperimentConfig
from classification.schemas import Neighbor, PointRecord, SourceIdentity

# Schema metadata key holding the ExperimentConfig JSON that created a table.
BUILD_SNAPSHOT_KEY = "classification.build"


class LanceDBIndex:
    """Store and query references through an open local LanceDB table.

    Use open() for existing tables or create() for a new fixed-dimension schema.
    """

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
    def create(
        cls,
        uri: str | Path,
        table: str,
        dimension: int,
        *,
        metadata: Mapping[str, str] | None = None,
    ) -> Self:
        """Create a typed table with schema metadata; refuse existing names."""
        schema = pa.schema(
            [
                pa.field(name, pa.string(), nullable=False)
                for name in ("id", "class", "source_kind", "source_identity", "value")
            ]
            + [pa.field("vector", pa.list_(pa.float32(), dimension), nullable=False)],
            metadata=metadata,
        )
        path = Path(uri).expanduser().resolve()
        opened = LanceDBConnection(path).create_table(
            table, schema=schema, mode="create", exist_ok=False
        )
        return cls(path, opened)

    @property
    def dimension(self) -> int:
        """Return the stored vector width.

        Raises:
            ValueError: If the schema lacks a positive fixed-size ``vector`` field.
        """
        schema = self.table.schema
        position = schema.get_field_index("vector")
        if position < 0:
            raise ValueError(f"Table {self.name!r} must have a fixed-size vector field")
        vector_type = schema.field(position).type
        if not pa.types.is_fixed_size_list(vector_type) or vector_type.list_size <= 0:
            raise ValueError(f"Table {self.name!r} must have a positive fixed-size vector field")
        return vector_type.list_size

    def derive_prefix(self, destination: str, dimension: int, *, batch_size: int) -> Self:
        """Open or create a table holding this table's rows at a shorter vector width.

        An existing destination is reused without reading this table. Otherwise one
        streaming creation copies every row, record ID, and metadata value, replacing
        each vector with its exact float32 prefix. The creation snapshot records the
        derived ``vector_dimension``; other schema metadata is kept verbatim.

        Args:
            destination: Name of the derived table in the same database.
            dimension: Prefix width, at most the stored width.
            batch_size: Rows per streamed batch.

        Returns:
            An index over the derived table.

        Raises:
            ValueError: If ``dimension`` exceeds the stored width.
            pydantic.ValidationError: If the stored creation snapshot is invalid.
        """
        connection = LanceDBConnection(self.uri)
        if destination in connection:
            return type(self)(self.uri, connection.open_table(destination))
        stored_width = self.dimension
        if dimension > stored_width:
            raise ValueError(
                f"Prefix width {dimension} exceeds the stored width {stored_width} "
                f"of table {self.name!r}"
            )
        schema = self._prefix_schema(dimension)
        reader = pa.RecordBatchReader.from_batches(
            schema, _prefix_batches(self.table, schema, dimension, batch_size)
        )
        created = connection.create_table(
            destination, data=reader, schema=schema, mode="create", exist_ok=False
        )
        return type(self)(self.uri, created)

    def _prefix_schema(self, dimension: int) -> pa.Schema:
        """Narrow the vector type and record the derived width in the snapshot."""
        schema = self.table.schema
        metadata = dict(schema.metadata or {})
        key = BUILD_SNAPSHOT_KEY.encode("utf-8")
        encoded = metadata.get(key)
        if encoded is not None:
            snapshot = ExperimentConfig.model_validate_json(encoded)
            snapshot.vector_dimension = dimension
            metadata[key] = snapshot.model_dump_json().encode("utf-8")
        position = schema.get_field_index("vector")
        vector = schema.field(position).with_type(pa.list_(pa.float32(), dimension))
        return schema.set(position, vector).with_metadata(metadata)

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
        return self._column_counts("class")

    def source_counts(self) -> Counter[str]:
        """Count every stored row, including duplicate source keys, by source kind.

        Streams only the ``source_kind`` column; vectors are never read.
        """
        return self._column_counts("source_kind")

    def _column_counts(self, column: str) -> Counter[str]:
        counts: Counter[str] = Counter()
        for batch in self.table.search().select([column]).limit(None).to_batches():
            counts.update(batch.column(0).to_pylist())
        return counts

    def get_record_ids(self) -> list[SourceIdentity]:
        """Scan all identity metadata without reading vectors."""
        rows = self.table.search().select(["source_kind", "source_identity"]).limit(None).to_list()
        return [(row["source_kind"], row["source_identity"]) for row in rows]

    def fetch_vectors(self, identities: Sequence[SourceIdentity]) -> NDArray[np.float32]:
        """Fetch vectors in the requested identity order, including repeats.

        Empty input returns shape (0, table dimension). A missing identity raises
        KeyError; storage errors propagate.
        """
        if not identities:
            return np.empty((0, self.dimension), dtype=np.float32)

        def literal(value: str) -> str:
            return "'" + value.replace("'", "''") + "'"

        predicate = " OR ".join(
            f"(source_kind = {literal(kind)} AND source_identity = {literal(identity)})"
            for kind, identity in identities
        )
        rows = (
            self.table.search()
            .where(predicate)
            .select(["source_kind", "source_identity", "vector"])
            .limit(None)
            .to_list()
        )
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


def _prefix_batches(
    source: LanceTable, schema: pa.Schema, dimension: int, batch_size: int
) -> Iterator[pa.RecordBatch]:
    """Stream every source row, replacing only the vector column with its prefix."""
    position = schema.get_field_index("vector")
    vector_type = schema.field(position).type
    for batch in source.search().limit(None).to_batches(batch_size=batch_size):
        vectors = np.asarray(batch.column(position).to_pylist(), dtype=np.float32)
        arrays = list(batch.columns)
        arrays[position] = pa.array(vectors[:, :dimension].tolist(), type=vector_type)
        yield pa.RecordBatch.from_arrays(arrays, schema=schema)
