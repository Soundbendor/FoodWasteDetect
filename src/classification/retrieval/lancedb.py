"""LanceDB storage and exact cosine retrieval."""

import functools
import json
import operator
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Self

import lance
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from lancedb.db import LanceDBConnection
from lancedb.table import LanceTable
from numpy.typing import NDArray

from classification.schemas import Neighbor, PointRecord, SourceIdentity


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
        payload_fields = ("id", "class", "source_kind", "source_identity", "value")
        schema = pa.schema(
            [pa.field(name, pa.string(), nullable=False) for name in payload_fields]
            + [pa.field("vector", pa.list_(pa.float32(), dimension), nullable=False)],
            metadata=dict(metadata or {}),
        )
        path = Path(uri).expanduser().resolve()
        opened = LanceDBConnection(path).create_table(
            table, schema=schema, mode="create", exist_ok=False
        )
        return cls(path, opened)

    @property
    def dimension(self) -> int:
        """Read the embedding vector width from table schema."""
        return self.table.schema.field("vector").type.list_size

    def create_prefix_table(
        self,
        destination: str,
        dimension: int,
    ) -> Self:
        """Reuse readable destinations; otherwise shallow-clone and L2-normalize prefixes."""
        if self.exists(self.uri, destination):
            return self.open(self.uri, destination)
        source_dim = self.dimension
        encoded = (self.table.schema.metadata or {}).get(b"classification.build")
        snapshot = None
        if encoded is not None:
            snapshot = json.loads(encoded) | {"vector_dimension": dimension}

        vector_type = pa.list_(pa.float32(), dimension)
        prefix_schema = pa.schema([pa.field("vector_prefix", vector_type, nullable=False)])

        @lance.batch_udf(output_schema=prefix_schema)
        def _truncate_and_normalize(batch: pa.RecordBatch) -> pa.RecordBatch:
            column = batch.column("vector")
            vectors = column.values.to_numpy().reshape(len(column), source_dim)[:, :dimension]
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            normalized = np.divide(
                vectors,
                norms,
                out=np.zeros_like(vectors, dtype=np.float32),
                where=norms > 0,
            )
            array = pa.FixedSizeListArray.from_arrays(
                pa.array(normalized.reshape(-1), type=pa.float32()),
                type=vector_type,
            )
            return pa.RecordBatch.from_arrays([array], schema=prefix_schema)

        connection = LanceDBConnection(self.uri)
        cloned = connection.clone_table(destination, str(self.table.uri))
        dataset = cloned.to_lance()
        dataset.add_columns(_truncate_and_normalize, read_columns=["vector"])
        dataset.drop_columns(["vector"])
        dataset.alter_columns({"path": "vector_prefix", "name": "vector"})  # ty: ignore[invalid-argument-type]
        if snapshot is not None:
            # Match Pydantic's compact JSON so the snapshot stays byte-comparable.
            encoded = json.dumps(snapshot, separators=(",", ":"), ensure_ascii=False)
            dataset.update_schema_metadata({"classification.build": encoded})
        return type(self)(self.uri, connection.open_table(destination))

    def append(
        self,
        records: Sequence[PointRecord],
        vectors: np.ndarray[tuple[int, int], np.dtype[np.float32 | np.float64]],
    ) -> None:
        """Append paired records; malformed vectors fail backend validation."""
        self.table.add(
            [
                dict(r.to_payload(), vector=v.tolist())
                for r, v in zip(records, vectors, strict=True)
            ],
            on_bad_vectors="error",
        )

    def optimize(self) -> None:
        """Compact sharded LanceDB table files into larger chunks for optimized queries."""
        self.table.optimize()

    def class_counts(self) -> Counter[str]:
        """Return count of items in table by class."""
        counts: Counter[str] = Counter()
        for batch in self.table.search().select(["class"]).limit(None).to_batches():
            counts.update(batch.column(0).to_pylist())
        return counts

    def get_record_ids(self) -> list[SourceIdentity]:
        """Get a list of files/source IDs currently stored in database."""
        table = self.table.to_lance().to_table(columns=["source_kind", "source_identity"])
        return list(zip(*(table[name].to_pylist() for name in table.column_names), strict=True))

    def fetch_vectors(self, identities: Sequence[SourceIdentity]) -> NDArray[np.float32]:
        """Fetch vectors in the requested identity order, including repeats.

        Empty input returns shape (0, table dimension). A missing identity raises
        KeyError; storage errors propagate.
        """
        if not identities:
            return np.empty((0, self.dimension), dtype=np.float32)

        # One isin per kind; a per-identity OR chain plans in exponential memory.
        by_kind: defaultdict[str, list[str]] = defaultdict(list)
        for kind, identity in identities:
            by_kind[kind].append(identity)
        predicate = functools.reduce(
            operator.or_,
            (
                (pc.field("source_kind") == kind) & pc.field("source_identity").isin(ids)
                for kind, ids in by_kind.items()
            ),
        )
        table = self.table.to_lance().to_table(
            columns=["source_kind", "source_identity", "vector"], filter=predicate
        )
        keys = zip(
            table["source_kind"].to_pylist(), table["source_identity"].to_pylist(), strict=True
        )
        rows = {key: row for row, key in enumerate(keys)}
        # Read vectors from the Arrow buffer instead of building Python floats.
        vectors = table["vector"].combine_chunks().flatten().to_numpy()
        return vectors.reshape(-1, self.dimension)[[rows[key] for key in identities]]

    def query(
        self, vector: np.ndarray[tuple[int], np.dtype[np.float32 | np.float64]], k: int
    ) -> list[Neighbor]:
        """Retrieve exact cosine neighbors, converting distance to similarity."""
        points = (
            self.table.search(vector.tolist())
            .distance_type("cosine")
            .bypass_vector_index()
            .select(["id", "class"])
            .limit(k)
            .to_list()
        )
        return [
            Neighbor(id=p["id"], label=p["class"], score=1 - float(p["_distance"])) for p in points
        ]
