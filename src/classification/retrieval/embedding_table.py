"""Embedding tables: headed LanceDB tables of records and vectors, with exact cosine search.

This module is the only place that knows storage column names, schema metadata keys, the
record-to-row mapping, or LanceDB and Arrow calls. Callers work in domain terms: roles,
source keys, records, and vectors.
"""

from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Self

import numpy as np
import pyarrow as pa
from lancedb.db import LanceDBConnection
from lancedb.table import LanceTable
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, PositiveInt, ValidationError

from classification.schemas import Neighbor, PointRecord, SourceKey

type TableRole = Literal["reference", "validation_cache"]
type FingerprintValue = str | int | bool | None
# Vector-affecting settings by name, such as configuration paths; compared generically.
type Fingerprint = Mapping[str, FingerprintValue]
# Each differing setting mapped to its stored and required values.
type Differences = dict[str, tuple[FingerprintValue, FingerprintValue]]

# Version of the store-owned header format; tables recording another version are legacy.
SCHEMA_VERSION = 2
_HEADER_KEY = "classification.embedding_table"
_SNAPSHOT_KEY = "classification.creation_snapshot"
_ADR = "docs/adr/0001-embedding-table-header.md"
# LanceDB version tag marking the version at which a table was verified complete.
_COMPLETE_TAG = "complete"
# How a researcher finishes an incomplete table of each role.
_RESUME: dict[TableRole, str] = {
    "reference": "rerun the index stage to resume and mark it",
    "validation_cache": "evaluate with validation caching enabled to fill and mark it",
}

_STRING_COLUMNS = ("id", "class", "source_kind", "source_identity", "value")
_VECTOR = "vector"


class EmbeddingTableError(ValueError):
    """An embedding table cannot be used as requested."""


class LegacyTableError(EmbeddingTableError):
    """A table lacks a valid header for the current schema version and must be rebuilt."""


class IncompatibleTableError(EmbeddingTableError):
    """A table's fingerprints or dimension differ from the values a caller requires.

    Attributes:
        differences: Each differing setting mapped to its stored and required values.
    """

    def __init__(self, message: str, *, differences: Differences) -> None:
        """Initialize the error.

        Args:
            message: Description listing every difference and the fix.
            differences: Each differing setting mapped to its stored and required
                values.
        """
        super().__init__(message)
        self.differences = differences


class IncompleteTableError(EmbeddingTableError):
    """A table's build has not finished, so it may lack expected source keys.

    Attributes:
        missing: How many expected source keys were absent when the error was raised.
    """

    def __init__(self, message: str, *, missing: int) -> None:
        super().__init__(message)
        self.missing = missing


@dataclass(frozen=True)
class PrefixSource:
    """The table a prefix table was derived from, and that table's dimension."""

    table: str
    dimension: int


class _Header(BaseModel):
    """Store-owned table header, written once at creation and never rewritten."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    schema_version: int
    role: TableRole
    dimension: PositiveInt
    encoder: dict[str, FingerprintValue]
    inputs: dict[str, FingerprintValue] | None
    source_table: str | None = None
    source_dimension: PositiveInt | None = None

    @property
    def prefix_source(self) -> PrefixSource | None:
        if self.source_table is None or self.source_dimension is None:
            return None
        return PrefixSource(self.source_table, self.source_dimension)


class EmbeddingTable:
    """A reference table or validation cache stored in one local LanceDB table.

    Every table carries a header recording its schema version, role, dimension,
    encoder fingerprint, inputs fingerprint (reference tables only), and, for prefix
    tables, the source table and source dimension. The experiment's creation snapshot
    is stored verbatim beside it. Opening a table without a valid header raises
    ``LegacyTableError``; opening it under another role raises
    ``EmbeddingTableError``; resuming it, or opening it for evaluation, with
    different fingerprints or dimension raises ``IncompatibleTableError``.

    Attributes:
        uri: Resolved database directory.
        name: Table name.
    """

    def __init__(self, uri: Path, table: LanceTable, header: _Header) -> None:
        self.uri = uri
        self.name = table.name
        self._table = table
        self._header = header

    @staticmethod
    def exists(uri: str | Path, name: str) -> bool:
        """Check for a table without creating a missing database directory.

        Args:
            uri: Local database directory.
            name: Table name.

        Returns:
            True when the directory and the named table both exist.
        """
        path = _resolve(uri)
        return path.is_dir() and name in LanceDBConnection(path)

    @classmethod
    def open(cls, uri: str | Path, name: str, role: TableRole) -> Self:
        """Open an existing table under the given role, without creating storage.

        Args:
            uri: Local database directory.
            name: Table name.
            role: The role the caller requires.

        Returns:
            The opened table.

        Raises:
            FileNotFoundError: If the database directory does not exist.
            LegacyTableError: If the table has no valid current header.
            EmbeddingTableError: If the table's role differs from ``role``.
        """
        path = _resolve(uri)
        if not path.is_dir():
            raise FileNotFoundError(f"LanceDB database does not exist: {path}")
        table = LanceDBConnection(path).open_table(name)
        header = _read_header(table)
        if header.role != role:
            raise EmbeddingTableError(
                f"Table {name!r} is a {header.role} table, not a {role} table"
            )
        return cls(path, table, header)

    @classmethod
    def open_complete(
        cls,
        uri: str | Path,
        name: str,
        role: TableRole,
        *,
        expected: Callable[[], Iterable[SourceKey]],
        encoder: Fingerprint | None = None,
        dimension: int | None = None,
    ) -> Self:
        """Open an existing complete table for reading under the given role.

        A complete table opens without computing the expected keys or scanning
        stored ones. Compatibility is checked before completion.

        Args:
            uri: Local database directory.
            name: Table name.
            role: The role the caller requires.
            expected: Returns the source keys the finished table must contain; called
                only for an incomplete table, to state how many are missing.
            encoder: Encoder fingerprint the table must record; None skips the check.
            dimension: Vector width the table must record; None skips the check.

        Returns:
            The opened complete table.

        Raises:
            FileNotFoundError: If the database directory does not exist.
            LegacyTableError: If the table has no valid current header.
            EmbeddingTableError: If the table's role differs from ``role``.
            IncompatibleTableError: If ``encoder`` or ``dimension`` differs from the
                stored value, listing each difference.
            IncompleteTableError: If the table is not marked complete at its current
                version, stating how many expected source keys are missing.
        """
        table = cls.open(uri, name, role)
        table.require_compatible(encoder=encoder, dimension=dimension)
        if not table.is_complete:
            raise table._incomplete(expected())
        return table

    @classmethod
    def create(
        cls,
        uri: str | Path,
        name: str,
        role: TableRole,
        dimension: int,
        *,
        snapshot: str,
        encoder: Fingerprint,
        inputs: Fingerprint | None = None,
    ) -> Self:
        """Create an empty table with a header and creation snapshot; refuse existing names.

        Args:
            uri: Local database directory, created when absent.
            name: Table name.
            role: Reference table or validation cache.
            dimension: Fixed vector width.
            snapshot: Creation snapshot, stored verbatim and never rewritten.
            encoder: Encoder fingerprint recorded in the header.
            inputs: Inputs fingerprint; required for reference tables and None for
                validation caches.

        Returns:
            The new, empty table.

        Raises:
            EmbeddingTableError: If ``inputs`` does not match the role; nothing is
                written.
        """
        if not _inputs_match_role(role, inputs):
            requirement = "requires an" if role == "reference" else "has no"
            raise EmbeddingTableError(
                f"Table {name!r}: a {role} table {requirement} inputs fingerprint"
            )
        header = _Header(
            schema_version=SCHEMA_VERSION,
            role=role,
            dimension=dimension,
            encoder=dict(encoder),
            inputs=None if inputs is None else dict(inputs),
        )
        return cls._create(_resolve(uri), name, header, snapshot)

    @classmethod
    def _create(
        cls,
        path: Path,
        name: str,
        header: _Header,
        snapshot: str,
        rows: pa.RecordBatchReader | None = None,
    ) -> Self:
        schema = _schema(header, snapshot)
        table = LanceDBConnection(path).create_table(
            name, data=rows, schema=schema, mode="create", exist_ok=False
        )
        return cls(path, table, header)

    @classmethod
    def open_or_create(
        cls,
        uri: str | Path,
        name: str,
        role: TableRole,
        dimension: Callable[[], int],
        *,
        snapshot: str,
        encoder: Fingerprint,
        inputs: Fingerprint | None = None,
    ) -> Self:
        """Open an existing table under ``role`` to resume it, or create one when absent.

        Resuming requires the stored fingerprints to equal ``encoder`` and ``inputs``,
        so one table never mixes vectors from two configurations.

        Args:
            uri: Local database directory.
            name: Table name.
            role: Reference table or validation cache.
            dimension: Returns the vector width; called only when creating, so an
                existing table never constructs an encoder.
            snapshot: Creation snapshot; ignored for an existing table.
            encoder: Encoder fingerprint of the current configuration.
            inputs: Inputs fingerprint of the current configuration; required for
                reference tables and None for validation caches.

        Returns:
            The opened or created table.

        Raises:
            IncompatibleTableError: If an existing table records different
                fingerprints, listing each differing setting.
        """
        if cls.exists(uri, name):
            table = cls.open(uri, name, role)
            table.require_compatible(encoder=encoder, inputs=inputs)
            return table
        return cls.create(
            uri, name, role, dimension(), snapshot=snapshot, encoder=encoder, inputs=inputs
        )

    @property
    def role(self) -> TableRole:
        """Return the role recorded in the header."""
        return self._header.role

    @property
    def dimension(self) -> int:
        """Return the vector width recorded in the header."""
        return self._header.dimension

    @property
    def encoder_fingerprint(self) -> dict[str, FingerprintValue]:
        """Return a copy of the encoder fingerprint recorded in the header."""
        return dict(self._header.encoder)

    @property
    def inputs_fingerprint(self) -> dict[str, FingerprintValue] | None:
        """Return a copy of the inputs fingerprint, or None for a validation cache."""
        inputs = self._header.inputs
        return None if inputs is None else dict(inputs)

    @property
    def prefix_source(self) -> PrefixSource | None:
        """Return the source of a prefix table, or None for a directly built table."""
        return self._header.prefix_source

    def require_compatible(
        self,
        *,
        encoder: Fingerprint | None = None,
        inputs: Fingerprint | None = None,
        dimension: int | None = None,
    ) -> None:
        """Require stored fingerprints and dimension to equal the given values.

        Fingerprints are compared setting by setting; an absent setting compares as
        None. There is no override: different values need a new table name.

        Args:
            encoder: Required encoder fingerprint; None skips the check.
            inputs: Required inputs fingerprint; None skips the check.
            dimension: Required vector width; None skips the check.

        Raises:
            IncompatibleTableError: If any value differs, listing each differing
                setting with its stored and required values.
        """
        differences: Differences = {}
        if encoder is not None:
            differences.update(_differences(self._header.encoder, encoder))
        if inputs is not None:
            differences.update(_differences(self._header.inputs or {}, inputs))
        if dimension is not None and dimension != self.dimension:
            differences["dimension"] = (self.dimension, dimension)
        if differences:
            listed = "; ".join(
                f"{key}: stored {stored!r}, required {required!r}"
                for key, (stored, required) in differences.items()
            )
            raise IncompatibleTableError(
                f"Table {self.name!r} is incompatible: {listed}. Settings that change "
                "vectors cannot change for an existing table; use a new table name "
                f"(ADR-0001, {_ADR}).",
                differences=differences,
            )

    @property
    def is_complete(self) -> bool:
        """Return whether the completion marker points at the table's current version.

        Any write after ``mark_complete()`` moves the version past the marker, so a
        changed table is never reported complete. Tables written before completion
        markers existed have none and are incomplete.
        """
        marker = self._table.tags.list().get(_COMPLETE_TAG)
        return marker is not None and marker["version"] == self._table.version

    @property
    def creation_snapshot(self) -> str:
        """Return the creation snapshot exactly as it was written."""
        return _metadata(self._table)[_SNAPSHOT_KEY]

    def derive_prefix(self, destination: str, dimension: int, *, batch_size: int) -> Self:
        """Open or create a prefix table holding this table's rows at a smaller dimension.

        An existing destination is opened under this table's role without reading this
        table, and must record this table's fingerprints. Otherwise one streaming creation copies every row, record ID, and
        metadata value, replacing each vector with its exact float32 prefix. The new
        header keeps this table's role and records this table as the source; the
        creation snapshot is copied verbatim. A new prefix of a complete table is
        marked complete once the copy finishes; a prefix of an incomplete table is not.
        The prefix inherits this table's encoder and inputs fingerprints.

        Args:
            destination: Name of the prefix table in the same database.
            dimension: Prefix width, at most this table's dimension.
            batch_size: Rows per streamed batch.

        Returns:
            The prefix table.

        Raises:
            EmbeddingTableError: If ``dimension`` exceeds this table's dimension.
            IncompatibleTableError: If an existing destination records other
                fingerprints; it is left unchanged.
        """
        if type(self).exists(self.uri, destination):
            existing = type(self).open(self.uri, destination, self.role)
            existing.require_compatible(encoder=self._header.encoder, inputs=self._header.inputs)
            return existing
        if dimension > self.dimension:
            raise EmbeddingTableError(
                f"Prefix width {dimension} exceeds the stored width {self.dimension} "
                f"of table {self.name!r}"
            )
        header = _Header(
            schema_version=SCHEMA_VERSION,
            role=self.role,
            dimension=dimension,
            encoder=self._header.encoder,
            inputs=self._header.inputs,
            source_table=self.name,
            source_dimension=self.dimension,
        )
        snapshot = self.creation_snapshot
        schema = _schema(header, snapshot)
        source_complete = self.is_complete
        rows = pa.RecordBatchReader.from_batches(
            schema, _prefix_batches(self._table, schema, dimension, batch_size)
        )
        prefix = type(self)._create(self.uri, destination, header, snapshot, rows)
        if source_complete:
            prefix._write_completion()
        return prefix

    def append(
        self,
        records: Sequence[PointRecord],
        vectors: np.ndarray[tuple[int, int], np.dtype[np.float32 | np.float64]],
    ) -> None:
        """Append records paired with vectors; malformed vectors fail backend validation.

        Args:
            records: Records to store, in order.
            vectors: One vector per record, of the table's dimension.

        Raises:
            ValueError: If the record and vector counts differ; nothing is written.
        """
        if len(records) != len(vectors):
            raise ValueError("Record/vector count mismatch")
        self._table.add(
            [_row(record, vector) for record, vector in zip(records, vectors, strict=True)],
            on_bad_vectors="error",
        )

    def mark_complete(self, expected: Iterable[SourceKey]) -> None:
        """Verify every expected source key, optimize reference tables, then mark complete.

        Validation caches are never optimized. The completion marker is persisted
        against the version reached after optimization, replacing any earlier marker.

        Args:
            expected: Source keys the finished table must contain; repeats count once.
                Extra stored rows are allowed.

        Raises:
            IncompleteTableError: If any expected key is absent, stating how many;
                nothing is optimized or marked.
        """
        error = self._incomplete(expected)
        if error.missing:
            raise error
        self._clear_completion()
        if self.role == "reference":
            self.optimize()
        self._write_completion()

    def _incomplete(self, expected: Iterable[SourceKey]) -> IncompleteTableError:
        """Count absent expected keys once and describe them for the researcher."""
        unique = set(expected)
        missing = len(self.missing(unique))
        return IncompleteTableError(
            f"Table {self.name!r} is incomplete: missing {missing} of {len(unique)} expected "
            f"source keys; {_RESUME[self.role]}",
            missing=missing,
        )

    def _write_completion(self) -> None:
        self._table.tags.create(_COMPLETE_TAG, self._table.version)

    def _clear_completion(self) -> None:
        if _COMPLETE_TAG in self._table.tags.list():
            self._table.tags.delete(_COMPLETE_TAG)

    def optimize(self) -> None:
        """Compact with LanceDB's default maintenance and version-retention policy.

        Clears the completion marker first: compaction writes new versions, and a
        marker on an old version would block LanceDB's cleanup of it.
        """
        self._clear_completion()
        self._table.optimize()

    def class_counts(self) -> Counter[str]:
        """Count every stored row by class label without reading vectors."""
        return self._column_counts("class")

    def source_counts(self) -> Counter[str]:
        """Count every stored row, including duplicate source keys, by source kind.

        Streams only the source-kind column; vectors are never read.
        """
        return self._column_counts("source_kind")

    def _column_counts(self, column: str) -> Counter[str]:
        counts: Counter[str] = Counter()
        for batch in self._table.search().select([column]).limit(None).to_batches():
            counts.update(batch.column(0).to_pylist())
        return counts

    def missing(self, keys: Iterable[SourceKey]) -> list[SourceKey]:
        """Return requested source keys absent from the table.

        Args:
            keys: Source keys; repeats are allowed.

        Returns:
            Each absent key once, in first-request order, from one metadata-only
            scan that never reads vectors.
        """
        rows = self._table.search().select(["source_kind", "source_identity"]).limit(None)
        stored = frozenset((row["source_kind"], row["source_identity"]) for row in rows.to_list())
        return [key for key in dict.fromkeys(keys) if key not in stored]

    def rows(self) -> list[tuple[PointRecord, NDArray[np.float32]]]:
        """Read every stored record with its vector, in storage order.

        Intended for inspection and small tables; it loads every vector.
        """
        return [
            (_record(row), np.asarray(row[_VECTOR], dtype=np.float32))
            for row in self._table.search().limit(None).to_list()
        ]

    def fetch_vectors(self, keys: Sequence[SourceKey]) -> NDArray[np.float32]:
        """Fetch validation-cache vectors in the requested key order, including repeats.

        Args:
            keys: Source keys to fetch; repeats are allowed.

        Returns:
            A float32 matrix with one row per requested key; empty input returns
            shape ``(0, dimension)``.

        Raises:
            EmbeddingTableError: If this table is not a validation cache.
            KeyError: If a requested key is absent; storage errors propagate.
        """
        self._require_role("validation_cache", "fetch vectors from")
        if not keys:
            return np.empty((0, self.dimension), dtype=np.float32)

        def literal(value: str) -> str:
            return "'" + value.replace("'", "''") + "'"

        predicate = " OR ".join(
            f"(source_kind = {literal(kind)} AND source_identity = {literal(identity)})"
            for kind, identity in keys
        )
        rows = (
            self._table.search()
            .where(predicate)
            .select(["source_kind", "source_identity", _VECTOR])
            .limit(None)
            .to_list()
        )
        vectors = {(row["source_kind"], row["source_identity"]): row[_VECTOR] for row in rows}
        return np.asarray([vectors[key] for key in keys], dtype=np.float32)

    def query(
        self, vector: np.ndarray[tuple[int], np.dtype[np.float32 | np.float64]], k: int
    ) -> list[Neighbor]:
        """Retrieve exact cosine neighbours from a reference table, in backend order.

        Args:
            vector: Query vector of the table's dimension.
            k: Maximum number of neighbours.

        Returns:
            Neighbours with record ID, label, and cosine similarity, which may be
            negative.

        Raises:
            EmbeddingTableError: If this table is not a reference table.
        """
        self._require_role("reference", "query")
        points = (
            self._table.search(vector.tolist())
            .distance_type("cosine")
            .bypass_vector_index()
            .limit(k)
            .to_list()
        )
        return [
            Neighbor(id=p["id"], label=p["class"], score=1 - float(p["_distance"])) for p in points
        ]

    def _require_role(self, role: TableRole, action: str) -> None:
        if self.role != role:
            raise EmbeddingTableError(
                f"Cannot {action} table {self.name!r}: it is a {self.role} table, not a {role} table"
            )


def _resolve(uri: str | Path) -> Path:
    return Path(uri).expanduser().resolve()


def _schema(header: _Header, snapshot: str) -> pa.Schema:
    fields = [pa.field(name, pa.string(), nullable=False) for name in _STRING_COLUMNS]
    fields.append(pa.field(_VECTOR, pa.list_(pa.float32(), header.dimension), nullable=False))
    metadata = {_HEADER_KEY: header.model_dump_json(), _SNAPSHOT_KEY: snapshot}
    return pa.schema(fields, metadata=metadata)


def _metadata(table: LanceTable) -> dict[str, str]:
    raw = table.schema.metadata or {}
    return {key.decode(): value.decode() for key, value in raw.items()}


def _read_header(table: LanceTable) -> _Header:
    """Parse and check the header; any missing, stale, or inconsistent header is legacy."""
    metadata = _metadata(table)
    try:
        header = _Header.model_validate_json(metadata[_HEADER_KEY])
    except (KeyError, ValidationError):
        header = None
    if (
        header is None
        or header.schema_version != SCHEMA_VERSION
        or not _inputs_match_role(header.role, header.inputs)
        or _SNAPSHOT_KEY not in metadata
        or table.schema.remove_metadata() != _schema(header, "").remove_metadata()
    ):
        raise LegacyTableError(
            f"Table {table.name!r} has no valid embedding table header for schema version "
            f"{SCHEMA_VERSION}. Rebuild it under a new table name; legacy tables are "
            f"refused, never migrated (ADR-0001, {_ADR})."
        )
    return header


def _inputs_match_role(role: TableRole, inputs: Fingerprint | None) -> bool:
    """Reference tables carry an inputs fingerprint; validation caches never do."""
    return (role == "reference") == (inputs is not None)


def _differences(stored: Fingerprint, required: Fingerprint) -> Differences:
    """Return each setting whose stored and required values differ, in sorted order."""
    return {
        key: (stored.get(key), required.get(key))
        for key in sorted(stored.keys() | required.keys())
        if stored.get(key) != required.get(key)
    }


def _row(record: PointRecord, vector: NDArray[Any]) -> dict[str, Any]:
    return {
        "id": record.id,
        "class": record.label,
        "source_kind": record.source_kind,
        "source_identity": record.source_identity,
        "value": record.value,
        _VECTOR: vector.tolist(),
    }


def _record(row: dict[str, Any]) -> PointRecord:
    return PointRecord(
        label=row["class"],
        source_kind=row["source_kind"],
        source_identity=row["source_identity"],
        value=row["value"],
        id=row["id"],
    )


def _prefix_batches(
    source: LanceTable, schema: pa.Schema, dimension: int, batch_size: int
) -> Iterator[pa.RecordBatch]:
    """Stream every source row, replacing only the vector column with its prefix."""
    position = schema.get_field_index(_VECTOR)
    vector_type = schema.field(position).type
    for batch in source.search().limit(None).to_batches(batch_size=batch_size):
        vectors = np.asarray(batch.column(position).to_pylist(), dtype=np.float32)
        arrays = list(batch.columns)
        arrays[position] = pa.array(vectors[:, :dimension].tolist(), type=vector_type)
        yield pa.RecordBatch.from_arrays(arrays, schema=schema)
